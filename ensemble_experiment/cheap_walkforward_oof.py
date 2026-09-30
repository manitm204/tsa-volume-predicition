"""
ensemble_experiment/cheap_walkforward_oof.py
=============================================
Cheap walk-forward CV to honestly backfill the OOF gap between the last
verified-honest walk-forward window (2025-06-05 -> 2026-06-03) and the
current production training cutoffs, for BOTH tabular and TS3.

Why this gap exists: dates after 2026-06-03 can only be added to the OOF
files via train_models.py's update_*_oof() once a production model's
training cutoff predates them (otherwise it'd be in-sample). The gap
2026-06-04 -> 2026-08-19 falls *before* TS3's current cutoff (2026-08-19)
and before any tabular cutoff at all, so it needs real walk-forward CV:
train on data strictly before each fold, predict the fold.

Budget: TIME_LIMIT_PER_FOLD seconds/fold x N_FOLDS folds, for EACH model.
Defaults (5 min x 3 folds) => ~15 min tabular + ~15 min ts3 = ~30 min total.
Lighter/faster presets are used than production training since the time
budget is intentionally small.

Usage:
    python ensemble_experiment/cheap_walkforward_oof.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from build_features import (
    load_master, get_feature_columns, TARGET_COL,
    transform_target, inverse_transform_target, USE_LOG_TARGET,
    load_and_merge_data, build_features_from_df,
)
from train_models import TS3_COVARIATES, ITEM_ID

OOF_DIR     = Path("./ensemble_experiment/output")
OOF_TABULAR = OOF_DIR / "ag_tabular_oof.csv"
OOF_TS      = OOF_DIR / "ag_timeseries_oof.csv"

GAP_START = pd.Timestamp("2026-06-04")
GAP_END   = pd.Timestamp("2026-08-19")   # inclusive — TS3's current training cutoff

N_FOLDS             = 3
TIME_LIMIT_PER_FOLD = 5 * 60     # seconds
TABULAR_PRESET      = "medium_quality"
TS3_PRESET          = "fast_training"
TS3_MIN_HISTORY     = 365

TMP_ROOT = Path("./output_autogluon_best/_cheap_cv_tmp")


def _folds(dates: pd.DatetimeIndex, n_folds: int):
    return np.array_split(np.asarray(dates), n_folds)


def _merge_and_save(oof_path: Path, pred_col: str, new_rows: list[dict]):
    if not new_rows:
        print(f"  {pred_col}: no new rows produced")
        return
    new_df = pd.DataFrame(new_rows)
    base = pd.read_csv(oof_path, parse_dates=["Date"]) if oof_path.exists() else pd.DataFrame()
    if not base.empty:
        existing = set(base["Date"].dt.date)
        new_df = new_df[~new_df["Date"].dt.date.isin(existing)]
    extended = pd.concat([base, new_df], ignore_index=True).sort_values("Date").reset_index(drop=True)
    extended.to_csv(oof_path, index=False)
    print(f"  {pred_col} -> {oof_path}  ({len(extended)} total rows, {len(new_df)} new)")
    return new_df


# ==========================================================================
# Tabular walk-forward
# ==========================================================================
def run_tabular_walkforward():
    print("\n" + "=" * 70)
    print("Tabular — cheap walk-forward CV")
    print("=" * 70)

    df = load_master()
    df["Date"] = pd.to_datetime(df["Date"])
    feature_cols = get_feature_columns(df)

    gap_dates = df.loc[(df["Date"] >= GAP_START) & (df["Date"] <= GAP_END), "Date"].sort_values()
    if gap_dates.empty:
        print("  nothing to do — no rows in gap window")
        return
    fold_date_groups = _folds(gap_dates.to_numpy(), N_FOLDS)

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    new_rows = []

    for i, fold_dates in enumerate(fold_date_groups, start=1):
        fold_dates = pd.to_datetime(fold_dates)
        fold_start, fold_end = fold_dates.min(), fold_dates.max()

        train_df = df[df["Date"] < fold_start][feature_cols + [TARGET_COL]].copy()
        test_df  = df[df["Date"].isin(fold_dates)][["Date"] + feature_cols + [TARGET_COL]].copy()
        y_test   = test_df[TARGET_COL]

        print(f"\nFold {i}/{len(fold_date_groups)}  train {len(train_df):,}  "
              f"test {len(test_df)}  ({fold_start.date()} -> {fold_end.date()})")

        target_col_ag = TARGET_COL
        if USE_LOG_TARGET:
            target_col_ag = "log_target"
            train_df[target_col_ag] = transform_target(train_df[TARGET_COL])
            train_df = train_df.drop(columns=[TARGET_COL])
        test_input = test_df[feature_cols]

        ag_path = TMP_ROOT / f"tab_fold_{i}"
        if ag_path.exists():
            shutil.rmtree(ag_path)

        predictor = TabularPredictor(
            label=target_col_ag,
            path=str(ag_path),
            eval_metric="mean_absolute_error",
            verbosity=1,
        ).fit(train_data=train_df, time_limit=TIME_LIMIT_PER_FOLD, presets=TABULAR_PRESET)

        pred_raw = predictor.predict(test_input)
        pred = inverse_transform_target(pred_raw.values) if USE_LOG_TARGET else pred_raw.values

        mae = mean_absolute_error(y_test, pred)
        print(f"  MAE {mae:>10,.0f}")

        for d, actual, p in zip(test_df["Date"], y_test, pred):
            new_rows.append({"Date": d, "Volume": float(actual), "pred_ag_tabular": float(p)})

        shutil.rmtree(ag_path, ignore_errors=True)

    _merge_and_save(OOF_TABULAR, "pred_ag_tabular", new_rows)


# ==========================================================================
# TS3 walk-forward
# ==========================================================================
def run_ts3_walkforward():
    print("\n" + "=" * 70)
    print("TS3 — cheap walk-forward CV")
    print("=" * 70)

    df = build_features_from_df(load_and_merge_data(), verbose=False, prune=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)
    available = [c for c in TS3_COVARIATES if c in df.columns]

    gap_dates = df.loc[(df["Date"] >= GAP_START) & (df["Date"] <= GAP_END), "Date"].sort_values()
    if gap_dates.empty:
        print("  nothing to do — no rows in gap window")
        return
    fold_date_groups = _folds(gap_dates.to_numpy(), N_FOLDS)

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    new_rows = []

    for i, fold_dates in enumerate(fold_date_groups, start=1):
        fold_dates = pd.to_datetime(fold_dates)
        fold_start, fold_end = fold_dates.min(), fold_dates.max()

        train_hist = df[df["Date"] < fold_start][["Date", TARGET_COL] + available].copy()
        print(f"\nFold {i}/{len(fold_date_groups)}  train {len(train_hist):,}  "
              f"test {len(fold_dates)}  ({fold_start.date()} -> {fold_end.date()})")

        train_hist["item_id"] = ITEM_ID
        train_ts = TimeSeriesDataFrame.from_data_frame(
            train_hist.rename(columns={"Date": "timestamp", TARGET_COL: "target"}),
            id_column="item_id", timestamp_column="timestamp",
        )

        ts_path = TMP_ROOT / f"ts_fold_{i}"
        if ts_path.exists():
            shutil.rmtree(ts_path)

        orig_dir = os.getcwd()
        tmp_dir = tempfile.mkdtemp(prefix="ag_ts3_cheap_cv_")
        try:
            os.chdir(tmp_dir)
            predictor = TimeSeriesPredictor(
                prediction_length=1, path=str(ts_path), target="target",
                eval_metric="MAE",
                known_covariates_names=available if available else None,
                verbosity=1,
            ).fit(train_ts, presets=TS3_PRESET, time_limit=TIME_LIMIT_PER_FOLD)

            fold_mae_vals = []
            for j, target_ts in enumerate(fold_dates, start=1):
                history = (
                    df[df["Date"] < target_ts][["Date", TARGET_COL] + available]
                    .dropna(subset=[TARGET_COL]).copy()
                )
                if len(history) < TS3_MIN_HISTORY:
                    continue
                history["item_id"] = ITEM_ID
                history_ts = TimeSeriesDataFrame.from_data_frame(
                    history.rename(columns={"Date": "timestamp", TARGET_COL: "target"}),
                    id_column="item_id", timestamp_column="timestamp",
                )
                future_row = df[df["Date"] == target_ts][["Date"] + available].copy()
                future_row["item_id"] = ITEM_ID
                known_cov = TimeSeriesDataFrame.from_data_frame(
                    future_row.rename(columns={"Date": "timestamp"}),
                    id_column="item_id", timestamp_column="timestamp",
                )
                try:
                    fc = predictor.predict(history_ts, known_covariates=known_cov)
                    col = "mean" if "mean" in fc.columns else fc.columns[0]
                    pred = float(fc.iloc[0][col])
                except Exception as e:
                    print(f"    WARNING: TS3 prediction failed for {target_ts.date()} ({e})")
                    continue
                actual = float(df.loc[df["Date"] == target_ts, TARGET_COL].iloc[0])
                new_rows.append({"Date": target_ts, "Volume": actual, "pred_ag_timeseries": pred})
                fold_mae_vals.append(abs(actual - pred))
        finally:
            os.chdir(orig_dir)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            shutil.rmtree(ts_path, ignore_errors=True)

        if fold_mae_vals:
            print(f"  MAE {np.mean(fold_mae_vals):>10,.0f}  ({len(fold_mae_vals)} days)")

    _merge_and_save(OOF_TS, "pred_ag_timeseries", new_rows)


def main():
    run_tabular_walkforward()
    run_ts3_walkforward()

    shutil.rmtree(TMP_ROOT, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Done. Now run:")
    print("  python3 -c \"import sys; sys.path.insert(0,'.'); from train_models import rebuild_combined_oof; rebuild_combined_oof()\"")
    print("to rebuild combined_oof.csv with the extended honest window.")
    print("=" * 70)


if __name__ == "__main__":
    main()
