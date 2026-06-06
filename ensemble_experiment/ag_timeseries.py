"""
ag_timeseries.py
================
AutoGluon TimeSeries on the TS3 covariate scope (calendar + holiday +
lag365 family — all known-ahead). Single config, best_quality, 600s
per fold.

  * Walk-forward OOF: 4 folds. Dates are pulled from
    ensemble_experiment/output/ag_tabular_oof.csv so the TS OOF rows line
    up 1:1 with the AG Tabular OOF. (Run ag_tabular.py first.)
  * Inside each fold: fit one TimeSeriesPredictor on data strictly before
    the fold start, then walk day-by-day with prediction_length=1. The
    input context is extended with each new observed Volume; model
    weights stay frozen within the fold (same convention as the existing
    timeseries_experiment.py).

Outputs (under ensemble_experiment/output/):
    ag_timeseries_oof.csv                 Date, Volume, pred_ag_timeseries
    fold_predictions/ag_timeseries_fold{1..4}.csv
    ag_timeseries_summary.json
"""

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from build_features import (
    TARGET_COL,
    build_features_from_df,
    load_and_merge_data,
)

OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)
FOLD_CACHE_DIR = OUT_DIR / "fold_predictions"
FOLD_CACHE_DIR.mkdir(exist_ok=True)

TABULAR_OOF_PATH = OUT_DIR / "ag_tabular_oof.csv"

PREDICTION_LENGTH = 1
AG_PRESET = "best_quality"
TIME_LIMIT = 600
ITEM_ID = "tsa"

# TS3 covariate scope (lifted from timeseries_experiment.py): calendar +
# holiday + lag365 family. All are known-ahead by construction.
TS3_COVARIATES = [
    # calendar
    "dayofweek", "month", "doy_cos", "doy_sin", "week_of_year",
    "quarter", "summer_flag", "holiday_season_flag",
    # holiday
    "is_holiday", "days_to_holiday", "days_to_holiday_x_dow",
    "holiday_decay_shape", "holiday_expected_x_dow",
    "nearest_holiday_aligned_lag", "nearest_holiday_expected_vol",
    "holiday_expected_vs_normal", "holiday_regime_strength",
    "is_long_weekend", "post_holiday_flag", "pre_holiday_regime",
    # lag365 family
    "lag365_residual_anchor", "lag365_same_dow_5w_mean",
    "lag365_error_7d", "yoy_ratio_7d",
]


def load_feature_df():
    print("Building feature dataframe (prune=False)...")
    df = load_and_merge_data()
    df = build_features_from_df(df, verbose=False, prune=False)
    df = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"  rows: {len(df):,}   date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_tabular_oof():
    if not TABULAR_OOF_PATH.exists():
        raise FileNotFoundError(
            f"{TABULAR_OOF_PATH} missing. Run ag_tabular.py first so we have "
            "the OOF date range to align against."
        )
    oof = pd.read_csv(TABULAR_OOF_PATH, parse_dates=["Date"])
    return oof[["Date", TARGET_COL]].rename(columns={TARGET_COL: "Volume"})


def fold_boundaries(tabular_oof):
    """4 contiguous chunks of the tabular OOF — identical date slicing to
    timeseries_experiment.py."""
    n = len(tabular_oof)
    fold_size = n // 4
    folds = []
    for f in range(4):
        start = tabular_oof.iloc[f * fold_size]["Date"]
        end_idx = (f + 1) * fold_size - 1 if f < 3 else n - 1
        end = tabular_oof.iloc[end_idx]["Date"]
        folds.append((start, end))
    return folds


def to_ts_df(df, covariate_cols):
    cols = ["Date", TARGET_COL] + covariate_cols
    out = df[cols].copy()
    out["item_id"] = ITEM_ID
    out = out.rename(columns={"Date": "timestamp", TARGET_COL: "target"})
    return TimeSeriesDataFrame.from_data_frame(
        out, id_column="item_id", timestamp_column="timestamp"
    )


def fold_cache_path(fold_idx):
    return FOLD_CACHE_DIR / f"ag_timeseries_fold{fold_idx}.csv"


def fit_and_predict_fold(df, fold_idx, fold_start, fold_end, covariate_cols,
                         use_cache=True):
    cache = fold_cache_path(fold_idx)
    if use_cache and cache.exists():
        cached = pd.read_csv(cache, parse_dates=["Date"])
        if not cached.empty:
            print(f"  fold {fold_idx}: cached ({len(cached)} preds)")
            return cached

    train_df = df[df["Date"] < fold_start].reset_index(drop=True)
    fold_df = df[(df["Date"] >= fold_start) & (df["Date"] <= fold_end)].reset_index(drop=True)
    print(f"  fitting fold {fold_idx}: train n={len(train_df)} "
          f"({train_df['Date'].min().date()} → {train_df['Date'].max().date()})  "
          f"val n={len(fold_df)} ({fold_start.date()} → {fold_end.date()})")

    train_ts = to_ts_df(train_df, covariate_cols)
    tmp_path = OUT_DIR / f"_tmp_ts_fold{fold_idx}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    predictor = TimeSeriesPredictor(
        prediction_length=PREDICTION_LENGTH,
        path=str(tmp_path),
        target="target",
        eval_metric="MAE",
        known_covariates_names=covariate_cols if covariate_cols else None,
        verbosity=1,
    ).fit(
        train_ts,
        presets=AG_PRESET,
        time_limit=TIME_LIMIT,
    )

    preds = []
    for i in range(len(fold_df)):
        forecast_date = fold_df.iloc[i]["Date"]
        history = df[df["Date"] < forecast_date]
        history_ts = to_ts_df(history, covariate_cols)

        known_kw = {}
        if covariate_cols:
            future_row = fold_df.iloc[[i]][["Date"] + covariate_cols].copy()
            future_row["item_id"] = ITEM_ID
            future_row = future_row.rename(columns={"Date": "timestamp"})
            known_kw["known_covariates"] = TimeSeriesDataFrame.from_data_frame(
                future_row, id_column="item_id", timestamp_column="timestamp"
            )
        forecast = predictor.predict(history_ts, **known_kw)
        col = "mean" if "mean" in forecast.columns else forecast.columns[0]
        preds.append({"Date": forecast_date, "pred": float(forecast.iloc[0][col])})

    out = pd.DataFrame(preds)
    out.to_csv(cache, index=False)
    shutil.rmtree(tmp_path, ignore_errors=True)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--time-limit", type=int, default=None,
                   help=f"Override per-fold time limit (default {TIME_LIMIT}s)")
    return p.parse_args()


def main():
    args = parse_args()
    global TIME_LIMIT
    if args.time_limit is not None:
        TIME_LIMIT = args.time_limit
        print(f"TIME_LIMIT overridden to {TIME_LIMIT}s")

    if args.clear_cache and FOLD_CACHE_DIR.exists():
        for f in FOLD_CACHE_DIR.glob("ag_timeseries_fold*.csv"):
            f.unlink()
        print("Cleared TS fold cache")

    oof_csv = OUT_DIR / "ag_timeseries_oof.csv"
    summary_path = OUT_DIR / "ag_timeseries_summary.json"

    tabular_oof = load_tabular_oof()

    if args.report_only:
        if not oof_csv.exists():
            raise FileNotFoundError(f"{oof_csv} missing — cannot --report-only")
        oof_df = pd.read_csv(oof_csv, parse_dates=["Date"])
        print(f"Loaded cached TS OOF: {len(oof_df)} rows  "
              f"MAE = {mean_absolute_error(oof_df['Volume'], oof_df['pred_ag_timeseries']):,.0f}")
        return

    df = load_feature_df()
    available = set(df.columns)
    covariates = [c for c in TS3_COVARIATES if c in available]
    missing = [c for c in TS3_COVARIATES if c not in available]
    print(f"\nTS3 covariates: {len(covariates)} present, {len(missing)} missing")
    if missing:
        print(f"  missing: {missing}")

    folds = fold_boundaries(tabular_oof)
    print("\n4-fold walk-forward (matching tabular OOF dates):")
    for i, (s, e) in enumerate(folds, 1):
        print(f"  fold {i}: {s.date()} → {e.date()}")

    fold_metrics = []
    pieces = []
    for fold_idx, (fold_start, fold_end) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold_idx}  ({fold_start.date()} → {fold_end.date()}) ---")
        preds_fold = fit_and_predict_fold(
            df, fold_idx, fold_start, fold_end, covariates,
            use_cache=not args.no_cache,
        )
        pieces.append(preds_fold)
        fold_actual = (
            df[(df["Date"] >= fold_start) & (df["Date"] <= fold_end)]
            [["Date", TARGET_COL]].reset_index(drop=True)
        )
        merged = fold_actual.merge(preds_fold, on="Date", how="left")
        fmae = float(mean_absolute_error(merged[TARGET_COL], merged["pred"]))
        fold_metrics.append({
            "fold": fold_idx,
            "start": str(fold_start.date()),
            "end": str(fold_end.date()),
            "n": int(len(merged)),
            "mae": fmae,
        })
        print(f"  fold {fold_idx} MAE: {fmae:,.0f}")

    stacked = pd.concat(pieces, ignore_index=True).rename(columns={"pred": "pred_ag_timeseries"})
    out = tabular_oof.merge(stacked, on="Date", how="left")
    out.to_csv(oof_csv, index=False)
    overall_mae = float(mean_absolute_error(out["Volume"], out["pred_ag_timeseries"]))
    print(f"\nSaved TS OOF → {oof_csv}  (n={len(out)}, overall MAE = {overall_mae:,.0f})")

    summary = {
        "n_folds": 4,
        "fold_time_limit_s": TIME_LIMIT,
        "preset": AG_PRESET,
        "covariate_scope": "TS3",
        "n_covariates": len(covariates),
        "covariates": covariates,
        "missing_covariates": missing,
        "fold_metrics": fold_metrics,
        "overall_oof_mae": overall_mae,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
