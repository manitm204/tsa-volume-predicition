"""
train_models.py
===============
Train and maintain the prediction models and update rolling OOF
predictions used for Platt / isotonic calibration.

Schedule:
  timeseries   weekly (slow, ~30 min)     --max-age-ts       (default 6)
  OOF update   weekly                     --max-age-oof      (default 6)

Model specs match ensemble_experiment/ exactly:
  TS3        TS3_COVARIATES (calendar + holiday + lag365 family)
             matching ensemble_experiment/ag_timeseries.py
  Tabular    managed by autogluon_full.py, NOT retrained here
  yoy_delta  deterministic formula (build_features.add_yoy_delta_feature),
             no training needed — replaces Prophet in the production ensemble

OOF strategy for new dates:
  TS3        — use current production TS3 to predict new dates day-by-day
  Tabular    — use current production tabular to predict new dates
  Anchor     — extract anchor_master feature directly (no model needed)
  yoy_delta  — extract pred_yoy_delta feature directly (no model needed)

All OOF files live in ensemble_experiment/output/ so calibration scripts
(platt_regime.py etc.) keep reading from the same location.

Usage:
    python train_models.py                      # respect staleness thresholds
    python train_models.py --ts                 # force timeseries retrain only
    python train_models.py --update-oof         # force OOF update only
    python train_models.py --all                # force retrain + OOF update
    python train_models.py --no-oof             # skip OOF update
    python train_models.py --ts-time-limit 900
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from build_features import (
    TARGET_COL, KEEP_FEATURES,
    build_features_from_df, load_and_merge_data,
)

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("./output_router_shadow/models")
OOF_DIR    = Path("./ensemble_experiment/output")
TABULAR_MODEL_DIR = Path("./output_autogluon_best/ag_final")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TS3_PATH          = MODELS_DIR / "ts3_predictor"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.json"

TS_TRAINED_AT      = MODELS_DIR / "ts_trained_at.json"
OOF_UPDATED_AT     = MODELS_DIR / "oof_updated_at.json"

OOF_TABULAR   = OOF_DIR / "ag_tabular_oof.csv"
OOF_TS        = OOF_DIR / "ag_timeseries_oof.csv"
OOF_ANCHOR    = OOF_DIR / "anchor_master_oof.csv"
OOF_YOY_DELTA = OOF_DIR / "yoy_delta_oof.csv"
OOF_COMBINED  = OOF_DIR / "combined_oof.csv"

ITEM_ID = "tsa"

# ── Model specs (must match ensemble_experiment/ scripts) ────────────────────
# TS3_COVARIATES identical to ensemble_experiment/ag_timeseries.py
TS3_COVARIATES = [
    "dayofweek", "month", "doy_cos", "doy_sin", "week_of_year",
    "quarter", "summer_flag", "holiday_season_flag",
    "is_holiday", "days_to_holiday", "days_to_holiday_x_dow",
    "holiday_decay_shape", "holiday_expected_x_dow",
    "nearest_holiday_aligned_lag", "nearest_holiday_expected_vol",
    "holiday_expected_vs_normal", "holiday_regime_strength",
    "is_long_weekend", "post_holiday_flag", "pre_holiday_regime",
    "lag365_residual_anchor", "lag365_same_dow_5w_mean",
    "lag365_error_7d", "yoy_ratio_7d",
]

DEFAULT_TS_TIME_LIMIT = 1800
DEFAULT_TS_PRESET     = "best_quality"


# ── Staleness helpers ─────────────────────────────────────────────────────────
def _age_days(path: Path) -> float:
    """Days since a staleness file was last written. Returns inf if missing."""
    if not path.exists():
        return float("inf")
    try:
        info = json.loads(path.read_text())
        last = datetime.fromisoformat(info["trained_at"])
        return (datetime.now(timezone.utc) - last).total_seconds() / 86400
    except Exception:
        return float("inf")


def _mark_trained(path: Path, extra: dict | None = None):
    info = {"trained_at": datetime.now(timezone.utc).isoformat()}
    if extra:
        info.update(extra)
    path.write_text(json.dumps(info, indent=2))


def _needs_update(path: Path, max_age: int, force: bool) -> bool:
    if force:
        return True
    age = _age_days(path)
    if age < max_age:
        print(f"  up to date (age {age:.1f}d < {max_age}d) — skipping")
        return False
    print(f"  stale (age {age:.1f}d >= {max_age}d) — updating")
    return True


# ── Data loading ──────────────────────────────────────────────────────────────
def load_feature_df(prune=False):
    df = load_and_merge_data()
    df = build_features_from_df(df, verbose=False, prune=prune)
    df = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def to_ts_df(df, covariate_cols):
    out = df[["Date", TARGET_COL] + covariate_cols].copy()
    out["item_id"] = ITEM_ID
    return TimeSeriesDataFrame.from_data_frame(
        out.rename(columns={"Date": "timestamp", TARGET_COL: "target"}),
        id_column="item_id", timestamp_column="timestamp",
    )


# ── TS3 training ──────────────────────────────────────────────────────────────
def train_timeseries(df: pd.DataFrame, time_limit: int, preset: str, force=False, max_age=6):
    print("\n[ TimeSeries (TS3) ]")
    if not _needs_update(TS_TRAINED_AT, max_age, force):
        return
    available = [c for c in TS3_COVARIATES if c in df.columns]
    print(f"  training on {len(df)} rows, {len(available)} covariates "
          f"(time_limit={time_limit}s, preset={preset})")
    if TS3_PATH.exists():
        shutil.rmtree(TS3_PATH)
    train_ts = to_ts_df(df, available)
    TimeSeriesPredictor(
        prediction_length=1, path=str(TS3_PATH), target="target",
        eval_metric="MAE",
        known_covariates_names=available if available else None,
        verbosity=1,
    ).fit(train_ts, presets=preset, time_limit=time_limit)
    _mark_trained(TS_TRAINED_AT, {"n_rows": len(df), "latest_date": str(df["Date"].max().date()),
                                   "preset": preset, "time_limit": time_limit})
    print(f"  saved → {TS3_PATH}")

    # Save feature_cols.json (read by daily_predict + autogluon_predict)
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump({"ts3_covariates": available,
                   "target_col": TARGET_COL, "item_id": ITEM_ID}, f, indent=2)


# ── OOF helpers ───────────────────────────────────────────────────────────────
def _new_oof_dates(oof_path: Path, full_df: pd.DataFrame) -> pd.DataFrame:
    """Rows in full_df whose dates are not yet in oof_path."""
    if not oof_path.exists():
        return full_df
    existing = set(pd.read_csv(oof_path, parse_dates=["Date"])["Date"].dt.date)
    return full_df[~full_df["Date"].dt.date.isin(existing)].reset_index(drop=True)


# ── TS3 OOF ───────────────────────────────────────────────────────────────────
# On the very first run the OOF file may be missing hundreds of historical dates.
# Predicting them all at once (one-by-one through AutoGluon) takes hours and
# scatters AutogluonModels/ temp dirs everywhere.  We cap at MAX_NEW_PER_RUN per
# invocation so the nightly cron backfills gradually; run `--update-oof` daily to
# fill faster, or run `ensemble_experiment/ag_timeseries.py` to bulk-generate the
# historical OOF in one shot.
_TS3_MAX_NEW_PER_RUN = 90
_TS3_MIN_HISTORY     = 365   # skip dates with fewer prior rows (too short for reliable TS3)


def update_ts_oof(full_df: pd.DataFrame):
    """
    Use current production TS3 model to predict new dates day-by-day.

    Only backfills dates strictly after TS_TRAINED_AT's recorded training
    cutoff (latest_date) — otherwise the model may have already seen that
    date during training, and predicting it would be in-sample, not honest
    OOF. (This bit us: dates were previously backfilled with no cutoff
    check, silently mixing in leaked/in-sample predictions.)

    Caps at _TS3_MAX_NEW_PER_RUN per call and requires >= _TS3_MIN_HISTORY prior
    rows so early-data edge cases don't trigger noisy SeasonalNaive fallbacks.
    AutoGluon temp dirs are created inside a temp directory and cleaned up.
    """
    if not TS3_PATH.exists():
        print("  TS3 OOF: production model not found — run --ts first")
        return

    if not TS_TRAINED_AT.exists():
        print("  ts3 OOF: no training-cutoff record (TS_TRAINED_AT) — skipping "
              "backfill until --ts has been run at least once.")
        return
    cutoff = pd.Timestamp(json.loads(TS_TRAINED_AT.read_text())["latest_date"])

    new_df = _new_oof_dates(OOF_TS, full_df)
    new_df = new_df[new_df["Date"] > cutoff]
    if new_df.empty:
        print(f"  ts3 OOF: up to date (nothing newer than training cutoff {cutoff.date()})")
        return

    new_df = new_df.sort_values("Date").reset_index(drop=True)
    total_new = len(new_df)

    if total_new > _TS3_MAX_NEW_PER_RUN:
        new_df = new_df.tail(_TS3_MAX_NEW_PER_RUN).reset_index(drop=True)
        print(f"  ts3 OOF: {total_new} new dates pending — processing most recent "
              f"{_TS3_MAX_NEW_PER_RUN} ({new_df['Date'].min().date()} → {new_df['Date'].max().date()})")
        print(f"           run --update-oof again to continue backfilling")
    else:
        print(f"  ts3 OOF: predicting {total_new} new dates "
              f"({new_df['Date'].min().date()} → {new_df['Date'].max().date()})")

    available = [c for c in TS3_COVARIATES if c in full_df.columns]
    # Resolve to absolute path before any chdir
    ts3_abs = str(TS3_PATH.resolve())
    predictor = TimeSeriesPredictor.load(ts3_abs)
    new_rows = []

    # Run predictions from a temp dir so AutoGluon's fallback model dirs
    # (SeasonalNaive etc.) don't pollute the project root.
    orig_dir = os.getcwd()
    tmp_dir  = tempfile.mkdtemp(prefix="ag_ts3_oof_")
    try:
        os.chdir(tmp_dir)
        for i, (_, row) in enumerate(new_df.iterrows(), 1):
            target_ts = row["Date"]
            history = (
                full_df[full_df["Date"] < target_ts][["Date", TARGET_COL] + available]
                .dropna(subset=[TARGET_COL])
                .copy()
            )
            if len(history) < _TS3_MIN_HISTORY:
                continue
            history["item_id"] = ITEM_ID
            history_ts = TimeSeriesDataFrame.from_data_frame(
                history.rename(columns={"Date": "timestamp", TARGET_COL: "target"}),
                id_column="item_id", timestamp_column="timestamp",
            )
            future_row = full_df[full_df["Date"] == target_ts][["Date"] + available].copy()
            future_row["item_id"] = ITEM_ID
            known_cov = TimeSeriesDataFrame.from_data_frame(
                future_row.rename(columns={"Date": "timestamp"}),
                id_column="item_id", timestamp_column="timestamp",
            )
            try:
                fc  = predictor.predict(history_ts, known_covariates=known_cov)
                col = "mean" if "mean" in fc.columns else fc.columns[0]
                pred = float(fc.iloc[0][col])
            except Exception as e:
                print(f"    WARNING: TS3 prediction failed for {target_ts.date()} ({e})")
                continue
            new_rows.append({
                "Date": target_ts,
                "Volume": float(row[TARGET_COL]),
                "pred_ag_timeseries": pred,
            })
            if i % 10 == 0:
                print(f"    {i}/{len(new_df)}  last={target_ts.date()}")
    finally:
        os.chdir(orig_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if new_rows:
        base = pd.read_csv(OOF_TS, parse_dates=["Date"]) if OOF_TS.exists() else pd.DataFrame()
        extended = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True).sort_values("Date")
        extended.to_csv(OOF_TS, index=False)
        remaining = total_new - len(new_rows)
        suffix = f"  ({remaining} dates still pending)" if remaining > 0 else ""
        print(f"  ts3 OOF → {OOF_TS}  ({len(extended)} total rows){suffix}")


# ── Tabular OOF ───────────────────────────────────────────────────────────────
def update_tabular_oof(full_df: pd.DataFrame):
    """
    Use current production tabular model to predict new dates.

    Only backfills dates strictly after ag_final_trained_through.json's
    recorded training cutoff — otherwise the model may have already seen
    that date during training, and predicting it would be in-sample, not
    honest OOF. (This bit us: ~78% of ag_tabular_oof.csv was previously
    leaked this way — see autogluon_full.py's TRAINED_THROUGH_PATH.)
    """
    if not TABULAR_MODEL_DIR.exists():
        print("  tabular OOF: production model not found — run autogluon_full.py first")
        return

    trained_through_path = TABULAR_MODEL_DIR.parent / "ag_final_trained_through.json"
    if not trained_through_path.exists():
        print("  tabular OOF: no training-cutoff record (ag_final_trained_through.json) — "
              "skipping backfill until autogluon_full.py has been rerun with the cutoff fix.")
        return
    cutoff = pd.Timestamp(json.loads(trained_through_path.read_text())["trained_through_date"])

    pruned_df = build_features_from_df(load_and_merge_data(), verbose=False, prune=True)
    pruned_df["Date"] = pd.to_datetime(pruned_df["Date"])
    pruned_df = pruned_df[pruned_df[TARGET_COL].notna()].copy()

    new_df = _new_oof_dates(OOF_TABULAR, pruned_df)
    new_df = new_df[new_df["Date"] > cutoff]
    if new_df.empty:
        print(f"  tabular OOF: up to date (nothing newer than training cutoff {cutoff.date()})")
        return

    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(str(TABULAR_MODEL_DIR))

    feat_cols = [c for c in KEEP_FEATURES if c in new_df.columns]
    preds = predictor.predict(new_df[feat_cols])

    new_rows = []
    for i, (_, row) in enumerate(new_df.iterrows()):
        new_rows.append({
            "Date":            row["Date"],
            "Volume":          float(row[TARGET_COL]),
            "pred_ag_tabular": float(preds.iloc[i]) if hasattr(preds, "iloc") else float(preds[i]),
        })

    base = pd.read_csv(OOF_TABULAR, parse_dates=["Date"]) if OOF_TABULAR.exists() else pd.DataFrame()
    extended = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True).sort_values("Date")
    extended.to_csv(OOF_TABULAR, index=False)
    print(f"  tabular OOF → {OOF_TABULAR}  ({len(extended)} total rows, {len(new_rows)} new)")


# ── Anchor OOF ────────────────────────────────────────────────────────────────
def update_anchor_oof(full_df: pd.DataFrame):
    """Extract anchor_master feature for any dates not yet in the anchor OOF."""
    if "anchor_master" not in full_df.columns:
        print("  anchor OOF: anchor_master column not found in features — check pipeline")
        return

    sub = full_df[["Date", TARGET_COL, "anchor_master"]].rename(
        columns={TARGET_COL: "Volume", "anchor_master": "pred_anchor_master"}
    )
    new_df = _new_oof_dates(OOF_ANCHOR, sub)
    if new_df.empty:
        print("  anchor OOF: up to date")
        return

    new_rows = sub[sub["Date"].isin(new_df["Date"])].copy()
    base = pd.read_csv(OOF_ANCHOR, parse_dates=["Date"]) if OOF_ANCHOR.exists() else pd.DataFrame()
    extended = pd.concat([base, new_rows], ignore_index=True).sort_values("Date")
    extended.to_csv(OOF_ANCHOR, index=False)
    print(f"  anchor OOF → {OOF_ANCHOR}  ({len(extended)} total rows, {len(new_rows)} new)")


# ── yoy_delta OOF ─────────────────────────────────────────────────────────────
def update_yoy_delta_oof(full_df: pd.DataFrame):
    """Extract pred_yoy_delta feature for any dates not yet in the OOF file."""
    if "pred_yoy_delta" not in full_df.columns:
        print("  yoy_delta OOF: pred_yoy_delta column not found in features — check pipeline")
        return

    sub = full_df[["Date", TARGET_COL, "pred_yoy_delta"]].rename(
        columns={TARGET_COL: "Volume"}
    )
    new_df = _new_oof_dates(OOF_YOY_DELTA, sub)
    if new_df.empty:
        print("  yoy_delta OOF: up to date")
        return

    new_rows = sub[sub["Date"].isin(new_df["Date"])].copy()
    base = pd.read_csv(OOF_YOY_DELTA, parse_dates=["Date"]) if OOF_YOY_DELTA.exists() else pd.DataFrame()
    extended = pd.concat([base, new_rows], ignore_index=True).sort_values("Date")
    extended.to_csv(OOF_YOY_DELTA, index=False)
    print(f"  yoy_delta OOF → {OOF_YOY_DELTA}  ({len(extended)} total rows, {len(new_rows)} new)")


# ── Combined OOF ──────────────────────────────────────────────────────────────
def rebuild_combined_oof():
    """Merge the four per-model OOF files into combined_oof.csv."""
    sources = [
        (OOF_TABULAR,   "pred_ag_tabular"),
        (OOF_TS,        "pred_ag_timeseries"),
        (OOF_ANCHOR,    "pred_anchor_master"),
        (OOF_YOY_DELTA, "pred_yoy_delta"),
    ]
    missing = [str(p) for p, _ in sources if not p.exists()]
    if missing:
        print(f"  combined OOF: missing source files: {missing}")
        return

    base = pd.read_csv(OOF_TABULAR, parse_dates=["Date"])[["Date", "Volume"]]
    for path, col in sources:
        df = pd.read_csv(path, parse_dates=["Date"])[["Date", col]]
        base = base.merge(df, on="Date", how="inner")

    pred_cols = [col for _, col in sources]
    base = base.dropna(subset=pred_cols).sort_values("Date").reset_index(drop=True)
    base.to_csv(OOF_COMBINED, index=False)

    # Print quick summary
    print(f"\n  combined OOF → {OOF_COMBINED}  ({len(base)} rows, "
          f"{base['Date'].min().date()} → {base['Date'].max().date()})")
    for col in pred_cols:
        mae = mean_absolute_error(base["Volume"], base[col])
        print(f"    {col:<25}  MAE {mae:>10,.0f}")


def update_oof(full_df: pd.DataFrame, force=False, max_age=6):
    print("\n[ OOF Update ]")
    if not _needs_update(OOF_UPDATED_AT, max_age, force):
        return

    update_ts_oof(full_df)
    update_tabular_oof(full_df)
    update_anchor_oof(full_df)
    update_yoy_delta_oof(full_df)
    rebuild_combined_oof()

    _mark_trained(OOF_UPDATED_AT, {"latest_date": str(full_df["Date"].max().date())})


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Train prediction models and update OOF")
    p.add_argument("--ts",           action="store_true", help="Force timeseries retrain")
    p.add_argument("--update-oof",   action="store_true", help="Force OOF update")
    p.add_argument("--all",          action="store_true", help="Force everything")
    p.add_argument("--no-oof",       action="store_true", help="Skip OOF update")
    p.add_argument("--ts-time-limit", type=int, default=DEFAULT_TS_TIME_LIMIT)
    p.add_argument("--ts-preset",    type=str, default=DEFAULT_TS_PRESET,
                   choices=["medium_quality", "good_quality", "high_quality", "best_quality"])
    p.add_argument("--max-age-ts",   type=int, default=6,
                   help="Max age (days) before TS retrain (default 6 = weekly)")
    p.add_argument("--max-age-oof",  type=int, default=6,
                   help="Max age (days) before OOF update (default 6 = weekly)")
    args = p.parse_args()

    print("Building feature dataframe...")
    df_full    = load_feature_df(prune=False)   # for ts3 + anchor + yoy_delta OOF
    print(f"  {len(df_full)} rows  {df_full['Date'].min().date()} → {df_full['Date'].max().date()}")

    train_timeseries(
        df_full,
        time_limit=args.ts_time_limit,
        preset=args.ts_preset,
        force=(args.ts or args.all),
        max_age=args.max_age_ts,
    )

    if not args.no_oof:
        update_oof(
            df_full,
            force=(args.update_oof or args.all),
            max_age=args.max_age_oof,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
