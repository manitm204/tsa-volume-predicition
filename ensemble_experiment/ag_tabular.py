"""
ag_tabular.py
=============
AutoGluon Tabular on KEEP_FEATURES (34 features) with best_quality preset.

  * Walk-forward OOF: 4 folds × 91 days (sklearn TimeSeriesSplit), matching
    the existing output_ensemble_experiment/oof_predictions.csv dates so the
    OOF rows produced here line up 1:1 with ag_timeseries.py / prophet_oof.py
    / anchor_master_oof.py.
  * Per-fold time limit: 300s.
  * After the 4 OOF folds, one final production-style model is trained on
    ALL available data with a 600s limit and saved under
    ensemble_experiment/output/ag_final/. This model is NOT part of OOF; it
    exists so the experiment's "best" tabular can be inspected / reused.

Outputs (all under ensemble_experiment/output/):
    ag_tabular_oof.csv            Date, Volume, pred_ag_tabular
    fold_predictions/ag_tabular_fold{1..4}.csv   per-fold cache (resumable)
    ag_final/                     final full-data AG model (production-style)
    ag_tabular_summary.json       fold MAEs + overall MAE

Resumable: re-running skips folds whose cache exists. Use --no-cache to
force retrain or --report-only to skip training entirely.
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
from sklearn.model_selection import TimeSeriesSplit

# Make `build_features` importable whether run from project root or this folder.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from autogluon.tabular import TabularPredictor

from build_features import (
    KEEP_FEATURES,
    TARGET_COL,
    build_features_from_df,
    load_and_merge_data,
)

OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)
FOLD_CACHE_DIR = OUT_DIR / "fold_predictions"
FOLD_CACHE_DIR.mkdir(exist_ok=True)
FINAL_MODEL_DIR = OUT_DIR / "ag_final"

N_CV_SPLITS = 4
CV_TEST_SIZE = 91
FOLD_TIME_LIMIT = 300       # 5 minutes per walk-forward fold
FINAL_TIME_LIMIT = 600      # 10 minutes for the final full-data fit
AG_PRESET = "best_quality"
AG_VERBOSITY = 1


def load_feature_df():
    print("Loading raw data + building all engineered features (prune=False)...")
    df = load_and_merge_data()
    df = build_features_from_df(df, verbose=False, prune=False)
    df = df[df[TARGET_COL].notna()].copy()
    essential = [f"{TARGET_COL}_lag{i}" for i in [1, 3, 7] if f"{TARGET_COL}_lag{i}" in df.columns]
    df = df.dropna(subset=essential).reset_index(drop=True)
    print(f"  rows: {len(df):,}   date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def select_keep_features(df):
    available = set(df.columns)
    present = [f for f in KEEP_FEATURES if f in available]
    missing = [f for f in KEEP_FEATURES if f not in available]
    print(f"\nKEEP_FEATURES: {len(present)} present, {len(missing)} missing")
    if missing:
        print(f"  missing: {missing}")
    return present


def fold_cache_path(fold_idx):
    return FOLD_CACHE_DIR / f"ag_tabular_fold{fold_idx}.csv"


def train_one_fold(train_df, val_df, features, fold_idx, use_cache=True):
    cache = fold_cache_path(fold_idx)
    if use_cache and cache.exists():
        cached = pd.read_csv(cache, parse_dates=["Date"])
        if len(cached) == len(val_df) and (cached["Date"].values == val_df["Date"].values).all():
            print(f"  fold {fold_idx}: cached ({len(cached)} preds)")
            return cached["pred"].values
        cache.unlink()

    tmp_dir = OUT_DIR / f"_tmp_fold{fold_idx}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    cols = features + [TARGET_COL]
    predictor = TabularPredictor(
        label=TARGET_COL,
        path=str(tmp_dir),
        eval_metric="mean_absolute_error",
        verbosity=AG_VERBOSITY,
    ).fit(
        train_data=train_df[cols],
        time_limit=FOLD_TIME_LIMIT,
        presets=AG_PRESET,
    )
    pred = predictor.predict(val_df[features]).values

    pd.DataFrame({"Date": val_df["Date"].values, "pred": pred}).to_csv(cache, index=False)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return pred


def run_walk_forward(df, features, use_cache=True):
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS, test_size=CV_TEST_SIZE, gap=0)
    oof = np.full(len(df), np.nan)
    fold_maes = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        print(f"\n--- Fold {fold_idx}/{N_CV_SPLITS}  "
              f"train n={len(train_df)} ({train_df['Date'].min().date()} → {train_df['Date'].max().date()})  "
              f"val n={len(val_df)} ({val_df['Date'].min().date()} → {val_df['Date'].max().date()}) ---")
        pred = train_one_fold(train_df, val_df, features, fold_idx, use_cache=use_cache)
        oof[val_idx] = pred
        fold_mae = float(mean_absolute_error(val_df[TARGET_COL].values, pred))
        fold_maes.append({
            "fold": fold_idx,
            "start": str(val_df["Date"].iloc[0].date()),
            "end": str(val_df["Date"].iloc[-1].date()),
            "n": int(len(val_df)),
            "mae": fold_mae,
        })
        print(f"  fold {fold_idx} MAE: {fold_mae:,.0f}")
    return oof, fold_maes


def fit_final_full_data(df, features):
    """Train one model on ALL data with the longer budget. Saved to
    ensemble_experiment/output/ag_final/ for downstream inspection/reuse."""
    print("\n" + "=" * 70)
    print(f"Final fit on all data (time_limit={FINAL_TIME_LIMIT}s, best_quality)")
    print(f"  rows: {len(df):,}   features: {len(features)}")
    print("=" * 70)
    if FINAL_MODEL_DIR.exists():
        shutil.rmtree(FINAL_MODEL_DIR)
    cols = features + [TARGET_COL]
    TabularPredictor(
        label=TARGET_COL,
        path=str(FINAL_MODEL_DIR),
        eval_metric="mean_absolute_error",
        verbosity=AG_VERBOSITY,
    ).fit(
        train_data=df[cols],
        time_limit=FINAL_TIME_LIMIT,
        presets=AG_PRESET,
    )
    print(f"Final model saved → {FINAL_MODEL_DIR}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore per-fold cache and retrain everything.")
    p.add_argument("--clear-cache", action="store_true",
                   help="Delete the fold_predictions cache before running.")
    p.add_argument("--report-only", action="store_true",
                   help="Skip training, recompute summary from existing OOF CSV.")
    p.add_argument("--skip-final", action="store_true",
                   help="Run only the 4 walk-forward folds; skip the final full-data fit.")
    return p.parse_args()


def main():
    args = parse_args()
    oof_csv = OUT_DIR / "ag_tabular_oof.csv"
    summary_path = OUT_DIR / "ag_tabular_summary.json"

    if args.clear_cache and FOLD_CACHE_DIR.exists():
        shutil.rmtree(FOLD_CACHE_DIR)
        FOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cleared {FOLD_CACHE_DIR}")

    df = load_feature_df()
    features = select_keep_features(df)

    if args.report_only:
        if not oof_csv.exists():
            raise FileNotFoundError(f"{oof_csv} missing — cannot --report-only")
        oof_df = pd.read_csv(oof_csv, parse_dates=["Date"])
        y = oof_df[TARGET_COL].values
        print(f"\nLoaded cached OOF: {len(oof_df)} rows  "
              f"MAE = {mean_absolute_error(y, oof_df['pred_ag_tabular'].values):,.0f}")
        return

    oof, fold_maes = run_walk_forward(df, features, use_cache=not args.no_cache)

    mask = ~np.isnan(oof)
    oof_df = pd.DataFrame({
        "Date": df["Date"].values[mask],
        TARGET_COL: df[TARGET_COL].values[mask],
        "pred_ag_tabular": oof[mask],
    })
    oof_df.to_csv(oof_csv, index=False)
    overall_mae = float(mean_absolute_error(oof_df[TARGET_COL], oof_df["pred_ag_tabular"]))
    print(f"\nSaved OOF → {oof_csv}  (n={len(oof_df)}, overall MAE = {overall_mae:,.0f})")

    summary = {
        "n_folds": N_CV_SPLITS,
        "fold_test_size": CV_TEST_SIZE,
        "fold_time_limit_s": FOLD_TIME_LIMIT,
        "final_time_limit_s": FINAL_TIME_LIMIT,
        "preset": AG_PRESET,
        "n_features": len(features),
        "features": features,
        "fold_metrics": fold_maes,
        "overall_oof_mae": overall_mae,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")

    if not args.skip_final:
        fit_final_full_data(df, features)


if __name__ == "__main__":
    main()
