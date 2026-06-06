"""
autogluon_full.py
=================
Production AutoGluon model — trains on ALL available data.

Workflow:
    1. Walk-forward CV (4 folds × 91 days) → OOF predictions + MAE
    2. Train final model on ALL data → saved to output_autogluon_best/ag_final
    3. OOF residuals saved → used by autogluon_predict.py for uncertainty

OOF MAE is reported for reference but is NOT a held-out evaluation
(the final model trains on more data than any CV fold).

Usage:
    python autogluon_full.py

Outputs to output_autogluon_best/
    ag_final/          — final model (used by autogluon_predict.py)
    oof_predictions.csv
    cv_fold_metrics.csv
    feature_importance.csv
"""

import warnings
warnings.filterwarnings("ignore")

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from autogluon.tabular import TabularPredictor

from build_features import (
    load_master, get_feature_columns,
    rmse, smape,
    seasonal_naive, lag365_naive,
    transform_target, inverse_transform_target, USE_LOG_TARGET,
    TARGET_COL, RANDOM_STATE,
)

OUT_DIR = Path("./output_autogluon_best_google_trends")
OUT_DIR.mkdir(exist_ok=True)

FINAL_MODEL_DIR = OUT_DIR / "ag_final"

N_CV_SPLITS = 4
CV_TEST_SIZE = 91
CV_GAP = 0
TIME_LIMIT_CV = 15 * 60       # seconds per fold
TIME_LIMIT_FINAL = 30 * 60   # seconds for final model
AG_PRESET = "best_quality"
AG_VERBOSITY = 1


# ==========================================================================
# Walk-forward CV
# ==========================================================================
def run_cv(df):
    feature_cols = get_feature_columns(df)
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS, test_size=CV_TEST_SIZE, gap=CV_GAP)

    oof = np.full(len(df), np.nan)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df[feature_cols]), start=1):
        train_df = df.iloc[train_idx][feature_cols + [TARGET_COL]].copy()
        test_df  = df.iloc[test_idx ][feature_cols + [TARGET_COL]].copy()
        y_train  = df.iloc[train_idx][TARGET_COL]
        y_test   = df.iloc[test_idx ][TARGET_COL]
        h        = len(y_test)

        print(f"\n{'=' * 60}")
        print(f"Fold {fold}  train {len(train_idx):,}  test {len(test_idx):,}  "
              f"({df.iloc[test_idx[0]]['Date'].date()} → {df.iloc[test_idx[-1]]['Date'].date()})")
        print("=" * 60)

        for bname, bpred in [("seasonal_naive", seasonal_naive(y_train, h)),
                              ("lag365",         lag365_naive(y_train, h))]:
            fold_results.append({
                "model": bname, "fold": fold,
                "mae":   mean_absolute_error(y_test, bpred),
                "rmse":  rmse(y_test, bpred),
                "smape": smape(y_test, bpred),
            })

        target_col_ag = TARGET_COL
        if USE_LOG_TARGET:
            target_col_ag = "log_target"
            train_df[target_col_ag] = transform_target(train_df[TARGET_COL])
            train_df = train_df.drop(columns=[TARGET_COL])
        test_input = test_df.drop(columns=[TARGET_COL])

        ag_path = OUT_DIR / f"ag_fold_{fold}"
        if ag_path.exists():
            shutil.rmtree(ag_path)

        predictor = TabularPredictor(
            label=target_col_ag,
            path=str(ag_path),
            eval_metric="mean_absolute_error",
            verbosity=AG_VERBOSITY,
        ).fit(train_data=train_df, time_limit=TIME_LIMIT_CV, presets=AG_PRESET)

        pred_raw = predictor.predict(test_input)
        pred = inverse_transform_target(pred_raw.values) if USE_LOG_TARGET else pred_raw.values
        oof[test_idx] = pred

        mae_val   = mean_absolute_error(y_test, pred)
        rmse_val  = rmse(y_test, pred)
        smape_val = smape(y_test, pred)
        print(f"\n  AutoGluon  MAE {mae_val:>10,.0f}  RMSE {rmse_val:>10,.0f}  sMAPE {smape_val:.2f}%")
        fold_results.append({"model": "autogluon", "fold": fold,
                              "mae": mae_val, "rmse": rmse_val, "smape": smape_val})

        lb = predictor.leaderboard(silent=True)
        print(lb.head(5)[["model", "score_val", "fit_time"]].to_string(index=False))
        shutil.rmtree(ag_path, ignore_errors=True)

    return pd.DataFrame(fold_results), oof


# ==========================================================================
# Final model on ALL data
# ==========================================================================
def train_final(df):
    feature_cols = get_feature_columns(df)
    if FINAL_MODEL_DIR.exists():
        shutil.rmtree(FINAL_MODEL_DIR)

    target_col_ag = TARGET_COL
    train_data = df[feature_cols + [TARGET_COL]].copy()
    if USE_LOG_TARGET:
        target_col_ag = "log_target"
        train_data[target_col_ag] = transform_target(train_data[TARGET_COL])
        train_data = train_data.drop(columns=[TARGET_COL])

    print(f"  Training on ALL {len(df):,} rows, {len(feature_cols)} features")
    print(f"  Preset: {AG_PRESET}  time_limit: {TIME_LIMIT_FINAL}s")

    predictor = TabularPredictor(
        label=target_col_ag,
        path=str(FINAL_MODEL_DIR),
        eval_metric="mean_absolute_error",
        verbosity=AG_VERBOSITY,
    ).fit(train_data=train_data, time_limit=TIME_LIMIT_FINAL, presets=AG_PRESET)

    lb = predictor.leaderboard(silent=True)
    print(lb.head(5)[["model", "score_val", "fit_time"]].to_string(index=False))
    return predictor


# ==========================================================================
# Main
# ==========================================================================
def main():
    df = load_master()
    feature_cols = get_feature_columns(df)
    print(f"Loaded master: {len(df):,} rows, {len(feature_cols)} features")
    print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"Preset: {AG_PRESET}  CV: {N_CV_SPLITS} folds × {CV_TEST_SIZE} days")

    # ── Walk-forward CV ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Walk-Forward CV (reference, not held-out)")
    print("=" * 80)
    fold_df, oof = run_cv(df)

    valid = ~np.isnan(oof)
    oof_mae   = mean_absolute_error(df.loc[valid, TARGET_COL], oof[valid])
    oof_rmse  = rmse(df.loc[valid, TARGET_COL], oof[valid])
    oof_smape = smape(df.loc[valid, TARGET_COL], oof[valid])
    print(f"\nOOF  MAE {oof_mae:,.0f}  RMSE {oof_rmse:,.0f}  sMAPE {oof_smape:.2f}%")

    print("\nPer-fold MAE:")
    for m in ["autogluon", "seasonal_naive", "lag365"]:
        rows = fold_df[fold_df["model"] == m]
        if not rows.empty:
            vals = rows["mae"].values
            print(f"  {m:>16}: {' | '.join(f'{v:>10,.0f}' for v in vals)}  avg {np.mean(vals):>10,.0f}")

    # ── Final model on ALL data ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("Training Final Model on ALL Data")
    print("=" * 80)
    predictor = train_final(df)

    # ── Feature importance ────────────────────────────────────────────
    try:
        eval_data = df.iloc[-CV_TEST_SIZE:][feature_cols + [TARGET_COL]]
        imp = predictor.feature_importance(data=eval_data, num_shuffle_sets=5)
        imp.to_csv(OUT_DIR / "feature_importance.csv")
        print(f"\nTop 20 features:")
        print(imp.head(20).to_string())
    except Exception as e:
        print(f"Feature importance skipped: {e}")

    # ── Save OOF predictions (used by autogluon_predict.py) ───────────
    oof_df = df.loc[valid, ["Date", TARGET_COL]].copy()
    oof_df["pred_autogluon"] = oof[valid]
    oof_df["abs_error"] = np.abs(oof_df[TARGET_COL] - oof_df["pred_autogluon"])
    oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    fold_df.to_csv(OUT_DIR / "cv_fold_metrics.csv", index=False)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"  OOF MAE:     {oof_mae:>10,.0f}")
    print(f"  OOF RMSE:    {oof_rmse:>10,.0f}")
    print(f"  Final model: {FINAL_MODEL_DIR}")
    print(f"  OOF file:    {OUT_DIR / 'oof_predictions.csv'}")
    print(f"  → Run autogluon_predict.py for weekly Kalshi predictions")


if __name__ == "__main__":
    main()
