"""
refresh_normal_weights.py
==========================
Refreshes the production NORMAL-regime ensemble weights on demand.

Trains a *quick* AutoGluon tabular model and a quick AutoGluon timeseries
(TS3) model with a hard total wall-clock budget (default 30 minutes,
medium_quality preset), via the same walk-forward OOF folding used by
ag_tabular.py / ag_timeseries.py — so the resulting predictions are true
out-of-sample, not in-sample.

Those two fresh OOF series are merged with the existing (fast, deterministic)
yoy_delta_oof.csv and anchor_master_oof.csv, filtered to NORMAL-regime days,
and used to fit two Direct-MAE weight vectors:

    - full history-to-date  (all NORMAL days)
    - last 30 NORMAL days

which are blended 50/50 and written to
ensemble_experiment/output/dynamic_normal_weights.json. autogluon_predict.py
loads that file at import time (if present) and uses it in place of the
hardcoded NORMAL weight tuple — so re-running this script is what makes the
blend "dynamic": no code edits needed downstream.

This does NOT touch the production tabular model used for the actual
forecast (autogluon_full.py) or the existing best_quality experiment OOF
files (ag_tabular_oof.csv / ag_timeseries_oof.csv) — it writes to separate
quick_* files so it can't silently degrade anything else that depends on
those.

Usage:
    python ensemble_experiment/refresh_normal_weights.py
    python ensemble_experiment/refresh_normal_weights.py --minutes 20
    python ensemble_experiment/refresh_normal_weights.py --no-cache
    python ensemble_experiment/refresh_normal_weights.py --dry-run   # compute, don't write the JSON
"""

import argparse
import json
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from build_features import KEEP_FEATURES, TARGET_COL, build_features_from_df, load_and_merge_data
from production_router import (
    get_major_holiday_dates,
    days_to_prior_and_next_major,
    STORM_TRIGGER_IMPACT,
    PEAK_HOLIDAY_WINDOW,
    SHOULDER_PRE_WINDOW,
    SHOULDER_POST_WINDOW,
)

import normal_weight_search as nws  # sibling module (fit_weights / insample_mae / mae)

OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)
FOLD_CACHE_DIR = OUT_DIR / "quick_fold_predictions"
FOLD_CACHE_DIR.mkdir(exist_ok=True)

QUICK_TABULAR_OOF = OUT_DIR / "quick_ag_tabular_oof.csv"
QUICK_TIMESERIES_OOF = OUT_DIR / "quick_ag_timeseries_oof.csv"
YOY_DELTA_OOF = OUT_DIR / "yoy_delta_oof.csv"
ANCHOR_MASTER_OOF = OUT_DIR / "anchor_master_oof.csv"
WEATHER_PATH = ROOT / "data" / "weather_national_features_with_lags.csv"
DYNAMIC_WEIGHTS_PATH = OUT_DIR / "dynamic_normal_weights.json"

N_FOLDS = 3
CV_TEST_SIZE = 60
AG_PRESET = "medium_quality"
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
ITEM_ID = "tsa"
# yoy_delta dropped from production (2026-09-21) — see normal_weight_search.py.
MODEL_COLS = ["pred_ag_tabular", "pred_ag_timeseries", "pred_anchor_master"]
MODEL_NAMES = ["tab", "ts3", "anchor"]
LAST_N_DAYS = 30


# ── feature loading ─────────────────────────────────────────────────────────
def load_feature_df():
    print("Loading raw data + building features (prune=False)...")
    df = load_and_merge_data()
    df = build_features_from_df(df, verbose=False, prune=False)
    df = df[df[TARGET_COL].notna()].copy()
    essential = [f"{TARGET_COL}_lag{i}" for i in [1, 3, 7] if f"{TARGET_COL}_lag{i}" in df.columns]
    df = df.dropna(subset=essential).reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"  rows: {len(df):,}   date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ── quick tabular walk-forward OOF ──────────────────────────────────────────
def run_quick_tabular(df, fold_budget_s, use_cache=True):
    features = [f for f in KEEP_FEATURES if f in df.columns]
    print(f"\n{'='*70}\nQUICK TABULAR — {N_FOLDS} folds x {fold_budget_s:.0f}s, preset={AG_PRESET}, "
          f"{len(features)} features\n{'='*70}")

    tscv = TimeSeriesSplit(n_splits=N_FOLDS, test_size=CV_TEST_SIZE, gap=0)
    oof = np.full(len(df), np.nan)
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df), start=1):
        cache = FOLD_CACHE_DIR / f"quick_tabular_fold{fold_idx}.csv"
        val_df = df.iloc[val_idx].reset_index(drop=True)
        if use_cache and cache.exists():
            cached = pd.read_csv(cache, parse_dates=["Date"])
            if len(cached) == len(val_df) and (cached["Date"].values == val_df["Date"].values).all():
                print(f"  fold {fold_idx}: cached")
                oof[val_idx] = cached["pred"].values
                continue
            cache.unlink()

        train_df = df.iloc[train_idx].reset_index(drop=True)
        print(f"  fold {fold_idx}/{N_FOLDS}: train n={len(train_df)} "
              f"({train_df['Date'].min().date()} → {train_df['Date'].max().date()})  "
              f"val n={len(val_df)} ({val_df['Date'].min().date()} → {val_df['Date'].max().date()})")

        tmp_dir = OUT_DIR / f"_tmp_quick_tab_fold{fold_idx}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        cols = features + [TARGET_COL]
        predictor = TabularPredictor(
            label=TARGET_COL, path=str(tmp_dir),
            eval_metric="mean_absolute_error", verbosity=1,
        ).fit(train_data=train_df[cols], time_limit=fold_budget_s, presets=AG_PRESET)
        pred = predictor.predict(val_df[features]).values
        shutil.rmtree(tmp_dir, ignore_errors=True)

        pd.DataFrame({"Date": val_df["Date"].values, "pred": pred}).to_csv(cache, index=False)
        oof[val_idx] = pred
        fold_mae = mean_absolute_error(val_df[TARGET_COL], pred)
        print(f"    fold {fold_idx} MAE: {fold_mae:,.0f}")

    mask = ~np.isnan(oof)
    oof_df = pd.DataFrame({
        "Date": df["Date"].values[mask],
        "Volume": df[TARGET_COL].values[mask],
        "pred_ag_tabular": oof[mask],
    })
    oof_df.to_csv(QUICK_TABULAR_OOF, index=False)
    overall_mae = mean_absolute_error(oof_df["Volume"], oof_df["pred_ag_tabular"])
    print(f"  Saved → {QUICK_TABULAR_OOF.name}  (n={len(oof_df)}, MAE={overall_mae:,.0f})")
    return oof_df


# ── quick timeseries walk-forward OOF ───────────────────────────────────────
def to_ts_df(df, covariate_cols):
    cols = ["Date", TARGET_COL] + covariate_cols
    out = df[cols].copy()
    out["item_id"] = ITEM_ID
    out = out.rename(columns={"Date": "timestamp", TARGET_COL: "target"})
    return TimeSeriesDataFrame.from_data_frame(out, id_column="item_id", timestamp_column="timestamp")


def run_quick_timeseries(df, tabular_oof, fold_budget_s, use_cache=True):
    available = set(df.columns)
    covariates = [c for c in TS3_COVARIATES if c in available]
    print(f"\n{'='*70}\nQUICK TIMESERIES — {N_FOLDS} folds x {fold_budget_s:.0f}s, preset={AG_PRESET}, "
          f"{len(covariates)} covariates\n{'='*70}")

    n = len(tabular_oof)
    fold_size = n // N_FOLDS
    folds = []
    for f in range(N_FOLDS):
        start = tabular_oof.iloc[f * fold_size]["Date"]
        end_idx = (f + 1) * fold_size - 1 if f < N_FOLDS - 1 else n - 1
        end = tabular_oof.iloc[end_idx]["Date"]
        folds.append((start, end))

    pieces = []
    for fold_idx, (fold_start, fold_end) in enumerate(folds, start=1):
        cache = FOLD_CACHE_DIR / f"quick_ts_fold{fold_idx}.csv"
        if use_cache and cache.exists():
            cached = pd.read_csv(cache, parse_dates=["Date"])
            if not cached.empty:
                print(f"  fold {fold_idx}: cached")
                pieces.append(cached)
                continue

        train_df = df[df["Date"] < fold_start].reset_index(drop=True)
        fold_df = df[(df["Date"] >= fold_start) & (df["Date"] <= fold_end)].reset_index(drop=True)
        print(f"  fold {fold_idx}/{N_FOLDS}: train n={len(train_df)} "
              f"({train_df['Date'].min().date()} → {train_df['Date'].max().date()})  "
              f"val {fold_start.date()} → {fold_end.date()}")

        train_ts = to_ts_df(train_df, covariates)
        tmp_path = OUT_DIR / f"_tmp_quick_ts_fold{fold_idx}"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        predictor = TimeSeriesPredictor(
            prediction_length=1, path=str(tmp_path), target="target", eval_metric="MAE",
            known_covariates_names=covariates if covariates else None, verbosity=1,
        ).fit(train_ts, presets=AG_PRESET, time_limit=fold_budget_s)

        preds = []
        for i in range(len(fold_df)):
            forecast_date = fold_df.iloc[i]["Date"]
            history = df[df["Date"] < forecast_date]
            history_ts = to_ts_df(history, covariates)
            known_kw = {}
            if covariates:
                future_row = fold_df.iloc[[i]][["Date"] + covariates].copy()
                future_row["item_id"] = ITEM_ID
                future_row = future_row.rename(columns={"Date": "timestamp"})
                known_kw["known_covariates"] = TimeSeriesDataFrame.from_data_frame(
                    future_row, id_column="item_id", timestamp_column="timestamp"
                )
            forecast = predictor.predict(history_ts, **known_kw)
            col = "mean" if "mean" in forecast.columns else forecast.columns[0]
            preds.append({"Date": forecast_date, "pred": float(forecast.iloc[0][col])})

        shutil.rmtree(tmp_path, ignore_errors=True)
        fold_out = pd.DataFrame(preds)
        fold_out.to_csv(cache, index=False)
        pieces.append(fold_out)

        merged = fold_df[["Date", TARGET_COL]].merge(fold_out, on="Date", how="left")
        fold_mae = mean_absolute_error(merged[TARGET_COL], merged["pred"])
        print(f"    fold {fold_idx} MAE: {fold_mae:,.0f}")

    stacked = pd.concat(pieces, ignore_index=True).rename(columns={"pred": "pred_ag_timeseries"})
    out = tabular_oof[["Date"]].merge(stacked, on="Date", how="left")
    out.to_csv(QUICK_TIMESERIES_OOF, index=False)
    valid = out.dropna(subset=["pred_ag_timeseries"])
    print(f"  Saved → {QUICK_TIMESERIES_OOF.name}  (n={len(out)})")
    return out


# ── regime classification (same convention as normal_weight_search.py) ─────
def classify_normal_regime(df):
    weather = (pd.read_csv(WEATHER_PATH, parse_dates=["date"])
               .rename(columns={"date": "Date"})[["Date", "vol_wtd_storm_impact"]])
    d = df.merge(weather, on="Date", how="left")
    d["vol_wtd_storm_impact"] = d["vol_wtd_storm_impact"].fillna(0.0)
    d["storm_impact_sq"] = d["vol_wtd_storm_impact"] ** 2
    d["storm_severe_flag"] = (d["vol_wtd_storm_impact"] > 0.5).astype(int)

    years = sorted(d["Date"].dt.year.unique())
    holidays = get_major_holiday_dates(range(min(years) - 1, max(years) + 2))
    prior_next = d["Date"].apply(
        lambda x: pd.Series(days_to_prior_and_next_major(x, holidays), index=["days_prior", "days_next"])
    )
    d[["days_prior", "days_next"]] = prior_next

    def classify(row):
        if row["storm_severe_flag"] == 1:
            return "severe_storm" if row["storm_impact_sq"] >= STORM_TRIGGER_IMPACT else "moderate_storm"
        p, n = int(row["days_prior"]), int(row["days_next"])
        if min(p, n) <= PEAK_HOLIDAY_WINDOW:
            return "peak_holiday"
        if PEAK_HOLIDAY_WINDOW < p <= SHOULDER_POST_WINDOW:
            return "shoulder_post"
        if PEAK_HOLIDAY_WINDOW < n <= SHOULDER_PRE_WINDOW:
            return "shoulder_pre"
        return "normal"

    d["regime"] = d.apply(classify, axis=1)
    return d[d["regime"] == "normal"].sort_values("Date").reset_index(drop=True)


# ── weight fitting ──────────────────────────────────────────────────────────
def fit_direct_mae(df):
    X = df[MODEL_COLS].values.astype(float)
    y = df["Volume"].values.astype(float)
    _, w = nws.insample_mae(X, y, "mae")
    return w


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=30.0,
                   help="Total wall-clock budget for the two quick AutoGluon retrains (default 30).")
    p.add_argument("--no-cache", action="store_true", help="Ignore per-fold cache, retrain everything.")
    p.add_argument("--dry-run", action="store_true", help="Compute the blend but don't write the JSON.")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    total_budget_s = args.minutes * 60
    tab_budget_s = total_budget_s * 0.5
    ts_budget_s = total_budget_s * 0.5
    tab_fold_budget = tab_budget_s / N_FOLDS
    ts_fold_budget = ts_budget_s / N_FOLDS
    print(f"Total budget: {args.minutes:.0f} min  →  tabular {tab_budget_s/60:.1f} min "
          f"({tab_fold_budget:.0f}s/fold), timeseries {ts_budget_s/60:.1f} min ({ts_fold_budget:.0f}s/fold)")

    if not YOY_DELTA_OOF.exists() or not ANCHOR_MASTER_OOF.exists():
        raise FileNotFoundError(
            "yoy_delta_oof.csv / anchor_master_oof.csv missing under ensemble_experiment/output/. "
            "Run yoy_delta_oof.py and anchor_master_oof.py first (they're fast, no AutoGluon)."
        )

    df = load_feature_df()
    tab_oof = run_quick_tabular(df, tab_fold_budget, use_cache=not args.no_cache)
    ts_oof = run_quick_timeseries(df, tab_oof, ts_fold_budget, use_cache=not args.no_cache)

    elapsed = time.time() - t0
    print(f"\nQuick retrain wall time: {elapsed/60:.1f} min")

    # ── merge the 4 model OOF series ────────────────────────────────────────
    yoy = pd.read_csv(YOY_DELTA_OOF, parse_dates=["Date"])[["Date", "pred_yoy_delta"]]
    anc = pd.read_csv(ANCHOR_MASTER_OOF, parse_dates=["Date"])[["Date", "pred_anchor_master"]]

    merged = tab_oof.merge(ts_oof[["Date", "pred_ag_timeseries"]], on="Date", how="inner")
    merged = merged.merge(yoy, on="Date", how="inner")
    merged = merged.merge(anc, on="Date", how="inner")
    merged = merged.dropna(subset=MODEL_COLS).reset_index(drop=True)
    print(f"\nMerged 4-model OOF: n={len(merged)}  "
          f"({merged['Date'].min().date()} → {merged['Date'].max().date()})")

    normal_df = classify_normal_regime(merged)
    print(f"NORMAL-regime days: n={len(normal_df)}  "
          f"({normal_df['Date'].min().date()} → {normal_df['Date'].max().date()})")
    if len(normal_df) < 60:
        raise RuntimeError(f"Only {len(normal_df)} NORMAL days available — too few to fit weights reliably.")

    last30 = normal_df.tail(LAST_N_DAYS)

    print("\nPer-model MAE (all NORMAL days):")
    for col, name in zip(MODEL_COLS, MODEL_NAMES):
        m = mean_absolute_error(normal_df["Volume"], normal_df[col])
        print(f"  {name:<12} {m:>10,.0f}")

    w_full = fit_direct_mae(normal_df)
    w_l30 = fit_direct_mae(last30)
    w_blend = 0.5 * w_full + 0.5 * w_l30
    w_blend = np.clip(w_blend, 0, None)
    w_blend = w_blend / w_blend.sum()

    print("\n" + "=" * 70)
    print(f"{'':14}" + "".join(f"{n:>12}" for n in MODEL_NAMES))
    print(f"{'full-history':14}" + "".join(f"{v:>12.3f}" for v in w_full))
    print(f"{'last-30d':14}" + "".join(f"{v:>12.3f}" for v in w_l30))
    print(f"{'BLEND (used)':14}" + "".join(f"{v:>12.3f}" for v in w_blend))
    print("=" * 70)

    def mae_of(w, d):
        pred = d[MODEL_COLS].values.astype(float) @ w
        return mean_absolute_error(d["Volume"], pred)

    print(f"\nIn-sample MAE check (full NORMAL set): full-history={mae_of(w_full, normal_df):,.0f}  "
          f"last-30={mae_of(w_l30, normal_df):,.0f}  blend={mae_of(w_blend, normal_df):,.0f}")

    if args.dry_run:
        print("\n--dry-run: not writing dynamic_normal_weights.json")
        return

    # yoy_delta dropped from production (2026-09-21) but consumers
    # (autogluon_predict.py / dashboard main.py) still unpack a 4-slot
    # (tab, ts3, yoy_delta, anchor) tuple — write it back in that shape
    # with yoy_delta pinned at 0 rather than changing every call site.
    def _with_yoy_zero(names, weights):
        out_names, out_weights = [], []
        for n, w in zip(names, weights):
            out_names.append(n)
            out_weights.append(round(float(w), 4))
            if n == "ts3":
                out_names.append("yoy_delta")
                out_weights.append(0.0)
        return out_names, out_weights

    order4, blend4 = _with_yoy_zero(MODEL_NAMES, w_blend)

    payload = {
        "regime": "NORMAL",
        "model_order": order4,
        "weights_tuple": blend4,
        "method": "50/50 blend of Direct-MAE full-history-to-date and last-30-NORMAL-day fits "
                  "(yoy_delta excluded from fitting, pinned to 0 — see normal_weight_search.py)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_full_history": int(len(normal_df)),
        "n_last30": int(len(last30)),
        "date_range": [str(normal_df["Date"].min().date()), str(normal_df["Date"].max().date())],
        "full_history_weights": [round(float(x), 4) for x in w_full],
        "last30_weights": [round(float(x), 4) for x in w_l30],
        "quick_tabular_oof_mae": round(float(mean_absolute_error(tab_oof["Volume"], tab_oof["pred_ag_tabular"])), 1),
    }
    with open(DYNAMIC_WEIGHTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote → {DYNAMIC_WEIGHTS_PATH}")
    print("autogluon_predict.py will pick this up automatically on its next run.")


if __name__ == "__main__":
    main()
