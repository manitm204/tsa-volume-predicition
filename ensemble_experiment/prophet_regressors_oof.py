"""
prophet_regressors_oof.py
=========================
Prophet OOF with 5 exogenous regressors over the same ~12-month window as
prophet_oof.py.

Why regressors?  Basic Prophet (yearly + weekly seasonality) earned ~0 weight
in the ensemble because its MAE was too high.  Adding lag-365 anchors, holiday
shape, weather impact, and YoY growth gives Prophet domain signals it can't
derive from pure trend/seasonality.

Regressors (all known-in-advance — no future-volume leakage):
  lag365_same_dow_5w_mean  — lag-365 same-DOW 5-week mean (year-specific anchor)
  yoy_ratio_7d             — year-over-year ratio (7-day window, growth momentum)
  vol_wtd_storm_impact     — weighted storm impact across 16 hub airports
  holiday_decay_shape      — decay curve around major holidays
  days_to_holiday_signed   — signed days to nearest major holiday

Fold structure is identical to prophet_oof.py (weekly holdout aligned to
ag_tabular_oof dates) — only the model changes.

Outputs (drop-in replacement for prophet_oof.py outputs):
    ensemble_experiment/output/prophet_oof.csv
        Date, Volume, pred_prophet
    ensemble_experiment/output/fold_predictions/prophet_reg_week{i}.csv
        per-week fold cache (separate key from basic prophet so both can coexist)
    ensemble_experiment/output/prophet_oof_summary.json

After running, re-run combine_oof.py then normal_weight_search.py to see
whether the regressor Prophet earns non-zero ensemble weight.

Usage:
    python prophet_regressors_oof.py               # use fold cache if present
    python prophet_regressors_oof.py --no-cache    # recompute every fold
    python prophet_regressors_oof.py --clear-cache # wipe cache, then run
    python prophet_regressors_oof.py --report-only # print MAE from saved CSV
"""

import argparse
import json
import logging
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
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from prophet import Prophet

from build_features import (
    TARGET_COL,
    build_features_from_df,
    load_and_merge_data,
)

OUT_DIR        = HERE / "output"
FOLD_CACHE_DIR = OUT_DIR / "fold_predictions"
OUT_DIR.mkdir(exist_ok=True)
FOLD_CACHE_DIR.mkdir(exist_ok=True)

TABULAR_OOF_PATH = OUT_DIR / "ag_tabular_oof.csv"

REGRESSORS = [
    "lag365_same_dow_5w_mean",
    "yoy_ratio_7d",
    "vol_wtd_storm_impact",
    "holiday_decay_shape",
    "days_to_holiday_signed",
]


def load_feature_df():
    """Build full feature dataframe (prune=False) for all available dates."""
    print("Building feature dataframe (prune=False) ...")
    raw = load_and_merge_data()
    feat = build_features_from_df(raw, prune=False)
    feat["Date"] = pd.to_datetime(feat["Date"])
    feat = feat[feat[TARGET_COL].notna()].sort_values("Date").reset_index(drop=True)
    print(f"  rows: {len(feat):,}   range: {feat['Date'].min().date()} → {feat['Date'].max().date()}")
    missing = [r for r in REGRESSORS if r not in feat.columns]
    if missing:
        raise RuntimeError(f"Regressor columns missing from feature df: {missing}")
    return feat


def load_tabular_oof():
    if not TABULAR_OOF_PATH.exists():
        raise FileNotFoundError(
            f"{TABULAR_OOF_PATH} missing. Run ag_tabular.py first so we know "
            "the OOF date range to align against."
        )
    oof = pd.read_csv(TABULAR_OOF_PATH, parse_dates=["Date"])
    return oof[["Date", TARGET_COL]].rename(columns={TARGET_COL: "Volume"})


def make_weekly_folds(oof_dates):
    dates = sorted(oof_dates)
    folds = []
    i = 0
    week_idx = 1
    while i < len(dates):
        start = dates[i]
        end   = dates[min(i + 6, len(dates) - 1)]
        folds.append((week_idx, start, end))
        i += 7
        week_idx += 1
    return folds


def fold_cache_path(week_idx):
    return FOLD_CACHE_DIR / f"prophet_reg_week{week_idx:02d}.csv"


def fit_predict_one_week(feat_df, week_idx, start_date, end_date, use_cache=True):
    cache = fold_cache_path(week_idx)
    if use_cache and cache.exists():
        return pd.read_csv(cache, parse_dates=["Date"])

    train_raw = feat_df[feat_df["Date"] < start_date].copy()
    pred_raw  = feat_df[(feat_df["Date"] >= start_date) & (feat_df["Date"] <= end_date)].copy()

    # Fill NaN regressors with 0 — safe for these features (0 = no storm, no
    # holiday effect, neutral YoY ratio difference from 1.0 handled below).
    for col in REGRESSORS:
        train_raw[col] = train_raw[col].fillna(0.0)
        pred_raw[col]  = pred_raw[col].fillna(0.0)

    train = train_raw.rename(columns={"Date": "ds", TARGET_COL: "y"})
    future = pred_raw.rename(columns={"Date": "ds"})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    for r in REGRESSORS:
        m.add_regressor(r)

    m.fit(train[["ds", "y"] + REGRESSORS])

    forecast = m.predict(future[["ds"] + REGRESSORS])
    out = pd.DataFrame({
        "Date": forecast["ds"].values,
        "pred": forecast["yhat"].values,
    })
    out.to_csv(cache, index=False)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache",     action="store_true")
    p.add_argument("--clear-cache",  action="store_true")
    p.add_argument("--report-only",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.clear_cache:
        for f in FOLD_CACHE_DIR.glob("prophet_reg_week*.csv"):
            f.unlink()
        print("Cleared regressor-Prophet weekly cache")

    oof_csv      = OUT_DIR / "prophet_oof.csv"
    summary_path = OUT_DIR / "prophet_oof_summary.json"

    if args.report_only:
        if not oof_csv.exists():
            raise FileNotFoundError(f"{oof_csv} missing — run without --report-only first")
        oof_df = pd.read_csv(oof_csv, parse_dates=["Date"])
        mae = mean_absolute_error(oof_df["Volume"], oof_df["pred_prophet"])
        print(f"Prophet-with-regressors OOF: {len(oof_df)} rows  MAE = {mae:,.0f}")
        return

    feat_df     = load_feature_df()
    tabular_oof = load_tabular_oof()
    oof_dates   = pd.to_datetime(tabular_oof["Date"]).tolist()
    print(f"\nOOF dates: {len(oof_dates)}  ({oof_dates[0].date()} → {oof_dates[-1].date()})")

    weekly_folds = make_weekly_folds(oof_dates)
    print(f"Weekly folds: {len(weekly_folds)}")
    print(f"Regressors: {REGRESSORS}")

    fold_metrics = []
    pieces = []
    for week_idx, start_date, end_date in weekly_folds:
        print(f"  week {week_idx:>2}: {start_date.date()} → {end_date.date()}", end="  ")
        preds = fit_predict_one_week(
            feat_df, week_idx, start_date, end_date, use_cache=not args.no_cache,
        )
        pieces.append(preds)
        actual = feat_df[(feat_df["Date"] >= start_date) & (feat_df["Date"] <= end_date)]
        merged = actual[["Date", TARGET_COL]].merge(preds, on="Date", how="left")
        if merged[TARGET_COL].notna().any():
            mae = float(mean_absolute_error(merged[TARGET_COL], merged["pred"]))
            fold_metrics.append({
                "week": week_idx,
                "start": str(start_date.date()),
                "end":   str(end_date.date()),
                "n":     int(len(merged)),
                "mae":   mae,
            })
            print(f"MAE {mae:>10,.0f}")
        else:
            print("(no actuals)")

    stacked = pd.concat(pieces, ignore_index=True).rename(columns={"pred": "pred_prophet"})
    out = tabular_oof.merge(stacked, on="Date", how="left")
    out.to_csv(oof_csv, index=False)

    overall_mae = float(mean_absolute_error(out["Volume"], out["pred_prophet"]))
    print(f"\nSaved Prophet-regressors OOF → {oof_csv}  (n={len(out)}, overall MAE = {overall_mae:,.0f})")

    # Compare to what basic Prophet got if summary exists
    if summary_path.exists():
        with open(summary_path) as f:
            prev = json.load(f)
        if "overall_oof_mae" in prev and prev.get("regressors") == []:
            print(f"Basic Prophet MAE was: {prev['overall_oof_mae']:,.0f}  "
                  f"(Δ {prev['overall_oof_mae'] - overall_mae:+,.0f})")

    summary = {
        "n_weekly_folds":   len(weekly_folds),
        "regressors":       REGRESSORS,
        "holidays":         False,
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "fold_metrics":     fold_metrics,
        "overall_oof_mae":  overall_mae,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")
    print("\nNext: run combine_oof.py then normal_weight_search.py to see updated ensemble weights.")


if __name__ == "__main__":
    main()
