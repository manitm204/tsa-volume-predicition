"""
prophet_oof.py
==============
Univariate Prophet OOF over the same ~12-month window as ag_tabular.py.

  * Folds are weekly: starting from the first AG-Tabular OOF date, step
    forward in 7-day chunks. For each chunk, fit Prophet on (Date, Volume)
    for all data strictly before the chunk start and predict the chunk.
    ~52 fits, each takes a few seconds.
  * No regressors, no holidays — just the default Prophet (yearly + weekly
    seasonality on the log of volume by default; we use additive linear
    regression on raw volume, matching what bare Prophet does).

Outputs (under ensemble_experiment/output/):
    prophet_oof.csv                       Date, Volume, pred_prophet
    fold_predictions/prophet_week{i}.csv  per-week cache (resumable)
    prophet_oof_summary.json
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
    load_and_merge_data,
)

OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)
FOLD_CACHE_DIR = OUT_DIR / "fold_predictions"
FOLD_CACHE_DIR.mkdir(exist_ok=True)

TABULAR_OOF_PATH = OUT_DIR / "ag_tabular_oof.csv"


def load_volume_df():
    """Load just Date + Volume from raw data (no feature engineering needed)."""
    df = load_and_merge_data()
    df = df[df[TARGET_COL].notna()].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", TARGET_COL]].sort_values("Date").reset_index(drop=True)
    print(f"  rows: {len(df):,}   date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_tabular_oof():
    if not TABULAR_OOF_PATH.exists():
        raise FileNotFoundError(
            f"{TABULAR_OOF_PATH} missing. Run ag_tabular.py first so we know "
            "the OOF date range to align against."
        )
    oof = pd.read_csv(TABULAR_OOF_PATH, parse_dates=["Date"])
    return oof[["Date", TARGET_COL]].rename(columns={TARGET_COL: "Volume"})


def make_weekly_folds(oof_dates):
    """Step through oof_dates in 7-day chunks. Returns list of (week_idx,
    chunk_start_date, chunk_end_date)."""
    dates = sorted(oof_dates)
    folds = []
    i = 0
    week_idx = 1
    while i < len(dates):
        start = dates[i]
        end = dates[min(i + 6, len(dates) - 1)]
        folds.append((week_idx, start, end))
        i += 7
        week_idx += 1
    return folds


def fold_cache_path(week_idx):
    return FOLD_CACHE_DIR / f"prophet_week{week_idx:02d}.csv"


def fit_predict_one_week(full_df, week_idx, start_date, end_date, use_cache=True):
    cache = fold_cache_path(week_idx)
    if use_cache and cache.exists():
        return pd.read_csv(cache, parse_dates=["Date"])

    train = full_df[full_df["Date"] < start_date].rename(
        columns={"Date": "ds", TARGET_COL: "y"}
    )
    future_dates = pd.date_range(start_date, end_date, freq="D")
    future = pd.DataFrame({"ds": future_dates})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    m.fit(train)
    forecast = m.predict(future)
    out = pd.DataFrame({
        "Date": forecast["ds"].values,
        "pred": forecast["yhat"].values,
    })
    out.to_csv(cache, index=False)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--report-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.clear_cache:
        for f in FOLD_CACHE_DIR.glob("prophet_week*.csv"):
            f.unlink()
        print("Cleared Prophet weekly cache")

    oof_csv = OUT_DIR / "prophet_oof.csv"
    summary_path = OUT_DIR / "prophet_oof_summary.json"

    if args.report_only:
        if not oof_csv.exists():
            raise FileNotFoundError(f"{oof_csv} missing — cannot --report-only")
        oof_df = pd.read_csv(oof_csv, parse_dates=["Date"])
        print(f"Loaded cached Prophet OOF: {len(oof_df)} rows  "
              f"MAE = {mean_absolute_error(oof_df['Volume'], oof_df['pred_prophet']):,.0f}")
        return

    full_df = load_volume_df()
    tabular_oof = load_tabular_oof()
    oof_dates = pd.to_datetime(tabular_oof["Date"]).tolist()
    print(f"\nOOF dates: {len(oof_dates)}  ({oof_dates[0].date()} → {oof_dates[-1].date()})")

    weekly_folds = make_weekly_folds(oof_dates)
    print(f"Weekly folds: {len(weekly_folds)}")

    fold_metrics = []
    pieces = []
    for week_idx, start_date, end_date in weekly_folds:
        print(f"  week {week_idx:>2}: {start_date.date()} → {end_date.date()}", end=" ")
        preds = fit_predict_one_week(
            full_df, week_idx, start_date, end_date, use_cache=not args.no_cache,
        )
        pieces.append(preds)
        actual = full_df[(full_df["Date"] >= start_date) & (full_df["Date"] <= end_date)]
        merged = actual.merge(preds, on="Date", how="left")
        if merged[TARGET_COL].notna().any():
            mae = float(mean_absolute_error(merged[TARGET_COL], merged["pred"]))
            fold_metrics.append({
                "week": week_idx,
                "start": str(start_date.date()),
                "end": str(end_date.date()),
                "n": int(len(merged)),
                "mae": mae,
            })
            print(f" MAE {mae:,.0f}")
        else:
            print(" (no actuals)")

    stacked = pd.concat(pieces, ignore_index=True).rename(columns={"pred": "pred_prophet"})
    out = tabular_oof.merge(stacked, on="Date", how="left")
    out.to_csv(oof_csv, index=False)
    overall_mae = float(mean_absolute_error(out["Volume"], out["pred_prophet"]))
    print(f"\nSaved Prophet OOF → {oof_csv}  (n={len(out)}, overall MAE = {overall_mae:,.0f})")

    summary = {
        "n_weekly_folds": len(weekly_folds),
        "regressors": [],
        "holidays": False,
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "fold_metrics": fold_metrics,
        "overall_oof_mae": overall_mae,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
