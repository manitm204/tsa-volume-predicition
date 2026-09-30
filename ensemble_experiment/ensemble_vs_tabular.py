"""
ensemble_vs_tabular.py
======================
Compare the new per-regime ensemble weights against pure tabular (ag_tabular)
on the OOF data.

Metrics per regime: N, MAE, Std(abs error), Bias (pred - actual).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from production_router import (
    get_major_holiday_dates, days_to_prior_and_next_major,
    STORM_TRIGGER_IMPACT, PEAK_HOLIDAY_WINDOW,
    SHOULDER_PRE_WINDOW, SHOULDER_POST_WINDOW,
)

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF_PATH     = os.path.join(ROOT, "ensemble_experiment", "output", "combined_oof.csv")
WEATHER_PATH = os.path.join(ROOT, "data", "weather_national_features_with_lags.csv")

MODEL_COLS  = ["pred_ag_tabular", "pred_ag_timeseries", "pred_prophet", "pred_anchor_master"]

# New per-regime ensemble weights  (tab, ts, prophet, anchor)
NEW_WEIGHTS = {
    "normal":       np.array([0.450, 0.300, 0.000, 0.250]),
    "shoulder_pre": np.array([0.000, 0.079, 0.001, 0.920]),
    "shoulder_post":np.array([0.950, 0.050, 0.000, 0.000]),
    "peak_holiday": np.array([1.000, 0.000, 0.000, 0.000]),
    "moderate_storm": np.array([1.000, 0.000, 0.000, 0.000]),  # not optimised → tabular
    "severe_storm":   np.array([1.000, 0.000, 0.000, 0.000]),  # not optimised → tabular
}

REGIME_ORDER = ["normal", "shoulder_pre", "shoulder_post", "peak_holiday",
                "moderate_storm", "severe_storm"]


def load_with_regimes():
    oof = pd.read_csv(OOF_PATH, parse_dates=["Date"])
    weather = (pd.read_csv(WEATHER_PATH, parse_dates=["date"])
               .rename(columns={"date": "Date"})[["Date", "vol_wtd_storm_impact"]])
    oof = oof.merge(weather, on="Date", how="left")
    oof["vol_wtd_storm_impact"] = oof["vol_wtd_storm_impact"].fillna(0.0)
    oof["storm_impact_sq"]   = oof["vol_wtd_storm_impact"] ** 2
    oof["storm_severe_flag"] = (oof["vol_wtd_storm_impact"] > 0.5).astype(int)

    years    = sorted(oof["Date"].dt.year.unique())
    holidays = get_major_holiday_dates(range(min(years) - 1, max(years) + 2))
    prior_next = oof["Date"].apply(
        lambda d: pd.Series(days_to_prior_and_next_major(d, holidays),
                            index=["days_prior", "days_next"])
    )
    oof[["days_prior", "days_next"]] = prior_next

    def classify(row):
        if row["storm_severe_flag"] == 1:
            return "severe_storm" if row["storm_impact_sq"] >= STORM_TRIGGER_IMPACT else "moderate_storm"
        p, n = int(row["days_prior"]), int(row["days_next"])
        if min(p, n) <= PEAK_HOLIDAY_WINDOW:                       return "peak_holiday"
        if PEAK_HOLIDAY_WINDOW < p <= SHOULDER_POST_WINDOW:        return "shoulder_post"
        if PEAK_HOLIDAY_WINDOW < n <= SHOULDER_PRE_WINDOW:         return "shoulder_pre"
        return "normal"

    oof["regime"] = oof.apply(classify, axis=1)

    # Apply per-regime ensemble weights
    X = oof[MODEL_COLS].values.astype(float)
    oof["pred_ensemble"] = 0.0
    for regime, w in NEW_WEIGHTS.items():
        mask = oof["regime"] == regime
        oof.loc[mask, "pred_ensemble"] = X[mask.values] @ w

    return oof


def stats(actual, pred):
    err = actual - pred
    ae  = err.abs()
    return {"mae": ae.mean(), "std": ae.std(), "bias": (-err).mean()}


def print_table(df):
    W = 90
    print("\n" + "=" * W)
    print(f"  {'Regime':<15}  {'N':>4}  "
          f"{'── Ensemble ──':^34}  {'── Tabular only ──':^34}")
    print(f"  {'':15}  {'':4}  "
          f"{'MAE':>10}  {'Std':>10}  {'Bias':>10}  "
          f"{'MAE':>10}  {'Std':>10}  {'Bias':>10}  {'Δ MAE':>8}")
    print("-" * W)

    actual = df["Volume"]

    def row(label, subset):
        if len(subset) == 0:
            print(f"  {label:<15}  {'0':>4}  {'—':>10}  {'—':>10}  {'—':>10}  "
                  f"{'—':>10}  {'—':>10}  {'—':>10}  {'—':>8}")
            return None, None

        se  = stats(subset["Volume"], subset["pred_ensemble"])
        st  = stats(subset["Volume"], subset["pred_ag_tabular"])
        d   = se["mae"] - st["mae"]
        arrow = "▼" if d < 0 else ("▲" if d > 0 else "=")
        print(f"  {label:<15}  {len(subset):>4}  "
              f"{se['mae']:>10,.0f}  {se['std']:>10,.0f}  {se['bias']:>+10,.0f}  "
              f"{st['mae']:>10,.0f}  {st['std']:>10,.0f}  {st['bias']:>+10,.0f}  "
              f"{d:>+7,.0f}{arrow}")
        return se, st

    ensemble_totals, tabular_totals = [], []

    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime]
        se, st = row(regime, sub)

    print("-" * W)

    # Overall
    se_all = stats(df["Volume"], df["pred_ensemble"])
    st_all = stats(df["Volume"], df["pred_ag_tabular"])
    d_all  = se_all["mae"] - st_all["mae"]
    arrow  = "▼" if d_all < 0 else ("▲" if d_all > 0 else "=")
    print(f"  {'OVERALL':<15}  {len(df):>4}  "
          f"{se_all['mae']:>10,.0f}  {se_all['std']:>10,.0f}  {se_all['bias']:>+10,.0f}  "
          f"{st_all['mae']:>10,.0f}  {st_all['std']:>10,.0f}  {st_all['bias']:>+10,.0f}  "
          f"{d_all:>+7,.0f}{arrow}")
    print("=" * W)
    print(f"\n  ▼ = ensemble better   ▲ = tabular better   Δ MAE = ensemble − tabular\n")


if __name__ == "__main__":
    df = load_with_regimes()
    print_table(df)
