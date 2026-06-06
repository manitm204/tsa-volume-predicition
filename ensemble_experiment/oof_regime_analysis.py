"""
oof_regime_analysis.py
======================
Break down each ensemble model's OOF errors by the production router's 6 regimes:
  severe_storm / moderate_storm / peak_holiday / shoulder_pre / shoulder_post / normal

Outputs a table of  N | MAE | std(abs_error) | bias  per model × regime.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from production_router import (
    get_major_holiday_dates,
    days_to_nearest_major_signed,
    STORM_TRIGGER_IMPACT,
    PEAK_HOLIDAY_WINDOW,
    SHOULDER_WINDOW,
)

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF_PATH     = os.path.join(ROOT, "ensemble_experiment", "output", "combined_oof.csv")
WEATHER_PATH = os.path.join(ROOT, "data", "weather_national_features_with_lags.csv")

# ── load OOF ───────────────────────────────────────────────────────────────
oof = pd.read_csv(OOF_PATH, parse_dates=["Date"])

MODELS = {
    "ag_tabular":    "pred_ag_tabular",
    "ag_timeseries": "pred_ag_timeseries",
    "prophet":       "pred_prophet",
    "anchor_master": "pred_anchor_master",
}

# ── attach storm features from weather ─────────────────────────────────────
weather = pd.read_csv(WEATHER_PATH, parse_dates=["date"]).rename(columns={"date": "Date"})
weather = weather[["Date", "vol_wtd_storm_impact"]].copy()
oof = oof.merge(weather, on="Date", how="left")
oof["vol_wtd_storm_impact"] = oof["vol_wtd_storm_impact"].fillna(0.0)
oof["storm_impact_sq"]   = oof["vol_wtd_storm_impact"] ** 2
oof["storm_severe_flag"] = (oof["vol_wtd_storm_impact"] > 0.5).astype(int)

# ── compute days_to_major_signed ───────────────────────────────────────────
years = sorted(oof["Date"].dt.year.unique())
holidays = get_major_holiday_dates([y for y in range(min(years) - 1, max(years) + 2)])

oof["days_to_major_signed"] = oof["Date"].apply(
    lambda d: days_to_nearest_major_signed(d, holidays)
)

# ── classify regimes ───────────────────────────────────────────────────────
def classify_regime(row):
    if row["storm_severe_flag"] == 1:
        return "severe_storm" if row["storm_impact_sq"] >= STORM_TRIGGER_IMPACT else "moderate_storm"
    d = row["days_to_major_signed"]
    if abs(d) <= PEAK_HOLIDAY_WINDOW:
        return "peak_holiday"
    if PEAK_HOLIDAY_WINDOW < abs(d) <= SHOULDER_WINDOW:
        return "shoulder_pre" if d > 0 else "shoulder_post"
    return "normal"

oof["regime"] = oof.apply(classify_regime, axis=1)

# ── compute per-regime per-model stats ─────────────────────────────────────
REGIME_ORDER = [
    "normal", "shoulder_pre", "shoulder_post", "peak_holiday",
    "moderate_storm", "severe_storm",
]

def regime_stats(df, pred_col):
    rows = []
    actual = df["Volume"]
    pred   = df[pred_col]
    abs_err = (actual - pred).abs()
    bias    = (pred - actual).mean()
    rows.append({
        "regime": "OVERALL",
        "n": len(df),
        "mae": abs_err.mean(),
        "std": abs_err.std(),
        "bias": bias,
    })
    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime]
        if len(sub) == 0:
            rows.append({"regime": regime, "n": 0, "mae": np.nan, "std": np.nan, "bias": np.nan})
            continue
        ae = (sub["Volume"] - sub[pred_col]).abs()
        rows.append({
            "regime": regime,
            "n": len(sub),
            "mae": ae.mean(),
            "std": ae.std(),
            "bias": (sub[pred_col] - sub["Volume"]).mean(),
        })
    return pd.DataFrame(rows).set_index("regime")

stats = {name: regime_stats(oof, col) for name, col in MODELS.items()}

# ── print ──────────────────────────────────────────────────────────────────
def fmt(v, digits=0):
    if np.isnan(v):
        return "   —  "
    if digits == 0:
        return f"{v:>8,.0f}"
    return f"{v:>8.{digits}f}"

REGIME_LABELS = {
    "OVERALL":       "OVERALL      ",
    "normal":        "normal       ",
    "shoulder_pre":  "shoulder_pre ",
    "shoulder_post": "shoulder_post",
    "peak_holiday":  "peak_holiday ",
    "moderate_storm":"moderate_storm",
    "severe_storm":  "severe_storm ",
}

model_names = list(MODELS.keys())
header_pad = 15

# Print regime day-count distribution first
print("\n" + "="*60)
print("REGIME DAY COUNTS (n=364 days, 2025-06-05 → 2026-06-03)")
print("="*60)
for regime in ["OVERALL"] + REGIME_ORDER:
    if regime == "OVERALL":
        n = len(oof)
    else:
        n = (oof["regime"] == regime).sum()
    label = REGIME_LABELS.get(regime, regime)
    bar = "█" * (n // 5)
    print(f"  {label:<15} {n:>4d}  {bar}")

# Print per-model breakdown
col_w = 26  # width per model column

print("\n" + "="*(header_pad + col_w * len(model_names)))
print(f"{'':>{header_pad}}", end="")
for m in model_names:
    print(f"{'  '+m:<{col_w}}", end="")
print()
print(f"{'Regime':<{header_pad}}", end="")
for _ in model_names:
    print(f"{'  N':>5}{'MAE':>9}{'Std':>9}{'Bias':>8}", end="  ")
print()
print("-"*(header_pad + col_w * len(model_names)))

for regime in ["OVERALL"] + REGIME_ORDER:
    label = REGIME_LABELS.get(regime, regime)
    print(f"{label:<{header_pad}}", end="")
    for m in model_names:
        s = stats[m].loc[regime]
        n = int(s["n"]) if not np.isnan(s["n"]) else 0
        print(f"  {n:>3d}{fmt(s['mae']):>9}{fmt(s['std']):>9}{fmt(s['bias']):>8}", end="  ")
    print()

print()

# ── best model per regime ──────────────────────────────────────────────────
print("="*60)
print("BEST MODEL BY MAE PER REGIME")
print("="*60)
for regime in ["OVERALL"] + REGIME_ORDER:
    label = REGIME_LABELS.get(regime, regime)
    maes = {m: stats[m].loc[regime]["mae"] for m in model_names}
    valid = {m: v for m, v in maes.items() if not np.isnan(v)}
    if not valid:
        print(f"  {label:<15}  — (no data)")
        continue
    best = min(valid, key=valid.get)
    worst = max(valid, key=valid.get)
    spread = valid[worst] - valid[best]
    print(f"  {label:<15}  best={best:<16} MAE={valid[best]:>8,.0f}  "
          f"(gap to worst: {spread:>7,.0f})")

print()
