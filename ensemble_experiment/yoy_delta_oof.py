"""
yoy_delta_oof.py
=================
YoY-delta baseline: pred(D) = Volume(D - 364) + delta_weight * weighted_delta,
where weighted_delta is a weighted average of how far recent same-weekday
volume is running above/below the same weekday one year ago:

    delta_k = Volume(D - 7k) - Volume(D - 7k - 364)   for k = 1, 2 weeks back

Sweeps lookback split (1wk only / 2wk 50-50 / 2wk 80-20) x delta_weight
(1.0 / 0.75 / 0.5) and reports standalone OOF MAE for every variant, then
saves the best variant's predictions in the same format as the other
per-model OOF files (Date, Volume, pred_yoy_delta) for combine_oof.py.

Outputs (under ensemble_experiment/output/):
    yoy_delta_oof.csv          Date, Volume, pred_yoy_delta  (best variant)
    yoy_delta_summary.json     variant sweep MAEs + chosen variant
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from production_router import (
    get_major_holiday_dates,
    days_to_prior_and_next_major,
    STORM_TRIGGER_IMPACT,
    PEAK_HOLIDAY_WINDOW,
    SHOULDER_PRE_WINDOW,
    SHOULDER_POST_WINDOW,
)

OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)
TABULAR_OOF_PATH = OUT_DIR / "ag_tabular_oof.csv"
VOLUME_PATH = ROOT / "data" / "tsa_volume.csv"
WEATHER_PATH = ROOT / "data" / "weather_national_features_with_lags.csv"

YEAR_LAG = 364


def classify_regime(dates):
    """Same regime classification used in normal_weight_search.py, so
    'normal' here means exactly what the production router means."""
    weather = (pd.read_csv(WEATHER_PATH, parse_dates=["date"])
               .rename(columns={"date": "Date"})[["Date", "vol_wtd_storm_impact"]])
    d = pd.DataFrame({"Date": dates}).merge(weather, on="Date", how="left")
    d["vol_wtd_storm_impact"] = d["vol_wtd_storm_impact"].fillna(0.0)
    d["storm_impact_sq"] = d["vol_wtd_storm_impact"] ** 2
    d["storm_severe_flag"] = (d["vol_wtd_storm_impact"] > 0.5).astype(int)

    years = sorted(d["Date"].dt.year.unique())
    holidays = get_major_holiday_dates(range(min(years) - 1, max(years) + 2))
    prior_next = d["Date"].apply(
        lambda dt: pd.Series(days_to_prior_and_next_major(dt, holidays),
                              index=["days_prior", "days_next"])
    )
    d[["days_prior", "days_next"]] = prior_next

    def _classify(row):
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

    return d["Date"].to_frame().assign(regime=d.apply(_classify, axis=1))

# (label, w1, w2) — weight on 1-week-back delta vs 2-weeks-back delta
SPLITS = [
    ("1wk_only", 1.0, 0.0),
    ("2wk_50_50", 0.5, 0.5),
    ("2wk_80_20", 0.8, 0.2),
]
DELTA_WEIGHTS = [1.0, 0.75, 0.5]


def main():
    if not TABULAR_OOF_PATH.exists():
        raise FileNotFoundError(
            f"{TABULAR_OOF_PATH} missing. Run ag_tabular.py first so we know "
            "the OOF date range to align against."
        )
    dates = pd.read_csv(TABULAR_OOF_PATH, parse_dates=["Date"])["Date"]
    vol = (pd.read_csv(VOLUME_PATH, parse_dates=["Date"])
           .set_index("Date")["Volume"])

    df = pd.DataFrame({"Date": dates})
    df["Volume"] = df["Date"].map(vol)
    df["ly"] = (df["Date"] - pd.Timedelta(days=YEAR_LAG)).map(vol)

    # delta_k = Volume(D - 7k) - Volume(D - 7k - YEAR_LAG)
    for k in (1, 2):
        cur = (df["Date"] - pd.Timedelta(days=7 * k)).map(vol)
        ly_k = (df["Date"] - pd.Timedelta(days=7 * k + YEAR_LAG)).map(vol)
        df[f"delta_{k}"] = cur - ly_k

    missing = df[["Volume", "ly", "delta_1", "delta_2"]].isna().any(axis=1)
    if missing.any():
        print(f"  Dropping {missing.sum()} rows with missing history "
              f"(need data back to D-{7*2+YEAR_LAG} days).")
    df = df.loc[~missing].reset_index(drop=True)

    regime = classify_regime(df["Date"])
    df = df.merge(regime, on="Date", how="left")
    print(f"Tuning variant sweep across all {len(df)} rows "
          f"({df['Date'].min().date()} -> {df['Date'].max().date()}); "
          "yoy_delta now replaces prophet in every regime.")

    y = df["Volume"].values

    print("=" * 70)
    print("YoY-delta variant sweep (all days)")
    print("=" * 70)
    print(f"{'split':<12} {'delta_weight':>12}   {'MAE':>12}")

    results = []
    for split_label, w1, w2 in SPLITS:
        weighted_delta = w1 * df["delta_1"] + w2 * df["delta_2"]
        for dw in DELTA_WEIGHTS:
            pred = df["ly"] + dw * weighted_delta
            m = float(mean_absolute_error(y, pred.values))
            results.append({"split": split_label, "w1": w1, "w2": w2,
                             "delta_weight": dw, "mae": m})
            print(f"{split_label:<12} {dw:>12.2f}   {m:>12,.0f}")

    # Raw seasonal-naive (delta_weight=0) for reference
    raw_mae = float(mean_absolute_error(y, df["ly"].values))
    print(f"{'(raw, no delta)':<12} {0.0:>12.2f}   {raw_mae:>12,.0f}")

    best = min(results, key=lambda r: r["mae"])
    print("-" * 70)
    print(f"Best variant (all days): split={best['split']} (w1={best['w1']}, w2={best['w2']}) "
          f"delta_weight={best['delta_weight']}  MAE={best['mae']:,.0f}")

    print("\nPer-regime MAE of the best variant:")
    weighted_delta_all = best["w1"] * df["delta_1"] + best["w2"] * df["delta_2"]
    df["_pred"] = df["ly"] + best["delta_weight"] * weighted_delta_all
    for r, g in df.groupby("regime"):
        print(f"  {r:<15} n={len(g):<5} MAE={mean_absolute_error(g['Volume'], g['_pred']):>10,.0f}")

    df["pred_yoy_delta"] = df.pop("_pred")

    out = df[["Date", "Volume", "pred_yoy_delta"]]
    out_csv = OUT_DIR / "yoy_delta_oof.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved OOF (all dates) -> {out_csv}  (n={len(out)})")

    summary = {
        "regime_filter": "all",
        "n_rows": int(len(out)),
        "date_start": str(out["Date"].min().date()),
        "date_end": str(out["Date"].max().date()),
        "year_lag_days": YEAR_LAG,
        "raw_seasonal_naive_mae": raw_mae,
        "sweep": results,
        "best_variant": best,
    }
    with open(OUT_DIR / "yoy_delta_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary -> {OUT_DIR / 'yoy_delta_summary.json'}")


if __name__ == "__main__":
    main()
