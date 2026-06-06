"""
daily_predict.py
================
Daily TSA passenger volume prediction for Kalshi daily markets.

Predicts today's (or a specified date's) volume and computes P(over/under)
for daily screening thresholds.

Pipeline:
  1. Tabular AutoGluon (autoregressive Monday chain for correct lag features)
  2. TS3 + Prophet shadow models for the target date
  3. Per-regime ensemble weights  (ensemble_experiment/normal_weight_search.py)
  4. Per-regime Platt σ_effective  (ensemble_experiment/platt_regime.py)

Prerequisites:
    - Run autogluon_full.py first (trains model + generates OOF)
    - Run build_features.py first (generates master_features.csv)
    - Run train_shadow_models.py first (saves Prophet + TS3 to output_router_shadow/models/)

Usage:
    python daily_predict.py
    python daily_predict.py --date 2026-05-19
    python daily_predict.py --thresholds 2.19 2.29 2.39

Outputs to output_autogluon_predict/
    daily_forecast.csv    — predicted volume + per-threshold probabilities
    daily_summary.csv     — one-row summary with p_over_XM / p_under_XM columns
"""

import json
import logging
import pickle
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from autogluon.tabular import TabularPredictor

from build_features import (
    load_master, build_features_from_df, load_and_merge_data,
    get_feature_columns, TARGET_COL,
)
from autogluon_predict import load_raw_data, build_features_for_date
from production_router import (
    classify_regime, storm_alpha_for,
    get_major_holiday_dates, days_to_nearest_major_signed,
)

MODEL_DIR         = Path("./output_autogluon_best/ag_final")
SHADOW_MODELS_DIR = Path("./output_router_shadow/models")
OUT_DIR           = Path("./output_autogluon_predict")
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_THRESHOLDS = [2.19, 2.29, 2.39]   # millions

# ── Per-regime ensemble weights (tab, ts3, prophet, anchor) ──────────────
# Source: ensemble_experiment/normal_weight_search.py — Direct-MAE LOO on OOF
# STORM: None → uses production_router's smooth α-blend with weather_anchor
ENSEMBLE_WEIGHTS = {
    "NORMAL":       (0.491, 0.250, 0.022, 0.238),
    "SHOULDER_PRE": (0.005, 0.103, 0.001, 0.891),
    "SHOULDER_POST":(0.881, 0.072, 0.047, 0.000),
    "PEAK_HOLIDAY": (1.000, 0.000, 0.000, 0.000),
    "STORM":        None,   # α-blend handled separately
}

# ── Per-regime σ for P(over/under) ───────────────────────────────────────
# All regimes: Platt σ_effective = σ_raw / A  (ensemble_experiment/platt_regime.py)
# STORM uses moderate_storm σ_eff (severe_storm n=2, not fitted separately).
REGIME_SIGMA = {
    "NORMAL":        41_617,
    "SHOULDER_PRE":  32_246,
    "SHOULDER_POST": 44_135,
    "PEAK_HOLIDAY":  82_271,
    "STORM":         57_780,
}

ITEM_ID = "tsa"


# ==========================================================================
# Load models
# ==========================================================================
def load_model() -> TabularPredictor:
    if not (MODEL_DIR / "predictor.pkl").exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_DIR}. Run autogluon_full.py first."
        )
    print(f"  Tabular model:  {MODEL_DIR}")
    return TabularPredictor.load(str(MODEL_DIR))


def load_shadow_models():
    """Load Prophet + TS3 shadow models. Returns (prophet_bundle, ts3, meta) or (None, None, None)."""
    prophet_path      = SHADOW_MODELS_DIR / "prophet_p3.pkl"
    ts3_path          = SHADOW_MODELS_DIR / "ts3_predictor"
    feature_cols_path = SHADOW_MODELS_DIR / "feature_cols.json"

    if not all(p.exists() for p in [prophet_path, ts3_path, feature_cols_path]):
        print(f"  WARNING: Shadow models not found in {SHADOW_MODELS_DIR}")
        print(f"  Run train_shadow_models.py first for ensemble routing.")
        print(f"  Falling back to tabular-only prediction.")
        return None, None, None

    from autogluon.timeseries import TimeSeriesPredictor
    with open(prophet_path, "rb") as f:
        prophet_bundle = pickle.load(f)
    ts3  = TimeSeriesPredictor.load(str(ts3_path))
    with open(feature_cols_path) as f:
        meta = json.load(f)

    print(f"  Shadow models:  {SHADOW_MODELS_DIR}")
    return prophet_bundle, ts3, meta


# ==========================================================================
# Build full feature dataframe for TS3 + Prophet
# ==========================================================================
def build_full_feature_df(target_date: pd.Timestamp) -> pd.DataFrame:
    """
    Build a full (prune=False) feature dataframe that includes target_date.
    Used by TS3 and Prophet — does NOT inject intra-week predicted volumes
    as history (TS3/Prophet use actual TSA history up to last known date).
    """
    df = load_and_merge_data()
    df = build_features_from_df(df, verbose=False, prune=False)
    df["Date"] = pd.to_datetime(df["Date"])

    if not (df["Date"] == target_date).any():
        extra = pd.DataFrame({"Date": [target_date], TARGET_COL: [np.nan]})
        df = pd.concat([df, extra], ignore_index=True).sort_values("Date").reset_index(drop=True)
        df = build_features_from_df(df, verbose=False, prune=False)
        df["Date"] = pd.to_datetime(df["Date"])

    return df


# ==========================================================================
# TS3 + Prophet predictions for a single date
# ==========================================================================
def get_ts3_prophet_preds(
    target_date: pd.Timestamp,
    feature_df: pd.DataFrame,
    prophet_bundle,
    ts3,
    meta: dict,
) -> tuple[float, float]:
    from autogluon.timeseries import TimeSeriesDataFrame

    row = feature_df[feature_df["Date"] == target_date]
    if row.empty:
        raise ValueError(f"No feature row for {target_date.date()} in feature_df")
    row = row.iloc[0]

    # ── Prophet ──────────────────────────────────────────────────────────
    prophet_model = prophet_bundle["model"]
    prophet_regs  = prophet_bundle["regressors"]
    future = pd.DataFrame({"ds": [target_date]})
    for r in prophet_regs:
        future[r] = [float(row[r]) if r in row.index and not pd.isna(row[r]) else 0.0]
    prophet_pred = float(prophet_model.predict(future).iloc[0]["yhat"])

    # ── TS3 ───────────────────────────────────────────────────────────────
    ts3_covs = [c for c in meta["ts3_covariates"] if c in feature_df.columns]

    history = feature_df[feature_df["Date"] < target_date][
        ["Date", TARGET_COL] + ts3_covs
    ].dropna(subset=[TARGET_COL]).copy()
    history["item_id"] = ITEM_ID
    history = history.rename(columns={"Date": "timestamp", TARGET_COL: "target"})
    history_ts = TimeSeriesDataFrame.from_data_frame(
        history, id_column="item_id", timestamp_column="timestamp"
    )

    future_row = feature_df[feature_df["Date"] == target_date][["Date"] + ts3_covs].copy()
    future_row["item_id"] = ITEM_ID
    future_row = future_row.rename(columns={"Date": "timestamp"})
    known_covariates = TimeSeriesDataFrame.from_data_frame(
        future_row, id_column="item_id", timestamp_column="timestamp"
    )

    forecast = ts3.predict(history_ts, known_covariates=known_covariates)
    col = "mean" if "mean" in forecast.columns else forecast.columns[0]
    ts3_pred = float(forecast.iloc[0][col])

    return ts3_pred, prophet_pred


# ==========================================================================
# Regime classification + ensemble routing
# ==========================================================================
def route_ensemble(
    target_date: pd.Timestamp,
    tabular_pred: float,
    ts3_pred: float | None,
    prophet_pred: float | None,
    feature_df: pd.DataFrame,
) -> tuple[float, str, float, int]:
    """
    Returns (pred_ensemble, regime, sigma, days_to_major).

    If ts3_pred / prophet_pred are None (shadow models unavailable), falls
    back to tabular-only for regimes that need them.
    """
    row = feature_df[feature_df["Date"] == target_date]
    if row.empty:
        return tabular_pred, "NORMAL", REGIME_SIGMA["NORMAL"], 0
    row = row.iloc[0]

    storm_flag    = int(row.get("storm_severe_flag", 0) or 0)
    impact_sq     = float(row.get("storm_impact_sq", 0.0) or 0.0)
    anchor_val    = float(row.get("anchor_master", tabular_pred) or tabular_pred)
    weather_anch  = float(row.get("weather_penalized_anchor", tabular_pred) or tabular_pred)

    holidays     = get_major_holiday_dates([target_date.year - 1, target_date.year, target_date.year + 1])
    days_to_major = days_to_nearest_major_signed(target_date, holidays=holidays)

    regime = classify_regime(
        storm_severe_flag=storm_flag,
        storm_impact_sq=impact_sq,
        days_to_major_signed=days_to_major,
    )

    if regime == "STORM":
        a = storm_alpha_for(impact_sq)
        pred_ensemble = a * tabular_pred + (1 - a) * weather_anch
    else:
        wt, ws, wp, wa = ENSEMBLE_WEIGHTS[regime]
        ts3_val    = ts3_pred    if ts3_pred    is not None else tabular_pred
        prophet_val = prophet_pred if prophet_pred is not None else tabular_pred
        pred_ensemble = (
            wt * tabular_pred +
            ws * ts3_val +
            wp * prophet_val +
            wa * anchor_val
        )

    sigma = REGIME_SIGMA[regime]
    return pred_ensemble, regime, sigma, days_to_major


# ==========================================================================
# Tabular prediction (autoregressive Monday chain)
# ==========================================================================
def predict_single_day(
    target_date: pd.Timestamp,
    predictor: TabularPredictor,
    master_df: pd.DataFrame,
    raw_tsa_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> tuple[float, str]:
    """
    Walk Monday → target_date, using actuals where available and tabular
    predictions otherwise, so lag features are correctly populated.

    Returns (tabular_predicted_volume, status).
    """
    feature_cols = get_feature_columns(master_df)
    dates_in_df  = set(raw_tsa_df["Date"].dt.date)

    if target_date.date() in dates_in_df:
        vol = float(raw_tsa_df.loc[
            raw_tsa_df["Date"].dt.date == target_date.date(), TARGET_COL
        ].values[0])
        return vol, "actual"

    days_since_monday = target_date.dayofweek
    week_monday       = target_date - pd.Timedelta(days=days_since_monday)
    predictions: dict[pd.Timestamp, float] = {}

    for offset in range(days_since_monday + 1):
        d  = week_monday + pd.Timedelta(days=offset)
        ts = pd.Timestamp(d)

        if ts.date() in dates_in_df:
            vol = float(raw_tsa_df.loc[
                raw_tsa_df["Date"].dt.date == ts.date(), TARGET_COL
            ].values[0])
            predictions[ts] = vol
            if offset < days_since_monday:
                print(f"    {ts.date()}  ACTUAL    {vol:>12,.0f}")
        else:
            print(f"    {ts.date()}  building features...", end="", flush=True)
            target_row   = build_features_for_date(ts, raw_tsa_df, weather_df, predictions)
            row_features = target_row[feature_cols].copy()
            pred         = predictor.predict(row_features)
            pred_vol     = float(pred.iloc[0]) if hasattr(pred, "iloc") else float(pred)
            predictions[ts] = pred_vol
            label = "TARGET" if ts == target_date else "INTERMED"
            print(f"\r    {ts.date()}  {label:<8}  {pred_vol:>12,.0f}")

    return predictions[target_date], "predicted"


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Daily TSA volume prediction")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
        help="Daily thresholds in millions (default: 2.19 2.29 2.39)"
    )
    args = parser.parse_args()

    target_date = pd.Timestamp(args.date) if args.date else pd.Timestamp(date.today())
    thresholds  = args.thresholds

    print("=" * 70)
    print(f"DAILY TSA PREDICTION  —  {target_date.date()}  ({target_date.strftime('%A')})")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[ Data ]")
    master_df              = load_master()
    raw_tsa_df, weather_df = load_raw_data()
    last_date = raw_tsa_df["Date"].max()
    print(f"  Last TSA date:   {last_date.date()}  ({last_date.strftime('%A')})")
    print(f"  Master features: {len(master_df):,} rows, {len(get_feature_columns(master_df))} features")

    # ── Load models ───────────────────────────────────────────────────
    print("\n[ Models ]")
    predictor = load_model()
    prophet_bundle, ts3, meta = load_shadow_models()
    shadow_available = prophet_bundle is not None

    # ── Tabular prediction (autoregressive chain) ─────────────────────
    print(f"\n[ Tabular forecast — {target_date.date()} ]")
    days_since_monday = target_date.dayofweek
    if days_since_monday > 0:
        print(f"  Building week chain (Mon → {target_date.strftime('%a')}):")
    tabular_pred, status = predict_single_day(
        target_date, predictor, master_df, raw_tsa_df, weather_df
    )

    if status == "actual":
        print(f"  {target_date.date()} already has actual data.")
        # No uncertainty needed — result is known
        print()
        print("=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"  Date:       {target_date.date()}  ({target_date.strftime('%A')})")
        print(f"  Status:     ACTUAL")
        print(f"  Volume:     {tabular_pred:>12,.0f}  ({tabular_pred/1e6:.4f}M)")

        prob_rows = []
        for t in thresholds:
            p_over  = 1.0 if tabular_pred > t * 1e6 else 0.0
            p_under = 1.0 - p_over
            print(f"  {t:.2f}M:  {'OVER' if p_over == 1 else 'UNDER'}")
            prob_rows.append({"threshold_millions": t, "p_over": p_over, "p_under": p_under})

        _save_outputs(target_date, tabular_pred, status, tabular_pred, "ACTUAL",
                      0, tabular_pred, tabular_pred, thresholds, prob_rows)
        return

    # ── TS3 + Prophet predictions ─────────────────────────────────────
    ts3_pred = prophet_pred = None
    if shadow_available:
        print(f"\n[ TS3 + Prophet — {target_date.date()} ]")
        try:
            feature_df = build_full_feature_df(target_date)
            ts3_pred, prophet_pred = get_ts3_prophet_preds(
                target_date, feature_df, prophet_bundle, ts3, meta
            )
            print(f"  pred_tabular:  {tabular_pred:>12,.0f}")
            print(f"  pred_ts3:      {ts3_pred:>12,.0f}")
            print(f"  pred_prophet:  {prophet_pred:>12,.0f}")
        except Exception as exc:
            print(f"  WARNING: TS3/Prophet failed ({exc}). Using tabular only.")
            feature_df = build_full_feature_df(target_date)
    else:
        feature_df = build_full_feature_df(target_date)

    # ── Regime routing + ensemble ─────────────────────────────────────
    print(f"\n[ Routing ]")
    pred_ensemble, regime, sigma, days_to_major = route_ensemble(
        target_date, tabular_pred, ts3_pred, prophet_pred, feature_df
    )

    print(f"  Regime:        {regime}")
    print(f"  Days to major: {days_to_major:+d}")
    if ts3_pred is not None:
        weights = ENSEMBLE_WEIGHTS.get(regime)
        if weights is not None:
            wt, ws, wp, wa = weights
            print(f"  Weights:       tab={wt:.3f}  ts3={ws:.3f}  prophet={wp:.3f}  anchor={wa:.3f}")
        else:
            print(f"  Weights:       storm α-blend")
    print(f"  pred_ensemble: {pred_ensemble:>12,.0f}")
    print(f"  σ ({regime:<12}): {sigma:>12,.0f}  (Platt σ_eff)")

    # ── Summary ───────────────────────────────────────────────────────
    ci_low  = pred_ensemble - 1.645 * sigma
    ci_high = pred_ensemble + 1.645 * sigma

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  Date:       {target_date.date()}  ({target_date.strftime('%A')})")
    print(f"  Regime:     {regime}")
    print(f"  Prediction: {pred_ensemble:>12,.0f}  ({pred_ensemble/1e6:.4f}M)")
    print(f"  Daily σ:    {sigma:>12,.0f}  ({sigma/1e6:.4f}M)")
    print(f"  90% CI:     [{ci_low/1e6:.4f}M — {ci_high/1e6:.4f}M]")

    # ── Probabilities ─────────────────────────────────────────────────
    print(f"\n  {'Threshold':>12}  {'P(Over)':>10}  {'P(Under)':>10}")
    print("  " + "-" * 36)

    prob_rows = []
    for t in thresholds:
        z       = (t * 1e6 - pred_ensemble) / sigma
        p_over  = float(1 - stats.norm.cdf(z))
        p_under = float(stats.norm.cdf(z))
        print(f"  {t:>10.2f}M  {p_over:>10.1%}  {p_under:>10.1%}")
        prob_rows.append({
            "threshold_millions": t,
            "p_over":             p_over,
            "p_under":            p_under,
        })

    _save_outputs(target_date, pred_ensemble, status, tabular_pred, regime,
                  sigma, ci_low, ci_high, thresholds, prob_rows)


# ==========================================================================
# Save outputs
# ==========================================================================
def _save_outputs(
    target_date, pred_vol, status, tabular_pred, regime,
    sigma, ci_low, ci_high, thresholds, prob_rows,
):
    forecast_df = pd.DataFrame([{
        "date":               target_date.date(),
        "day_name":           target_date.strftime("%A"),
        "status":             status,
        "predicted_volume":   pred_vol,
        "predicted_millions": pred_vol / 1e6,
        "tabular_pred":       tabular_pred,
        "regime":             regime,
        "daily_sigma":        sigma,
        "ci90_low":           ci_low,
        "ci90_high":          ci_high,
        **{f"p_over_{t}M":  r["p_over"]  for t, r in zip(thresholds, prob_rows)},
        **{f"p_under_{t}M": r["p_under"] for t, r in zip(thresholds, prob_rows)},
    }])

    forecast_path = OUT_DIR / "daily_forecast.csv"
    summary_path  = OUT_DIR / "daily_summary.csv"
    forecast_df.to_csv(forecast_path, index=False)
    forecast_df.to_csv(summary_path,  index=False)

    print(f"\n  Saved → {forecast_path}")
    print(f"  Saved → {summary_path}")

    try:
        import db
        db.write_daily_forecasts(forecast_df)
    except Exception as exc:
        print(f"[db] write failed (non-fatal): {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
