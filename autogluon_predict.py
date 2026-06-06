"""
autogluon_predict.py
====================
Weekly average prediction for Kalshi betting.

Uses the production model from autogluon_full.py.
Uses the EXACT SAME feature pipeline for inference as training.

Prediction pipeline:
  1. Tabular AutoGluon (autoregressive Monday chain for correct lag features)
  2. TS3 + Prophet shadow models for each predicted day
  3. Per-regime ensemble weights  (ensemble_experiment/normal_weight_search.py)
  4. Weekly σ from ensemble OOF + per-regime Platt ratio  (platt_regime.py)

Prerequisites:
    - Run autogluon_full.py first (trains model + generates OOF)
    - Run build_features.py first (generates master_features.csv)
    - Run train_shadow_models.py first (Prophet + TS3 for ensemble routing)

Usage:
    python autogluon_predict.py
    python autogluon_predict.py --thresholds 2.5 2.55 2.6 2.65

Outputs to output_autogluon_predict/
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import shutil

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from autogluon.tabular import TabularPredictor

from build_features import (
    load_master, build_features_from_df,
    get_feature_columns, TARGET_COL,
    WEATHER_PATH as _WEATHER_PATH,
)
from production_router import (
    classify_regime, storm_alpha_for,
    get_major_holiday_dates, days_to_nearest_major_signed,
)

OUT_DIR = Path("./output_autogluon_predict")
OUT_DIR.mkdir(exist_ok=True)


def _move_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.move(src, dst)

EVAL_MODEL_DIR = Path("./output_autogluon_best/ag_final")
OOF_PATH = Path("./output_autogluon_best/oof_predictions.csv")
FEATURE_IMPORTANCE_PATH = Path("./output_autogluon_best/feature_importance.csv")
SHADOW_MODELS_DIR = Path("./output_router_shadow/models")
ENSEMBLE_OOF_PATH = Path("./ensemble_experiment/output/combined_oof.csv")

DEFAULT_THRESHOLDS = [2.45, 2.5, 2.55, 2.6, 2.65, 2.7]  # in millions
TOP_N_DRIVERS = 5  # number of features to ablate for the dashboard explainability panel

# Per-regime ensemble weights (tab, ts3, prophet, anchor)
# Source: ensemble_experiment/normal_weight_search.py — Direct-MAE LOO
# STORM: None → production_router's smooth α-blend with weather_penalized_anchor
ENSEMBLE_WEIGHTS = {
    "NORMAL":       (0.491, 0.250, 0.022, 0.238),
    "SHOULDER_PRE": (0.005, 0.103, 0.001, 0.891),
    "SHOULDER_POST":(0.881, 0.072, 0.047, 0.000),
    "PEAK_HOLIDAY": (1.000, 0.000, 0.000, 0.000),
    "STORM":        None,
}

# Platt ratio = σ_eff / σ_raw per regime (ensemble_experiment/platt_regime.py)
# All regimes now have Platt shrinkage. STORM uses moderate_storm σ_eff.
PLATT_RATIO = {
    "NORMAL":        41_617 / 76_004,   # ≈ 0.548
    "SHOULDER_PRE":  32_246 / 56_245,   # ≈ 0.573
    "SHOULDER_POST": 44_135 / 72_708,   # ≈ 0.607
    "PEAK_HOLIDAY":  82_271 / 124_151,  # ≈ 0.663
    "STORM":         57_780 / 85_228,   # ≈ 0.678
}

ITEM_ID = "tsa"


# ==========================================================================
# Load model from autogluon_evaluate
# ==========================================================================
def load_eval_model():
    if not EVAL_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model not found at {EVAL_MODEL_DIR}\n"
            f"Run autogluon_evaluate.py first to train the model."
        )

    predictor_pkl = EVAL_MODEL_DIR / "predictor.pkl"
    if not predictor_pkl.exists():
        raise FileNotFoundError(
            f"Model incomplete at {EVAL_MODEL_DIR} (no predictor.pkl)\n"
            f"Re-run autogluon_evaluate.py to train the model."
        )

    print(f"  Loading model from {EVAL_MODEL_DIR}")
    predictor = TabularPredictor.load(str(EVAL_MODEL_DIR))
    return predictor


# ==========================================================================
# Shadow model helpers (Prophet + TS3)
# ==========================================================================
def _load_shadow_models():
    """Load Prophet + TS3. Returns (prophet_bundle, ts3, meta) or (None, None, None)."""
    prophet_path      = SHADOW_MODELS_DIR / "prophet_p3.pkl"
    ts3_path          = SHADOW_MODELS_DIR / "ts3_predictor"
    feature_cols_path = SHADOW_MODELS_DIR / "feature_cols.json"

    if not all(p.exists() for p in [prophet_path, ts3_path, feature_cols_path]):
        return None, None, None

    from autogluon.timeseries import TimeSeriesPredictor
    with open(prophet_path, "rb") as f:
        prophet_bundle = pickle.load(f)
    ts3  = TimeSeriesPredictor.load(str(ts3_path))
    with open(feature_cols_path) as f:
        meta = json.load(f)
    return prophet_bundle, ts3, meta


def _get_ts3_prophet_preds(target_date, feature_df, prophet_bundle, ts3, meta):
    """Get TS3 and Prophet predictions for a single date."""
    from autogluon.timeseries import TimeSeriesDataFrame

    row = feature_df[feature_df["Date"] == target_date]
    if row.empty:
        return None, None
    row = row.iloc[0]

    # Prophet
    prophet_model = prophet_bundle["model"]
    prophet_regs  = prophet_bundle["regressors"]
    future = pd.DataFrame({"ds": [target_date]})
    for r in prophet_regs:
        future[r] = [float(row[r]) if r in row.index and not pd.isna(row[r]) else 0.0]
    prophet_pred = float(prophet_model.predict(future).iloc[0]["yhat"])

    # TS3
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
    known_cov = TimeSeriesDataFrame.from_data_frame(
        future_row, id_column="item_id", timestamp_column="timestamp"
    )
    forecast = ts3.predict(history_ts, known_covariates=known_cov)
    col = "mean" if "mean" in forecast.columns else forecast.columns[0]
    ts3_pred = float(forecast.iloc[0][col])

    return ts3_pred, prophet_pred


def _route_day(target_date, tabular_pred, ts3_pred, prophet_pred, feature_df):
    """Apply regime classification + ensemble weights for one day.
    Returns (pred_ensemble, regime).
    """
    row = feature_df[feature_df["Date"] == target_date]
    if row.empty:
        return tabular_pred, "NORMAL"
    row = row.iloc[0]

    storm_flag   = int(row.get("storm_severe_flag", 0) or 0)
    impact_sq    = float(row.get("storm_impact_sq", 0.0) or 0.0)
    anchor_val   = float(row.get("anchor_master", tabular_pred) or tabular_pred)
    weather_anch = float(row.get("weather_penalized_anchor", tabular_pred) or tabular_pred)

    holidays      = get_major_holiday_dates([target_date.year - 1, target_date.year, target_date.year + 1])
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
        ts3_v    = ts3_pred    if ts3_pred    is not None else tabular_pred
        prophet_v = prophet_pred if prophet_pred is not None else tabular_pred
        pred_ensemble = wt * tabular_pred + ws * ts3_v + wp * prophet_v + wa * anchor_val

    return pred_ensemble, regime


# ==========================================================================
# Compute weekly uncertainty from ensemble OOF
# ==========================================================================
def _estimate_from_ensemble_oof():
    """Weekly std and autocorrelation from ensemble OOF (combined_oof.csv)."""
    oof = pd.read_csv(ENSEMBLE_OOF_PATH, parse_dates=["Date"])
    weather = (
        pd.read_csv(_WEATHER_PATH, parse_dates=["date"])
        .rename(columns={"date": "Date"})[["Date", "vol_wtd_storm_impact"]]
    )
    oof = oof.merge(weather, on="Date", how="left")
    oof["vol_wtd_storm_impact"] = oof["vol_wtd_storm_impact"].fillna(0.0)
    oof["storm_impact_sq"]   = oof["vol_wtd_storm_impact"] ** 2
    oof["storm_severe_flag"] = (oof["vol_wtd_storm_impact"] > 0.5).astype(int)

    years    = sorted(oof["Date"].dt.year.unique())
    holidays = get_major_holiday_dates(range(min(years) - 1, max(years) + 2))
    oof["days_to_major_signed"] = oof["Date"].apply(
        lambda d: days_to_nearest_major_signed(d, holidays)
    )
    oof["regime"] = oof.apply(
        lambda r: classify_regime(
            storm_severe_flag=r["storm_severe_flag"],
            storm_impact_sq=r["storm_impact_sq"],
            days_to_major_signed=r["days_to_major_signed"],
        ), axis=1
    )

    # Apply ensemble weights per row (STORM → tabular fallback, no weather_anchor in OOF)
    MODEL_COLS = ["pred_ag_tabular", "pred_ag_timeseries", "pred_prophet", "pred_anchor_master"]
    X = oof[MODEL_COLS].values.astype(float)
    oof["pred_ensemble"] = 0.0
    for regime, w in ENSEMBLE_WEIGHTS.items():
        mask = oof["regime"] == regime
        if not mask.any():
            continue
        if w is not None:
            oof.loc[mask, "pred_ensemble"] = X[mask.values] @ np.array(w)
        else:
            oof.loc[mask, "pred_ensemble"] = oof.loc[mask, "pred_ag_tabular"]

    oof["residual"] = oof["Volume"] - oof["pred_ensemble"]
    oof_sorted      = oof.sort_values("Date").reset_index(drop=True)
    residuals       = oof_sorted["residual"].values

    daily_std = float(oof["residual"].std())

    weekly_residuals = []
    for i in range(len(residuals) - 6):
        dates      = oof_sorted["Date"].iloc[i:i + 7].values
        date_range = (dates[-1] - dates[0]).astype("timedelta64[D]").astype(int)
        if date_range == 6:
            weekly_residuals.append(np.sum(residuals[i:i + 7]))

    weekly_total_std = np.std(weekly_residuals) if weekly_residuals else daily_std * 2.5

    if len(residuals) > 2:
        autocorr_1 = np.clip(np.corrcoef(residuals[:-1], residuals[1:])[0, 1], 0.0, 0.95)
    else:
        autocorr_1 = 0.4
    alpha = 0.5 + 0.5 * autocorr_1

    print(f"  Ensemble OOF ({len(oof)} days)")
    print(f"  Daily residual std:  {daily_std:>10,.0f}")
    print(f"  Weekly total std:    {weekly_total_std:>10,.0f}  ({len(weekly_residuals)} windows)")
    print(f"  Weekly avg std:      {weekly_total_std / 7:>10,.0f}")
    print(f"  Lag-1 autocorr:      {autocorr_1:>10.3f}")
    print(f"  Scaling exponent:    {alpha:>10.3f}")

    return weekly_total_std, alpha


# ==========================================================================
# Estimate weekly uncertainty from OOF residuals
# ==========================================================================
def estimate_weekly_uncertainty(oof_path=OOF_PATH):
    """
    Compute weekly total std and autocorrelation from OOF residuals.
    Prefers ensemble OOF (combined_oof.csv) for tighter, regime-aware estimates.
    Falls back to single-model OOF if ensemble OOF not found.
    """
    if ENSEMBLE_OOF_PATH.exists():
        try:
            return _estimate_from_ensemble_oof()
        except Exception as exc:
            print(f"  WARNING: ensemble OOF failed ({exc}), falling back to single-model OOF.")

    if not oof_path.exists():
        print(f"  WARNING: OOF not found at {oof_path}. Using fallback estimates.")
        return 400_000, 0.65

    oof = pd.read_csv(oof_path, parse_dates=["Date"])
    pred_col = next((c for c in ["pred_autogluon", "pred_blend"] if c in oof.columns), None)
    if pred_col is None:
        print(f"  WARNING: No prediction column in OOF file.")
        return 400_000, 0.65

    oof["residual"] = oof[TARGET_COL] - oof[pred_col]
    oof_sorted = oof.sort_values("Date").reset_index(drop=True)
    residuals  = oof_sorted["residual"].values
    daily_std  = oof["residual"].std()

    weekly_residuals = []
    for i in range(len(residuals) - 6):
        dates      = oof_sorted["Date"].iloc[i:i + 7].values
        date_range = (dates[-1] - dates[0]).astype("timedelta64[D]").astype(int)
        if date_range == 6:
            weekly_residuals.append(np.sum(residuals[i:i + 7]))

    weekly_total_std = np.std(weekly_residuals) if weekly_residuals else daily_std * 2.5

    if len(residuals) > 2:
        autocorr_1 = np.clip(np.corrcoef(residuals[:-1], residuals[1:])[0, 1], 0.0, 0.95)
    else:
        autocorr_1 = 0.4
    alpha = 0.5 + 0.5 * autocorr_1

    print(f"  Single-model OOF ({len(oof)} days)")
    print(f"  Daily residual std:  {daily_std:>10,.0f}")
    print(f"  Weekly total std:    {weekly_total_std:>10,.0f}  ({len(weekly_residuals)} windows)")
    print(f"  Weekly avg std:      {weekly_total_std / 7:>10,.0f}")
    print(f"  Lag-1 autocorr:      {autocorr_1:>10.3f}")
    print(f"  Scaling exponent:    {alpha:>10.3f}")

    return weekly_total_std, alpha


# ==========================================================================
# Load raw data for inference
# ==========================================================================
def load_raw_data():
    from build_features import TSA_PATH, WEATHER_PATH, START_DATE

    tsa = pd.read_csv(TSA_PATH)
    tsa.columns = [c.strip() for c in tsa.columns]
    tsa["Date"] = pd.to_datetime(tsa["Date"])
    tsa = tsa.sort_values("Date").drop_duplicates("Date")
    tsa = tsa[tsa["Date"] >= pd.to_datetime(START_DATE)].copy()

    weather = pd.read_csv(WEATHER_PATH)
    weather.columns = [c.strip() for c in weather.columns]
    weather.rename(columns={"date": "Date"}, inplace=True)
    if "Date" not in weather.columns and "ate" in weather.columns:
        weather.rename(columns={"ate": "Date"}, inplace=True)
    weather["Date"] = pd.to_datetime(weather["Date"])
    weather = weather.sort_values("Date").drop_duplicates("Date")

    print(f"  TSA data: {len(tsa)} days ({tsa['Date'].min().date()} → {tsa['Date'].max().date()})")
    print(f"  Weather:  {len(weather)} days ({weather['Date'].min().date()} → {weather['Date'].max().date()})")

    last_tsa = tsa["Date"].max()
    future_weather = weather[weather["Date"] > last_tsa]
    if len(future_weather) > 0:
        print(f"  Weather forecasts: {len(future_weather)} future days "
              f"(through {future_weather['Date'].max().date()})")

    return tsa, weather


# ==========================================================================
# Build features for a future date using the REAL pipeline
# ==========================================================================
def build_features_for_date(target_date, raw_tsa_df, weather_df, predictions):
    """
    Build a complete feature row for target_date by:
    1. Appending all prior predictions to raw TSA data
    2. Appending target_date with NaN volume
    3. Merging with weather
    4. Running the full feature pipeline
    5. Extracting the target date's feature row
    """
    tsa_extended = raw_tsa_df.copy()

    # Use date objects for reliable comparison (avoids Timestamp vs datetime64 issues)
    existing_dates = set(pd.to_datetime(tsa_extended["Date"]).dt.date)

    # Append prior predictions as if they were actual data
    for pred_date, pred_vol in sorted(predictions.items()):
        pred_date = pd.Timestamp(pred_date)
        if pred_date.date() not in existing_dates:
            new_row = pd.DataFrame({"Date": [pred_date], TARGET_COL: [pred_vol]})
            tsa_extended = pd.concat([tsa_extended, new_row], ignore_index=True)
            existing_dates.add(pred_date.date())

    # Append target date with NaN volume
    target_ts = pd.Timestamp(target_date)
    if target_ts.date() not in existing_dates:
        new_row = pd.DataFrame({"Date": [target_ts], TARGET_COL: [np.nan]})
        tsa_extended = pd.concat([tsa_extended, new_row], ignore_index=True)

    tsa_extended = tsa_extended.sort_values("Date").reset_index(drop=True)

    # Merge with weather
    merged = tsa_extended.merge(weather_df, on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)

    # Run the FULL feature pipeline
    featured = build_features_from_df(merged, verbose=False)

    # Extract the row for target_date (use .dt.date for reliable match)
    target_row = featured[pd.to_datetime(featured["Date"]).dt.date == target_ts.date()]

    if target_row.empty:
        raise ValueError(f"Target date {target_date} not found after feature pipeline")

    return target_row


# ==========================================================================
# Determine current week boundaries
# ==========================================================================
def get_current_week_info(tsa_df):
    last_date = tsa_df["Date"].max()

    days_since_monday = last_date.dayofweek
    week_monday = last_date - pd.Timedelta(days=days_since_monday)

    # If entire week is complete, advance to next week
    dates_in_df = set(tsa_df["Date"].dt.date)
    week_dates_check = [week_monday + pd.Timedelta(days=i) for i in range(7)]
    all_present = all(d.date() in dates_in_df for d in week_dates_check)

    if all_present:
        week_monday = week_monday + pd.Timedelta(days=7)
        print(f"  Last week complete — predicting next week")

    week_sunday = week_monday + pd.Timedelta(days=6)

    week_dates = [week_monday + pd.Timedelta(days=i) for i in range(7)]
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

    actual_days = {}
    predict_days = []

    for i, d in enumerate(week_dates):
        ts = pd.Timestamp(d)
        if ts.date() in dates_in_df:
            vol = tsa_df.loc[tsa_df["Date"].dt.date == ts.date(), TARGET_COL].values[0]
            actual_days[ts] = vol
        else:
            predict_days.append(ts)

    return week_monday, week_sunday, week_dates, day_names, actual_days, predict_days


# ==========================================================================
# Weekly forecast
# ==========================================================================
def weekly_forecast(predictor, master_df, raw_tsa_df, weather_df,
                     weekly_total_std=None, alpha=0.65,
                     prophet_bundle=None, ts3=None, meta=None):
    shadow_available = prophet_bundle is not None
    feature_cols = get_feature_columns(master_df)
    week_monday, week_sunday, week_dates, day_names, actual_days, predict_days = \
        get_current_week_info(raw_tsa_df)

    n_actual  = len(actual_days)
    n_predict = len(predict_days)

    print(f"\n  Current week: {week_monday.date()} (Mon) → {week_sunday.date()} (Sun)")
    print(f"  Actual days:     {n_actual}")
    print(f"  Days to predict: {n_predict}")

    # Build full (prune=False) feature_df for ts3/prophet — uses actual TSA history,
    # no autoregressive injection needed for these models.
    feature_df_full = None
    if shadow_available and n_predict > 0:
        try:
            merged = raw_tsa_df.copy().merge(weather_df, on="Date", how="left")
            existing_dates = set(pd.to_datetime(merged["Date"]).dt.date)
            for d in week_dates:
                ts = pd.Timestamp(d)
                if ts.date() not in existing_dates:
                    merged = pd.concat(
                        [merged, pd.DataFrame({"Date": [ts], TARGET_COL: [np.nan]})],
                        ignore_index=True,
                    )
            merged = merged.sort_values("Date").reset_index(drop=True)
            feature_df_full = build_features_from_df(merged, verbose=False, prune=False)
            feature_df_full["Date"] = pd.to_datetime(feature_df_full["Date"])
        except Exception as exc:
            print(f"  WARNING: could not build feature_df for routing ({exc})")
            feature_df_full = None

    predictions = {}
    for d, v in actual_days.items():
        predictions[d] = v

    daily_results = []

    print(f"\n  {'Day':>12} {'Date':>12}  {'Status':>10}  {'Prediction':>12}  {'Regime':>12}")
    print("  " + "-" * 68)

    for i, d in enumerate(week_dates):
        ts       = pd.Timestamp(d)
        day_name = day_names[i]

        if ts in actual_days:
            vol = actual_days[ts]
            print(f"  {day_name:>12} {ts.date()}  {'ACTUAL':>10}  {vol:>12,.0f}  {'—':>12}")
            daily_results.append({
                "Date": ts, "day_name": day_name, "status": "actual",
                "volume": vol, "pred_tabular": vol, "regime": "ACTUAL",
            })
        else:
            print(f"  {day_name:>12} {ts.date()}  building...", end="", flush=True)

            # 1. Tabular prediction (autoregressive chain for correct lag features)
            target_row   = build_features_for_date(ts, raw_tsa_df, weather_df, predictions)
            row_features = target_row[feature_cols].copy()
            pred         = predictor.predict(row_features)
            tabular_pred = float(pred.iloc[0]) if hasattr(pred, "iloc") else float(pred)

            # 2. Ensemble routing (ts3 + prophet + per-regime weights)
            regime      = "NORMAL"
            pred_vol    = tabular_pred
            ts3_pred    = None
            prophet_pred = None
            anchor_pred  = None
            if shadow_available and feature_df_full is not None:
                try:
                    ts3_pred, prophet_pred = _get_ts3_prophet_preds(
                        ts, feature_df_full, prophet_bundle, ts3, meta
                    )
                    pred_vol, regime = _route_day(ts, tabular_pred, ts3_pred, prophet_pred, feature_df_full)
                    row = feature_df_full[feature_df_full["Date"] == ts]
                    if not row.empty:
                        anchor_pred = float(row.iloc[0].get("anchor_master", np.nan) or np.nan)
                        if np.isnan(anchor_pred):
                            anchor_pred = None
                except Exception:
                    pass   # keep tabular_pred and NORMAL regime

            predictions[ts] = pred_vol

            print(f"\r  {day_name:>12} {ts.date()}  {'PREDICTED':>10}  {pred_vol:>12,.0f}  {regime:>12}")
            daily_results.append({
                "Date": ts, "day_name": day_name, "status": "predicted",
                "volume": pred_vol, "pred_tabular": tabular_pred, "regime": regime,
                "pred_ts3": ts3_pred, "pred_prophet": prophet_pred, "pred_anchor": anchor_pred,
            })

    results_df = pd.DataFrame(daily_results)

    # ── Weekly average ────────────────────────────────────────────────
    weekly_total = results_df["volume"].sum()
    weekly_avg   = weekly_total / 7

    # ── Weekly uncertainty ────────────────────────────────────────────
    if n_predict == 0:
        weekly_avg_std = 0
    elif weekly_total_std is not None:
        scale = (n_predict / 7) ** alpha
        weekly_avg_std = (weekly_total_std * scale) / 7
    else:
        weekly_avg_std = 80_000 * np.sqrt(n_predict) / 7

    return results_df, weekly_avg, weekly_avg_std, n_actual, n_predict


# ==========================================================================
# Kalshi odds
# ==========================================================================
def compute_kalshi_odds(weekly_avg, weekly_avg_std, thresholds_millions):
    print(f"\n{'=' * 82}")
    print(f"KALSHI ODDS")
    print("=" * 82)
    print(f"\n  Weekly avg prediction:  {weekly_avg:>12,.0f}  ({weekly_avg/1e6:.4f}M)")
    print(f"  Weekly avg std:         {weekly_avg_std:>12,.0f}  ({weekly_avg_std/1e6:.4f}M)")

    if weekly_avg_std <= 0:
        print(f"\n  All days are actual — weekly average is known with certainty.")
        print(f"  Weekly average = {weekly_avg/1e6:.4f}M")
        for t in thresholds_millions:
            result = "OVER" if weekly_avg > t * 1e6 else "UNDER"
            print(f"    {t:.2f}M: {result}")
        return

    print(f"\n  {'Threshold':>12}  {'P(Over)':>10}  {'P(Under)':>10}")
    print("  " + "-" * 36)

    for t in thresholds_millions:
        threshold = t * 1e6
        z = (threshold - weekly_avg) / weekly_avg_std
        p_over = 1 - stats.norm.cdf(z)
        p_under = stats.norm.cdf(z)
        print(f"  {t:>10.2f}M  {p_over:>10.1%}  {p_under:>10.1%}")


# ==========================================================================
# Feature attribution for "Why the model predicts this"
# ==========================================================================
def attribute_features(
    predictor,
    master_df,
    results_df,
    raw_tsa_df,
    weather_df,
    n_top=TOP_N_DRIVERS,
):
    """Leave-one-out feature attribution for the predicted days of the week.

    For each top-N feature (by training-time permutation importance):
        contribution[feature] = Σ (baseline_pred - ablated_pred) / 7
    summed across the week's predicted days. `ablated_pred` replaces the
    feature with its training-set median; the result is a signed contribution
    to the weekly average, in passengers.
    """
    if not FEATURE_IMPORTANCE_PATH.exists():
        print(f"  WARNING: feature_importance.csv not found at {FEATURE_IMPORTANCE_PATH}")
        return pd.DataFrame()

    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH, index_col=0)
    feature_cols = get_feature_columns(master_df)
    valid = [f for f in importance.index if f in feature_cols]
    top_features = valid[:n_top]
    if not top_features:
        return pd.DataFrame()

    medians = master_df[top_features].median()

    predict_rows = results_df[results_df["status"] == "predicted"]
    if predict_rows.empty:
        print(f"  All days are actual — no features to attribute.")
        return pd.DataFrame()

    # Rebuild the predictions dict from the final results so each day's feature
    # row reflects the full autoregressive chain that produced results_df.
    predictions = {pd.Timestamp(r["Date"]): float(r["volume"]) for _, r in results_df.iterrows()}

    feature_rows = {}
    for _, r in predict_rows.iterrows():
        ts = pd.Timestamp(r["Date"])
        target = build_features_for_date(ts, raw_tsa_df, weather_df, predictions)
        feature_rows[ts] = target[feature_cols].copy()

    # Baseline re-predictions for each predict day. These should match
    # results_df volumes exactly; we re-run them so the attribution math is
    # internally consistent with the ablated calls.
    baseline_preds = {}
    for ts, fr in feature_rows.items():
        p = predictor.predict(fr)
        baseline_preds[ts] = float(p.iloc[0]) if hasattr(p, "iloc") else float(p)

    records = []
    for feat in top_features:
        delta_total = 0.0
        for ts, fr in feature_rows.items():
            ablated = fr.copy()
            ablated[feat] = medians[feat]
            p = predictor.predict(ablated)
            ablated_pred = float(p.iloc[0]) if hasattr(p, "iloc") else float(p)
            delta_total += baseline_preds[ts] - ablated_pred

        contribution_weekly_avg = delta_total / 7.0

        # Representative value from the first predict day's feature row.
        first_ts = next(iter(feature_rows))
        current_val = float(feature_rows[first_ts][feat].iloc[0])

        records.append({
            "feature": feat,
            "contribution_passengers": round(contribution_weekly_avg),
            "current_value": current_val,
            "baseline_value": float(medians[feat]),
            "importance_rank": int(list(importance.index).index(feat)) + 1,
            "importance_passengers": round(float(importance.loc[feat, "importance"])),
        })

    df = pd.DataFrame(records)
    df = df.reindex(df["contribution_passengers"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="Weekly average prediction for Kalshi")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=DEFAULT_THRESHOLDS,
                        help="Kalshi thresholds in millions (default: 2.5 2.55 2.6)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────
    print(f"{'=' * 82}")
    print("Loading Data")
    print("=" * 82)

    master_df = load_master()
    print(f"  Master features: {len(master_df):,} rows, {len(get_feature_columns(master_df))} features")

    raw_tsa_df, weather_df = load_raw_data()
    last_date = raw_tsa_df["Date"].max()
    print(f"  Last TSA date: {last_date.date()} ({last_date.strftime('%A')})")

    # ── Load model ────────────────────────────────────────────────────
    print(f"\n{'=' * 82}")
    print("Model")
    print("=" * 82)
    predictor = load_eval_model()

    # ── Shadow models (Prophet + TS3) ─────────────────────────────────
    print(f"\n{'=' * 82}")
    print("Shadow Models (ensemble routing)")
    print("=" * 82)
    prophet_bundle, ts3, meta = _load_shadow_models()
    if prophet_bundle is not None:
        print(f"  Loaded Prophet + TS3 from {SHADOW_MODELS_DIR}")
    else:
        print(f"  Shadow models not found — falling back to tabular-only routing.")
        print(f"  Run train_shadow_models.py to enable full ensemble.")

    # ── Uncertainty ───────────────────────────────────────────────────
    print(f"\n{'=' * 82}")
    print("Uncertainty (from OOF)")
    print("=" * 82)
    weekly_total_std, alpha = estimate_weekly_uncertainty()

    # ── Weekly forecast ───────────────────────────────────────────────
    print(f"\n{'=' * 82}")
    print("WEEKLY FORECAST")
    print("=" * 82)

    results_df, weekly_avg, weekly_avg_std, n_actual, n_predict = \
        weekly_forecast(predictor, master_df, raw_tsa_df, weather_df,
                        weekly_total_std=weekly_total_std, alpha=alpha,
                        prophet_bundle=prophet_bundle, ts3=ts3, meta=meta)

    # ── Platt-adjusted weekly σ ───────────────────────────────────────
    weekly_avg_std_platt = weekly_avg_std
    if weekly_avg_std > 0 and n_predict > 0:
        predicted_regimes = [
            r["regime"] for r in results_df.to_dict("records")
            if r["status"] == "predicted"
        ]
        platt_ratio_avg = float(np.mean([PLATT_RATIO.get(r, 1.0) for r in predicted_regimes]))
        weekly_avg_std_platt = weekly_avg_std * platt_ratio_avg

    # ── Summary ───────────────────────────────────────────────────────
    weekly_total = results_df["volume"].sum()

    print(f"\n  {'─' * 66}")
    print(f"  Weekly total:       {weekly_total:>12,.0f}")
    print(f"  Weekly average:     {weekly_avg:>12,.0f}  ({weekly_avg/1e6:.4f}M)")
    print(f"  Confidence:         {n_actual}/7 days actual, {n_predict}/7 predicted")

    if "regime" in results_df.columns:
        regime_counts = results_df[results_df["status"] == "predicted"]["regime"].value_counts()
        print(f"  Predicted regimes:  {dict(regime_counts)}")

    if weekly_avg_std > 0:
        print(f"\n  Weekly avg σ (raw ensemble OOF): {weekly_avg_std:>10,.0f}")
        if weekly_avg_std_platt != weekly_avg_std:
            platt_ratio_avg = weekly_avg_std_platt / weekly_avg_std
            print(f"  Platt ratio (avg):               {platt_ratio_avg:>10.3f}")
            print(f"  Weekly avg σ (Platt-adjusted):   {weekly_avg_std_platt:>10,.0f}  ← used for odds")
        ci90_low  = weekly_avg - 1.645 * weekly_avg_std_platt
        ci90_high = weekly_avg + 1.645 * weekly_avg_std_platt
        print(f"  90% CI (Platt):     [{ci90_low/1e6:.4f}M — {ci90_high/1e6:.4f}M]")

    # ── Kalshi odds ───────────────────────────────────────────────────
    compute_kalshi_odds(weekly_avg, weekly_avg_std_platt, args.thresholds)

    # ── Save ──────────────────────────────────────────────────────────
    _move_if_exists(OUT_DIR / "weekly_forecast.csv", OUT_DIR / "prev_weekly_forecast.csv")
    results_df.to_csv(OUT_DIR / "weekly_forecast.csv", index=False)

    summary = pd.DataFrame([{
        "week_monday":                results_df["Date"].iloc[0],
        "week_sunday":                results_df["Date"].iloc[-1],
        "n_actual":                   n_actual,
        "n_predicted":                n_predict,
        "weekly_total":               weekly_total,
        "weekly_avg":                 weekly_avg,
        "weekly_avg_millions":        weekly_avg / 1e6,
        "weekly_avg_std":             weekly_avg_std,             # raw ensemble OOF σ
        "weekly_avg_std_millions":    weekly_avg_std / 1e6,
        "weekly_avg_std_platt":       weekly_avg_std_platt,       # Platt-adjusted σ (used for odds)
        "weekly_avg_std_platt_millions": weekly_avg_std_platt / 1e6,
    }])

    for t in args.thresholds:
        if weekly_avg_std_platt > 0:
            z = (t * 1e6 - weekly_avg) / weekly_avg_std_platt
            summary[f"p_over_{t}M"]  = 1 - stats.norm.cdf(z)
            summary[f"p_under_{t}M"] = stats.norm.cdf(z)
        else:
            summary[f"p_over_{t}M"]  = 1.0 if weekly_avg > t * 1e6 else 0.0
            summary[f"p_under_{t}M"] = 1.0 if weekly_avg <= t * 1e6 else 0.0

    _move_if_exists(OUT_DIR / "weekly_summary.csv", OUT_DIR / "prev_weekly_summary.csv")
    summary.to_csv(OUT_DIR / "weekly_summary.csv", index=False)

    # ── Append to prediction history (local CSV fallback for DB) ─────
    run_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    predicted_rows = results_df[results_df["status"] == "predicted"].copy()
    if not predicted_rows.empty:
        predicted_rows = predicted_rows.rename(columns={"volume": "predicted_volume"})
        predicted_rows["run_date"] = run_date
        predicted_rows["weekly_avg_millions"] = weekly_avg / 1e6
        history_cols = ["run_date", "Date", "day_name", "predicted_volume",
                        "weekly_avg_millions", "pred_tabular", "regime"]
        history_cols = [c for c in history_cols if c in predicted_rows.columns]
        history_path = OUT_DIR / "prediction_history.csv"
        if history_path.exists():
            existing = pd.read_csv(history_path)
            merged = pd.concat([existing, predicted_rows[history_cols]], ignore_index=True)
            merged.drop_duplicates(subset=["run_date", "Date"], keep="last", inplace=True)
        else:
            merged = predicted_rows[history_cols]
        merged.to_csv(history_path, index=False)
        print(f"  Prediction history: {history_path} ({len(merged)} total rows)")

    # ── Feature attributions (top-5 leave-one-out) ───────────────────
    print(f"\n{'=' * 82}")
    print("FEATURE ATTRIBUTIONS (top-5 drivers of this week's forecast)")
    print("=" * 82)
    try:
        attributions_df = attribute_features(predictor, master_df, results_df, raw_tsa_df, weather_df)
        if not attributions_df.empty:
            for _, r in attributions_df.iterrows():
                sign = "+" if r["contribution_passengers"] >= 0 else "−"
                print(f"  {r['feature']:>32}  {sign}{abs(int(r['contribution_passengers'])):>6,}  "
                      f"(rank {int(r['importance_rank'])}, current {r['current_value']:.2f} vs median {r['baseline_value']:.2f})")
            attributions_df.to_csv(OUT_DIR / "feature_attributions.csv", index=False)
        else:
            print(f"  No attributions computed.")
    except Exception as exc:
        print(f"  WARNING: attribution failed (non-fatal): {exc}")

    print(f"\n{'=' * 82}")
    print("FILES SAVED")
    print("=" * 82)
    print(f"  Daily forecast:    {OUT_DIR / 'weekly_forecast.csv'}")
    print(f"  Weekly summary:    {OUT_DIR / 'weekly_summary.csv'}")
    print(f"  Attributions:      {OUT_DIR / 'feature_attributions.csv'}")

    # ── Persist to DB ─────────────────────────────────────────────────
    try:
        import db
        db.write_daily_forecasts(results_df)
        db.write_weekly_summary(summary.iloc[0].to_dict())
    except Exception as exc:
        print(f"[db] write failed (non-fatal): {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()