"""
normal_weight_search.py  (generalized to all learnable regimes)
===============================================================
Find optimal ensemble weights for normal / shoulder_pre / shoulder_post /
peak_holiday using three methods:

  1. NNLS        – non-negative least squares (minimises SSE, SLSQP)
  2. Direct MAE  – minimises MAE directly (SLSQP)
  3. Grid search – exhaustive search on a 0.05 grid, LOO-selected

All methods use leave-one-out (LOO) CV for honest out-of-sample MAE.
Baselines (current production weights, equal, inverse-MAE, tabular-only)
are included per regime for comparison.

Outputs per regime
------------------
  - LOO MAE table: all methods + baselines
  - Final weights (fit on all days in that regime)
  - In-sample MAE and optimism gap
  - Small-n warning when n < 30
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from production_router import (
    get_major_holiday_dates,
    days_to_nearest_major_signed,
    STORM_TRIGGER_IMPACT,
    PEAK_HOLIDAY_WINDOW,
    SHOULDER_WINDOW,
    NORMAL_WEIGHTS,
    SHOULDER_PRE_WEIGHTS,
    SHOULDER_POST_WEIGHTS,
)

# ── paths ──────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF_PATH     = os.path.join(ROOT, "ensemble_experiment", "output", "combined_oof.csv")
WEATHER_PATH = os.path.join(ROOT, "data", "weather_national_features_with_lags.csv")

MODEL_COLS  = ["pred_ag_tabular", "pred_ag_timeseries", "pred_prophet", "pred_anchor_master"]
MODEL_NAMES = ["ag_tabular", "ag_timeseries", "prophet", "anchor_master"]

# Current production weights per regime (tab, ts, prophet, anchor)
PRODUCTION_WEIGHTS = {
    "normal":       np.array(NORMAL_WEIGHTS),
    "shoulder_pre": np.array(SHOULDER_PRE_WEIGHTS),
    "shoulder_post":np.array(SHOULDER_POST_WEIGHTS),
    "peak_holiday": np.array([1.0, 0.0, 0.0, 0.0]),   # tabular only
}

REGIMES_TO_RUN = ["normal", "shoulder_pre", "shoulder_post", "peak_holiday"]

# ── load full OOF with regime labels ───────────────────────────────────────
def load_all_data():
    oof = pd.read_csv(OOF_PATH, parse_dates=["Date"])
    weather = (pd.read_csv(WEATHER_PATH, parse_dates=["date"])
               .rename(columns={"date": "Date"})
               [["Date", "vol_wtd_storm_impact"]])
    oof = oof.merge(weather, on="Date", how="left")
    oof["vol_wtd_storm_impact"] = oof["vol_wtd_storm_impact"].fillna(0.0)
    oof["storm_impact_sq"]   = oof["vol_wtd_storm_impact"] ** 2
    oof["storm_severe_flag"] = (oof["vol_wtd_storm_impact"] > 0.5).astype(int)

    years    = sorted(oof["Date"].dt.year.unique())
    holidays = get_major_holiday_dates(range(min(years) - 1, max(years) + 2))
    oof["days_to_major_signed"] = oof["Date"].apply(
        lambda d: days_to_nearest_major_signed(d, holidays)
    )

    def classify(row):
        if row["storm_severe_flag"] == 1:
            return "severe_storm" if row["storm_impact_sq"] >= STORM_TRIGGER_IMPACT else "moderate_storm"
        d = row["days_to_major_signed"]
        if abs(d) <= PEAK_HOLIDAY_WINDOW:
            return "peak_holiday"
        if PEAK_HOLIDAY_WINDOW < abs(d) <= SHOULDER_WINDOW:
            return "shoulder_pre" if d > 0 else "shoulder_post"
        return "normal"

    oof["regime"] = oof.apply(classify, axis=1)
    return oof


# ── shared helpers ──────────────────────────────────────────────────────────
def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def apply_weights(X, w):
    """X: (n, 4)  w: (4,) → (n,)"""
    return X @ w

def sse_objective(w, X, y):
    return np.sum((y - X @ w) ** 2)

def mae_objective(w, X, y):
    return np.mean(np.abs(y - X @ w))

def fit_weights(X, y, method):
    """Fit weights on (X, y) using the given method. Returns w (4,)."""
    w0 = np.ones(4) / 4
    bounds = [(0, 1)] * 4
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}

    obj = sse_objective if method == "nnls" else mae_objective
    result = minimize(obj, w0, args=(X, y), method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-9, "maxiter": 2000})
    w = result.x
    # Normalise to exactly sum to 1 (numerical noise from SLSQP)
    w = np.clip(w, 0, None)
    w /= w.sum()
    return w


# ── grid search helpers ─────────────────────────────────────────────────────
GRID_STEP = 0.05

def build_weight_grid(step=GRID_STEP):
    """All (w0,w1,w2,w3) with step resolution that sum to 1, each >= 0."""
    n = round(1.0 / step)
    combos = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                d = n - a - b - c
                combos.append((a / n, b / n, c / n, d / n))
    return np.array(combos)   # shape (n_combos, 4)

WEIGHT_GRID = build_weight_grid()

def best_grid_weights(X_train, y_train):
    """Pick the grid combination that minimises MAE on (X_train, y_train)."""
    preds = X_train @ WEIGHT_GRID.T          # (n_train, n_combos)
    maes  = np.mean(np.abs(y_train[:, None] - preds), axis=0)
    return WEIGHT_GRID[np.argmin(maes)]


# ── LOO evaluation ──────────────────────────────────────────────────────────
def loo_mae(X, y, method):
    """
    Leave-one-out CV.
    For each fold: fit weights on (n-1) days → predict held-out day.
    Returns (loo_mae, per-fold errors).
    """
    n = len(y)
    errors = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]

        if method == "grid":
            w = best_grid_weights(X_tr, y_tr)
        else:
            w = fit_weights(X_tr, y_tr, method)

        errors[i] = abs(y[i] - X[i] @ w)

    return errors.mean(), errors


def insample_mae(X, y, method):
    """Fit on all data, return in-sample MAE + weights."""
    if method == "grid":
        w = best_grid_weights(X, y)
    else:
        w = fit_weights(X, y, method)
    return mae(y, X @ w), w


# ── baseline weights ────────────────────────────────────────────────────────
def inverse_mae_weights(X, y):
    per_model_mae = np.array([mae(y, X[:, k]) for k in range(4)])
    inv = 1.0 / np.maximum(per_model_mae, 1)
    return inv / inv.sum()

# ── per-regime runner ──────────────────────────────────────────────────────
def run_regime(regime, df_all):
    df = df_all[df_all["regime"] == regime].reset_index(drop=True)
    X  = df[MODEL_COLS].values.astype(float)
    y  = df["Volume"].values.astype(float)
    n  = len(df)

    W  = 110  # table width
    print("\n" + "█" * W)
    print(f"  REGIME: {regime.upper()}   n={n}  "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    if n < 30:
        print(f"  ⚠  Small sample (n={n}): LOO estimates have higher variance — "
              f"treat optimised weights with caution.")
    print("█" * W)

    # Individual model MAE
    print("\nIndividual model MAE:")
    for k, m in enumerate(MODEL_NAMES):
        print(f"  {m:<18}  {mae(y, X[:, k]):>9,.0f}")

    # Baselines
    results = {}
    w_prod  = PRODUCTION_WEIGHTS[regime]
    w_equal = np.array([0.25, 0.25, 0.25, 0.25])
    w_tab   = np.array([1.00, 0.00, 0.00, 0.00])
    w_inv   = inverse_mae_weights(X, y)

    results["production (current)"] = {"loo_mae": mae(y, X @ w_prod),  "weights": w_prod,  "loo_note": "(no fit)"}
    results["equal weights"]        = {"loo_mae": mae(y, X @ w_equal), "weights": w_equal, "loo_note": "(no fit)"}
    results["tabular only"]         = {"loo_mae": mae(y, X @ w_tab),   "weights": w_tab,   "loo_note": "(no fit)"}
    results["inverse-MAE"]          = {"loo_mae": mae(y, X @ w_inv),   "weights": w_inv,   "loo_note": "(in-sample)"}

    # Optimised methods
    print()
    for method in ("nnls", "mae", "grid"):
        label = {"nnls": "NNLS (min-SSE)", "mae": "Direct MAE",
                 "grid": f"Grid search ({GRID_STEP} step)"}[method]
        print(f"  Running LOO: {label}...", end="  ", flush=True)
        loo, _ = loo_mae(X, y, method)
        is_m, w_final = insample_mae(X, y, method)
        results[label] = {"loo_mae": loo, "is_mae": is_m, "weights": w_final, "loo_note": "LOO"}
        print(f"LOO={loo:,.0f}  in-sample={is_m:,.0f}  optimism={loo-is_m:+,.0f}")

    # Table
    col_w = 14
    print("\n" + "-" * W)
    print(f"{'Method':<30}  {'LOO MAE':>12}  {'In-sample':>10}  {'Optimism':>9}  "
          + "  ".join(f"{m:<{col_w}}" for m in MODEL_NAMES))
    print("-" * W)

    for name, r in results.items():
        w   = r["weights"]
        loo = r["loo_mae"]
        ism = r.get("is_mae", loo)
        opt = loo - ism
        note = r["loo_note"]
        w_str = "  ".join(f"{wi:.3f}" for wi in w)
        loo_disp = f"{loo:>9,.0f} {note}"
        print(f"{name:<30}  {loo_disp:<20}  {ism:>10,.0f}  {opt:>9,.0f}  {w_str}")

    print("-" * W)

    # Best optimised method
    optimised = {k: v for k, v in results.items() if v["loo_note"] == "LOO"}
    best_name = min(optimised, key=lambda k: optimised[k]["loo_mae"])
    best      = optimised[best_name]
    prod_mae  = results["production (current)"]["loo_mae"]
    delta     = best["loo_mae"] - prod_mae
    print(f"\n  Best → {best_name}  |  LOO MAE={best['loo_mae']:,.0f}  "
          f"vs production={prod_mae:,.0f}  (Δ={delta:+,.0f})")
    print("  Recommended weights: "
          + ", ".join(f"{m}={w:.3f}" for m, w in zip(MODEL_NAMES, best["weights"])))

    return {
        "regime": regime, "n": n,
        "best_method": best_name,
        "best_loo_mae": best["loo_mae"],
        "best_weights": best["weights"],
        "prod_mae": prod_mae,
        "results": results,
    }


# ── main ───────────────────────────────────────────────────────────────────
def main():
    print(f"\nGrid step: {GRID_STEP}  →  {len(WEIGHT_GRID):,} weight combinations")
    df_all = load_all_data()

    summaries = []
    for regime in REGIMES_TO_RUN:
        s = run_regime(regime, df_all)
        summaries.append(s)

    # Cross-regime summary
    W = 110
    print("\n\n" + "=" * W)
    print("  CROSS-REGIME SUMMARY — recommended weights (best LOO method per regime)")
    print("=" * W)
    print(f"{'Regime':<15}  {'n':>4}  {'Method':<26}  {'LOO MAE':>9}  {'vs prod':>8}  "
          + "  ".join(f"{m:<14}" for m in MODEL_NAMES))
    print("-" * W)
    for s in summaries:
        w     = s["best_weights"]
        delta = s["best_loo_mae"] - s["prod_mae"]
        w_str = "  ".join(f"{wi:.3f}" for wi in w)
        print(f"{s['regime']:<15}  {s['n']:>4}  {s['best_method']:<26}  "
              f"{s['best_loo_mae']:>9,.0f}  {delta:>+8,.0f}  {w_str}")
    print("=" * W)
    print()

    # Save best weights per regime so downstream scripts (platt_regime.py, etc.)
    # always use the most recent LOO-selected weights without manual copying.
    OUT_DIR = os.path.join(ROOT, "ensemble_experiment", "output")
    weights_json = {
        s["regime"]: [round(float(w), 6) for w in s["best_weights"]]
        for s in summaries
    }
    # Regimes not in the search default to tabular-only.
    for r in ("moderate_storm", "severe_storm"):
        weights_json[r] = [1.0, 0.0, 0.0, 0.0]

    weights_path = os.path.join(OUT_DIR, "best_weights.json")
    import json
    with open(weights_path, "w") as f:
        json.dump(weights_json, f, indent=2)
    print(f"  Saved best weights → {weights_path}\n")


if __name__ == "__main__":
    main()
