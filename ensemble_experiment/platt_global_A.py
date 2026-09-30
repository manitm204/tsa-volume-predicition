"""
platt_global_A.py
=================
Tests whether a single global A (estimated from all 364 OOF days pooled
across regimes) improves calibration over raw Gaussian.

Motivation: per-regime LOO had too little statistical power on n=25–31 to
confirm what the A > 1 signal clearly suggests — the residual σ is too wide.
Pooling all regimes gives n=364 and a tight, reliable A estimate.

Three comparisons (all evaluated with LOO):
  1. Gaussian       – A=1, B=0 fixed
  2. Global Fixed-A – A estimated once from all 364 days (B=0 fixed);
                      in each LOO fold A is RE-ESTIMATED from n-1 days so
                      the held-out day is truly out-of-sample
  3. Per-regime A   – each regime's own A (from platt_regime.py) applied
                      as a fixed correction; A never refitted in LOO

Metrics per method:
  Brier | Log-loss | ECE | MCE | Sharpness | AUC | Calibration slope
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from production_router import (
    get_major_holiday_dates, days_to_prior_and_next_major,
    STORM_TRIGGER_IMPACT, PEAK_HOLIDAY_WINDOW,
    SHOULDER_PRE_WINDOW, SHOULDER_POST_WINDOW,
)

# ── paths ──────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF_PATH     = os.path.join(ROOT, "ensemble_experiment", "output", "combined_oof.csv")
WEATHER_PATH = os.path.join(ROOT, "data", "weather_national_features_with_lags.csv")
PLATT_JSON   = os.path.join(ROOT, "ensemble_experiment", "output", "platt_params.json")

MODEL_COLS = ["pred_ag_tabular", "pred_ag_timeseries", "pred_prophet", "pred_anchor_master"]

NEW_WEIGHTS = {
    "normal":        np.array([0.450, 0.300, 0.000, 0.250]),
    "shoulder_pre":  np.array([0.000, 0.079, 0.001, 0.920]),
    "shoulder_post": np.array([0.950, 0.050, 0.000, 0.000]),
    "peak_holiday":  np.array([1.000, 0.000, 0.000, 0.000]),
    "moderate_storm":np.array([1.000, 0.000, 0.000, 0.000]),
    "severe_storm":  np.array([1.000, 0.000, 0.000, 0.000]),
}

N_THRESHOLDS = 20


# ── data loading ────────────────────────────────────────────────────────────
def load_data():
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

    X = oof[MODEL_COLS].values.astype(float)
    oof["pred_ensemble"] = 0.0
    for regime, w in NEW_WEIGHTS.items():
        mask = oof["regime"] == regime
        oof.loc[mask, "pred_ensemble"] = X[mask.values] @ w

    # Regime-specific σ
    sigma_map = (oof.groupby("regime")
                 .apply(lambda g: (g["Volume"] - g["pred_ensemble"]).std())
                 .to_dict())
    oof["sigma"] = oof["regime"].map(sigma_map)

    return oof.reset_index(drop=True), sigma_map


# ── pair generation ─────────────────────────────────────────────────────────
def make_pairs_df(df, n_thresh=N_THRESHOLDS):
    """
    Build a flat DataFrame of (day_idx, z, y, threshold) across all days.
    z = (pred - T) / sigma_regime.
    """
    pred  = df["pred_ensemble"].values
    act   = df["Volume"].values
    sigma = df["sigma"].values

    pcts       = np.linspace(5, 95, n_thresh)
    thresholds = np.percentile(act, pcts)

    rows = []
    for i, T in enumerate(thresholds):
        z = (pred - T) / sigma
        y = (act > T).astype(float)
        for j in range(len(df)):
            rows.append({"day_idx": j, "z": z[j], "y": y[j], "T": T})

    return pd.DataFrame(rows), thresholds


# ── fit single A ─────────────────────────────────────────────────────────────
def fit_A(z, y):
    """Fit logistic regression with intercept=0 (B fixed at 0): P = sigmoid(A*z)."""
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000, fit_intercept=False)
    lr.fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0][0])


# ── metrics ──────────────────────────────────────────────────────────────────
EPS = 1e-7

def clip_prob(p):
    return np.clip(p, EPS, 1 - EPS)

def brier_score(p, y):
    return float(np.mean((p - y) ** 2))

def log_loss_score(p, y):
    p = clip_prob(np.asarray(p))
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

def ece_score(p, y, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (p >= lo) & (p < hi)
        if idx.sum() == 0: continue
        total += idx.sum() * abs(p[idx].mean() - y[idx].mean())
    return total / len(p)

def mce_score(p, y, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (p >= lo) & (p < hi)
        if idx.sum() == 0: continue
        worst = max(worst, abs(p[idx].mean() - y[idx].mean()))
    return worst

def sharpness(p):
    return float(np.std(p))

def auc_score(p, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))

def calibration_slope(p, y):
    """
    Regress y on logit(p). Slope ≈ 1 → well calibrated width.
    Slope < 1 → overconfident (probabilities too extreme).
    Slope > 1 → underconfident (probabilities too narrow / σ too wide).
    """
    p = clip_prob(np.asarray(p, dtype=float))
    lp = logit(p)
    # Simple OLS: slope = cov(lp, y) / var(lp)
    slope = np.cov(lp, y)[0, 1] / np.var(lp)
    return float(slope)

def all_metrics(p, y, label):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return {
        "label":    label,
        "brier":    brier_score(p, y),
        "logloss":  log_loss_score(p, y),
        "ece":      ece_score(p, y),
        "mce":      mce_score(p, y),
        "sharp":    sharpness(p),
        "auc":      auc_score(p, y),
        "cal_slope":calibration_slope(p, y),
    }


# ── LOO evaluation ────────────────────────────────────────────────────────────
def run_loo(df, pairs_df, per_regime_A):
    """
    For each held-out day, compute probabilities under 3 methods,
    record all (threshold, y, p_gauss, p_global, p_per_regime) rows.
    """
    n    = len(df)
    pred = df["pred_ensemble"].values
    act  = df["Volume"].values
    sig  = df["sigma"].values
    pcts = np.linspace(5, 95, N_THRESHOLDS)

    rows = []

    print(f"  Running LOO over {n} days × {N_THRESHOLDS} thresholds...", flush=True)

    for i in range(n):
        mask   = np.ones(n, dtype=bool); mask[i] = False
        act_tr = act[mask]; pred_tr = pred[mask]; sig_tr = sig[mask]
        act_ho = act[i]; pred_ho = pred[i]; sig_ho = sig[i]
        regime_ho = df["regime"].iloc[i]

        thresholds = np.percentile(act_tr, pcts)

        # ── Re-estimate global A from n-1 days ──
        z_tr = np.concatenate([(pred_tr - T) / sig_tr for T in thresholds])
        y_tr = np.concatenate([(act_tr > T).astype(float) for T in thresholds])
        try:
            A_global = fit_A(z_tr, y_tr)
        except Exception:
            A_global = 1.0

        # ── Per-regime A (never re-estimated) ──
        A_regime = per_regime_A.get(regime_ho, 1.0)

        for T in thresholds:
            z_ho = (pred_ho - T) / sig_ho
            y_ho = float(act_ho > T)
            rows.append({
                "day_idx":  i,
                "regime":   regime_ho,
                "T":        T,
                "y":        y_ho,
                "p_gauss":  norm.cdf(z_ho),
                "p_global": norm.cdf(A_global * z_ho),
                "p_regime": norm.cdf(A_regime * z_ho),
            })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n} done", flush=True)

    print(f"    {n}/{n} done")
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("\nLoading data...")
    df, sigma_map = load_data()
    n = len(df)
    print(f"  Total days: {n}  |  Regimes: {df['regime'].value_counts().to_dict()}")

    # ── Load per-regime A from prior run ──────────────────────────────────
    with open(PLATT_JSON) as f:
        platt_params = json.load(f)
    per_regime_A = {r: p["A"] for r, p in platt_params.items() if "A" in p and isinstance(p["A"], float)}
    print(f"\n  Per-regime A values: { {r: round(a, 3) for r, a in per_regime_A.items()} }")

    # ── Full-data global A (B=0 fixed) ────────────────────────────────────
    pairs_df, _ = make_pairs_df(df)
    A_global_full = fit_A(pairs_df["z"].values, pairs_df["y"].values)
    print(f"\n  Global A (full data, B=0 fixed): {A_global_full:.4f}")
    print(f"  Implied σ shrinkage: {1/A_global_full:.3f}×  (σ_eff = σ / {A_global_full:.3f})")

    # ── Run LOO ───────────────────────────────────────────────────────────
    print("\nRunning LOO (global A re-estimated each fold)...")
    loo_df = run_loo(df, pairs_df, per_regime_A)

    y   = loo_df["y"].values
    p_g = loo_df["p_gauss"].values
    p_gl= loo_df["p_global"].values
    p_r = loo_df["p_regime"].values

    # ── Overall metrics ───────────────────────────────────────────────────
    print("\n" + "="*95)
    print("  OVERALL LOO METRICS  (all regimes pooled, all thresholds)")
    print("="*95)

    results = [
        all_metrics(p_g,  y, "Gaussian     (A=1)"),
        all_metrics(p_gl, y, "Global Fixed-A (LOO re-fit)"),
        all_metrics(p_r,  y, "Per-regime A   (fixed from full data)"),
    ]

    hdr = f"  {'Method':<32}  {'Brier':>8}  {'LogLoss':>8}  {'ECE':>7}  {'MCE':>7}  {'Sharp':>7}  {'AUC':>7}  {'CalSlope':>9}"
    print(hdr)
    print("  " + "-" * 93)

    baseline_brier   = results[0]["brier"]
    baseline_logloss = results[0]["logloss"]

    for r in results:
        d_b = (baseline_brier   - r["brier"])   / baseline_brier   * 100
        d_l = (baseline_logloss - r["logloss"]) / baseline_logloss * 100
        arrow_b = "▼" if r["brier"]   < baseline_brier   else ("=" if abs(d_b) < 0.01 else "▲")
        arrow_l = "▼" if r["logloss"] < baseline_logloss else ("=" if abs(d_l) < 0.01 else "▲")
        print(f"  {r['label']:<32}  {r['brier']:>8.5f}  {r['logloss']:>8.5f}  "
              f"{r['ece']:>7.5f}  {r['mce']:>7.5f}  {r['sharp']:>7.5f}  "
              f"{r['auc']:>7.5f}  {r['cal_slope']:>9.4f}  "
              f"Brier{arrow_b}{d_b:+.2f}%  LogLoss{arrow_l}{d_l:+.2f}%")

    # ── Per-regime breakdown ───────────────────────────────────────────────
    print("\n" + "="*95)
    print("  PER-REGIME LOO METRICS")
    print("="*95)

    for regime in sorted(loo_df["regime"].unique()):
        sub = loo_df[loo_df["regime"] == regime]
        y_r   = sub["y"].values
        p_g_r = sub["p_gauss"].values
        p_gl_r= sub["p_global"].values
        p_r_r = sub["p_regime"].values
        n_days = sub["day_idx"].nunique()

        r_gauss  = all_metrics(p_g_r,  y_r, "Gaussian")
        r_global = all_metrics(p_gl_r, y_r, "Global-A")
        r_regime = all_metrics(p_r_r,  y_r, "Regime-A")

        print(f"\n  {regime.upper()}  (n={n_days} days)")
        print(f"  {'Method':<20}  {'Brier':>8}  {'LogLoss':>8}  {'ECE':>7}  {'MCE':>7}  "
              f"{'Sharp':>7}  {'AUC':>7}  {'CalSlope':>9}  {'BrierΔ%':>8}  {'LogLossΔ%':>10}")
        print("  " + "-" * 93)
        for r in [r_gauss, r_global, r_regime]:
            d_b = (r_gauss["brier"]   - r["brier"])   / r_gauss["brier"]   * 100
            d_l = (r_gauss["logloss"] - r["logloss"]) / r_gauss["logloss"] * 100
            print(f"  {r['label']:<20}  {r['brier']:>8.5f}  {r['logloss']:>8.5f}  "
                  f"{r['ece']:>7.5f}  {r['mce']:>7.5f}  {r['sharp']:>7.5f}  "
                  f"{r['auc']:>7.5f}  {r['cal_slope']:>9.4f}  "
                  f"{d_b:>+7.2f}%  {d_l:>+9.2f}%")

    # ── Calibration slope summary ─────────────────────────────────────────
    print("\n" + "="*70)
    print("  CALIBRATION SLOPE SUMMARY  (target = 1.0)")
    print("  Slope < 1 → overconfident  |  Slope > 1 → σ too wide")
    print("="*70)
    print(f"  {'Method':<32}  {'Overall slope':>14}")
    print("  " + "-" * 48)
    for r in results:
        delta = r["cal_slope"] - 1.0
        verdict = "✓ well calibrated" if abs(delta) < 0.05 else (
                  "σ too wide — shrink" if delta > 0 else "overconfident — widen")
        print(f"  {r['label']:<32}  {r['cal_slope']:>14.4f}  ({verdict})")

    print()


if __name__ == "__main__":
    main()
