"""
platt_regime.py
===============
Fit a separate Platt scaling function per regime on the ensemble OOF predictions.

Platt model per regime r:
    P(actual > T) = sigmoid(A_r * z + B_r)
    where z = (pred_ensemble - T) / σ_r
    and σ_r = std of OOF residuals within regime r

Compared against:
    Gaussian baseline: P = Φ(z)   (i.e. A=1, B=0 assumed)

Evaluation
----------
  - LOO Brier score (leave one day out, all its threshold pairs drop with it)
  - Bootstrap 95% CIs on A and B (1000 resamples)
  - Calibration plot: predicted prob vs empirical freq in 10 equal-width bins
  - Parameters saved to output/platt_params.json for production use

Notes
-----
  - Pooling across thresholds multiplies observations (n_days × n_thresh per regime).
    Observations sharing the same day are not independent, so bootstrap resamples
    by day (not by row) to respect that structure.
  - severe_storm (n=2) is skipped — not enough data to fit.
  - moderate_storm (n=11) is flagged as low-confidence.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.special import expit          # sigmoid
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from production_router import (
    get_major_holiday_dates, days_to_nearest_major_signed,
    STORM_TRIGGER_IMPACT, PEAK_HOLIDAY_WINDOW, SHOULDER_WINDOW,
)

# ── paths ──────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF_PATH     = os.path.join(ROOT, "ensemble_experiment", "output", "combined_oof.csv")
WEATHER_PATH = os.path.join(ROOT, "data", "weather_national_features_with_lags.csv")
OUT_JSON     = os.path.join(ROOT, "ensemble_experiment", "output", "platt_params.json")
WEIGHTS_JSON = os.path.join(ROOT, "ensemble_experiment", "output", "best_weights.json")

MODEL_COLS = ["pred_ag_tabular", "pred_ag_timeseries", "pred_prophet", "pred_anchor_master"]

# Load weights from normal_weight_search.py output if available; fall back to
# the last-known good values so the script can still run standalone.
_FALLBACK_WEIGHTS = {
    "normal":        [0.450, 0.300, 0.000, 0.250],
    "shoulder_pre":  [0.000, 0.079, 0.001, 0.920],
    "shoulder_post": [0.950, 0.050, 0.000, 0.000],
    "peak_holiday":  [1.000, 0.000, 0.000, 0.000],
    "moderate_storm":[1.000, 0.000, 0.000, 0.000],
    "severe_storm":  [1.000, 0.000, 0.000, 0.000],
}
if os.path.exists(WEIGHTS_JSON):
    with open(WEIGHTS_JSON) as _f:
        _raw = json.load(_f)
    NEW_WEIGHTS = {r: np.array(w) for r, w in _raw.items()}
    print(f"[platt_regime] Loaded ensemble weights from {WEIGHTS_JSON}")
else:
    NEW_WEIGHTS = {r: np.array(w) for r, w in _FALLBACK_WEIGHTS.items()}
    print(f"[platt_regime] best_weights.json not found — using fallback weights. "
          f"Run normal_weight_search.py to generate it.")

REGIMES_FIT  = ["normal", "shoulder_pre", "shoulder_post",
                "peak_holiday", "moderate_storm"]
REGIME_SKIP  = {"severe_storm"}          # n=2, skip
N_THRESHOLDS = 20                        # percentile thresholds per regime
N_BOOT       = 1000                      # bootstrap resamples for CIs


# ── data loading ────────────────────────────────────────────────────────────
def load_ensemble_oof():
    oof = pd.read_csv(OOF_PATH, parse_dates=["Date"])
    weather = (pd.read_csv(WEATHER_PATH, parse_dates=["date"])
               .rename(columns={"date": "Date"})[["Date", "vol_wtd_storm_impact"]])
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
        if abs(d) <= PEAK_HOLIDAY_WINDOW:                         return "peak_holiday"
        if PEAK_HOLIDAY_WINDOW < abs(d) <= SHOULDER_WINDOW:
            return "shoulder_pre" if d > 0 else "shoulder_post"
        return "normal"

    oof["regime"] = oof.apply(classify, axis=1)

    # Apply per-regime ensemble weights
    X = oof[MODEL_COLS].values.astype(float)
    oof["pred_ensemble"] = 0.0
    for regime, w in NEW_WEIGHTS.items():
        mask = oof["regime"] == regime
        oof.loc[mask, "pred_ensemble"] = X[mask.values] @ w

    oof["residual"] = oof["Volume"] - oof["pred_ensemble"]
    return oof


# ── threshold + pair generation ─────────────────────────────────────────────
def make_pairs(pred, actual, sigma, n_thresh=N_THRESHOLDS):
    """
    For n_thresh percentile thresholds, create (z, y) pairs.
    Returns arrays z, y and the thresholds used.
    """
    pcts = np.linspace(5, 95, n_thresh)
    thresholds = np.percentile(actual, pcts)

    z_all, y_all, day_idx = [], [], []
    for T in thresholds:
        z = (pred - T) / sigma
        y = (actual > T).astype(float)
        z_all.extend(z.tolist())
        y_all.extend(y.tolist())
        day_idx.extend(range(len(pred)))

    return (np.array(z_all), np.array(y_all),
            np.array(day_idx), thresholds)


# ── Platt fit / predict ──────────────────────────────────────────────────────
def fit_platt(z, y):
    """Fit logistic regression P(y=1) = sigmoid(A*z + B). Returns (A, B)."""
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000, fit_intercept=True)
    lr.fit(z.reshape(-1, 1), y)
    A = float(lr.coef_[0][0])
    B = float(lr.intercept_[0])
    return A, B


def platt_prob(z, A, B):
    return expit(A * z + B)


def gaussian_prob(z):
    return norm.cdf(z)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


# ── LOO Brier ────────────────────────────────────────────────────────────────
def loo_brier(pred, actual, sigma, method="platt", fixed_A=None):
    """
    Leave one DAY out.  Three modes:
      method="platt"    – re-fit A and B on the n-1 training days each fold
      method="gaussian" – A=1, B=0 fixed (raw Gaussian)
      method="fixed_A"  – use fixed_A estimated once from full data; B=0 fixed
                          This tests whether the one-time σ correction helps
                          without the per-fold refit variance that hurt Platt.
    """
    n    = len(pred)
    pcts = np.linspace(5, 95, N_THRESHOLDS)

    total_sq, total_n = 0.0, 0

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        p_tr, a_tr = pred[mask], actual[mask]
        p_ho, a_ho = pred[i],    actual[i]

        thresholds = np.percentile(a_tr, pcts)

        if method == "platt":
            z_tr = np.concatenate([(p_tr - T) / sigma for T in thresholds])
            y_tr = np.concatenate([(a_tr > T).astype(float) for T in thresholds])
            A, B = fit_platt(z_tr, y_tr)
            probs = [platt_prob((p_ho - T) / sigma, A, B) for T in thresholds]
        elif method == "fixed_A":
            probs = [gaussian_prob(fixed_A * (p_ho - T) / sigma) for T in thresholds]
        else:  # gaussian
            probs = [gaussian_prob((p_ho - T) / sigma) for T in thresholds]

        ys = [(a_ho > T) for T in thresholds]
        total_sq += sum((pr - y) ** 2 for pr, y in zip(probs, ys))
        total_n  += len(thresholds)

    return total_sq / total_n


# ── bootstrap CIs ────────────────────────────────────────────────────────────
def bootstrap_AB(pred, actual, sigma, n_boot=N_BOOT, seed=42):
    """Resample days (not rows) to get bootstrap distribution of (A, B)."""
    rng = np.random.default_rng(seed)
    n   = len(pred)
    As, Bs = [], []

    pcts       = np.linspace(5, 95, N_THRESHOLDS)
    thresholds = np.percentile(actual, pcts)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p_b, a_b = pred[idx], actual[idx]
        z_b = np.concatenate([(p_b - T) / sigma for T in thresholds])
        y_b = np.concatenate([(a_b > T).astype(float) for T in thresholds])
        try:
            A, B = fit_platt(z_b, y_b)
            As.append(A); Bs.append(B)
        except Exception:
            pass

    return np.array(As), np.array(Bs)


# ── calibration data for plotting ───────────────────────────────────────────
def calibration_data(pred, actual, sigma, A, B, n_bins=10):
    """
    Bucket all (z, y) pairs by predicted probability into n_bins.
    Returns (bin_centre_pred_prob, empirical_freq, bin_count).
    """
    z_all, y_all, _, _ = make_pairs(pred, actual, sigma)
    p_platt = platt_prob(z_all, A, B)
    p_gauss = gaussian_prob(z_all)

    edges = np.linspace(0, 1, n_bins + 1)

    def bin_it(p):
        centres, freqs, counts = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = (p >= lo) & (p < hi)
            if idx.sum() == 0:
                continue
            centres.append(p[idx].mean())
            freqs.append(y_all[idx].mean())
            counts.append(int(idx.sum()))
        return np.array(centres), np.array(freqs), np.array(counts)

    return bin_it(p_platt), bin_it(p_gauss)


# ── ECE ──────────────────────────────────────────────────────────────────────
def ece(pred_prob, actual_y, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (pred_prob >= lo) & (pred_prob < hi)
        if idx.sum() == 0:
            continue
        total += idx.sum() * abs(pred_prob[idx].mean() - actual_y[idx].mean())
    return total / len(pred_prob)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    df = load_ensemble_oof()

    all_params = {}
    W = 100

    print(f"\nPlatt scaling — per-regime calibration")
    print(f"Thresholds per day: {N_THRESHOLDS}  |  Bootstrap resamples: {N_BOOT}\n")

    for regime in REGIMES_FIT:
        sub  = df[df["regime"] == regime].reset_index(drop=True)
        n    = len(sub)
        pred = sub["pred_ensemble"].values
        act  = sub["Volume"].values
        sigma = sub["residual"].values.std()

        print("█" * W)
        flag = "  ⚠  Low-confidence (n<15)" if n < 15 else ""
        print(f"  REGIME: {regime.upper()}   n={n}   σ={sigma:,.0f}{flag}")
        print("█" * W)

        # ── Full-data Platt fit ────────────────────────────────────────────
        z_all, y_all, day_idx, thresholds = make_pairs(pred, act, sigma)
        A, B = fit_platt(z_all, y_all)

        # ── Bootstrap CIs ─────────────────────────────────────────────────
        print(f"  Bootstrapping CIs ({N_BOOT} resamples)...", end="  ", flush=True)
        As, Bs = bootstrap_AB(pred, act, sigma)
        A_ci = (np.percentile(As, 2.5), np.percentile(As, 97.5))
        B_ci = (np.percentile(Bs, 2.5), np.percentile(Bs, 97.5))
        print("done")

        # ── LOO Brier ─────────────────────────────────────────────────────
        print(f"  Running LOO Brier (3 methods)...", end="  ", flush=True)
        loo_gauss   = loo_brier(pred, act, sigma, method="gaussian")
        loo_fixedA  = loo_brier(pred, act, sigma, method="fixed_A", fixed_A=A)
        loo_platt   = loo_brier(pred, act, sigma, method="platt")
        print("done")

        # ── In-sample Brier (for reference) ───────────────────────────────
        p_platt_is = platt_prob(z_all, A, B)
        p_gauss_is = gaussian_prob(z_all)
        brier_platt_is = brier(p_platt_is, y_all)
        brier_gauss_is = brier(p_gauss_is, y_all)

        # ── ECE ───────────────────────────────────────────────────────────
        ece_platt = ece(p_platt_is, y_all)
        ece_gauss = ece(p_gauss_is, y_all)

        # ── Calibration bins ──────────────────────────────────────────────
        (cp_p, cf_p, cn_p), (cp_g, cf_g, cn_g) = calibration_data(pred, act, sigma, A, B)

        # ── Print results ──────────────────────────────────────────────────
        print(f"\n  Fitted parameters:")
        print(f"    A = {A:+.4f}   95% CI [{A_ci[0]:+.4f}, {A_ci[1]:+.4f}]"
              f"   {'✓ consistent with A=1' if A_ci[0] <= 1 <= A_ci[1] else '✗ A significantly ≠ 1'}")
        print(f"    B = {B:+.4f}   95% CI [{B_ci[0]:+.4f}, {B_ci[1]:+.4f}]"
              f"   {'✓ consistent with B=0' if B_ci[0] <= 0 <= B_ci[1] else '✗ B significantly ≠ 0'}")

        print(f"\n  σ (regime residual std): {sigma:,.0f}")

        pct_fixedA = (loo_gauss - loo_fixedA) / loo_gauss * 100
        pct_platt  = (loo_gauss - loo_platt)  / loo_gauss * 100

        print(f"\n  Brier scores (lower = better):")
        print(f"    {'Method':<25}  {'LOO (honest)':>14}  {'In-sample':>12}  {'ECE':>8}  {'vs Gaussian':>12}")
        print(f"    {'Gaussian (A=1, B=0)':<25}  {loo_gauss:>14.5f}  {brier_gauss_is:>12.5f}  {ece_gauss:>8.5f}  {'baseline':>12}")
        arrow_f = "▼ better" if loo_fixedA < loo_gauss else "▲ worse"
        print(f"    {'Fixed-A  (A={A:.2f}, B=0)':<25}  {loo_fixedA:>14.5f}  {'—':>12}  {'—':>8}  {pct_fixedA:>+10.2f}%  {arrow_f}")
        arrow_p = "▼ better" if loo_platt  < loo_gauss else "▲ worse"
        print(f"    {'Platt    (A,B refitted)':<25}  {loo_platt:>14.5f}  {brier_platt_is:>12.5f}  {ece_platt:>8.5f}  {pct_platt:>+10.2f}%  {arrow_p}")

        print(f"\n  Calibration bins (Platt):  predicted prob → empirical freq")
        for p, f, cnt in zip(cp_p, cf_p, cn_p):
            bar = "█" * int(round(f * 20))
            gap = abs(p - f)
            print(f"    [{p:.2f}]  empirical={f:.3f}  n={cnt:>5}  gap={gap:.3f}  {bar}")

        print(f"\n  Calibration bins (Gaussian):  predicted prob → empirical freq")
        for p, f, cnt in zip(cp_g, cf_g, cn_g):
            bar = "█" * int(round(f * 20))
            gap = abs(p - f)
            print(f"    [{p:.2f}]  empirical={f:.3f}  n={cnt:>5}  gap={gap:.3f}  {bar}")

        print()

        all_params[regime] = {
            "A": round(A, 6), "B": round(B, 6),
            "sigma": round(float(sigma), 2),
            "sigma_effective": round(float(sigma / A), 2),
            "n": n,
            "A_ci_95": [round(A_ci[0], 6), round(A_ci[1], 6)],
            "B_ci_95": [round(B_ci[0], 6), round(B_ci[1], 6)],
            "brier_gaussian_loo":  round(loo_gauss,  6),
            "brier_fixed_A_loo":   round(loo_fixedA, 6),
            "brier_platt_loo":     round(loo_platt,  6),
            "ece_gaussian": round(ece_gauss, 6),
            "ece_platt":    round(ece_platt, 6),
        }

    # ── severe_storm: use global fallback ─────────────────────────────────
    all_params["severe_storm"] = {
        "A": 1.0, "B": 0.0,
        "sigma": round(float(df["residual"].std()), 2),
        "n": int((df["regime"] == "severe_storm").sum()),
        "note": "n=2, skipped — using Gaussian fallback (A=1, B=0)",
    }

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  SUMMARY — per-regime Platt parameters")
    print("=" * W)
    print(f"  {'Regime':<15}  {'n':>4}  {'A':>6}  {'σ_raw':>8}  {'σ_eff':>8}  "
          f"{'Gauss LOO':>11}  {'FixedA LOO':>11}  {'Platt LOO':>11}  "
          f"{'FixedA Δ%':>10}  {'Platt Δ%':>9}")
    print("-" * W)
    for regime, p in all_params.items():
        if "note" in p:
            print(f"  {regime:<15}  {p['n']:>4}  {'—':>6}  {p['sigma']:>8,.0f}  "
                  f"{'—':>8}  {'n/a':>11}  {'n/a':>11}  {'n/a':>11}  {'—':>10}  {'—':>9}")
            continue
        d_f = (p["brier_gaussian_loo"] - p["brier_fixed_A_loo"]) / p["brier_gaussian_loo"] * 100
        d_p = (p["brier_gaussian_loo"] - p["brier_platt_loo"])    / p["brier_gaussian_loo"] * 100
        print(f"  {regime:<15}  {p['n']:>4}  {p['A']:>6.3f}  {p['sigma']:>8,.0f}  "
              f"{p['sigma_effective']:>8,.0f}  "
              f"{p['brier_gaussian_loo']:>11.5f}  {p['brier_fixed_A_loo']:>11.5f}  "
              f"{p['brier_platt_loo']:>11.5f}  {d_f:>+9.2f}%  {d_p:>+8.2f}%")
    print("=" * W)

    # ── Save JSON ──────────────────────────────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump(all_params, f, indent=2)
    print(f"\n  Parameters saved → {OUT_JSON}\n")


if __name__ == "__main__":
    main()
