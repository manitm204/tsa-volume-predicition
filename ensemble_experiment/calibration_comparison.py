"""
calibration_comparison.py
=========================
Compare three probability calibration methods on the ensemble OOF:

  1. Gaussian        – P = Φ(z)           current baseline
  2. Student's t     – P = t_cdf(z, df)   heavier tails; df fit by MLE per regime
  3. Isotonic        – P = iso(z)          non-parametric; directly learns z → P

All evaluated with LOO.  Metrics are split into:
  - OVERALL  : all (day, threshold) pairs
  - CORE     : pairs where Gaussian predicted probability is in [0.20, 0.80]
               (the "tradeable" zone — where calibration actually matters most)
  - TAILS    : remainder (p < 0.20 or p > 0.80)

Per-regime results are also shown.

z = (pred_ensemble - T) / σ_regime  throughout.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, t as tdist
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from production_router import (
    get_major_holiday_dates, days_to_nearest_major_signed,
    STORM_TRIGGER_IMPACT, PEAK_HOLIDAY_WINDOW, SHOULDER_WINDOW,
)

# ── config ─────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF_PATH     = os.path.join(ROOT, "ensemble_experiment", "output", "combined_oof.csv")
WEATHER_PATH = os.path.join(ROOT, "data", "weather_national_features_with_lags.csv")
OUT_JSON     = os.path.join(ROOT, "ensemble_experiment", "output", "calibration_params.json")

MODEL_COLS = ["pred_ag_tabular", "pred_ag_timeseries", "pred_prophet", "pred_anchor_master"]
NEW_WEIGHTS = {
    "normal":        np.array([0.450, 0.300, 0.000, 0.250]),
    "shoulder_pre":  np.array([0.000, 0.079, 0.001, 0.920]),
    "shoulder_post": np.array([0.950, 0.050, 0.000, 0.000]),
    "peak_holiday":  np.array([1.000, 0.000, 0.000, 0.000]),
    "moderate_storm":np.array([1.000, 0.000, 0.000, 0.000]),
    "severe_storm":  np.array([1.000, 0.000, 0.000, 0.000]),
}

N_THRESHOLDS  = 20
CORE_LO, CORE_HI = 0.20, 0.80   # "tradeable" zone


# ── data ────────────────────────────────────────────────────────────────────
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

    X = oof[MODEL_COLS].values.astype(float)
    oof["pred_ensemble"] = 0.0
    for regime, w in NEW_WEIGHTS.items():
        mask = oof["regime"] == regime
        oof.loc[mask, "pred_ensemble"] = X[mask.values] @ w

    oof["residual"] = oof["Volume"] - oof["pred_ensemble"]

    sigma_map = {}
    for r, g in oof.groupby("regime"):
        sigma_map[r] = float(g["residual"].std())
    oof["sigma"] = oof["regime"].map(sigma_map)

    return oof.reset_index(drop=True), sigma_map


# ── Student's t fit ──────────────────────────────────────────────────────────
def fit_t_df(residuals, sigma):
    """
    Given standardised residuals z = res / sigma, fit the degrees of freedom
    of a Student's t by minimising negative log-likelihood.
    Search over df in [2.1, 50].
    """
    z = residuals / sigma
    def neg_ll(df):
        return -np.sum(tdist.logpdf(z, df=df))
    result = minimize_scalar(neg_ll, bounds=(2.1, 50.0), method="bounded")
    return float(result.x)


# ── isotonic calibration ─────────────────────────────────────────────────────
def fit_isotonic(z_train, y_train):
    """
    Fit monotone non-decreasing function z → P(y=1).
    Returns a callable that interpolates on new z values.
    """
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(z_train, y_train)
    return iso


def iso_predict(iso, z_new):
    return iso.predict(np.atleast_1d(z_new))


# ── probability functions ────────────────────────────────────────────────────
def prob_gaussian(z):
    return norm.cdf(z)

def prob_t(z, df):
    return tdist.cdf(z, df=df)


# ── metrics ──────────────────────────────────────────────────────────────────
EPS = 1e-7

def clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)

def brier(p, y):       return float(np.mean((p - y) ** 2))
def logloss(p, y):     p = clip(p); return float(np.mean(-(y*np.log(p) + (1-y)*np.log(1-p))))
def ece(p, y, b=10):
    edges = np.linspace(0, 1, b + 1)
    total = sum(
        idx.sum() * abs(p[idx].mean() - y[idx].mean())
        for lo, hi in zip(edges[:-1], edges[1:])
        if (idx := (p >= lo) & (p < hi)).sum() > 0
    )
    return total / len(p)

def mce(p, y, b=10):
    edges = np.linspace(0, 1, b + 1)
    return max(
        (abs(p[idx].mean() - y[idx].mean())
         for lo, hi in zip(edges[:-1], edges[1:])
         if (idx := (p >= lo) & (p < hi)).sum() > 0),
        default=0.0
    )

def sharpness(p):    return float(np.std(p))

def auc(p, y):
    if len(np.unique(y)) < 2: return float("nan")
    return float(roc_auc_score(y, p))

def metrics_dict(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return dict(brier=brier(p,y), logloss=logloss(p,y),
                ece=ece(p,y), mce=mce(p,y),
                sharp=sharpness(p), auc=auc(p,y), n=len(y))

def split_core_tails(p_gauss, p_method, y):
    """Split into core (Gaussian p in tradeable zone) and tails."""
    core = (p_gauss >= CORE_LO) & (p_gauss <= CORE_HI)
    tail = ~core
    return (metrics_dict(p_method[core], y[core]),
            metrics_dict(p_method[tail], y[tail]))


# ── LOO runner ───────────────────────────────────────────────────────────────
def run_loo(df):
    n     = len(df)
    pred  = df["pred_ensemble"].values
    act   = df["Volume"].values
    resid = df["residual"].values
    sigma = df["sigma"].values
    pcts  = np.linspace(5, 95, N_THRESHOLDS)

    # Storage: one row per (day × threshold)
    records = []

    print(f"  LOO over {n} days × {N_THRESHOLDS} thresholds...", flush=True)

    for i in range(n):
        mask  = np.ones(n, bool); mask[i] = False
        res_tr = resid[mask]; sig_i  = sigma[i]
        act_tr = act[mask];   pred_tr = pred[mask]
        act_ho = act[i];      pred_ho = pred[i]
        sig_tr = sigma[mask]

        thresholds = np.percentile(act_tr, pcts)

        # ── Fit t df from training residuals ──
        df_t = fit_t_df(res_tr, sig_tr.mean())

        # ── Fit isotonic from training (z, y) pairs ──
        z_tr_all = np.concatenate([(pred_tr - T) / sig_tr for T in thresholds])
        y_tr_all = np.concatenate([(act_tr > T).astype(float) for T in thresholds])
        iso_model = fit_isotonic(z_tr_all, y_tr_all)

        for T in thresholds:
            z_ho  = (pred_ho - T) / sig_i
            y_ho  = float(act_ho > T)
            p_g   = float(prob_gaussian(z_ho))

            records.append({
                "day":     i,
                "regime":  df["regime"].iloc[i],
                "T":       T, "z": z_ho, "y": y_ho,
                "p_gauss": p_g,
                "p_t":     float(prob_t(z_ho, df_t)),
                "p_iso":   float(iso_predict(iso_model, z_ho)[0]),
                "df_t":    df_t,
            })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n}", flush=True)

    print(f"    {n}/{n} — done")
    return pd.DataFrame(records)


# ── printing ─────────────────────────────────────────────────────────────────
def print_comparison(label, gauss_m, t_m, iso_m, zone=""):
    prefix = f"  {zone}" if zone else "  "
    def row(name, m, base):
        db = (base["brier"]   - m["brier"])   / base["brier"]   * 100
        dl = (base["logloss"] - m["logloss"]) / base["logloss"] * 100
        ab = "▼" if db > 0 else ("=" if abs(db) < 0.01 else "▲")
        al = "▼" if dl > 0 else ("=" if abs(dl) < 0.01 else "▲")
        return (f"{prefix}{name:<12}  n={m['n']:>5}  "
                f"Brier={m['brier']:.5f} {ab}{db:+.1f}%  "
                f"LogLoss={m['logloss']:.5f} {al}{dl:+.1f}%  "
                f"ECE={m['ece']:.4f}  MCE={m['mce']:.4f}  "
                f"Sharp={m['sharp']:.4f}  AUC={m['auc']:.5f}")
    print(f"\n  ── {label} ──")
    print(row("Gaussian",   gauss_m, gauss_m))
    print(row("Student-t",  t_m,     gauss_m))
    print(row("Isotonic",   iso_m,   gauss_m))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("\nLoading data...")
    df, sigma_map = load_data()

    print("\nRunning LOO calibration comparison...")
    loo = run_loo(df)

    y   = loo["y"].values
    p_g = loo["p_gauss"].values
    p_t = loo["p_t"].values
    p_i = loo["p_iso"].values

    # ── Overall metrics ───────────────────────────────────────────────────
    print("\n" + "="*100)
    print("  OVERALL  (all regimes, all thresholds)")
    print("="*100)
    core_g, tail_g = split_core_tails(p_g, p_g, y)
    core_t, tail_t = split_core_tails(p_g, p_t, y)
    core_i, tail_i = split_core_tails(p_g, p_i, y)

    print_comparison("ALL PAIRS", metrics_dict(p_g,y), metrics_dict(p_t,y), metrics_dict(p_i,y))
    print_comparison(f"CORE  p∈[{CORE_LO},{CORE_HI}]  — tradeable zone",
                     core_g, core_t, core_i, zone="")
    print_comparison(f"TAILS p<{CORE_LO} or p>{CORE_HI}",
                     tail_g, tail_t, tail_i, zone="")

    # ── Per-regime breakdown ───────────────────────────────────────────────
    print("\n" + "="*100)
    print("  PER-REGIME  (core zone only — the part that matters for trading)")
    print("="*100)

    regime_params = {}
    for regime in sorted(loo["regime"].unique()):
        sub = loo[loo["regime"] == regime]
        y_r   = sub["y"].values
        p_g_r = sub["p_gauss"].values
        p_t_r = sub["p_t"].values
        p_i_r = sub["p_iso"].values
        n_days = sub["day"].nunique()
        df_t_median = sub["df_t"].median()

        core_g_r, _ = split_core_tails(p_g_r, p_g_r, y_r)
        core_t_r, _ = split_core_tails(p_g_r, p_t_r, y_r)
        core_i_r, _ = split_core_tails(p_g_r, p_i_r, y_r)

        print(f"\n  {regime.upper()}  (n={n_days} days, median df_t={df_t_median:.1f})")
        print_comparison(f"CORE p∈[{CORE_LO},{CORE_HI}]",
                         core_g_r, core_t_r, core_i_r)

        regime_params[regime] = {
            "sigma":       round(sigma_map.get(regime, 0), 0),
            "df_t_median": round(float(df_t_median), 2),
            "core_brier_gauss": round(core_g_r["brier"], 6),
            "core_brier_t":     round(core_t_r["brier"], 6),
            "core_brier_iso":   round(core_i_r["brier"], 6),
            "core_logloss_gauss": round(core_g_r["logloss"], 6),
            "core_logloss_t":     round(core_t_r["logloss"], 6),
            "core_logloss_iso":   round(core_i_r["logloss"], 6),
        }

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n\n" + "="*100)
    print(f"  SUMMARY — CORE ZONE (p∈[{CORE_LO},{CORE_HI}]) per regime")
    print("  Winner = method with lowest Brier in core zone")
    print("="*100)
    print(f"  {'Regime':<15}  {'df_t':>6}  "
          f"{'GaussBrier':>11}  {'t Brier':>11}  {'IsoBrier':>11}  "
          f"{'GaussLL':>10}  {'t LL':>10}  {'IsoLL':>10}  {'Winner':>10}")
    print("  " + "-"*98)

    for regime, p in regime_params.items():
        briers   = {"Gaussian": p["core_brier_gauss"],
                    "Student-t": p["core_brier_t"],
                    "Isotonic":  p["core_brier_iso"]}
        lls      = {"Gaussian": p["core_logloss_gauss"],
                    "Student-t": p["core_logloss_t"],
                    "Isotonic":  p["core_logloss_iso"]}
        winner_b = min(briers, key=briers.get)
        winner_l = min(lls,    key=lls.get)
        winner   = winner_b if winner_b == winner_l else f"{winner_b}/LL:{winner_l}"
        print(f"  {regime:<15}  {p['df_t_median']:>6.1f}  "
              f"{p['core_brier_gauss']:>11.5f}  {p['core_brier_t']:>11.5f}  "
              f"{p['core_brier_iso']:>11.5f}  "
              f"{p['core_logloss_gauss']:>10.5f}  {p['core_logloss_t']:>10.5f}  "
              f"{p['core_logloss_iso']:>10.5f}  {winner:>10}")

    print("="*100)

    # Save
    with open(OUT_JSON, "w") as f:
        json.dump(regime_params, f, indent=2)
    print(f"\n  Params saved → {OUT_JSON}\n")


if __name__ == "__main__":
    main()
