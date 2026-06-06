"""
plot_calibration.py
===================
Two plots per regime, side by side:
  LEFT  — Residual distribution: histogram + Gaussian fit + Student-t fit
  RIGHT — Isotonic calibration curve: z → P(actual>T), vs Gaussian CDF

Saves: ensemble_experiment/output/calibration_plots.png
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, t as tdist
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from production_router import (
    get_major_holiday_dates, days_to_nearest_major_signed,
    STORM_TRIGGER_IMPACT, PEAK_HOLIDAY_WINDOW, SHOULDER_WINDOW,
)

# ── config ──────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OOF_PATH     = os.path.join(ROOT, "ensemble_experiment", "output", "combined_oof.csv")
WEATHER_PATH = os.path.join(ROOT, "data", "weather_national_features_with_lags.csv")
OUT_PATH     = os.path.join(ROOT, "ensemble_experiment", "output", "calibration_plots.png")

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
REGIME_ORDER = ["normal", "shoulder_pre", "shoulder_post",
                "peak_holiday", "moderate_storm", "severe_storm"]

COLORS = {
    "hist":     "#4C72B0",
    "gaussian": "#DD4949",
    "t_dist":   "#F28E2B",
    "isotonic": "#2CA02C",
    "scatter":  "#AAAAAA",
    "core":     "#E8F4E8",
}


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
    return oof.reset_index(drop=True)


def fit_t_df(residuals, sigma):
    z = residuals / sigma
    result = minimize_scalar(
        lambda df: -np.sum(tdist.logpdf(z, df=df)),
        bounds=(2.1, 50.0), method="bounded")
    return float(result.x)


# ── LEFT: residual distribution ──────────────────────────────────────────────
def plot_distribution(ax, residuals, sigma, df_t, regime, n):
    """Histogram of residuals + Gaussian + Student-t overlays."""
    res = residuals / sigma   # standardised

    # histogram
    counts, bins, _ = ax.hist(
        res, bins=min(20, max(5, n // 3)),
        density=True, color=COLORS["hist"], alpha=0.6,
        edgecolor="white", linewidth=0.5, label="Residuals (std)")

    x = np.linspace(min(-4, res.min() - 0.3), max(4, res.max() + 0.3), 300)

    # Gaussian
    ax.plot(x, norm.pdf(x), color=COLORS["gaussian"],
            linewidth=2, label=f"Gaussian(0,1)")

    # Student-t
    ax.plot(x, tdist.pdf(x, df=df_t), color=COLORS["t_dist"],
            linewidth=2, linestyle="--", label=f"Student-t (df={df_t:.1f})")

    # shade core zone  ±1.28σ ≈ p∈[0.10,0.90]
    ax.axvspan(-1.28, 1.28, alpha=0.08, color=COLORS["isotonic"],
               label="±1.28σ  (80% core)")

    ax.set_title(f"{regime.upper()}  (n={n})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Standardised residual  (actual−pred) / σ", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    ax.set_xlim(x[0], x[-1])

    # annotate σ and df
    ax.text(0.02, 0.97, f"σ = {sigma:,.0f}\ndf_t = {df_t:.1f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


# ── RIGHT: isotonic calibration curve ────────────────────────────────────────
def plot_isotonic(ax, pred, actual, sigma, regime, n):
    """
    Scatter of (z, y) pairs binned into 15 bins + isotonic fit + Gaussian CDF.
    Shaded band shows the tradeable zone p∈[0.20,0.80].
    """
    pcts       = np.linspace(5, 95, N_THRESHOLDS)
    thresholds = np.percentile(actual, pcts)

    z_all = np.concatenate([(pred - T) / sigma for T in thresholds])
    y_all = np.concatenate([(actual > T).astype(float) for T in thresholds])

    # Isotonic fit (full data — for display, not LOO)
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(z_all, y_all)

    # Sort for smooth line
    z_sorted = np.sort(np.unique(z_all))
    p_iso    = iso.predict(z_sorted)

    # Gaussian CDF
    z_fine  = np.linspace(z_sorted[0] - 0.2, z_sorted[-1] + 0.2, 400)
    p_gauss = norm.cdf(z_fine)

    # Binned empirical scatter (15 bins)
    n_bins   = 15
    bin_edges= np.percentile(z_all, np.linspace(0, 100, n_bins + 1))
    bin_z, bin_p, bin_n = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        idx = (z_all >= lo) & (z_all < hi)
        if idx.sum() > 0:
            bin_z.append(z_all[idx].mean())
            bin_p.append(y_all[idx].mean())
            bin_n.append(idx.sum())

    # Shade tradeable zone (p∈[0.20, 0.80])
    z_lo = norm.ppf(0.20)
    z_hi = norm.ppf(0.80)
    ax.axvspan(z_lo, z_hi, alpha=0.10, color=COLORS["isotonic"],
               label="Tradeable zone\np∈[0.20,0.80]")

    # Reference diagonal
    ax.plot(z_fine, p_gauss, color=COLORS["gaussian"],
            linewidth=2, label="Gaussian CDF")
    ax.plot(z_sorted, p_iso, color=COLORS["isotonic"],
            linewidth=2, label="Isotonic fit")

    # Empirical scatter, sized by count
    sizes = [max(20, min(200, c * 3)) for c in bin_n]
    ax.scatter(bin_z, bin_p, s=sizes, color=COLORS["scatter"],
               edgecolors="dimgray", linewidths=0.5, zorder=5,
               label="Empirical (binned)", alpha=0.85)

    ax.set_title(f"{regime.upper()}  calibration curve", fontsize=11, fontweight="bold")
    ax.set_xlabel("z-score  (pred − T) / σ", fontsize=8)
    ax.set_ylabel("P(actual > T)", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Calibration error annotation in core zone
    core = (np.array(bin_z) >= z_lo) & (np.array(bin_z) <= z_hi)
    if core.sum() > 0:
        iso_core_preds = iso.predict(np.array(bin_z)[core])
        avg_err = np.mean(np.abs(iso_core_preds - norm.cdf(np.array(bin_z)[core])))
        ax.text(0.98, 0.04,
                f"Avg |iso−gauss| in core: {avg_err:.3f}",
                transform=ax.transAxes, fontsize=7, ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_data()

    n_regimes = len(REGIME_ORDER)
    fig = plt.figure(figsize=(16, n_regimes * 3.8))
    fig.suptitle(
        "Per-regime calibration: residual distribution (left)  vs  isotonic curve (right)",
        fontsize=13, fontweight="bold", y=0.995)

    gs = gridspec.GridSpec(n_regimes, 2, figure=fig,
                           hspace=0.55, wspace=0.30)

    for row, regime in enumerate(REGIME_ORDER):
        sub = df[df["regime"] == regime]
        n   = len(sub)

        if n == 0:
            for col in range(2):
                ax = fig.add_subplot(gs[row, col])
                ax.text(0.5, 0.5, f"{regime}\n(no data)",
                        ha="center", va="center", transform=ax.transAxes)
            continue

        residuals = sub["residual"].values
        sigma     = float(np.std(residuals))
        pred      = sub["pred_ensemble"].values
        actual    = sub["Volume"].values

        df_t = fit_t_df(residuals, sigma)

        ax_left  = fig.add_subplot(gs[row, 0])
        ax_right = fig.add_subplot(gs[row, 1])

        plot_distribution(ax_left,  residuals, sigma, df_t, regime, n)
        plot_isotonic(ax_right, pred, actual, sigma, regime, n)

        print(f"  {regime:<15}  n={n:>3}  σ={sigma:>8,.0f}  df_t={df_t:.1f}")

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
