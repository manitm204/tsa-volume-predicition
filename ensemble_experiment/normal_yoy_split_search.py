"""
normal_yoy_split_search.py
===========================
Does re-admitting yoy_delta help once NORMAL is split by
yoy_delta_trend_available (build_features.add_yoy_delta_feature)?

combined_oof.csv's existing pred_yoy_delta column (from yoy_delta_oof.py) is
UNGUARDED — no holiday-contamination zeroing at all, unlike the production
feature (build_features.add_yoy_delta_feature). This script recomputes
pred_yoy_delta the way the live feature actually does it (holiday guard +
the new delta_1 fallback), splits the "normal" regime rows from
normal_weight_search.py into normal_yoy_available / normal_yoy_unavailable
by that guard, and runs the same LOO NNLS/MAE/grid search as
normal_weight_search.py, now with yoy_delta as a 4th candidate model, on
each half.

Answers: is NORMAL_YOY_AVAILABLE_WEIGHTS's current (0.25,0.25,0.25,0.25)
placeholder in production_router.py actually beaten by giving yoy_delta
real weight, or is dropping it to 0 (same as NORMAL_YOY_UNAVAILABLE) still
best even on the "clean" half?
"""

import os
import sys

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from production_router import (
    get_major_holiday_dates,
    days_to_prior_and_next_major,
    STORM_TRIGGER_IMPACT,
    PEAK_HOLIDAY_WINDOW,
    SHOULDER_PRE_WINDOW,
    SHOULDER_POST_WINDOW,
)
from build_features import _get_major_holiday_dates

OOF_PATH     = os.path.join(ROOT, "ensemble_experiment", "output", "combined_oof.csv")
WEATHER_PATH = os.path.join(ROOT, "data", "weather_national_features_with_lags.csv")
VOLUME_PATH  = os.path.join(ROOT, "data", "tsa_volume.csv")

MODEL_COLS  = ["pred_ag_tabular", "pred_ag_timeseries", "pred_yoy_delta_guarded", "pred_anchor_master"]
MODEL_NAMES = ["ag_tabular", "ag_timeseries", "yoy_delta", "anchor_master"]
N_MODELS    = len(MODEL_COLS)
GRID_STEP   = 0.05


# ── recompute pred_yoy_delta the way the live feature does (guard + fallback) ──
def compute_guarded_yoy_delta(dates):
    vol_df = pd.read_csv(VOLUME_PATH, parse_dates=["Date"])
    d_all = vol_df["Date"]
    v_all = vol_df["Volume"]

    cal = USFederalHolidayCalendar()
    fed = pd.to_datetime(cal.holidays(start=d_all.min(), end=d_all.max() + pd.Timedelta(days=400)))
    major = _get_major_holiday_dates(sorted(d_all.dt.year.unique()))
    all_h = pd.DatetimeIndex(
        pd.concat([fed.to_series(), major["easter"].to_series(), major["halloween"].to_series()])
    ).sort_values().drop_duplicates()

    hdays = all_h.values.astype("datetime64[D]")
    alld = d_all.values.astype("datetime64[D]")
    signed = []
    for day in alld:
        diffs = hdays - day
        idx = np.argmin(np.abs(diffs))
        signed.append(int(diffs[idx].astype("timedelta64[D]").astype(int)))
    dth = pd.Series([-x for x in signed], index=vol_df.index)  # + = post-holiday

    def adj(x):
        return (x >= -SHOULDER_PRE_WINDOW) & (x <= SHOULDER_POST_WINDOW)

    c_D = adj(dth)
    contaminated_1 = c_D | adj(dth.shift(7)) | adj(dth.shift(371))
    contaminated_2 = c_D | adj(dth.shift(14)) | adj(dth.shift(378))

    ly = v_all.shift(364)
    delta_1 = v_all.shift(7) - v_all.shift(371)
    delta_2 = v_all.shift(14) - v_all.shift(378)
    delta_1 = delta_1.where(~contaminated_1, 0.0)
    delta_2 = delta_2.where(~contaminated_2, delta_1)  # the fallback

    vol_df["pred_yoy_delta_guarded"] = ly + 0.8 * delta_1 + 0.2 * delta_2
    vol_df["yoy_delta_trend_available"] = (~contaminated_1).astype(int)

    lookup = vol_df.set_index("Date")[["pred_yoy_delta_guarded", "yoy_delta_trend_available"]]
    out = lookup.reindex(dates)
    return out["pred_yoy_delta_guarded"].values, out["yoy_delta_trend_available"].values


def load_all_data():
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
        if min(p, n) <= PEAK_HOLIDAY_WINDOW:
            return "peak_holiday"
        if PEAK_HOLIDAY_WINDOW < p <= SHOULDER_POST_WINDOW:
            return "shoulder_post"
        if PEAK_HOLIDAY_WINDOW < n <= SHOULDER_PRE_WINDOW:
            return "shoulder_pre"
        return "normal"

    oof["regime"] = oof.apply(classify, axis=1)

    guarded, avail = compute_guarded_yoy_delta(oof["Date"])
    oof["pred_yoy_delta_guarded"] = guarded
    oof["yoy_delta_trend_available"] = avail

    return oof


# ── fit / LOO machinery (same as normal_weight_search.py) ──────────────────
def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def fit_weights(X, y, method):
    k = X.shape[1]
    w0 = np.ones(k) / k
    bounds = [(0, 1)] * k
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    obj = (lambda w, X, y: np.sum((y - X @ w) ** 2)) if method == "nnls" \
        else (lambda w, X, y: np.mean(np.abs(y - X @ w)))
    result = minimize(obj, w0, args=(X, y), method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-9, "maxiter": 2000})
    w = np.clip(result.x, 0, None)
    w /= w.sum()
    return w

def build_weight_grid(step=GRID_STEP, dims=N_MODELS):
    n = round(1.0 / step)
    def compositions(total, parts):
        if parts == 1:
            yield (total,); return
        for head in range(total + 1):
            for rest in compositions(total - head, parts - 1):
                yield (head,) + rest
    return np.array(list(compositions(n, dims))) / n

WEIGHT_GRID = build_weight_grid()

def best_grid_weights(X_train, y_train, grid):
    preds = X_train @ grid.T
    maes  = np.mean(np.abs(y_train[:, None] - preds), axis=0)
    return grid[np.argmin(maes)]

def loo_mae(X, y, method):
    n = len(y)
    errors = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        w = best_grid_weights(X_tr, y_tr, WEIGHT_GRID) if method == "grid" else fit_weights(X_tr, y_tr, method)
        errors[i] = abs(y[i] - X[i] @ w)
    return errors.mean(), errors

def insample_mae(X, y, method):
    w = best_grid_weights(X, y, WEIGHT_GRID) if method == "grid" else fit_weights(X, y, method)
    return mae(y, X @ w), w

def inverse_mae_weights(X, y):
    per_model_mae = np.array([mae(y, X[:, k]) for k in range(X.shape[1])])
    inv = 1.0 / np.maximum(per_model_mae, 1)
    return inv / inv.sum()


def run_bucket(label, df):
    X = df[MODEL_COLS].values.astype(float)
    y = df["Volume"].values.astype(float)
    n = len(df)

    W = 100
    print("\n" + "█" * W)
    print(f"  BUCKET: {label}   n={n}  ({df['Date'].min().date()} -> {df['Date'].max().date()})")
    if n < 30:
        print(f"  ⚠  Small sample (n={n}): LOO estimates have higher variance.")
    print("█" * W)

    print("\nIndividual model MAE:")
    for k, m in enumerate(MODEL_NAMES):
        print(f"  {m:<18}  {mae(y, X[:, k]):>9,.0f}")

    results = {}
    w_prod_no_yoy = np.array([0.333, 0.333, 0.000, 0.333])   # current production (yoy pinned 0)
    w_placeholder = np.array([0.250, 0.250, 0.250, 0.250])   # NORMAL_YOY_AVAILABLE_WEIGHTS placeholder
    w_tab_only    = np.array([1.0, 0.0, 0.0, 0.0])
    w_inv         = inverse_mae_weights(X, y)

    results["production (yoy=0)"]      = {"loo_mae": mae(y, X @ w_prod_no_yoy), "weights": w_prod_no_yoy, "loo_note": "(no fit)"}
    results["placeholder (equal 0.25)"]= {"loo_mae": mae(y, X @ w_placeholder), "weights": w_placeholder, "loo_note": "(no fit)"}
    results["tabular only"]            = {"loo_mae": mae(y, X @ w_tab_only),    "weights": w_tab_only,    "loo_note": "(no fit)"}
    results["inverse-MAE"]             = {"loo_mae": mae(y, X @ w_inv),         "weights": w_inv,         "loo_note": "(in-sample)"}

    print()
    for method in ("nnls", "mae", "grid"):
        mlabel = {"nnls": "NNLS (min-SSE)", "mae": "Direct MAE", "grid": f"Grid search ({GRID_STEP} step)"}[method]
        print(f"  Running LOO: {mlabel}...", end="  ", flush=True)
        loo, _ = loo_mae(X, y, method)
        is_m, w_final = insample_mae(X, y, method)
        results[mlabel] = {"loo_mae": loo, "is_mae": is_m, "weights": w_final, "loo_note": "LOO"}
        print(f"LOO={loo:,.0f}  in-sample={is_m:,.0f}  optimism={loo-is_m:+,.0f}")

    print("\n" + "-" * W)
    print(f"{'Method':<30}  {'LOO MAE':>18}  {'In-sample':>10}  " + "  ".join(f"{m:<10}" for m in MODEL_NAMES))
    print("-" * W)
    for name, r in results.items():
        w = r["weights"]
        ism = r.get("is_mae", r["loo_mae"])
        w_str = "  ".join(f"{wi:.3f}" for wi in w)
        print(f"{name:<30}  {r['loo_mae']:>9,.0f} {r['loo_note']:<8}  {ism:>10,.0f}  {w_str}")
    print("-" * W)

    prod_mae = results["production (yoy=0)"]["loo_mae"]
    optimised = {k: v for k, v in results.items() if v["loo_note"] == "LOO"}
    best_name = min(optimised, key=lambda k: optimised[k]["loo_mae"])
    best = optimised[best_name]
    print(f"\n  Best fitted → {best_name}  |  LOO MAE={best['loo_mae']:,.0f}  "
          f"vs production(yoy=0)={prod_mae:,.0f}  (Δ={best['loo_mae']-prod_mae:+,.0f})")
    print("  Recommended weights: " + ", ".join(f"{m}={w:.3f}" for m, w in zip(MODEL_NAMES, best["weights"])))
    return {"label": label, "n": n, "results": results, "best_name": best_name, "best": best, "prod_mae": prod_mae}


def main():
    df_all = load_all_data()
    normal = df_all[df_all["regime"] == "normal"].dropna(subset=MODEL_COLS + ["Volume"]).reset_index(drop=True)
    avail   = normal[normal["yoy_delta_trend_available"] == 1].reset_index(drop=True)
    unavail = normal[normal["yoy_delta_trend_available"] == 0].reset_index(drop=True)

    print(f"\nNORMAL regime OOF rows: {len(normal)}  "
          f"(trend_available={len(avail)}, unavailable={len(unavail)})")

    s_avail   = run_bucket("NORMAL_YOY_AVAILABLE", avail)
    s_unavail = run_bucket("NORMAL_YOY_UNAVAILABLE", unavail)

    print("\n\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    for s in (s_avail, s_unavail):
        print(f"{s['label']:<24} n={s['n']:<5} best={s['best_name']:<20} "
              f"LOO={s['best']['loo_mae']:>9,.0f}  vs_prod(yoy=0)={s['best']['loo_mae']-s['prod_mae']:>+8,.0f}")


if __name__ == "__main__":
    main()
