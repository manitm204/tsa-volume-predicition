#!/usr/bin/env python3
# Run: uvicorn main:app --reload --port 8000
from __future__ import annotations

import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).parent.parent.parent  # tsa/
sys.path.insert(0, str(BASE))

# Load .env so Kalshi auth + Postgres work inside uvicorn
_env_path = BASE / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip('"').strip("'")
                if _key and _key not in os.environ:
                    os.environ[_key] = _val

import db

app = FastAPI(title="TSA Dashboard API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── TTL cache (avoids hammering Kalshi API on every request) ──────────────────
_cache: dict[str, tuple[float, Any]] = {}

def _cached(key: str, ttl: int, fn):
    now = time.time()
    if key in _cache and (now - _cache[key][0]) < ttl:
        return _cache[key][1]
    result = fn()
    _cache[key] = (now, result)
    return result

# ── Data readers ──────────────────────────────────────────────────────────────
def _safe(val, divisor: float = 1.0) -> float | None:
    try:
        v = float(val)
        return None if pd.isna(v) else v / divisor
    except (TypeError, ValueError):
        return None

def _strike(ticker: str) -> float | None:
    for part in ticker.split("-"):
        if part.startswith("A"):
            try: return float(part[1:])
            except ValueError: pass
    return None

def _weekly_forecast() -> pd.DataFrame:
    p = BASE / "output_autogluon_predict/weekly_forecast.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _weekly_summary() -> dict:
    p = BASE / "output_autogluon_predict/weekly_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df.iloc[0].to_dict() if not df.empty else {}
    return {}

def _prev_weekly_summary() -> dict:
    p = BASE / "output_autogluon_predict/prev_weekly_summary.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df.iloc[0].to_dict() if not df.empty else {}
    return {}

def _snapshot() -> pd.DataFrame:
    p = BASE / "output_kalshi/market_snapshot_latest.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _tsa() -> pd.DataFrame:
    p = BASE / "data/tsa_volume.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["Date"])
    df.columns = [c.strip() for c in df.columns]
    return df.sort_values("Date").reset_index(drop=True)

def _portfolio() -> dict:
    def _fetch():
        try:
            from kalshi import fetch_portfolio_summary
            result = fetch_portfolio_summary()
            result["source"] = "live"
            # Update cache with richer data for fallback
            try:
                cache_path = BASE / "output_kalshi/positions.json"
                cache_data = {}
                for p in result.get("positions", []):
                    cache_data[p["ticker"]] = {
                        "yes": p["yes"], "no": p["no"],
                        "avg_price": p.get("avg_price"),
                        "cost_dollars": p.get("cost_dollars", 0),
                    }
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f, indent=2)
            except Exception:
                pass
            return result
        except Exception as exc:
            print(f"[warn] portfolio API: {exc}", file=sys.stderr)
            positions_path = BASE / "output_kalshi/positions.json"
            positions = []
            if positions_path.exists():
                with open(positions_path) as f:
                    cache = json.load(f)
                for ticker, counts in cache.items():
                    positions.append({
                        "ticker": ticker,
                        "yes": counts.get("yes", 0),
                        "no": counts.get("no", 0),
                        "avg_price": counts.get("avg_price"),
                        "cost_dollars": counts.get("cost_dollars", 0.0),
                    })
            return {"positions": positions, "open_orders": [], "source": "cache"}
    return _cached("portfolio", 60, _fetch)

def _actions() -> pd.DataFrame:
    files = sorted(glob.glob(str(BASE / "output_kalshi/actions_*.csv")))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        ts = Path(f).stem.replace("actions_", "")
        try:
            df = pd.read_csv(f)
            df["run_ts"] = ts
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _last_run() -> dict:
    snaps = sorted(glob.glob(str(BASE / "output_kalshi/market_snapshot_2*.csv")))
    if not snaps:
        return {"last_run_at": None, "status": "unknown"}
    ts = Path(snaps[-1]).stem.replace("market_snapshot_", "")
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        return {"last_run_at": dt.isoformat(), "status": "ok"}
    except ValueError:
        return {"last_run_at": None, "status": "unknown"}

def _thresholds_from_summary(summary: dict) -> list[float]:
    import re
    thresholds = []
    for col in summary:
        m = re.match(r"^p_over_([\d.]+)M$", col)
        if m:
            thresholds.append(float(m.group(1)))
    return sorted(thresholds)

THRESHOLDS = [2.30, 2.35, 2.40, 2.45, 2.50, 2.55, 2.60, 2.65, 2.70, 2.75, 2.80]  # fallback only
EDGE_MIN   = 0.03
KELLY_MIN_PCT = 0.05
BANKROLL   = float(os.environ.get("KALSHI_BANKROLL", "250"))

# ── Ensemble routing constants (yoy_delta dropped from production 2026-09-21
# — its holiday-contamination guard zeroes the trend-correction term for
# ~70% of all history, degrading it to a naive last-year lookup; see
# production_router.py docstring) ──────────────────────────────────────────
# Re-fit 2026-09-22 after fixing a day-of-week misalignment bug in
# recent_vol_vs_lag365_7d/14d, lag365_error_7d, and lag365_residual_anchor
# (build_features.py) and retraining TS3 on the corrected covariates — ts3's
# honest OOF error rose enough that it now gets zero weight in NORMAL and
# SHOULDER_POST. Kept in sync with daily_predict.py's ENSEMBLE_WEIGHTS.
# NORMAL is split on yoy_delta_trend_available (build_features.add_yoy_delta_
# feature). The router emits NORMAL_YOY_AVAILABLE / NORMAL_YOY_UNAVAILABLE as
# regime names, so both must be keys here or the day-detail endpoint returns
# no weights for those days. "NORMAL" kept as a legacy alias (= UNAVAILABLE)
# so older prediction_history rows tagged plain "NORMAL" still render.
# NORMAL_YOY_AVAILABLE re-fit 2026-09-30 after the yoy_delta rebuild (5-week
# drop-variant, holiday-safe base).
ENSEMBLE_WEIGHTS: dict[str, dict[str, float | None]] = {
    "NORMAL_YOY_UNAVAILABLE": {"tab": 0.600, "ts3": 0.000, "yoy_delta": 0.000, "anchor": 0.400},
    "NORMAL_YOY_AVAILABLE":   {"tab": 0.350, "ts3": 0.000, "yoy_delta": 0.450, "anchor": 0.200},
    "NORMAL":        {"tab": 0.600, "ts3": 0.000, "yoy_delta": 0.000, "anchor": 0.400},
    "SHOULDER_PRE":  {"tab": 0.333, "ts3": 0.333, "yoy_delta": 0.000, "anchor": 0.333},
    "SHOULDER_POST": {"tab": 0.500, "ts3": 0.000, "yoy_delta": 0.000, "anchor": 0.500},
    "PEAK_HOLIDAY":  {"tab": 1.000, "ts3": 0.000, "yoy_delta": 0.000, "anchor": 0.000},
    "STORM":         {"tab": None,  "ts3": None,   "yoy_delta": None,  "anchor": None},
}

# NORMAL weights can be refreshed dynamically without a code edit — see
# ensemble_experiment/refresh_normal_weights.py, which fits a 50/50 blend of
# full-history-to-date and last-30-day weights and writes this file. Mirrors
# the loader in autogluon_predict.py so the dashboard and the live predict
# pipeline never disagree.
_DYNAMIC_NORMAL_WEIGHTS_PATH = BASE / "ensemble_experiment" / "output" / "dynamic_normal_weights.json"


def _load_dynamic_normal_weights() -> dict | None:
    if not _DYNAMIC_NORMAL_WEIGHTS_PATH.exists():
        return None
    try:
        with open(_DYNAMIC_NORMAL_WEIGHTS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


_dyn = _load_dynamic_normal_weights()
if _dyn is not None:
    _order = _dyn["model_order"]
    ENSEMBLE_WEIGHTS["NORMAL"] = dict(zip(_order, _dyn["weights_tuple"]))

# Re-fit 2026-09-22 against the corrected anchor_master/TS3 OOF (see
# ENSEMBLE_WEIGHTS comment above). Kept in sync with daily_predict.py's
# REGIME_SIGMA.
PLATT_SIGMA: dict[str, dict[str, int]] = {
    "NORMAL":        {"sigma_raw": 67302,  "sigma_eff": 37058},
    "SHOULDER_PRE":  {"sigma_raw": 93078,  "sigma_eff": 52956},
    "SHOULDER_POST": {"sigma_raw": 82353,  "sigma_eff": 45824},
    "PEAK_HOLIDAY":  {"sigma_raw": 123198, "sigma_eff": 70017},
    "STORM":         {"sigma_raw": 85228,  "sigma_eff": 57780},
}


def _prediction_history() -> pd.DataFrame:
    """Loads prediction_history.csv with a `_row_order` column so callers can
    break run_date ties (date-only, no time — collides if the pipeline runs
    more than once on the same day) in favor of the most-recently-appended
    row instead of an arbitrary one from pandas' stable sort."""
    p = BASE / "output_autogluon_predict/prediction_history.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["_row_order"] = range(len(df))
    return df


def _kalshi_implied_avg(snap: pd.DataFrame) -> float | None:
    """Compute Kalshi's implied weekly avg from binary threshold market probabilities."""
    if snap.empty or "strike_millions" not in snap.columns:
        return None
    rows = snap[snap["strike_millions"].between(2.0, 3.0)].sort_values("strike_millions")
    if rows.empty:
        return None
    strikes = rows["strike_millions"].tolist()
    probs = [float(r.get("market_prob", 0) or 0) for _, r in rows.iterrows()]
    above = None
    below = None
    for i, p in enumerate(probs):
        if p >= 0.5:
            below = i
        if p < 0.5 and above is None:
            above = i
    if below is not None and above is not None and above > below:
        p_low = probs[below]
        p_high = probs[above]
        s_low = strikes[below]
        s_high = strikes[above]
        frac = (p_low - 0.5) / (p_low - p_high) if p_low != p_high else 0.5
        return round(s_low + frac * (s_high - s_low), 4)
    if probs and probs[0] < 0.5:
        return strikes[0] - 0.025
    if probs and probs[-1] >= 0.5:
        return strikes[-1] + 0.025
    return None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "api_version": "1.0", **_last_run()}


@app.post("/api/refresh-positions")
def refresh_positions():
    """Bust the portfolio cache and force a re-fetch from Kalshi."""
    _cache.pop("portfolio", None)
    portfolio = _portfolio()
    return {
        "ok":       True,
        "source":   portfolio.get("source", "unknown"),
        "n_positions":  len(portfolio.get("positions", [])),
        "n_open_orders": len(portfolio.get("open_orders", [])),
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/overview")
def overview():
    summary      = _weekly_summary()
    tsa          = _tsa()
    portfolio    = _portfolio()

    yesterday_actual = yesterday_predicted = yesterday_error = None
    if not tsa.empty:
        last = tsa.dropna(subset=["Volume"]).iloc[-1]
        yesterday_actual = float(last["Volume"])
        last_date        = str(last["Date"].date())

        # Look up prediction from history (before actual overwrote it)
        hist = _prediction_history()
        if not hist.empty and "Date" in hist.columns:
            h = hist[hist["Date"].apply(lambda x: str(pd.Timestamp(x).date())) == last_date].sort_values("run_date", ascending=False)
            if not h.empty:
                yesterday_predicted = float(h.iloc[0]["predicted_volume"])

        # Fallback: DB
        if yesterday_predicted is None:
            db_pred = db.query(
                """SELECT volume_forecast FROM daily_forecast_snapshots
                   WHERE forecast_date = %s AND status = 'predicted'
                   ORDER BY run_at DESC LIMIT 1""",
                (last_date,),
            )
            if db_pred:
                yesterday_predicted = float(db_pred[0]["volume_forecast"])

        if yesterday_predicted is not None:
            yesterday_error = yesterday_actual - yesterday_predicted

    today_avg = _safe(summary.get("weekly_avg_millions"))

    # Derive prev_avg from prediction_history run dates (prev_weekly_summary.csv is
    # overwritten by send_update.py immediately after the Telegram message is built,
    # making it identical to weekly_summary.csv for the rest of the day)
    prev_avg: float | None = None
    hist_for_change = _prediction_history()
    if not hist_for_change.empty and "weekly_avg_millions" in hist_for_change.columns:
        run_dates = sorted(hist_for_change["run_date"].unique())
        if len(run_dates) >= 2:
            prev_run  = run_dates[-2]
            prev_rows = hist_for_change[hist_for_change["run_date"] == prev_run]
            if not prev_rows.empty:
                prev_avg = float(prev_rows.iloc[0]["weekly_avg_millions"])

    # DB fallback
    if prev_avg is None:
        prev_avg = _safe(_prev_weekly_summary().get("weekly_avg_millions"))

    forecast_change = (today_avg - prev_avg) * 1e6 if (today_avg and prev_avg) else None

    positions = [p for p in portfolio.get("positions", []) if "KXTSAW" in p.get("ticker", "")]
    exposure  = sum(p.get("cost_dollars", 0) for p in positions)

    actions     = _actions()
    orders_today = 0
    if not actions.empty and "run_ts" in actions.columns:
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        ta = actions[actions["run_ts"].str.startswith(today_str)]
        if not ta.empty and "dry_run" in ta.columns:
            orders_today = int((ta["dry_run"].astype(str).str.lower() == "false").sum())

    auto_trading = os.environ.get("KALSHI_AUTO_TRADING", "true").lower() != "false"

    return {
        "weekly_avg_forecast":         _safe(summary.get("weekly_avg")),
        "weekly_avg_millions":         _safe(summary.get("weekly_avg_millions")),
        "weekly_avg_std_millions":     _safe(summary.get("weekly_avg_std_millions")),
        "n_actual":                    summary.get("n_actual"),
        "n_predicted":                 summary.get("n_predicted"),
        "yesterday_actual":            yesterday_actual,
        "yesterday_predicted":         yesterday_predicted,
        "yesterday_error":             yesterday_error,
        "forecast_change_from_yesterday": forecast_change,
        "bankroll":                    BANKROLL,
        "open_exposure":               round(exposure, 2),
        "exposure_pct":                round(exposure / BANKROLL * 100, 1) if BANKROLL else 0,
        "orders_today":                orders_today,
        "auto_trading":                auto_trading,
        "portfolio_source":            portfolio.get("source", "unknown"),
        **_last_run(),
    }


@app.get("/api/forecast/daily")
def forecast_daily(
    start: Optional[str] = Query(None),
    end:   Optional[str] = Query(None),
):
    tsa = _tsa()
    if tsa.empty:
        return []

    df = tsa.copy()
    if start:
        df = df[df["Date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["Date"] <= pd.Timestamp(end)]

    result = [
        {"date": str(r["Date"].date()), "actual": _safe(r["Volume"]), "predicted": None}
        for _, r in df.sort_values("Date").iterrows()
    ]
    result_map = {r["date"]: r for r in result}

    # Historical predictions from local prediction_history.csv
    hist = _prediction_history()
    if not hist.empty and "Date" in hist.columns:
        latest_per_date = hist.sort_values(["run_date", "_row_order"], ascending=False).drop_duplicates(subset=["Date"], keep="first")
        for _, row in latest_per_date.iterrows():
            d = str(pd.Timestamp(row["Date"]).date())
            if start and d < start:
                continue
            if end and d > end:
                continue
            if d in result_map:
                result_map[d]["predicted"] = float(row["predicted_volume"])
            else:
                entry = {"date": d, "actual": None, "predicted": float(row["predicted_volume"])}
                result.append(entry)
                result_map[d] = entry

    # Historical predictions from DB (one prediction per forecast_date: latest run's value)
    db_rows = db.query(
        """
        SELECT DISTINCT ON (forecast_date)
            forecast_date::text AS date,
            volume_forecast      AS predicted
        FROM daily_forecast_snapshots
        WHERE status = 'predicted'
          AND (%(start)s IS NULL OR forecast_date >= %(start)s::date)
          AND (%(end)s   IS NULL OR forecast_date <= %(end)s::date)
        ORDER BY forecast_date, run_at DESC
        """,
        {"start": start, "end": end},
    )
    for row in db_rows:
        d = str(row["date"])
        if d in result_map:
            if result_map[d]["predicted"] is None:
                result_map[d]["predicted"] = row["predicted"]
        else:
            entry = {"date": d, "actual": None, "predicted": row["predicted"]}
            result.append(entry)
            result_map[d] = entry

    # Also overlay current-week CSV predictions (most up-to-date)
    fc = _weekly_forecast()
    if not fc.empty:
        for _, r in fc.iterrows():
            if r["status"] == "predicted":
                d = str(pd.Timestamp(r["Date"]).date())
                if d in result_map:
                    result_map[d]["predicted"] = float(r["volume"])
                else:
                    result.append({"date": d, "actual": None, "predicted": float(r["volume"])})

    result.sort(key=lambda x: x["date"])
    return result


@app.get("/api/forecast/accuracy")
def forecast_accuracy(days: int = Query(90)):
    """Historical MAE per run — shows how model accuracy changed over time."""
    rows = db.query(
        """
        SELECT
            d.run_at::date                           AS run_date,
            round(avg(abs(a.volume - d.volume_forecast))::numeric, 0) AS mae,
            count(*)                                 AS n_days
        FROM daily_forecast_snapshots d
        JOIN tsa_actuals a ON a.date = d.forecast_date
        WHERE d.status = 'predicted'
          AND d.run_at >= now() - interval '1 day' * %s
        GROUP BY d.run_at::date
        ORDER BY d.run_at::date
        """,
        (days,),
    )
    return [{"date": str(r["run_date"]), "mae": float(r["mae"]) if r["mae"] else None, "n_days": r["n_days"]} for r in rows]


@app.get("/api/runs")
def recent_runs(limit: int = Query(30)):
    """Recent pipeline runs derived from DB timestamps."""
    rows = db.query(
        """
        SELECT
            run_at,
            weekly_avg_millions,
            n_actual,
            n_predicted
        FROM weekly_summary_snapshots
        ORDER BY run_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    return [
        {
            "run_at":              r["run_at"].isoformat() if r["run_at"] else None,
            "weekly_avg_millions": r["weekly_avg_millions"],
            "n_actual":            r["n_actual"],
            "n_predicted":         r["n_predicted"],
        }
        for r in rows
    ]


@app.get("/api/tomorrow")
def tomorrow_forecast():
    """
    Return per-model predictions + ensemble blend for the next unpredicted day.
    Individual model columns (pred_ts3, pred_yoy_delta, pred_anchor) are written
    by autogluon_predict.py when shadow models are available.
    """
    from datetime import date as date_cls
    df = _weekly_forecast()
    if df.empty:
        return {"has_data": False, "reason": "weekly_forecast.csv not found"}

    df["Date"] = pd.to_datetime(df["Date"])
    predicted = df[df["status"] == "predicted"].sort_values("Date")
    if predicted.empty:
        return {"has_data": False, "reason": "All days in current week are actual — no predicted days"}

    # Take the first predicted day (tomorrow or next future day)
    r = predicted.iloc[0]
    target_date = str(r["Date"].date())
    regime      = str(r["regime"]) if pd.notna(r.get("regime")) else "NORMAL"

    pred_tabular   = _safe(r.get("pred_tabular"))
    pred_ts3       = _safe(r.get("pred_ts3"))
    pred_yoy_delta = _safe(r.get("pred_yoy_delta"))
    pred_anchor    = _safe(r.get("pred_anchor"))
    ensemble       = _safe(r.get("volume"))

    weights  = ENSEMBLE_WEIGHTS.get(regime, {})
    sigma_d  = PLATT_SIGMA.get(regime, {})
    sigma_eff = sigma_d.get("sigma_eff")
    sigma_raw = sigma_d.get("sigma_raw")

    # Build the bell-curve data: discretise N(mu, sigma_eff) over ±4σ
    bell_points = []
    if ensemble is not None and sigma_eff:
        import math
        mu, sig = ensemble, sigma_eff
        n_points = 200
        x_min = mu - 4 * sig
        x_max = mu + 4 * sig
        step  = (x_max - x_min) / n_points
        for i in range(n_points + 1):
            x = x_min + i * step
            y = math.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
            bell_points.append({"x": round(x / 1e6, 5), "y": round(y * 1e6, 6)})

    # Pull daily Kalshi markets for the target date (live, 10-min cache).
    # On any failure or no matching event → fallback thresholds around μ.
    kalshi_thresholds, kalshi_source, kalshi_error = _daily_kalshi_for_date(target_date)

    from scipy.stats import norm as _norm
    thresholds: list[dict] = []

    def _model_p_over(t_m: float) -> float | None:
        if ensemble is None or not sigma_eff:
            return None
        z = (t_m * 1e6 - ensemble) / sigma_eff
        return round(float(1 - _norm.cdf(z)), 4)

    if kalshi_thresholds:
        # Real Kalshi markets — attach model + market + edge
        for m in kalshi_thresholds:
            t_m = m["threshold_millions"]
            p_o = _model_p_over(t_m)
            p_u = round(1 - p_o, 4) if p_o is not None else None
            mkt_p_over = m.get("market_prob_over")
            mkt_p_under = (
                round(1 - mkt_p_over, 4)
                if mkt_p_over is not None else None
            )
            edge_over = (
                round(p_o - mkt_p_over, 4)
                if (p_o is not None and mkt_p_over is not None) else None
            )
            edge_under = (
                round(p_u - mkt_p_under, 4)
                if (p_u is not None and mkt_p_under is not None) else None
            )
            thresholds.append({
                "threshold_millions": t_m,
                "ticker":             m.get("ticker"),
                "p_over":             p_o,
                "p_under":            p_u,
                "market_p_over":      mkt_p_over,
                "market_p_under":     mkt_p_under,
                "edge_over":          edge_over,
                "edge_under":         edge_under,
                "yes_bid_cents":      m.get("yes_bid_cents"),
                "yes_ask_cents":      m.get("yes_ask_cents"),
                "volume":             m.get("volume"),
                "open_interest":      m.get("open_interest"),
            })
    elif ensemble is not None and sigma_eff:
        # Fallback: μ ± {0.5, 1, 1.5}σ — no Kalshi, no edge
        for mult in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5):
            t_m = round((ensemble + mult * sigma_eff) / 1e6, 2)
            p_o = _model_p_over(t_m)
            p_u = round(1 - p_o, 4) if p_o is not None else None
            thresholds.append({
                "threshold_millions": t_m,
                "ticker":             None,
                "p_over":             p_o,
                "p_under":            p_u,
                "market_p_over":      None,
                "market_p_under":     None,
                "edge_over":          None,
                "edge_under":         None,
                "yes_bid_cents":      None,
                "yes_ask_cents":      None,
                "volume":             None,
                "open_interest":      None,
            })

    # Build daily portfolio data (positions, open orders, orderbook) for the
    # daily KXTRUFTSA markets of target_date. Only populated when Kalshi
    # has a live market for this date.
    daily_positions: list[dict] = []
    daily_open_orders: list[dict] = []
    daily_orderbook: list[dict] = []

    if kalshi_source == "kalshi" and kalshi_thresholds:
        thr_by_ticker = {t["ticker"]: t for t in kalshi_thresholds if t.get("ticker")}
        model_p_by_ticker = {
            row["ticker"]: row["p_over"]
            for row in thresholds
            if row.get("ticker") and row.get("p_over") is not None
        }
        portfolio = _portfolio()

        def _yes_mid_dollars(t: dict) -> float | None:
            yb, ya = t.get("yes_bid_cents"), t.get("yes_ask_cents")
            if yb is not None and ya is not None:
                return round((yb + ya) / 200.0, 4)
            return t.get("market_prob_over")

        for pos in portfolio.get("positions", []):
            ticker = pos.get("ticker", "")
            if ticker not in thr_by_ticker:
                continue
            t = thr_by_ticker[ticker]
            yes_mid = _yes_mid_dollars(t)
            no_mid  = round(1 - yes_mid, 4) if yes_mid is not None else None
            avg     = _safe(pos.get("avg_price"))
            cost    = float(pos.get("cost_dollars", 0))

            unrealized = None
            if yes_mid is not None and avg is not None:
                unrealized = 0.0
                if pos["yes"] > 0:
                    unrealized += (yes_mid - avg) * pos["yes"]
                if pos["no"] > 0 and no_mid is not None:
                    unrealized += (no_mid - avg) * pos["no"]
                unrealized = round(unrealized, 2)

            daily_positions.append({
                "ticker":            ticker,
                "strike_millions":   t["threshold_millions"],
                "yes_shares":        pos["yes"],
                "no_shares":         pos["no"],
                "avg_price":         avg,
                "cost_dollars":      cost,
                "yes_current_price": yes_mid,
                "no_current_price":  no_mid,
                "unrealized_pnl":    unrealized,
            })

        for o in portfolio.get("open_orders", []):
            ticker = o.get("ticker", "")
            if ticker not in thr_by_ticker:
                continue
            t = thr_by_ticker[ticker]
            yes_mid = _yes_mid_dollars(t)
            curr    = yes_mid if o["side"] == "yes" else (round(1 - yes_mid, 4) if yes_mid is not None else None)
            daily_open_orders.append({
                "ticker":          ticker,
                "strike_millions": t["threshold_millions"],
                "side":            o["side"],
                "price":           o["price"],
                "remaining":       o["remaining"],
                "current_price":   curr,
                "total_value":     round(o["price"] * o["remaining"], 2),
            })

        for t in kalshi_thresholds:
            ticker  = t.get("ticker")
            yes_bid = t.get("yes_bid_cents")
            yes_ask = t.get("yes_ask_cents")
            yes_mid_c = (
                (yes_bid + yes_ask) / 2.0
                if (yes_bid is not None and yes_ask is not None) else
                (yes_ask if yes_ask is not None else yes_bid)
            )
            mkt_p = t.get("market_prob_over")
            mod_p = model_p_by_ticker.get(ticker)
            daily_orderbook.append({
                "ticker":          ticker,
                "strike_millions": t["threshold_millions"],
                "yes_bid":         round(yes_bid / 100, 4) if yes_bid is not None else None,
                "yes_ask":         round(yes_ask / 100, 4) if yes_ask is not None else None,
                "no_bid":          None,
                "no_ask":          None,
                "yes_mid":         round(yes_mid_c / 100, 4) if yes_mid_c is not None else None,
                "market_prob":     mkt_p,
                "model_prob":      mod_p,
                "spread":          (
                    round((yes_ask - yes_bid) / 100, 4)
                    if (yes_ask is not None and yes_bid is not None) else None
                ),
                "volume":          t.get("volume"),
                "open_interest":   t.get("open_interest"),
            })

    return {
        "has_data":      True,
        "target_date":   target_date,
        "day_name":      str(r.get("day_name", "")),
        "regime":        regime,
        "pred_tabular":  pred_tabular,
        "pred_ts3":      pred_ts3,
        "pred_yoy_delta": pred_yoy_delta,
        "pred_anchor":   pred_anchor,
        "pred_ensemble": ensemble,
        "weights":       weights,
        "sigma_eff":     sigma_eff,
        "sigma_raw":     sigma_raw,
        "bell_curve":    bell_points,
        "thresholds":    thresholds,
        "thresholds_source": kalshi_source,  # "kalshi" | "fallback"
        "kalshi_error":  kalshi_error,
        "daily_positions":   daily_positions,
        "daily_open_orders": daily_open_orders,
        "daily_orderbook":   daily_orderbook,
    }


# ── Daily Kalshi market fetch (live, 10-min cache) ────────────────────────────
_MONTH_3 = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}


def _parse_event_ticker_date(event_ticker: str) -> str | None:
    """Parse KXTRUFTSA-26JUN04 → '2026-06-04'. Returns None on failure."""
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None
    seg = parts[1]
    if len(seg) != 7:
        return None
    try:
        yy = int(seg[0:2])
        mon = _MONTH_3.get(seg[2:5].upper())
        dd = int(seg[5:7])
        if mon is None:
            return None
        return f"20{yy:02d}-{mon:02d}-{dd:02d}"
    except ValueError:
        return None


def _daily_kalshi_for_date(target_date: str) -> tuple[list[dict], str, str | None]:
    """Return (thresholds, source, error).

    source ∈ {"kalshi", "fallback"}.
    Thresholds list is empty when source == "fallback".
    Cached for 10 minutes per target_date.
    """
    cache_key = f"daily_kalshi:{target_date}"
    cached = _cached(cache_key, 600, lambda: _daily_kalshi_fetch(target_date))
    return cached


def _daily_kalshi_fetch(target_date: str) -> tuple[list[dict], str, str | None]:
    try:
        from kalshi import fetch_tsa_daily_markets
        markets = fetch_tsa_daily_markets(debug=False)
    except Exception as exc:
        print(f"[warn] daily-kalshi fetch failed: {exc}", file=sys.stderr)
        return ([], "fallback", str(exc))

    if not markets:
        return ([], "fallback", "no_open_daily_markets")

    matched: list[dict] = []
    for m in markets:
        ev_date = _parse_event_ticker_date(m.event_ticker)
        if ev_date != target_date:
            continue
        yes_bid = m.yes_bid_cents
        yes_ask = m.yes_ask_cents
        if yes_bid is not None and yes_ask is not None:
            yes_mid = (yes_bid + yes_ask) / 200.0
        elif m.yes_mid_cents is not None:
            yes_mid = m.yes_mid_cents / 100.0
        elif m.market_mid_cents is not None:
            yes_mid = m.market_mid_cents / 100.0
        else:
            yes_mid = None
        matched.append({
            "ticker":             m.ticker,
            "threshold_millions": round(m.strike_millions, 4),
            "market_prob_over":   round(yes_mid, 4) if yes_mid is not None else None,
            "yes_bid_cents":      yes_bid,
            "yes_ask_cents":      yes_ask,
            "volume":             m.volume,
            "open_interest":      m.open_interest,
        })

    if not matched:
        return ([], "fallback", "no_market_for_target_date")

    matched.sort(key=lambda x: x["threshold_millions"])
    return (matched, "kalshi", None)


@app.get("/api/forecast/current-week")
def forecast_current_week():
    df      = _weekly_forecast()
    summary = _weekly_summary()
    snap    = _snapshot()
    if df.empty:
        return {"days": [], "summary": {}}

    # Build prediction lookup from history
    hist = _prediction_history()
    pred_lookup: dict[str, float] = {}
    if not hist.empty and "Date" in hist.columns:
        latest = hist.sort_values(["run_date", "_row_order"], ascending=False).drop_duplicates(subset=["Date"], keep="first")
        for _, h in latest.iterrows():
            pred_lookup[str(pd.Timestamp(h["Date"]).date())] = float(h["predicted_volume"])

    # DB fallback for predictions
    week_dates = [str(pd.Timestamp(r["Date"]).date()) for _, r in df.iterrows()]
    if week_dates:
        db_preds = db.query(
            """SELECT DISTINCT ON (forecast_date)
                   forecast_date::text AS date, volume_forecast
               FROM daily_forecast_snapshots
               WHERE status = 'predicted' AND forecast_date = ANY(%s::date[])
               ORDER BY forecast_date, run_at DESC""",
            (week_dates,),
        )
        for row in db_preds:
            d = str(row["date"])
            if d not in pred_lookup:
                pred_lookup[d] = float(row["volume_forecast"])

    days = []
    errors = []
    for _, r in df.iterrows():
        date_str = str(pd.Timestamp(r["Date"]).date())
        vol = _safe(r["volume"])
        predicted = pred_lookup.get(date_str)
        error = None
        if r.get("status") == "actual" and vol is not None and predicted is not None:
            error = round(vol - predicted)
            errors.append(abs(error))
        # Ensemble enrichment
        status = r.get("status", "")
        regime = r.get("regime") if status == "predicted" else None
        if isinstance(regime, float) and pd.isna(regime):
            regime = None
        pred_tabular = _safe(r.get("pred_tabular")) if status == "predicted" else None
        weights = ENSEMBLE_WEIGHTS.get(regime) if regime else None
        sigma_eff = PLATT_SIGMA.get(regime, {}).get("sigma_eff") if regime else None
        days.append({
            "date":         date_str,
            "day_name":     r.get("day_name", ""),
            "status":       status,
            "volume":       vol,
            "predicted":    predicted,
            "error":        error,
            "regime":       regime,
            "pred_tabular": pred_tabular,
            "weights":      weights,
            "sigma_eff":    sigma_eff,
        })

    mae = round(sum(errors) / len(errors)) if errors else None

    # Kalshi implied weekly avg
    kalshi_avg = _kalshi_implied_avg(snap)

    clean_summary = {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in summary.items()}

    return {
        "days":       days,
        "summary":    clean_summary,
        "mae":        mae,
        "kalshi_avg": kalshi_avg,
    }


def _bell_curve_points(mu: float, sigma: float, n_points: int = 200, extra_x: float | None = None) -> list[dict]:
    """extra_x: a value (e.g. the actual outcome) that must fall within the
    plotted range even if it's an outlier beyond ±4σ, so its reference line
    is never silently clipped off-chart."""
    import math
    x_min = mu - 4 * sigma
    x_max = mu + 4 * sigma
    if extra_x is not None:
        pad = 0.5 * sigma
        x_min = min(x_min, extra_x - pad)
        x_max = max(x_max, extra_x + pad)
    step  = (x_max - x_min) / n_points
    points = []
    for i in range(n_points + 1):
        x = x_min + i * step
        y = math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        points.append({"x": round(x / 1e6, 5), "y": round(y * 1e6, 6)})
    return points


@app.get("/api/forecast/day-detail")
def forecast_day_detail(weeks_back: int = Query(0, ge=0)):
    """Per-day breakdown for one Mon–Sun week: ensemble + per-model
    predictions, and (for days that have already occurred) the actual
    value, error, and a forecast-distribution bell curve — built for the
    'cycle through each day' explorer page.

    weeks_back=0 is the week containing today; weeks_back=1 is the prior
    week, etc. Predictions come from the latest run per date in
    prediction_history.csv, so this works for any week that history
    covers (not just the live weekly_forecast.csv week).

    Per-model columns (pred_ts3/pred_yoy_delta/pred_anchor) are only
    available for already-actual days going forward from when this was
    added — older history only has pred_tabular (prediction_history.csv
    didn't retain the others before)."""
    hist = _prediction_history()
    if hist.empty or "Date" not in hist.columns:
        return {"days": [], "week_start": None, "has_older_week": False}

    hist["_date"] = pd.to_datetime(hist["Date"], format="mixed").dt.normalize()
    latest = hist.sort_values(["run_date", "_row_order"], ascending=False).drop_duplicates(subset=["_date"], keep="first")
    hist_lookup: dict[str, dict] = {str(h["_date"].date()): h.to_dict() for _, h in latest.iterrows()}

    tsa = _tsa()
    actual_lookup: dict[str, float] = {}
    if not tsa.empty:
        for _, t in tsa.iterrows():
            actual_lookup[str(pd.Timestamp(t["Date"]).date())] = _safe(t["Volume"])

    today = pd.Timestamp.now().normalize()
    this_monday = today - pd.Timedelta(days=today.dayofweek)
    week_monday = this_monday - pd.Timedelta(weeks=weeks_back)
    week_dates = [week_monday + pd.Timedelta(days=i) for i in range(7)]

    oldest_hist_date = hist["_date"].min()
    has_older_week = (week_monday - pd.Timedelta(weeks=1)) >= (oldest_hist_date - pd.Timedelta(days=oldest_hist_date.dayofweek))

    MODEL_COLS = ["pred_tabular", "pred_ts3", "pred_yoy_delta", "pred_anchor"]

    days = []
    for wd in week_dates:
        date_str = str(wd.date())
        h = hist_lookup.get(date_str, {})
        actual_vol = actual_lookup.get(date_str)
        is_actual = actual_vol is not None

        regime = h.get("regime") or None
        if isinstance(regime, float) and pd.isna(regime):
            regime = None

        models: dict[str, float | None] = {col: _safe(h.get(col)) for col in MODEL_COLS}

        predicted_vol = _safe(h.get("predicted_volume"))
        error = round(actual_vol - predicted_vol) if (actual_vol is not None and predicted_vol is not None) else None

        weights  = ENSEMBLE_WEIGHTS.get(regime) if regime else None
        sigma_d  = PLATT_SIGMA.get(regime, {}) if regime else {}
        sigma_eff = sigma_d.get("sigma_eff")
        sigma_raw = sigma_d.get("sigma_raw")

        bell_curve = []
        if is_actual and predicted_vol is not None and sigma_eff:
            bell_curve = _bell_curve_points(predicted_vol, sigma_eff, extra_x=actual_vol)

        days.append({
            "date":          date_str,
            "day_name":      wd.strftime("%A"),
            "status":        "actual" if is_actual else "predicted",
            "regime":        regime,
            "ensemble":      actual_vol if is_actual else predicted_vol,
            "predicted":     predicted_vol,
            "actual":        actual_vol,
            "error":         error,
            "pred_tabular":  models["pred_tabular"],
            "pred_ts3":      models["pred_ts3"],
            "pred_yoy_delta": models["pred_yoy_delta"],
            "pred_anchor":   models["pred_anchor"],
            "weights":       weights,
            "sigma_eff":     sigma_eff,
            "sigma_raw":     sigma_raw,
            "bell_curve":    bell_curve,
        })

    return {"days": days, "week_start": str(week_monday.date()), "has_older_week": bool(has_older_week)}


def _week_monday_str(d: pd.Timestamp) -> str:
    return str((d - pd.Timedelta(days=d.weekday())).date())


@app.get("/api/forecast/weeks")
def forecast_weeks():
    """List available target weeks (Monday) for weekly-avg-tracker selection.

    Sourced from prediction_history.csv distinct target week-mondays, with the
    current week's monday (from weekly_forecast.csv) appended if missing.
    Each entry includes the settled actual average if all 7 days are confirmed.
    """
    weeks: dict[str, dict] = {}

    hist = _prediction_history()
    if not hist.empty and "Date" in hist.columns:
        dates = pd.to_datetime(hist["Date"], errors="coerce").dropna()
        for d in dates.unique():
            wm = _week_monday_str(pd.Timestamp(d))
            weeks.setdefault(wm, {"week_monday": wm})

    # Always include the current forecast week
    df = _weekly_forecast()
    if not df.empty:
        cur_wm = _week_monday_str(pd.Timestamp(df["Date"].iloc[0]))
        weeks.setdefault(cur_wm, {"week_monday": cur_wm})

    # Compute settled average from tsa_volume.csv where all 7 days exist
    tsa = _tsa()
    for wm, entry in weeks.items():
        wm_ts = pd.Timestamp(wm)
        end_ts = wm_ts + pd.Timedelta(days=6)
        entry["settled_avg_millions"] = None
        entry["is_current"] = (not df.empty and wm == _week_monday_str(pd.Timestamp(df["Date"].iloc[0])))
        if not tsa.empty:
            wk = tsa[(tsa["Date"] >= wm_ts) & (tsa["Date"] <= end_ts)]
            vols = wk["Volume"].dropna()
            if len(vols) == 7:
                entry["settled_avg_millions"] = round(float(vols.mean()) / 1e6, 4)

    return sorted(weeks.values(), key=lambda w: w["week_monday"], reverse=True)


@app.get("/api/forecast/weekly-avg-tracker")
def weekly_avg_tracker(week_monday: str | None = None):
    """Day-by-day evolution of model and Kalshi weekly avg predictions.

    Defaults to the current forecast week. Pass `week_monday=YYYY-MM-DD` to
    inspect a past week — past weeks include `settled_avg_millions` (the actual
    weekly average) when all 7 days are confirmed in tsa_volume.csv.
    """
    if week_monday is None:
        df = _weekly_forecast()
        if df.empty:
            return {"week_monday": None, "points": [], "settled_avg_millions": None}
        week_monday = _week_monday_str(pd.Timestamp(df["Date"].iloc[0]))

    week_end = str((pd.Timestamp(week_monday) + pd.Timedelta(days=6)).date())

    # Model: from prediction_history grouped by run_date, filtered to this target week
    hist = _prediction_history()
    model_points: list[dict] = []
    if not hist.empty and "Date" in hist.columns and "weekly_avg_millions" in hist.columns:
        week_hist = hist[(hist["Date"] >= week_monday) & (hist["Date"] <= week_end)]
        if not week_hist.empty:
            by_run = week_hist.groupby("run_date").first().reset_index()
            for _, row in by_run.sort_values("run_date").iterrows():
                model_points.append({
                    "run_date":            str(row["run_date"]),
                    "model_avg_millions":  round(float(row["weekly_avg_millions"]), 4),
                })

    # DB fallback (current week only — DB snapshots may not cover past weeks reliably)
    if not model_points:
        db_rows = db.query(
            """SELECT run_at::date AS run_date, weekly_avg_millions
               FROM weekly_summary_snapshots
               WHERE week_monday = %s
               ORDER BY run_at""",
            (week_monday,),
        )
        for r in db_rows:
            model_points.append({
                "run_date":           str(r["run_date"]),
                "model_avg_millions": round(float(r["weekly_avg_millions"]), 4) if r["weekly_avg_millions"] else None,
            })

    # Kalshi: from historical market snapshot CSVs (snapshots taken within the target week)
    snap_files = sorted(glob.glob(str(BASE / "output_kalshi/archive/market_snapshot_2*.csv")))
    kalshi_by_date: dict[str, float | None] = {}
    for f in snap_files:
        ts = Path(f).stem.replace("market_snapshot_", "")
        try:
            snap_date = datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d")
        except ValueError:
            continue
        if snap_date < week_monday or snap_date > week_end:
            continue
        try:
            snap_df = pd.read_csv(f)
            implied = _kalshi_implied_avg(snap_df)
            if implied is not None:
                kalshi_by_date[snap_date] = implied
        except Exception:
            continue

    # Settled average: if all 7 days exist in tsa_volume.csv, compute actual avg
    settled_avg = None
    tsa = _tsa()
    if not tsa.empty:
        wm_ts = pd.Timestamp(week_monday)
        end_ts = pd.Timestamp(week_end)
        wk = tsa[(tsa["Date"] >= wm_ts) & (tsa["Date"] <= end_ts)]
        vols = wk["Volume"].dropna()
        if len(vols) == 7:
            settled_avg = round(float(vols.mean()) / 1e6, 4)

    # Merge model + kalshi
    all_dates = sorted(set([p["run_date"] for p in model_points] + list(kalshi_by_date.keys())))
    points = []
    for d in all_dates:
        mp = next((p for p in model_points if p["run_date"] == d), None)
        points.append({
            "date":                d,
            "model_avg_millions":  mp["model_avg_millions"] if mp else None,
            "kalshi_avg_millions": kalshi_by_date.get(d),
        })
    return {
        "week_monday":          week_monday,
        "points":               points,
        "settled_avg_millions": settled_avg,
    }


def _kelly_would_trade(model_prob: float, ask_price: float | None, fee_rate: float = 0.07) -> bool:
    """Check if Kelly at 5% bankroll level would produce a viable limit price."""
    if model_prob <= 0 or model_prob >= 1:
        return False
    denom = 1.0 + fee_rate - KELLY_MIN_PCT
    if denom <= 0:
        return False
    limit_price = max(0.01, min(0.99, (model_prob - KELLY_MIN_PCT) / denom))
    if limit_price < 0.05 or limit_price > 0.95:
        return False
    if ask_price is not None and ask_price > limit_price:
        return False
    return True


@app.get("/api/probabilities")
def probabilities():
    summary = _weekly_summary()
    snap    = _snapshot()

    thresholds = _thresholds_from_summary(summary) or THRESHOLDS

    result = []
    for t in thresholds:
        model_over  = _safe(summary.get(f"p_over_{t}M"))
        model_under = _safe(summary.get(f"p_under_{t}M"))

        snap_row = snap[abs(snap["strike_millions"] - t) < 0.001] if (not snap.empty and "strike_millions" in snap.columns) else pd.DataFrame()

        kalshi_prob = _safe(snap_row.iloc[0]["market_prob"]) if not snap_row.empty else None
        yes_bid     = _safe(snap_row.iloc[0]["yes_bid_cents"], 100) if not snap_row.empty else None
        yes_ask     = _safe(snap_row.iloc[0]["yes_ask_cents"], 100) if not snap_row.empty else None
        yes_mid     = _safe(snap_row.iloc[0]["yes_mid_cents"], 100) if not snap_row.empty else None
        no_mid      = round(1 - yes_mid, 4) if yes_mid is not None else None

        over_edge  = round(model_over  - kalshi_prob,       4) if (model_over  and kalshi_prob is not None) else None
        under_edge = round(model_under - (1 - kalshi_prob), 4) if (model_under and kalshi_prob is not None) else None

        over_has_edge = over_edge is not None and over_edge >= EDGE_MIN
        under_has_edge = under_edge is not None and under_edge >= EDGE_MIN

        over_kelly_ok = model_over is not None and _kelly_would_trade(model_over, yes_ask)
        no_ask = _safe(snap_row.iloc[0]["no_ask_cents"], 100) if not snap_row.empty else None
        under_kelly_ok = model_under is not None and _kelly_would_trade(model_under, no_ask)

        result.append({
            "threshold_millions": t,
            "model_p_over":       model_over,
            "model_p_under":      model_under,
            "kalshi_prob":        kalshi_prob,
            "over_edge":          over_edge,
            "under_edge":         under_edge,
            "yes_bid":            yes_bid,
            "yes_ask":            yes_ask,
            "yes_mid":            yes_mid,
            "no_mid":             no_mid,
            "over_action":        "buy_yes" if (over_has_edge and over_kelly_ok) else "skip",
            "under_action":       "buy_no"  if (under_has_edge and under_kelly_ok) else "skip",
        })

    return result


@app.get("/api/positions")
def positions():
    portfolio = _portfolio()
    snap      = _snapshot()
    snap_map  = {} if snap.empty else {r["ticker"]: r for _, r in snap.iterrows()}

    result = []
    for pos in portfolio.get("positions", []):
        ticker = pos.get("ticker", "")
        if "KXTSAW" not in ticker:
            continue

        sr      = snap_map.get(ticker)
        yes_mid = _safe(sr["yes_mid_cents"], 100) if sr is not None else None
        no_mid  = round(1 - yes_mid, 4)          if yes_mid is not None else None
        avg     = _safe(pos.get("avg_price"))
        cost    = float(pos.get("cost_dollars", 0))

        unrealized = None
        if yes_mid is not None and avg is not None:
            unrealized = 0.0
            if pos["yes"] > 0:
                unrealized += (yes_mid - avg) * pos["yes"]
            if pos["no"] > 0 and no_mid is not None:
                unrealized += (no_mid - avg) * pos["no"]
            unrealized = round(unrealized, 2)

        result.append({
            "ticker":            ticker,
            "strike_millions":   _strike(ticker),
            "yes_shares":        pos["yes"],
            "no_shares":         pos["no"],
            "avg_price":         avg,
            "cost_dollars":      cost,
            "yes_current_price": yes_mid,
            "no_current_price":  no_mid,
            "unrealized_pnl":    unrealized,
        })

    return result


@app.get("/api/orders/open")
def open_orders():
    portfolio = _portfolio()
    snap      = _snapshot()
    snap_map  = {} if snap.empty else {r["ticker"]: r for _, r in snap.iterrows()}

    result = []
    for o in portfolio.get("open_orders", []):
        ticker = o.get("ticker", "")
        if "KXTSAW" not in ticker:
            continue

        sr      = snap_map.get(ticker)
        yes_mid = _safe(sr["yes_mid_cents"], 100) if sr is not None else None
        curr    = yes_mid if o["side"] == "yes" else (round(1 - yes_mid, 4) if yes_mid else None)

        result.append({
            "ticker":          ticker,
            "strike_millions": _strike(ticker),
            "side":            o["side"],
            "price":           o["price"],
            "remaining":       o["remaining"],
            "current_price":   curr,
            "total_value":     round(o["price"] * o["remaining"], 2),
        })

    return result


@app.get("/api/orders/history")
def orders_history(limit: int = Query(200)):
    df = _actions()
    if df.empty:
        return []

    result = []
    for _, row in df.sort_values("run_ts", ascending=False).head(limit).iterrows():
        run_ts = str(row.get("run_ts", ""))
        try:
            dt = datetime.strptime(run_ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            run_at = dt.isoformat()
        except ValueError:
            run_at = run_ts

        dry_run_val = str(row.get("dry_run", "True")).strip().lower()
        is_live     = dry_run_val == "false"
        price       = _safe(row.get("price"))
        shares      = int(row["shares"]) if pd.notna(row.get("shares")) else None

        result.append({
            "run_at":        run_at,
            "ticker":        row.get("ticker", ""),
            "strike_millions": _strike(str(row.get("ticker", ""))),
            "type":          row.get("type", ""),
            "side":          row.get("side", ""),
            "price":         price,
            "shares":        shares,
            "pct":           _safe(row.get("pct")),
            "model_prob":    _safe(row.get("model_prob")),
            "market_prob":   _safe(row.get("market_prob")),
            "dry_run":       not is_live,
            "cost":          round(price * shares, 2) if (price and shares) else None,
        })

    return result


@app.get("/api/orderbook")
def orderbook():
    snap = _snapshot()
    if snap.empty:
        return []

    thresholds = _thresholds_from_summary(_weekly_summary()) or THRESHOLDS
    relevant = snap[snap["strike_millions"].isin(thresholds)] if "strike_millions" in snap.columns else snap
    result = []
    for _, row in relevant.iterrows():
        result.append({
            "ticker":          row.get("ticker", ""),
            "strike_millions": _safe(row.get("strike_millions")),
            "yes_bid":         _safe(row.get("yes_bid_cents"),  100),
            "yes_ask":         _safe(row.get("yes_ask_cents"),  100),
            "no_bid":          _safe(row.get("no_bid_cents"),   100),
            "no_ask":          _safe(row.get("no_ask_cents"),   100),
            "yes_mid":         _safe(row.get("yes_mid_cents"),  100),
            "market_prob":     _safe(row.get("market_prob")),
            "model_prob":      _safe(row.get("model_prob")),
            "spread":          _safe(row.get("spread")),
            "volume":          _safe(row.get("volume")),
            "open_interest":   _safe(row.get("open_interest")),
        })
    return result


@app.get("/api/pnl")
def pnl():
    pos_data  = positions()
    portfolio = _portfolio()

    total_invested = sum(p.get("cost_dollars", 0) for p in portfolio.get("positions", []) if "KXTSAW" in p.get("ticker", ""))
    unrealized     = sum(p["unrealized_pnl"] for p in pos_data if p.get("unrealized_pnl") is not None)

    # Build P&L per position for the table
    by_position = [
        {
            "ticker":          p["ticker"],
            "strike_millions": p["strike_millions"],
            "side":            "Over" if p["yes_shares"] > 0 else "Under",
            "shares":          p["yes_shares"] if p["yes_shares"] > 0 else p["no_shares"],
            "avg_price":       p["avg_price"],
            "current_price":   p["yes_current_price"] if p["yes_shares"] > 0 else p["no_current_price"],
            "cost_dollars":    p["cost_dollars"],
            "unrealized_pnl":  p["unrealized_pnl"],
        }
        for p in pos_data
    ]

    return {
        "total_invested":  round(total_invested, 2),
        "current_value":   round(total_invested + unrealized, 2),
        "unrealized_pnl":  round(unrealized, 2),
        "realized_pnl":    None,
        "total_pnl":       round(unrealized, 2),
        "bankroll":        BANKROLL,
        "by_position":     by_position,
    }


@app.get("/api/bankroll")
def bankroll():
    portfolio  = _portfolio()
    positions_ = [p for p in portfolio.get("positions", []) if "KXTSAW" in p.get("ticker", "")]
    orders_    = [o for o in portfolio.get("open_orders", []) if "KXTSAW" in o.get("ticker", "")]

    invested = sum(p.get("cost_dollars", 0) for p in positions_)
    reserved = sum(o.get("price", 0) * o.get("remaining", 0) for o in orders_)
    cash     = max(BANKROLL - invested - reserved, 0)

    return {
        "bankroll":           BANKROLL,
        "cash":               round(cash, 2),
        "invested":           round(invested, 2),
        "reserved_for_orders": round(reserved, 2),
        "cash_pct":           round(cash     / BANKROLL * 100, 1),
        "invested_pct":       round(invested  / BANKROLL * 100, 1),
        "reserved_pct":       round(reserved  / BANKROLL * 100, 1),
    }


# ── New: changes / weather / drift ────────────────────────────

def _snapshot_dir_for_current_week() -> Path | None:
    """Find the current week's snapshot directory under output_autogluon_predict/snapshots/.

    Snapshot dirs are named `week_YYYY-MM-DD` using the week's Monday. We require
    an exact match — picking the last dir alphabetically would silently return
    a previous week's snapshots once a new week starts.
    """
    base = BASE / "output_autogluon_predict" / "snapshots"
    if not base.exists():
        return None
    today = pd.Timestamp.now().normalize()
    monday = today - pd.Timedelta(days=today.dayofweek)
    target = base / f"week_{monday.date().isoformat()}"
    return target if target.is_dir() else None


def _reconstruct_pair_from_history() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame] | None:
    """Reconstruct the (today, previous) snapshot pair from prediction_history.csv.

    Used when day-by-day snapshots for the current week are missing (the common
    case — daily snapshot generation isn't part of the pipeline yet). Returns
    summaries with only `weekly_avg_millions` populated; per-strike probability
    and confidence-sigma deltas are unavailable from this source and the changes
    endpoint already handles those being absent.
    """
    hist = _prediction_history()
    if hist.empty or "run_date" not in hist.columns or "weekly_avg_millions" not in hist.columns:
        return None
    run_dates = sorted(hist["run_date"].unique())
    if len(run_dates) < 2:
        return None

    today_run, prev_run = run_dates[-1], run_dates[-2]
    today_rows = hist[hist["run_date"] == today_run].copy()
    prev_rows  = hist[hist["run_date"] == prev_run].copy()
    if today_rows.empty or prev_rows.empty:
        return None

    today_rows["Date"] = pd.to_datetime(today_rows["Date"]).dt.normalize()
    prev_rows["Date"]  = pd.to_datetime(prev_rows["Date"]).dt.normalize()

    # weekly_avg_millions is refined autoregressively within a run; the last
    # row by Date holds the final value for that run.
    today_avg = float(today_rows.sort_values("Date").iloc[-1]["weekly_avg_millions"])
    prev_avg  = float(prev_rows.sort_values("Date").iloc[-1]["weekly_avg_millions"])

    today_summary = {"weekly_avg_millions": today_avg}
    prev_summary  = {"weekly_avg_millions": prev_avg}

    # Build per-day forecast frames anchored to the current week. For each day:
    #   - actual if the date is on or before the run's run_date and TSA has a value
    #   - else predicted (from prediction_history's row for that run_date)
    today_ts = pd.Timestamp.now().normalize()
    monday   = today_ts - pd.Timedelta(days=today_ts.dayofweek)
    sunday   = monday + pd.Timedelta(days=6)

    tsa = _tsa()
    tsa_map: dict[pd.Timestamp, float] = {}
    if not tsa.empty:
        for _, r in tsa.iterrows():
            if pd.notna(r.get("Volume")):
                tsa_map[pd.Timestamp(r["Date"]).normalize()] = float(r["Volume"])

    def _frame_for_run(rows: pd.DataFrame, run_date_str: str) -> pd.DataFrame:
        run_ts = pd.Timestamp(run_date_str).normalize()
        records: list[dict] = []
        current = monday
        while current <= sunday:
            day_name = current.strftime("%A")
            actual = tsa_map.get(current)
            # An "actual" relative to this run requires both: the TSA date is
            # strictly before the run_date AND we have a value for it. Same-day
            # predictions are still predictions until tomorrow's TSA scrape.
            if actual is not None and current < run_ts:
                records.append({"Date": current, "day_name": day_name, "status": "actual", "volume": actual})
            else:
                r = rows[rows["Date"] == current]
                if not r.empty:
                    records.append({
                        "Date": current,
                        "day_name": day_name,
                        "status": "predicted",
                        "volume": float(r.iloc[0]["predicted_volume"]),
                    })
            current += pd.Timedelta(days=1)
        return pd.DataFrame(records)

    return today_summary, prev_summary, _frame_for_run(today_rows, today_run), _frame_for_run(prev_rows, prev_run)


def _load_snapshot_pair() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame] | None:
    """Return (today_summary, prev_summary, today_forecast_df, prev_forecast_df).

    Resolution order:
      1. Day-by-day snapshots for the *current* week (when available).
      2. Reconstruction from prediction_history.csv (most reliable in practice).
      3. Top-level weekly_summary.csv + prev_weekly_summary.csv (often broken —
         send_update.py overwrites prev_* with today's value right after sending).
    """
    week_dir = _snapshot_dir_for_current_week()
    if week_dir:
        days = sorted(
            [p for p in week_dir.iterdir() if p.is_dir() and p.name.startswith("day_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        if len(days) >= 2:
            today_dir, prev_dir = days[-1], days[-2]
            try:
                today_summary = pd.read_csv(today_dir / "weekly_summary.csv").iloc[0].to_dict()
                prev_summary  = pd.read_csv(prev_dir  / "weekly_summary.csv").iloc[0].to_dict()
                today_fc      = pd.read_csv(today_dir / "weekly_forecast.csv")
                prev_fc       = pd.read_csv(prev_dir  / "weekly_forecast.csv")
                return today_summary, prev_summary, today_fc, prev_fc
            except Exception as exc:
                print(f"[warn] snapshot read: {exc}", file=sys.stderr)

    reconstructed = _reconstruct_pair_from_history()
    if reconstructed is not None:
        return reconstructed

    today_summary = _weekly_summary()
    prev_summary  = _prev_weekly_summary()
    today_fc = _weekly_forecast()
    prev_p = BASE / "output_autogluon_predict/prev_weekly_forecast.csv"
    prev_fc = pd.read_csv(prev_p) if prev_p.exists() else pd.DataFrame()
    if not today_summary or not prev_summary:
        return None
    return today_summary, prev_summary, today_fc, prev_fc


@app.get("/api/changes")
def changes():
    """What changed since the previous pipeline run.

    Returns per-day forecast revisions, weekly-avg delta, confidence (σ) delta,
    per-strike model+kalshi probability shifts, and a coarse attribution of
    the forecast revision into new-actual surprise vs forward revisions.
    """
    pair = _load_snapshot_pair()
    if not pair:
        return {"has_data": False, "reason": "Need at least two pipeline runs for the current week"}
    today_s, prev_s, today_fc, prev_fc = pair

    avg_today = _safe(today_s.get("weekly_avg_millions"))
    avg_prev  = _safe(prev_s.get("weekly_avg_millions"))
    forecast_delta_passengers = (avg_today - avg_prev) * 1e6 if (avg_today and avg_prev) else None
    forecast_delta_pct = (avg_today - avg_prev) / avg_prev if (avg_today and avg_prev and avg_prev > 0) else None

    std_today = _safe(today_s.get("weekly_avg_std_millions"))
    std_prev  = _safe(prev_s.get("weekly_avg_std_millions"))
    confidence_delta_std = (std_today - std_prev) if (std_today is not None and std_prev is not None) else None

    # Per-day deltas
    per_day_changes = []
    new_actual_contribution = 0.0
    forecast_revision_contribution = 0.0
    if not today_fc.empty and not prev_fc.empty:
        prev_map = {str(pd.Timestamp(r["Date"]).date()): r for _, r in prev_fc.iterrows()}
        for _, r in today_fc.iterrows():
            d = str(pd.Timestamp(r["Date"]).date())
            now_status = r.get("status", "")
            now_vol = _safe(r.get("volume"))
            pr = prev_map.get(d)
            if pr is None or now_vol is None:
                continue
            prev_status = pr.get("status", "")
            prev_vol = _safe(pr.get("volume"))
            if prev_vol is None:
                continue
            if prev_status == "predicted" and now_status == "actual":
                surprise = now_vol - prev_vol
                new_actual_contribution += surprise / 7.0  # weekly-avg effect
                per_day_changes.append({
                    "date": d,
                    "day_name": r.get("day_name", ""),
                    "kind": "new_actual",
                    "prev_predicted": prev_vol,
                    "now_actual": now_vol,
                    "surprise": round(surprise),
                })
            elif prev_status == "predicted" and now_status == "predicted":
                delta = now_vol - prev_vol
                forecast_revision_contribution += delta / 7.0
                per_day_changes.append({
                    "date": d,
                    "day_name": r.get("day_name", ""),
                    "kind": "forecast_revision",
                    "prev_predicted": prev_vol,
                    "now_predicted": now_vol,
                    "delta": round(delta),
                })

    # Probability shifts per strike
    prob_shifts = []
    for col in sorted(today_s):
        m = col.startswith("p_over_") and col.endswith("M")
        if not m:
            continue
        strike_str = col.replace("p_over_", "").replace("M", "")
        try:
            strike = float(strike_str)
        except ValueError:
            continue
        now_p = _safe(today_s.get(col))
        prev_p = _safe(prev_s.get(col))
        if now_p is None or prev_p is None:
            continue
        prob_shifts.append({
            "strike_millions": strike,
            "prev_model_p_over": prev_p,
            "now_model_p_over": now_p,
            "delta": round(now_p - prev_p, 4),
        })

    # Kalshi delta from snapshot history (use last 2 daily implied avgs)
    snaps = sorted(glob.glob(str(BASE / "output_kalshi/market_snapshot_2*.csv")))
    kalshi_delta = None
    by_day: dict[str, float] = {}
    for f in snaps[-30:]:
        ts = Path(f).stem.replace("market_snapshot_", "")
        try:
            day = datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d")
        except ValueError:
            continue
        try:
            df = pd.read_csv(f)
            implied = _kalshi_implied_avg(df)
            if implied is not None:
                by_day[day] = implied  # keep last seen value per day
        except Exception:
            continue
    days_sorted = sorted(by_day)
    if len(days_sorted) >= 2:
        kalshi_delta = by_day[days_sorted[-1]] - by_day[days_sorted[-2]]

    return {
        "has_data": True,
        "today_run_date": today_s.get("week_monday") and datetime.utcnow().date().isoformat(),
        "forecast_delta_passengers": round(forecast_delta_passengers) if forecast_delta_passengers is not None else None,
        "forecast_delta_pct": round(forecast_delta_pct, 5) if forecast_delta_pct is not None else None,
        "confidence_delta_std_millions": round(confidence_delta_std, 5) if confidence_delta_std is not None else None,
        "kalshi_delta_millions": round(kalshi_delta, 5) if kalshi_delta is not None else None,
        "per_day_changes": per_day_changes,
        "probability_shifts": prob_shifts,
        "attribution": {
            "new_actual_contribution_k": round(new_actual_contribution / 1000, 1),
            "forecast_revision_contribution_k": round(forecast_revision_contribution / 1000, 1),
            "kalshi_delta_pct": round(kalshi_delta * 100, 2) if kalshi_delta is not None else None,
        },
    }


def _weather_features() -> pd.DataFrame:
    p = BASE / "data/weather_national_features_with_lags.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


@app.get("/api/weather-impact")
def weather_impact():
    """Current-week weather load + trailing 4-week comparison."""
    df = _weather_features()
    if df.empty:
        return {"has_data": False}

    today = pd.Timestamp.now().normalize()
    week_monday = today - pd.Timedelta(days=today.dayofweek)
    week_sunday = week_monday + pd.Timedelta(days=6)
    trail_start = week_monday - pd.Timedelta(weeks=4)

    cur = df[(df["date"] >= week_monday) & (df["date"] <= week_sunday)].copy()
    trail = df[(df["date"] >= trail_start) & (df["date"] < week_monday)].copy()

    if cur.empty:
        return {"has_data": False, "reason": "No weather data for current week"}

    feat_cols = [
        "wt_avg_snow_depth_max",
        "wt_avg_snowfall_sum",
        "n_hubs_snowing",
        "top3_hubs_snowfall_mean",
        "vol_wtd_storm_impact",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]

    days = []
    for _, r in cur.iterrows():
        days.append({
            "date": str(r["date"].date()),
            **{c: _safe(r.get(c)) for c in feat_cols},
        })

    cur_mean   = {c: _safe(cur[c].mean())   if c in cur.columns   else None for c in feat_cols}
    trail_mean = {c: _safe(trail[c].mean()) if c in trail.columns else None for c in feat_cols}

    # Composite weather penalty: weighted sum of normalised storm + snowfall + snowing-hubs
    def _pen(metrics: dict) -> float | None:
        storm = metrics.get("vol_wtd_storm_impact") or 0.0
        snow  = metrics.get("wt_avg_snowfall_sum") or 0.0
        hubs  = metrics.get("n_hubs_snowing") or 0.0
        return round(storm * 1.0 + snow * 0.5 + hubs * 0.05, 4)

    cur_pen   = _pen(cur_mean)
    trail_pen = _pen(trail_mean)
    delta = None if (cur_pen is None or trail_pen is None) else round(cur_pen - trail_pen, 4)

    if cur_pen is None:
        label = "Unknown"
    elif cur_pen < 0.10:
        label = "Clear week — minimal weather drag"
    elif cur_pen < 0.30:
        label = "Mild weather — minor headwinds at a few hubs"
    elif cur_pen < 0.60:
        label = "Active weather — moderate drag on volume"
    else:
        label = "Severe weather — major demand suppression likely"

    # Approximate per-week volume impact in passengers. Empirical scale from
    # the model: roughly 6k weekly-avg passengers per unit of composite penalty.
    impact_k = None if cur_pen is None else round(-cur_pen * 6.0, 1)

    return {
        "has_data": True,
        "days": days,
        "current_week_mean": cur_mean,
        "trailing_4w_mean": trail_mean,
        "composite_penalty": cur_pen,
        "trailing_penalty": trail_pen,
        "penalty_delta": delta,
        "qualitative_label": label,
        "approx_volume_impact_k": impact_k,  # ± k passengers per day
    }


@app.get("/api/drift")
def drift():
    """Compare recent (14d) model accuracy to trailing (90d) baseline."""
    rows = db.query(
        """
        SELECT DISTINCT ON (d.forecast_date)
            d.forecast_date::text AS date,
            d.volume_forecast      AS predicted,
            a.volume               AS actual
        FROM daily_forecast_snapshots d
        JOIN tsa_actuals a ON a.date = d.forecast_date
        WHERE d.status = 'predicted'
          AND d.forecast_date >= (current_date - interval '120 days')
        ORDER BY d.forecast_date, d.run_at DESC
        """,
        (),
    )
    # Fallback: derive from local prediction_history + tsa_volume csv
    if not rows:
        hist = _prediction_history()
        tsa = _tsa()
        if hist.empty or tsa.empty:
            return {"has_data": False}
        tsa_map = {str(r["Date"].date()): float(r["Volume"]) for _, r in tsa.iterrows() if pd.notna(r["Volume"])}
        latest_pred = hist.sort_values(["run_date", "_row_order"], ascending=False).drop_duplicates(subset=["Date"], keep="first")
        rows = []
        for _, r in latest_pred.iterrows():
            d = str(pd.Timestamp(r["Date"]).date())
            if d in tsa_map:
                rows.append({"date": d, "predicted": float(r["predicted_volume"]), "actual": tsa_map[d]})

    if not rows:
        return {"has_data": False}

    today = pd.Timestamp.now().normalize()
    recent_cutoff   = today - pd.Timedelta(days=14)
    trailing_cutoff = today - pd.Timedelta(days=90)

    def _stats(window):
        if not window:
            return {"mae": None, "bias": None, "residual_std": None, "n": 0}
        errs = [r["actual"] - r["predicted"] for r in window]
        n = len(errs)
        mae  = sum(abs(e) for e in errs) / n
        bias = sum(errs) / n
        std  = (sum((e - bias) ** 2 for e in errs) / n) ** 0.5 if n > 1 else 0.0
        return {"mae": round(mae), "bias": round(bias), "residual_std": round(std), "n": n}

    recent_rows   = [r for r in rows if pd.Timestamp(r["date"]) >= recent_cutoff]
    trailing_rows = [r for r in rows if pd.Timestamp(r["date"]) >= trailing_cutoff and pd.Timestamp(r["date"]) < recent_cutoff]

    recent   = _stats(recent_rows)
    trailing = _stats(trailing_rows)

    mae_ratio = None
    bias_flip = None
    drift_flag = False
    messages = []
    if recent["mae"] is not None and trailing["mae"] is not None and trailing["mae"] > 0:
        mae_ratio = round(recent["mae"] / trailing["mae"], 3)
        if mae_ratio >= 1.40:
            drift_flag = True
            messages.append(f"Recent MAE is {int((mae_ratio-1)*100)}% worse than trailing 90d")
        elif mae_ratio <= 0.75:
            messages.append(f"Recent MAE is {int((1-mae_ratio)*100)}% better than trailing 90d")
    if recent["bias"] is not None and trailing["bias"] is not None:
        if recent["bias"] * trailing["bias"] < 0 and abs(recent["bias"]) > 5_000:
            bias_flip = True
            messages.append("Bias has flipped sign relative to trailing baseline")
        elif abs(recent["bias"]) > 30_000:
            drift_flag = True
            messages.append(f"Recent bias is large ({recent['bias']:+,}); model is consistently off-direction")

    return {
        "has_data": True,
        "recent_14d": recent,
        "trailing_90d": trailing,
        "mae_ratio": mae_ratio,
        "bias_flip": bias_flip,
        "drift_flag": drift_flag,
        "messages": messages or ["Model is stable vs trailing baseline"],
    }


FEATURE_LABELS: dict[str, tuple[str, str]] = {
    # feature_name → (human label, category)
    "lag365_residual_anchor":         ("Same week last year",          "history"),
    "vol_diff_1":                     ("Yesterday's day-over-day",     "momentum"),
    "anchor_accel_adjusted":          ("YoY momentum",                 "momentum"),
    "Volume_lag3":                    ("Volume 3 days ago",            "history"),
    "days_to_holiday_x_dow":          ("Holiday × day-of-week",        "calendar"),
    "dow_weekly_share":               ("Day-of-week share",            "calendar"),
    "dow_sin":                        ("Day-of-week cycle",            "calendar"),
    "nearest_holiday_expected_vol":   ("Holiday expected volume",      "calendar"),
    "holiday_asymmetry":              ("Pre/post-holiday asymmetry",   "calendar"),
    "week_vol_cumsum":                ("Week-to-date cumulative",      "trend"),
    "regime_gap_3d_7d":               ("3-day vs 7-day regime",        "trend"),
    "dayofweek":                      ("Day of week",                  "calendar"),
    "holiday_decay_shape":            ("Holiday decay curve",          "calendar"),
    "lag7_x_dow_share":               ("Same DOW last week × share",   "history"),
    "days_to_holiday":                ("Days to nearest holiday",      "calendar"),
    "week_progress_vs_4w_avg":        ("Week pace vs 4w avg",          "trend"),
    "Volume_lag7":                    ("Volume same day last week",    "history"),
    "holiday_expected_x_dow":         ("Holiday-expected × DOW",       "calendar"),
    "log_Volume_lag7":                ("Log-volume last week",         "history"),
    "month":                          ("Month of year",                "calendar"),
    "doy_cos":                        ("Day-of-year cycle",            "calendar"),
    "dow_x_month":                    ("DOW × month interaction",      "calendar"),
    "last_week_avg":                  ("Last week's average",          "history"),
    "pre_holiday_regime":             ("Pre-holiday regime",           "calendar"),
    "prior_7d_total":                 ("Prior 7-day total",            "history"),
    "week_accel_ratio":               ("Week acceleration ratio",      "trend"),
    "vol_same_dow_trimmed_8w":        ("Trimmed 8w same-DOW mean",     "history"),
    "holiday_expected_vs_normal":     ("Holiday vs normal expected",   "calendar"),
    "same_dow_acceleration":          ("Same-DOW acceleration",        "momentum"),
    "nearest_holiday_aligned_lag":    ("Holiday-aligned lag",          "calendar"),
}


def _label_feature(name: str) -> tuple[str, str]:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    return (name.replace("_", " ").title(), "other")


def _attribution_reason(current: float, baseline: float, contribution: float) -> str:
    push = "pushing forecast up" if contribution >= 0 else "pulling forecast down"

    # Features centered near zero (day-over-day diffs, accel ratios) make
    # percentage talk nonsensical. Detect sign flips and bucket magnitudes.
    if baseline == 0 or (current * baseline < 0 and abs(baseline) > 1):
        delta_desc = "opposite sign from typical"
    else:
        pct = abs(current - baseline) / max(abs(baseline), 1e-9) * 100
        direction = "above" if current >= baseline else "below"
        if pct < 5:
            delta_desc = "near typical level"
        elif pct < 25:
            delta_desc = f"slightly {direction} typical"
        elif pct < 75:
            delta_desc = f"clearly {direction} typical"
        else:
            delta_desc = f"well {direction} typical"

    return f"{delta_desc} · {push}"


@app.get("/api/feature-attributions")
def feature_attributions():
    """Real per-feature contributions to this week's weekly_avg forecast.

    Computed by autogluon_predict.py via leave-one-out ablation against the
    training-set median; results saved to
    output_autogluon_predict/feature_attributions.csv on each pipeline run.
    """
    csv_path = BASE / "output_autogluon_predict/feature_attributions.csv"
    if not csv_path.exists():
        return {
            "has_data": False,
            "reason": "Attributions not yet computed — wait for next pipeline run",
            "drivers": [],
        }

    df = pd.read_csv(csv_path)
    if df.empty:
        return {
            "has_data": False,
            "reason": "All days are actual — no features left to attribute",
            "drivers": [],
        }

    drivers = []
    for _, r in df.iterrows():
        feat = str(r["feature"])
        label, category = _label_feature(feat)
        contribution = int(r["contribution_passengers"])
        current = float(r["current_value"])
        baseline = float(r["baseline_value"])
        drivers.append({
            "feature": feat,
            "label": label,
            "category": category,
            "reason": _attribution_reason(current, baseline, contribution),
            "contribution_passengers": contribution,
            "contribution_millions": round(contribution / 1e6, 6),
            "current_value": current,
            "baseline_value": baseline,
            "importance_rank": int(r["importance_rank"]),
            "importance_passengers": int(r["importance_passengers"]),
        })

    summary = _weekly_summary()
    return {
        "has_data": True,
        "computed_at": datetime.fromtimestamp(csv_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "baseline_prediction_weekly_avg_millions": _safe(summary.get("weekly_avg_millions")),
        "method": "Leave-one-out ablation vs training-set median, summed across predicted days, divided by 7",
        "drivers": drivers,
    }


@app.get("/api/ev-journal")
def ev_journal():
    """Theoretical EV at entry vs. realized P&L at settlement, per fill."""
    try:
        from kalshi import load_ev_journal
        return load_ev_journal()
    except Exception as exc:
        print(f"[warn] ev_journal: {exc}", file=sys.stderr)
        return {"fills": [], "totals": {
            "theoretical_ev": 0.0, "realized_pnl": 0.0, "capture_ratio": None,
            "n_settled": 0, "n_pending": 0, "n_total": 0,
        }}


# ── Backtest endpoints ───────────────────────────────────────────────────────

def _backtest_dir() -> Path:
    return BASE / "output_backtest"


def _backtest_daily() -> pd.DataFrame:
    p = _backtest_dir() / "daily_snapshots.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _backtest_settle() -> pd.DataFrame:
    p = _backtest_dir() / "weekly_settlements.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _backtest_trades() -> pd.DataFrame:
    p = _backtest_dir() / "paper_trades.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _backtest_pnl_curve() -> pd.DataFrame:
    p = _backtest_dir() / "pnl_curve.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@app.get("/api/backtest/summary")
def backtest_summary():
    """Headline KPIs across the full backtest window."""
    daily = _backtest_daily()
    settle = _backtest_settle()
    trades = _backtest_trades()

    if daily.empty or settle.empty:
        return {"has_data": False, "reason": "Backtest hasn't been run yet (output_backtest/ empty)"}

    # Model accuracy across Monday-of-week predictions (the most-uncertain snapshot)
    daily["week_monday"] = pd.to_datetime(daily["week_monday"])
    settle["week_monday"] = pd.to_datetime(settle["week_monday"])
    monday_view = daily[daily["n_actual"] == 0].copy()  # full-week predictions
    merged = monday_view.merge(settle[["week_monday", "actual_weekly_avg_millions"]],
                                on="week_monday", how="inner")
    if not merged.empty:
        merged["err_millions"] = merged["weekly_avg_millions"] - merged["actual_weekly_avg_millions"]
        merged["abs_err_millions"] = merged["err_millions"].abs()
        model_mae_k    = float(merged["abs_err_millions"].mean() * 1000)
        model_bias_k   = float(merged["err_millions"].mean() * 1000)
        weeks_covered  = int(len(merged))
        within_50k_pct = float((merged["abs_err_millions"] < 0.05).sum() / len(merged) * 100)
    else:
        model_mae_k = model_bias_k = None
        weeks_covered = 0
        within_50k_pct = None

    # Trading P&L (only available if paper trades have been simulated)
    trade_count = won_count = settled_count = 0
    realized = capital = roi = None
    if not trades.empty:
        trade_count   = int(len(trades))
        settled_count = int(trades.get("settled", pd.Series(dtype=bool)).fillna(False).sum())
        won_count     = int((trades.get("outcome", pd.Series(dtype=str)) == "won").sum())
        settled_df = trades[trades["settled"].astype(bool) == True] if "settled" in trades.columns else pd.DataFrame()
        if not settled_df.empty:
            realized = float(settled_df["realized_pnl"].sum())
            capital  = float(settled_df["entry_cost"].sum())
            roi = (realized / capital) if capital > 0 else None

    return {
        "has_data":          True,
        "window_start":      daily["sim_date"].min(),
        "window_end":        daily["sim_date"].max(),
        "sim_days":          int(len(daily)),
        "weeks_settled":     int(len(settle)),
        "model_mae_k":       round(model_mae_k, 1) if model_mae_k is not None else None,
        "model_bias_k":      round(model_bias_k, 1) if model_bias_k is not None else None,
        "weeks_compared":    weeks_covered,
        "within_50k_pct":    round(within_50k_pct, 1) if within_50k_pct is not None else None,
        "trade_count":       trade_count,
        "trades_settled":    settled_count,
        "trades_won":        won_count,
        "hit_rate_pct":      round(won_count / settled_count * 100, 1) if settled_count else None,
        "realized_pnl":      round(realized, 2) if realized is not None else None,
        "capital_deployed":  round(capital, 2) if capital is not None else None,
        "roi_pct":           round(roi * 100, 1) if roi is not None else None,
    }


@app.get("/api/backtest/pnl-curve")
def backtest_pnl_curve():
    """Cumulative realized P&L expanded to one row per sim day. Trades settle
    on Sundays only, so the curve is flat between settlements and steps up on
    settlement days — gives a true day-by-day growth view."""
    pnl = _backtest_pnl_curve()
    daily = _backtest_daily()
    if pnl.empty or daily.empty:
        return []
    pnl = pnl.copy()
    pnl["settle_date"] = pd.to_datetime(pnl["settle_date"])
    daily = daily.copy()
    daily["sim_date"] = pd.to_datetime(daily["sim_date"])

    cal = pd.DataFrame({"date": sorted(daily["sim_date"].unique())})
    cal = cal.merge(
        pnl[["settle_date", "realized_pnl", "cumulative_pnl", "trade_count"]],
        left_on="date", right_on="settle_date", how="left",
    )
    cal["realized_pnl"]   = cal["realized_pnl"].fillna(0.0)
    cal["cumulative_pnl"] = cal["cumulative_pnl"].ffill().fillna(0.0)
    cal["trade_count"]    = cal["trade_count"].fillna(0).astype(int)
    return [
        {
            "date":           r["date"].date().isoformat(),
            "realized_pnl":   float(r["realized_pnl"]),
            "cumulative_pnl": float(r["cumulative_pnl"]),
            "trade_count":    int(r["trade_count"]),
        }
        for _, r in cal.iterrows()
    ]


@app.get("/api/backtest/per-week")
def backtest_per_week():
    """Per-settled-week: predicted (Monday-of-week) vs actual + trades placed."""
    daily  = _backtest_daily()
    settle = _backtest_settle()
    trades = _backtest_trades()
    if daily.empty or settle.empty:
        return []

    daily["week_monday"]  = pd.to_datetime(daily["week_monday"])
    settle["week_monday"] = pd.to_datetime(settle["week_monday"])
    monday_view = daily[daily["n_actual"] == 0].copy()

    if not trades.empty and "week_monday" in trades.columns:
        trades["week_monday"] = pd.to_datetime(trades["week_monday"])
        trade_agg = trades.groupby("week_monday").agg(
            trades_placed=("ticker", "count"),
            trades_won=("outcome", lambda s: int((s == "won").sum())),
            realized_pnl=("realized_pnl", "sum"),
        ).reset_index()
    else:
        trade_agg = pd.DataFrame(columns=["week_monday", "trades_placed", "trades_won", "realized_pnl"])

    out = []
    for _, s in settle.iterrows():
        wm = s["week_monday"]
        pred_row = monday_view[monday_view["week_monday"] == wm]
        ta = trade_agg[trade_agg["week_monday"] == wm] if not trade_agg.empty else pd.DataFrame()
        out.append({
            "week_monday":               wm.date().isoformat(),
            "actual_weekly_avg_millions": _safe(s.get("actual_weekly_avg_millions")),
            "predicted_weekly_avg_millions": _safe(pred_row.iloc[0]["weekly_avg_millions"]) if not pred_row.empty else None,
            "predicted_weekly_avg_std_millions": _safe(pred_row.iloc[0]["weekly_avg_std_millions"]) if not pred_row.empty else None,
            "err_millions":              (_safe(pred_row.iloc[0]["weekly_avg_millions"]) - _safe(s.get("actual_weekly_avg_millions"))) if not pred_row.empty else None,
            "trades_placed":             int(ta.iloc[0]["trades_placed"]) if not ta.empty else 0,
            "trades_won":                int(ta.iloc[0]["trades_won"])    if not ta.empty else 0,
            "realized_pnl":              round(float(ta.iloc[0]["realized_pnl"]), 2) if not ta.empty else 0.0,
        })
    return out


@app.get("/api/backtest/by-strike")
def backtest_by_strike():
    """Trade count + P&L per strike threshold."""
    trades = _backtest_trades()
    if trades.empty:
        return []
    grouped = trades.groupby(["strike_millions", "side"]).agg(
        trades=("ticker", "count"),
        settled=("settled", lambda s: int(s.astype(bool).sum())),
        won=("outcome", lambda s: int((s == "won").sum())),
        realized_pnl=("realized_pnl", "sum"),
        avg_edge=("edge", "mean"),
        capital=("entry_cost", "sum"),
    ).reset_index()
    return [
        {
            "strike_millions": float(r["strike_millions"]),
            "side":            str(r["side"]),
            "trades":          int(r["trades"]),
            "settled":         int(r["settled"]),
            "won":             int(r["won"]),
            "hit_rate_pct":    round(r["won"] / r["settled"] * 100, 1) if r["settled"] else None,
            "realized_pnl":    round(float(r["realized_pnl"]), 2),
            "avg_edge":        round(float(r["avg_edge"]), 4),
            "capital":         round(float(r["capital"]), 2),
        }
        for _, r in grouped.iterrows()
    ]


@app.get("/api/backtest/trades")
def backtest_trades_list(limit: int = Query(500)):
    """Recent simulated paper trades."""
    df = _backtest_trades()
    if df.empty:
        return []
    df = df.sort_values("entry_date", ascending=False).head(limit)
    return [
        {
            "entry_date":      str(r["entry_date"]),
            "week_monday":     str(r["week_monday"]),
            "strike_millions": float(r["strike_millions"]),
            "side":            str(r["side"]),
            "model_prob":      float(r["model_prob"]),
            "market_price":    float(r["market_price"]),
            "edge":            float(r["edge"]),
            "shares":          int(r["shares"]),
            "entry_cost":      float(r["entry_cost"]),
            "settled":         bool(r.get("settled", False)),
            "outcome":         str(r.get("outcome", "pending")),
            "realized_pnl":    float(r["realized_pnl"]) if pd.notna(r.get("realized_pnl")) else None,
            "actual_weekly_avg": float(r["actual_weekly_avg"]) if pd.notna(r.get("actual_weekly_avg")) else None,
        }
        for _, r in df.iterrows()
    ]


def _backtest_oof_dir() -> Path:
    return BASE / "output_backtest_oof"


def _read_csv_safe(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _spy_history() -> pd.DataFrame:
    p = BASE / "data" / "spy_history.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)


@app.get("/api/backtest/oof-comparison")
def backtest_oof_comparison(bankroll: float = Query(1000.0)):
    """OOF-window backtest comparison: raw P&L, bias-adjusted P&L, and a SPY
    buy-and-hold benchmark over the same window.

    Bankroll is the buy-and-hold amount used for the SPY benchmark — defaults
    to the live backtest_trades default ($1000).
    """
    out_dir = _backtest_oof_dir()
    raw_daily = _read_csv_safe(out_dir / "daily_snapshots_raw.csv")
    if raw_daily.empty:
        return {
            "has_data": False,
            "reason":   "OOF backtest hasn't been run yet (output_backtest_oof/ empty)",
        }

    # All bias-correction strategies the engine produced. Order = legend order.
    variants = ["raw", "bias", "bias_med", "bias_dow", "bias_lag365"]

    raw_daily["sim_date"] = pd.to_datetime(raw_daily["sim_date"])
    dates = sorted(raw_daily["sim_date"].unique())
    date_ts = [pd.Timestamp(d) for d in dates]

    def _cum_map(pnl_df: pd.DataFrame) -> dict[pd.Timestamp, float]:
        if pnl_df.empty:
            return {d: 0.0 for d in date_ts}
        p = pnl_df.copy()
        p["settle_date"] = pd.to_datetime(p["settle_date"])
        p = p.sort_values("settle_date")
        by_date = dict(zip(p["settle_date"], p["cumulative_pnl"].astype(float)))
        out: dict[pd.Timestamp, float] = {}
        running = 0.0
        for d in date_ts:
            if d in by_date:
                running = float(by_date[d])
            out[d] = running
        return out

    cums: dict[str, dict[pd.Timestamp, float]] = {}
    available: dict[str, bool] = {}
    for v in variants:
        pnl_df = _read_csv_safe(out_dir / f"pnl_curve_{v}.csv")
        cums[v] = _cum_map(pnl_df)
        available[v] = not pnl_df.empty

    # SPY buy-and-hold benchmark over the same window.
    spy = _spy_history()
    spy_cum: dict[pd.Timestamp, float | None] = {d: None for d in date_ts}
    spy_available = False
    if not spy.empty and date_ts:
        win_start, win_end = date_ts[0], date_ts[-1]
        spy_win = spy[(spy["date"] >= win_start) & (spy["date"] <= win_end)].copy()
        if not spy_win.empty:
            spy_available = True
            base_close = float(spy_win["close"].iloc[0])
            spy_map = dict(zip(spy_win["date"], spy_win["close"].astype(float)))
            last_close = base_close
            for d in date_ts:
                if d in spy_map:
                    last_close = spy_map[d]
                spy_cum[d] = float(bankroll * (last_close / base_close - 1.0))

    rows = []
    for d in date_ts:
        row = {"date": d.date().isoformat()}
        for v in variants:
            row[f"{v}_pnl"] = round(cums[v].get(d, 0.0), 2)
        row["spy_pnl"] = round(spy_cum[d], 2) if spy_cum.get(d) is not None else None
        rows.append(row)

    def _final(k: str) -> float | None:
        if not rows:
            return None
        v = rows[-1].get(k)
        return float(v) if v is not None else None

    finals = {f"{v}_pnl": _final(f"{v}_pnl") for v in variants}
    finals["spy_pnl"] = _final("spy_pnl")

    return {
        "has_data":       True,
        "bankroll":       float(bankroll),
        "window_start":   date_ts[0].date().isoformat() if date_ts else None,
        "window_end":     date_ts[-1].date().isoformat() if date_ts else None,
        "spy_available":  spy_available,
        "variant_available": available,
        "variants":       variants,
        "finals":         finals,
        "rows":           rows,
    }


@app.get("/api/yoy-comparison")
def yoy_comparison():
    """3-week window aligned across 2024, 2025, 2026 for year-over-year chart."""
    tsa = _tsa()
    if tsa.empty:
        return []

    today = pd.Timestamp.now().normalize()
    week_monday  = today - pd.Timedelta(days=today.dayofweek)
    week_sunday  = week_monday + pd.Timedelta(days=6)
    window_start = week_monday - pd.Timedelta(weeks=3)  # Monday 3 weeks ago

    tsa_map = {str(r["Date"].date()): float(r["Volume"]) for _, r in tsa.iterrows()}

    fc = _weekly_forecast()
    pred_map: dict[str, float] = {}
    for _, r in fc.iterrows():
        if r.get("status") == "predicted":
            pred_map[str(pd.Timestamp(r["Date"]).date())] = float(r["volume"])

    result = []
    current = window_start
    while current <= week_sunday:
        date_str  = str(current.date())
        # 52-week offset preserves exact day-of-week across years
        date_2025 = str((current - pd.Timedelta(weeks=52)).date())
        date_2024 = str((current - pd.Timedelta(weeks=104)).date())

        y2026_actual    = tsa_map.get(date_str)
        y2026_predicted = pred_map.get(date_str) if date_str not in tsa_map else None

        result.append({
            "date":             date_str,
            "y2024":            tsa_map.get(date_2024),
            "y2025":            tsa_map.get(date_2025),
            "y2026_actual":     y2026_actual,
            "y2026_predicted":  y2026_predicted,
        })
        current += pd.Timedelta(days=1)

    return result


# ── Ensemble router (reads from prediction_history.csv + weekly_forecast.csv) ─
def _ensemble_history_rows() -> list[dict]:
    """Build ensemble history rows from prediction_history.csv enriched with
    regime/pred_tabular from weekly_forecast.csv."""
    hist = _prediction_history()
    if hist.empty:
        return []

    fc = _weekly_forecast()
    # Build lookup: date → {regime, pred_tabular}
    fc_lookup: dict[str, dict] = {}
    if not fc.empty:
        for _, r in fc.iterrows():
            d = str(pd.Timestamp(r["Date"]).date())
            regime = r.get("regime")
            if isinstance(regime, float) and pd.isna(regime):
                regime = None
            fc_lookup[d] = {
                "regime": regime,
                "pred_tabular": _safe(r.get("pred_tabular")),
            }

    rows = []
    for _, r in hist.iterrows():
        date_str = str(pd.Timestamp(r["Date"]).date())
        regime_raw = r.get("regime")
        if pd.isna(regime_raw) if isinstance(regime_raw, float) else False:
            regime_raw = None
        regime = regime_raw or fc_lookup.get(date_str, {}).get("regime")
        pred_tabular_val = _safe(r.get("pred_tabular")) or fc_lookup.get(date_str, {}).get("pred_tabular")
        predicted_volume = _safe(r.get("predicted_volume"))
        weights = ENSEMBLE_WEIGHTS.get(regime) if regime else None
        sigma_eff = PLATT_SIGMA.get(regime, {}).get("sigma_eff") if regime else None
        run_date = r.get("run_date")
        rows.append({
            "target_date":       date_str,
            "made_on_date":      str(run_date) if run_date and not (isinstance(run_date, float) and pd.isna(run_date)) else None,
            "predicted_volume":  predicted_volume,
            "pred_tabular":      pred_tabular_val,
            "regime":            regime,
            "weights":           weights,
            "sigma_eff":         sigma_eff,
        })
    return rows


def _ts_trained_at() -> dict | None:
    """TS3 is the only remaining trained shadow model — yoy_delta is a
    deterministic feature (build_features.add_yoy_delta_feature), no training."""
    p = BASE / "output_router_shadow/models/ts_trained_at.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


@app.get("/api/shadow/predictions")
@app.get("/api/ensemble/history")
def ensemble_history():
    """Prediction history enriched with regime, ensemble weights and sigma_eff."""
    rows = _ensemble_history_rows()
    # Most recent 30, descending
    rows.sort(key=lambda x: x["target_date"], reverse=True)
    return rows[:30]


@app.get("/api/shadow/summary")
@app.get("/api/ensemble/summary")
def ensemble_summary():
    """Ensemble summary: n_predictions, regime distribution, date range."""
    rows = _ensemble_history_rows()
    if not rows:
        return {
            "n_predictions": 0,
            "regime_distribution": {},
            "models_last_trained": _ts_trained_at(),
            "first_target_date": None,
            "last_target_date": None,
        }
    regime_dist: dict[str, int] = {}
    dates = []
    for r in rows:
        dates.append(r["target_date"])
        reg = r.get("regime") or "UNKNOWN"
        regime_dist[reg] = regime_dist.get(reg, 0) + 1
    return {
        "n_predictions": len(rows),
        "regime_distribution": regime_dist,
        "models_last_trained": _ts_trained_at(),
        "first_target_date": min(dates) if dates else None,
        "last_target_date": max(dates) if dates else None,
    }


# ── Backlog / performance-tracking endpoints ─────────────────────────────────
# Thin wrappers over backlog.py (CLI module). Returns NaN-safe JSON.
import backlog as _backlog


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and (pd.isna(obj) or obj != obj):
        return None
    return obj


def _df_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return _nan_to_none(df.to_dict(orient="records"))


@app.get("/api/backlog")
def api_backlog(
    status: str = Query("all", description="open | resolved | all"),
    market: Optional[str] = Query(None, description="daily | weekly"),
    limit: int = Query(500),
):
    """Unified backlog rows: prediction × market × outcome × trade."""
    def _fetch():
        return _backlog.build_backlog(market)
    df = _cached(f"backlog:{market}", 60, _fetch)
    if df is None or df.empty:
        return {"rows": [], "n": 0}
    if status == "open":
        df = df[~df["resolved"]]
    elif status == "resolved":
        df = df[df["resolved"]]
    df = df.head(limit)
    return {"rows": _df_records(df), "n": int(len(df))}


@app.get("/api/backlog/perf")
def api_backlog_perf(market: Optional[str] = Query(None)):
    df = _cached(f"backlog:{market}", 60, lambda: _backlog.build_backlog(market))
    return _nan_to_none(_backlog.performance_summary(df) if df is not None and not df.empty
                        else {"overall": {"n": 0}, "by_market": {}, "by_weekday": {},
                              "by_edge_bucket": {}})


@app.get("/api/backlog/calibration")
def api_backlog_calibration(market: Optional[str] = Query(None), bins: int = Query(10)):
    df = _cached(f"backlog:{market}", 60, lambda: _backlog.build_backlog(market))
    if df is None or df.empty:
        return {"bins": []}
    return {"bins": _nan_to_none(_backlog.calibration_table(df, bins=bins))}


@app.get("/api/ensemble/weights")
def ensemble_weights():
    """Static ensemble weight table with Platt sigma values per regime."""
    REGIME_NOTES = {
        "NORMAL":        "dynamic — refreshed by refresh_normal_weights.py" if _dyn else "",
        "SHOULDER_PRE":  "",
        "SHOULDER_POST": "",
        "PEAK_HOLIDAY":  "",
        "STORM":         "α-blend with weather anchor",
    }
    result = []
    for regime in ["NORMAL", "SHOULDER_PRE", "SHOULDER_POST", "PEAK_HOLIDAY", "STORM"]:
        w = ENSEMBLE_WEIGHTS[regime]
        s = PLATT_SIGMA[regime]
        result.append({
            "regime":     regime,
            "tab":        w["tab"],
            "ts3":        w["ts3"],
            "yoy_delta":  w["yoy_delta"],
            "anchor":     w["anchor"],
            "sigma_raw":  s["sigma_raw"],
            "sigma_eff":  s["sigma_eff"],
            "note":       REGIME_NOTES.get(regime, ""),
        })
    return result


@app.get("/api/ensemble/dynamic-weights")
def ensemble_dynamic_weights():
    """Full breakdown behind the dynamic NORMAL weights: full-history fit,
    last-30-NORMAL-day fit, and the 50/50 blend actually in use — as written
    by ensemble_experiment/refresh_normal_weights.py."""
    dyn = _load_dynamic_normal_weights()
    if dyn is None:
        return {"has_data": False}

    order = dyn["model_order"]

    def as_dict(key: str) -> dict[str, float]:
        return dict(zip(order, dyn[key]))

    return {
        "has_data":              True,
        "regime":                dyn.get("regime", "NORMAL"),
        "model_order":           order,
        "full_history_weights":  as_dict("full_history_weights"),
        "last30_weights":        as_dict("last30_weights"),
        "blend_weights":         as_dict("weights_tuple"),
        "method":                dyn.get("method"),
        "generated_at":          dyn.get("generated_at"),
        "n_full_history":        dyn.get("n_full_history"),
        "n_last30":              dyn.get("n_last30"),
        "date_range":            dyn.get("date_range"),
        "quick_tabular_oof_mae": dyn.get("quick_tabular_oof_mae"),
    }
