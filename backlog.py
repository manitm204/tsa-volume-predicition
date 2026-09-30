#!/usr/bin/env python3
"""backlog.py — Unified prediction/market/outcome/trade backlog.

Read-only join layer over the existing storage:
    - Postgres market_snapshots   (per-run model_prob, market_prob, edge)
    - Postgres tsa_actuals        (per-date actual volume)
    - SQLite trades.db fills      (real Kalshi fills)
    - SQLite trades.db market_results (settlement)

No new schema. Always uses the *latest* run_at per (ticker, side) as the
canonical "frozen" pre-resolution prediction.

Subcommands:
    python backlog.py snapshot              # latest run, per-ticker view
    python backlog.py open [--last N]       # unresolved predictions
    python backlog.py resolved [--last N]   # resolved predictions w/ correctness
    python backlog.py perf                  # Brier, log-loss, win-rate, by bucket
    python backlog.py calibration [--bins K]
    python backlog.py export [--out PATH]   # CSV exports for dashboard / analysis

All commands support --daily / --weekly to filter market_type.
Set --dry-run on any command to short-circuit DB writes (there are none today,
but the flag is reserved so the pipeline contract stays consistent).
"""
from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

import db
import trades

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output_backlog"

logger = logging.getLogger("backlog")


# ── Resolution helpers ───────────────────────────────────────────────────────
def _resolve_daily(actual_volume: float | None, strike_m: float) -> int | None:
    """1 if YES (over) wins, 0 if NO (under) wins, None if unknown."""
    if actual_volume is None or pd.isna(actual_volume):
        return None
    return 1 if actual_volume / 1e6 > strike_m else 0


def _weekly_window(event_date: str) -> tuple[date, date]:
    """KXTSAW tickers expire on Sunday; the scored week is Mon..Sun."""
    d = datetime.strptime(event_date, "%Y-%m-%d").date()
    sunday = d
    monday = sunday - timedelta(days=6)
    return monday, sunday


def _weekly_actual(actuals: dict[date, int], event_date: str) -> tuple[float | None, int]:
    """Return (weekly mean volume, days available) for the Mon..Sun window."""
    monday, sunday = _weekly_window(event_date)
    vals = []
    cur = monday
    while cur <= sunday:
        v = actuals.get(cur)
        if v is not None:
            vals.append(v)
        cur += timedelta(days=1)
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


# ── Data loaders ─────────────────────────────────────────────────────────────
def _load_actuals() -> dict[date, int]:
    rows = db.query("SELECT date, volume FROM tsa_actuals", ())
    return {r["date"]: int(r["volume"]) for r in rows}


def _load_latest_snapshots(market_type: str | None) -> pd.DataFrame:
    """One row per ticker — the most recent run's market_snapshot row.

    Returns columns:
        run_at, ticker, strike_millions, yes_bid_cents, yes_ask_cents,
        yes_mid_cents, market_prob, model_prob, edge, volume, open_interest.
    """
    sql = """
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker ORDER BY run_at DESC
            ) AS rn
            FROM market_snapshots
        )
        SELECT * FROM ranked WHERE rn = 1
    """
    rows = db.query(sql, ())
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ticker"] = df["ticker"].astype(str)
    parsed = df["ticker"].apply(trades.parse_ticker)
    df["market_type"] = parsed.apply(lambda t: t[0])
    df["parsed_strike"] = parsed.apply(lambda t: t[1])
    df["event_date"] = parsed.apply(lambda t: t[2])
    # Prefer the snapshot's stored strike; fall back to ticker-parsed.
    df["strike_millions"] = df["strike_millions"].fillna(df["parsed_strike"])
    df = df.drop(columns=["parsed_strike", "rn"], errors="ignore")
    if market_type:
        df = df[df["market_type"] == market_type]
    return df.reset_index(drop=True)


def _load_trade_rows(market_type: str | None) -> list[dict]:
    """Aggregate Postgres `fills` + `settlements` into position-level rows.

    Same shape as trades.aggregate_positions(), but Postgres-backed so the
    backlog joins live in one store. Falls back to the SQLite path if
    Postgres has no fills yet (e.g., pre-migration).
    """
    if not db.available():
        return _load_trade_rows_sqlite(market_type)

    sql_where = ""
    params: list = []
    if market_type:
        sql_where = "WHERE market_type = %s"
        params = [market_type]

    rows = db.query(
        f"""
        SELECT
            f.ticker, f.side, f.market_type, f.strike_millions, f.event_date,
            SUM(CASE WHEN f.action='buy'  THEN f.count ELSE 0 END) AS buy_shares,
            SUM(CASE WHEN f.action='sell' THEN f.count ELSE 0 END) AS sell_shares,
            SUM(CASE WHEN f.action='buy' THEN
                CASE WHEN f.side='yes' THEN f.yes_price_cents * f.count
                     ELSE f.no_price_cents  * f.count END
                ELSE 0 END) AS buy_cost_cents,
            MAX(s.status)       AS market_status,
            MAX(s.result)       AS result,
            MAX(s.settled_time) AS settled_time
        FROM fills f
        LEFT JOIN settlements s USING (ticker)
        {sql_where}
        GROUP BY f.ticker, f.side, f.market_type, f.strike_millions, f.event_date
        """,
        tuple(params),
    )

    if not rows:
        return _load_trade_rows_sqlite(market_type)

    out: list[dict] = []
    for d in rows:
        buy  = int(d["buy_shares"]  or 0)
        sell = int(d["sell_shares"] or 0)
        if buy <= 0:
            continue  # skip phantom rows
        net = buy - sell
        avg_cents = (d["buy_cost_cents"] / buy) if buy > 0 else None
        cost_dollars = (d["buy_cost_cents"] / 100.0) if d["buy_cost_cents"] else 0.0
        result = d["result"] or None
        settled = result is not None
        realized = ((net - cost_dollars) if (result == d["side"]) else (-cost_dollars)) if settled else None

        model_p = trades.get_model_prob(
            d["ticker"], d["market_type"], d["side"],
            str(d["event_date"]) if d["event_date"] else None,
            float(d["strike_millions"]) if d["strike_millions"] is not None else None,
        )
        market_p = (avg_cents / 100.0) if avg_cents is not None else None
        edge = (model_p - market_p) if (model_p is not None and market_p is not None) else None

        out.append({
            "ticker":           d["ticker"],
            "side":             d["side"],
            "market_type":      d["market_type"],
            "strike_millions":  d["strike_millions"],
            "event_date":       str(d["event_date"]) if d["event_date"] else None,
            "net_shares":       net,
            "buy_shares":       buy,
            "avg_price_cents":  avg_cents,
            "buy_cost_dollars": cost_dollars,
            "settled":          settled,
            "result":           result,
            "realized_dollars": realized,
            "settled_time":     d["settled_time"],
            "market_status":    d["market_status"],
            "market_prob":      market_p,
            "model_prob":       model_p,
            "edge":             edge,
            "win":              (result == d["side"]) if settled else None,
        })
    return out


def _load_trade_rows_sqlite(market_type: str | None) -> list[dict]:
    """Legacy SQLite path — used only if Postgres fills are empty."""
    if not trades.DB_PATH.exists():
        logger.warning("trades.db not found at %s — trade columns will be blank",
                       trades.DB_PATH)
        return []
    conn = sqlite3.connect(trades.DB_PATH); conn.row_factory = sqlite3.Row
    try:
        return trades.aggregate_positions(conn, market_type)
    finally:
        conn.close()


# ── Backlog assembly ─────────────────────────────────────────────────────────
def build_backlog(market_type: str | None = None) -> pd.DataFrame:
    """Return one row per (ticker, side) joining prediction × market × outcome × trade."""
    snap = _load_latest_snapshots(market_type)
    if snap.empty:
        return pd.DataFrame()

    actuals = _load_actuals()

    # Snapshot is per ticker (YES side prob). Expand to one row per side so
    # we can attach trades + correctness on either side.
    rows: list[dict] = []
    for _, s in snap.iterrows():
        ticker = s["ticker"]
        mt = s.get("market_type")
        strike = s.get("strike_millions")
        event_date = s.get("event_date")
        if strike is None or event_date is None:
            continue
        p_over_model = s.get("model_prob")
        p_over_market = s.get("market_prob")

        if mt == "daily":
            actual_vol = actuals.get(datetime.strptime(event_date, "%Y-%m-%d").date())
            weekly_days = None
        elif mt == "weekly":
            actual_vol, weekly_days = _weekly_actual(actuals, event_date)
        else:
            continue

        yes_wins = _resolve_daily(actual_vol, strike) if actual_vol is not None else None

        for side in ("yes", "no"):
            if side == "yes":
                model_p = p_over_model
                market_p = p_over_market
                won = yes_wins
            else:
                model_p = (1.0 - p_over_model) if p_over_model is not None else None
                market_p = (1.0 - p_over_market) if p_over_market is not None else None
                won = (1 - yes_wins) if yes_wins is not None else None

            edge = None
            if model_p is not None and market_p is not None:
                edge = model_p - market_p

            rows.append({
                "ticker":          ticker,
                "event_ticker":    "-".join(ticker.split("-")[:-1]),
                "market_type":     mt,
                "event_date":      event_date,
                "weekday":         datetime.strptime(event_date, "%Y-%m-%d").strftime("%A"),
                "strike_millions": float(strike),
                "side":            side,
                "direction":       "over" if side == "yes" else "under",
                "model_prob":      model_p,
                "market_prob":     market_p,
                "edge":            edge,
                "yes_bid_cents":   s.get("yes_bid_cents"),
                "yes_ask_cents":   s.get("yes_ask_cents"),
                "yes_mid_cents":   s.get("yes_mid_cents"),
                "run_at":          s.get("run_at"),
                "actual_volume":   actual_vol,
                "actual_millions": (actual_vol / 1e6) if actual_vol else None,
                "weekly_days_known": weekly_days,
                "resolved":        won is not None and (
                    mt == "daily" or weekly_days == 7
                ),
                "won":             won,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Attach trade columns.
    trade_rows = _load_trade_rows(market_type)
    trade_idx: dict[tuple[str, str], dict] = {(t["ticker"], t["side"]): t for t in trade_rows}
    df["trade_shares"]     = df.apply(lambda r: trade_idx.get((r["ticker"], r["side"]), {}).get("net_shares"), axis=1)
    df["trade_buy_shares"] = df.apply(lambda r: trade_idx.get((r["ticker"], r["side"]), {}).get("buy_shares"), axis=1)
    df["trade_avg_cents"]  = df.apply(lambda r: trade_idx.get((r["ticker"], r["side"]), {}).get("avg_price_cents"), axis=1)
    df["trade_cost_dollars"] = df.apply(lambda r: trade_idx.get((r["ticker"], r["side"]), {}).get("buy_cost_dollars"), axis=1)
    df["trade_realized_dollars"] = df.apply(lambda r: trade_idx.get((r["ticker"], r["side"]), {}).get("realized_dollars"), axis=1)
    df["trade_model_prob"] = df.apply(lambda r: trade_idx.get((r["ticker"], r["side"]), {}).get("model_prob"), axis=1)
    df["trade_edge"]       = df.apply(lambda r: trade_idx.get((r["ticker"], r["side"]), {}).get("edge"), axis=1)
    df["has_trade"]        = df["trade_buy_shares"].apply(lambda v: bool(v) and v > 0)

    df = df.sort_values(["event_date", "ticker", "side"], ascending=[False, True, True])
    return df.reset_index(drop=True)


# ── Analytics ────────────────────────────────────────────────────────────────
def _safe_log(p: float) -> float:
    eps = 1e-9
    return math.log(min(max(p, eps), 1 - eps))


def _yes_only(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical scoring view: one YES-side prediction per ticker."""
    return df[df["side"] == "yes"].copy()


def performance_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Brier + log loss + win-rate, overall and by slice."""
    base = _yes_only(df)
    resolved = base[base["resolved"] & base["model_prob"].notna() & base["won"].notna()]

    def _scores(sub: pd.DataFrame) -> dict[str, Any]:
        if sub.empty:
            return {"n": 0}
        actual = sub["won"].astype(float)
        model_p = sub["model_prob"].astype(float)
        market_p = sub["market_prob"].astype(float)
        return {
            "n":             int(len(sub)),
            "model_brier":   float(((model_p - actual) ** 2).mean()),
            "model_logloss": float(-(actual * model_p.apply(_safe_log) +
                                     (1 - actual) * (1 - model_p).apply(_safe_log)).mean()),
            "market_brier":  None if market_p.isna().any() else float(((market_p - actual) ** 2).mean()),
            "yes_hit_rate":  float(actual.mean()),
            "model_avg_p":   float(model_p.mean()),
            "market_avg_p":  None if market_p.isna().any() else float(market_p.mean()),
        }

    out = {
        "overall":      _scores(resolved),
        "by_market":    {k: _scores(g) for k, g in resolved.groupby("market_type")},
        "by_weekday":   {k: _scores(g) for k, g in resolved.groupby("weekday")},
    }

    # Edge-bucketed realized return (only over rows that were traded).
    traded = df[df["has_trade"] & df["resolved"] & df["trade_edge"].notna() &
                df["trade_realized_dollars"].notna()]
    edge_buckets: dict[str, dict[str, float]] = {}
    if not traded.empty:
        traded = traded.copy()
        traded["edge_bucket"] = (traded["trade_edge"] * 100 // 5 * 5).astype(int)
        for b, g in traded.groupby("edge_bucket"):
            edge_buckets[f"{int(b):+d}%–{int(b)+5:+d}%"] = {
                "n":            int(len(g)),
                "wins":         int((g["trade_realized_dollars"] > 0).sum()),
                "realized_pnl": float(g["trade_realized_dollars"].sum()),
                "avg_return":   float(
                    (g["trade_realized_dollars"] / g["trade_cost_dollars"].replace(0, pd.NA)).mean(skipna=True)
                ),
            }
    out["by_edge_bucket"] = edge_buckets

    return out


def calibration_table(df: pd.DataFrame, bins: int = 10) -> list[dict[str, Any]]:
    """Bucket model_prob in equal-width bins, return predicted vs empirical hit rate."""
    base = _yes_only(df)
    resolved = base[base["resolved"] & base["model_prob"].notna() & base["won"].notna()]
    if resolved.empty:
        return []
    out = []
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (resolved["model_prob"] >= lo) & (resolved["model_prob"] < hi if i < bins - 1
                                                  else resolved["model_prob"] <= hi)
        sub = resolved[mask]
        if sub.empty:
            out.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                        "model_mean": None, "empirical_hit_rate": None})
            continue
        out.append({
            "bin_lo":             lo,
            "bin_hi":             hi,
            "n":                  int(len(sub)),
            "model_mean":         float(sub["model_prob"].mean()),
            "empirical_hit_rate": float(sub["won"].mean()),
        })
    return out


# ── Rendering ────────────────────────────────────────────────────────────────
def _fmt(v, spec="") -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def _render_backlog(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        print(f"\n=== {title}: (none) ===")
        return
    print(f"\n=== {title} ({len(df)} rows) ===")
    cols = [
        ("event_date",   12, "<"),
        ("market_type",  7, "<"),
        ("ticker",      36, "<"),
        ("side",         4, "<"),
        ("strike",       7, ">"),
        ("model%",       7, ">"),
        ("mkt%",         7, ">"),
        ("edge%",        7, ">"),
        ("act_M",        7, ">"),
        ("won",          3, "<"),
        ("shares",       6, ">"),
        ("pnl$",         9, ">"),
    ]
    header = "  ".join(f"{n:{a}{w}}" for n, w, a in cols)
    print(header); print("-" * len(header))
    for _, r in df.iterrows():
        cells = (
            f"{r['event_date']:<12}",
            f"{r['market_type']:<7}",
            f"{r['ticker']:<36}",
            f"{r['side']:<4}",
            f"{_fmt(r['strike_millions'], '.2f'):>7}",
            f"{_fmt(r['model_prob']*100 if r['model_prob'] is not None else None, '.1f'):>7}",
            f"{_fmt(r['market_prob']*100 if r['market_prob'] is not None else None, '.1f'):>7}",
            f"{_fmt(r['edge']*100 if r['edge'] is not None else None, '+.1f'):>7}",
            f"{_fmt(r['actual_millions'], '.3f') if r['resolved'] else '--':>7}",
            f"{(_fmt(int(r['won'])) if r['resolved'] and r['won'] is not None and not pd.isna(r['won']) else '--'):<3}",
            f"{int(r['trade_buy_shares']) if pd.notna(r['trade_buy_shares']) and r['trade_buy_shares'] else 0:>6d}",
            f"{_fmt(r['trade_realized_dollars'], '+.2f'):>9}",
        )
        print("  ".join(cells))


def _render_perf(perf: dict[str, Any]) -> None:
    print("\n=== PERFORMANCE — overall (YES-side scoring) ===")
    o = perf["overall"]
    if not o["n"]:
        print("(no resolved predictions yet)")
        return
    print(f"  n resolved:        {o['n']}")
    print(f"  model Brier:       {o['model_brier']:.4f}")
    print(f"  model log loss:    {o['model_logloss']:.4f}")
    if o["market_brier"] is not None:
        print(f"  market Brier:      {o['market_brier']:.4f}")
    print(f"  YES hit rate:      {o['yes_hit_rate']*100:.1f}%")
    print(f"  model avg p(yes):  {o['model_avg_p']*100:.1f}%")

    for label, table in (("by market type", perf["by_market"]),
                         ("by weekday",     perf["by_weekday"])):
        print(f"\n=== PERFORMANCE — {label} ===")
        print(f"  {'slice':<12} {'n':>4} {'brier':>8} {'logloss':>9} {'hit%':>7}")
        for k, s in sorted(table.items()):
            if not s.get("n"):
                continue
            print(f"  {k:<12} {s['n']:>4d} {s['model_brier']:>8.4f} "
                  f"{s['model_logloss']:>9.4f} {s['yes_hit_rate']*100:>6.1f}%")

    if perf["by_edge_bucket"]:
        print("\n=== TRADE P&L — by edge bucket at trade time ===")
        print(f"  {'bucket':<14} {'n':>4} {'wins':>5} {'pnl$':>10} {'avg ret':>9}")
        for k, s in sorted(perf["by_edge_bucket"].items(),
                           key=lambda kv: int(kv[0].split("%")[0])):
            print(f"  {k:<14} {s['n']:>4d} {s['wins']:>5d} "
                  f"{s['realized_pnl']:>+10.2f} {_fmt(s['avg_return'], '+.2%'):>9}")


def _render_calibration(table: list[dict[str, Any]]) -> None:
    print("\n=== CALIBRATION — YES-side prediction buckets ===")
    if not table:
        print("(no resolved predictions yet)")
        return
    print(f"  {'bin':<14} {'n':>4} {'model_mean':>11} {'empirical':>10}  diff")
    for b in table:
        if not b["n"]:
            print(f"  {b['bin_lo']*100:>4.0f}%–{b['bin_hi']*100:>4.0f}%   "
                  f"{'0':>4} {'--':>11} {'--':>10}  --")
            continue
        diff = b["empirical_hit_rate"] - b["model_mean"]
        print(f"  {b['bin_lo']*100:>4.0f}%–{b['bin_hi']*100:>4.0f}%   "
              f"{b['n']:>4d} {b['model_mean']*100:>10.1f}% "
              f"{b['empirical_hit_rate']*100:>9.1f}%  {diff*100:+5.1f}%")


# ── CSV export ───────────────────────────────────────────────────────────────
def export_csvs(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    p = out_dir / "backlog.csv"; df.to_csv(p, index=False); written["backlog"] = p

    p = out_dir / "open.csv"
    df[~df["resolved"]].to_csv(p, index=False); written["open"] = p

    p = out_dir / "resolved.csv"
    df[df["resolved"]].to_csv(p, index=False); written["resolved"] = p

    perf = performance_summary(df)
    perf_rows: list[dict] = []
    for slice_name, table in (("overall", {"overall": perf["overall"]}),
                              ("by_market", perf["by_market"]),
                              ("by_weekday", perf["by_weekday"])):
        for k, s in table.items():
            if not s.get("n"):
                continue
            perf_rows.append({"slice": slice_name, "key": k, **s})
    p = out_dir / "perf_summary.csv"
    pd.DataFrame(perf_rows).to_csv(p, index=False); written["perf_summary"] = p

    cal = calibration_table(df, bins=10)
    p = out_dir / "calibration.csv"
    pd.DataFrame(cal).to_csv(p, index=False); written["calibration"] = p
    return written


# ── CLI ──────────────────────────────────────────────────────────────────────
def _market_filter(args) -> str | None:
    if getattr(args, "daily", False):  return "daily"
    if getattr(args, "weekly", False): return "weekly"
    return None


def _load_or_fail(args) -> pd.DataFrame:
    if not db.available():
        print("[backlog] POSTGRES_URL not set — cannot build backlog.", file=sys.stderr)
        sys.exit(1)
    df = build_backlog(_market_filter(args))
    if df.empty:
        print("[backlog] No market_snapshots in Postgres yet.", file=sys.stderr)
    return df


def cmd_snapshot(args) -> None:
    df = _load_or_fail(args)
    if df.empty: return
    df = df[df["run_at"] == df["run_at"].max()]
    _render_backlog(df, "LATEST RUN — backlog snapshot")


def cmd_open(args) -> None:
    df = _load_or_fail(args)
    if df.empty: return
    df = df[~df["resolved"]]
    if args.last: df = df.head(args.last)
    _render_backlog(df, "OPEN predictions")


def cmd_resolved(args) -> None:
    df = _load_or_fail(args)
    if df.empty: return
    df = df[df["resolved"]]
    if args.last: df = df.head(args.last)
    _render_backlog(df, "RESOLVED predictions")


def cmd_perf(args) -> None:
    df = _load_or_fail(args)
    if df.empty: return
    _render_perf(performance_summary(df))


def cmd_calibration(args) -> None:
    df = _load_or_fail(args)
    if df.empty: return
    _render_calibration(calibration_table(df, bins=args.bins))


# ── Jarvis-friendly Python API ───────────────────────────────────────────────
# These return plain dicts/lists keyed on simple types so callers (jarvis.py,
# notebooks, the dashboard) can `import backlog` and consume directly.
# All functions are read-only and resilient to missing data.

def _latest_prediction(forecast_date, market_type: str = "daily") -> dict | None:
    rows = db.query(
        """
        SELECT * FROM predictions
        WHERE forecast_date = %s AND market_type = %s
        ORDER BY run_at DESC LIMIT 1
        """,
        (forecast_date, market_type),
    )
    return rows[0] if rows else None


def _actual_for(forecast_date) -> int | None:
    rows = db.query("SELECT volume FROM tsa_actuals WHERE date = %s", (forecast_date,))
    return int(rows[0]["volume"]) if rows else None


def yesterday_brief(today: date | None = None) -> dict:
    """Compact summary of the prior day: prediction, actual, error, trade P&L.

    Returns:
        {
            "date":              YYYY-MM-DD string,
            "day_name":          "Tuesday",
            "regime":            "NORMAL" | ...,
            "predicted_volume":  float (passengers),
            "sigma":             float | None,
            "actual_volume":     float | None,
            "error":             float | None     (predicted − actual, passengers),
            "abs_error_k":       float | None     (|error| in thousands),
            "z_score":           float | None,
            "n_trades":          int,
            "realized_pnl":      float            (dollars, daily fills for that event_date),
            "trades":            [
                {ticker, side, shares, avg_cents, edge_pct, won, pnl_dollars}
            ],
        }
    """
    today = today or date.today()
    target = today - timedelta(days=1)

    pred = _latest_prediction(target, "daily")
    actual_vol = _actual_for(target)

    out: dict = {
        "date":             target.isoformat(),
        "day_name":         target.strftime("%A"),
        "regime":           pred.get("regime") if pred else None,
        "predicted_volume": float(pred["predicted_volume"]) if pred and pred.get("predicted_volume") is not None else None,
        "sigma":            float(pred["sigma"]) if pred and pred.get("sigma") is not None else None,
        "actual_volume":    float(actual_vol) if actual_vol is not None else None,
        "error":            None,
        "abs_error_k":      None,
        "z_score":          None,
        "n_trades":         0,
        "realized_pnl":     0.0,
        "trades":           [],
    }
    if out["predicted_volume"] is not None and out["actual_volume"] is not None:
        out["error"]       = out["predicted_volume"] - out["actual_volume"]
        out["abs_error_k"] = abs(out["error"]) / 1000.0
        if out["sigma"]:
            out["z_score"] = out["error"] / out["sigma"]

    # Trades for that event_date (daily fills)
    trade_rows = db.query(
        """
        SELECT f.ticker, f.side, f.market_type, f.strike_millions,
               SUM(CASE WHEN f.action='buy'  THEN f.count ELSE 0 END) AS buy_shares,
               SUM(CASE WHEN f.action='sell' THEN f.count ELSE 0 END) AS sell_shares,
               SUM(CASE WHEN f.action='buy' THEN
                   CASE WHEN f.side='yes' THEN f.yes_price_cents*f.count
                        ELSE f.no_price_cents*f.count END ELSE 0 END) AS buy_cost_cents,
               MAX(s.result) AS result
        FROM fills f
        LEFT JOIN settlements s USING (ticker)
        WHERE f.event_date = %s AND f.market_type = 'daily'
        GROUP BY f.ticker, f.side, f.market_type, f.strike_millions
        """,
        (target,),
    )

    for r in trade_rows:
        buy = int(r["buy_shares"] or 0)
        if buy <= 0: continue
        net = buy - int(r["sell_shares"] or 0)
        avg_cents = (r["buy_cost_cents"] / buy) if buy > 0 else None
        cost_dollars = (r["buy_cost_cents"] or 0) / 100.0
        result = r["result"] or None
        won = (result == r["side"]) if result else None
        pnl = ((net - cost_dollars) if won else (-cost_dollars)) if result else None
        model_p = trades.get_model_prob(
            r["ticker"], r["market_type"], r["side"],
            target.isoformat(),
            float(r["strike_millions"]) if r["strike_millions"] is not None else None,
        )
        market_p = (avg_cents / 100.0) if avg_cents is not None else None
        edge = (model_p - market_p) if (model_p is not None and market_p is not None) else None

        out["trades"].append({
            "ticker":      r["ticker"],
            "side":        r["side"],
            "strike":      float(r["strike_millions"]) if r["strike_millions"] is not None else None,
            "shares":      buy,
            "avg_cents":   round(avg_cents, 2) if avg_cents is not None else None,
            "edge_pct":    round(edge * 100, 2) if edge is not None else None,
            "won":         won,
            "pnl_dollars": round(pnl, 2) if pnl is not None else None,
        })
        if pnl is not None:
            out["realized_pnl"] += pnl
    out["n_trades"] = len(out["trades"])
    out["realized_pnl"] = round(out["realized_pnl"], 2)
    return out


def today_brief(today: date | None = None) -> dict:
    """Today's prediction + candidate daily-market edges, ranked.

    Returns:
        {
            "date": ..., "day_name": ..., "regime": ...,
            "predicted_volume": ..., "sigma": ...,
            "ci90_low": ..., "ci90_high": ...,
            "candidates": [          # ranked by |edge|
                {ticker, strike, side, model_prob, market_prob, edge,
                 yes_bid_cents, yes_ask_cents}
            ]
        }
    """
    today = today or date.today()
    pred = _latest_prediction(today, "daily")
    out: dict = {
        "date":             today.isoformat(),
        "day_name":         today.strftime("%A"),
        "regime":           pred.get("regime") if pred else None,
        "predicted_volume": float(pred["predicted_volume"]) if pred and pred.get("predicted_volume") is not None else None,
        "sigma":            float(pred["sigma"]) if pred and pred.get("sigma") is not None else None,
        "ci90_low":         float(pred["ci90_low"]) if pred and pred.get("ci90_low") is not None else None,
        "ci90_high":        float(pred["ci90_high"]) if pred and pred.get("ci90_high") is not None else None,
        "candidates":       [],
    }

    # Latest market snapshot per daily ticker for today.
    today_yy = today.strftime("%y%b%d").upper()        # e.g. '26JUN14'
    ticker_like = f"KXTRUFTSA-{today_yy}-%"
    rows = db.query(
        """
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY run_at DESC) rn
            FROM market_snapshots WHERE ticker LIKE %s
        )
        SELECT ticker, strike_millions, yes_bid_cents, yes_ask_cents,
               yes_mid_cents, market_prob, model_prob, edge
        FROM ranked WHERE rn = 1
        """,
        (ticker_like,),
    )
    for r in rows:
        for side in ("yes", "no"):
            mp = r["model_prob"] if side == "yes" else (1 - r["model_prob"] if r["model_prob"] is not None else None)
            kp = r["market_prob"] if side == "yes" else (1 - r["market_prob"] if r["market_prob"] is not None else None)
            edge = (mp - kp) if (mp is not None and kp is not None) else None
            out["candidates"].append({
                "ticker":         r["ticker"],
                "strike":         float(r["strike_millions"]) if r["strike_millions"] is not None else None,
                "side":           side,
                "model_prob":     float(mp) if mp is not None else None,
                "market_prob":    float(kp) if kp is not None else None,
                "edge":           float(edge) if edge is not None else None,
                "yes_bid_cents":  float(r["yes_bid_cents"]) if r["yes_bid_cents"] is not None else None,
                "yes_ask_cents":  float(r["yes_ask_cents"]) if r["yes_ask_cents"] is not None else None,
            })
    out["candidates"].sort(key=lambda c: abs(c["edge"] or 0), reverse=True)
    return out


def running_perf(days: int = 30) -> dict:
    """Trailing-N-day performance: MAE, |z|≤1σ rate, Brier, log loss, P&L.

    Returns:
        {
            "window_days": int,
            "n_predictions_resolved": int,
            "mae_k": float | None,
            "z_within_1sigma_rate": float | None,
            "n_markets_resolved": int,
            "brier_yes_side": float | None,
            "logloss_yes_side": float | None,
            "n_trades_settled": int,
            "total_realized_pnl": float,
            "by_edge_bucket": {bucket: {n, wins, pnl}},
        }
    """
    cutoff = date.today() - timedelta(days=days)

    # 1. Daily prediction accuracy (predictions × tsa_actuals)
    pred_rows = db.query(
        """
        WITH latest AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY forecast_date ORDER BY run_at DESC) rn
            FROM predictions WHERE market_type = 'daily' AND forecast_date >= %s
        )
        SELECT p.forecast_date, p.predicted_volume, p.sigma, a.volume AS actual
        FROM latest p
        LEFT JOIN tsa_actuals a ON a.date = p.forecast_date
        WHERE p.rn = 1 AND a.volume IS NOT NULL
        """,
        (cutoff,),
    )
    abs_err_k: list[float] = []
    n_within_sigma = 0
    n_with_sigma = 0
    for r in pred_rows:
        err = float(r["predicted_volume"]) - float(r["actual"])
        abs_err_k.append(abs(err) / 1000.0)
        if r["sigma"]:
            n_with_sigma += 1
            if abs(err) / float(r["sigma"]) <= 1.0:
                n_within_sigma += 1

    # 2. Market calibration over the window (YES-side scoring per ticker)
    cal_rows = db.query(
        """
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY run_at DESC) rn
            FROM market_snapshots
        )
        SELECT m.ticker, m.strike_millions, m.model_prob, s.result, f.event_date
        FROM ranked m
        LEFT JOIN settlements s ON s.ticker = m.ticker
        LEFT JOIN (SELECT DISTINCT ticker, event_date FROM fills) f ON f.ticker = m.ticker
        WHERE m.rn = 1 AND s.result IS NOT NULL AND m.model_prob IS NOT NULL
              AND f.event_date >= %s
        """,
        (cutoff,),
    )
    briers: list[float] = []
    losses: list[float] = []
    for r in cal_rows:
        actual = 1.0 if r["result"] == "yes" else 0.0
        p = max(min(float(r["model_prob"]), 1 - 1e-9), 1e-9)
        briers.append((p - actual) ** 2)
        losses.append(-(actual * math.log(p) + (1 - actual) * math.log(1 - p)))

    # 3. Trade P&L over window (event_date >= cutoff, settled)
    trade_rows = db.query(
        """
        SELECT f.ticker, f.side, f.event_date,
               SUM(CASE WHEN f.action='buy'  THEN f.count ELSE 0 END) AS buy_shares,
               SUM(CASE WHEN f.action='sell' THEN f.count ELSE 0 END) AS sell_shares,
               SUM(CASE WHEN f.action='buy' THEN
                   CASE WHEN f.side='yes' THEN f.yes_price_cents*f.count
                        ELSE f.no_price_cents*f.count END ELSE 0 END) AS buy_cost_cents,
               MAX(s.result) AS result
        FROM fills f
        LEFT JOIN settlements s USING (ticker)
        WHERE f.event_date >= %s
        GROUP BY f.ticker, f.side, f.event_date
        """,
        (cutoff,),
    )
    total_pnl = 0.0
    n_settled = 0
    edge_buckets: dict[str, dict] = {}
    for r in trade_rows:
        buy = int(r["buy_shares"] or 0)
        if buy <= 0: continue
        net = buy - int(r["sell_shares"] or 0)
        cost = (r["buy_cost_cents"] or 0) / 100.0
        if not r["result"]:
            continue
        n_settled += 1
        won = (r["result"] == r["side"])
        pnl = (net - cost) if won else (-cost)
        total_pnl += pnl

        # Edge at trade time (from order_actions snapshot)
        snap = db.query(
            "SELECT AVG(model_prob - market_prob) AS edge FROM order_actions WHERE ticker=%s AND side=%s",
            (r["ticker"], r["side"]),
        )
        edge = snap[0]["edge"] if snap and snap[0]["edge"] is not None else None
        bucket = "no-edge-data" if edge is None else f"{int(edge*100//5*5):+d}%–{int(edge*100//5*5)+5:+d}%"
        b = edge_buckets.setdefault(bucket, {"n": 0, "wins": 0, "pnl": 0.0})
        b["n"] += 1
        if won: b["wins"] += 1
        b["pnl"] += pnl

    return {
        "window_days":            days,
        "n_predictions_resolved": len(pred_rows),
        "mae_k":                  round(sum(abs_err_k)/len(abs_err_k), 2) if abs_err_k else None,
        "z_within_1sigma_rate":   round(n_within_sigma / n_with_sigma, 3) if n_with_sigma else None,
        "n_markets_resolved":     len(briers),
        "brier_yes_side":         round(sum(briers)/len(briers), 4) if briers else None,
        "logloss_yes_side":       round(sum(losses)/len(losses), 4) if losses else None,
        "n_trades_settled":       n_settled,
        "total_realized_pnl":     round(total_pnl, 2),
        "by_edge_bucket":         {k: {**v, "pnl": round(v["pnl"], 2)} for k, v in edge_buckets.items()},
    }


def open_positions_with_edge() -> list[dict]:
    """Mark-to-market view of every open position w/ current market state.

    A position is 'open' if (sum buy_shares > sum sell_shares) AND settlement.result IS NULL.
    Joined to the most recent market_snapshot for that ticker, so you get the
    current market_prob, model_prob (if recorded), and resulting edge.
    """
    rows = db.query(
        """
        WITH agg AS (
            SELECT ticker, side,
                   SUM(CASE WHEN action='buy'  THEN count ELSE 0 END) AS buy_shares,
                   SUM(CASE WHEN action='sell' THEN count ELSE 0 END) AS sell_shares,
                   SUM(CASE WHEN action='buy' THEN
                       CASE WHEN side='yes' THEN yes_price_cents*count
                            ELSE no_price_cents*count END
                       ELSE 0 END) AS buy_cost_cents
            FROM fills GROUP BY ticker, side
        ),
        latest_snap AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY run_at DESC) rn
            FROM market_snapshots
        )
        SELECT a.ticker, a.side,
               (a.buy_shares - a.sell_shares) AS net_shares,
               a.buy_shares, a.buy_cost_cents,
               s.market_prob AS yes_market_prob,
               s.model_prob  AS yes_model_prob,
               s.strike_millions, s.yes_mid_cents, st.result
        FROM agg a
        LEFT JOIN latest_snap s ON s.ticker = a.ticker AND s.rn = 1
        LEFT JOIN settlements st ON st.ticker = a.ticker
        WHERE a.buy_shares > 0 AND st.result IS NULL
        """,
        (),
    )
    out: list[dict] = []
    for r in rows:
        if (r["net_shares"] or 0) <= 0:
            continue
        avg_cents = (r["buy_cost_cents"] / r["buy_shares"]) if r["buy_shares"] else None
        cost = (r["buy_cost_cents"] or 0) / 100.0
        # Per-side market/model prob
        mp_yes = r["yes_model_prob"]; kp_yes = r["yes_market_prob"]
        if r["side"] == "yes":
            mp, kp = mp_yes, kp_yes
        else:
            mp = (1 - mp_yes) if mp_yes is not None else None
            kp = (1 - kp_yes) if kp_yes is not None else None
        edge = (mp - kp) if (mp is not None and kp is not None) else None
        mtm = None
        if kp is not None and r["net_shares"]:
            # Mark-to-market: current market price × net shares − cost basis
            mtm = (kp * 100 * r["net_shares"]) / 100.0 - cost
        out.append({
            "ticker":       r["ticker"],
            "side":         r["side"],
            "shares":       int(r["net_shares"]),
            "avg_cents":    round(avg_cents, 2) if avg_cents is not None else None,
            "cost_dollars": round(cost, 2),
            "market_prob":  float(kp) if kp is not None else None,
            "model_prob":   float(mp) if mp is not None else None,
            "edge":         float(edge) if edge is not None else None,
            "mtm_dollars":  round(mtm, 2) if mtm is not None else None,
            "strike":       float(r["strike_millions"]) if r["strike_millions"] is not None else None,
        })
    out.sort(key=lambda p: -(p["mtm_dollars"] or 0))
    return out


def forecast_change(forecast_date, market_type: str = "weekly") -> dict | None:
    """How did the prediction for `forecast_date` move between the two latest runs.

    Returns None if there's only one (or zero) prediction recorded.
    """
    rows = db.query(
        """
        SELECT run_at, predicted_volume, regime, sigma
        FROM predictions
        WHERE forecast_date = %s AND market_type = %s
        ORDER BY run_at DESC LIMIT 2
        """,
        (forecast_date, market_type),
    )
    if len(rows) < 2:
        return None
    latest, prior = rows[0], rows[1]
    delta = float(latest["predicted_volume"]) - float(prior["predicted_volume"])
    return {
        "forecast_date":   forecast_date.isoformat() if hasattr(forecast_date, "isoformat") else str(forecast_date),
        "market_type":     market_type,
        "latest_run_at":   latest["run_at"].isoformat() if hasattr(latest["run_at"], "isoformat") else str(latest["run_at"]),
        "prior_run_at":    prior["run_at"].isoformat() if hasattr(prior["run_at"], "isoformat") else str(prior["run_at"]),
        "latest_prediction": float(latest["predicted_volume"]),
        "prior_prediction":  float(prior["predicted_volume"]),
        "delta":              delta,
        "delta_k":            delta / 1000.0,
        "regime_changed":     latest.get("regime") != prior.get("regime"),
        "latest_regime":      latest.get("regime"),
        "prior_regime":       prior.get("regime"),
    }


def jarvis_brief(today: date | None = None, perf_window_days: int = 30) -> dict:
    """One-shot bundle for jarvis. Calls all of the above and packages the result."""
    today = today or date.today()
    return {
        "today":              today.isoformat(),
        "yesterday":          yesterday_brief(today),
        "today_forecast":     today_brief(today),
        "open_positions":     open_positions_with_edge(),
        "running_perf":       running_perf(days=perf_window_days),
        "weekly_change":      forecast_change(
            today + timedelta(days=(6 - today.weekday()) % 7),  # next Sunday
            market_type="weekly",
        ),
    }


def cmd_brief(args) -> None:
    """Print the jarvis bundle as JSON for inspection / piping."""
    import json
    brief = jarvis_brief(perf_window_days=args.window)
    print(json.dumps(brief, default=str, indent=2))


def cmd_predictions(args) -> None:
    """Predictions backlog: predicted vs actual, error, ±σ, per forecast date.

    Joins predictions × tsa_actuals × order_actions × fills so you can see for
    every past prediction: what I predicted, what edge I had at trade time,
    whether I actually traded, and whether the outcome agreed with the model.
    Doesn't depend on market_snapshots — works for daily even when daily
    market snapshots aren't being recorded.
    """
    if not db.available():
        print("[backlog] POSTGRES_URL not set.", file=sys.stderr); sys.exit(1)

    mt = _market_filter(args) or "daily"
    # Latest run per (forecast_date) to avoid double-counting multi-run days.
    rows = db.query(
        """
        WITH latest AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY forecast_date ORDER BY run_at DESC
            ) AS rn
            FROM predictions WHERE market_type = %s
        ),
        trade_summary AS (
            SELECT
                event_date,
                COUNT(DISTINCT ticker) AS n_tickers,
                SUM(count) AS shares,
                SUM(CASE WHEN action='buy' THEN
                    CASE WHEN side='yes' THEN yes_price_cents*count
                         ELSE no_price_cents*count END
                    ELSE 0 END)/100.0 AS cost_dollars
            FROM fills WHERE market_type = %s
            GROUP BY event_date
        ),
        edge_summary AS (
            -- Average proposed model edge from order_actions at any run that day.
            SELECT date_trunc('day', run_at)::date AS run_date,
                   AVG(model_prob - market_prob) AS avg_edge
            FROM order_actions
            WHERE model_prob IS NOT NULL AND market_prob IS NOT NULL
            GROUP BY 1
        )
        SELECT
            p.forecast_date,
            p.day_name,
            p.regime,
            p.predicted_volume,
            p.sigma,
            a.volume AS actual_volume,
            (p.predicted_volume - a.volume) AS error,
            CASE WHEN a.volume IS NOT NULL AND p.sigma IS NOT NULL THEN
                (p.predicted_volume - a.volume) / p.sigma
            END AS z_score,
            ts.n_tickers AS daily_trade_tickers,
            ts.shares    AS daily_shares,
            ts.cost_dollars AS daily_cost_dollars,
            es.avg_edge  AS avg_edge_at_run
        FROM latest p
        LEFT JOIN tsa_actuals    a  ON a.date       = p.forecast_date
        LEFT JOIN trade_summary  ts ON ts.event_date = p.forecast_date
        LEFT JOIN edge_summary   es ON es.run_date  = p.forecast_date
        WHERE p.rn = 1
        ORDER BY p.forecast_date DESC
        """,
        (mt, mt),
    )
    if args.last:
        rows = rows[:args.last]

    if not rows:
        print("[backlog] no predictions yet.")
        return

    print(f"\n=== PREDICTIONS BACKLOG ({mt}, {len(rows)} rows) ===")
    cols = (
        ("date",          12, "<"),
        ("day",            4, "<"),
        ("regime",        14, "<"),
        ("pred(M)",        8, ">"),
        ("σ(M)",           7, ">"),
        ("act(M)",         8, ">"),
        ("err(k)",         8, ">"),
        ("z",              6, ">"),
        ("trades",         7, ">"),
        ("cost$",          8, ">"),
        ("avg_edge",       9, ">"),
    )
    hdr = "  ".join(f"{n:{a}{w}}" for n, w, a in cols)
    print(hdr); print("-" * len(hdr))

    n_within_1sigma = n_resolved = 0
    abs_errors_k: list[float] = []
    for r in rows:
        pv  = r["predicted_volume"]
        av  = r["actual_volume"]
        sig = r["sigma"]
        err = r["error"]
        z   = r["z_score"]
        if av is not None:
            n_resolved += 1
            if err is not None: abs_errors_k.append(abs(err) / 1000)
            if z is not None and abs(z) <= 1.0: n_within_1sigma += 1

        cells = (
            f"{str(r['forecast_date']):<12}",
            f"{(r['day_name'] or '')[:3]:<4}",
            f"{(r['regime'] or '')[:14]:<14}",
            f"{_fmt(pv/1e6 if pv else None, '.3f'):>8}",
            f"{_fmt(sig/1e6 if sig else None, '.3f'):>7}",
            f"{_fmt(av/1e6 if av else None, '.3f'):>8}",
            f"{_fmt(err/1000 if err is not None else None, '+.1f'):>8}",
            f"{_fmt(z, '+.2f'):>6}",
            f"{_fmt(int(r['daily_trade_tickers']) if r['daily_trade_tickers'] else 0):>7}",
            f"{_fmt(r['daily_cost_dollars'], '.2f'):>8}",
            f"{_fmt(r['avg_edge_at_run']*100 if r['avg_edge_at_run'] is not None else None, '+.1f'):>9}",
        )
        print("  ".join(cells))

    if n_resolved:
        mae_k = sum(abs_errors_k) / len(abs_errors_k)
        print("-" * len(hdr))
        print(f"resolved: {n_resolved}/{len(rows)}   "
              f"MAE: {mae_k:.1f}k   "
              f"|z|≤1σ: {n_within_1sigma}/{n_resolved} ({n_within_1sigma/n_resolved*100:.0f}%)")


def cmd_export(args) -> None:
    df = _load_or_fail(args)
    if df.empty: return
    out_dir = Path(args.out) if args.out else OUT_DIR
    if args.dry_run:
        print(f"[backlog] DRY-RUN — would write {len(df)} rows to {out_dir}")
        return
    written = export_csvs(df, out_dir)
    for name, p in written.items():
        print(f"[backlog] wrote {name} -> {p}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[backlog] %(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_common(sp):
        sp.add_argument("--daily",  action="store_true", help="only daily markets")
        sp.add_argument("--weekly", action="store_true", help="only weekly markets")
        sp.add_argument("--dry-run", action="store_true", help="no writes")

    sp = sub.add_parser("snapshot");    _add_common(sp); sp.set_defaults(fn=cmd_snapshot)
    sp = sub.add_parser("open");        _add_common(sp); sp.add_argument("--last", type=int, default=0); sp.set_defaults(fn=cmd_open)
    sp = sub.add_parser("resolved");    _add_common(sp); sp.add_argument("--last", type=int, default=0); sp.set_defaults(fn=cmd_resolved)
    sp = sub.add_parser("perf");        _add_common(sp); sp.set_defaults(fn=cmd_perf)
    sp = sub.add_parser("calibration"); _add_common(sp); sp.add_argument("--bins", type=int, default=10); sp.set_defaults(fn=cmd_calibration)
    sp = sub.add_parser("export");      _add_common(sp); sp.add_argument("--out", type=str, default=None); sp.set_defaults(fn=cmd_export)
    sp = sub.add_parser("predictions",  help="predicted vs actual w/ error, |z|, trade summary");
    _add_common(sp); sp.add_argument("--last", type=int, default=0); sp.set_defaults(fn=cmd_predictions)
    sp = sub.add_parser("brief",        help="jarvis bundle: yesterday + today + perf + positions (JSON)");
    _add_common(sp); sp.add_argument("--window", type=int, default=30, help="rolling window days"); sp.set_defaults(fn=cmd_brief)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
