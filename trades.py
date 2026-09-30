#!/usr/bin/env python3
"""trades.py — Local SQLite trade tracker for Kalshi TSA markets.

Pulls fills (truth) + settled positions from the Kalshi API into
output_kalshi/trades.db, aggregates fills into position-level rows
(avg fill price, total shares), and reports W/L + model edge.

Usage:
    python trades.py refresh                # incremental sync from API
    python trades.py positions              # open positions
    python trades.py positions --daily      # only daily markets
    python trades.py history                # settled positions w/ W/L + P&L
    python trades.py history --weekly       # only weekly markets
    python trades.py bins                   # win-rate by price-bin + edge-bin
    python trades.py bins --daily
    python trades.py all                    # refresh + positions + history + bins

Notes:
    - Daily ticker  KXTRUFTSA-26JUN12-T2120000 -> threshold 2.12M, date 2026-06-12
    - Weekly ticker KXTSAW-26JUN14-A1.80       -> threshold 1.80M, week-end 2026-06-14
    - Model prob: daily markets only (looked up from daily_forecast.csv).
      YES side = p_over_<strike>M, NO side = 1 - p_over_<strike>M.
    - Price bins are 20c wide: 0-20, 20-40, 40-60, 60-80, 80-100.
    - Edge bins are 5% wide.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import kalshi

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output_kalshi"
DB_PATH = OUT_DIR / "trades.db"
DAILY_FORECAST_CSV = ROOT / "output_autogluon_predict" / "daily_forecast.csv"

PRICE_BIN_WIDTH = 20.0   # cents
EDGE_BIN_WIDTH = 5.0     # percent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (
    trade_id        TEXT PRIMARY KEY,
    order_id        TEXT,
    ticker          TEXT NOT NULL,
    side            TEXT NOT NULL,        -- 'yes' / 'no'
    action          TEXT,                 -- 'buy' / 'sell'
    count           INTEGER,
    yes_price_cents REAL,
    no_price_cents  REAL,
    is_taker        INTEGER,
    created_time    TEXT,
    market_type     TEXT,                 -- 'daily' / 'weekly'
    strike_millions REAL,
    event_date      TEXT
);
CREATE INDEX IF NOT EXISTS idx_fills_ticker     ON fills(ticker);
CREATE INDEX IF NOT EXISTS idx_fills_event_date ON fills(event_date);

CREATE TABLE IF NOT EXISTS market_results (
    ticker       TEXT PRIMARY KEY,
    status       TEXT,                    -- 'finalized' / 'settled' / 'active' / ...
    result       TEXT,                    -- 'yes' / 'no' / NULL
    settled_time TEXT,
    refreshed_at TEXT
);

CREATE TABLE IF NOT EXISTS sync_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _conn() -> sqlite3.Connection:
    OUT_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


# ── Ticker parsing ───────────────────────────────────────────────────────────
def parse_ticker(ticker: str) -> tuple[str | None, float | None, str | None]:
    """Return (market_type, strike_millions, event_date) or (None, None, None)."""
    parts = ticker.split("-")
    if len(parts) < 3:
        return None, None, None

    if parts[0] == "KXTRUFTSA":
        market_type = "daily"
    elif parts[0] == "KXTSAW":
        market_type = "weekly"
    else:
        return None, None, None

    try:
        event_date = datetime.strptime(parts[1], "%y%b%d").strftime("%Y-%m-%d")
    except ValueError:
        event_date = None

    strike = None
    last = parts[-1]
    if last.startswith("T") and last[1:].isdigit():
        strike = round(int(last[1:]) / 1e6, 4)
    elif last.startswith("A"):
        try:
            strike = float(last[1:])
        except ValueError:
            pass

    return market_type, strike, event_date


# ── Kalshi API: fills ────────────────────────────────────────────────────────
def fetch_fills(min_ts_iso: str | None = None) -> list[dict]:
    """Pull all fills, paginated. `min_ts_iso` is an inclusive lower-bound."""
    auth = kalshi._load_auth()
    params: dict[str, str] = {"limit": "1000"}
    if min_ts_iso:
        try:
            ts = datetime.fromisoformat(min_ts_iso.replace("Z", "+00:00"))
            params["min_ts"] = str(int(ts.timestamp()))
        except Exception:
            pass

    out: list[dict] = []
    while True:
        data = kalshi._get("/portfolio/fills", auth, params=params)
        chunk = data.get("fills") or []
        out.extend(chunk)
        cursor = data.get("cursor")
        if not cursor:
            break
        params["cursor"] = cursor
    return out


def _price_to_cents(v) -> float | None:
    """Kalshi fills return prices as dollar strings (e.g., '0.9300'). Convert to cents."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f * 100.0 if f <= 1.0 else f


def _to_int(v) -> int:
    """Kalshi returns counts as fractional strings (e.g., '5.00'). Round to int."""
    if v is None or v == "":
        return 0
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return 0


def refresh() -> None:
    conn = _conn()

    row = conn.execute("SELECT value FROM sync_meta WHERE key='last_fill_ts'").fetchone()
    last_ts = row[0] if row else None
    print(f"[trades] Fetching fills since {last_ts or 'beginning'}…")
    try:
        fills = fetch_fills(min_ts_iso=last_ts)
    except Exception as e:
        print(f"[trades] fills fetch failed: {e}")
        fills = []
    print(f"[trades] Got {len(fills)} fills from API")

    new_count = 0
    max_ts = last_ts
    tickers_seen: set[str] = set()
    for f in fills:
        ticker = f.get("ticker") or f.get("market_ticker") or ""
        if "KXTSAW" not in ticker and "KXTRUFTSA" not in ticker:
            continue
        trade_id = f.get("trade_id") or f.get("fill_id")
        if not trade_id:
            continue
        mt, strike, event_date = parse_ticker(ticker)
        yp = _price_to_cents(f.get("yes_price_dollars"))
        np_ = _price_to_cents(f.get("no_price_dollars"))
        count = _to_int(f.get("count_fp") or f.get("count"))
        conn.execute(
            """
            INSERT OR REPLACE INTO fills
              (trade_id, order_id, ticker, side, action, count,
               yes_price_cents, no_price_cents, is_taker, created_time,
               market_type, strike_millions, event_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id, f.get("order_id"), ticker,
                f.get("side"), f.get("action"),
                count, yp, np_,
                1 if f.get("is_taker") else 0,
                f.get("created_time"),
                mt, strike, event_date,
            ),
        )
        new_count += 1
        tickers_seen.add(ticker)
        ct = f.get("created_time")
        if ct and (max_ts is None or ct > max_ts):
            max_ts = ct

    if max_ts:
        conn.execute(
            "INSERT INTO sync_meta(key, value) VALUES('last_fill_ts', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (max_ts,),
        )
    conn.commit()

    # Refresh market results for every ticker we have fills for.
    # Skip tickers we've already marked finalized (immutable).
    all_tickers = {row[0] for row in conn.execute("SELECT DISTINCT ticker FROM fills")}
    finalized = {row[0] for row in conn.execute(
        "SELECT ticker FROM market_results WHERE status IN ('finalized','settled')"
    )}
    to_refresh = sorted(all_tickers - finalized)
    print(f"[trades] Refreshing market_results for {len(to_refresh)} ticker(s)…")
    now = datetime.utcnow().isoformat() + "Z"
    mr_count = 0
    for ticker in to_refresh:
        try:
            mr = kalshi.fetch_market_result(ticker) or {}
        except Exception as e:
            print(f"  [trades] fetch_market_result({ticker}) failed: {e}")
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO market_results
              (ticker, status, result, settled_time, refreshed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticker, mr.get("status"), (mr.get("result") or None),
             mr.get("settled_time"), now),
        )
        mr_count += 1

    conn.commit()
    conn.close()
    print(f"[trades] DB updated: {new_count} fills, {mr_count} market_results rows")

    # Mirror to Postgres so all storage joins live in one place.
    # SQLite remains the local fast backup; Postgres becomes canonical.
    try:
        import db as _pgdb
        if _pgdb.available():
            _mirror_to_postgres()
    except Exception as exc:
        print(f"[trades] postgres mirror failed (non-fatal): {exc}", file=sys.stderr)


def _mirror_to_postgres() -> None:
    """Push all SQLite fills + market_results into Postgres. Idempotent."""
    import db as _pgdb
    sconn = sqlite3.connect(DB_PATH); sconn.row_factory = sqlite3.Row
    try:
        fills = [dict(r) for r in sconn.execute("SELECT * FROM fills").fetchall()]
        results = [dict(r) for r in sconn.execute("SELECT * FROM market_results").fetchall()]
    finally:
        sconn.close()
    if fills:    _pgdb.write_fills(fills)
    if results:  _pgdb.write_settlements(results)


# ── Model probability lookup ─────────────────────────────────────────────────
# Priority 1: postgres order_actions.model_prob (recorded at order placement).
# Priority 2: today's daily_forecast.csv (only useful for current-day open positions).
_pg_cache: dict[tuple[str, str], float] | None = None
_daily_cache: pd.DataFrame | None = None


def _load_pg_model_probs() -> dict[tuple[str, str], float]:
    """Pull model_prob per (ticker, side) from postgres. Average over runs."""
    global _pg_cache
    if _pg_cache is not None:
        return _pg_cache
    try:
        import db
        if not db.available():
            _pg_cache = {}
            return _pg_cache
        rows = db.query(
            """
            SELECT ticker, side, AVG(model_prob) AS model_prob
            FROM order_actions
            WHERE model_prob IS NOT NULL
            GROUP BY ticker, side
            """,
            (),
        )
        _pg_cache = {(r["ticker"], r["side"]): float(r["model_prob"]) for r in rows}
    except Exception as e:
        print(f"[trades] postgres lookup failed: {e}", file=sys.stderr)
        _pg_cache = {}
    return _pg_cache


def _load_daily_forecast() -> pd.DataFrame:
    global _daily_cache
    if _daily_cache is not None:
        return _daily_cache
    _daily_cache = pd.read_csv(DAILY_FORECAST_CSV) if DAILY_FORECAST_CSV.exists() else pd.DataFrame()
    return _daily_cache


def get_model_prob(ticker: str, market_type: str | None, side: str | None,
                   event_date: str | None, strike_millions: float | None) -> float | None:
    """Return P(this side wins) at the time the order was placed.

    Lookup order:
      1. postgres order_actions(ticker, side) → recorded model_prob
      2. today's daily_forecast.csv (only matches current-day open daily positions)
    """
    if not side:
        return None

    pg = _load_pg_model_probs()
    if (ticker, side) in pg:
        return pg[(ticker, side)]

    if market_type == "daily" and event_date and strike_millions is not None:
        df = _load_daily_forecast()
        if not df.empty:
            row = df[df["date"] == event_date]
            if not row.empty:
                col = f"p_over_{strike_millions:.2f}M"
                if col in row.columns:
                    try:
                        p_over = float(row.iloc[0][col])
                        return p_over if side == "yes" else (1.0 - p_over)
                    except (TypeError, ValueError):
                        pass
    return None


# ── Aggregate positions ──────────────────────────────────────────────────────
def aggregate_positions(conn: sqlite3.Connection, market: str | None = None) -> list[dict]:
    where = ""
    params: list = []
    if market:
        where = "WHERE market_type = ?"
        params.append(market)

    sql = f"""
        SELECT
          ticker, side, market_type, strike_millions, event_date,
          SUM(CASE WHEN action='buy'  THEN count ELSE 0 END) AS buy_shares,
          SUM(CASE WHEN action='sell' THEN count ELSE 0 END) AS sell_shares,
          SUM(CASE WHEN action='buy' THEN
              CASE WHEN side='yes' THEN yes_price_cents * count
                   ELSE no_price_cents  * count END
              ELSE 0 END) AS buy_cost_cents
        FROM fills
        {where}
        GROUP BY ticker, side, market_type, strike_millions, event_date
    """
    rows: list[dict] = []
    for r in conn.execute(sql, params).fetchall():
        d = dict(r)
        buy = d["buy_shares"] or 0
        sell = d["sell_shares"] or 0
        net = buy - sell
        avg_cents = (d["buy_cost_cents"] / buy) if buy > 0 else None

        mr = conn.execute(
            "SELECT status, result, settled_time FROM market_results WHERE ticker=?",
            (d["ticker"],),
        ).fetchone()
        result = mr["result"] if (mr and mr["result"]) else None
        settled = result is not None
        cost_dollars = (d["buy_cost_cents"] / 100.0) if d["buy_cost_cents"] else 0.0

        if settled:
            realized = (net - cost_dollars) if (result == d["side"]) else (-cost_dollars)
            # net shares is what was held at settlement (buys minus sells)
        else:
            realized = None

        out = {
            "ticker":          d["ticker"],
            "side":            d["side"],
            "market_type":     d["market_type"],
            "strike_millions": d["strike_millions"],
            "event_date":      d["event_date"],
            "net_shares":      net,
            "buy_shares":      buy,
            "avg_price_cents": avg_cents,
            "buy_cost_dollars": cost_dollars,
            "settled":         settled,
            "result":          result,
            "realized_dollars": realized,
            "settled_time":    mr["settled_time"] if mr else None,
            "market_status":   mr["status"] if mr else None,
        }
        out["market_prob"] = (avg_cents / 100.0) if avg_cents is not None else None
        out["model_prob"]  = get_model_prob(
            out["ticker"], out["market_type"], out["side"],
            out["event_date"], out["strike_millions"],
        )
        if out["model_prob"] is not None and out["market_prob"] is not None:
            out["edge"] = out["model_prob"] - out["market_prob"]
        else:
            out["edge"] = None
        out["win"] = (result == d["side"]) if settled else None

        # Skip phantom rows (Kalshi cross-side encoding or pre-window holdings):
        # if we have no buys for this side, we have no cost basis to report.
        if buy > 0:
            rows.append(out)
    return rows


# ── Rendering ────────────────────────────────────────────────────────────────
def _fmt(v, spec=""):
    if v is None:
        return "--"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def render_positions(rows: list[dict], title: str) -> None:
    if not rows:
        print(f"\n=== {title}: (none) ===")
        return
    print(f"\n=== {title} ({len(rows)}) ===")
    cols = (
        ("ticker",      32, "<"),
        ("side",         4, "<"),
        ("shares",       6, ">"),
        ("avg¢",         6, ">"),
        ("mkt%",         6, ">"),
        ("model%",       7, ">"),
        ("edge%",        7, ">"),
        ("result",       6, "<"),
        ("pnl$",         9, ">"),
    )
    header = "  ".join(f"{name:{align}{w}}" for name, w, align in cols)
    print(header)
    print("-" * len(header))

    tot_pnl = 0.0
    n_win = n_settled = 0
    for r in sorted(rows, key=lambda x: (x.get("event_date") or "", x["ticker"])):
        mp = r["model_prob"] * 100 if r["model_prob"] is not None else None
        mk = r["market_prob"] * 100 if r["market_prob"] is not None else None
        ed = r["edge"] * 100 if r["edge"] is not None else None
        pnl = r["realized_dollars"]
        if pnl is not None:
            tot_pnl += pnl
        if r["settled"]:
            n_settled += 1
            if r["win"]:
                n_win += 1
        cells = (
            f"{r['ticker']:<32}",
            f"{r['side']:<4}",
            f"{r['net_shares'] if r['net_shares'] else r['buy_shares']:>6d}",
            f"{_fmt(r['avg_price_cents'], '.1f'):>6}",
            f"{_fmt(mk, '.1f'):>6}",
            f"{_fmt(mp, '.1f'):>7}",
            f"{_fmt(ed, '+.1f'):>7}",
            f"{r['result'] or '--':<6}",
            f"{_fmt(pnl, '+.2f'):>9}",
        )
        print("  ".join(cells))

    if n_settled:
        wr = n_win / n_settled * 100
        print("-" * len(header))
        print(f"settled: {n_settled}  wins: {n_win}  win-rate: {wr:.1f}%  realized P&L: ${tot_pnl:+.2f}")


# ── Bins ─────────────────────────────────────────────────────────────────────
def _price_bin_label(c: float) -> str:
    idx = min(int(c // PRICE_BIN_WIDTH), 4)
    lo = idx * PRICE_BIN_WIDTH
    hi = lo + PRICE_BIN_WIDTH
    return f"{int(lo):>3}–{int(hi):>3}¢"


def _edge_bin_label(e_pct: float) -> str:
    # Floor to nearest 5% (signed)
    import math
    lo = math.floor(e_pct / EDGE_BIN_WIDTH) * EDGE_BIN_WIDTH
    hi = lo + EDGE_BIN_WIDTH
    return f"{int(lo):>+4}%–{int(hi):>+4}%"


def render_bins(rows: list[dict], title: str) -> None:
    settled = [r for r in rows if r["settled"] and r["result"] is not None]
    if not settled:
        print(f"\n=== {title}: (no settled trades) ===")
        return

    print(f"\n=== {title} — price bins ({int(PRICE_BIN_WIDTH)}¢ wide) ===")
    price_buckets: dict[str, list[dict]] = {}
    for r in settled:
        if r["avg_price_cents"] is None:
            continue
        price_buckets.setdefault(_price_bin_label(r["avg_price_cents"]), []).append(r)
    _print_buckets(price_buckets)

    print(f"\n=== {title} — edge bins ({int(EDGE_BIN_WIDTH)}% wide) ===")
    edge_buckets: dict[str, list[dict]] = {}
    for r in settled:
        if r["edge"] is None:
            continue
        edge_buckets.setdefault(_edge_bin_label(r["edge"] * 100), []).append(r)
    if not edge_buckets:
        print("(no edge data — no model_prob found for any settled trade)")
    else:
        _print_buckets(edge_buckets)


def _print_buckets(buckets: dict[str, list[dict]]) -> None:
    hdr = f"{'bin':<14}  {'n':>3}  {'wins':>4}  {'win%':>6}  {'pnl$':>10}  {'avg edge%':>10}"
    print(hdr)
    print("-" * len(hdr))

    def sort_key(label: str) -> float:
        # Extract the lower bound numeric for sorting (handles '+10%–+15%' and '  0– 20¢')
        tok = label.strip().split("–")[0].strip()
        try:
            return float(tok.rstrip("¢").rstrip("%"))
        except ValueError:
            return 0.0

    for label in sorted(buckets.keys(), key=sort_key):
        rs = buckets[label]
        n = len(rs)
        wins = sum(1 for r in rs if r["win"])
        wr = wins / n * 100
        pnl = sum((r["realized_dollars"] or 0) for r in rs)
        edges = [r["edge"] for r in rs if r["edge"] is not None]
        avg_edge = (sum(edges) / len(edges) * 100) if edges else None
        print(f"{label:<14}  {n:>3d}  {wins:>4d}  {wr:>5.1f}%  {pnl:>+10.2f}  "
              f"{_fmt(avg_edge, '+.1f'):>10}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def cmd_positions(args) -> None:
    conn = _conn()
    market = "daily" if args.daily else ("weekly" if args.weekly else None)
    rows = [r for r in aggregate_positions(conn, market) if not r["settled"]]
    render_positions(rows, "OPEN POSITIONS" + (f" ({market})" if market else ""))


def cmd_history(args) -> None:
    conn = _conn()
    market = "daily" if args.daily else ("weekly" if args.weekly else None)
    rows = [r for r in aggregate_positions(conn, market) if r["settled"]]
    rows.sort(key=lambda r: (r.get("settled_time") or ""), reverse=True)
    render_positions(rows, "TRADE HISTORY" + (f" ({market})" if market else ""))


def cmd_bins(args) -> None:
    conn = _conn()
    market = "daily" if args.daily else ("weekly" if args.weekly else None)
    rows = aggregate_positions(conn, market)
    render_bins(rows, "WIN-RATE BINS" + (f" ({market})" if market else ""))


def cmd_all(args) -> None:
    refresh()
    conn = _conn()
    market = "daily" if args.daily else ("weekly" if args.weekly else None)
    rows = aggregate_positions(conn, market)
    suffix = f" ({market})" if market else ""
    render_positions([r for r in rows if not r["settled"]], "OPEN POSITIONS" + suffix)
    render_positions(sorted([r for r in rows if r["settled"]],
                            key=lambda r: r.get("settled_time") or "", reverse=True),
                     "TRADE HISTORY" + suffix)
    render_bins(rows, "WIN-RATE BINS" + suffix)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("refresh", help="pull new fills + settled positions from Kalshi API")

    def _add_filters(sp):
        sp.add_argument("--daily",  action="store_true", help="only daily markets (KXTRUFTSA)")
        sp.add_argument("--weekly", action="store_true", help="only weekly markets (KXTSAW)")

    sp = sub.add_parser("positions", help="open positions w/ avg price + edge")
    _add_filters(sp); sp.set_defaults(fn=cmd_positions)

    sp = sub.add_parser("history",   help="settled positions w/ W/L + P&L")
    _add_filters(sp); sp.set_defaults(fn=cmd_history)

    sp = sub.add_parser("bins",      help="win-rate by price bin + edge bin")
    _add_filters(sp); sp.set_defaults(fn=cmd_bins)

    sp = sub.add_parser("all",       help="refresh + positions + history + bins")
    _add_filters(sp); sp.set_defaults(fn=cmd_all)

    args = p.parse_args()
    if args.cmd == "refresh":
        refresh()
    else:
        args.fn(args)


if __name__ == "__main__":
    main()
