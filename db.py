"""
db.py — Postgres persistence layer for the TSA trading pipeline.

Set POSTGRES_URL in .env:
    POSTGRES_URL=postgresql://user:password@localhost:5432/tsa_trading

If POSTGRES_URL is not set, or psycopg2 is not installed, all writes are
silently skipped so the pipeline never fails because of the DB.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
    _HAVE_PSYCOPG2 = True
except ImportError:
    _HAVE_PSYCOPG2 = False

try:
    import pandas as pd
    _HAVE_PANDAS = True
except ImportError:
    _HAVE_PANDAS = False

POSTGRES_URL = os.environ.get("POSTGRES_URL", "")


def available() -> bool:
    return _HAVE_PSYCOPG2 and bool(POSTGRES_URL)


def _conn():
    if not _HAVE_PSYCOPG2:
        raise RuntimeError("psycopg2 not installed: pip install psycopg2-binary")
    if not POSTGRES_URL:
        raise RuntimeError("POSTGRES_URL not set")
    return psycopg2.connect(POSTGRES_URL)


def _run_at() -> datetime:
    """Return the pipeline run timestamp from env, or now()."""
    ts = os.environ.get("PIPELINE_RUN_TS", "")
    if ts:
        try:
            return datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _guard(fn):
    """Decorator: skip silently if DB not available; log and skip on errors."""
    import functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not available():
            return
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            print(f"[db] {fn.__name__} failed: {exc}", file=sys.stderr)
    return wrapper


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tsa_actuals (
    date         DATE PRIMARY KEY,
    volume       BIGINT NOT NULL,
    recorded_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS daily_forecast_snapshots (
    id             SERIAL PRIMARY KEY,
    run_at         TIMESTAMPTZ NOT NULL,
    forecast_date  DATE        NOT NULL,
    day_name       TEXT,
    status         TEXT,
    volume_forecast FLOAT,
    UNIQUE (run_at, forecast_date)
);
CREATE INDEX IF NOT EXISTS idx_dfs_forecast_date ON daily_forecast_snapshots (forecast_date);
CREATE INDEX IF NOT EXISTS idx_dfs_run_at        ON daily_forecast_snapshots (run_at);

CREATE TABLE IF NOT EXISTS weekly_summary_snapshots (
    id                      SERIAL PRIMARY KEY,
    run_at                  TIMESTAMPTZ NOT NULL,
    week_monday             DATE,
    week_sunday             DATE,
    n_actual                INT,
    n_predicted             INT,
    weekly_avg              FLOAT,
    weekly_avg_millions     FLOAT,
    weekly_avg_std          FLOAT,
    weekly_avg_std_millions FLOAT,
    p_over_240              FLOAT,
    p_over_245              FLOAT,
    p_over_250              FLOAT,
    p_over_255              FLOAT,
    p_over_260              FLOAT,
    UNIQUE (run_at)
);
CREATE INDEX IF NOT EXISTS idx_wss_run_at ON weekly_summary_snapshots (run_at);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id              SERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ NOT NULL,
    ticker          TEXT        NOT NULL,
    strike_millions FLOAT,
    yes_bid_cents   FLOAT,
    yes_ask_cents   FLOAT,
    yes_mid_cents   FLOAT,
    market_prob     FLOAT,
    model_prob      FLOAT,
    edge            FLOAT,
    volume          FLOAT,
    open_interest   FLOAT
);
CREATE INDEX IF NOT EXISTS idx_ms_run_at ON market_snapshots (run_at);

CREATE TABLE IF NOT EXISTS positions_snapshots (
    id           SERIAL PRIMARY KEY,
    run_at       TIMESTAMPTZ NOT NULL,
    ticker       TEXT        NOT NULL,
    yes_shares   INT,
    no_shares    INT,
    avg_price    FLOAT,
    cost_dollars FLOAT
);
CREATE INDEX IF NOT EXISTS idx_ps_run_at ON positions_snapshots (run_at);

CREATE TABLE IF NOT EXISTS order_actions (
    id              SERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ NOT NULL,
    ticker          TEXT        NOT NULL,
    strike_millions FLOAT,
    action_type     TEXT,
    side            TEXT,
    price           FLOAT,
    shares          INT,
    pct             FLOAT,
    model_prob      FLOAT,
    market_prob     FLOAT,
    dry_run         BOOLEAN,
    cost_dollars    FLOAT
);
CREATE INDEX IF NOT EXISTS idx_oa_run_at ON order_actions (run_at);
CREATE INDEX IF NOT EXISTS idx_oa_ticker ON order_actions (ticker);
"""


def init_schema() -> None:
    """Create all tables. Safe to run repeatedly (uses CREATE IF NOT EXISTS)."""
    if not available():
        print("[db] POSTGRES_URL not set — skipping schema init")
        return
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)
        conn.commit()
    print("[db] Schema ready")


# ── Write helpers ─────────────────────────────────────────────────────────────

@_guard
def write_tsa_actuals(df: Any) -> None:
    """Upsert TSA daily actuals. df must have Date and Volume columns."""
    run_at = _run_at()
    with _conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO tsa_actuals (date, volume, recorded_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (date) DO UPDATE SET
                        volume      = EXCLUDED.volume,
                        recorded_at = EXCLUDED.recorded_at
                    """,
                    (row["Date"].date(), int(row["Volume"]), run_at),
                )
        conn.commit()
    print(f"[db] tsa_actuals: upserted {len(df)} rows")


@_guard
def write_daily_forecasts(df: Any) -> None:
    """Insert the weekly_forecast.csv rows for this run."""
    import pandas as pd
    run_at = _run_at()
    with _conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO daily_forecast_snapshots
                        (run_at, forecast_date, day_name, status, volume_forecast)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (run_at, forecast_date) DO NOTHING
                    """,
                    (
                        run_at,
                        pd.Timestamp(row["Date"]).date(),
                        row.get("day_name"),
                        row.get("status"),
                        float(row["volume"]) if pd.notna(row["volume"]) else None,
                    ),
                )
        conn.commit()
    print(f"[db] daily_forecast_snapshots: inserted {len(df)} rows for run_at={run_at}")


@_guard
def write_weekly_summary(summary_dict: dict) -> None:
    """Insert the weekly_summary.csv row for this run."""
    run_at = _run_at()
    import pandas as pd

    def _f(key: str):
        v = summary_dict.get(key)
        return None if (v is None or (isinstance(v, float) and pd.isna(v))) else float(v)

    def _i(key: str):
        v = summary_dict.get(key)
        return None if v is None else int(v)

    def _d(key: str):
        v = summary_dict.get(key)
        return pd.Timestamp(v).date() if v else None

    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO weekly_summary_snapshots (
                    run_at, week_monday, week_sunday,
                    n_actual, n_predicted,
                    weekly_avg, weekly_avg_millions,
                    weekly_avg_std, weekly_avg_std_millions,
                    p_over_240, p_over_245, p_over_250, p_over_255, p_over_260
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (run_at) DO NOTHING
                """,
                (
                    run_at,
                    _d("week_monday"), _d("week_sunday"),
                    _i("n_actual"),   _i("n_predicted"),
                    _f("weekly_avg"), _f("weekly_avg_millions"),
                    _f("weekly_avg_std"), _f("weekly_avg_std_millions"),
                    _f("p_over_2.4M"),  _f("p_over_2.45M"),
                    _f("p_over_2.5M"),  _f("p_over_2.55M"), _f("p_over_2.6M"),
                ),
            )
        conn.commit()
    print(f"[db] weekly_summary_snapshots: inserted row for run_at={run_at}")


@_guard
def write_market_snapshot(snap_df: Any) -> None:
    """Insert market snapshot rows for this run."""
    import pandas as pd
    run_at = _run_at()
    with _conn() as conn:
        with conn.cursor() as cur:
            for _, row in snap_df.iterrows():
                def _f(col):
                    v = row.get(col)
                    return None if (v is None or (isinstance(v, float) and pd.isna(v))) else float(v)
                cur.execute(
                    """
                    INSERT INTO market_snapshots (
                        run_at, ticker, strike_millions,
                        yes_bid_cents, yes_ask_cents, yes_mid_cents,
                        market_prob, model_prob, edge, volume, open_interest
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_at, row["ticker"], _f("strike_millions"),
                        _f("yes_bid_cents"), _f("yes_ask_cents"), _f("yes_mid_cents"),
                        _f("market_prob"), _f("model_prob"), _f("edge"),
                        _f("volume"), _f("open_interest"),
                    ),
                )
        conn.commit()
    print(f"[db] market_snapshots: inserted {len(snap_df)} rows for run_at={run_at}")


@_guard
def write_positions_snapshot(positions: list[dict]) -> None:
    """Insert current portfolio positions for this run."""
    run_at = _run_at()
    with _conn() as conn:
        with conn.cursor() as cur:
            for pos in positions:
                if "KXTSAW" not in pos.get("ticker", ""):
                    continue
                avg = pos.get("avg_price")
                cur.execute(
                    """
                    INSERT INTO positions_snapshots
                        (run_at, ticker, yes_shares, no_shares, avg_price, cost_dollars)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_at, pos["ticker"],
                        pos.get("yes", 0), pos.get("no", 0),
                        float(avg) if avg is not None else None,
                        float(pos.get("cost_dollars", 0)),
                    ),
                )
        conn.commit()
    print(f"[db] positions_snapshots: inserted {len(positions)} rows for run_at={run_at}")


@_guard
def write_order_actions(actions: list[dict]) -> None:
    """Insert order action rows for this run."""
    import pandas as pd
    run_at = _run_at()

    def _strike(ticker: str):
        for part in ticker.split("-"):
            if part.startswith("A"):
                try: return float(part[1:])
                except ValueError: pass
        return None

    def _f(v): return None if (v is None or (isinstance(v, float) and pd.isna(v))) else float(v)

    with _conn() as conn:
        with conn.cursor() as cur:
            for a in actions:
                price  = _f(a.get("price"))
                shares = int(a["shares"]) if a.get("shares") is not None else None
                cost   = round(price * shares, 4) if (price and shares) else None
                dry_run_val = str(a.get("dry_run", "True")).strip().lower()
                cur.execute(
                    """
                    INSERT INTO order_actions (
                        run_at, ticker, strike_millions, action_type, side,
                        price, shares, pct, model_prob, market_prob, dry_run, cost_dollars
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_at, a.get("ticker", ""),
                        _strike(str(a.get("ticker", ""))),
                        a.get("type"), a.get("side"),
                        price, shares,
                        _f(a.get("pct")), _f(a.get("model_prob")), _f(a.get("market_prob")),
                        dry_run_val == "false",
                        cost,
                    ),
                )
        conn.commit()
    print(f"[db] order_actions: inserted {len(actions)} rows for run_at={run_at}")


# ── Read helpers (used by dashboard) ─────────────────────────────────────────

def query(sql: str, params=()) -> list[dict]:
    """Run a SELECT and return list of dicts. Returns [] if DB unavailable."""
    if not available():
        return []
    try:
        with _conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        print(f"[db] query failed: {exc}", file=sys.stderr)
        return []


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"POSTGRES_URL={'set' if POSTGRES_URL else 'NOT SET'}")
    print(f"psycopg2={'available' if _HAVE_PSYCOPG2 else 'NOT INSTALLED'}")
    if available():
        init_schema()
