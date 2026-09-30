#!/usr/bin/env python3
"""migrate_consolidate.py — one-shot backfill into the consolidated tables.

Idempotent. Safe to re-run. Reads from existing stores and upserts into:
    predictions    ← daily_forecast_snapshots + weekly_summary_snapshots
    fills          ← output_kalshi/trades.db fills
    settlements    ← output_kalshi/trades.db market_results

Source tables are NOT modified or deleted. Run again any time the underlying
data changes — upserts keep new rows in sync.

Usage:
    python migrate_consolidate.py            # all sources
    python migrate_consolidate.py --predictions
    python migrate_consolidate.py --fills
    python migrate_consolidate.py --settlements
    python migrate_consolidate.py --dry-run  # report only, no writes
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import db

ROOT = Path(__file__).resolve().parent
SQLITE_PATH = ROOT / "output_kalshi" / "trades.db"


# ── Predictions backfill ─────────────────────────────────────────────────────
def _daily_rows() -> list[dict]:
    """Pull daily snapshots from daily_forecast_snapshots."""
    rows = db.query(
        """
        SELECT run_at, forecast_date, day_name, status, volume_forecast
        FROM daily_forecast_snapshots
        """,
        (),
    )
    return [
        {
            "run_at":           r["run_at"],          # used by upsert override below
            "forecast_date":    r["forecast_date"],
            "market_type":      "daily",
            "day_name":         r["day_name"],
            "status":           r["status"],
            "predicted_volume": r["volume_forecast"],
        }
        for r in rows
    ]


def _weekly_rows() -> list[dict]:
    """Pull weekly snapshots from weekly_summary_snapshots.

    Uses week_sunday as the forecast_date (the canonical event date for KXTSAW).
    """
    rows = db.query(
        """
        SELECT run_at, week_sunday, weekly_avg, weekly_avg_std
        FROM weekly_summary_snapshots
        WHERE week_sunday IS NOT NULL
        """,
        (),
    )
    return [
        {
            "run_at":           r["run_at"],
            "forecast_date":    r["week_sunday"],
            "market_type":      "weekly",
            "predicted_volume": r["weekly_avg"],
            "sigma":            r["weekly_avg_std"],
        }
        for r in rows
    ]


def migrate_predictions(dry_run: bool = False) -> None:
    daily  = _daily_rows()
    weekly = _weekly_rows()
    all_rows = daily + weekly
    print(f"[migrate] predictions: {len(daily)} daily + {len(weekly)} weekly = {len(all_rows)}")
    if dry_run or not all_rows:
        return
    # write_predictions uses the *current* run_at from env. For a backfill,
    # the source's run_at is the truth — go through psycopg2 directly.
    import os
    from datetime import datetime
    import psycopg2.extras as _pgex
    with db._conn() as conn:
        with conn.cursor() as cur:
            for r in all_rows:
                cur.execute(
                    """
                    INSERT INTO predictions
                        (run_at, forecast_date, market_type, day_name, status,
                         regime, predicted_volume, sigma, ci90_low, ci90_high)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_at, forecast_date, market_type) DO UPDATE SET
                        day_name         = EXCLUDED.day_name,
                        status           = EXCLUDED.status,
                        predicted_volume = EXCLUDED.predicted_volume,
                        sigma            = COALESCE(EXCLUDED.sigma, predictions.sigma)
                    """,
                    (
                        r["run_at"], r["forecast_date"], r["market_type"],
                        r.get("day_name"), r.get("status"), r.get("regime"),
                        r.get("predicted_volume"), r.get("sigma"),
                        r.get("ci90_low"), r.get("ci90_high"),
                    ),
                )
        conn.commit()
    print(f"[migrate] predictions: upserted {len(all_rows)} rows")


# ── Fills + settlements backfill (from SQLite) ───────────────────────────────
def _sqlite_fills() -> list[dict]:
    if not SQLITE_PATH.exists():
        print(f"[migrate] no SQLite at {SQLITE_PATH}, skipping fills")
        return []
    conn = sqlite3.connect(SQLITE_PATH); conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM fills").fetchall()]
    finally:
        conn.close()


def _sqlite_settlements() -> list[dict]:
    if not SQLITE_PATH.exists():
        return []
    conn = sqlite3.connect(SQLITE_PATH); conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM market_results").fetchall()]
    finally:
        conn.close()


def migrate_fills(dry_run: bool = False) -> None:
    fills = _sqlite_fills()
    print(f"[migrate] fills: {len(fills)} rows in SQLite")
    if dry_run or not fills:
        return
    n = db.write_fills(fills)
    print(f"[migrate] fills: {n} rows upserted in Postgres")


def migrate_settlements(dry_run: bool = False) -> None:
    rows = _sqlite_settlements()
    print(f"[migrate] settlements: {len(rows)} rows in SQLite")
    if dry_run or not rows:
        return
    n = db.write_settlements(rows)
    print(f"[migrate] settlements: {n} rows upserted in Postgres")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", action="store_true")
    p.add_argument("--fills",       action="store_true")
    p.add_argument("--settlements", action="store_true")
    p.add_argument("--dry-run",     action="store_true")
    args = p.parse_args()

    if not db.available():
        print("[migrate] POSTGRES_URL not set — abort.", file=sys.stderr)
        sys.exit(1)

    all_ = not (args.predictions or args.fills or args.settlements)
    if all_ or args.predictions: migrate_predictions(dry_run=args.dry_run)
    if all_ or args.fills:       migrate_fills(dry_run=args.dry_run)
    if all_ or args.settlements: migrate_settlements(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
