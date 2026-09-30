#!/usr/bin/env python3
"""jarvis.py — Mini Jarvis morning/evening brief for TSA trading.

Pulls structured data from `backlog` (Postgres-backed) and asks Claude to
narrate a short, sir-style briefing.

Env:
    ANTHROPIC_API_KEY  — required

Usage:
    python jarvis.py                    # auto: today = system date
    python jarvis.py --date 2026-06-12  # treat the given date as 'today'
    python jarvis.py --raw              # print the structured data dict, skip the LLM
    python jarvis.py --evening          # force greeting_time=evening
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

import anthropic

import backlog
import db

EDGE_VOICE = os.environ.get("EDGE_VOICE", "en-US-ChristopherNeural")

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 700

SYSTEM_PROMPT = """You are Mini Jarvis, a concise trading assistant for the user's \
TSA passenger volume Kalshi prediction system. You produce a short, natural \
morning/evening briefing in a respectful "sir" style. Easy to read, confident, \
not robotic. Mention only sections that apply based on the data provided.

Rules:
1. Start with "Good morning sir," or "Good evening sir," based on greeting_time.
2. On Tuesday–Friday:
   - Recap yesterday's daily prediction: predicted, actual, error.
   - Recap thresholds we bet, amounts, win/loss, P&L.
   - Emphasize the main bet (model %, Kalshi %, edge).
   - Discuss today's prediction, today's bets, capital deployed, best edge.
   - Note how yesterday's actual moved the weekly average prediction.
3. On Monday:
   - Recap Friday's daily market.
   - Recap weekly market resolution.
   - Mention total weekly P&L, final weekly average, and Fri/Sat/Sun errors.
   - Preview the coming week: our prediction, Kalshi's implied prediction, \
current bets, main threshold, side, capital deployed, best edge.
   - If a Monday daily market exists, include it briefly.
4. On Saturday/Sunday: if no major updates, keep it very short.
5. Tone: smart personal assistant. Honest but encouraging. Use phrases like \
"solid result," "not perfect, but acceptable," "this was the important one," \
"our edge is still meaningful," "we should be careful here." Don't overhype losses.
6. Under 250 words unless weekly resolution is included.
7. Numbers are heard. Round aggressively: volumes to nearest 0.1M, errors to \
nearest 5k or 10k, P&L to nearest dollar, edges to nearest whole percent. \
No markdown, no bullets, no headings, no emoji. One number per sentence, max.
8. If a field is null or missing, briefly acknowledge it ("yesterday's actual \
hasn't published yet") rather than inventing numbers."""


# ── Time helpers ─────────────────────────────────────────────────────────────
def _greeting_time(force: str | None = None) -> str:
    if force in {"morning", "evening"}:
        return force
    h = datetime.now().hour
    return "morning" if h < 14 else "evening"


def _next_sunday(d: date) -> date:
    return d + timedelta(days=(6 - d.weekday()) % 7)


def _prev_sunday(d: date) -> date:
    """Most recent past Sunday (yesterday if d is Monday)."""
    return d - timedelta(days=(d.weekday() + 1) % 7 or 7)


# ── Data pulls ───────────────────────────────────────────────────────────────
def _recent_daily_errors(today: date, n: int = 3) -> list[dict]:
    """Last n resolved daily predictions: forecast_date, predicted, actual, error_k."""
    rows = db.query(
        """
        WITH latest AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY forecast_date ORDER BY run_at DESC) rn
            FROM predictions WHERE market_type = 'daily' AND forecast_date < %s
        )
        SELECT p.forecast_date, p.day_name, p.predicted_volume, a.volume AS actual
        FROM latest p
        JOIN tsa_actuals a ON a.date = p.forecast_date
        WHERE p.rn = 1
        ORDER BY p.forecast_date DESC LIMIT %s
        """,
        (today, n),
    )
    out = []
    for r in rows:
        pred = float(r["predicted_volume"]) if r["predicted_volume"] is not None else None
        act = float(r["actual"])
        err = pred - act if pred is not None else None
        out.append({
            "date":      r["forecast_date"].isoformat() if hasattr(r["forecast_date"], "isoformat") else str(r["forecast_date"]),
            "day":       r["day_name"],
            "predicted": pred,
            "actual":    act,
            "error_k":   round(err / 1000.0, 1) if err is not None else None,
        })
    return out


def _weekly_prediction_now(target_sunday: date) -> dict | None:
    rows = db.query(
        """
        SELECT run_at, predicted_volume, sigma, regime
        FROM predictions
        WHERE forecast_date = %s AND market_type = 'weekly'
        ORDER BY run_at DESC LIMIT 1
        """,
        (target_sunday,),
    )
    if not rows:
        return None
    r = rows[0]
    return {
        "predicted_volume": float(r["predicted_volume"]) if r["predicted_volume"] is not None else None,
        "sigma":            float(r["sigma"]) if r["sigma"] is not None else None,
        "regime":           r.get("regime"),
    }


def _kalshi_implied_weekly(target_sunday: date) -> float | None:
    """Estimate Kalshi-implied weekly volume by finding the strike where the
    YES market_prob is closest to 0.5 across the latest snapshot."""
    yy = target_sunday.strftime("%y%b%d").upper()
    ticker_like = f"KXTSAW-{yy}-%"
    rows = db.query(
        """
        WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY run_at DESC) rn
            FROM market_snapshots WHERE ticker LIKE %s
        )
        SELECT strike_millions, market_prob FROM ranked
        WHERE rn = 1 AND market_prob IS NOT NULL AND strike_millions IS NOT NULL
        ORDER BY strike_millions
        """,
        (ticker_like,),
    )
    if len(rows) < 2:
        return None
    # Linear interpolate strike where market_prob crosses 0.5
    for a, b in zip(rows, rows[1:]):
        pa, pb = float(a["market_prob"]), float(b["market_prob"])
        sa, sb = float(a["strike_millions"]), float(b["strike_millions"])
        if (pa - 0.5) * (pb - 0.5) <= 0 and pa != pb:
            frac = (0.5 - pa) / (pb - pa)
            return round(sa + frac * (sb - sa), 3) * 1_000_000
    return None


def _weekly_open_positions(target_sunday: date) -> list[dict]:
    yy = target_sunday.strftime("%y%b%d").upper()
    prefix = f"KXTSAW-{yy}-"
    return [p for p in backlog.open_positions_with_edge() if p["ticker"].startswith(prefix)]


def _weekly_main(positions: list[dict]) -> dict | None:
    if not positions:
        return None
    return max(positions, key=lambda p: p["cost_dollars"] or 0)


def _weekly_resolution(target_sunday: date) -> dict | None:
    """Settled trades for the weekly tickers expiring on target_sunday."""
    yy = target_sunday.strftime("%y%b%d").upper()
    ticker_like = f"KXTSAW-{yy}-%"
    rows = db.query(
        """
        SELECT f.ticker, f.side, st.result,
               SUM(CASE WHEN f.action='buy'  THEN f.count ELSE 0 END) AS buy_shares,
               SUM(CASE WHEN f.action='sell' THEN f.count ELSE 0 END) AS sell_shares,
               SUM(CASE WHEN f.action='buy' THEN
                   CASE WHEN f.side='yes' THEN f.yes_price_cents*f.count
                        ELSE f.no_price_cents*f.count END ELSE 0 END) AS buy_cost_cents
        FROM fills f
        LEFT JOIN settlements st ON st.ticker = f.ticker
        WHERE f.ticker LIKE %s
        GROUP BY f.ticker, f.side, st.result
        """,
        (ticker_like,),
    )
    if not rows:
        return None
    total_pnl = 0.0
    n_settled = 0
    main_result = None
    main_cost = -1.0
    for r in rows:
        buy = int(r["buy_shares"] or 0)
        if buy <= 0:
            continue
        net = buy - int(r["sell_shares"] or 0)
        cost = (r["buy_cost_cents"] or 0) / 100.0
        if not r["result"]:
            continue
        n_settled += 1
        won = (r["result"] == r["side"])
        pnl = (net - cost) if won else (-cost)
        total_pnl += pnl
        if cost > main_cost:
            main_cost = cost
            main_result = {"ticker": r["ticker"], "side": r["side"],
                           "won": won, "pnl_dollars": round(pnl, 2)}
    if n_settled == 0:
        return None
    return {
        "week_sunday":      target_sunday.isoformat(),
        "n_settled_trades": n_settled,
        "total_pnl":        round(total_pnl, 2),
        "main_bet_result":  main_result,
    }


def _weekly_actual_avg(target_sunday: date) -> float | None:
    """Mean TSA volume for Mon..Sun of the target week, if all 7 days are known."""
    monday = target_sunday - timedelta(days=6)
    rows = db.query(
        "SELECT volume FROM tsa_actuals WHERE date BETWEEN %s AND %s",
        (monday, target_sunday),
    )
    if len(rows) < 7:
        return None
    return sum(int(r["volume"]) for r in rows) / 7.0


def _today_capital_deployed(positions: list[dict]) -> float:
    return round(sum((p["cost_dollars"] or 0) for p in positions), 2)


def _is_daily_for(today: date, ticker: str) -> bool:
    yy = today.strftime("%y%b%d").upper()
    return ticker.startswith(f"KXTRUFTSA-{yy}-")


# ── Brief assembly ───────────────────────────────────────────────────────────
def build_brief_data(target: date, greeting: str) -> dict:
    weekday = target.strftime("%A")
    this_sunday = _next_sunday(target)

    yest = backlog.yesterday_brief(target)
    today_fc = backlog.today_brief(target)

    open_pos = backlog.open_positions_with_edge()
    today_daily_pos = [p for p in open_pos if _is_daily_for(target, p["ticker"])]
    weekly_pos = [p for p in open_pos if p["ticker"].startswith(f"KXTSAW-{this_sunday.strftime('%y%b%d').upper()}-")]

    daily_main_bet = max(yest["trades"], key=lambda t: t["shares"] or 0) if yest["trades"] else None

    weekly_change = backlog.forecast_change(this_sunday, "weekly")
    weekly_now = _weekly_prediction_now(this_sunday)
    kalshi_weekly = _kalshi_implied_weekly(this_sunday)
    weekly_main = _weekly_main(weekly_pos)

    data: dict = {
        "current_day":   weekday,
        "greeting_time": greeting,
        "today_date":    target.isoformat(),

        # Yesterday recap (Tue–Fri)
        "yesterday_daily_prediction": yest["predicted_volume"],
        "yesterday_actual":           yest["actual_volume"],
        "yesterday_error":            yest["error"],
        "daily_bets":                 yest["trades"],
        "daily_main_bet":             daily_main_bet,
        "daily_pnl":                  yest["realized_pnl"],

        # Today
        "today_daily_prediction":    today_fc["predicted_volume"],
        "today_daily_bets":          today_daily_pos,
        "today_capital_deployed":    _today_capital_deployed(today_daily_pos),
        "today_edges_vs_kalshi":     today_fc["candidates"][:5],

        # Weekly average shift driven by yesterday's actual
        "weekly_prediction_before":  weekly_change["prior_prediction"] if weekly_change else None,
        "weekly_prediction_after":   weekly_now["predicted_volume"] if weekly_now else None,

        # Weekly market state
        "kalshi_weekly_prediction":              kalshi_weekly,
        "weekly_bets":                           weekly_pos,
        "weekly_capital_deployed":               _today_capital_deployed(weekly_pos),
        "weekly_main_market_probability_model":  weekly_main["model_prob"] if weekly_main else None,
        "weekly_main_market_probability_kalshi": weekly_main["market_prob"] if weekly_main else None,
        "weekly_edge":                           weekly_main["edge"] if weekly_main else None,
    }

    # Monday-only fields
    if weekday == "Monday":
        prev_sun = target - timedelta(days=1)
        coming_sun = _next_sunday(target + timedelta(days=1))
        coming_pos = _weekly_open_positions(coming_sun)
        coming_main = _weekly_main(coming_pos)
        coming_now = _weekly_prediction_now(coming_sun)

        last3 = _recent_daily_errors(target, n=3)
        err_by_day = {row["day"]: row["error_k"] for row in last3 if row["day"]}

        data.update({
            "weekly_resolution_result": _weekly_resolution(prev_sun),
            "weekly_actual_average":    _weekly_actual_avg(prev_sun),
            "friday_error":             err_by_day.get("Friday"),
            "saturday_error":           err_by_day.get("Saturday"),
            "sunday_error":             err_by_day.get("Sunday"),
            "coming_week_prediction":   coming_now["predicted_volume"] if coming_now else None,
            "coming_week_kalshi_prediction": _kalshi_implied_weekly(coming_sun),
            "coming_week_bets":         coming_pos,
            "coming_week_main_bet":     coming_main,
        })

    return data


# ── LLM narration ────────────────────────────────────────────────────────────
def narrate(data: dict) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": json.dumps(data, indent=2, default=str),
        }],
    )
    return next(b.text for b in response.content if b.type == "text").strip()


# ── TTS ──────────────────────────────────────────────────────────────────────
def speak(text: str) -> None:
    try:
        import asyncio
        import edge_tts
    except ImportError:
        print("[jarvis] edge-tts not installed — run: pip install edge-tts", file=sys.stderr)
        return

    async def _speak() -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        try:
            await edge_tts.Communicate(text, EDGE_VOICE).save(path)
            import subprocess
            subprocess.run(
                ["gst-play-1.0", "--no-interactive", path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            Path(path).unlink(missing_ok=True)

    asyncio.run(_speak())


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date",     help="treat this date as 'today' (YYYY-MM-DD)")
    p.add_argument("--evening",  action="store_true", help="force greeting_time=evening")
    p.add_argument("--morning",  action="store_true", help="force greeting_time=morning")
    p.add_argument("--raw",      action="store_true",
                   help="print the structured data dict instead of calling Claude")
    p.add_argument("--no-speak", action="store_true",
                   help="print the narration but skip TTS playback")
    args = p.parse_args()

    target = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date else date.today()
    )
    forced = "evening" if args.evening else ("morning" if args.morning else None)
    data = build_brief_data(target, _greeting_time(forced))

    if args.raw:
        print(json.dumps(data, indent=2, default=str))
        return

    text = narrate(data)
    print(text)
    if not args.no_speak:
        speak(text)


if __name__ == "__main__":
    main()
