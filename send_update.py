#!/usr/bin/env python3
"""
send_update.py
==============
Sends a daily TSA Kalshi trading update via Telegram.

Reads outputs from:
  - get_new_tsa.py        → data/tsa_volume.csv
  - autogluon_predict.py  → output_autogluon_predict/weekly_{summary,forecast}.csv
  - kalshi.py             → output_kalshi/market_snapshot_latest.csv
                            Kalshi API: /portfolio/positions, /portfolio/orders

Environment (set in .env or shell):
  TELEGRAM_BOT_TOKEN     Telegram bot API token
  TELEGRAM_CHAT_ID       Target chat/user ID
  KALSHI_BANKROLL        Total bankroll dollars (default: 250)
  KALSHI_AUTO_TRADING    "true"/"false" — whether live trading is enabled (default: true)

Usage:
  python send_update.py            # build + send
  python send_update.py --dry-run  # print message without sending
"""

from __future__ import annotations

import html
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────
BASE             = Path(__file__).parent
TSA_DATA         = BASE / "data/tsa_volume.csv"
WEEKLY_SUMMARY   = BASE / "output_autogluon_predict/weekly_summary.csv"
WEEKLY_FORECAST  = BASE / "output_autogluon_predict/weekly_forecast.csv"
PREV_SUMMARY     = BASE / "output_autogluon_predict/prev_weekly_summary.csv"
PREV_FORECAST    = BASE / "output_autogluon_predict/prev_weekly_forecast.csv"
SNAPSHOT            = BASE / "output_kalshi/market_snapshot_latest.csv"
PREDICTION_HISTORY  = BASE / "output_autogluon_predict/prediction_history.csv"

# ── Config ─────────────────────────────────────────────────────────────────
_bankroll_default = 250.0
BANKROLL     = float(os.environ.get("KALSHI_BANKROLL", str(_bankroll_default)))
AUTO_TRADING = os.environ.get("KALSHI_AUTO_TRADING", "true").lower() != "false"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT_ID", "")

THRESHOLDS = [2.30, 2.35, 2.40, 2.45, 2.50, 2.55, 2.60, 2.65, 2.70, 2.75, 2.80]
# Show edge only when |edge| is at least this many percentage points
EDGE_THRESHOLD_PCT = 3.0


# ── Formatting helpers ─────────────────────────────────────────────────────
def _fmt_vol(n: float | None) -> str:
    if n is None:
        return "N/A"
    return f"{n / 1e6:.2f}M"


def _fmt_k(delta: float | None) -> str:
    if delta is None:
        return "N/A"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{round(delta / 1000)}k"


def _strike_label(ticker: str) -> str:
    """KXTSAW-26MAY17-A2.50 → 2.50M"""
    for part in ticker.split("-"):
        if part.startswith("A"):
            try:
                return f"{float(part[1:]):.2f}M"
            except ValueError:
                pass
    return ticker


# ── Section: Yesterday ─────────────────────────────────────────────────────
def section_yesterday() -> str:
    actual_vol: float | None  = None
    actual_date: str | None   = None

    if datetime.today().weekday() == 0:
        return ""

    if TSA_DATA.exists():
        df = pd.read_csv(TSA_DATA, header=None, names=["date", "volume"])
        last = df.dropna().iloc[-1]
        actual_date = str(last["date"])
        actual_vol  = float(last["volume"])

    pred_vol: float | None = None
    if actual_date:
        # Use prediction_history for the original model forecast (before actuals overwrote it)
        if PREDICTION_HISTORY.exists():
            hist = pd.read_csv(PREDICTION_HISTORY)
            hist["_date"] = pd.to_datetime(hist["Date"], format="mixed").dt.strftime("%Y-%m-%d")
            match = hist[hist["_date"] == actual_date].sort_values("run_date")
            if not match.empty:
                pred_vol = float(match.iloc[0]["predicted_volume"])

        # Fall back to forecast files, but only rows that were still "predicted"
        if pred_vol is None:
            for fc_path in [PREV_FORECAST, WEEKLY_FORECAST]:
                if not fc_path.exists():
                    continue
                df = pd.read_csv(fc_path)
                match = df[(df["Date"] == actual_date) & (df["status"] == "predicted")]
                if not match.empty:
                    pred_vol = float(match.iloc[0]["volume"])
                    break

    error_str = _fmt_k(actual_vol - pred_vol) + " passengers" if (actual_vol and pred_vol) else "N/A"

    return "\n".join([
        "Yesterday:",
        f"Actual TSA: {_fmt_vol(actual_vol)}",
        f"Yesterday prediction: {_fmt_vol(pred_vol)}",
        f"Error: {error_str}",
    ])


# ── Section: Weekly forecast ───────────────────────────────────────────────
def section_weekly_forecast() -> str:
    today_avg: float | None = None
    prev_avg:  float | None = None

    if WEEKLY_SUMMARY.exists():
        today_avg = float(pd.read_csv(WEEKLY_SUMMARY).iloc[0]["weekly_avg_millions"])
    if PREV_SUMMARY.exists():
        prev_avg = float(pd.read_csv(PREV_SUMMARY).iloc[0]["weekly_avg_millions"])

    change_str = _fmt_k((today_avg - prev_avg) * 1e6) if (today_avg and prev_avg) else "N/A"
    prev_str   = f"{prev_avg:.3f}M"  if prev_avg  is not None else "N/A"
    today_str  = f"{today_avg:.3f}M" if today_avg is not None else "N/A"

    if datetime.today().weekday() == 0:
        return "\n".join([
            f"Weekly forecast: {today_str}",
        ])   

    return "\n".join([
        "Weekly forecast:",
        f"Yesterday weekly avg forecast: {prev_str}",
        f"Today weekly avg forecast: {today_str}",
        f"Change: {change_str}",
    ])


# ── Section: Weekly forecast table ────────────────────────────────────────────
def section_weekly_forecast_table() -> str:
    if not WEEKLY_FORECAST.exists():
        return ""

    df = pd.read_csv(WEEKLY_FORECAST)
    col_w = [11, 11, 10]
    header = (
        f"{'Day':<{col_w[0]}}"
        f"{'Status':<{col_w[1]}}"
        f"{'Volume':>{col_w[2]}}"
    )
    sep = "─" * sum(col_w)
    rows = [header, sep]

    for _, row in df.iterrows():
        day   = str(row["day_name"])[:col_w[0] - 1]
        stat  = str(row["status"])[:col_w[1] - 1]
        vol   = f"{float(row['volume']) / 1e6:.3f}M"
        rows.append(
            f"{day:<{col_w[0]}}"
            f"{stat:<{col_w[1]}}"
            f"{vol:>{col_w[2]}}"
        )

    return "Daily forecast:\n<pre>" + html.escape("\n".join(rows)) + "</pre>"


# ── Section: Model probabilities YES ───────────────────────────────────────────
def section_model_prob_yes() -> str:
    if not WEEKLY_SUMMARY.exists() or not SNAPSHOT.exists():
        return "Model probabilities: N/A"

    summary = pd.read_csv(WEEKLY_SUMMARY).iloc[0]
    snap    = pd.read_csv(SNAPSHOT)


    col_w = [9, 12, 13, 8]
    header = (
        f"{'Threshold':<{col_w[0]}}"
        f"{'Model Over':>{col_w[1]}}"
        f"{'Kalshi Over':>{col_w[2]}}"
        f"{'Edge':>{col_w[3]}}"
    )
    sep  = "─" * len(header)
    rows = [header, sep]

    for t in THRESHOLDS:
        col     = f"p_over_{t}M"
        model_p = float(summary.get(col, float("nan")))
        if pd.isna(model_p):
            continue

        snap_row = snap[abs(snap["strike_millions"] - t) < 0.001]
        market_p = float(snap_row.iloc[0]["market_prob"]) if not snap_row.empty else float("nan")

        mo = f"{model_p * 100:.1f}%"
        ko = f"{market_p * 100:.1f}%" if not pd.isna(market_p) else "N/A"

        if pd.isna(market_p):
            edge_str = "N/A"
        else:
            edge = (model_p - market_p) * 100
            if (edge) < EDGE_THRESHOLD_PCT:
                edge_str = "Skip"
            else:
                edge_str = f"+{edge:.1f}%" if edge > 0 else f"{edge:.1f}%"

        rows.append(
            f"{f'{t:.2f}M':<{col_w[0]}}"
            f"{mo:>{col_w[1]}}"
            f"{ko:>{col_w[2]}}"
            f"{edge_str:>{col_w[3]}}"
        )

    return "Model probabilities Over:\n<pre>" + html.escape("\n".join(rows)) + "</pre>"

# ── Section: Model probabilities NO───────────────────────────────────────────
def section_model_prob_no() -> str:
    if not WEEKLY_SUMMARY.exists() or not SNAPSHOT.exists():
        return "Model probabilities: N/A"

    summary = pd.read_csv(WEEKLY_SUMMARY).iloc[0]
    snap    = pd.read_csv(SNAPSHOT)


    col_w = [9, 13, 14, 8]
    header = (
        f"{'Threshold':<{col_w[0]}}"
        f"{'Model Under':>{col_w[1]}}"
        f"{'Kalshi Under':>{col_w[2]}}"
        f"{'Edge':>{col_w[3]}}"
    )
    sep  = "─" * len(header)
    rows = [header, sep]

    for t in THRESHOLDS:
        col     = f"p_over_{t}M"
        model_p = 1 - float(summary.get(col, float("nan")))
        if pd.isna(model_p):
            continue

        snap_row = snap[abs(snap["strike_millions"] - t) < 0.001]
        market_p = 1 - float(snap_row.iloc[0]["market_prob"]) if not snap_row.empty else float("nan")

        mu = f"{(model_p) * 100:.1f}%"
        ku = f"{(market_p) * 100:.1f}%" if not pd.isna(market_p) else "N/A"

        if pd.isna(market_p):
            edge_str = "N/A"
        else:
            edge = (model_p - market_p) * 100
            if (edge) < EDGE_THRESHOLD_PCT:
                edge_str = "Skip"
            else:
                edge_str = f"+{edge:.1f}%" if edge > 0 else f"{edge:.1f}%"

        rows.append(
            f"{f'{t:.2f}M':<{col_w[0]}}"
            f"{mu:>{col_w[1]}}"
            f"{ku:>{col_w[2]}}"
            f"{edge_str:>{col_w[3]}}"
        )

    return "Model probabilities Under:\n<pre>" + html.escape("\n".join(rows)) + "</pre>"


# ── Live portfolio data ────────────────────────────────────────────────────
def _fetch_portfolio() -> dict:
    """Call kalshi.fetch_portfolio_summary(); return empty structure on failure."""
    try:
        sys.path.insert(0, str(BASE))
        from kalshi import fetch_portfolio_summary  # noqa: PLC0415
        return fetch_portfolio_summary()
    except Exception as exc:
        print(f"[warn] Could not fetch live portfolio: {exc}", file=sys.stderr)
        return {"positions": [], "open_orders": []}


# ── Section: Positions ─────────────────────────────────────────────────────
def _current_price(ticker: str, side: str, snap: pd.DataFrame) -> float | None:
    """Return current mid price in dollars for a YES or NO position."""
    row = snap[snap["ticker"] == ticker]
    if row.empty:
        return None
    mid_yes = row.iloc[0].get("yes_mid_cents")
    if pd.isna(mid_yes):
        return None
    if side == "yes":
        return float(mid_yes) / 100.0
    return (100.0 - float(mid_yes)) / 100.0


def section_positions(portfolio: dict) -> tuple[str, float]:
    """Returns (formatted text, total cost basis in dollars)."""
    lines    = ["Positions:"]
    exposure = 0.0

    snap = pd.read_csv(SNAPSHOT) if SNAPSHOT.exists() else pd.DataFrame()

    positions = [p for p in portfolio.get("positions", []) if "KXTSAW" in p["ticker"]]
    if not positions:
        lines.append("None")
        return "\n".join(lines), 0.0

    col_w = [7, 6, 5, 6, 6, 10]
    header = (
        f"{'Strike':<{col_w[0]}}"
        f"{'Side':<{col_w[1]}}"
        f"{'Qty':>{col_w[2]}}"
        f"{'Avg':>{col_w[3]}}"
        f"{'Now':>{col_w[4]}}"
        f"{'Invested':>{col_w[5]}}"
    )
    sep = "─" * sum(col_w)
    rows = [header, sep]

    for pos in sorted(positions, key=lambda p: p["ticker"]):
        label = _strike_label(pos["ticker"])
        avg   = pos.get("avg_price")
        cost  = pos.get("cost_dollars", 0.0)
        avg_str  = f"${avg:.2f}" if avg is not None else "N/A"
        cost_str = f"${cost:.2f}"

        for side, shares in [("yes", pos["yes"]), ("no", pos["no"])]:
            if shares == 0:
                continue
            side_label = "Over" if side == "yes" else "Under"
            curr = _current_price(pos["ticker"], side, snap)
            curr_str = f"${curr:.2f}" if curr is not None else "N/A"
            rows.append(
                f"{label:<{col_w[0]}}"
                f"{side_label:<{col_w[1]}}"
                f"{shares:>{col_w[2]}}"
                f"{avg_str:>{col_w[3]}}"
                f"{curr_str:>{col_w[4]}}"
                f"{cost_str:>{col_w[5]}}"
            )
            exposure += cost

    lines.append("<pre>" + html.escape("\n".join(rows)) + "</pre>")
    return "\n".join(lines), round(exposure, 2)


# ── Section: Open orders ───────────────────────────────────────────────────
def section_orders(portfolio: dict) -> str:
    # Minimum price to show — filters out sub-nickel resting orders
    MIN_PRICE = 0.05

    orders = [o for o in portfolio.get("open_orders", []) if o["price"] >= MIN_PRICE]

    lines = ["Open orders:"]
    if not orders:
        lines.append("None")
        return "\n".join(lines)

    grouped: dict[tuple[str, str], dict[float, int]] = {}
    for o in orders:
        key = (o["ticker"], o["side"])
        price = round(o["price"], 2)
        bucket = grouped.setdefault(key, {})
        bucket[price] = bucket.get(price, 0) + o["remaining"]

    for (ticker, side), price_map in sorted(grouped.items()):
        side_lbl = "YES" if side == "yes" else "NO"
        lines.append(f"Buy {side_lbl} {_strike_label(ticker)}:")
        for price, shares in sorted(price_map.items(), reverse=True):
            lines.append(f"  - {shares} shares @ {price:.2f}")

    return "\n".join(lines)


# ── Section: Risk ──────────────────────────────────────────────────────────
def section_risk(exposure: float) -> str:
    return "\n".join([
        "Risk:",
        f"Bankroll: ${BANKROLL:.0f}",
        f"Open exposure: ${exposure:.2f}",
        f"Auto-trading: {'ON' if AUTO_TRADING else 'OFF'}",
    ])


# ── Message builder ────────────────────────────────────────────────────────
def build_message() -> str:
    today     = datetime.now(timezone.utc).strftime("%B %-d, %Y")
    portfolio = _fetch_portfolio()

    pos_text, exposure = section_positions(portfolio)

    parts = [
        f"<b>TSA Kalshi Daily Update — {today}</b>",
        "",
        section_yesterday(),
        "",
        section_weekly_forecast(),
        "",
        section_weekly_forecast_table(),
        "",
        section_model_prob_yes(),
        "",
        section_model_prob_no(),
        "",
        pos_text,
        "",
        section_orders(portfolio),
        "",
        section_risk(exposure),
    ]
    return "\n".join(parts)


# ── Telegram ───────────────────────────────────────────────────────────────
def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        raise EnvironmentError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment or .env"
        )
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT, "text": text, "parse_mode": "HTML"},
        timeout=15,
    )
    resp.raise_for_status()


def save_prev_snapshots() -> None:
    """Copy today's summary/forecast → prev_* so tomorrow's run can compare."""
    if WEEKLY_SUMMARY.exists():
        shutil.copy(WEEKLY_SUMMARY, PREV_SUMMARY)
    if WEEKLY_FORECAST.exists():
        shutil.copy(WEEKLY_FORECAST, PREV_FORECAST)


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE / ".env")
    except ImportError:
        pass

    dry_run = "--dry-run" in sys.argv

    msg = build_message()
    print(msg)

    # Always rotate prev snapshots so tomorrow's change comparison is fresh,
    # regardless of whether the Telegram send succeeds.
    save_prev_snapshots()

    if dry_run:
        print("\n[dry run] message not sent")
        return

    send_telegram(msg)
    print("\n[OK] Telegram message sent")


if __name__ == "__main__":
    main()
