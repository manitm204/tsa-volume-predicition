#!/usr/bin/env python3
"""
kalshi.py
=========
Kalshi TSA weekly-volume ladder trader.

Reads model probabilities from output_autogluon_predict/weekly_summary.csv,
fetches current TSA market orderbooks, saves a market snapshot, then runs a
Kelly ladder to buy YES shares or place passive limit orders.

Environment:
  KALSHI_KEY_ID              API key UUID from Kalshi dashboard
  KALSHI_PRIVATE_KEY_PATH    Absolute path to RSA private key PEM
  KALSHI_ENV                 "prod" (default) or "demo"

Usage:
  python kalshi.py                    # live trading
  python kalshi.py --dry-run          # simulate without placing orders
  python kalshi.py --bankroll 500     # override bankroll
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


# ── Endpoints ─────────────────────────────────────────────────────────────────
_PROD_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_DEMO_BASE = "https://demo-api.kalshi.co/trade-api/v2"
_API_PATH_PREFIX = "/trade-api/v2"

_DEFAULT_SERIES    = "KXTSAW"
_DAILY_SERIES      = "KXTRUFTSA"

# Anchor all paths to the kalshi.py file location so behaviour is identical
# whether kalshi.py is run as a CLI from the repo root or imported by the
# FastAPI dashboard (whose CWD is dashboard/backend/).
_REPO_ROOT       = Path(__file__).resolve().parent
OUT_DIR          = _REPO_ROOT / "output_kalshi"
SUMMARY_PATH       = _REPO_ROOT / "output_autogluon_predict" / "weekly_summary.csv"
DAILY_SUMMARY_PATH = _REPO_ROOT / "output_autogluon_predict" / "daily_summary.csv"
POSITIONS_PATH     = OUT_DIR / "positions.json"   # local position cache
SETTLEMENTS_PATH   = OUT_DIR / "settlements.csv"  # one row per settled (ticker, side, fill)


# ── Market dataclass ──────────────────────────────────────────────────────────
@dataclass
class Market:
    ticker:              str
    event_ticker:        str
    strike_millions:     float
    strike_passengers:   float
    yes_bid_cents:       float | None = None
    yes_ask_cents:       float | None = None
    no_bid_cents:        float | None = None
    no_ask_cents:        float | None = None
    yes_mid_cents:       float | None = None
    no_mid_cents:        float | None = None
    market_mid_cents:    float | None = None
    volume:              float        = 0.0
    open_interest:       float        = 0.0
    spread:              float        = 0.0
    market_prob:         float        = 0.5


# ── Auth ──────────────────────────────────────────────────────────────────────
class _KalshiAuth:
    def __init__(self, key_id: str, private_key_pem: bytes) -> None:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        self._key_id      = key_id
        self._private_key = serialization.load_pem_private_key(private_key_pem, password=None)
        # Elections API (api.elections.kalshi.com) requires RSA-PSS-SHA256
        self._padding     = padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        )
        self._hash        = hashes.SHA256()

    def _sign(self, msg: str) -> str:
        sig = self._private_key.sign(msg.encode("utf-8"), self._padding, self._hash)
        return base64.b64encode(sig).decode("ascii")

    def headers(self, method: str, path: str) -> dict[str, str]:
        ts_ms = str(int(time.time() * 1000))
        msg   = ts_ms + method.upper() + path
        sig   = self._sign(msg)
        return {
            "KALSHI-ACCESS-KEY":       self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "Content-Type":            "application/json",
        }


def _load_auth() -> _KalshiAuth:
    key_id   = os.environ.get("KALSHI_KEY_ID", "").strip()
    pem_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not key_id:
        raise EnvironmentError("KALSHI_KEY_ID env var not set")
    if not pem_path:
        raise EnvironmentError("KALSHI_PRIVATE_KEY_PATH env var not set")
    return _KalshiAuth(key_id, Path(pem_path).read_bytes())


def _base() -> str:
    env = os.environ.get("KALSHI_ENV", "prod").strip().lower()
    return _DEMO_BASE if env == "demo" else _PROD_BASE


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _get(path: str, auth: _KalshiAuth, *, params: dict | None = None, retries: int = 4) -> dict:
    url  = _base() + path
    last: BaseException | None = None
    for attempt in range(1, retries):
        headers = auth.headers("GET", _API_PATH_PREFIX + path)
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                print(f"  [kalshi] rate-limited; retry in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code is not None and 400 <= code < 500 and code != 429:
                raise
            last = e
        except Exception as e:
            last = e
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"Kalshi GET {path} failed after {retries} attempts: {last}")


def _post(path: str, auth: _KalshiAuth, *, payload: dict, retries: int = 4) -> dict:
    url  = _base() + path
    last: BaseException | None = None
    for attempt in range(1, retries):
        headers = auth.headers("POST", _API_PATH_PREFIX + path)
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                print(f"  [kalshi] rate-limited; retry in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 401:
                raise RuntimeError(_portfolio_401_hint()) from e
            if code is not None and 400 <= code < 500 and code != 429:
                body = e.response.text if e.response is not None else ""
                raise RuntimeError(
                    f"Kalshi POST {path} {code}: {body}\npayload={json.dumps(payload)}"
                ) from e
            last = e
        except Exception as e:
            last = e
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"Kalshi POST {path} failed after {retries} attempts: {last}")


def cancel_order(order_id: str) -> dict:
    """Cancel an open order by ID."""
    auth = _load_auth()
    path = f"/portfolio/orders/{order_id}"
    url  = _base() + path
    last: BaseException | None = None
    for attempt in range(4):
        headers = auth.headers("DELETE", _API_PATH_PREFIX + path)
        try:
            resp = requests.delete(url, headers=headers, timeout=20)
            if resp.status_code == 429:
                time.sleep(min(2 ** attempt, 30))
                continue
            resp.raise_for_status()
            return resp.json() if resp.text else {}
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code is not None and 400 <= code < 500 and code != 429:
                raise
            last = e
        except Exception as e:
            last = e
        if attempt + 1 < 4:
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"cancel_order {order_id} failed: {last}")


# ── Strike extraction ─────────────────────────────────────────────────────────
def _extract_strike_millions(mkt: dict) -> float | None:
    subtitle = (mkt.get("subtitle") or "").lower()
    for word in subtitle.split():
        try:
            val = float(word.rstrip("m").rstrip(","))
            if 1.0 < val < 10.0:
                return val
        except ValueError:
            pass

    for key in ("floor_strike", "cap_strike", "strike"):
        raw = mkt.get(key)
        if raw is not None:
            try:
                val = float(raw)
                if val > 100:
                    val /= 1e6
                if 1.0 < val < 10.0:
                    return val
            except (TypeError, ValueError):
                pass

    title = (mkt.get("title") or "").lower()
    for word in title.split():
        try:
            val = float(word.rstrip("m").rstrip(","))
            if 1.0 < val < 10.0:
                return val
        except ValueError:
            pass

    return None


def _extract_daily_threshold_millions(ticker: str) -> float | None:
    """Parse T{N} threshold from KXTRUFTSA-26JUN04-T2820000 style tickers."""
    for part in reversed(ticker.split("-")):
        if part.startswith("T") and part[1:].isdigit():
            val = int(part[1:])
            if 500_000 < val < 10_000_000:
                return round(val / 1e6, 6)
    return None


# ── Price enrichment helpers ──────────────────────────────────────────────────
def _extract_prices(mkt: dict) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (yes_bid, yes_ask, no_bid, no_ask) from a market dict, all in cents."""
    return (
        (float(mkt["yes_bid"]) if mkt.get("yes_bid") is not None else None),
        (float(mkt["yes_ask"]) if mkt.get("yes_ask") is not None else None),
        (float(mkt["no_bid"])  if mkt.get("no_bid")  is not None else None),
        (float(mkt["no_ask"])  if mkt.get("no_ask")  is not None else None),
    )


def _prices_from_orderbook(auth: _KalshiAuth, ticker: str) -> tuple[float | None, float | None]:
    """
    Return (yes_bid_cents, yes_ask_cents) derived from the live orderbook.
    Orderbook prices are in dollars (0-1); we convert to cents.

    YES bid  = best YES bid level
    YES ask  = 1 - best NO bid level  (someone buying NO at 0.62 => YES costs 0.38)
    """
    ob_data = _get(f"/markets/{ticker}/orderbook", auth)
    ob = ob_data.get("orderbook_fp") or ob_data.get("orderbook") or {}

    no_levels  = ob.get("no_dollars")  or ob.get("no")  or []
    yes_levels = ob.get("yes_dollars") or ob.get("yes") or []

    yes_bid_cents: float | None = None
    yes_ask_cents: float | None = None

    yes_active = [(float(l[0]), int(float(l[1]))) for l in yes_levels if int(float(l[1])) > 0]
    if yes_active:
        best_yes_bid_dollars = max(p for p, _ in yes_active)
        yes_bid_cents = round(best_yes_bid_dollars * 100, 2)

    no_active = [(float(l[0]), int(float(l[1]))) for l in no_levels if int(float(l[1])) > 0]
    if no_active:
        best_no_bid_dollars = max(p for p, _ in no_active)
        yes_ask_cents = round((1.0 - best_no_bid_dollars) * 100, 2)

    return yes_bid_cents, yes_ask_cents


# ── Market fetch ──────────────────────────────────────────────────────────────
def fetch_tsa_weekly_markets(
    series_ticker: str = _DEFAULT_SERIES,
    debug: bool = False,
) -> list[Market]:
    """Fetch all open TSA weekly-volume markets with live bid/ask prices.

    Price enrichment order:
      1. /markets list response  (fastest; often missing prices)
      2. /markets/{ticker}       (individual market detail)
      3. /markets/{ticker}/orderbook  (derived from best bid levels)
    """
    auth = _load_auth()

    print(f"[Kalshi] Fetching open events for series={series_ticker}...")
    events_data = _get("/events", auth, params={"series_ticker": series_ticker, "status": "open"})
    events = events_data.get("events", [])
    if not events:
        print(f"  No open events found for {series_ticker}")
        return []

    markets_out: list[Market] = []
    for event in events:
        event_ticker = event.get("event_ticker", "")
        print(f"  Event: {event_ticker}")

        mkt_data    = _get("/markets", auth, params={"event_ticker": event_ticker, "status": "open"})
        raw_markets = mkt_data.get("markets", [])

        for mkt in raw_markets:
            ticker   = mkt.get("ticker", "")
            strike_m = _extract_strike_millions(mkt)
            if strike_m is None:
                print(f"    WARNING: could not parse strike for {ticker}; skipping")
                continue

            yes_bid, yes_ask, no_bid, no_ask = _extract_prices(mkt)
            volume   = float(mkt.get("volume", 0) or 0)
            open_int = float(mkt.get("open_interest", 0) or 0)

            if debug:
                print(f"    DEBUG list: {ticker}  yes_bid={yes_bid} yes_ask={yes_ask} "
                      f"last_price={mkt.get('last_price')} vol={volume}")

            # Fallback 1: individual market endpoint
            if yes_bid is None and yes_ask is None:
                try:
                    detail   = _get(f"/markets/{ticker}", auth)
                    dm       = detail.get("market", detail)
                    yes_bid, yes_ask, no_bid, no_ask = _extract_prices(dm)
                    if volume == 0:
                        volume = float(dm.get("volume", 0) or 0)
                    if debug:
                        print(f"    DEBUG individual: yes_bid={yes_bid} yes_ask={yes_ask} "
                              f"last_price={dm.get('last_price')}")
                except Exception as e:
                    if debug:
                        print(f"    DEBUG individual fetch failed: {e}")

            # Fallback 2: orderbook
            if yes_bid is None and yes_ask is None:
                try:
                    yes_bid, yes_ask = _prices_from_orderbook(auth, ticker)
                    if debug:
                        print(f"    DEBUG orderbook: yes_bid={yes_bid}c yes_ask={yes_ask}c")
                except Exception as e:
                    if debug:
                        print(f"    DEBUG orderbook fetch failed: {e}")

            # Compute mid
            if yes_ask is not None and yes_bid is not None:
                yes_mid   = (yes_ask + yes_bid) / 2.0
                no_mid    = ((no_ask or (100 - yes_bid)) + (no_bid or (100 - yes_ask))) / 2.0
                mid_cents = yes_mid
            elif yes_ask is not None:
                yes_mid   = yes_ask
                no_mid    = 100.0 - yes_ask
                mid_cents = yes_mid
            elif yes_bid is not None:
                yes_mid   = yes_bid
                no_mid    = 100.0 - yes_bid
                mid_cents = yes_mid
            else:
                mid_cents = float(mkt.get("last_price") or 50)
                yes_mid   = mid_cents
                no_mid    = 100.0 - mid_cents

            spread_val  = abs(yes_ask - yes_bid) if (yes_ask is not None and yes_bid is not None) else 0.0
            market_prob = mid_cents / 100.0

            markets_out.append(Market(
                ticker            = ticker,
                event_ticker      = event_ticker,
                strike_millions   = strike_m,
                strike_passengers = strike_m * 1e6,
                yes_bid_cents     = yes_bid,
                yes_ask_cents     = yes_ask,
                no_bid_cents      = no_bid,
                no_ask_cents      = no_ask,
                yes_mid_cents     = yes_mid,
                no_mid_cents      = no_mid,
                market_mid_cents  = mid_cents,
                volume            = volume,
                open_interest     = open_int,
                spread            = spread_val,
                market_prob       = market_prob,
            ))
            print(f"    {ticker:40s}  strike={strike_m:.2f}M  "
                  f"yes_bid/ask={yes_bid}/{yes_ask}  mid={mid_cents:.1f}c  vol={volume:.0f}")

    markets_out.sort(key=lambda m: m.strike_millions)
    print(f"[Kalshi] {len(markets_out)} open market(s) found.")
    return markets_out


def fetch_tsa_daily_markets(debug: bool = False) -> list[Market]:
    """Fetch open KXTRUFTSA daily markets with live bid/ask prices."""
    auth = _load_auth()
    print(f"[Kalshi] Fetching open daily events for series={_DAILY_SERIES}...")
    events_data = _get("/events", auth, params={"series_ticker": _DAILY_SERIES, "status": "open"})
    events = events_data.get("events", [])
    if not events:
        print(f"  No open daily events found for {_DAILY_SERIES}")
        return []

    markets_out: list[Market] = []
    for event in events:
        event_ticker = event.get("event_ticker", "")
        print(f"  Event: {event_ticker}")
        mkt_data    = _get("/markets", auth, params={"event_ticker": event_ticker, "status": "open"})
        raw_markets = mkt_data.get("markets", [])

        for mkt in raw_markets:
            ticker   = mkt.get("ticker", "")
            strike_m = _extract_daily_threshold_millions(ticker) or _extract_strike_millions(mkt)
            if strike_m is None:
                print(f"    WARNING: could not parse threshold for {ticker}; skipping")
                continue

            yes_bid, yes_ask, no_bid, no_ask = _extract_prices(mkt)
            volume   = float(mkt.get("volume", 0) or 0)
            open_int = float(mkt.get("open_interest", 0) or 0)

            if debug:
                print(f"    DEBUG list: {ticker}  yes_bid={yes_bid} yes_ask={yes_ask} "
                      f"last_price={mkt.get('last_price')} vol={volume}")

            if yes_bid is None and yes_ask is None:
                try:
                    detail = _get(f"/markets/{ticker}", auth)
                    dm = detail.get("market", detail)
                    yes_bid, yes_ask, no_bid, no_ask = _extract_prices(dm)
                    if volume == 0:
                        volume = float(dm.get("volume", 0) or 0)
                    if debug:
                        print(f"    DEBUG individual: yes_bid={yes_bid} yes_ask={yes_ask}")
                except Exception as e:
                    if debug:
                        print(f"    DEBUG individual fetch failed: {e}")

            if yes_bid is None and yes_ask is None:
                try:
                    yes_bid, yes_ask = _prices_from_orderbook(auth, ticker)
                    if debug:
                        print(f"    DEBUG orderbook: yes_bid={yes_bid}c yes_ask={yes_ask}c")
                except Exception as e:
                    if debug:
                        print(f"    DEBUG orderbook fetch failed: {e}")

            if yes_ask is not None and yes_bid is not None:
                yes_mid   = (yes_ask + yes_bid) / 2.0
                no_mid    = ((no_ask or (100 - yes_bid)) + (no_bid or (100 - yes_ask))) / 2.0
                mid_cents = yes_mid
            elif yes_ask is not None:
                yes_mid   = yes_ask
                no_mid    = 100.0 - yes_ask
                mid_cents = yes_mid
            elif yes_bid is not None:
                yes_mid   = yes_bid
                no_mid    = 100.0 - yes_bid
                mid_cents = yes_mid
            else:
                mid_cents = float(mkt.get("last_price") or 50)
                yes_mid   = mid_cents
                no_mid    = 100.0 - mid_cents

            spread_val  = abs(yes_ask - yes_bid) if (yes_ask is not None and yes_bid is not None) else 0.0
            market_prob = mid_cents / 100.0

            markets_out.append(Market(
                ticker            = ticker,
                event_ticker      = event_ticker,
                strike_millions   = strike_m,
                strike_passengers = strike_m * 1e6,
                yes_bid_cents     = yes_bid,
                yes_ask_cents     = yes_ask,
                no_bid_cents      = no_bid,
                no_ask_cents      = no_ask,
                yes_mid_cents     = yes_mid,
                no_mid_cents      = no_mid,
                market_mid_cents  = mid_cents,
                volume            = volume,
                open_interest     = open_int,
                spread            = spread_val,
                market_prob       = market_prob,
            ))
            print(f"    {ticker:50s}  threshold={strike_m:.4f}M  "
                  f"yes_bid/ask={yes_bid}/{yes_ask}  mid={mid_cents:.1f}c  vol={volume:.0f}")

    markets_out.sort(key=lambda m: m.strike_millions)
    print(f"[Kalshi] {len(markets_out)} open daily market(s) found.")
    return markets_out


# ── Orderbook fetch ───────────────────────────────────────────────────────────
def fetch_orderbook(ticker: str) -> dict:
    auth = _load_auth()
    return _get(f"/markets/{ticker}/orderbook", auth)


def yes_asks_from_orderbook(orderbook_data: dict) -> list[tuple[float, int]]:
    """
    YES asks are derived from NO bids: yes_ask_price = 1.00 - no_bid_price.
    """
    ob = orderbook_data.get("orderbook_fp") or orderbook_data.get("orderbook") or {}
    no_levels = ob.get("no_dollars") or ob.get("no") or []

    asks: list[tuple[float, int]] = []
    for level in no_levels:
        no_bid_price  = float(level[0])
        qty           = int(float(level[1]))
        yes_ask_price = round(1.0 - no_bid_price, 4)
        if qty > 0 and 0.01 <= yes_ask_price <= 0.99:
            asks.append((yes_ask_price, qty))

    asks.sort(key=lambda x: x[0])
    return asks


def no_asks_from_orderbook(orderbook_data: dict) -> list[tuple[float, int]]:
    """
    NO asks are derived from YES bids: no_ask_price = 1.00 - yes_bid_price.
    """
    ob = orderbook_data.get("orderbook_fp") or orderbook_data.get("orderbook") or {}
    yes_levels = ob.get("yes_dollars") or ob.get("yes") or []

    asks: list[tuple[float, int]] = []
    for level in yes_levels:
        yes_bid_price = float(level[0])
        qty           = int(float(level[1]))
        no_ask_price  = round(1.0 - yes_bid_price, 4)
        if qty > 0 and 0.01 <= no_ask_price <= 0.99:
            asks.append((no_ask_price, qty))

    asks.sort(key=lambda x: x[0])
    return asks


# ── Portfolio helpers ─────────────────────────────────────────────────────────
def _portfolio_401_hint() -> str:
    return (
        "401 Unauthorized on portfolio endpoint.\n"
        "Possible causes:\n"
        "  1. API key doesn't have trading/portfolio permissions (check Kalshi dashboard)\n"
        "  2. KALSHI_ENV=demo but using prod key (or vice-versa)\n"
        "  3. Private key file doesn't match the registered KALSHI_KEY_ID\n"
        "  4. System clock is skewed >5s from server time\n"
        "Run with --dry-run to simulate without portfolio access."
    )


def fetch_open_orders(ticker: str | None = None) -> list[dict]:
    auth   = _load_auth()
    params = {"status": "resting"}
    if ticker:
        params["ticker"] = ticker
    try:
        data = _get("/portfolio/orders", auth, params=params)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            raise RuntimeError(_portfolio_401_hint()) from e
        raise
    return data.get("orders", [])


def fetch_positions(ticker: str | None = None) -> list[dict]:
    auth   = _load_auth()
    params = {"limit": "200", "settlement_status": "unsettled"}
    if ticker:
        params["ticker"] = ticker
    try:
        data = _get("/portfolio/positions", auth, params=params)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            raise RuntimeError(_portfolio_401_hint()) from e
        raise
    positions = data.get("market_positions", data.get("positions", []))
    cursor = data.get("cursor")
    while cursor:
        params["cursor"] = cursor
        data = _get("/portfolio/positions", auth, params=params)
        positions.extend(data.get("market_positions", data.get("positions", [])))
        cursor = data.get("cursor")
    return positions


def get_yes_position(ticker: str) -> int:
    for pos in fetch_positions(ticker):
        if pos.get("ticker") == ticker:
            if "position" in pos:
                return int(float(pos.get("position") or 0))
            if "yes_position" in pos:
                return int(float(pos.get("yes_position") or 0))
            if "market_exposure" in pos:
                return int(float(pos.get("market_exposure") or 0))
    return 0


def get_open_yes_buy_orders(ticker: str) -> int:
    total = 0
    for order in fetch_open_orders(ticker):
        if order.get("ticker") != ticker:
            continue
        if order.get("action") != "buy" or order.get("side") != "yes":
            continue
        remaining = (
            order.get("remaining_count")
            or order.get("remaining_count_fp")
            or order.get("count")
            or order.get("count_fp")
            or 0
        )
        total += int(float(remaining))
    return total


def get_no_position(ticker: str) -> int:
    for pos in fetch_positions(ticker):
        if pos.get("ticker") == ticker:
            if "no_position" in pos:
                return int(float(pos.get("no_position") or 0))
            # Some responses encode NO as a negative position value
            if "position" in pos:
                return max(0, -int(float(pos.get("position") or 0)))
    return 0


def get_open_no_buy_orders(ticker: str) -> int:
    total = 0
    for order in fetch_open_orders(ticker):
        if order.get("ticker") != ticker:
            continue
        if order.get("action") != "buy" or order.get("side") != "no":
            continue
        remaining = (
            order.get("remaining_count")
            or order.get("remaining_count_fp")
            or order.get("count")
            or order.get("count_fp")
            or 0
        )
        total += int(float(remaining))
    return total


def fetch_portfolio_summary() -> dict:
    """Fetch live positions and open orders from the Kalshi API.

    Returns:
        {
            "positions": [
                {
                    "ticker":       str,
                    "yes":          int,   # shares held (0 if none)
                    "no":           int,   # shares held (0 if none)
                    "avg_price":    float | None,  # dollars per share
                    "cost_dollars": float,
                }
            ],
            "open_orders": [
                {
                    "ticker":    str,
                    "side":      "yes" | "no",
                    "price":     float,   # dollars
                    "remaining": int,
                }
            ],
        }
    """
    positions_raw  = fetch_positions()
    open_orders_raw = fetch_open_orders()

    positions = []
    for pos in positions_raw:
        ticker = pos.get("ticker", "")
        if not ticker:
            continue

        # Net field (positive = YES, negative = NO); fall back to separate yes/no fields
        net_fp = float(pos.get("position_fp", pos.get("position", 0)) or 0)
        if net_fp != 0:
            yes = max(round(net_fp), 0)
            no  = max(round(-net_fp), 0)
        else:
            yes = max(int(float(pos.get("yes_position") or pos.get("market_exposure") or 0)), 0)
            no  = max(int(float(pos.get("no_position") or 0)), 0)

        if yes == 0 and no == 0:
            continue

        cost_dollars = float(pos.get("total_traded_dollars", 0) or 0)
        if cost_dollars == 0:
            total_traded_cents = float(pos.get("total_traded", 0) or 0)
            cost_dollars = total_traded_cents / 100.0
        total_shares = yes + no if net_fp == 0 else abs(net_fp)
        avg_price = None
        if total_shares > 0 and cost_dollars > 0:
            avg_price = cost_dollars / total_shares
        elif total_shares > 0:
            resting_price = float(pos.get("resting_orders_count", 0) or 0)
            if resting_price == 0:
                market_exp = float(pos.get("market_exposure", 0) or 0)
                if market_exp > 0:
                    avg_price = market_exp / total_shares / 100.0

        positions.append({
            "ticker":       ticker,
            "yes":          yes,
            "no":           no,
            "avg_price":    avg_price,
            "cost_dollars": cost_dollars,
        })

    open_orders = []
    for order in open_orders_raw:
        ticker = order.get("ticker", "")
        side   = order.get("side", "")
        action = order.get("action", "")
        if not ticker or side not in ("yes", "no") or action != "buy":
            continue

        remaining = round(float(
            order.get("remaining_count_fp")
            or order.get("remaining_count")
            or order.get("count_fp")
            or order.get("count")
            or 0
        ))
        if remaining <= 0:
            continue

        price_str = (
            order.get("yes_price_dollars") if side == "yes"
            else order.get("no_price_dollars")
        )
        try:
            price = float(price_str)
        except (TypeError, ValueError):
            price = float(
                order.get("yes_price") if side == "yes" else order.get("no_price") or 0
            ) / 100

        open_orders.append({
            "ticker":    ticker,
            "side":      side,
            "price":     price,
            "remaining": remaining,
        })

    return {"positions": positions, "open_orders": open_orders}


# ── Settlements + EV journal ─────────────────────────────────────────────────
def fetch_settled_positions(ticker: str | None = None) -> list[dict]:
    """All settled positions for this account (paged through Kalshi cursor)."""
    auth   = _load_auth()
    params = {"limit": "200", "settlement_status": "settled"}
    if ticker:
        params["ticker"] = ticker
    try:
        data = _get("/portfolio/positions", auth, params=params)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            raise RuntimeError(_portfolio_401_hint()) from e
        raise
    positions = data.get("market_positions", data.get("positions", []))
    cursor = data.get("cursor")
    while cursor:
        params["cursor"] = cursor
        data = _get("/portfolio/positions", auth, params=params)
        positions.extend(data.get("market_positions", data.get("positions", [])))
        cursor = data.get("cursor")
    return positions


def fetch_market_result(ticker: str) -> dict:
    """Look up a market and return {result, settled_time, status, expiration_value}.

    `result` is "yes" or "no" once the market is settled. Returns empty dict for
    unsettled markets or on 404.
    """
    auth = _load_auth()
    try:
        data = _get(f"/markets/{ticker}", auth)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code in (401, 404):
            return {}
        raise
    mkt = data.get("market", {}) or {}
    return {
        "ticker":         mkt.get("ticker", ticker),
        "status":         mkt.get("status"),
        "result":         mkt.get("result"),                # "yes" / "no" / ""
        "settled_time":   mkt.get("settled_time"),
        "expiration_value": mkt.get("expiration_value"),    # numeric outcome (e.g., 2.687M for TSA)
    }


def sync_settlements() -> pd.DataFrame:
    """Refresh ./output_kalshi/settlements.csv with all settled KXTSAW positions.

    For each settled position we record:
        ticker, side (yes/no), shares, avg_price, cost_dollars,
        result (yes/no), realized_dollars, settled_time

    Realized P&L per position = (1 if our side wins else 0) × shares − cost_dollars.
    """
    OUT_DIR.mkdir(exist_ok=True)

    # Load existing settlements so we don't re-query markets we've already cached
    existing: pd.DataFrame = (
        pd.read_csv(SETTLEMENTS_PATH) if SETTLEMENTS_PATH.exists() else pd.DataFrame()
    )
    seen_keys = set()
    if not existing.empty:
        for _, r in existing.iterrows():
            seen_keys.add((str(r["ticker"]), str(r["side"])))

    try:
        settled_raw = fetch_settled_positions()
    except Exception as exc:
        print(f"[settle] fetch_settled_positions failed: {exc}")
        return existing

    new_rows: list[dict] = []
    result_cache: dict[str, dict] = {}

    for pos in settled_raw:
        ticker = pos.get("ticker", "")
        if not ticker or "KXTSAW" not in ticker:
            continue

        net_fp = float(pos.get("position_fp", pos.get("position", 0)) or 0)
        if net_fp != 0:
            yes = max(round(net_fp), 0)
            no  = max(round(-net_fp), 0)
        else:
            yes = max(int(float(pos.get("yes_position") or pos.get("market_exposure") or 0)), 0)
            no  = max(int(float(pos.get("no_position") or 0)), 0)

        cost_dollars = float(pos.get("total_traded_dollars", 0) or 0)
        if cost_dollars == 0:
            cost_dollars = float(pos.get("total_traded", 0) or 0) / 100.0

        sides: list[tuple[str, int]] = []
        if yes > 0: sides.append(("yes", yes))
        if no  > 0: sides.append(("no",  no))
        if not sides:
            continue

        if ticker not in result_cache:
            result_cache[ticker] = fetch_market_result(ticker)
        mr = result_cache[ticker]
        result = (mr.get("result") or "").lower()
        settled_time = mr.get("settled_time") or pos.get("last_updated_ts") or ""

        for side, shares in sides:
            if (ticker, side) in seen_keys:
                continue
            # Cost allocated proportionally to this side's share of net cost
            share_of_cost = cost_dollars * (shares / max(yes + no, 1)) if (yes + no) > 0 else 0.0
            avg_price = (share_of_cost / shares) if shares > 0 else None

            if result == "yes":
                realized = (1.0 if side == "yes" else 0.0) * shares - share_of_cost
            elif result == "no":
                realized = (1.0 if side == "no"  else 0.0) * shares - share_of_cost
            else:
                # Settled but no result field — leave realized null
                realized = None

            new_rows.append({
                "ticker":          ticker,
                "side":            side,
                "shares":          shares,
                "avg_price":       round(avg_price, 4) if avg_price is not None else None,
                "cost_dollars":    round(share_of_cost, 4),
                "result":          result or None,
                "realized_dollars": round(realized, 4) if realized is not None else None,
                "settled_time":    settled_time,
            })

    if not new_rows:
        return existing

    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if not existing.empty else pd.DataFrame(new_rows)
    combined.to_csv(SETTLEMENTS_PATH, index=False)
    print(f"[settle] +{len(new_rows)} settled rows -> {SETTLEMENTS_PATH}")
    return combined


def load_actions_history() -> pd.DataFrame:
    """Aggregate every actions_*.csv into one frame, with parsed run timestamps."""
    files = sorted(OUT_DIR.glob("actions_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        ts = f.stem.replace("actions_", "")
        try:
            df = pd.read_csv(f)
            df["run_ts"] = ts
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_ev_journal() -> dict:
    """Join live order actions with settlements to produce theoretical-vs-realized.

    Returns:
      {
        "fills":   [ {run_ts, ticker, strike, side, shares, price, model_prob,
                      market_prob, theoretical_ev, settled, result, realized_pnl} ],
        "totals":  {theoretical_ev, realized_pnl, capture_ratio,
                    n_settled, n_pending, n_total}
      }
    """
    actions = load_actions_history()
    if actions.empty:
        return {"fills": [], "totals": {
            "theoretical_ev": 0.0, "realized_pnl": 0.0, "capture_ratio": None,
            "n_settled": 0, "n_pending": 0, "n_total": 0,
        }}

    # Live fills only — dry_run can be string or bool depending on csv source
    actions["is_live"] = actions["dry_run"].astype(str).str.lower() == "false"
    live = actions[actions["is_live"]].copy()
    if live.empty:
        return {"fills": [], "totals": {
            "theoretical_ev": 0.0, "realized_pnl": 0.0, "capture_ratio": None,
            "n_settled": 0, "n_pending": 0, "n_total": 0,
        }}

    # Settlements lookup — keyed by (ticker, side); realized P&L distributed
    # across fills proportionally by cost.
    settle_map: dict[tuple[str, str], dict] = {}
    if SETTLEMENTS_PATH.exists():
        s = pd.read_csv(SETTLEMENTS_PATH)
        for _, r in s.iterrows():
            settle_map[(str(r["ticker"]), str(r["side"]))] = {
                "result":   r.get("result"),
                "realized": float(r.get("realized_dollars") or 0),
                "total_cost": float(r.get("cost_dollars") or 0),
                "settled_time": r.get("settled_time"),
            }

    fills: list[dict] = []
    for _, a in live.iterrows():
        ticker = str(a.get("ticker", ""))
        side   = str(a.get("side", ""))
        price  = float(a.get("price") or 0)
        shares = int(float(a.get("shares") or 0))
        if shares == 0 or price <= 0:
            continue
        cost = price * shares

        try:
            model_p  = float(a.get("model_prob"))
            market_p = float(a.get("market_prob"))
        except (TypeError, ValueError):
            model_p = market_p = float("nan")

        # Theoretical EV from model: (model_prob / price − 1) × cost
        if model_p == model_p and price > 0:                # not NaN
            theo_ev = (model_p / price - 1) * cost
        else:
            theo_ev = None

        # Strike from ticker (KXTSAW-26MAY24-A2.65 -> 2.65)
        strike = None
        for part in ticker.split("-"):
            if part.startswith("A"):
                try: strike = float(part[1:])
                except ValueError: pass

        st = settle_map.get((ticker, side))
        if st is not None:
            # Distribute the position's total realized P&L across each fill by cost weight
            total_cost = st["total_cost"] or cost
            realized = st["realized"] * (cost / total_cost) if total_cost > 0 else 0.0
            fills.append({
                "run_ts":         str(a.get("run_ts", "")),
                "ticker":         ticker,
                "strike":         strike,
                "side":           side,
                "shares":         shares,
                "price":          round(price, 4),
                "cost":           round(cost, 4),
                "model_prob":     None if model_p != model_p  else round(model_p,  4),
                "market_prob":    None if market_p != market_p else round(market_p, 4),
                "theoretical_ev": round(theo_ev, 4) if theo_ev is not None else None,
                "settled":        True,
                "result":         st.get("result"),
                "realized_pnl":   round(realized, 4),
            })
        else:
            fills.append({
                "run_ts":         str(a.get("run_ts", "")),
                "ticker":         ticker,
                "strike":         strike,
                "side":           side,
                "shares":         shares,
                "price":          round(price, 4),
                "cost":           round(cost, 4),
                "model_prob":     None if model_p != model_p  else round(model_p,  4),
                "market_prob":    None if market_p != market_p else round(market_p, 4),
                "theoretical_ev": round(theo_ev, 4) if theo_ev is not None else None,
                "settled":        False,
                "result":         None,
                "realized_pnl":   None,
            })

    n_settled = sum(1 for f in fills if f["settled"])
    n_pending = len(fills) - n_settled

    # Totals computed only on settled fills (apples-to-apples)
    settled_fills = [f for f in fills if f["settled"]]
    theo_total = sum((f["theoretical_ev"] or 0) for f in settled_fills)
    real_total = sum((f["realized_pnl"]   or 0) for f in settled_fills)
    capture = (real_total / theo_total) if theo_total != 0 else None

    return {
        "fills": sorted(fills, key=lambda x: x["run_ts"], reverse=True),
        "totals": {
            "theoretical_ev": round(theo_total, 4),
            "realized_pnl":   round(real_total, 4),
            "capture_ratio":  round(capture, 4) if capture is not None else None,
            "n_settled":      n_settled,
            "n_pending":      n_pending,
            "n_total":        len(fills),
        },
    }


# ── Kelly math ────────────────────────────────────────────────────────────────
def price_for_bankroll_pct(model_prob: float, pct: float, fee_rate: float = 0.07) -> float:
    """Max limit price at which buying pct% of bankroll has positive Kelly EV."""
    denom = 1.0 + fee_rate - pct
    if denom <= 0:
        return 0.0
    return max(0.01, min(0.99, (model_prob - pct) / denom))


def fee_adjusted_cost_per_share(price: float, fee_rate: float = 0.07) -> float:
    return price + fee_rate * price * (1.0 - price)


def shares_for_pct(
    *,
    pct: float,
    bankroll: float,
    price: float,
    kelly_fraction: float = 0.5,
    fee_rate: float = 0.07,
) -> int:
    dollars_to_risk = pct * bankroll * kelly_fraction
    cost_per_share  = fee_adjusted_cost_per_share(price, fee_rate)
    if cost_per_share <= 0:
        return 0
    return int(dollars_to_risk // cost_per_share)


# ── Order placement ───────────────────────────────────────────────────────────
def _end_of_day_expiration_ts() -> int:
    now = datetime.now(timezone.utc)
    eod = now.replace(hour=23, minute=59, second=0, microsecond=0)
    if eod <= now:
        eod += timedelta(days=1)
    return int(eod.timestamp())


def place_yes_limit_order(
    *,
    ticker: str,
    price: float,
    shares: int,
    expiration_ts: int | None = None,
    post_only: bool = False,
) -> dict:
    if shares < 1:
        raise ValueError("shares must be >= 1")
    price_cents = int(round(price * 100))
    if not 1 <= price_cents <= 99:
        raise ValueError(f"yes_price must be 1..99 cents, got {price_cents} (from {price})")
    payload = {
        "ticker":          ticker,
        "side":            "yes",
        "action":          "buy",
        "type":            "limit",
        "client_order_id": str(uuid.uuid4()),
        "count":           int(shares),
        "yes_price":       price_cents,
        "post_only":       post_only,
    }
    if expiration_ts is not None:
        payload["expiration_ts"] = expiration_ts
    auth = _load_auth()
    return _post("/portfolio/orders", auth, payload=payload)


def buy_yes_aggressive_limit(*, ticker: str, ask_price: float, shares: int) -> dict:
    return place_yes_limit_order(
        ticker=ticker,
        price=ask_price,
        shares=shares,
        expiration_ts=_end_of_day_expiration_ts(),
        post_only=False,
    )


def place_yes_passive_limit(*, ticker: str, limit_price: float, shares: int) -> dict:
    return place_yes_limit_order(
        ticker=ticker,
        price=limit_price,
        shares=shares,
        expiration_ts=_end_of_day_expiration_ts(),
        post_only=True,
    )


def place_no_limit_order(
    *,
    ticker: str,
    price: float,
    shares: int,
    expiration_ts: int | None = None,
    post_only: bool = False,
) -> dict:
    if shares < 1:
        raise ValueError("shares must be >= 1")
    price_cents = int(round(price * 100))
    if not 1 <= price_cents <= 99:
        raise ValueError(f"no_price must be 1..99 cents, got {price_cents} (from {price})")
    payload = {
        "ticker":          ticker,
        "side":            "no",
        "action":          "buy",
        "type":            "limit",
        "client_order_id": str(uuid.uuid4()),
        "count":           int(shares),
        "no_price":        price_cents,
        "post_only":       post_only,
    }
    if expiration_ts is not None:
        payload["expiration_ts"] = expiration_ts
    auth = _load_auth()
    return _post("/portfolio/orders", auth, payload=payload)


def buy_no_aggressive_limit(*, ticker: str, ask_price: float, shares: int) -> dict:
    return place_no_limit_order(
        ticker=ticker,
        price=ask_price,
        shares=shares,
        expiration_ts=_end_of_day_expiration_ts(),
        post_only=False,
    )


def place_no_passive_limit(*, ticker: str, limit_price: float, shares: int) -> dict:
    return place_no_limit_order(
        ticker=ticker,
        price=limit_price,
        shares=shares,
        expiration_ts=_end_of_day_expiration_ts(),
        post_only=True,
    )


# ── Ladder execution ──────────────────────────────────────────────────────────
def execute_over_ladder(
    *,
    ticker: str,
    model_prob: float,
    bankroll: float = 250.0,
    pct_levels: list[float] | None = None,
    kelly_fraction: float = 0.5,
    fee_rate: float = 0.07,
    dry_run: bool = True,
    initial_position: int = 0,
    initial_open_orders: int = 0,
) -> list[dict]:
    """
    For each bankroll pct level (15%..40%), compute the max limit price, then:
      - Walk YES asks and aggressively fill any shares priced <= limit.
      - If nothing filled at this level, place a passive limit order instead.
    Pass initial_position + initial_open_orders to account for shares already held.
    """
    if pct_levels is None:
        pct_levels = [0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    orderbook = fetch_orderbook(ticker)
    yes_asks  = yes_asks_from_orderbook(orderbook)

    # Mutable remaining quantity per price level — decremented as we consume shares.
    remaining: dict[float, int] = {price: qty for price, qty in yes_asks}

    def best_remaining_ask() -> float | None:
        for price, _ in yes_asks:
            if remaining.get(price, 0) > 0:
                return price
        return None

    actual_position = initial_position
    open_orders     = initial_open_orders
    filled_now      = 0

    actions: list[dict] = []

    for pct in pct_levels:
        limit_price        = price_for_bankroll_pct(model_prob, pct, fee_rate)
        bought_at_this_pct = 0

        if limit_price < 0.15 or limit_price > 0.97:
            continue

        for ask_price, _ in yes_asks:
            if ask_price > limit_price:
                break
            if ask_price < 0.15 or ask_price > 0.97:
                continue

            available = remaining.get(ask_price, 0)
            if available <= 0:
                continue

            target_shares      = shares_for_pct(
                pct=pct, bankroll=bankroll, price=ask_price,
                kelly_fraction=kelly_fraction, fee_rate=fee_rate,
            )
            exposure           = actual_position + open_orders + filled_now
            incremental_needed = target_shares - exposure
            shares_to_buy      = min(int(incremental_needed), available)

            if shares_to_buy <= 0:
                continue

            action = {
                "type":             "aggressive_buy",
                "pct":              pct,
                "ticker":           ticker,
                "price":            ask_price,
                "limit_price":      limit_price,
                "available_shares": available,
                "target_shares":    target_shares,
                "exposure_before":  exposure,
                "shares":           shares_to_buy,
                "dry_run":          dry_run,
            }

            if not dry_run:
                action["response"] = buy_yes_aggressive_limit(
                    ticker=ticker, ask_price=ask_price, shares=shares_to_buy
                )

            actions.append(action)
            remaining[ask_price] -= shares_to_buy
            filled_now            += shares_to_buy
            bought_at_this_pct    += shares_to_buy

        if bought_at_this_pct == 0:
            target_shares      = shares_for_pct(
                pct=pct, bankroll=bankroll, price=limit_price,
                kelly_fraction=kelly_fraction, fee_rate=fee_rate,
            )
            exposure           = actual_position + open_orders + filled_now
            incremental_needed = target_shares - exposure
            cur_best_ask       = best_remaining_ask()

            should_place_passive = (
                incremental_needed > 0
                and (cur_best_ask is None or limit_price < cur_best_ask)
            )

            if should_place_passive:
                shares_to_place = int(incremental_needed)
                action = {
                    "type":            "passive_limit",
                    "pct":             pct,
                    "ticker":          ticker,
                    "price":           limit_price,
                    "best_ask":        cur_best_ask,
                    "target_shares":   target_shares,
                    "exposure_before": exposure,
                    "shares":          shares_to_place,
                    "dry_run":         dry_run,
                }

                if not dry_run:
                    try:
                        action["response"] = place_yes_passive_limit(
                            ticker=ticker, limit_price=limit_price, shares=shares_to_place
                        )
                    except RuntimeError as e:
                        if "post only cross" in str(e):
                            print(f"    [YES passive] skipped at {limit_price:.3f} — would cross spread")
                            continue
                        raise

                actions.append(action)
                open_orders += shares_to_place

    return actions


def execute_under_ladder(
    *,
    ticker: str,
    model_prob: float,        # P(under) = 1 - p_over
    bankroll: float = 250.0,
    pct_levels: list[float] | None = None,
    kelly_fraction: float = 0.5,
    fee_rate: float = 0.07,
    dry_run: bool = True,
    initial_position: int = 0,
    initial_open_orders: int = 0,
) -> list[dict]:
    """
    UNDER / NO ladder. Mirrors execute_over_ladder for the NO side.
    NO asks are derived from YES bids: no_ask = 1 - yes_bid.
    Pass initial_position + initial_open_orders to account for shares already held.
    """
    if pct_levels is None:
        pct_levels = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    orderbook = fetch_orderbook(ticker)
    no_asks   = no_asks_from_orderbook(orderbook)

    remaining: dict[float, int] = {price: qty for price, qty in no_asks}

    def best_remaining_ask() -> float | None:
        for price, _ in no_asks:
            if remaining.get(price, 0) > 0:
                return price
        return None

    actual_position = initial_position
    open_orders     = initial_open_orders
    filled_now      = 0

    actions: list[dict] = []

    for pct in pct_levels:
        limit_price        = price_for_bankroll_pct(model_prob, pct, fee_rate)
        bought_at_this_pct = 0

        if limit_price < 0.15 or limit_price > 0.97:
            continue

        for ask_price, _ in no_asks:
            if ask_price > limit_price:
                break
            if ask_price < 0.15 or ask_price > 0.97:
                continue

            available = remaining.get(ask_price, 0)
            if available <= 0:
                continue

            target_shares      = shares_for_pct(
                pct=pct, bankroll=bankroll, price=ask_price,
                kelly_fraction=kelly_fraction, fee_rate=fee_rate,
            )
            exposure           = actual_position + open_orders + filled_now
            incremental_needed = target_shares - exposure
            shares_to_buy      = min(int(incremental_needed), available)

            if shares_to_buy <= 0:
                continue

            action = {
                "type":             "aggressive_buy",
                "side":             "no",
                "pct":              pct,
                "ticker":           ticker,
                "price":            ask_price,
                "limit_price":      limit_price,
                "available_shares": available,
                "target_shares":    target_shares,
                "exposure_before":  exposure,
                "shares":           shares_to_buy,
                "dry_run":          dry_run,
            }

            if not dry_run:
                action["response"] = buy_no_aggressive_limit(
                    ticker=ticker, ask_price=ask_price, shares=shares_to_buy
                )

            actions.append(action)
            remaining[ask_price] -= shares_to_buy
            filled_now            += shares_to_buy
            bought_at_this_pct    += shares_to_buy

        if bought_at_this_pct == 0:
            target_shares      = shares_for_pct(
                pct=pct, bankroll=bankroll, price=limit_price,
                kelly_fraction=kelly_fraction, fee_rate=fee_rate,
            )
            exposure           = actual_position + open_orders + filled_now
            incremental_needed = target_shares - exposure
            cur_best_ask       = best_remaining_ask()

            should_place_passive = (
                incremental_needed > 0
                and (cur_best_ask is None or limit_price < cur_best_ask)
            )

            if should_place_passive:
                shares_to_place = int(incremental_needed)
                action = {
                    "type":            "passive_limit",
                    "side":            "no",
                    "pct":             pct,
                    "ticker":          ticker,
                    "price":           limit_price,
                    "best_ask":        cur_best_ask,
                    "target_shares":   target_shares,
                    "exposure_before": exposure,
                    "shares":          shares_to_place,
                    "dry_run":         dry_run,
                }

                if not dry_run:
                    try:
                        action["response"] = place_no_passive_limit(
                            ticker=ticker, limit_price=limit_price, shares=shares_to_place
                        )
                    except RuntimeError as e:
                        if "post only cross" in str(e):
                            print(f"    [NO  passive] skipped at {limit_price:.3f} — would cross spread")
                            continue
                        raise

                actions.append(action)
                open_orders += shares_to_place


    return actions


# ── Local position cache ──────────────────────────────────────────────────────
def _load_positions() -> dict:
    """Return {ticker: {"yes": N, "no": M}} from the local cache file."""
    if POSITIONS_PATH.exists():
        with open(POSITIONS_PATH) as f:
            return json.load(f)
    return {}


def _save_positions(positions: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    with open(POSITIONS_PATH, "w") as f:
        json.dump(positions, f, indent=2)


def _get_position(ticker: str, side: str, dry_run: bool) -> tuple[int, int]:
    """
    Return (position, open_orders) for ticker/side.

    Priority:
      1. Kalshi portfolio API  (requires auth with portfolio permissions)
      2. Local positions cache (fallback when API returns 401)
    Always queries real positions even in dry-run; order placement is gated separately.
    """
    try:
        summary = fetch_portfolio_summary()

        pos = 0
        for p in summary["positions"]:
            if p["ticker"] == ticker:
                pos = p[side]
                break

        orders = 0
        for o in summary["open_orders"]:
            if o["ticker"] == ticker and o["side"] == side:
                orders += o["remaining"]

        print(f"  [{side.upper()}] portfolio API: position={pos}  open_orders={orders}")
        return pos, orders
    except RuntimeError as e:
        print(f"  [{side.upper()}] portfolio API unavailable ({e!s:.80}); using local cache")

    positions = _load_positions()
    pos = positions.get(ticker, {}).get(side, 0)
    print(f"  [{side.upper()}] local cache: position={pos}")
    return pos, 0  # can't track open orders locally


def _record_fills(ticker: str, yes_bought: int, no_bought: int, dry_run: bool) -> None:
    """Add newly filled shares to the local positions cache."""
    if dry_run or (yes_bought == 0 and no_bought == 0):
        return
    positions = _load_positions()
    entry = positions.setdefault(ticker, {"yes": 0, "no": 0})
    entry["yes"] += yes_bought
    entry["no"]  += no_bought
    _save_positions(positions)
    print(f"  [cache] updated {ticker}: yes={entry['yes']}  no={entry['no']}")


def _reset_positions_for_week(week_monday: str) -> None:
    """
    Clear the local cache when a new week starts so old positions
    from last week's market don't carry over.
    """
    marker_path = OUT_DIR / "positions_week.txt"
    OUT_DIR.mkdir(exist_ok=True)
    if marker_path.exists() and marker_path.read_text().strip() == week_monday:
        return  # same week, keep existing positions
    print(f"[cache] New week detected ({week_monday}). Clearing local positions cache.")
    _save_positions({})
    marker_path.write_text(week_monday)


# ── Daily probability helper ──────────────────────────────────────────────────
def _ensure_daily_probs(thresholds: list[float]) -> "pd.Series | None":
    """
    Ensure daily_summary.csv is current for today with the needed thresholds.
    If stale or missing columns, runs daily_predict.py as a subprocess.
    Returns the summary row, or None on failure.
    """
    import subprocess
    import sys as _sys
    from datetime import date as _date

    today_str = str(_date.today())

    if DAILY_SUMMARY_PATH.exists():
        try:
            df = pd.read_csv(DAILY_SUMMARY_PATH)
            if not df.empty:
                row   = df.iloc[0]
                needed = [f"p_over_{t}M" for t in thresholds]
                if str(row.get("date", "")) == today_str and all(c in df.columns for c in needed):
                    print(f"[Daily] daily_summary.csv already current — skipping re-compute")
                    return row
        except Exception:
            pass

    script = _REPO_ROOT / "daily_predict.py"
    if not script.exists():
        print(f"[Daily] daily_predict.py not found at {script} — skipping daily trading")
        return None

    cmd = [_sys.executable, str(script), "--thresholds"] + [str(t) for t in thresholds]
    print(f"[Daily] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[Daily] daily_predict.py exited {result.returncode} — skipping daily trading")
        return None

    if not DAILY_SUMMARY_PATH.exists():
        print(f"[Daily] daily_summary.csv not found after run — skipping")
        return None

    df = pd.read_csv(DAILY_SUMMARY_PATH)
    return df.iloc[0] if not df.empty else None


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi TSA weekly-volume ladder trader")
    parser.add_argument("--dry-run",        action="store_true", help="Simulate without placing orders")
    parser.add_argument("--bankroll",       type=float, default=250.0, help="Weekly market bankroll in dollars")
    parser.add_argument("--daily-bankroll", type=float, default=None,
                        help="Bankroll for KXTRUFTSA daily markets (omit to skip daily trading)")
    parser.add_argument("--series",         default=_DEFAULT_SERIES, help="Weekly Kalshi series ticker")
    parser.add_argument("--debug",          action="store_true", help="Print raw API field values")
    parser.add_argument("--sync-settlements", action="store_true",
                        help="Refresh settlements.csv and exit (no trading)")
    args = parser.parse_args()

    if args.sync_settlements:
        sync_settlements()
        return

    if args.dry_run:
        print("*** DRY RUN — no orders will be placed ***\n")

    # Load model probabilities from autogluon_predict output
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"weekly_summary.csv not found at {SUMMARY_PATH}\n"
            f"Run autogluon_predict.py first."
        )

    summary = pd.read_csv(SUMMARY_PATH).iloc[0]
    week_monday = str(summary["week_monday"])
    print(f"[Summary] week {week_monday} -> {summary['week_sunday']}")
    print(f"  weekly_avg={summary['weekly_avg_millions']:.4f}M  "
          f"std={summary['weekly_avg_std_millions']:.4f}M")

    # Reset local positions cache if this is a new week
    _reset_positions_for_week(week_monday)

    # Fetch live markets with price enrichment
    markets = fetch_tsa_weekly_markets(args.series, debug=args.debug)
    if not markets:
        print("No open markets found — exiting.")
        return

    # Save market snapshot
    OUT_DIR.mkdir(exist_ok=True)
    snapshot_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    snapshot_rows = []
    for m in markets:
        prob_col  = f"p_over_{m.strike_millions}M"
        model_prob = float(summary.get(prob_col, float("nan")))
        snapshot_rows.append({
            "snapshot_ts":    snapshot_ts,
            "ticker":         m.ticker,
            "event_ticker":   m.event_ticker,
            "strike_millions": m.strike_millions,
            "yes_bid_cents":  m.yes_bid_cents,
            "yes_ask_cents":  m.yes_ask_cents,
            "no_bid_cents":   m.no_bid_cents,
            "no_ask_cents":   m.no_ask_cents,
            "yes_mid_cents":  m.yes_mid_cents,
            "market_prob":    m.market_prob,
            "model_prob":     model_prob,
            "edge":           model_prob - m.market_prob,
            "volume":         m.volume,
            "open_interest":  m.open_interest,
            "spread":         m.spread,
        })

    snap_df = pd.DataFrame(snapshot_rows)
    snap_df.to_csv(OUT_DIR / f"market_snapshot_{snapshot_ts}.csv", index=False)
    snap_df.to_csv(OUT_DIR / "market_snapshot_latest.csv", index=False)
    print(f"\n[Snapshot] saved -> {OUT_DIR / 'market_snapshot_latest.csv'}")
    print(snap_df[["ticker", "strike_millions", "market_prob", "model_prob", "edge"]].to_string(index=False))

    try:
        import db
        db.write_market_snapshot(snap_df)
    except Exception as exc:
        import sys as _sys
        print(f"[db] market snapshot write failed (non-fatal): {exc}", file=_sys.stderr)

    # Run ladder for every market that has a model probability
    all_actions: list[dict] = []

    for m in markets:
        prob_col   = f"p_over_{m.strike_millions}M"
        if prob_col not in summary.index or pd.isna(summary[prob_col]):
            print(f"\n[{m.ticker}] no model prob for strike {m.strike_millions}M — skipping")
            continue

        model_prob  = float(summary[prob_col])
        market_prob = m.market_prob
        edge        = model_prob - market_prob

        if market_prob >= 0.99 or market_prob <= 0.01:
            print(f"\n[{m.ticker}] market_prob={market_prob:.3f} — at limit, skipping")
            continue

        model_prob_no  = 1.0 - model_prob
        market_prob_no = 1.0 - market_prob
        edge_no        = model_prob_no - market_prob_no

        print(f"\n{'='*60}")
        print(f"Market: {m.ticker}  strike={m.strike_millions}M")
        print(f"  YES — model={model_prob:.3f}  market={market_prob:.3f}  edge={edge:+.3f}")
        print(f"  NO  — model={model_prob_no:.3f}  market={market_prob_no:.3f}  edge={edge_no:+.3f}")

        # Fetch existing positions (API → local cache fallback)
        yes_pos, yes_orders = _get_position(m.ticker, "yes", args.dry_run)
        no_pos,  no_orders  = _get_position(m.ticker, "no",  args.dry_run)

        # YES (OVER) ladder
        yes_actions = execute_over_ladder(
            ticker=m.ticker,
            model_prob=model_prob,
            bankroll=args.bankroll,
            dry_run=args.dry_run,
            initial_position=yes_pos,
            initial_open_orders=yes_orders,
        )
        for a in yes_actions:
            a["side"]        = "yes"
            a["model_prob"]  = model_prob
            a["market_prob"] = market_prob
            all_actions.append(a)
            print(f"  [YES {a['type']}] pct={a['pct']:.0%}  "
                  f"price={a['price']:.3f}  shares={a['shares']}")

        if not yes_actions:
            print("  YES: no actions at any level.")

        # NO (UNDER) ladder
        no_actions = execute_under_ladder(
            ticker=m.ticker,
            model_prob=model_prob_no,
            bankroll=args.bankroll,
            dry_run=args.dry_run,
            initial_position=no_pos,
            initial_open_orders=no_orders,
        )
        for a in no_actions:
            a["side"]        = "no"
            a["model_prob"]  = model_prob_no
            a["market_prob"] = market_prob_no
            all_actions.append(a)
            print(f"  [NO  {a['type']}] pct={a['pct']:.0%}  "
                  f"price={a['price']:.3f}  shares={a['shares']}")

        if not no_actions:
            print("  NO: no actions at any level.")

        # Record aggressive fills in the local cache
        yes_bought = sum(a["shares"] for a in yes_actions if a["type"] == "aggressive_buy")
        no_bought  = sum(a["shares"] for a in no_actions  if a["type"] == "aggressive_buy")
        _record_fills(m.ticker, yes_bought, no_bought, args.dry_run)

    # ── Daily market trading (KXTRUFTSA) ─────────────────────────────────────
    if args.daily_bankroll is not None and args.daily_bankroll > 0:
        print(f"\n{'='*60}")
        print(f"[Daily] KXTRUFTSA markets  bankroll=${args.daily_bankroll:.0f}")

        daily_markets = fetch_tsa_daily_markets(debug=args.debug)
        if daily_markets:
            thresholds    = sorted({m.strike_millions for m in daily_markets})
            daily_summary = _ensure_daily_probs(thresholds)

            if daily_summary is not None:
                for m in daily_markets:
                    prob_col = f"p_over_{m.strike_millions}M"
                    if prob_col not in daily_summary.index or pd.isna(daily_summary[prob_col]):
                        print(f"\n[{m.ticker}] no model prob for {m.strike_millions}M — skipping")
                        continue

                    model_prob     = float(daily_summary[prob_col])
                    market_prob    = m.market_prob
                    model_prob_no  = 1.0 - model_prob
                    market_prob_no = 1.0 - market_prob

                    if market_prob >= 0.99 or market_prob <= 0.01:
                        print(f"\n[{m.ticker}] market_prob={market_prob:.3f} — at limit, skipping")
                        continue

                    print(f"\n{'='*60}")
                    print(f"[Daily] {m.ticker}  threshold={m.strike_millions}M")
                    print(f"  YES — model={model_prob:.3f}  market={market_prob:.3f}  "
                          f"edge={model_prob - market_prob:+.3f}")
                    print(f"  NO  — model={model_prob_no:.3f}  market={market_prob_no:.3f}  "
                          f"edge={model_prob_no - market_prob_no:+.3f}")

                    yes_pos, yes_orders = _get_position(m.ticker, "yes", args.dry_run)
                    no_pos,  no_orders  = _get_position(m.ticker, "no",  args.dry_run)

                    yes_daily = execute_over_ladder(
                        ticker=m.ticker,
                        model_prob=model_prob,
                        bankroll=args.daily_bankroll,
                        dry_run=args.dry_run,
                        initial_position=yes_pos,
                        initial_open_orders=yes_orders,
                    )
                    for a in yes_daily:
                        a["side"]        = "yes"
                        a["model_prob"]  = model_prob
                        a["market_prob"] = market_prob
                        all_actions.append(a)
                        print(f"  [YES {a['type']}] pct={a['pct']:.0%}  "
                              f"price={a['price']:.3f}  shares={a['shares']}")
                    if not yes_daily:
                        print("  YES: no actions at any level.")

                    no_daily = execute_under_ladder(
                        ticker=m.ticker,
                        model_prob=model_prob_no,
                        bankroll=args.daily_bankroll,
                        dry_run=args.dry_run,
                        initial_position=no_pos,
                        initial_open_orders=no_orders,
                    )
                    for a in no_daily:
                        a["side"]        = "no"
                        a["model_prob"]  = model_prob_no
                        a["market_prob"] = market_prob_no
                        all_actions.append(a)
                        print(f"  [NO  {a['type']}] pct={a['pct']:.0%}  "
                              f"price={a['price']:.3f}  shares={a['shares']}")
                    if not no_daily:
                        print("  NO: no actions at any level.")

                    yes_bought = sum(a["shares"] for a in yes_daily if a["type"] == "aggressive_buy")
                    no_bought  = sum(a["shares"] for a in no_daily  if a["type"] == "aggressive_buy")
                    _record_fills(m.ticker, yes_bought, no_bought, args.dry_run)
            else:
                print("[Daily] Could not compute probabilities — skipping daily trading.")
        else:
            print("[Daily] No open daily markets found.")

    # Save actions
    if all_actions:
        actions_df = pd.DataFrame(all_actions)
        actions_path = OUT_DIR / f"actions_{snapshot_ts}.csv"
        actions_df.to_csv(actions_path, index=False)
        print(f"\n[Actions] {len(all_actions)} action(s) saved -> {actions_path}")
    else:
        print("\n[Actions] No actions taken (no edge on any market).")

    # Write positions + actions to DB
    try:
        import db
        portfolio = fetch_portfolio_summary()
        db.write_positions_snapshot(portfolio.get("positions", []))
        if all_actions:
            db.write_order_actions(all_actions)
    except Exception as exc:
        import sys as _sys
        print(f"[db] portfolio/actions write failed (non-fatal): {exc}", file=_sys.stderr)

    if args.dry_run:
        print("\n*** DRY RUN — no orders were placed ***")
    else:
        # Refresh settlements after a live run so the EV journal stays fresh
        try:
            sync_settlements()
        except Exception as exc:
            import sys as _sys
            print(f"[settle] sync failed (non-fatal): {exc}", file=_sys.stderr)


if __name__ == "__main__":
    main()
