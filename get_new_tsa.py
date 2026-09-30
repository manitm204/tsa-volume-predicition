#!/usr/bin/env python3
"""Scrape tsa.gov passenger volumes and update data/tsa_volume.csv."""

import time
from pathlib import Path

import pandas as pd
import requests

TSA_PATH = Path(__file__).resolve().parent / "data" / "tsa_volume.csv"


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
}


def _get(url: str, *, label: str, timeout: float = 20.0, retries: int = 4) -> requests.Response:
    last: BaseException | None = None
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=_BROWSER_HEADERS)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            last = e
            if e.response is not None and e.response.status_code < 500:
                raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError) as e:
            last = e
        if i + 1 < retries:
            wait = min(8.0 * (2 ** i), 90.0)
            print(f"  [{label}] attempt {i+1}/{retries} failed ({last}); retry in {wait:.0f}s...")
            time.sleep(wait)
    assert last is not None
    raise last


def refresh_tsa() -> int:
    """Scrape tsa.gov and append new rows to tsa_volume.csv. Returns rows added."""
    from bs4 import BeautifulSoup

    print("[TSA] Fetching latest passenger volumes from tsa.gov...")
    resp = _get("https://www.tsa.gov/travel/passenger-volumes", label="TSA")
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise RuntimeError("No <table> found on tsa.gov — HTML structure may have changed")

    scraped = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) >= 2:
            try:
                dt = pd.to_datetime(cells[0]).normalize()
                vol = int(cells[1].replace(",", ""))
                scraped.append({"Date": dt, "Volume": vol})
            except (ValueError, IndexError):
                continue

    if not scraped:
        raise RuntimeError("Parsed 0 rows from tsa.gov page")

    new_df = pd.DataFrame(scraped)

    existing = pd.read_csv(TSA_PATH)
    existing.columns = [c.strip() for c in existing.columns]
    existing["Date"] = pd.to_datetime(existing["Date"])
    existing = existing.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    last_existing = existing["Date"].max()
    new_rows = new_df[new_df["Date"] > last_existing].copy()

    if new_rows.empty:
        print(f"  Already up-to-date through {last_existing.date()}")
        return 0

    print(
        f"  + {len(new_rows)} new row(s) "
        f"({new_rows['Date'].min().date()} -> {new_rows['Date'].max().date()})"
    )

    updated = pd.concat(
        [existing[["Date", "Volume"]], new_rows[["Date", "Volume"]]],
        ignore_index=True,
    )
    updated = updated.sort_values("Date").drop_duplicates("Date", keep="last")
    updated.to_csv(TSA_PATH, index=False, date_format="%Y-%m-%d")
    print(f"  Saved -> {TSA_PATH.name}")

    try:
        import db
        db.write_tsa_actuals(new_rows)
    except Exception as exc:
        import sys
        print(f"[db] write failed (non-fatal): {exc}", file=sys.stderr)

    return len(new_rows)


if __name__ == "__main__":
    refresh_tsa()
