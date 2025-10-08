"""HTTP helpers for Alpaca market data."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import requests

from .env import market_data_base_url


def fetch_bars_http(
    symbols: List[str],
    start: str,
    end: str,
    timeframe: str = "1Day",
    feed: str = "iex",
    per_page: int = 10000,
    sleep_s: float = 0.35,
    verify_hook=None,
) -> Tuple[List[dict], Dict[str, int]]:
    base = market_data_base_url()
    url = f"{base}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID"),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY"),
    }
    out: List[dict] = []
    http_404 = 0
    http_empty = 0
    rate_limited = 0
    first = True
    for i in range(0, len(symbols), 50):
        chunk = symbols[i : i + 50]
        page = None
        while True:
            params = {
                "symbols": ",".join(chunk),
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "feed": feed,
                "limit": per_page,
            }
            if page:
                params["page_token"] = page
            if verify_hook and first:
                verify_hook(url, params)
                first = False

            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                rate_limited += 1
                time.sleep(1.0)
                resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 404:
                http_404 += 1
                break
            resp.raise_for_status()
            data = resp.json()
            bars = data.get("bars", []) if isinstance(data, dict) else []
            if not bars:
                http_empty += 1
            else:
                out.extend(bars)
            page = data.get("next_page_token") if isinstance(data, dict) else None
            if not page:
                break
            time.sleep(sleep_s)
        time.sleep(sleep_s)
    total = len(out)
    return out, {
        "http_404_batches": http_404,
        "http_empty_batches": http_empty,
        "rate_limited": rate_limited,
        "raw_bars_count": total,
        "parsed_rows_count": total,
    }
