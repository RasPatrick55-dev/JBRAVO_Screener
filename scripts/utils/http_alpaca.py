"""HTTP helpers for Alpaca market data."""
from __future__ import annotations

import os
import time
from typing import Iterable, List

import requests

from .rate import TokenBucket


def _batched(symbols: Iterable[str], size: int) -> Iterable[List[str]]:
    batch: List[str] = []
    for sym in symbols:
        batch.append(sym)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def fetch_bars_http(
    symbols: list[str],
    start: str,
    end: str,
    *,
    timeframe: str = "1Day",
    feed: str = "iex",
    per_page: int = 10_000,
    chunk_size: int = 50,
    rate_limit: int = 200,
    sleep_s: float = 0.35,
) -> list[dict]:
    """Fetch daily bars via Alpaca's REST API with pagination support."""

    if not symbols:
        return []

    base = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    url = f"{base}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID"),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY"),
    }
    limiter = TokenBucket(rate_limit)
    output: list[dict] = []

    for chunk in _batched(symbols, max(1, min(chunk_size, 50))):
        page_token: str | None = None
        while True:
            params = {
                "symbols": ",".join(chunk),
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "feed": feed,
                "limit": per_page,
            }
            if page_token:
                params["page_token"] = page_token
            limiter.acquire()
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json() or {}
            output.extend(payload.get("bars", []) or [])
            page_token = payload.get("next_page_token")
            if not page_token:
                break
            time.sleep(sleep_s)
        time.sleep(sleep_s)
    return output


__all__ = ["fetch_bars_http"]

