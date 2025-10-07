"""HTTP helpers for Alpaca market data."""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Iterable, List, Tuple

import requests

from .env import market_data_base_url
from .rate import TokenBucket

LOGGER = logging.getLogger(__name__)


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
) -> Tuple[list[dict], Dict[str, int]]:
    """Fetch daily bars via Alpaca's REST API with pagination support."""

    if not symbols:
        return [], {"rate_limited": 0, "pages": 0, "requests": 0, "chunks": 0}

    base = market_data_base_url()
    url = f"{base}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID"),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY"),
    }
    limiter = TokenBucket(rate_limit)
    output: list[dict] = []
    metrics: Dict[str, int] = {
        "rate_limited": 0,
        "pages": 0,
        "requests": 0,
        "chunks": 0,
    }
    rate_logged = False
    not_found_logged = False

    for chunk in _batched(symbols, max(1, min(chunk_size, 50))):
        metrics["chunks"] += 1
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
            metrics["requests"] += 1
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 404:
                if not not_found_logged:
                    sample = ",".join(chunk[:5])
                    LOGGER.info(
                        "No bars returned for request chunk (size=%d sample=%s)",
                        len(chunk),
                        sample,
                    )
                    not_found_logged = True
                break
            if response.status_code == 429:
                if not rate_logged:
                    LOGGER.warning("Alpaca rate limit hit when fetching bars; retrying once")
                    rate_logged = True
                metrics["rate_limited"] += 1
                time.sleep(1.0)
                metrics["requests"] += 1
                response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json() or {}
            output.extend(payload.get("bars", []) or [])
            metrics["pages"] += 1
            page_token = payload.get("next_page_token")
            if not page_token:
                break
            time.sleep(sleep_s)
        time.sleep(sleep_s)
    return output, metrics


__all__ = ["fetch_bars_http"]

