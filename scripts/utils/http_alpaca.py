"""HTTP helpers for Alpaca market data."""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Iterable, List, Tuple

import requests

from .env import market_data_base_url

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


def _coerce_bars(payload: dict) -> list[dict]:
    bars = payload.get("bars", []) if isinstance(payload, dict) else []
    if isinstance(bars, list):
        return [bar for bar in bars if isinstance(bar, dict)]
    if isinstance(bars, dict):
        collected: list[dict] = []
        for symbol, entries in bars.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                record = dict(entry)
                record.setdefault("symbol", symbol)
                collected.append(record)
        return collected
    return []


def fetch_bars_http(
    symbols: list[str],
    start: str,
    end: str,
    *,
    timeframe: str = "1Day",
    feed: str = "iex",
    per_page: int = 10_000,
    sleep_s: float = 0.35,
    verify_hook=None,
    first_page_callback=None,
) -> Tuple[list[dict], Dict[str, int]]:
    """Fetch daily bars via Alpaca's REST API with pagination support."""

    if not symbols:
        empty_metrics = {
            "rate_limited": 0,
            "pages": 0,
            "requests": 0,
            "chunks": 0,
            "http_404_batches": 0,
            "http_empty_batches": 0,
        }
        return [], empty_metrics

    base = market_data_base_url()
    url = f"{base}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID"),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY"),
    }
    output: list[dict] = []
    metrics: Dict[str, int] = {
        "rate_limited": 0,
        "pages": 0,
        "requests": 0,
        "chunks": 0,
        "http_404_batches": 0,
        "http_empty_batches": 0,
    }
    first_request_logged = False
    first_page_notified = False

    for chunk in _batched(symbols, 50):
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
            if verify_hook and not first_request_logged:
                try:
                    verify_hook(url, params)
                except Exception:  # pragma: no cover - diagnostics must not break fetch
                    LOGGER.debug("Verify hook failed for request preview", exc_info=True)
                first_request_logged = True
            metrics["requests"] += 1
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 429:
                metrics["rate_limited"] += 1
                time.sleep(1.0)
                metrics["requests"] += 1
                response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 404:
                metrics["http_404_batches"] += 1
                break
            response.raise_for_status()
            payload = response.json() or {}
            if not first_page_notified:
                first_page_notified = True
                if first_page_callback is not None:
                    try:
                        first_page_callback(payload)
                    except Exception:  # pragma: no cover - diagnostics only
                        LOGGER.debug("First-page callback failed", exc_info=True)
            bars = _coerce_bars(payload)
            if not bars:
                metrics["http_empty_batches"] += 1
            else:
                output.extend(bars)
            metrics["pages"] += 1
            page_token = payload.get("next_page_token") if isinstance(payload, dict) else None
            if not page_token:
                break
            time.sleep(sleep_s)
        time.sleep(sleep_s)

    return output, metrics


__all__ = ["fetch_bars_http"]

