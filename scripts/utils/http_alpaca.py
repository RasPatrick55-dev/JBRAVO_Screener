"""HTTP helpers for Alpaca market data."""

from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Tuple

import requests

from utils.env import AlpacaUnauthorizedError


def _flatten_to_canonical(data: Dict) -> List[Dict]:
    """
    Canonicalize to a flat list of dicts with keys:
      symbol, timestamp, open, high, low, close, volume
    Handles both:
      A) {"bars": {"AAPL":[{t,o,h,l,c,v},...], "MSFT":[...]}}
      B) {"bars": [{ "S":"AAPL", "t":..., "o":..., "h":..., "l":..., "c":..., "v":...}, ...]}
    """

    out: List[Dict] = []
    bars = data.get("bars", []) if isinstance(data, dict) else []

    def canon_one(sym: str, b: Dict) -> Dict:
        return {
            "symbol": (sym or b.get("S") or b.get("symbol") or "").upper(),
            "timestamp": b.get("t") or b.get("T") or b.get("time") or b.get("timestamp"),
            "open": b.get("o") or b.get("open"),
            "high": b.get("h") or b.get("high"),
            "low": b.get("l") or b.get("low"),
            "close": b.get("c") or b.get("close"),
            "volume": b.get("v") or b.get("volume"),
        }

    if isinstance(bars, dict):
        for sym, arr in bars.items():
            for b in arr or []:
                if isinstance(b, dict):
                    out.append(canon_one(sym, b))
    elif isinstance(bars, list):
        for b in bars:
            if not isinstance(b, dict):
                continue
            sym = b.get("S") or b.get("symbol")
            out.append(canon_one(sym, b))
    return out


def _rate_limit_guard(last_call: List[float], min_interval: float) -> None:
    """Sleep when the last request was within ``min_interval`` seconds."""

    now = time.monotonic()
    if last_call:
        elapsed = now - last_call[0]
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
    last_call[:] = [time.monotonic()]


def _chunk_symbols(symbols: Iterable[str], batch: int) -> Iterable[List[str]]:
    batch = max(1, int(batch))
    cleaned: List[str] = []
    for sym in symbols:
        if not sym:
            continue
        cleaned.append(str(sym).strip().upper())
    for start_idx in range(0, len(cleaned), batch):
        yield cleaned[start_idx : start_idx + batch]


def fetch_bars_http(
    symbols: List[str],
    start: str,
    end: str,
    feed: str = "iex",
    batch: int = 50,
    *,
    timeframe: str = "1Day",
    per_page: int = 10000,
    verify_hook=None,
) -> Tuple[List[dict], Dict[str, int]]:
    """Fetch daily bars for ``symbols`` via Alpaca's HTTP API.

    The client issues batched requests, following pagination tokens until
    exhaustion. A light-weight rate limiter targets ~3-4 requests per second.
    When a 429 response is observed the client backs off exponentially before
    retrying. HTTP 404 responses mark every symbol in the batch as a miss.
    """

    base = (os.getenv("APCA_DATA_API_BASE_URL") or "https://data.alpaca.markets").rstrip("/")
    url = f"{base}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID"),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY"),
    }

    rows: List[dict] = []
    stats: Dict[str, int] = {
        "requests": 0,
        "rows": 0,
        "rate_limit_hits": 0,
        "retries": 0,
        "pages": 0,
        "http_404_batches": 0,
        "http_empty_batches": 0,
        "chunks": 0,
    }
    missed: set[str] = set()
    first_hook = True
    last_call: List[float] = []

    for chunk in _chunk_symbols(symbols, batch):
        if not chunk:
            continue
        stats["chunks"] += 1
        page_token = None
        consecutive_429 = 0
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

            if verify_hook and first_hook:
                verify_hook(url, params)
                first_hook = False

            _rate_limit_guard(last_call, 0.28)
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            stats["requests"] += 1

            if resp.status_code in (401, 403):
                endpoint = getattr(resp.request, "path_url", "/v2/stocks/bars")
                raise AlpacaUnauthorizedError(endpoint=endpoint, feed=feed)

            if resp.status_code == 429:
                stats["rate_limit_hits"] += 1
                stats["retries"] += 1
                consecutive_429 += 1
                delay = min(4.0, 0.5 * (2 ** (consecutive_429 - 1)))
                time.sleep(delay)
                continue

            consecutive_429 = 0
            if resp.status_code == 404:
                stats["http_404_batches"] += 1
                missed.update(chunk)
                break

            resp.raise_for_status()

            try:
                data = resp.json()
            except Exception:
                data = {}

            flattened = _flatten_to_canonical(data if isinstance(data, dict) else {})
            if flattened:
                rows.extend(flattened)
                stats["rows"] += len(flattened)
                stats["pages"] += 1
            else:
                stats["http_empty_batches"] += 1

            page_token = data.get("next_page_token") if isinstance(data, dict) else None
            if not page_token:
                break
            stats["retries"] += 1

    if missed:
        stats["miss_symbols"] = len(missed)
        stats["miss_list"] = sorted(missed)
    else:
        stats["miss_symbols"] = 0
        stats["miss_list"] = []

    stats["rate_limited"] = stats["rate_limit_hits"]

    return rows, stats
