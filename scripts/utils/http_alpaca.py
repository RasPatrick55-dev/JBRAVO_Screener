"""HTTP helpers for Alpaca market data."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import requests


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
    base = (os.getenv("APCA_DATA_API_BASE_URL") or "https://data.alpaca.markets").rstrip("/")
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
            flattened = _flatten_to_canonical(data if isinstance(data, dict) else {})
            if flattened:
                out.extend(flattened)
            else:
                http_empty += 1
            page = data.get("next_page_token") if isinstance(data, dict) else None
            if not page:
                break
            time.sleep(sleep_s)
        time.sleep(sleep_s)
    return out, {
        "http_404_batches": http_404,
        "http_empty_batches": http_empty,
        "rate_limited": rate_limited,
    }
