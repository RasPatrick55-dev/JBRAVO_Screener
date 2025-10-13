"""Connectivity health probe for Alpaca trading and data APIs."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Mapping

import requests

from scripts.utils.env import market_data_base_url, trading_base_url
from utils.env import get_alpaca_creds, load_env


LOG = logging.getLogger("health_check")

CONNECTIVITY_PATH = os.path.join("data", "health", "connectivity.json")
_PROBE_SYMBOLS = ("AAPL", "MSFT", "SPY")


def _alpaca_headers() -> Dict[str, str]:
    key, secret, _, _ = get_alpaca_creds()
    headers: Dict[str, str] = {}
    if key:
        headers["APCA-API-KEY-ID"] = key.strip()
    if secret:
        headers["APCA-API-SECRET-KEY"] = secret.strip()
    return headers


def _error_message(response: requests.Response, url: str) -> str:
    snippet = ""
    try:
        snippet = response.text.strip()
    except Exception:  # pragma: no cover - defensive
        snippet = ""
    if not snippet:
        try:
            payload = response.json()
        except Exception:  # pragma: no cover - defensive
            payload = None
        if isinstance(payload, Mapping):
            snippet = json.dumps(payload)
        elif payload is not None:
            snippet = str(payload)
    snippet = " ".join(str(snippet).split())
    reason = response.reason or "error"
    return f"GET {url} -> {response.status_code} {reason}: {snippet}".strip()


def _probe_trading(session: requests.Session, headers: Mapping[str, str]) -> Dict[str, Any]:
    url = f"{trading_base_url().rstrip('/')}/v2/account"
    if "APCA-API-KEY-ID" not in headers or "APCA-API-SECRET-KEY" not in headers:
        return {
            "ok": False,
            "status": 0,
            "message": "missing APCA_API_KEY_ID/APCA_API_SECRET_KEY",
        }
    try:
        resp = session.get(url, headers=headers, timeout=10)
    except Exception as exc:  # pragma: no cover - network errors
        return {"ok": False, "status": 0, "message": f"GET {url} -> {exc}"}

    if resp.ok:
        message = ""
        try:
            payload = resp.json()
        except Exception:  # pragma: no cover - non JSON
            payload = None
        if isinstance(payload, Mapping):
            message = str(payload.get("status") or payload.get("account_status") or "ok")
        else:
            message = resp.text.strip() or resp.reason or "ok"
    else:
        message = _error_message(resp, url)

    return {"ok": resp.ok, "status": resp.status_code, "message": message}


def _probe_data(
    session: requests.Session, headers: Mapping[str, str], feed: str
) -> Dict[str, Any]:
    base = market_data_base_url().rstrip("/")
    url = f"{base}/v2/stocks/bars"
    params = {
        "symbols": ",".join(_PROBE_SYMBOLS),
        "timeframe": "1Day",
        "limit": 252,
        "feed": feed or "iex",
    }
    try:
        resp = session.get(url, headers=headers, params=params, timeout=10)
    except Exception as exc:  # pragma: no cover - network errors
        return {"ok": False, "status": 0, "message": f"GET {url} -> {exc}"}

    if resp.ok:
        message = ""
        try:
            payload = resp.json()
        except Exception:  # pragma: no cover - non JSON
            payload = None
        if isinstance(payload, Mapping):
            bars = payload.get("bars")
            if isinstance(bars, Mapping):
                parts = []
                for symbol in _PROBE_SYMBOLS:
                    series = bars.get(symbol)
                    count = len(series) if isinstance(series, list) else 0
                    parts.append(f"{symbol}:{count}")
                message = ",".join(parts)
            else:
                message = json.dumps(payload)[:200]
        else:
            message = resp.text.strip() or resp.reason or "ok"
    else:
        message = _error_message(resp, f"{url}?symbols={params['symbols']}&feed={params['feed']}")

    return {"ok": resp.ok, "status": resp.status_code, "message": message}


def run_health_check(*, write: bool = True) -> Dict[str, Any]:
    """Run connectivity probes and optionally persist their outcome."""

    load_env()
    headers = _alpaca_headers()
    _, _, _, feed = get_alpaca_creds()

    session = requests.Session()
    try:
        trading = _probe_trading(session, headers)
        data = _probe_data(session, headers, (feed or "iex").lower())
    finally:
        session.close()

    report: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "trading": trading,
        "data": data,
    }

    if write:
        os.makedirs(os.path.dirname(CONNECTIVITY_PATH), exist_ok=True)
        with open(CONNECTIVITY_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    return report


def probe_trading_only() -> Dict[str, Any]:
    """Run only the trading probe (without writing artifacts)."""

    load_env()
    headers = _alpaca_headers()
    session = requests.Session()
    try:
        return _probe_trading(session, headers)
    finally:
        session.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = run_health_check(write=True)
    trading = report.get("trading", {})
    data = report.get("data", {})
    LOG.info(
        "[INFO] HEALTH trading_ok=%s data_ok=%s trading_status=%s data_status=%s",
        trading.get("ok"),
        data.get("ok"),
        trading.get("status"),
        data.get("status"),
    )
    return 0 if trading.get("ok") and data.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
