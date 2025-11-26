from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urljoin

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

CONNECTION_HEALTH_JSON = DATA_DIR / "connection_health.json"
ERROR_LOG = LOGS_DIR / "error.log"
LOCAL_LOG = LOGS_DIR / "connection_health.log"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv_if_needed(path: Path | None = None) -> None:
    """
    Lightweight .env loader for when tasks forgot to `set -a; . .env; set +a`.

    Only fills in keys that are currently missing from os.environ.
    """

    if path is None:
        path = Path(os.path.expanduser("~/.config/jbravo/.env"))
    if not path.exists():
        return

    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Best-effort helper; don't crash probe if .env parsing fails
        pass


def _setup_logger() -> logging.Logger:
    _ensure_dirs()
    logger = logging.getLogger("connection_health")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(LOCAL_LOG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    err_fh = logging.FileHandler(ERROR_LOG)
    err_fh.setLevel(logging.ERROR)
    err_fh.setFormatter(fmt)
    logger.addHandler(err_fh)

    return logger


def _probe_trading(logger: logging.Logger) -> Dict[str, Any]:
    base = (os.getenv("APCA_API_BASE_URL") or "").rstrip("/")
    key = os.getenv("APCA_API_KEY_ID") or ""
    secret = os.getenv("APCA_API_SECRET_KEY") or ""

    if not base or not key or not secret:
        msg = "Missing trading env vars (APCA_API_BASE_URL/APCA_API_KEY_ID/APCA_API_SECRET_KEY)"
        logger.error(msg)
        return {"ok": False, "status": 0, "message": msg}

    url = urljoin(base + "/", "v2/account")
    try:
        resp = requests.get(
            url,
            headers={
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            },
            timeout=10,
        )
        ok = resp.status_code == 200
        message = "" if ok else f"HTTP {resp.status_code}: {resp.text[:200]}"
        if ok:
            logger.info("Trading probe OK status=%s", resp.status_code)
        else:
            logger.error(
                "Trading probe FAILED status=%s body=%s", resp.status_code, resp.text[:200]
            )
        return {"ok": ok, "status": resp.status_code, "message": message}
    except Exception as exc:  # pragma: no cover - network guard
        logger.exception("Trading probe exception")
        return {"ok": False, "status": 0, "message": str(exc)}


def _probe_data(logger: logging.Logger) -> Dict[str, Any]:
    """
    Use the v2 'latest bars' endpoint as a lightweight data health check.

    https://data.alpaca.markets/v2/stocks/bars/latest
    """

    base = (os.getenv("APCA_DATA_API_BASE_URL") or "https://data.alpaca.markets").rstrip("/")
    key = os.getenv("APCA_API_KEY_ID") or ""
    secret = os.getenv("APCA_API_SECRET_KEY") or ""
    feed = os.getenv("ALPACA_DATA_FEED") or "iex"

    if not key or not secret:
        msg = "Missing data env vars (APCA_API_KEY_ID/APCA_API_SECRET_KEY)"
        logger.error(msg)
        return {"ok": False, "status": 0, "message": msg}

    url = urljoin(base + "/", "v2/stocks/bars/latest")
    params = {"symbols": "AAPL", "feed": feed}
    try:
        resp = requests.get(
            url,
            params=params,
            headers={
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            },
            timeout=10,
        )
        ok = resp.status_code == 200
        message = "" if ok else f"HTTP {resp.status_code}: {resp.text[:200]}"
        if ok:
            logger.info("Data probe OK status=%s", resp.status_code)
        else:
            logger.error(
                "Data probe FAILED status=%s body=%s", resp.status_code, resp.text[:200]
            )
        return {"ok": ok, "status": resp.status_code, "message": message}
    except Exception as exc:  # pragma: no cover - network guard
        logger.exception("Data probe exception")
        return {"ok": False, "status": 0, "message": str(exc)}


def run_probe() -> Dict[str, Any]:
    """
    Run both probes and persist connection_health.json.
    Returns the dict that was written.
    """

    _load_dotenv_if_needed()
    logger = _setup_logger()

    trading = _probe_trading(logger)
    data = _probe_data(logger)

    payload: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trading_ok": bool(trading.get("ok")),
        "data_ok": bool(data.get("ok")),
        "trading": trading,
        "data": data,
    }

    try:
        _ensure_dirs()
        with CONNECTION_HEALTH_JSON.open("w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote %s", CONNECTION_HEALTH_JSON)
    except Exception as exc:  # pragma: no cover - filesystem guard
        logger.exception("Failed to write connection_health.json: %s", exc)

    return payload


if __name__ == "__main__":
    run_probe()
