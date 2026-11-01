"""Status-only PAPER connectivity checks and the canonical saved-health artifact.

Exactly two fixed GETs at most. Response bodies and exception text are never
read, returned or logged. This module does not run a pipeline or place orders.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CONNECTION_HEALTH_JSON = DATA_DIR / "connection_health.json"
ERROR_LOG = LOGS_DIR / "error.log"
LOCAL_LOG = LOGS_DIR / "connection_health.log"
PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv_if_needed(path: Path | None = None) -> None:
    # Reuse the application's parser; quoted values/aliases must mean the same
    # thing here as in normal startup. Never echo values or source the shell.
    from utils.env import _load_env_file, _normalize_env_aliases
    paths = (path,) if path is not None else (
        Path.home() / ".config" / "jbravo" / ".env", BASE_DIR / ".env")
    for target in paths:
        _load_env_file(target, override=False)
    _normalize_env_aliases()


def _setup_logger() -> logging.Logger:
    _ensure_dirs()
    logger = logging.getLogger("connection_health")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        for path, level in ((LOCAL_LOG, logging.INFO), (ERROR_LOG, logging.ERROR)):
            handler = logging.FileHandler(path)
            handler.setLevel(level)
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger


def _refused(reason: str) -> Dict[str, Any]:
    return {"ok": False, "status": 0, "message": reason}


def _status_get(logger: logging.Logger, token: str, url: str, *, params=None) -> Dict[str, Any]:
    key = os.getenv("APCA_API_KEY_ID") or ""
    secret = os.getenv("APCA_API_SECRET_KEY") or ""
    if not key or not secret:
        result = _refused("CREDENTIALS_MISSING")
    else:
        response = None
        try:
            response = requests.get(url, params=params,
                headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
                timeout=(3, 5), allow_redirects=False, stream=True)
            status = response.status_code
            if type(status) is not int or not 100 <= status <= 599:
                result = _refused("HTTP_STATUS_INVALID")
            else:
                result = {"ok": status == 200, "status": status,
                          "message": "" if status == 200 else "HTTP_STATUS_NOT_200"}
        except requests.RequestException:
            result = _refused("REQUEST_FAILED")
        finally:
            if response is not None:
                try:
                    response.close()
                except (OSError, requests.RequestException):
                    result = _refused("RESPONSE_CLOSE_FAILED")
    logger.log(logging.INFO if result["ok"] else logging.ERROR,
               "%s probe ok=%s status=%s reason=%s", token,
               result["ok"], result["status"], result["message"])
    return result


def _probe_trading(logger: logging.Logger) -> Dict[str, Any]:
    if os.getenv("APCA_API_BASE_URL") != PAPER_URL:
        return _refused("PAPER_ENDPOINT_REQUIRED")
    if os.getenv("JBR_EXEC_PAPER") not in (None, "true"):
        return _refused("PAPER_CONTEXT_CONFLICT")
    return _status_get(logger, "trading", PAPER_URL + "/v2/account")


def _probe_data(logger: logging.Logger) -> Dict[str, Any]:
    if (os.getenv("APCA_DATA_API_BASE_URL") or DATA_URL) != DATA_URL:
        return _refused("DATA_ENDPOINT_REQUIRED")
    feed = os.getenv("ALPACA_DATA_FEED") or "iex"
    if feed not in ("iex", "sip"):
        return _refused("DATA_FEED_UNSUPPORTED")
    return _status_get(logger, "data", DATA_URL + "/v2/stocks/bars/latest",
                       params={"symbols": "AAPL", "feed": feed})


def _write_health(payload: Dict[str, Any]) -> None:
    wire = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(wire) > 65536:
        raise ValueError("HEALTH_ARTIFACT_OVERSIZE")
    _ensure_dirs()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=DATA_DIR, prefix=".connection_health.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(wire)
        if CONNECTION_HEALTH_JSON.exists():
            os.chmod(temporary, CONNECTION_HEALTH_JSON.stat().st_mode & 0o777)
        os.replace(temporary, CONNECTION_HEALTH_JSON)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_probe() -> Dict[str, Any]:
    _load_dotenv_if_needed()
    # Refuse a different account context before logging, API access or writes.
    if os.getenv("APCA_API_BASE_URL") != PAPER_URL or os.getenv("JBR_EXEC_PAPER") not in (None, "true"):
        raise ValueError("PAPER_CONTEXT_REQUIRED")
    logger = _setup_logger()
    trading, data = _probe_trading(logger), _probe_data(logger)
    payload = {"ts_utc": datetime.now(timezone.utc).isoformat(),
               "trading_ok": trading["ok"], "data_ok": data["ok"],
               "trading": trading, "data": data}
    _write_health(payload)
    logger.info("Wrote canonical connection_health.json")
    return payload


def main() -> int:
    try:
        payload = run_probe()
        return 0 if payload["trading_ok"] is True and payload["data_ok"] is True else 1
    except Exception:
        # A normal command must not leak private loader, request or filesystem
        # exceptions through a traceback. Failure never prints a fresh success.
        print(json.dumps({"status": "failed", "reason": "HEALTH_CHECK_FAILED"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
