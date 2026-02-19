"""Fetch and persist an Alpaca account snapshot.

Usage:
    APCA_API_BASE_URL=... APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... DATABASE_URL=... \\
        python -m scripts.fetch_account_snapshot

The script is paper-mode safe (logs the target base_url) and writes structured
messages to ``logs/account_snapshot.log``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import requests

from scripts import db
from scripts.utils.env import load_env, trading_base_url

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "account_snapshot.log"


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("account_snapshot")
    if logger.handlers:
        return logger

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger


def _alpaca_headers() -> Mapping[str, str]:
    return {
        "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
    }


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _fetch_snapshot(
    session: requests.Session, base_url: str, logger: logging.Logger
) -> Mapping[str, Any] | None:
    url = f"{base_url}/v2/account"
    logger.info(
        "[INFO] ACCT_SNAP_START base_url=%s paper_mode=%s", base_url, "paper" in base_url.lower()
    )
    try:
        response = session.get(url, headers=_alpaca_headers(), timeout=15)
    except Exception as exc:
        logger.error("[ERROR] ACCT_SNAP_FAIL step=fetch err=%s", exc)
        return None

    body_snippet = ""
    try:
        body_snippet = " ".join((response.text or "").split())[:500]
    except Exception:
        body_snippet = ""

    if not response.ok:
        logger.error(
            "[ERROR] ACCT_SNAP_FAIL step=fetch status=%s reason=%s body=%s",
            response.status_code,
            response.reason or "error",
            body_snippet,
        )
        return None

    try:
        payload: Mapping[str, Any] = response.json()
    except Exception as exc:
        logger.error(
            "[ERROR] ACCT_SNAP_FAIL step=parse status=%s err=%s body=%s",
            response.status_code,
            exc,
            body_snippet,
        )
        return None

    logger.info(
        "[INFO] ACCT_SNAP_FETCH_OK status=%s account_id=%s",
        response.status_code,
        payload.get("id") or payload.get("account_id") or "",
    )
    return payload


def _build_insert_payload(snapshot: Mapping[str, Any]) -> MutableMapping[str, Any]:
    taken_at = datetime.now(timezone.utc)
    return {
        "taken_at": taken_at,
        "account_id": snapshot.get("id") or snapshot.get("account_id"),
        "status": snapshot.get("status") or snapshot.get("account_status"),
        "cash": _coerce_float(snapshot.get("cash")),
        "cash_withdrawable": _coerce_float(snapshot.get("cash_withdrawable")),
        "portfolio_value": _coerce_float(snapshot.get("portfolio_value")),
        "equity": _coerce_float(snapshot.get("equity")),
        "buying_power": _coerce_float(snapshot.get("buying_power")),
        "effective_buying_power": _coerce_float(snapshot.get("effective_buying_power")),
        "maintenance_margin": _coerce_float(snapshot.get("maintenance_margin")),
        "daytrading_buying_power": _coerce_float(snapshot.get("daytrading_buying_power")),
        "multiplier": snapshot.get("multiplier"),
        "default_currency": snapshot.get("currency") or snapshot.get("default_currency"),
        "raw": json.dumps(snapshot),
    }


def _persist_snapshot(payload: Mapping[str, Any], logger: logging.Logger) -> bool:
    conn = db.get_db_conn()
    if conn is None:
        logger.error("[ERROR] ACCT_SNAP_FAIL step=db err=%s", "db_disabled")
        return False

    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO alpaca_account_snapshots (
                        taken_at,
                        account_id,
                        status,
                        cash,
                        cash_withdrawable,
                        portfolio_value,
                        equity,
                        buying_power,
                        effective_buying_power,
                        maintenance_margin,
                        daytrading_buying_power,
                        multiplier,
                        default_currency,
                        raw
                    )
                    VALUES (
                        %(taken_at)s,
                        %(account_id)s,
                        %(status)s,
                        %(cash)s,
                        %(cash_withdrawable)s,
                        %(portfolio_value)s,
                        %(equity)s,
                        %(buying_power)s,
                        %(effective_buying_power)s,
                        %(maintenance_margin)s,
                        %(daytrading_buying_power)s,
                        %(multiplier)s,
                        %(default_currency)s,
                        CAST(%(raw)s AS JSONB)
                    )
                    """,
                    payload,
                )
    except Exception as exc:
        logger.error("[ERROR] ACCT_SNAP_FAIL step=db err=%s", exc)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass

    logger.info(
        "[INFO] ACCT_SNAP_DB_OK inserted=1 taken_at=%s equity=%s cash=%s buying_power=%s",
        payload.get("taken_at"),
        payload.get("equity"),
        payload.get("cash"),
        payload.get("buying_power"),
    )
    # metrics_daily schema does not currently expose equity/cash/buying_power; skipping rollup.
    return True


def main() -> int:
    logger = _setup_logger()
    _, missing = load_env(
        required_keys=("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "APCA_API_BASE_URL")
    )
    if missing:
        logger.error("[ERROR] ACCT_SNAP_FAIL step=env missing=%s", ",".join(missing))
        return 1

    base_url = trading_base_url().rstrip("/")
    session = requests.Session()
    snapshot = _fetch_snapshot(session, base_url, logger)
    if not snapshot:
        return 1

    payload = _build_insert_payload(snapshot)
    ok = _persist_snapshot(payload, logger)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
