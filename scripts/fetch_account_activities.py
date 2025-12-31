import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from dateutil import parser as date_parser
from dotenv import load_dotenv
from sqlalchemy import text

from scripts import db


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "logs", "activities_ingestor.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

WATERMARK_KEY = "alpaca_activities_watermark"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


def _parse_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = date_parser.isoparse(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _table_columns(engine, table: str) -> set[str]:
    stmt = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name=:table
        """
    )
    try:
        with engine.connect() as connection:
            rows = connection.execute(stmt, {"table": table}).fetchall()
            return {row[0] for row in rows}
    except Exception:
        return set()


def load_watermark(engine) -> Optional[str]:
    columns = _table_columns(engine, "reconcile_state")
    if {"key", "value"} <= columns:
        stmt = text("SELECT value FROM reconcile_state WHERE key=:key LIMIT 1")
        try:
            with engine.connect() as connection:
                row = connection.execute(stmt, {"key": WATERMARK_KEY}).scalar()
                return str(row) if row is not None else None
        except Exception:
            return None

    if {"id", "last_after"} <= columns:
        stmt = text("SELECT last_after FROM reconcile_state WHERE id=:id LIMIT 1")
        try:
            with engine.connect() as connection:
                row = connection.execute(stmt, {"id": 2}).scalar()
                if row is None:
                    return None
                if isinstance(row, datetime):
                    return row.isoformat()
                return str(row)
        except Exception:
            return None

    return None


def save_watermark(engine, value: str) -> bool:
    columns = _table_columns(engine, "reconcile_state")
    now_ts = datetime.now(timezone.utc)

    if {"key", "value"} <= columns:
        stmt = text(
            """
            INSERT INTO reconcile_state (key, value, updated_at)
            VALUES (:key, :value, :updated_at)
            ON CONFLICT (key) DO UPDATE
            SET value=EXCLUDED.value,
                updated_at=EXCLUDED.updated_at
            """
        )
        try:
            with engine.begin() as connection:
                connection.execute(stmt, {"key": WATERMARK_KEY, "value": value, "updated_at": now_ts})
            return True
        except Exception:
            return False

    if {"id", "last_after"} <= columns:
        stmt = text(
            """
            INSERT INTO reconcile_state (id, last_after, last_ran_at, updated_at)
            VALUES (:id, :last_after, :last_ran_at, :updated_at)
            ON CONFLICT (id) DO UPDATE SET
                last_after=EXCLUDED.last_after,
                last_ran_at=EXCLUDED.last_ran_at,
                updated_at=EXCLUDED.updated_at
            """
        )
        try:
            parsed = _parse_timestamp(value)
            with engine.begin() as connection:
                connection.execute(
                    stmt,
                    {"id": 2, "last_after": parsed or value, "last_ran_at": now_ts, "updated_at": now_ts},
                )
            return True
        except Exception:
            return False

    return False


def _build_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _pagination_token(response: requests.Response, payload: Any) -> Optional[str]:
    header_keys = [
        "Next-Page-Token",
        "next-page-token",
        "X-Next-Page-Token",
        "x-next-page-token",
        "apca-next-page-token",
    ]
    for key in header_keys:
        token = response.headers.get(key)
        if token:
            return token
    if isinstance(payload, dict):
        token = payload.get("next_page_token") or payload.get("page_token") or payload.get("next_token")
        if token:
            return str(token)
    return None


def fetch_activities(base_url: str, since_iso: str | None) -> List[Dict[str, Any]]:
    base = base_url.rstrip("/")
    url = f"{base}/v2/account/activities"
    headers = _build_headers()
    params: Dict[str, Any] = {"page_size": 100, "direction": "asc"}
    if since_iso:
        params["after"] = since_iso

    activities: List[Dict[str, Any]] = []
    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"activities_request_failed status={resp.status_code} body={resp.text}")
        try:
            payload = resp.json()
        except Exception as exc:
            raise RuntimeError(f"invalid_json_response err={exc}") from exc

        if isinstance(payload, list):
            activities.extend(payload)
        elif isinstance(payload, dict) and "activities" in payload and isinstance(payload["activities"], list):
            activities.extend(payload["activities"])
        else:
            break

        token = _pagination_token(resp, payload)
        if not token:
            break
        params["page_token"] = token

    return activities


def _normalize_activity(activity: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    txn_time = _parse_timestamp(
        activity.get("transaction_time")
        or activity.get("processed_at")
        or activity.get("date")
        or activity.get("timestamp")
    )
    normalized = {
        "activity_id": activity.get("id") or activity.get("activity_id"),
        "activity_type": activity.get("activity_type") or activity.get("type"),
        "transaction_time": txn_time,
        "symbol": activity.get("symbol"),
        "side": activity.get("side"),
        "qty": _parse_decimal(activity.get("qty") or activity.get("quantity")),
        "price": _parse_decimal(activity.get("price")),
        "amount": _parse_decimal(activity.get("amount") or activity.get("net_amount")),
        "order_id": activity.get("order_id"),
        "description": activity.get("description") or activity.get("details"),
        "raw": json.dumps(activity),
    }

    watermark_candidate = None
    if txn_time is not None:
        watermark_candidate = txn_time.isoformat()
    elif normalized["activity_id"]:
        watermark_candidate = str(normalized["activity_id"])

    return normalized, watermark_candidate


def insert_activities(engine, activities: Iterable[Dict[str, Any]]) -> Tuple[int, Optional[str]]:
    normalized_rows: List[Dict[str, Any]] = []
    watermark: Optional[str] = None

    for activity in activities:
        normalized, wm = _normalize_activity(activity)
        if not normalized.get("activity_id") or not normalized.get("activity_type"):
            continue
        normalized_rows.append(normalized)
        if wm:
            watermark = wm

    if not normalized_rows:
        return 0, watermark

    stmt = text(
        """
        INSERT INTO alpaca_activities (
            activity_id, activity_type, transaction_time, symbol, side, qty, price,
            amount, order_id, description, raw
        )
        VALUES (
            :activity_id, :activity_type, :transaction_time, :symbol, :side, :qty, :price,
            :amount, :order_id, :description, CAST(:raw AS JSONB)
        )
        ON CONFLICT (activity_id, activity_type) DO NOTHING
        """
    )
    try:
        with engine.begin() as connection:
            connection.execute(stmt, normalized_rows)
        return len(normalized_rows), watermark
    except Exception as exc:
        raise RuntimeError(f"db_insert_failed err={exc}") from exc


def compute_since(args_since: Optional[str], watermark: Optional[str], lookback_days: int) -> Tuple[Optional[str], str]:
    since_candidate = args_since or watermark
    origin = "args" if args_since else "watermark" if watermark else "lookback"

    if since_candidate:
        ts = _parse_timestamp(since_candidate)
        if ts:
            return ts.isoformat(), origin

    fallback_ts = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    return fallback_ts.isoformat(), origin


def validate_env() -> None:
    missing = []
    if not os.getenv("APCA_API_KEY_ID"):
        missing.append("APCA_API_KEY_ID")
    if not os.getenv("APCA_API_SECRET_KEY"):
        missing.append("APCA_API_SECRET_KEY")
    if not os.getenv("DATABASE_URL"):
        missing.append("DATABASE_URL")
    if missing:
        raise EnvironmentError(f"missing_env {' '.join(missing)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Alpaca account activities into Postgres.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Number of days to look back when no watermark exists.",
    )
    parser.add_argument(
        "--since-ts",
        type=str,
        default=None,
        help="ISO-8601 timestamp to start fetching from (overrides lookback when provided).",
    )
    args = parser.parse_args(argv)

    try:
        load_dotenv()
        validate_env()
        base_url = os.getenv("APCA_API_BASE_URL", DEFAULT_BASE_URL)
        engine = db.get_engine()
        if engine is None:
            raise RuntimeError("db_unavailable")

        watermark = load_watermark(engine)
        since_iso, origin = compute_since(args.since_ts, watermark, args.lookback_days)

        logger.info(
            "[INFO] ACT_START base_url=%s since=%s lookback_days=%s origin=%s",
            base_url,
            since_iso,
            args.lookback_days,
            origin,
        )

        activities = fetch_activities(base_url, since_iso)
        logger.info("[INFO] ACT_FETCH_OK count=%s", len(activities))

        inserted, watermark_candidate = insert_activities(engine, activities)
        logger.info("[INFO] ACT_DB_OK inserted=%s", inserted)

        if watermark_candidate:
            if save_watermark(engine, watermark_candidate):
                logger.info("[INFO] ACT_WATERMARK_OK value=%s", watermark_candidate)
            else:
                logger.error("[ERROR] ACT_WATERMARK_FAIL value=%s", watermark_candidate)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("[ERROR] ACT_FAIL %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
