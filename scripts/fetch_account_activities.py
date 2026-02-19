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
from psycopg2 import extras
from psycopg2.extensions import connection as PGConnection

from scripts import db


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "logs", "activities_ingestor.log")
BACKFILL_LOG_PATH = os.path.join(BASE_DIR, "logs", "activities_backfill.log")

logger = logging.getLogger(__name__)

WATERMARK_KEY = "alpaca_activities_watermark"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_PAGE_SIZE = 100
DEFAULT_CHUNK_DAYS = 30
DEFAULT_MAX_PAGES = 2000
MAX_PAGE_SIZE = 100


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


def _table_columns(engine: Optional[PGConnection], table: str) -> set[str]:
    if engine is None:
        return set()
    try:
        with engine.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema=current_schema()
                  AND table_name=%(table)s
                """,
                {"table": table},
            )
            rows = cursor.fetchall()
            return {row[0] for row in rows}
    except Exception:
        return set()


def load_watermark(engine: Optional[PGConnection]) -> Optional[str]:
    columns = _table_columns(engine, "reconcile_state")
    if {"key", "value"} <= columns:
        try:
            with engine.cursor() as cursor:
                cursor.execute(
                    "SELECT value FROM reconcile_state WHERE key=%(key)s LIMIT 1",
                    {"key": WATERMARK_KEY},
                )
                row = cursor.fetchone()
                return str(row[0]) if row and row[0] is not None else None
        except Exception:
            return None

    if {"id", "last_after"} <= columns:
        try:
            with engine.cursor() as cursor:
                cursor.execute(
                    "SELECT last_after FROM reconcile_state WHERE id=%(id)s LIMIT 1",
                    {"id": 2},
                )
                row = cursor.fetchone()
                if not row or row[0] is None:
                    return None
                if isinstance(row[0], datetime):
                    return row[0].isoformat()
                return str(row[0])
        except Exception:
            return None

    return None


def save_watermark(engine: Optional[PGConnection], value: str) -> bool:
    columns = _table_columns(engine, "reconcile_state")
    now_ts = datetime.now(timezone.utc)

    if {"key", "value"} <= columns:
        try:
            with engine:
                with engine.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO reconcile_state (key, value, updated_at)
                        VALUES (%(key)s, %(value)s, %(updated_at)s)
                        ON CONFLICT (key) DO UPDATE
                        SET value=EXCLUDED.value,
                            updated_at=EXCLUDED.updated_at
                        """,
                        {"key": WATERMARK_KEY, "value": value, "updated_at": now_ts},
                    )
            return True
        except Exception:
            return False

    if {"id", "last_after"} <= columns:
        try:
            parsed = _parse_timestamp(value)
            with engine:
                with engine.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO reconcile_state (id, last_after, last_ran_at, updated_at)
                        VALUES (%(id)s, %(last_after)s, %(last_ran_at)s, %(updated_at)s)
                        ON CONFLICT (id) DO UPDATE SET
                            last_after=EXCLUDED.last_after,
                            last_ran_at=EXCLUDED.last_ran_at,
                            updated_at=EXCLUDED.updated_at
                        """,
                        {
                            "id": 2,
                            "last_after": parsed or value,
                            "last_ran_at": now_ts,
                            "updated_at": now_ts,
                        },
                    )
            return True
        except Exception:
            return False

    return False


def _configure_logging(mode: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_file = BACKFILL_LOG_PATH if mode == "backfill" else LOG_PATH
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


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
        token = (
            payload.get("next_page_token") or payload.get("page_token") or payload.get("next_token")
        )
        if token:
            return str(token)
    return None


def _request_activity_page(
    base_url: str, params: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    base = base_url.rstrip("/")
    url = f"{base}/v2/account/activities"
    headers = _build_headers()
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"activities_request_failed status={resp.status_code} body={resp.text}")
    try:
        payload = resp.json()
    except Exception as exc:
        raise RuntimeError(f"invalid_json_response err={exc}") from exc

    activities: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        activities.extend(payload)
    elif (
        isinstance(payload, dict)
        and "activities" in payload
        and isinstance(payload["activities"], list)
    ):
        activities.extend(payload["activities"])

    token = _pagination_token(resp, payload)
    return activities, token


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


def insert_activities(
    engine: Optional[PGConnection],
    activities: Iterable[Dict[str, Any]],
) -> Tuple[int, Optional[str], Optional[str]]:
    normalized_rows: List[Dict[str, Any]] = []
    watermark: Optional[str] = None
    earliest_ts: Optional[datetime] = None

    for activity in activities:
        normalized, wm = _normalize_activity(activity)
        if not normalized.get("activity_id") or not normalized.get("activity_type"):
            continue
        normalized_rows.append(normalized)
        if wm:
            watermark = wm
        if normalized.get("transaction_time"):
            candidate_ts = normalized["transaction_time"]
            if earliest_ts is None or (candidate_ts is not None and candidate_ts < earliest_ts):
                earliest_ts = candidate_ts

    if not normalized_rows:
        earliest_iso = earliest_ts.isoformat() if earliest_ts else None
        return 0, watermark, earliest_iso

    try:
        if engine is None:
            raise RuntimeError("db_disabled")
        with engine:
            with engine.cursor() as cursor:
                extras.execute_batch(
                    cursor,
                    """
                    INSERT INTO alpaca_activities (
                        activity_id, activity_type, transaction_time, symbol, side, qty, price,
                        amount, order_id, description, raw
                    )
                    VALUES (
                        %(activity_id)s, %(activity_type)s, %(transaction_time)s, %(symbol)s, %(side)s,
                        %(qty)s, %(price)s, %(amount)s, %(order_id)s, %(description)s, CAST(%(raw)s AS JSONB)
                    )
                    ON CONFLICT (activity_id, activity_type) DO NOTHING
                    """,
                    normalized_rows,
                    page_size=200,
                )
        earliest_iso = earliest_ts.isoformat() if earliest_ts else None
        return len(normalized_rows), watermark, earliest_iso
    except Exception as exc:
        raise RuntimeError(f"db_insert_failed err={exc}") from exc


def compute_since(
    args_since: Optional[str], watermark: Optional[str], lookback_days: int
) -> Tuple[Optional[str], str]:
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
    if missing:
        raise EnvironmentError(f"missing_env {' '.join(missing)}")


def _normalize_page_size(requested: int) -> int:
    try:
        sanitized = max(1, int(requested))
    except Exception:
        sanitized = 1

    if sanitized > MAX_PAGE_SIZE:
        logger.warning("[WARN] ACT_PAGE_SIZE_CLAMP requested=%s using=%s", sanitized, MAX_PAGE_SIZE)
        return MAX_PAGE_SIZE
    return sanitized


def fetch_activities(
    base_url: str,
    after: str | None,
    until: str | None,
    page_size: int,
    max_pages: int,
    *,
    direction: str = "asc",
) -> List[Dict[str, Any]]:
    max_pages = max(1, int(max_pages))
    page_size = _normalize_page_size(page_size)
    params: Dict[str, Any] = {
        "page_size": page_size,
        "direction": direction,
    }
    if after:
        params["after"] = after
    if until:
        params["until"] = until

    activities: List[Dict[str, Any]] = []
    page_count = 0
    while True:
        page_count += 1
        if page_count > max_pages:
            logger.warning(
                "[WARN] ACT_MAX_PAGES_REACHED page_count=%s max_pages=%s after=%s until=%s",
                page_count - 1,
                max_pages,
                after or "",
                until or "",
            )
            break
        results, token = _request_activity_page(base_url, params)
        activities.extend(results)
        if not token:
            break
        params["page_token"] = token

    return activities


def _is_newer_watermark(candidate: Optional[str], current: Optional[str]) -> bool:
    if not candidate:
        return False
    candidate_ts = _parse_timestamp(candidate)
    current_ts = _parse_timestamp(current) if current else None

    if candidate_ts and current_ts:
        return candidate_ts > current_ts
    if candidate_ts and not current_ts:
        return True
    if not candidate_ts and current_ts:
        return False
    if current is None:
        return True
    return str(candidate) > str(current)


def _run_incremental(
    args: argparse.Namespace, base_url: str, engine, watermark: Optional[str]
) -> None:
    since_iso, origin = compute_since(args.since_ts, watermark, args.lookback_days)
    max_pages = max(1, int(args.max_pages))
    page_size = _normalize_page_size(args.page_size)

    logger.info(
        "[INFO] ACT_START base_url=%s since=%s lookback_days=%s origin=%s page_size=%s max_pages=%s",
        base_url,
        since_iso,
        args.lookback_days,
        origin,
        page_size,
        max_pages,
    )

    activities = fetch_activities(base_url, since_iso, None, page_size, max_pages)
    logger.info("[INFO] ACT_FETCH_OK count=%s", len(activities))

    inserted, watermark_candidate, _ = insert_activities(engine, activities)
    logger.info("[INFO] ACT_DB_OK inserted=%s", inserted)

    if watermark_candidate:
        if save_watermark(engine, watermark_candidate):
            logger.info("[INFO] ACT_WATERMARK_OK value=%s", watermark_candidate)
        else:
            logger.error("[ERROR] ACT_WATERMARK_FAIL value=%s", watermark_candidate)


def _build_backfill_windows(
    start_ts: Optional[datetime], end_ts: Optional[datetime], chunk_days: int, backfill_all: bool
) -> Iterable[Tuple[datetime, datetime]]:
    chunk = max(1, int(chunk_days))
    now_ts = datetime.now(timezone.utc)
    effective_end = end_ts or now_ts
    effective_start = start_ts
    if effective_start is None:
        effective_start = effective_end - timedelta(days=chunk)
    if end_ts is None and start_ts is not None:
        effective_end = now_ts
    if end_ts is not None and start_ts is None:
        effective_start = end_ts - timedelta(days=chunk)

    if not backfill_all:
        yield effective_start, effective_end
        return

    current_end = effective_end
    current_start = effective_start
    while current_start < current_end:
        yield current_start, current_end
        current_end = current_start
        current_start = current_end - timedelta(days=chunk)


def _run_backfill(
    args: argparse.Namespace, base_url: str, engine, watermark: Optional[str]
) -> None:
    start_ts = _parse_timestamp(args.from_ts)
    end_ts = _parse_timestamp(args.to_ts)
    backfill_all = bool(args.backfill_all and not (start_ts and end_ts))
    chunk_days = max(1, int(args.chunk_days))
    page_size = _normalize_page_size(args.page_size)
    max_pages = max(1, int(args.max_pages))
    logger.info(
        "[INFO] ACT_BACKFILL_START from=%s to=%s chunk_days=%s page_size=%s max_pages=%s backfill_all=%s",
        start_ts.isoformat() if start_ts else "",
        end_ts.isoformat() if end_ts else "",
        chunk_days,
        page_size,
        max_pages,
        backfill_all,
    )

    total_inserted = 0
    earliest_seen_dt: Optional[datetime] = None
    watermark_candidate: Optional[str] = None

    for window_start, window_end in _build_backfill_windows(
        start_ts, end_ts, chunk_days, backfill_all
    ):
        if window_start >= window_end:
            logger.warning(
                "[WARN] ACT_SKIP_WINDOW reason=ambiguous_range window_start=%s window_end=%s",
                window_start.isoformat(),
                window_end.isoformat(),
            )
            continue

        params: Dict[str, Any] = {
            "after": window_start.isoformat(),
            "until": window_end.isoformat(),
            "page_size": page_size,
            "direction": "asc",
        }
        page_count = 0
        window_had_results = False
        page_token: Optional[str] = None

        while True:
            page_count += 1
            if page_count > max_pages:
                logger.warning(
                    "[WARN] ACT_MAX_PAGES_REACHED page_count=%s max_pages=%s window_start=%s window_end=%s",
                    page_count - 1,
                    max_pages,
                    window_start.isoformat(),
                    window_end.isoformat(),
                )
                break

            if page_token:
                params["page_token"] = page_token
            elif "page_token" in params:
                params.pop("page_token", None)

            page_results, token = _request_activity_page(base_url, params)
            inserted, wm_candidate, earliest_iso = insert_activities(engine, page_results)
            window_had_results = window_had_results or bool(page_results)
            total_inserted += inserted

            if earliest_iso:
                parsed_earliest = _parse_timestamp(earliest_iso)
                if parsed_earliest and (
                    earliest_seen_dt is None or parsed_earliest < earliest_seen_dt
                ):
                    earliest_seen_dt = parsed_earliest

            if wm_candidate and _is_newer_watermark(wm_candidate, watermark_candidate):
                watermark_candidate = wm_candidate

            logger.info(
                "[INFO] ACT_BACKFILL_PAGE fetched=%s inserted=%s window_start=%s window_end=%s",
                len(page_results),
                inserted,
                window_start.isoformat(),
                window_end.isoformat(),
            )

            if not token:
                break
            page_token = token

        if backfill_all and not window_had_results:
            break

    earliest_seen = earliest_seen_dt.isoformat() if earliest_seen_dt else ""
    logger.info(
        "[INFO] ACT_BACKFILL_DONE total_inserted=%s earliest_seen=%s", total_inserted, earliest_seen
    )

    if watermark_candidate and _is_newer_watermark(watermark_candidate, watermark):
        if save_watermark(engine, watermark_candidate):
            logger.info("[INFO] ACT_WATERMARK_OK value=%s", watermark_candidate)
        else:
            logger.error("[ERROR] ACT_WATERMARK_FAIL value=%s", watermark_candidate)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Alpaca account activities into Postgres.")
    parser.add_argument(
        "--mode",
        type=str,
        default="incremental",
        choices=["incremental", "backfill"],
        help="Ingestion mode: incremental (default) or backfill.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Number of days to look back when no watermark exists (incremental mode).",
    )
    parser.add_argument(
        "--since-ts",
        type=str,
        default=None,
        help="ISO-8601 timestamp to start fetching from (overrides lookback when provided, incremental mode).",
    )
    parser.add_argument(
        "--from-ts", type=str, default=None, help="Backfill start timestamp (ISO-8601)."
    )
    parser.add_argument(
        "--to-ts", type=str, default=None, help="Backfill end timestamp (ISO-8601)."
    )
    parser.add_argument(
        "--backfill-all",
        action="store_true",
        help="When set, walk backwards in chunked windows until no more data is returned.",
    )
    parser.add_argument(
        "--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="API page size (default 100)."
    )
    parser.add_argument(
        "--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS, help="Days per backfill chunk."
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Maximum pages to request per window (safety cap).",
    )
    args = parser.parse_args(argv)

    try:
        _configure_logging(args.mode)
        load_dotenv()
        validate_env()
        base_url = os.getenv("APCA_API_BASE_URL", DEFAULT_BASE_URL)
        conn = db.get_db_conn()
        if conn is None:
            raise RuntimeError("db_unavailable")

        try:
            watermark = load_watermark(conn)
            if args.mode == "backfill":
                _run_backfill(args, base_url, conn, watermark)
            else:
                _run_incremental(args, base_url, conn, watermark)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("[ERROR] ACT_FAIL %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
