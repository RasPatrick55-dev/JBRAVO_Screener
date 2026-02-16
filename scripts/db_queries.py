"""Database query helpers for run-scoped screener candidate reads."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

import pandas as pd

from scripts import db

LOGGER = logging.getLogger("db_queries")


def _coerce_run_date(run_date: Any | None) -> Optional[date]:
    if run_date is None:
        return None
    try:
        value = pd.to_datetime(run_date, utc=True)
    except Exception:
        return None
    if pd.isna(value):
        return None
    return value.date()


def get_latest_screener_candidates(
    run_date: Any,
    *,
    limit: int | None = None,
) -> tuple[pd.DataFrame, Any | None]:
    """Return candidates scoped to the latest screener run timestamp for ``run_date``."""

    conn = db.get_db_conn()
    if conn is None:
        return pd.DataFrame(), None

    run_date_value = _coerce_run_date(run_date)
    if run_date_value is None:
        try:
            conn.close()
        except Exception:
            pass
        return pd.DataFrame(), None

    latest_run_ts = None
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT COALESCE(
                    max(run_ts_utc),
                    max(created_at)
                ) AS latest_run_ts
                FROM screener_candidates
                WHERE run_date = %(run_date)s
                """,
                {"run_date": run_date_value},
            )
            row = cursor.fetchone()
            latest_run_ts = row[0] if row else None

            if latest_run_ts is None:
                LOGGER.info(
                    "DB_QUERY latest_screener_candidates run_date=%s latest_run_ts=NULL count=0",
                    run_date_value,
                )
                return pd.DataFrame(), None

            params: dict[str, Any] = {
                "run_date": run_date_value,
                "latest_run_ts": latest_run_ts,
            }
            limit_sql = ""
            if limit is not None and int(limit) > 0:
                params["limit"] = int(limit)
                limit_sql = " LIMIT %(limit)s"

            cursor.execute(
                (
                    """
                    SELECT run_date, timestamp, symbol, score, exchange, close, volume,
                           universe_count, score_breakdown, entry_price, adv20, atrp, source,
                           final_score, sma9, ema20, sma180, rsi14, passed_gates,
                           gate_fail_reason, ml_weight_used, run_ts_utc, created_at
                    FROM screener_candidates
                    WHERE run_date = %(run_date)s
                      AND run_ts_utc = %(latest_run_ts)s
                    ORDER BY score DESC NULLS LAST, symbol ASC
                    """
                    + limit_sql
                ),
                params,
            )
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description or []]
    finally:
        try:
            conn.close()
        except Exception:
            pass

    frame = pd.DataFrame(rows, columns=columns)
    LOGGER.info(
        "DB_QUERY latest_screener_candidates run_date=%s latest_run_ts=%s count=%s",
        run_date_value,
        latest_run_ts,
        len(frame.index),
    )
    return frame, latest_run_ts
