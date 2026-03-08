"""Database query helpers for run-scoped screener candidate reads."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

import pandas as pd

from scripts import db

LOGGER = logging.getLogger("db_queries")


def _coerce_symbol(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _log_model_score_join_diag(
    frame: pd.DataFrame,
    *,
    scores_rows_for_run: int,
    latest_run_ts: Any,
    score_col: str,
    reason_override: str | None = None,
) -> None:
    candidates = int(len(frame.index))
    if score_col in frame.columns:
        joined_series = pd.to_numeric(frame[score_col], errors="coerce")
    else:
        joined_series = pd.Series([None] * candidates, index=frame.index)
    joined_non_null = int(joined_series.notna().sum())
    joined_null = max(candidates - joined_non_null, 0)
    LOGGER.info(
        "[INFO] MODEL_SCORE_JOIN_DIAG candidates=%s scores_rows_for_run=%s joined_non_null=%s joined_null=%s run_ts_utc=%s score_col=%s",
        candidates,
        int(max(scores_rows_for_run, 0)),
        joined_non_null,
        joined_null,
        latest_run_ts,
        score_col,
    )
    if joined_null <= 0:
        return

    reason = reason_override
    if reason is None:
        if int(scores_rows_for_run or 0) <= 0:
            reason = "missing_scores_for_run"
        elif joined_non_null <= 0:
            reason = "symbol_mismatch_or_join_key_mismatch"
        else:
            reason = "partial_missing_scores"
    sample_symbols: list[str] = []
    if "symbol" in frame.columns:
        sample = frame.loc[joined_series.isna(), "symbol"].head(10)
        sample_symbols = [_coerce_symbol(value) for value in sample if _coerce_symbol(value)]
    LOGGER.info(
        "[INFO] MODEL_SCORE_JOIN_SAMPLE_UNMATCHED symbols=%s reason=%s",
        sample_symbols,
        reason,
    )


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


def _fetch_latest_candidate_rows(
    cursor: Any,
    *,
    run_date_value: date,
    latest_run_ts: Any,
    limit: int | None = None,
    include_ranker_scores: bool = True,
) -> tuple[list[Any], list[str]]:
    params: dict[str, Any] = {
        "run_date": run_date_value,
        "latest_run_ts": latest_run_ts,
    }
    limit_sql = ""
    if limit is not None and int(limit) > 0:
        params["limit"] = int(limit)
        limit_sql = " LIMIT %(limit)s"

    ranker_join = ""
    ranker_select = ""
    if include_ranker_scores:
        ranker_join = """
            LEFT JOIN screener_ranker_scores_app rs
              ON rs.run_ts_utc = c.run_ts_utc
             AND rs.symbol = UPPER(BTRIM(c.symbol))
        """
        ranker_select = ", rs.model_score_5d AS model_score_5d, rs.model_score_5d AS model_score"

    cursor.execute(
        (
            f"""
            SELECT c.run_date, c.timestamp, c.symbol, c.score, c.exchange, c.close, c.volume,
                   c.universe_count, c.score_breakdown, c.entry_price, c.adv20, c.atrp, c.source,
                   c.final_score, c.sma9, c.ema20, c.sma180, c.rsi14, c.passed_gates,
                   c.gate_fail_reason, c.ml_weight_used, c.run_ts_utc, c.created_at
                   {ranker_select}
            FROM screener_candidates c
            {ranker_join}
            WHERE c.run_date = %(run_date)s
              AND c.run_ts_utc = %(latest_run_ts)s
            ORDER BY c.score DESC NULLS LAST, c.symbol ASC
            """
            + limit_sql
        ),
        params,
    )
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description or []]
    return rows, columns


def _count_scores_rows_for_run(cursor: Any, latest_run_ts: Any) -> int:
    try:
        cursor.execute(
            """
            SELECT COUNT(*) AS row_count
            FROM screener_ranker_scores_app
            WHERE run_ts_utc = %(latest_run_ts)s
            """,
            {"latest_run_ts": latest_run_ts},
        )
        row = cursor.fetchone()
        return int((row[0] if row else 0) or 0)
    except Exception:
        return 0


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
    include_ranker_scores = True
    scores_rows_for_run = 0
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

            try:
                rows, columns = _fetch_latest_candidate_rows(
                    cursor,
                    run_date_value=run_date_value,
                    latest_run_ts=latest_run_ts,
                    limit=limit,
                    include_ranker_scores=True,
                )
                scores_rows_for_run = _count_scores_rows_for_run(cursor, latest_run_ts)
            except Exception as exc:
                LOGGER.warning("[WARN] RANKER_SCORE_JOIN_SKIPPED err=%s", exc)
                try:
                    conn.rollback()
                except Exception:
                    pass
                include_ranker_scores = False
                rows, columns = _fetch_latest_candidate_rows(
                    cursor,
                    run_date_value=run_date_value,
                    latest_run_ts=latest_run_ts,
                    limit=limit,
                    include_ranker_scores=False,
                )
                scores_rows_for_run = 0
    finally:
        try:
            conn.close()
        except Exception:
            pass

    frame = pd.DataFrame(rows, columns=columns)
    if "model_score" not in frame.columns and "model_score_5d" in frame.columns:
        frame["model_score"] = pd.to_numeric(frame["model_score_5d"], errors="coerce")
    score_col = "model_score_5d" if "model_score_5d" in frame.columns else "model_score"
    _log_model_score_join_diag(
        frame,
        scores_rows_for_run=scores_rows_for_run,
        latest_run_ts=latest_run_ts,
        score_col=score_col,
        reason_override="ranker_join_unavailable" if not include_ranker_scores else None,
    )
    LOGGER.info(
        "DB_QUERY latest_screener_candidates run_date=%s latest_run_ts=%s count=%s",
        run_date_value,
        latest_run_ts,
        len(frame.index),
    )
    return frame, latest_run_ts
