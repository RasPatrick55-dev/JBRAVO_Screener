"""Evaluate closed-trade outcomes by model score-at-entry buckets.

This script is evaluation-only and paper-mode safe. It does not modify trading
execution behavior.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from psycopg2 import extras

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts import db  # noqa: E402
from utils.env import load_env  # noqa: E402

load_env()

LOG = logging.getLogger("ranker_trade_attribution")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_MIN_TRADES = 20
DEFAULT_SCORE_COL = "model_score"
DEFAULT_BINS = 10
DEFAULT_TRADES_PATH = BASE_DIR / "data" / "trades.csv"
DEFAULT_OOS_PATH = BASE_DIR / "data" / "ranker_walkforward" / "oos_predictions.csv"
VALID_MATCH_MODES = {"auto", "entry_context", "run_map", "scores_direct", "oos_predictions"}


@dataclass
class AttributionArgs:
    lookback_days: int
    min_trades: int
    score_col: str
    bins: int
    run_date: date | None
    source: str
    match_mode: str
    self_test: bool
    trades_path: Path
    output_dir: Path


def _parse_run_date(value: str | None) -> date | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Invalid --run-date: {value}")
    return parsed.date()


def _window_bounds(run_date: date | None, lookback_days: int) -> tuple[datetime, datetime]:
    if run_date is None:
        end_dt = datetime.now(timezone.utc)
    else:
        end_dt = datetime.combine(run_date, time.max, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=max(int(lookback_days), 1))
    return start_dt, end_dt


def _ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "trade_id",
        "symbol",
        "entry_order_id",
        "qty",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "realized_pnl",
        "exit_reason",
        "status",
        "run_map_ts_utc",
        "screener_run_ts_utc",
        "score_run_ts_utc",
        "model_score_5d",
        "model_score",
        "score_source",
        "score_timestamp_utc",
        "match_status",
        "match_reason",
        "entry_to_score_lag_minutes",
        "symbol_has_entry_context",
        "symbol_has_score",
        "symbol_has_run_map",
        "symbol_has_oos",
    ]
    out = frame.copy()
    datetime_cols = {
        "entry_time",
        "exit_time",
        "run_map_ts_utc",
        "screener_run_ts_utc",
        "score_run_ts_utc",
        "score_timestamp_utc",
    }
    text_cols = {
        "symbol",
        "entry_order_id",
        "exit_reason",
        "status",
        "score_source",
        "match_status",
        "match_reason",
    }
    bool_cols = {
        "symbol_has_entry_context",
        "symbol_has_score",
        "symbol_has_run_map",
        "symbol_has_oos",
    }
    for column in required:
        if column not in out.columns:
            if column in datetime_cols:
                out[column] = pd.Series([pd.NaT] * len(out), index=out.index, dtype="object")
            elif column in text_cols or column in bool_cols:
                out[column] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
            else:
                out[column] = np.nan
    return out


def _table_exists(conn: Any, table_name: str) -> bool:
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (table_name,))
            row = cursor.fetchone()
        return bool(row and row[0])
    except Exception:
        return False


def _diag_counts(
    conn: Any,
    *,
    start_dt: datetime,
    end_dt: datetime,
    entry_context_table_exists: bool,
    score_table_exists: bool,
    run_map_table_exists: bool,
) -> dict[str, int]:
    out = {
        "entry_context_rows_available": 0,
        "score_rows_available": 0,
        "score_runs_available": 0,
        "run_map_rows_available": 0,
        "oos_rows_available": 0,
    }
    try:
        with conn.cursor() as cursor:
            if entry_context_table_exists:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM trade_entry_ml_context_app
                    WHERE COALESCE(entry_time, created_at) >= %(start_dt)s
                      AND COALESCE(entry_time, created_at) <= %(end_dt)s
                    """,
                    {"start_dt": start_dt, "end_dt": end_dt},
                )
                row = cursor.fetchone()
                if row:
                    out["entry_context_rows_available"] = int(row[0] or 0)
            if score_table_exists:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS rows_count,
                        COUNT(DISTINCT date_trunc('second', run_ts_utc)) AS run_count
                    FROM screener_ranker_scores_app
                    WHERE run_ts_utc >= %(start_dt)s
                      AND run_ts_utc <= %(end_dt)s
                    """,
                    {"start_dt": start_dt, "end_dt": end_dt},
                )
                row = cursor.fetchone()
                if row:
                    out["score_rows_available"] = int(row[0] or 0)
                    out["score_runs_available"] = int(row[1] or 0)
            if run_map_table_exists:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM screener_run_map_app
                    WHERE run_ts_utc >= %(start_dt)s
                      AND run_ts_utc <= %(end_dt)s
                    """,
                    {"start_dt": start_dt, "end_dt": end_dt},
                )
                row = cursor.fetchone()
                if row:
                    out["run_map_rows_available"] = int(row[0] or 0)
    except Exception:
        pass
    return out


def _query_closed_trades_base(conn: Any, *, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    query = """
        SELECT
            t.trade_id,
            t.symbol,
            t.entry_order_id,
            t.qty,
            t.entry_time,
            t.exit_time,
            t.entry_price,
            t.exit_price,
            t.realized_pnl,
            t.exit_reason,
            t.status,
            NULL::timestamptz AS run_map_ts_utc,
            NULL::timestamptz AS screener_run_ts_utc,
            NULL::timestamptz AS score_run_ts_utc,
            NULL::double precision AS model_score_5d,
            NULL::double precision AS model_score,
            NULL::text AS score_source,
            NULL::boolean AS symbol_has_entry_context,
            NULL::boolean AS symbol_has_score,
            NULL::boolean AS symbol_has_run_map,
            NULL::boolean AS symbol_has_oos
        FROM trades t
        WHERE UPPER(COALESCE(t.status, '')) = 'CLOSED'
          AND COALESCE(t.exit_time, t.entry_time) >= %(start_dt)s
          AND COALESCE(t.exit_time, t.entry_time) <= %(end_dt)s
        ORDER BY COALESCE(t.exit_time, t.entry_time) DESC NULLS LAST, t.trade_id DESC
    """
    with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
        cursor.execute(query, {"start_dt": start_dt, "end_dt": end_dt})
        rows = cursor.fetchall()
    return _ensure_columns(pd.DataFrame([dict(row) for row in rows]))


def _query_entry_context(conn: Any, *, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    query = """
        WITH closed_trades AS (
            SELECT
                t.trade_id,
                UPPER(TRIM(COALESCE(t.symbol, ''))) AS symbol_norm,
                t.symbol,
                t.entry_order_id,
                t.qty,
                t.entry_time,
                t.exit_time,
                t.entry_price,
                t.exit_price,
                t.realized_pnl,
                t.exit_reason,
                t.status
            FROM trades t
            WHERE UPPER(COALESCE(t.status, '')) = 'CLOSED'
              AND COALESCE(t.exit_time, t.entry_time) >= %(start_dt)s
              AND COALESCE(t.exit_time, t.entry_time) <= %(end_dt)s
        )
        SELECT
            ct.trade_id,
            ct.symbol,
            ct.entry_order_id,
            ct.qty,
            ct.entry_time,
            ct.exit_time,
            ct.entry_price,
            ct.exit_price,
            ct.realized_pnl,
            ct.exit_reason,
            ct.status,
            NULL::timestamptz AS run_map_ts_utc,
            ec.screener_run_ts_utc AS screener_run_ts_utc,
            ec.screener_run_ts_utc AS score_run_ts_utc,
            ec.model_score_5d AS model_score_5d,
            COALESCE(ec.model_score, ec.model_score_5d) AS model_score,
            CASE
                WHEN ec.order_id IS NOT NULL THEN 'entry_context'
                ELSE NULL
            END AS score_source,
            CASE WHEN ec.order_id IS NOT NULL THEN TRUE ELSE FALSE END AS symbol_has_entry_context,
            NULL::boolean AS symbol_has_score,
            NULL::boolean AS symbol_has_run_map,
            NULL::boolean AS symbol_has_oos
        FROM closed_trades ct
        LEFT JOIN LATERAL (
            SELECT e.*
            FROM trade_entry_ml_context_app e
            WHERE e.order_id = ct.entry_order_id
            LIMIT 1
        ) ec ON TRUE
        ORDER BY COALESCE(ct.exit_time, ct.entry_time) DESC NULLS LAST, ct.trade_id DESC
    """
    with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
        cursor.execute(query, {"start_dt": start_dt, "end_dt": end_dt})
        rows = cursor.fetchall()
    return _ensure_columns(pd.DataFrame([dict(row) for row in rows]))


def _query_scores_direct(conn: Any, *, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    query = """
        WITH closed_trades AS (
            SELECT
                t.trade_id,
                UPPER(TRIM(COALESCE(t.symbol, ''))) AS symbol_norm,
                t.symbol,
                t.entry_order_id,
                t.qty,
                t.entry_time,
                t.exit_time,
                t.entry_price,
                t.exit_price,
                t.realized_pnl,
                t.exit_reason,
                t.status
            FROM trades t
            WHERE UPPER(COALESCE(t.status, '')) = 'CLOSED'
              AND COALESCE(t.exit_time, t.entry_time) >= %(start_dt)s
              AND COALESCE(t.exit_time, t.entry_time) <= %(end_dt)s
        )
        SELECT
            ct.trade_id,
            ct.symbol,
            ct.entry_order_id,
            ct.qty,
            ct.entry_time,
            ct.exit_time,
            ct.entry_price,
            ct.exit_price,
            ct.realized_pnl,
            ct.exit_reason,
            ct.status,
            NULL::timestamptz AS run_map_ts_utc,
            sd.run_ts_utc AS screener_run_ts_utc,
            sd.run_ts_utc AS score_run_ts_utc,
            sd.model_score_5d AS model_score_5d,
            sd.model_score_5d AS model_score,
            CASE WHEN sd.run_ts_utc IS NOT NULL THEN 'scores_direct' ELSE NULL END AS score_source,
            NULL::boolean AS symbol_has_entry_context,
            CASE WHEN sym_score.has_score = 1 THEN TRUE ELSE FALSE END AS symbol_has_score,
            NULL::boolean AS symbol_has_run_map,
            NULL::boolean AS symbol_has_oos
        FROM closed_trades ct
        LEFT JOIN LATERAL (
            SELECT s.run_ts_utc, s.model_score_5d
            FROM screener_ranker_scores_app s
            WHERE UPPER(TRIM(COALESCE(s.symbol, ''))) = ct.symbol_norm
              AND date_trunc('second', s.run_ts_utc)
                    <= date_trunc('second', COALESCE(ct.entry_time, ct.exit_time))
            ORDER BY s.run_ts_utc DESC
            LIMIT 1
        ) sd ON TRUE
        LEFT JOIN LATERAL (
            SELECT 1 AS has_score
            FROM screener_ranker_scores_app s2
            WHERE UPPER(TRIM(COALESCE(s2.symbol, ''))) = ct.symbol_norm
            LIMIT 1
        ) sym_score ON TRUE
        ORDER BY COALESCE(ct.exit_time, ct.entry_time) DESC NULLS LAST, ct.trade_id DESC
    """
    with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
        cursor.execute(query, {"start_dt": start_dt, "end_dt": end_dt})
        rows = cursor.fetchall()
    return _ensure_columns(pd.DataFrame([dict(row) for row in rows]))


def _query_run_map(conn: Any, *, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    query = """
        WITH closed_trades AS (
            SELECT
                t.trade_id,
                UPPER(TRIM(COALESCE(t.symbol, ''))) AS symbol_norm,
                t.symbol,
                t.entry_order_id,
                t.qty,
                t.entry_time,
                t.exit_time,
                t.entry_price,
                t.exit_price,
                t.realized_pnl,
                t.exit_reason,
                t.status
            FROM trades t
            WHERE UPPER(COALESCE(t.status, '')) = 'CLOSED'
              AND COALESCE(t.exit_time, t.entry_time) >= %(start_dt)s
              AND COALESCE(t.exit_time, t.entry_time) <= %(end_dt)s
        )
        SELECT
            ct.trade_id,
            ct.symbol,
            ct.entry_order_id,
            ct.qty,
            ct.entry_time,
            ct.exit_time,
            ct.entry_price,
            ct.exit_price,
            ct.realized_pnl,
            ct.exit_reason,
            ct.status,
            rm.run_ts_utc AS run_map_ts_utc,
            rm.run_ts_utc AS screener_run_ts_utc,
            rs.run_ts_utc AS score_run_ts_utc,
            rs.model_score_5d AS model_score_5d,
            rs.model_score_5d AS model_score,
            CASE WHEN rs.run_ts_utc IS NOT NULL THEN 'run_map' ELSE NULL END AS score_source,
            NULL::boolean AS symbol_has_entry_context,
            CASE WHEN sym_score.has_score = 1 THEN TRUE ELSE FALSE END AS symbol_has_score,
            CASE WHEN sym_run_map.has_run_map = 1 THEN TRUE ELSE FALSE END AS symbol_has_run_map,
            NULL::boolean AS symbol_has_oos
        FROM closed_trades ct
        LEFT JOIN LATERAL (
            SELECT m.run_ts_utc
            FROM screener_run_map_app m
            WHERE UPPER(TRIM(COALESCE(m.symbol, ''))) = ct.symbol_norm
              AND date_trunc('second', m.run_ts_utc)
                    <= date_trunc('second', COALESCE(ct.entry_time, ct.exit_time))
            ORDER BY m.run_ts_utc DESC
            LIMIT 1
        ) rm ON TRUE
        LEFT JOIN LATERAL (
            SELECT s.run_ts_utc, s.model_score_5d
            FROM screener_ranker_scores_app s
            WHERE UPPER(TRIM(COALESCE(s.symbol, ''))) = ct.symbol_norm
              AND date_trunc('second', s.run_ts_utc) = date_trunc('second', rm.run_ts_utc)
            ORDER BY s.run_ts_utc DESC
            LIMIT 1
        ) rs ON TRUE
        LEFT JOIN LATERAL (
            SELECT 1 AS has_score
            FROM screener_ranker_scores_app s2
            WHERE UPPER(TRIM(COALESCE(s2.symbol, ''))) = ct.symbol_norm
            LIMIT 1
        ) sym_score ON TRUE
        LEFT JOIN LATERAL (
            SELECT 1 AS has_run_map
            FROM screener_run_map_app m2
            WHERE UPPER(TRIM(COALESCE(m2.symbol, ''))) = ct.symbol_norm
            LIMIT 1
        ) sym_run_map ON TRUE
        ORDER BY COALESCE(ct.exit_time, ct.entry_time) DESC NULLS LAST, ct.trade_id DESC
    """
    with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
        cursor.execute(query, {"start_dt": start_dt, "end_dt": end_dt})
        rows = cursor.fetchall()
    return _ensure_columns(pd.DataFrame([dict(row) for row in rows]))


def _select_oos_score_column(frame: pd.DataFrame) -> str | None:
    preferred = ["score_oos", "score_5d", "model_score", "model_score_5d"]
    for col in preferred:
        if col in frame.columns:
            return col
    for col in frame.columns:
        if str(col).startswith("score_"):
            return str(col)
    return None


def _load_oos_predictions_frame(run_date: date | None) -> tuple[pd.DataFrame, str]:
    if db.db_enabled():
        record = db.fetch_ml_artifact("ranker_oos_predictions", run_date=run_date)
        if record is None and run_date is not None:
            LOG.warning(
                "[WARN] TRADE_ATTRIBUTION_RUN_DATE_FALLBACK artifact=ranker_oos_predictions run_date=%s",
                run_date,
            )
            record = db.fetch_latest_ml_artifact("ranker_oos_predictions")
        if record is not None:
            csv_data = record.get("csv_data")
            if csv_data:
                try:
                    frame = pd.read_csv(io.StringIO(str(csv_data)))
                    return frame, "db://ml_artifacts/ranker_oos_predictions"
                except Exception as exc:
                    LOG.warning(
                        "[WARN] TRADE_ATTRIBUTION_OOS_LOAD_FAILED source=db err=%s",
                        exc,
                    )
            else:
                LOG.warning(
                    "[WARN] TRADE_ATTRIBUTION_DB_ARTIFACT_EMPTY artifact=ranker_oos_predictions "
                    "falling_back=filesystem"
                )

    if DEFAULT_OOS_PATH.exists():
        try:
            frame = pd.read_csv(DEFAULT_OOS_PATH)
            return frame, str(DEFAULT_OOS_PATH)
        except Exception as exc:
            LOG.warning("[WARN] TRADE_ATTRIBUTION_OOS_LOAD_FAILED source=filesystem err=%s", exc)
    return pd.DataFrame(), "missing://ranker_oos_predictions"


def _prepare_oos_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if frame is None or frame.empty:
        return pd.DataFrame(), None
    if "symbol" not in frame.columns or "timestamp" not in frame.columns:
        return pd.DataFrame(), None

    score_col = _select_oos_score_column(frame)
    if not score_col:
        return pd.DataFrame(), None

    work = frame.copy()
    work["symbol_norm"] = work["symbol"].astype("string").str.upper().str.strip()
    work["score_run_ts_utc"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["score_value"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["symbol_norm", "score_run_ts_utc", "score_value"]).copy()
    if work.empty:
        return pd.DataFrame(), score_col
    work = work.sort_values(["symbol_norm", "score_run_ts_utc"]).reset_index(drop=True)
    return work, score_col


def _build_oos_match_frame(trades_frame: pd.DataFrame, oos_prepared: pd.DataFrame) -> pd.DataFrame:
    base = _ensure_columns(trades_frame).copy()
    if base.empty:
        return base

    trade_keys = pd.DataFrame(
        {
            "trade_id": base["trade_id"],
            "symbol_norm": base["symbol"].astype("string").str.upper().str.strip(),
            "entry_time": pd.to_datetime(base["entry_time"], utc=True, errors="coerce"),
        }
    )
    symbol_has_oos = pd.Series(False, index=trade_keys.index, dtype="bool")
    if not oos_prepared.empty:
        oos_symbols = set(oos_prepared["symbol_norm"].dropna().astype(str).tolist())
        symbol_has_oos = trade_keys["symbol_norm"].astype(str).isin(oos_symbols)

    out = _ensure_columns(pd.DataFrame({"trade_id": trade_keys["trade_id"]}))
    out["symbol_has_oos"] = symbol_has_oos.to_numpy()

    valid_trades = trade_keys.dropna(subset=["trade_id", "symbol_norm", "entry_time"]).copy()
    if valid_trades.empty or oos_prepared.empty:
        return _ensure_columns(out)

    valid_trades["entry_time"] = pd.to_datetime(
        valid_trades["entry_time"], utc=True, errors="coerce"
    )
    valid_trades = valid_trades.dropna(subset=["entry_time"]).copy()
    if valid_trades.empty:
        return _ensure_columns(out)

    valid_trades["entry_date"] = valid_trades["entry_time"].dt.floor("D")
    oos = oos_prepared.copy()
    oos["score_run_ts_utc"] = pd.to_datetime(oos["score_run_ts_utc"], utc=True, errors="coerce")
    oos = oos.dropna(subset=["score_run_ts_utc"]).copy()
    if oos.empty:
        return _ensure_columns(out)
    oos["oos_date"] = oos["score_run_ts_utc"].dt.floor("D")

    # Pass 1: same-symbol and same UTC date, latest score on that date.
    oos_daily_latest = oos.drop_duplicates(subset=["symbol_norm", "oos_date"], keep="last")
    exact = valid_trades.merge(
        oos_daily_latest[["symbol_norm", "oos_date", "score_run_ts_utc", "score_value"]],
        left_on=["symbol_norm", "entry_date"],
        right_on=["symbol_norm", "oos_date"],
        how="left",
    )
    exact["match_kind"] = np.where(exact["score_run_ts_utc"].notna(), "exact_date", "")

    exact_trade_ids = set(exact.loc[exact["score_run_ts_utc"].notna(), "trade_id"].tolist())
    remaining = valid_trades.loc[~valid_trades["trade_id"].isin(exact_trade_ids)].copy()

    asof = pd.DataFrame(columns=["trade_id", "score_run_ts_utc", "score_value", "match_kind"])
    if not remaining.empty:
        left = remaining.sort_values(["symbol_norm", "entry_time"])
        right = oos.sort_values(["symbol_norm", "score_run_ts_utc"])
        asof_parts: list[pd.DataFrame] = []
        for symbol_norm, left_group in left.groupby("symbol_norm", dropna=False):
            right_group = right.loc[right["symbol_norm"] == symbol_norm]
            if right_group.empty:
                unmatched_group = left_group.copy()
                unmatched_group["score_run_ts_utc"] = pd.NaT
                unmatched_group["score_value"] = np.nan
                asof_parts.append(unmatched_group)
                continue
            merged_group = pd.merge_asof(
                left_group.sort_values("entry_time"),
                right_group[["score_run_ts_utc", "score_value"]].sort_values("score_run_ts_utc"),
                left_on="entry_time",
                right_on="score_run_ts_utc",
                direction="backward",
                allow_exact_matches=True,
            )
            merged_group["symbol_norm"] = symbol_norm
            asof_parts.append(merged_group)
        asof = pd.concat(asof_parts, ignore_index=True) if asof_parts else asof
        if not asof.empty:
            asof["entry_time"] = pd.to_datetime(asof["entry_time"], utc=True, errors="coerce")
            asof["score_run_ts_utc"] = pd.to_datetime(
                asof["score_run_ts_utc"], utc=True, errors="coerce"
            )
            asof["score_value"] = pd.to_numeric(asof["score_value"], errors="coerce")
        asof["match_kind"] = np.where(asof["score_run_ts_utc"].notna(), "asof", "")

    matched = pd.concat(
        [
            exact[["trade_id", "entry_time", "score_run_ts_utc", "score_value", "match_kind"]],
            asof[["trade_id", "entry_time", "score_run_ts_utc", "score_value", "match_kind"]],
        ],
        ignore_index=True,
    )
    matched = matched.loc[matched["score_run_ts_utc"].notna()].copy()
    if matched.empty:
        return _ensure_columns(out)
    matched["entry_time"] = pd.to_datetime(matched["entry_time"], utc=True, errors="coerce")
    matched["score_run_ts_utc"] = pd.to_datetime(
        matched["score_run_ts_utc"], utc=True, errors="coerce"
    )
    matched = matched.dropna(subset=["entry_time", "score_run_ts_utc"]).copy()
    if matched.empty:
        return _ensure_columns(out)

    priority_map = {"exact_date": 0, "asof": 1}
    matched["match_priority"] = matched["match_kind"].map(priority_map).fillna(2)
    matched.sort_values(["trade_id", "match_priority", "score_run_ts_utc"], inplace=True)
    matched = matched.drop_duplicates(subset=["trade_id"], keep="first")
    matched["entry_to_score_lag_minutes"] = (
        matched["entry_time"] - matched["score_run_ts_utc"]
    ).dt.total_seconds() / 60.0

    matched_frame = _ensure_columns(
        pd.DataFrame(
            {
                "trade_id": matched["trade_id"],
                "run_map_ts_utc": pd.NaT,
                "screener_run_ts_utc": matched["score_run_ts_utc"],
                "score_run_ts_utc": matched["score_run_ts_utc"],
                "score_timestamp_utc": matched["score_run_ts_utc"],
                "model_score_5d": matched["score_value"],
                "model_score": matched["score_value"],
                "score_source": "oos_predictions",
                "entry_to_score_lag_minutes": matched["entry_to_score_lag_minutes"],
                "symbol_has_oos": True,
            }
        )
    )
    return _merge_match_frames(out, matched_frame)


def _query_oos_predictions(
    trades_frame: pd.DataFrame,
    *,
    run_date: date | None,
) -> tuple[pd.DataFrame, int, str, str | None]:
    raw_oos, source = _load_oos_predictions_frame(run_date)
    prepared, score_col = _prepare_oos_frame(raw_oos)
    oos_rows_available = int(prepared.shape[0]) if prepared is not None else 0
    matched_frame = _build_oos_match_frame(trades_frame, prepared)
    return matched_frame, oos_rows_available, source, score_col


def _merge_match_frames(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if primary is None or primary.empty:
        return _ensure_columns(fallback if fallback is not None else pd.DataFrame())
    if fallback is None or fallback.empty:
        return _ensure_columns(primary)

    left = _ensure_columns(primary).copy()
    right = _ensure_columns(fallback).copy()
    right = right.drop_duplicates(subset=["trade_id"], keep="first")
    merge_cols = [
        "trade_id",
        "run_map_ts_utc",
        "screener_run_ts_utc",
        "score_run_ts_utc",
        "score_timestamp_utc",
        "model_score_5d",
        "model_score",
        "score_source",
        "entry_to_score_lag_minutes",
        "symbol_has_entry_context",
        "symbol_has_score",
        "symbol_has_run_map",
        "symbol_has_oos",
    ]
    merged = left.merge(right[merge_cols], on="trade_id", how="left", suffixes=("", "__fb"))

    score_missing = merged["model_score"].isna() & merged["model_score_5d"].isna()
    fill_cols = [
        "run_map_ts_utc",
        "screener_run_ts_utc",
        "score_run_ts_utc",
        "score_timestamp_utc",
        "model_score_5d",
        "model_score",
        "score_source",
        "entry_to_score_lag_minutes",
    ]
    for col in fill_cols:
        fb_col = f"{col}__fb"
        if fb_col in merged.columns:
            merged.loc[score_missing, col] = merged.loc[score_missing, fb_col]

    def _as_bool(series: pd.Series) -> pd.Series:
        return series.apply(_truthy)

    for flag in (
        "symbol_has_entry_context",
        "symbol_has_score",
        "symbol_has_run_map",
        "symbol_has_oos",
    ):
        fb_col = f"{flag}__fb"
        if fb_col in merged.columns:
            merged[flag] = _as_bool(merged[flag]) | _as_bool(merged[fb_col])

    drop_cols = [col for col in merged.columns if col.endswith("__fb")]
    if drop_cols:
        merged.drop(columns=drop_cols, inplace=True, errors="ignore")
    return _ensure_columns(merged)


def _load_closed_trades_db(
    *,
    start_dt: datetime,
    end_dt: datetime,
    match_mode: str,
    run_date: date | None,
) -> tuple[pd.DataFrame, str, str, dict[str, int]]:
    conn = db.get_db_conn()
    if conn is None:
        raise RuntimeError("DB disabled or unreachable")
    db.ensure_screener_ranker_scores_app_table()
    db.ensure_trade_entry_ml_context_app_table()
    try:
        entry_context_table_exists = _table_exists(conn, "trade_entry_ml_context_app")
        score_table_exists = _table_exists(conn, "screener_ranker_scores_app")
        run_map_table_exists = _table_exists(conn, "screener_run_map_app")
        diag = _diag_counts(
            conn,
            start_dt=start_dt,
            end_dt=end_dt,
            entry_context_table_exists=entry_context_table_exists,
            score_table_exists=score_table_exists,
            run_map_table_exists=run_map_table_exists,
        )
        requested_mode = str(match_mode).strip().lower()
        if requested_mode not in VALID_MATCH_MODES:
            requested_mode = "auto"

        base_frame = _query_closed_trades_base(conn, start_dt=start_dt, end_dt=end_dt)
        mode_used = requested_mode
        if requested_mode == "entry_context" and not entry_context_table_exists:
            raise RuntimeError("entry_context requested but trade_entry_ml_context_app unavailable")
        elif requested_mode == "scores_direct" and not score_table_exists:
            raise RuntimeError("scores_direct requested but screener_ranker_scores_app unavailable")
        elif requested_mode == "run_map" and (not run_map_table_exists or not score_table_exists):
            raise RuntimeError("run_map requested but required tables unavailable")

        if requested_mode == "auto":
            mode_used = "auto"
            frame = base_frame
            if entry_context_table_exists:
                try:
                    frame = _merge_match_frames(
                        frame,
                        _query_entry_context(conn, start_dt=start_dt, end_dt=end_dt),
                    )
                except Exception as exc:
                    LOG.warning(
                        "[WARN] TRADE_ATTRIBUTION_MATCH_FALLBACK from=entry_context reason=%s",
                        exc,
                    )
            if score_table_exists:
                try:
                    frame = _merge_match_frames(
                        frame,
                        _query_scores_direct(conn, start_dt=start_dt, end_dt=end_dt),
                    )
                except Exception as exc:
                    LOG.warning(
                        "[WARN] TRADE_ATTRIBUTION_MATCH_FALLBACK from=scores_direct reason=%s",
                        exc,
                    )
            if run_map_table_exists and score_table_exists:
                try:
                    frame = _merge_match_frames(
                        frame,
                        _query_run_map(conn, start_dt=start_dt, end_dt=end_dt),
                    )
                except Exception as exc:
                    LOG.warning(
                        "[WARN] TRADE_ATTRIBUTION_MATCH_FALLBACK from=run_map reason=%s",
                        exc,
                    )
            try:
                oos_frame, oos_rows_available, _, _ = _query_oos_predictions(
                    frame,
                    run_date=run_date,
                )
                diag["oos_rows_available"] = int(oos_rows_available)
                frame = _merge_match_frames(frame, oos_frame)
            except Exception as exc:
                LOG.warning(
                    "[WARN] TRADE_ATTRIBUTION_MATCH_FALLBACK from=oos_predictions reason=%s",
                    exc,
                )
        elif mode_used == "entry_context":
            frame = _query_entry_context(conn, start_dt=start_dt, end_dt=end_dt)
        elif mode_used == "scores_direct":
            frame = _query_scores_direct(conn, start_dt=start_dt, end_dt=end_dt)
        elif mode_used == "run_map":
            frame = _query_run_map(conn, start_dt=start_dt, end_dt=end_dt)
        else:
            frame = base_frame
            oos_frame, oos_rows_available, _, _ = _query_oos_predictions(
                frame,
                run_date=run_date,
            )
            diag["oos_rows_available"] = int(oos_rows_available)
            frame = _merge_match_frames(frame, oos_frame)

        source = f"db://trades+{mode_used}"
        return _ensure_columns(frame), source, mode_used, diag
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _load_closed_trades_fs(
    *,
    trades_path: Path,
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[pd.DataFrame, str, str, dict[str, int]]:
    if not trades_path.exists():
        return (
            _ensure_columns(pd.DataFrame()),
            str(trades_path),
            "fs",
            {
                "entry_context_rows_available": 0,
                "score_rows_available": 0,
                "score_runs_available": 0,
                "run_map_rows_available": 0,
                "oos_rows_available": 0,
            },
        )
    frame = pd.read_csv(trades_path)
    if frame.empty:
        return (
            _ensure_columns(frame),
            str(trades_path),
            "fs",
            {
                "entry_context_rows_available": 0,
                "score_rows_available": 0,
                "score_runs_available": 0,
                "run_map_rows_available": 0,
                "oos_rows_available": 0,
            },
        )

    work = _ensure_columns(frame)
    status_series = work["status"].astype("string").str.upper()
    if status_series.notna().any():
        work = work.loc[status_series == "CLOSED"].copy()
    if work.empty:
        return (
            _ensure_columns(work),
            str(trades_path),
            "fs",
            {
                "entry_context_rows_available": 0,
                "score_rows_available": 0,
                "score_runs_available": 0,
                "run_map_rows_available": 0,
                "oos_rows_available": 0,
            },
        )

    for column in ("entry_time", "exit_time"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="coerce")
    ref_ts = pd.to_datetime(
        work["exit_time"].where(work["exit_time"].notna(), work["entry_time"]),
        utc=True,
        errors="coerce",
    )
    mask = (ref_ts >= pd.Timestamp(start_dt)) & (ref_ts <= pd.Timestamp(end_dt))
    work = work.loc[mask].copy()
    work.sort_values(by=["exit_time", "entry_time", "symbol"], inplace=True)
    return (
        _ensure_columns(work.reset_index(drop=True)),
        str(trades_path),
        "fs",
        {
            "entry_context_rows_available": 0,
            "score_rows_available": 0,
            "score_runs_available": 0,
            "run_map_rows_available": 0,
            "oos_rows_available": 0,
        },
    )


def _load_trades(
    args: AttributionArgs,
    *,
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[pd.DataFrame, str, str, dict[str, int]]:
    source = str(args.source).strip().lower()
    if source not in {"auto", "db", "fs"}:
        source = "auto"

    if source in {"auto", "db"}:
        if db.db_enabled():
            try:
                return _load_closed_trades_db(
                    start_dt=start_dt,
                    end_dt=end_dt,
                    match_mode=args.match_mode,
                    run_date=args.run_date,
                )
            except Exception as exc:
                if source == "db":
                    raise
                LOG.warning(
                    "[WARN] TRADE_ATTRIBUTION_DB_FALLBACK reason=%s fallback=filesystem",
                    exc,
                )
        elif source == "db":
            raise RuntimeError("DB requested but disabled/unreachable")

    frame, source_used, mode_used, diag = _load_closed_trades_fs(
        trades_path=args.trades_path,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    requested_mode = str(args.match_mode).strip().lower()
    if requested_mode not in VALID_MATCH_MODES:
        requested_mode = "auto"
    if requested_mode in {"auto", "oos_predictions"}:
        oos_frame, oos_rows_available, _, _ = _query_oos_predictions(frame, run_date=args.run_date)
        diag["oos_rows_available"] = int(oos_rows_available)
        frame = _merge_match_frames(frame, oos_frame)
        mode_used = requested_mode
    return _ensure_columns(frame), source_used, mode_used, diag


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        if value.tz is None:
            value = value.tz_localize("UTC")
        return value.isoformat()
    return value


def _resolve_score_column(work: pd.DataFrame, requested: str) -> tuple[pd.Series, str]:
    requested_col = str(requested or "").strip()
    if requested_col and requested_col in work.columns:
        return pd.to_numeric(work[requested_col], errors="coerce"), requested_col
    if requested_col == "model_score" and "model_score_5d" in work.columns:
        return pd.to_numeric(work["model_score_5d"], errors="coerce"), "model_score_5d"
    if "model_score" in work.columns:
        return pd.to_numeric(work["model_score"], errors="coerce"), "model_score"
    if "model_score_5d" in work.columns:
        return pd.to_numeric(work["model_score_5d"], errors="coerce"), "model_score_5d"
    return pd.Series(np.nan, index=work.index, dtype="float64"), "missing"


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on", "t"}


def _annotate_match_status(
    work: pd.DataFrame,
    *,
    match_mode_used: str,
) -> tuple[
    pd.DataFrame, dict[str, int], dict[str, float | None], dict[str, dict[str, float | None]]
]:
    out = work.copy()
    match_status: list[str] = []
    match_reason: list[str] = []
    score_source: list[str] = []

    for _, row in out.iterrows():
        score_val = row.get("score_at_entry")
        score_ts = row.get("score_run_ts_utc")
        run_map_ts = row.get("run_map_ts_utc")
        has_entry_context = _truthy(row.get("symbol_has_entry_context"))
        has_score = _truthy(row.get("symbol_has_score"))
        has_run_map = _truthy(row.get("symbol_has_run_map"))
        has_oos = _truthy(row.get("symbol_has_oos"))
        source_cell = str(row.get("score_source") or "").strip().lower()

        if score_val is not None and not (isinstance(score_val, float) and np.isnan(score_val)):
            match_status.append("matched")
            match_reason.append("matched")
            score_source.append(
                source_cell or ("fs" if match_mode_used == "fs" else match_mode_used)
            )
            continue

        reason = "missing_scores"
        if match_mode_used == "entry_context":
            if not has_entry_context:
                reason = "missing_entry_context"
            else:
                reason = "missing_overlay_score"
            score_source.append("entry_context")
        elif match_mode_used == "scores_direct":
            if pd.notna(score_ts):
                reason = "missing_scores"
            elif not has_score:
                reason = "symbol_mismatch"
            else:
                reason = "time_mismatch"
            score_source.append("scores_direct")
        elif match_mode_used == "run_map":
            if pd.notna(score_ts):
                reason = "missing_scores"
            elif pd.notna(run_map_ts):
                if has_score:
                    reason = "time_mismatch"
                else:
                    reason = "missing_scores"
            else:
                if has_score and not has_run_map:
                    reason = "missing_run_map"
                elif has_run_map and not has_score:
                    reason = "missing_scores"
                elif not has_score and not has_run_map:
                    reason = "symbol_mismatch"
                else:
                    reason = "time_mismatch"
            score_source.append("run_map")
        elif match_mode_used == "oos_predictions":
            if has_oos:
                reason = "time_mismatch"
            else:
                reason = "missing_oos_predictions"
            score_source.append("oos_predictions")
        elif match_mode_used == "auto":
            if has_entry_context:
                reason = "missing_overlay_score"
            elif has_score and has_run_map and pd.isna(score_ts):
                reason = "time_mismatch"
            elif has_score:
                reason = "missing_scores"
            elif has_run_map:
                reason = "missing_scores"
            elif has_oos:
                reason = "time_mismatch"
            else:
                reason = "missing_oos_predictions"
            score_source.append(source_cell or "auto")
        else:
            reason = "missing_scores"
            score_source.append(source_cell or "fs")

        match_status.append("unmatched")
        match_reason.append(reason)

    out["score_source"] = score_source
    out["match_status"] = match_status
    out["match_reason"] = match_reason
    unmatched = out.loc[out["match_status"] == "unmatched", "match_reason"]
    unmatched_reason_counts = {
        str(k): int(v) for k, v in unmatched.value_counts(dropna=False).to_dict().items()
    }

    lag_stats = {"min": None, "median": None, "p95": None}
    lag_stats_by_source: dict[str, dict[str, float | None]] = {}
    matched_mask = (
        (out["match_status"] == "matched")
        & out["entry_time"].notna()
        & out["score_run_ts_utc"].notna()
    )
    if bool(matched_mask.any()):
        lag_minutes = (
            pd.to_datetime(out.loc[matched_mask, "entry_time"], utc=True, errors="coerce")
            - pd.to_datetime(out.loc[matched_mask, "score_run_ts_utc"], utc=True, errors="coerce")
        ).dt.total_seconds() / 60.0
        lag_minutes = pd.to_numeric(lag_minutes, errors="coerce")
        lag_minutes_clean = lag_minutes.dropna()
        if not lag_minutes_clean.empty:
            lag_stats = {
                "min": _safe_float(lag_minutes_clean.min()),
                "median": _safe_float(lag_minutes_clean.median()),
                "p95": _safe_float(lag_minutes_clean.quantile(0.95)),
            }
            out.loc[lag_minutes_clean.index, "entry_to_score_lag_minutes"] = lag_minutes_clean

            matched_rows = out.loc[lag_minutes_clean.index].copy()
            matched_rows["lag_minutes"] = lag_minutes_clean
            matched_rows["score_source_norm"] = (
                matched_rows["score_source"].astype("string").str.lower().fillna("unknown")
            )
            for source_name, group in matched_rows.groupby("score_source_norm", dropna=False):
                series = pd.to_numeric(group["lag_minutes"], errors="coerce").dropna()
                if series.empty:
                    continue
                lag_stats_by_source[str(source_name)] = {
                    "min": _safe_float(series.min()),
                    "median": _safe_float(series.median()),
                    "p95": _safe_float(series.quantile(0.95)),
                }

    return out, unmatched_reason_counts, lag_stats, lag_stats_by_source


def _compute_bucket_metrics(scored: pd.DataFrame, *, bins: int) -> tuple[list[dict[str, Any]], int]:
    if scored.empty:
        return [], 0
    scores = pd.to_numeric(scored["score_at_entry"], errors="coerce")
    n_rows = int(scores.notna().sum())
    if n_rows <= 0:
        return [], 0
    q = max(1, min(int(bins), n_rows))
    try:
        bucket_series = pd.qcut(scores, q=q, labels=False, duplicates="drop")
    except Exception:
        ranked = scores.rank(method="first")
        bucket_series = pd.qcut(ranked, q=q, labels=False, duplicates="drop")
    bucket_series = pd.to_numeric(bucket_series, errors="coerce")

    work = scored.copy()
    work["score_bucket"] = bucket_series
    work = work.loc[work["score_bucket"].notna()].copy()
    if work.empty:
        return [], 0

    win_numeric = pd.to_numeric(work["win"], errors="coerce")
    grouped = (
        work.groupby("score_bucket", dropna=True)
        .agg(
            count=("trade_id", "count"),
            score_min=("score_at_entry", "min"),
            score_max=("score_at_entry", "max"),
            avg_return=("trade_return_pct", "mean"),
            median_return=("trade_return_pct", "median"),
        )
        .reset_index()
    )
    grouped["win_rate"] = (
        pd.DataFrame({"score_bucket": work["score_bucket"], "win_numeric": win_numeric})
        .groupby("score_bucket", dropna=True)["win_numeric"]
        .mean()
        .reindex(grouped["score_bucket"])
        .to_numpy()
    )
    grouped.sort_values(by="score_bucket", ascending=False, inplace=True)

    actual_bins = int(grouped["score_bucket"].nunique())
    metrics: list[dict[str, Any]] = []
    for _, row in grouped.iterrows():
        bucket_idx = int(float(row["score_bucket"]))
        metrics.append(
            {
                "bucket": bucket_idx,
                "bucket_label": f"{bucket_idx + 1}/{max(actual_bins, 1)}",
                "count": int(row["count"]),
                "score_min": _safe_float(row["score_min"]),
                "score_max": _safe_float(row["score_max"]),
                "win_rate": _safe_float(row.get("win_rate")),
                "avg_return": _safe_float(row.get("avg_return")),
                "median_return": _safe_float(row.get("median_return")),
            }
        )
    return metrics, actual_bins


def run_trade_attribution(args: AttributionArgs) -> dict[str, Any]:
    LOG.info(
        "[INFO] TRADE_ATTRIBUTION_START lookback_days=%d score_col=%s bins=%d",
        int(args.lookback_days),
        args.score_col,
        int(args.bins),
    )

    start_dt, end_dt = _window_bounds(args.run_date, args.lookback_days)
    trades, source_used, match_mode_used, diag_counts = _load_trades(
        args, start_dt=start_dt, end_dt=end_dt
    )
    work = _ensure_columns(trades).copy()
    if "symbol" in work.columns:
        work["symbol"] = work["symbol"].astype("string").str.upper()
    for column in (
        "entry_time",
        "exit_time",
        "score_run_ts_utc",
        "screener_run_ts_utc",
        "run_map_ts_utc",
    ):
        work[column] = pd.to_datetime(work[column], utc=True, errors="coerce")
    for column in (
        "entry_price",
        "exit_price",
        "qty",
        "realized_pnl",
        "model_score_5d",
        "model_score",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    score_series, score_col_used = _resolve_score_column(work, args.score_col)
    work["score_at_entry"] = pd.to_numeric(score_series, errors="coerce")
    work, unmatched_reason_counts, lag_minutes_stats, lag_minutes_stats_by_source = (
        _annotate_match_status(work, match_mode_used=match_mode_used)
    )
    work["score_timestamp_utc"] = pd.to_datetime(
        work["score_timestamp_utc"].where(
            work["score_timestamp_utc"].notna(), work["score_run_ts_utc"]
        ),
        utc=True,
        errors="coerce",
    )

    trades_total = int(work.shape[0])
    LOG.info(
        "[INFO] TRADE_ATTRIBUTION_DIAG trades_total=%d entry_context_rows_available=%d score_rows_available=%d score_runs_available=%d run_map_rows_available=%d oos_rows_available=%d match_mode=%s",
        trades_total,
        int(diag_counts.get("entry_context_rows_available", 0)),
        int(diag_counts.get("score_rows_available", 0)),
        int(diag_counts.get("score_runs_available", 0)),
        int(diag_counts.get("run_map_rows_available", 0)),
        int(diag_counts.get("oos_rows_available", 0)),
        match_mode_used,
    )

    entry_price = pd.to_numeric(work["entry_price"], errors="coerce")
    exit_price = pd.to_numeric(work["exit_price"], errors="coerce")
    valid_return = entry_price.notna() & (entry_price != 0) & exit_price.notna()
    work["trade_return_pct"] = np.nan
    work.loc[valid_return, "trade_return_pct"] = (
        exit_price.loc[valid_return] - entry_price.loc[valid_return]
    ) / entry_price.loc[valid_return]

    realized = pd.to_numeric(work["realized_pnl"], errors="coerce")
    win = pd.Series(pd.NA, index=work.index, dtype="boolean")
    realized_mask = realized.notna()
    win.loc[realized_mask] = realized.loc[realized_mask] > 0
    fallback_mask = (~realized_mask) & work["trade_return_pct"].notna()
    win.loc[fallback_mask] = work.loc[fallback_mask, "trade_return_pct"] > 0
    work["win"] = win

    scored = work.loc[work["match_status"] == "matched"].copy()
    matched_by_source = {
        str(k): int(v)
        for k, v in scored["score_source"]
        .fillna("unknown")
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }
    for source_name in ("entry_context", "scores_direct", "run_map", "oos_predictions"):
        matched_by_source.setdefault(source_name, 0)
    LOG.info(
        "[INFO] TRADE_ATTRIBUTION_DIAG matched_by_source_entry_context=%d matched_by_source_scores_direct=%d matched_by_source_run_map=%d matched_by_source_oos_predictions=%d",
        int(matched_by_source.get("entry_context", 0)),
        int(matched_by_source.get("scores_direct", 0)),
        int(matched_by_source.get("run_map", 0)),
        int(matched_by_source.get("oos_predictions", 0)),
    )
    trades_scored = int(scored.shape[0])
    trades_unmatched = int(max(trades_total - trades_scored, 0))
    LOG.info(
        "[INFO] TRADE_ATTRIBUTION_MATCHED trades_total=%d matched=%d unmatched=%d",
        trades_total,
        trades_scored,
        trades_unmatched,
    )

    if trades_total < int(args.min_trades):
        LOG.warning(
            "[WARN] TRADE_ATTRIBUTION_INSUFFICIENT trades=%d required=%d",
            trades_total,
            int(args.min_trades),
        )

    win_numeric = pd.to_numeric(scored["win"], errors="coerce")
    win_rate_scored = _safe_float(win_numeric.mean()) if not scored.empty else None
    avg_return_scored = _safe_float(
        pd.to_numeric(scored["trade_return_pct"], errors="coerce").mean()
    )
    median_return_scored = _safe_float(
        pd.to_numeric(scored["trade_return_pct"], errors="coerce").median()
    )

    bucket_metrics, actual_bins = _compute_bucket_metrics(scored, bins=int(args.bins))

    brier = None
    brier_rows = 0
    brier_skipped_reason = None
    if not scored.empty:
        score_vals = pd.to_numeric(scored["score_at_entry"], errors="coerce")
        brier_mask = score_vals.between(0.0, 1.0, inclusive="both") & win_numeric.notna()
        brier_rows = int(brier_mask.sum())
        if brier_rows >= 5:
            score_used = score_vals.loc[brier_mask]
            win_used = win_numeric.loc[brier_mask]
            brier = _safe_float(float(np.mean((score_used - win_used) ** 2)))
        else:
            brier_skipped_reason = "insufficient_prob_rows_or_out_of_range_scores"
    else:
        brier_skipped_reason = "no_scored_trades"
    if brier is None and brier_skipped_reason:
        LOG.info(
            "[INFO] TRADE_ATTRIBUTION_BRIER_SKIPPED reason=%s rows=%d",
            brier_skipped_reason,
            int(brier_rows),
        )

    status = "ok"
    if trades_total <= 0:
        status = "no_data"
    elif trades_total < int(args.min_trades):
        status = "insufficient_trades"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.json"
    trades_path = output_dir / "trades_scored.csv"

    output_frame = work.copy()
    output_frame.sort_values(
        by=["exit_time", "entry_time", "symbol"],
        ascending=[False, False, True],
        inplace=True,
    )
    output_frame.reset_index(drop=True, inplace=True)
    output_frame.to_csv(trades_path, index=False)

    payload: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "run_date": str(args.run_date) if args.run_date else None,
        "status": status,
        "validation_scheme": "trade_outcome_attribution",
        "source": source_used,
        "score_col_requested": args.score_col,
        "score_col_used": score_col_used,
        "lookback_days": int(args.lookback_days),
        "min_trades": int(args.min_trades),
        "bins_requested": int(args.bins),
        "bins_used": int(actual_bins),
        "window_start": start_dt.isoformat(),
        "window_end": end_dt.isoformat(),
        "match_mode_used": match_mode_used,
        "entry_context_rows_available": int(diag_counts.get("entry_context_rows_available", 0)),
        "score_rows_available": int(diag_counts.get("score_rows_available", 0)),
        "score_runs_available": int(diag_counts.get("score_runs_available", 0)),
        "run_map_rows_available": int(diag_counts.get("run_map_rows_available", 0)),
        "oos_rows_available": int(diag_counts.get("oos_rows_available", 0)),
        "matched_by_source": matched_by_source,
        "unmatched_reason_counts": unmatched_reason_counts,
        "entry_to_score_lag_minutes": lag_minutes_stats,
        "entry_to_score_lag_minutes_by_source": lag_minutes_stats_by_source,
        "summary": {
            "trades_total": trades_total,
            "trades_scored": trades_scored,
            "trades_unmatched": trades_unmatched,
            "win_rate_scored": win_rate_scored,
            "avg_return_scored": avg_return_scored,
            "median_return_scored": median_return_scored,
            "brier": brier,
            "brier_rows": int(brier_rows),
            "brier_skipped_reason": brier_skipped_reason,
        },
        "buckets": bucket_metrics,
        "output_files": {
            "latest_json": str(latest_path),
            "trades_scored_csv": str(trades_path),
        },
    }
    payload = _to_json_safe(payload)
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    run_date_for_artifact = args.run_date or datetime.now(timezone.utc).date()
    if db.db_enabled():
        if db.upsert_ml_artifact(
            "ranker_trade_attribution",
            run_date_for_artifact,
            payload=payload,
            rows_count=int(trades_total),
            source="ranker_trade_attribution",
            file_name=latest_path.name,
        ):
            LOG.info(
                "[INFO] TRADE_ATTRIBUTION_DB_WRITTEN artifact_type=ranker_trade_attribution run_date=%s",
                run_date_for_artifact,
            )
        if db.upsert_ml_artifact_frame(
            "ranker_trade_attribution_trades",
            run_date_for_artifact,
            output_frame,
            source="ranker_trade_attribution",
            file_name=trades_path.name,
        ):
            LOG.info(
                "[INFO] TRADE_ATTRIBUTION_DB_WRITTEN artifact_type=ranker_trade_attribution_trades run_date=%s",
                run_date_for_artifact,
            )
    else:
        LOG.warning("[WARN] DB_DISABLED ranker_trade_attribution_using_fs_fallback=true")

    LOG.info(
        "[INFO] TRADE_ATTRIBUTION_END win_rate=%s avg_return=%s brier=%s output=%s",
        win_rate_scored,
        avg_return_scored,
        brier,
        latest_path,
    )
    return payload


def parse_args(argv: list[str] | None = None) -> AttributionArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--score-col", type=str, default=DEFAULT_SCORE_COL)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--run-date", type=str, default=None)
    parser.add_argument(
        "--source",
        choices=("auto", "db", "fs"),
        default="auto",
        help="Input source selection. auto prefers DB then falls back to filesystem.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("auto", "entry_context", "run_map", "scores_direct", "oos_predictions"),
        default="auto",
        help="Point-in-time join mode for trade->score attribution.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run an in-memory join/matching self-test and exit.",
    )
    parser.add_argument(
        "--trades-path",
        type=Path,
        default=DEFAULT_TRADES_PATH,
        help="Filesystem trades CSV fallback path (default: data/trades.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "ranker_trade_attribution",
        help="Output directory for attribution artifacts.",
    )
    parsed = parser.parse_args(argv)

    if int(parsed.lookback_days) < 1:
        parser.error("--lookback-days must be >= 1")
    if int(parsed.min_trades) < 1:
        parser.error("--min-trades must be >= 1")
    if int(parsed.bins) < 1:
        parser.error("--bins must be >= 1")
    try:
        run_date = _parse_run_date(parsed.run_date)
    except ValueError as exc:
        parser.error(str(exc))
        raise  # pragma: no cover - argparse exits

    return AttributionArgs(
        lookback_days=int(parsed.lookback_days),
        min_trades=int(parsed.min_trades),
        score_col=str(parsed.score_col),
        bins=int(parsed.bins),
        run_date=run_date,
        source=str(parsed.source),
        match_mode=str(parsed.match_mode),
        self_test=bool(parsed.self_test),
        trades_path=Path(parsed.trades_path),
        output_dir=Path(parsed.output_dir),
    )


def run_self_test() -> int:
    frame = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "symbol": "AAPL",
                "entry_order_id": "OID-1",
                "entry_time": "2026-02-01T14:30:00Z",
                "exit_time": "2026-02-05T14:30:00Z",
                "model_score": 0.72,
                "model_score_5d": 0.72,
                "score_source": "entry_context",
                "symbol_has_entry_context": True,
            },
            {
                "trade_id": 2,
                "symbol": "MSFT",
                "entry_order_id": "OID-2",
                "entry_time": "2026-02-02T14:30:00Z",
                "exit_time": "2026-02-06T14:30:00Z",
                "model_score": np.nan,
                "model_score_5d": np.nan,
                "score_source": "",
                "symbol_has_entry_context": False,
                "symbol_has_score": False,
                "symbol_has_run_map": False,
            },
        ]
    )
    work = _ensure_columns(frame)
    oos_frame = pd.DataFrame(
        [
            {
                "symbol": "MSFT",
                "timestamp": "2026-02-02T13:00:00Z",
                "score_oos": 0.61,
            }
        ]
    )
    oos_prepared, _ = _prepare_oos_frame(oos_frame)
    work = _merge_match_frames(work, _build_oos_match_frame(work, oos_prepared))
    score_series, _ = _resolve_score_column(work, "model_score")
    work["score_at_entry"] = pd.to_numeric(score_series, errors="coerce")
    out, unmatched_reason_counts, _, _ = _annotate_match_status(work, match_mode_used="auto")
    matched = int((out["match_status"] == "matched").sum())
    unmatched = int((out["match_status"] == "unmatched").sum())
    matched_oos = int((out["score_source"].astype("string").str.lower() == "oos_predictions").sum())
    if matched == 2 and unmatched == 0 and matched_oos >= 1:
        LOG.info("[INFO] TRADE_ATTRIBUTION_SELF_TEST_PASS matched=%d", matched)
        return 0
    LOG.error(
        "TRADE_ATTRIBUTION_SELF_TEST_FAIL matched=%d unmatched=%d reasons=%s",
        matched,
        unmatched,
        unmatched_reason_counts,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        if args.self_test:
            return run_self_test()
        run_trade_attribution(args)
        return 0
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("TRADE_ATTRIBUTION_FAILED err=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
