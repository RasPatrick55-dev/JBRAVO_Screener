import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any, Mapping, Optional

import pandas as pd
from dateutil import parser
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def db_enabled() -> bool:
    """Return True if DATABASE_URL is present."""

    return bool(os.environ.get("DATABASE_URL"))


def get_engine() -> Optional[Engine]:
    """Return SQLAlchemy engine with pool_pre_ping=True."""

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.warning("[WARN] DB_DISABLED Missing DATABASE_URL")
        return None

    try:
        return create_engine(database_url, pool_pre_ping=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_ENGINE %s", exc)
        return None


def safe_connect_test() -> bool:
    """Return True if DB reachable, else False (log warning)."""

    try:
        engine = get_engine()
        if engine is None:
            return False

        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[WARN] DB_CONNECT_TEST %s", exc)
            return False
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_CONNECT_TEST_SETUP %s", exc)
        return False

    return True


def _engine_or_none() -> Optional[Engine]:
    if not db_enabled():
        return None
    return get_engine()


def _coerce_date(run_date: Any) -> str:
    if isinstance(run_date, datetime):
        return run_date.date().isoformat()
    if isinstance(run_date, date):
        return run_date.isoformat()
    try:
        return date.fromisoformat(str(run_date)).isoformat()
    except Exception:
        return str(run_date)


def _json_dumps_or_none(payload: Any) -> Optional[str]:
    if payload is None:
        return None
    try:
        return json.dumps(payload)
    except Exception:
        try:
            return json.dumps({"raw": str(payload)})
        except Exception:
            return None


def normalize_ts(value: Any, field: str | None = None) -> datetime | None:
    """Return a timezone-aware datetime parsed from ``value`` or None on failure."""

    if value is None:
        return None

    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            parsed = parser.isoparse(candidate)
        except Exception:
            logger.warning(
                "[WARN] EXECUTED_TRADES_TS_PARSE_FAIL field=%s value=%s", field or "unknown", value
            )
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    logger.warning("[WARN] EXECUTED_TRADES_TS_PARSE_FAIL field=%s value=%s", field or "unknown", value)
    return None


def normalize_score_breakdown(value: Any, symbol: str | None = None) -> Optional[str]:
    """Return a JSON-serialised score_breakdown or None on failure."""

    if value is None:
        return None

    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass

    normalized: Any
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            normalized = json.loads(stripped)
        except Exception:
            normalized = {"raw": value}
    elif isinstance(value, (Mapping, list)):
        normalized = value
    else:
        normalized = {"raw": value}

    try:
        return json.dumps(normalized)
    except Exception as exc:
        logger.warning(
            "[WARN] SCORE_BREAKDOWN_JSON_FAIL symbol=%s detail=%s", (symbol or "UNKNOWN").upper(), exc
        )
        return None


def _log_write_result(ok: bool, table: str, rows: int, err: Exception | None = None) -> None:
    if ok:
        logger.info("[INFO] DB_WRITE_OK table=%s rows=%s", table, rows)
    else:
        logger.warning("[WARN] DB_WRITE_FAILED table=%s err=%s", table, err)


def upsert_pipeline_run(
    run_date: Any,
    started_at: datetime | None,
    ended_at: datetime | None,
    rc: int,
    summary_dict: Mapping[str, Any] | None,
) -> None:
    engine = _engine_or_none()
    if engine is None:
        return

    payload = {
        "run_date": _coerce_date(run_date),
        "started_at": started_at,
        "ended_at": ended_at,
        "rc": int(rc),
        "summary": _json_dumps_or_none(summary_dict or {}),
    }
    stmt = text(
        """
        INSERT INTO pipeline_runs (run_date, started_at, ended_at, rc, summary)
        VALUES (:run_date, :started_at, :ended_at, :rc, CAST(:summary AS JSONB))
        ON CONFLICT (run_date) DO UPDATE
        SET started_at=EXCLUDED.started_at,
            ended_at=EXCLUDED.ended_at,
            rc=EXCLUDED.rc,
            summary=EXCLUDED.summary
        """
    )
    try:
        with engine.begin() as connection:
            connection.execute(stmt, payload)
        _log_write_result(True, "pipeline_runs", 1)
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_write_result(False, "pipeline_runs", 0, exc)


def insert_screener_candidates(run_date: Any, df_candidates: pd.DataFrame | None) -> None:
    if df_candidates is None or df_candidates.empty:
        return
    engine = _engine_or_none()
    if engine is None:
        return

    schema_stmt = text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'screener_candidates'
          AND column_name = 'run_date'
        LIMIT 1
        """
    )
    try:
        with engine.connect() as connection:
            has_run_date = connection.execute(schema_stmt).scalar() is not None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_INGEST_FAILED table=screener_candidates err=%s", exc)
        return

    if not has_run_date:
        logger.warning("[WARN] DB_SCHEMA_MISMATCH screener_candidates missing run_date")
        return

    normalized = df_candidates.copy()
    columns = {
        "timestamp": None,
        "symbol": None,
        "score": None,
        "exchange": None,
        "close": None,
        "volume": None,
        "universe_count": None,
        "score_breakdown": None,
        "entry_price": None,
        "adv20": None,
        "atrp": None,
        "source": None,
    }
    rows: list[dict[str, Any]] = []
    for _, row in normalized.iterrows():
        record = {}
        for key in columns:
            record[key] = row.get(key) if isinstance(row, Mapping) else row[key] if key in row else None
        symbol = (record.get("symbol") or "").upper()
        payload = {
            "run_date": _coerce_date(run_date),
            "timestamp": record.get("timestamp"),
            "symbol": symbol,
            "score": record.get("score"),
            "exchange": record.get("exchange"),
            "close": record.get("close"),
            "volume": record.get("volume"),
            "universe_count": record.get("universe_count"),
            "score_breakdown": normalize_score_breakdown(record.get("score_breakdown"), symbol=symbol),
            "entry_price": record.get("entry_price"),
            "adv20": record.get("adv20"),
            "atrp": record.get("atrp"),
            "source": record.get("source"),
        }
        rows.append(payload)

    stmt = text(
        """
        INSERT INTO screener_candidates (
            run_date, timestamp, symbol, score, exchange, close, volume,
            universe_count, score_breakdown, entry_price, adv20, atrp, source
        )
        VALUES (
            :run_date, :timestamp, :symbol, :score, :exchange, :close, :volume,
            :universe_count, CAST(:score_breakdown AS JSONB), :entry_price, :adv20, :atrp, :source
        )
        ON CONFLICT (run_date, symbol) DO UPDATE SET
            timestamp=EXCLUDED.timestamp,
            score=EXCLUDED.score,
            exchange=EXCLUDED.exchange,
            close=EXCLUDED.close,
            volume=EXCLUDED.volume,
            universe_count=EXCLUDED.universe_count,
            score_breakdown=EXCLUDED.score_breakdown,
            entry_price=EXCLUDED.entry_price,
            adv20=EXCLUDED.adv20,
            atrp=EXCLUDED.atrp,
            source=EXCLUDED.source
        """
    )
    try:
        with engine.begin() as connection:
            connection.execute(stmt, rows)
        logger.info("[INFO] DB_INGEST_OK table=screener_candidates rows=%s", len(rows))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_INGEST_FAILED table=screener_candidates err=%s", exc)


def insert_backtest_results(run_date: Any, df_results: pd.DataFrame | None) -> None:
    if df_results is None or df_results.empty:
        return
    engine = _engine_or_none()
    if engine is None:
        return

    rows: list[dict[str, Any]] = []
    for _, row in df_results.iterrows():
        payload = {
            "run_date": _coerce_date(run_date),
            "symbol": (row.get("symbol") or "").upper(),
            "trades": row.get("trades"),
            "win_rate": row.get("win_rate"),
            "net_pnl": row.get("net_pnl"),
            "expectancy": row.get("expectancy"),
            "profit_factor": row.get("profit_factor"),
            "max_drawdown": row.get("max_drawdown"),
            "sharpe": row.get("sharpe"),
            "sortino": row.get("sortino"),
        }
        rows.append(payload)

    stmt = text(
        """
        INSERT INTO backtest_results (
            run_date, symbol, trades, win_rate, net_pnl, expectancy,
            profit_factor, max_drawdown, sharpe, sortino
        )
        VALUES (
            :run_date, :symbol, :trades, :win_rate, :net_pnl, :expectancy,
            :profit_factor, :max_drawdown, :sharpe, :sortino
        )
        ON CONFLICT (run_date, symbol) DO UPDATE SET
            trades=EXCLUDED.trades,
            win_rate=EXCLUDED.win_rate,
            net_pnl=EXCLUDED.net_pnl,
            expectancy=EXCLUDED.expectancy,
            profit_factor=EXCLUDED.profit_factor,
            max_drawdown=EXCLUDED.max_drawdown,
            sharpe=EXCLUDED.sharpe,
            sortino=EXCLUDED.sortino
        """
    )
    try:
        with engine.begin() as connection:
            connection.execute(stmt, rows)
        _log_write_result(True, "backtest_results", len(rows))
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_write_result(False, "backtest_results", 0, exc)


def upsert_metrics_daily(run_date: Any, summary_metrics_dict: Mapping[str, Any] | None) -> None:
    if not summary_metrics_dict:
        return
    engine = _engine_or_none()
    if engine is None:
        return

    payload = {
        "run_date": _coerce_date(run_date),
        "total_trades": summary_metrics_dict.get("total_trades"),
        "win_rate": summary_metrics_dict.get("win_rate"),
        "net_pnl": summary_metrics_dict.get("net_pnl"),
        "expectancy": summary_metrics_dict.get("expectancy"),
        "profit_factor": summary_metrics_dict.get("profit_factor"),
        "max_drawdown": summary_metrics_dict.get("max_drawdown"),
        "sharpe": summary_metrics_dict.get("sharpe"),
        "sortino": summary_metrics_dict.get("sortino"),
    }
    stmt = text(
        """
        INSERT INTO metrics_daily (
            run_date, total_trades, win_rate, net_pnl, expectancy,
            profit_factor, max_drawdown, sharpe, sortino
        )
        VALUES (
            :run_date, :total_trades, :win_rate, :net_pnl, :expectancy,
            :profit_factor, :max_drawdown, :sharpe, :sortino
        )
        ON CONFLICT (run_date) DO UPDATE SET
            total_trades=EXCLUDED.total_trades,
            win_rate=EXCLUDED.win_rate,
            net_pnl=EXCLUDED.net_pnl,
            expectancy=EXCLUDED.expectancy,
            profit_factor=EXCLUDED.profit_factor,
            max_drawdown=EXCLUDED.max_drawdown,
            sharpe=EXCLUDED.sharpe,
            sortino=EXCLUDED.sortino
        """
    )
    try:
        with engine.begin() as connection:
            connection.execute(stmt, payload)
        _log_write_result(True, "metrics_daily", 1)
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_write_result(False, "metrics_daily", 0, exc)


def insert_executed_trade(row_dict: Mapping[str, Any] | None) -> bool:
    if not row_dict:
        return False
    engine = _engine_or_none()
    event_label = (row_dict.get("event_type") or row_dict.get("status") or row_dict.get("order_status") or "").upper()
    order_id = row_dict.get("order_id")
    if engine is None:
        logger.warning(
            "[WARN] DB_WRITE_FAILED table=executed_trades event=%s order_id=%s err=%s",
            event_label,
            order_id or "",
            "disabled",
        )
        _log_write_result(False, "executed_trades", 0, RuntimeError("db_disabled"))
        return False

    entry_time = normalize_ts(row_dict.get("entry_time"), field="entry_time") or datetime.now(timezone.utc)
    exit_time = normalize_ts(row_dict.get("exit_time"), field="exit_time")

    payload = {
        "symbol": (row_dict.get("symbol") or "").upper(),
        "qty": row_dict.get("qty"),
        "entry_time": entry_time,
        "entry_price": row_dict.get("entry_price"),
        "exit_time": exit_time,
        "exit_price": row_dict.get("exit_price"),
        "pnl": row_dict.get("pnl"),
        "net_pnl": row_dict.get("net_pnl"),
        "order_id": row_dict.get("order_id"),
        "status": row_dict.get("order_status") or row_dict.get("status"),
        "raw": _json_dumps_or_none(row_dict),
    }
    stmt = text(
        """
        INSERT INTO executed_trades (
            symbol, qty, entry_time, entry_price, exit_time, exit_price,
            pnl, net_pnl, order_id, status, raw
        )
        VALUES (
            :symbol, :qty, :entry_time, :entry_price, :exit_time, :exit_price,
            :pnl, :net_pnl, :order_id, :status, CAST(:raw AS JSONB)
        )
        """
    )
    try:
        with engine.begin() as connection:
            connection.execute(stmt, payload)
        logger.info(
            "[INFO] DB_WRITE_OK table=executed_trades event=%s order_id=%s symbol=%s",
            event_label or "",
            order_id or "",
            payload["symbol"],
        )
        _log_write_result(True, "executed_trades", 1)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "[WARN] DB_WRITE_FAILED table=executed_trades event=%s order_id=%s err=%s",
            event_label or "",
            order_id or "",
            exc,
        )
        _log_write_result(False, "executed_trades", 0, exc)
        return False
