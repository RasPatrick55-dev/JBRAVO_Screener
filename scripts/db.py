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


def insert_order_event(
    event: Mapping[str, Any] | None = None,
    *,
    engine: Optional[Engine] = None,
    event_type: str | None = None,
    symbol: str | None = None,
    qty: Any | None = None,
    order_id: str | None = None,
    status: str | None = None,
    event_time: Any | None = None,
    raw: Any | None = None,
) -> bool:
    if event is None:
        event = {}
    payload = dict(event)
    if event_type is not None:
        payload.setdefault("event_type", event_type)
    if symbol is not None:
        payload.setdefault("symbol", symbol)
    if qty is not None:
        payload.setdefault("qty", qty)
    if order_id is not None:
        payload.setdefault("order_id", order_id)
    if status is not None:
        payload.setdefault("status", status)
    if event_time is not None:
        payload.setdefault("event_time", event_time)
    payload.setdefault("event_type", "UNKNOWN")

    db_engine = engine or _engine_or_none()
    event_label = (payload.get("event_type") or "").upper()
    order_id_value = payload.get("order_id")
    if db_engine is None:
        logger.warning(
            "[WARN] DB_WRITE_FAILED table=order_events event=%s order_id=%s err=%s",
            event_label,
            order_id_value or "",
            "disabled",
        )
        _log_write_result(False, "order_events", 0, RuntimeError("db_disabled"))
        return False

    normalized_event_time = normalize_ts(payload.get("event_time"), field="event_time") or datetime.now(timezone.utc)
    raw_payload = raw if raw is not None else payload
    stmt_payload = {
        "symbol": (payload.get("symbol") or "").upper(),
        "qty": payload.get("qty"),
        "order_id": order_id_value,
        "status": payload.get("status"),
        "event_type": payload.get("event_type") or "UNKNOWN",
        "event_time": normalized_event_time,
        "raw": _json_dumps_or_none(raw_payload),
    }
    stmt = text(
        """
        INSERT INTO order_events (
            symbol, qty, order_id, status, event_type, event_time, raw
        )
        VALUES (
            :symbol, :qty, :order_id, :status, :event_type, :event_time, CAST(:raw AS JSONB)
        )
        """
    )
    try:
        with db_engine.begin() as connection:
            connection.execute(stmt, stmt_payload)
        _log_write_result(True, "order_events", 1)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "[WARN] DB_WRITE_FAILED table=order_events event=%s order_id=%s err=%s",
            event_label,
            order_id_value or "",
            exc,
        )
        _log_write_result(False, "order_events", 0, exc)
        return False


def get_open_trades(engine: Optional[Engine] = None, limit: int = 200) -> list[dict[str, Any]]:
    db_engine = engine or _engine_or_none()
    if db_engine is None:
        return []
    limit = max(1, int(limit or 0))
    stmt = text(
        """
        SELECT trade_id, symbol, qty, entry_time, entry_price, entry_order_id
        FROM trades
        WHERE status='OPEN'
        ORDER BY entry_time DESC NULLS LAST, trade_id DESC
        LIMIT :limit
        """
    )
    try:
        with db_engine.connect() as connection:
            rows = connection.execute(stmt, {"limit": limit}).mappings().all()
            return [dict(row) for row in rows]
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_TRADE_FETCH_FAILED err=%s", exc)
        return []


def close_trade(
    engine: Optional[Engine],
    trade_id: Any,
    exit_order_id: str | None,
    exit_time: Any,
    exit_price: Any,
    exit_reason: str | None,
) -> bool:
    db_engine = engine or _engine_or_none()
    if db_engine is None:
        logger.warning(
            "[WARN] DB_TRADE_CLOSE_FAILED trade_id=%s exit_order_id=%s err=%s",
            trade_id,
            exit_order_id or "",
            "disabled",
        )
        return False

    normalized_exit_time = normalize_ts(exit_time, field="exit_time") or datetime.now(timezone.utc)
    try:
        with db_engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT status, qty, entry_price
                    FROM trades
                    WHERE trade_id=:trade_id
                    """
                ),
                {"trade_id": trade_id},
            ).fetchone()
            if row is None:
                logger.warning(
                    "[WARN] DB_TRADE_CLOSE_FAILED trade_id=%s exit_order_id=%s err=%s",
                    trade_id,
                    exit_order_id or "",
                    "trade_not_found",
                )
                return False
            status_value = (row[0] or "").upper()
            if status_value == "CLOSED":
                logger.info(
                    "[INFO] DB_TRADE_CLOSE_SKIPPED trade_id=%s exit_order_id=%s reason=%s",
                    trade_id,
                    exit_order_id or "",
                    "status_closed",
                )
                return False
            qty_value = row[1]
            entry_price_value = row[2]
            realized_pnl = None
            try:
                if exit_price is not None and entry_price_value is not None:
                    realized_pnl = float(exit_price) - float(entry_price_value)
                    if qty_value is not None:
                        realized_pnl *= float(qty_value)
            except Exception:
                realized_pnl = None
            connection.execute(
                text(
                    """
                    UPDATE trades
                    SET exit_order_id=:exit_order_id,
                        exit_time=:exit_time,
                        exit_price=:exit_price,
                        realized_pnl=:realized_pnl,
                        exit_reason=:exit_reason,
                        status='CLOSED',
                        updated_at=now()
                    WHERE trade_id=:trade_id
                    """
                ),
                {
                    "exit_order_id": exit_order_id,
                    "exit_time": normalized_exit_time,
                    "exit_price": exit_price,
                    "realized_pnl": realized_pnl,
                    "exit_reason": exit_reason,
                    "trade_id": trade_id,
                },
            )
        _log_write_result(True, "trades", 1)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "[WARN] DB_TRADE_CLOSE_FAILED trade_id=%s exit_order_id=%s err=%s",
            trade_id,
            exit_order_id or "",
            exc,
        )
        return False


def update_trade_exit_fields(
    engine: Optional[Engine],
    trade_id: Any,
    *,
    exit_order_id: str | None = None,
    exit_time: Any | None = None,
    exit_price: Any | None = None,
    exit_reason: str | None = None,
) -> bool:
    db_engine = engine or _engine_or_none()
    if db_engine is None:
        logger.warning(
            "[WARN] DB_TRADE_EXIT_UPDATE_FAILED trade_id=%s exit_order_id=%s err=%s",
            trade_id,
            exit_order_id or "",
            "disabled",
        )
        return False

    normalized_exit_time = None
    if exit_time is not None:
        normalized_exit_time = normalize_ts(exit_time, field="exit_time") or datetime.now(timezone.utc)

    try:
        with db_engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT status, qty, entry_price, exit_price, realized_pnl
                    FROM trades
                    WHERE trade_id=:trade_id
                    """
                ),
                {"trade_id": trade_id},
            ).fetchone()
            if row is None:
                logger.warning(
                    "[WARN] DB_TRADE_EXIT_UPDATE_FAILED trade_id=%s exit_order_id=%s err=%s",
                    trade_id,
                    exit_order_id or "",
                    "trade_not_found",
                )
                return False

            status_value = (row[0] or "").upper()
            qty_value = row[1]
            entry_price_value = row[2]
            existing_exit_price = row[3]
            existing_realized_pnl = row[4]
            final_exit_price = exit_price if exit_price is not None else existing_exit_price

            realized_pnl = existing_realized_pnl
            try:
                if final_exit_price is not None and entry_price_value is not None:
                    realized_pnl = float(final_exit_price) - float(entry_price_value)
                    if qty_value is not None:
                        realized_pnl *= float(qty_value)
            except Exception:
                realized_pnl = existing_realized_pnl

            should_skip = (
                status_value == "CLOSED"
                and exit_order_id is None
                and normalized_exit_time is None
                and exit_price is None
                and exit_reason is None
            )
            if should_skip:
                logger.info(
                    "[INFO] DB_TRADE_EXIT_UPDATE_SKIPPED trade_id=%s reason=%s",
                    trade_id,
                    "no_fields",
                )
                return False

            connection.execute(
                text(
                    """
                    UPDATE trades
                    SET exit_order_id=COALESCE(:exit_order_id, exit_order_id),
                        exit_time=COALESCE(:exit_time, exit_time),
                        exit_price=COALESCE(:exit_price, exit_price),
                        realized_pnl=:realized_pnl,
                        exit_reason=COALESCE(:exit_reason, exit_reason),
                        status='CLOSED',
                        updated_at=now()
                    WHERE trade_id=:trade_id
                    """
                ),
                {
                    "exit_order_id": exit_order_id,
                    "exit_time": normalized_exit_time,
                    "exit_price": exit_price,
                    "realized_pnl": realized_pnl,
                    "exit_reason": exit_reason,
                    "trade_id": trade_id,
                },
            )
        _log_write_result(True, "trades", 1)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "[WARN] DB_TRADE_EXIT_UPDATE_FAILED trade_id=%s exit_order_id=%s err=%s",
            trade_id,
            exit_order_id or "",
            exc,
        )
        return False


def upsert_trade_on_buy_fill(
    symbol: str,
    qty: Any,
    entry_order_id: str | None,
    entry_time: Any,
    entry_price: Any,
) -> bool:
    engine = _engine_or_none()
    if engine is None:
        logger.warning(
            "[WARN] DB_TRADE_UPSERT_FAILED symbol=%s entry_order_id=%s err=%s",
            symbol or "",
            entry_order_id or "",
            "disabled",
        )
        return False

    normalized_entry_time = normalize_ts(entry_time, field="entry_time") or datetime.now(timezone.utc)
    payload = {
        "symbol": (symbol or "").upper(),
        "qty": qty,
        "entry_order_id": entry_order_id,
        "entry_time": normalized_entry_time,
        "entry_price": entry_price,
    }
    stmt = text(
        """
        INSERT INTO trades (
            symbol, qty, entry_order_id, entry_time, entry_price, status, updated_at
        )
        VALUES (
            :symbol, :qty, :entry_order_id, :entry_time, :entry_price, 'OPEN', now()
        )
        ON CONFLICT (entry_order_id) DO UPDATE SET
            symbol=EXCLUDED.symbol,
            qty=EXCLUDED.qty,
            entry_time=EXCLUDED.entry_time,
            entry_price=EXCLUDED.entry_price,
            status='OPEN',
            updated_at=now()
        """
    )
    try:
        with engine.begin() as connection:
            connection.execute(stmt, payload)
        _log_write_result(True, "trades", 1)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "[WARN] DB_TRADE_UPSERT_FAILED symbol=%s entry_order_id=%s err=%s",
            symbol or "",
            entry_order_id or "",
            exc,
        )
        return False


def close_trade_on_sell_fill(
    symbol: str,
    exit_order_id: str | None,
    exit_time: Any,
    exit_price: Any,
    exit_reason: str | None = None,
) -> bool:
    engine = _engine_or_none()
    if engine is None:
        logger.warning(
            "[WARN] DB_TRADE_CLOSE_FAILED symbol=%s exit_order_id=%s err=%s",
            symbol or "",
            exit_order_id or "",
            "disabled",
        )
        return False

    normalized_exit_time = normalize_ts(exit_time, field="exit_time") or datetime.now(timezone.utc)
    symbol_value = (symbol or "").upper()
    try:
        with engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT trade_id, qty, entry_price
                    FROM trades
                    WHERE symbol=:symbol AND status='OPEN'
                    ORDER BY entry_time DESC NULLS LAST, trade_id DESC
                    LIMIT 1
                    """
                ),
                {"symbol": symbol_value},
            ).fetchone()
            if row is None:
                logger.warning(
                    "[WARN] DB_TRADE_CLOSE_FAILED symbol=%s exit_order_id=%s err=%s",
                    symbol_value,
                    exit_order_id or "",
                    "no_open_trade",
                )
                return False
            trade_id = row[0]
            qty_value = row[1]
            entry_price_value = row[2]
            realized_pnl = None
            try:
                if exit_price is not None and entry_price_value is not None:
                    realized_pnl = float(exit_price) - float(entry_price_value)
                    if qty_value is not None:
                        realized_pnl *= float(qty_value)
            except Exception:
                realized_pnl = None
            connection.execute(
                text(
                    """
                    UPDATE trades
                    SET exit_order_id=:exit_order_id,
                        exit_time=:exit_time,
                        exit_price=:exit_price,
                        realized_pnl=:realized_pnl,
                        exit_reason=:exit_reason,
                        status='CLOSED',
                        updated_at=now()
                    WHERE trade_id=:trade_id
                    """
                ),
                {
                    "exit_order_id": exit_order_id,
                    "exit_time": normalized_exit_time,
                    "exit_price": exit_price,
                    "realized_pnl": realized_pnl,
                    "exit_reason": exit_reason,
                    "trade_id": trade_id,
                },
            )
        _log_write_result(True, "trades", 1)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "[WARN] DB_TRADE_CLOSE_FAILED symbol=%s exit_order_id=%s err=%s",
            symbol_value,
            exit_order_id or "",
            exc,
        )
        return False


def count_open_trades(symbol: str | None = None) -> int:
    engine = _engine_or_none()
    if engine is None:
        return 0
    try:
        query = """
            SELECT COUNT(*) FROM trades
            WHERE status='OPEN'
        """
        params: dict[str, Any] = {}
        if symbol:
            query += " AND symbol=:symbol"
            params["symbol"] = symbol.upper()
        with engine.connect() as connection:
            result = connection.execute(text(query), params).scalar()
            return int(result or 0)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_TRADE_COUNT_FAILED symbol=%s err=%s", symbol or "", exc)
        return 0
