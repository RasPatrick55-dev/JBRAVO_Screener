import json
import logging
import os
from pathlib import Path
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Iterator, Mapping, Optional
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
import psycopg2
from dateutil import parser
from psycopg2 import extras
from psycopg2.extensions import connection as PGConnection

logger = logging.getLogger(__name__)

_DUMPED_CANDIDATES = False

DEFAULT_DB_HOST = "localhost"
DEFAULT_DB_PORT = 9999
DEFAULT_DB_NAME = "jbravo"
DEFAULT_DB_USER = "super"
DEFAULT_CONNECT_TIMEOUT = 10

_EXPLICIT_ENV_KEYS = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_USERNAME",
    "DB_PASSWORD",
    "DB_PASS",
    "DB_SSLMODE",
)


def _env_truthy(key: str) -> bool:
    value = os.environ.get(key)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_database_url(database_url: str) -> dict[str, Any]:
    parsed = urlparse(database_url)
    query = parse_qs(parsed.query or "")
    return {
        "host": parsed.hostname,
        "port": parsed.port,
        "dbname": parsed.path.lstrip("/") if parsed.path else None,
        "user": unquote(parsed.username) if parsed.username else None,
        "password": unquote(parsed.password) if parsed.password else None,
        "sslmode": (query.get("sslmode", [None]) or [None])[0],
    }


def _resolve_db_config() -> Optional[dict[str, Any]]:
    if _env_truthy("DB_DISABLED"):
        return None

    has_explicit = any(os.environ.get(key) for key in _EXPLICIT_ENV_KEYS)
    if has_explicit:
        return {
            "host": os.environ.get("DB_HOST", DEFAULT_DB_HOST),
            "port": _coerce_int(os.environ.get("DB_PORT"), DEFAULT_DB_PORT),
            "dbname": os.environ.get("DB_NAME", DEFAULT_DB_NAME),
            "user": os.environ.get("DB_USER") or os.environ.get("DB_USERNAME") or DEFAULT_DB_USER,
            "password": os.environ.get("DB_PASSWORD") or os.environ.get("DB_PASS"),
            "sslmode": os.environ.get("DB_SSLMODE"),
            "connect_timeout": _coerce_int(
                os.environ.get("DB_CONNECT_TIMEOUT"), DEFAULT_CONNECT_TIMEOUT
            ),
        }

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        parsed = _parse_database_url(database_url)
        return {
            "host": parsed.get("host") or DEFAULT_DB_HOST,
            "port": _coerce_int(parsed.get("port"), DEFAULT_DB_PORT),
            "dbname": parsed.get("dbname") or DEFAULT_DB_NAME,
            "user": parsed.get("user") or DEFAULT_DB_USER,
            "password": parsed.get("password"),
            "sslmode": parsed.get("sslmode"),
            "connect_timeout": _coerce_int(
                os.environ.get("DB_CONNECT_TIMEOUT"), DEFAULT_CONNECT_TIMEOUT
            ),
        }

    return {
        "host": DEFAULT_DB_HOST,
        "port": DEFAULT_DB_PORT,
        "dbname": DEFAULT_DB_NAME,
        "user": DEFAULT_DB_USER,
        "password": os.environ.get("DB_PASSWORD") or os.environ.get("DB_PASS"),
        "sslmode": os.environ.get("DB_SSLMODE"),
        "connect_timeout": _coerce_int(
            os.environ.get("DB_CONNECT_TIMEOUT"), DEFAULT_CONNECT_TIMEOUT
        ),
    }


def db_enabled() -> bool:
    """Return True if database access is enabled."""

    return _resolve_db_config() is not None


def get_db_conn() -> Optional[PGConnection]:
    """Return a psycopg2 connection using resolved environment config."""

    config = _resolve_db_config()
    if config is None:
        logger.warning("[WARN] DB_DISABLED via DB_DISABLED")
        return None

    try:
        connect_kwargs = {
            "host": config.get("host"),
            "port": config.get("port"),
            "dbname": config.get("dbname"),
            "user": config.get("user"),
            "connect_timeout": config.get("connect_timeout", DEFAULT_CONNECT_TIMEOUT),
        }
        password = config.get("password")
        if password:
            connect_kwargs["password"] = password
        sslmode = config.get("sslmode")
        if sslmode:
            connect_kwargs["sslmode"] = sslmode
        return psycopg2.connect(**connect_kwargs)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "[WARN] DB_CONNECT_FAIL host=%s port=%s dbname=%s user=%s err=%s",
            config.get("host"),
            config.get("port"),
            config.get("dbname"),
            config.get("user"),
            exc,
        )
        return None


def get_engine() -> Optional[PGConnection]:
    """Compatibility alias for get_db_conn (psycopg2 connection)."""

    return get_db_conn()


def check_db_connection() -> bool:
    """Return True if DB reachable, else False (log warning)."""

    try:
        conn = get_db_conn()
        if conn is None:
            return False

        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[WARN] DB_CONNECT_TEST %s", exc)
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_CONNECT_TEST_SETUP %s", exc)
        return False

    return True


def safe_connect_test() -> bool:
    return check_db_connection()


@contextmanager
def _maybe_conn(engine: Optional[PGConnection]) -> Iterator[Optional[PGConnection]]:
    if engine is not None:
        yield engine
        return

    conn = get_db_conn() if db_enabled() else None
    if conn is None:
        yield None
        return
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _conn_or_none() -> Optional[PGConnection]:
    if not db_enabled():
        return None
    return get_db_conn()


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


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return None


def normalize_gate_fail_reason(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    if isinstance(value, (list, tuple, set)):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(cleaned) if cleaned else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            cleaned = [str(item).strip() for item in parsed if str(item).strip()]
            return ",".join(cleaned) if cleaned else None
        raw = text
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        parts = [
            part.strip().strip("\"'") for part in raw.split(",") if part.strip().strip("\"'")
        ]
        return ",".join(parts) if parts else text
    rendered = str(value).strip()
    return rendered or None


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


def get_reconcile_state(engine: Optional[PGConnection] = None) -> dict[str, Any]:
    with _maybe_conn(engine) as conn:
        if conn is None:
            return {}
        try:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
                cursor.execute("SELECT last_after, last_ran_at FROM reconcile_state WHERE id=1")
                row = cursor.fetchone()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[WARN] DB_RECONCILE_STATE_FETCH err=%s", exc)
            return {}

    if not row:
        return {}

    last_after = normalize_ts(row.get("last_after"), field="last_after")
    last_ran_at = normalize_ts(row.get("last_ran_at"), field="last_ran_at")
    return {"last_after": last_after, "last_ran_at": last_ran_at}


def set_reconcile_state(
    engine: Optional[PGConnection], last_after: datetime | None, last_ran_at: datetime | None
) -> bool:
    with _maybe_conn(engine) as conn:
        if conn is None:
            logger.warning("[WARN] DB_RECONCILE_STATE_WRITE_FAILED err=%s", "db_disabled")
            return False

        payload = {
            "last_after": normalize_ts(last_after, field="last_after"),
            "last_ran_at": normalize_ts(last_ran_at, field="last_ran_at"),
        }
        stmt = """
            INSERT INTO reconcile_state (id, last_after, last_ran_at)
            VALUES (1, %(last_after)s, %(last_ran_at)s)
            ON CONFLICT (id) DO UPDATE SET
                last_after=EXCLUDED.last_after,
                last_ran_at=EXCLUDED.last_ran_at,
                updated_at=now()
        """
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(stmt, payload)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[WARN] DB_RECONCILE_STATE_WRITE_FAILED err=%s", exc)
            return False


def upsert_pipeline_run(
    run_date: Any,
    started_at: datetime | None,
    ended_at: datetime | None,
    rc: int,
    summary_dict: Mapping[str, Any] | None,
) -> None:
    conn = _conn_or_none()
    if conn is None:
        return

    payload = {
        "run_date": _coerce_date(run_date),
        "started_at": started_at,
        "ended_at": ended_at,
        "rc": int(rc),
        "summary": _json_dumps_or_none(summary_dict or {}),
    }
    stmt = """
        INSERT INTO pipeline_runs (run_date, started_at, ended_at, rc, summary)
        VALUES (%(run_date)s, %(started_at)s, %(ended_at)s, %(rc)s, CAST(%(summary)s AS JSONB))
        ON CONFLICT (run_date) DO UPDATE
        SET started_at=EXCLUDED.started_at,
            ended_at=EXCLUDED.ended_at,
            rc=EXCLUDED.rc,
            summary=EXCLUDED.summary
    """
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(stmt, payload)
        _log_write_result(True, "pipeline_runs", 1)
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_write_result(False, "pipeline_runs", 0, exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def insert_screener_candidates(
    run_date: Any,
    df_candidates: pd.DataFrame | None,
    *,
    run_ts_utc: Any | None = None,
) -> None:
    if df_candidates is None or df_candidates.empty:
        return
    global _DUMPED_CANDIDATES
    if _env_truthy("DEBUG_DUMP_CANDIDATES") and not _DUMPED_CANDIDATES:
        try:
            diagnostics_dir = Path("data") / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            dump_cols = [
                "timestamp",
                "symbol",
                "score",
                "sma9",
                "ema20",
                "sma180",
                "rsi14",
                "passed_gates",
                "gate_fail_reason",
                "source",
            ]
            dump_frame = df_candidates.reindex(columns=dump_cols, fill_value=pd.NA)
            dump_path = diagnostics_dir / f"candidates_insert_{_coerce_date(run_date)}.csv"
            dump_frame.to_csv(dump_path, index=False)
            logger.info("[INFO] DEBUG_DUMP_CANDIDATES path=%s rows=%d", dump_path, len(dump_frame))
            _DUMPED_CANDIDATES = True
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.warning("[WARN] DEBUG_DUMP_CANDIDATES_FAILED err=%s", exc)
    conn = _conn_or_none()
    if conn is None:
        return

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'screener_candidates'
                  AND column_name = 'run_date'
                LIMIT 1
                """
            )
            has_run_date = cursor.fetchone() is not None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_INGEST_FAILED table=screener_candidates err=%s", exc)
        try:
            conn.close()
        except Exception:
            pass
        return

    if not has_run_date:
        logger.warning("[WARN] DB_SCHEMA_MISMATCH screener_candidates missing run_date")
        try:
            conn.close()
        except Exception:
            pass
        return

    normalized = df_candidates.copy()
    columns = {
        "timestamp": None,
        "symbol": None,
        "score": None,
        "final_score": None,
        "exchange": None,
        "close": None,
        "volume": None,
        "universe_count": None,
        "score_breakdown": None,
        "entry_price": None,
        "adv20": None,
        "atrp": None,
        "source": None,
        "sma9": None,
        "ema20": None,
        "sma180": None,
        "rsi14": None,
        "passed_gates": None,
        "gate_fail_reason": None,
        "ml_weight_used": None,
    }
    aliases = {
        "timestamp": ("timestamp",),
        "symbol": ("symbol",),
        "score": ("score", "Score"),
        "final_score": ("final_score", "ml_adjusted_score", "Final Score"),
        "exchange": ("exchange", "Exchange"),
        "close": ("close", "Close", "price", "last"),
        "volume": ("volume", "Volume"),
        "universe_count": ("universe_count", "Universe Count", "universe count"),
        "score_breakdown": ("score_breakdown", "Score Breakdown", "score breakdown"),
        "entry_price": ("entry_price", "Entry", "entry"),
        "adv20": ("adv20", "ADV20", "adv_20"),
        "atrp": ("atrp", "ATR_pct", "ATR%", "atr_percent", "ATR"),
        "source": ("source",),
        "sma9": ("sma9", "SMA9"),
        "ema20": ("ema20", "EMA20"),
        "sma180": ("sma180", "SMA180"),
        "rsi14": ("rsi14", "RSI14"),
        "passed_gates": ("passed_gates", "gates_passed"),
        "gate_fail_reason": ("gate_fail_reason",),
        "ml_weight_used": ("ml_weight_used", "ML_WEIGHT"),
    }

    def _row_value(row: Mapping[str, Any], key: str) -> Any:
        for alias in aliases.get(key, (key,)):
            if alias in row:
                return row.get(alias) if hasattr(row, "get") else row[alias]
        if hasattr(row, "get"):
            return row.get(key)
        return row[key] if key in row else None

    rows: list[dict[str, Any]] = []
    for _, row in normalized.iterrows():
        record = {}
        for key in columns:
            record[key] = _row_value(row, key)
        symbol = (record.get("symbol") or "").upper()
        payload = {
            "run_date": _coerce_date(run_date),
            "timestamp": record.get("timestamp"),
            "symbol": symbol,
            "score": record.get("score"),
            "final_score": record.get("final_score"),
            "exchange": record.get("exchange"),
            "close": record.get("close"),
            "volume": record.get("volume"),
            "universe_count": record.get("universe_count"),
            "score_breakdown": normalize_score_breakdown(record.get("score_breakdown"), symbol=symbol),
            "entry_price": record.get("entry_price"),
            "adv20": record.get("adv20"),
            "atrp": record.get("atrp"),
            "source": record.get("source"),
            "sma9": record.get("sma9"),
            "ema20": record.get("ema20"),
            "sma180": record.get("sma180"),
            "rsi14": record.get("rsi14"),
            "passed_gates": _coerce_bool(record.get("passed_gates")),
            "gate_fail_reason": normalize_gate_fail_reason(record.get("gate_fail_reason")),
            "ml_weight_used": record.get("ml_weight_used"),
        }
        rows.append(payload)

    inserted = False
    try:
        with conn:
            with conn.cursor() as cursor:
                extras.execute_batch(
                    cursor,
                    """
                    INSERT INTO screener_candidates (
                        run_date, timestamp, symbol, score, final_score, exchange, close, volume,
                        universe_count, score_breakdown, entry_price, adv20, atrp, source,
                        sma9, ema20, sma180, rsi14, passed_gates, gate_fail_reason, ml_weight_used
                    )
                    VALUES (
                        %(run_date)s, %(timestamp)s, %(symbol)s, %(score)s, %(final_score)s, %(exchange)s, %(close)s, %(volume)s,
                        %(universe_count)s, CAST(%(score_breakdown)s AS JSONB), %(entry_price)s, %(adv20)s,
                        %(atrp)s, %(source)s, %(sma9)s, %(ema20)s, %(sma180)s, %(rsi14)s, %(passed_gates)s,
                        %(gate_fail_reason)s, %(ml_weight_used)s
                    )
                    ON CONFLICT (run_date, symbol) DO UPDATE SET
                        timestamp=EXCLUDED.timestamp,
                        score=EXCLUDED.score,
                        final_score=EXCLUDED.final_score,
                        exchange=EXCLUDED.exchange,
                        close=EXCLUDED.close,
                        volume=EXCLUDED.volume,
                        universe_count=EXCLUDED.universe_count,
                        score_breakdown=EXCLUDED.score_breakdown,
                        entry_price=EXCLUDED.entry_price,
                        adv20=EXCLUDED.adv20,
                        atrp=EXCLUDED.atrp,
                        source=EXCLUDED.source,
                        sma9=EXCLUDED.sma9,
                        ema20=EXCLUDED.ema20,
                        sma180=EXCLUDED.sma180,
                        rsi14=EXCLUDED.rsi14,
                        passed_gates=EXCLUDED.passed_gates,
                        gate_fail_reason=EXCLUDED.gate_fail_reason,
                        ml_weight_used=EXCLUDED.ml_weight_used
                    """,
                    rows,
                    page_size=200,
                )
        inserted = True
        logger.info("[INFO] DB_INGEST_OK table=screener_candidates rows=%s", len(rows))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_INGEST_FAILED table=screener_candidates err=%s", exc)

    # Run scoping is stored in screener_run_map_app because screener_candidates is owned by super.
    if inserted and run_ts_utc is not None:
        normalized_run_ts = normalize_ts(run_ts_utc, field="run_ts_utc")
        if normalized_run_ts is not None:
            normalized_run_ts = normalized_run_ts.replace(microsecond=0)
            map_rows = [
                {"run_ts_utc": normalized_run_ts, "symbol": row.get("symbol")}
                for row in rows
                if row.get("symbol")
            ]
            create_stmt = """
                CREATE TABLE IF NOT EXISTS screener_run_map_app (
                    run_ts_utc TIMESTAMPTZ,
                    symbol TEXT,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (run_ts_utc, symbol)
                )
            """
            insert_stmt = """
                INSERT INTO screener_run_map_app (run_ts_utc, symbol)
                VALUES (%(run_ts_utc)s, %(symbol)s)
                ON CONFLICT (run_ts_utc, symbol) DO NOTHING
            """
            try:
                with conn:
                    with conn.cursor() as cursor:
                        cursor.execute(create_stmt)
                        extras.execute_batch(cursor, insert_stmt, map_rows, page_size=200)
                _log_write_result(True, "screener_run_map_app", len(map_rows))
            except Exception as exc:  # pragma: no cover - defensive logging
                _log_write_result(False, "screener_run_map_app", 0, exc)

    try:
        conn.close()
    except Exception:
        pass


def insert_pipeline_health(record: Mapping[str, Any]) -> None:
    if not record:
        return
    conn = _conn_or_none()
    if conn is None:
        return

    def _coerce_ts(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    payload = {
        "run_date": _coerce_date(record.get("run_date")),
        "run_ts_utc": _coerce_ts(record.get("run_ts_utc")),
        "mode": record.get("mode"),
        "symbols_in": record.get("symbols_in"),
        "with_bars": record.get("with_bars"),
        "coarse_rows": record.get("coarse_rows"),
        "shortlist_rows": record.get("shortlist_rows"),
        "final_rows": record.get("final_rows"),
        "gated_rows": record.get("gated_rows"),
        "fallback_used": record.get("fallback_used"),
        "db_ingest_rows": record.get("db_ingest_rows"),
        "notes": record.get("notes"),
    }
    create_stmt = """
        CREATE TABLE IF NOT EXISTS pipeline_health_app (
            run_date DATE,
            run_ts_utc TIMESTAMPTZ,
            mode TEXT,
            symbols_in INT,
            with_bars INT,
            coarse_rows INT,
            shortlist_rows INT,
            final_rows INT,
            gated_rows INT,
            fallback_used BOOLEAN,
            db_ingest_rows INT,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (run_ts_utc, mode)
        )
    """
    insert_stmt = """
        INSERT INTO pipeline_health_app (
            run_date, run_ts_utc, mode, symbols_in, with_bars, coarse_rows, shortlist_rows,
            final_rows, gated_rows, fallback_used, db_ingest_rows, notes
        )
        VALUES (
            %(run_date)s, %(run_ts_utc)s, %(mode)s, %(symbols_in)s, %(with_bars)s, %(coarse_rows)s,
            %(shortlist_rows)s, %(final_rows)s, %(gated_rows)s, %(fallback_used)s,
            %(db_ingest_rows)s, %(notes)s
        )
        ON CONFLICT (run_ts_utc, mode) DO UPDATE SET
            run_date=EXCLUDED.run_date,
            symbols_in=EXCLUDED.symbols_in,
            with_bars=EXCLUDED.with_bars,
            coarse_rows=EXCLUDED.coarse_rows,
            shortlist_rows=EXCLUDED.shortlist_rows,
            final_rows=EXCLUDED.final_rows,
            gated_rows=EXCLUDED.gated_rows,
            fallback_used=EXCLUDED.fallback_used,
            db_ingest_rows=EXCLUDED.db_ingest_rows,
            notes=EXCLUDED.notes
    """
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(create_stmt)
                cursor.execute(insert_stmt, payload)
        _log_write_result(True, "pipeline_health", 1)
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_write_result(False, "pipeline_health", 0, exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def insert_bar_coverage(record: Mapping[str, Any]) -> None:
    if not record:
        return
    conn = _conn_or_none()
    if conn is None:
        return

    payload = {
        "run_ts": record.get("run_ts"),
        "run_date": _coerce_date(record.get("run_date")),
        "mode": record.get("mode"),
        "feed": record.get("feed"),
        "symbols_requested": record.get("symbols_requested"),
        "symbols_with_bars": record.get("symbols_with_bars"),
        "symbols_missing": record.get("symbols_missing"),
    }
    create_stmt = """
        CREATE TABLE IF NOT EXISTS bar_coverage_app (
            run_ts TIMESTAMPTZ,
            run_date DATE,
            mode TEXT,
            feed TEXT,
            symbols_requested INT,
            symbols_with_bars INT,
            symbols_missing INT,
            created_at TIMESTAMPTZ DEFAULT now()
        )
    """
    insert_stmt = """
        INSERT INTO bar_coverage_app (
            run_ts, run_date, mode, feed, symbols_requested,
            symbols_with_bars, symbols_missing
        )
        VALUES (
            %(run_ts)s, %(run_date)s, %(mode)s, %(feed)s, %(symbols_requested)s,
            %(symbols_with_bars)s, %(symbols_missing)s
        )
    """
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(create_stmt)
                cursor.execute(insert_stmt, payload)
        _log_write_result(True, "bar_coverage_app", 1)
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_write_result(False, "bar_coverage_app", 0, exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_latest_pipeline_health_run_ts() -> Optional[datetime]:
    conn = _conn_or_none()
    if conn is None:
        return None
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_ts_utc
                FROM pipeline_health_app
                ORDER BY run_ts_utc DESC NULLS LAST
                LIMIT 1
                """
            )
            row = cursor.fetchone()
            return row[0] if row else None
    except Exception:  # pragma: no cover - defensive guard
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def insert_backtest_results(run_date: Any, df_results: pd.DataFrame | None) -> bool:
    if df_results is None or df_results.empty:
        return False
    conn = _conn_or_none()
    if conn is None:
        return False

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

    try:
        with conn:
            with conn.cursor() as cursor:
                extras.execute_batch(
                    cursor,
                    """
                    INSERT INTO backtest_results (
                        run_date, symbol, trades, win_rate, net_pnl, expectancy,
                        profit_factor, max_drawdown, sharpe, sortino
                    )
                    VALUES (
                        %(run_date)s, %(symbol)s, %(trades)s, %(win_rate)s, %(net_pnl)s, %(expectancy)s,
                        %(profit_factor)s, %(max_drawdown)s, %(sharpe)s, %(sortino)s
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
                    """,
                    rows,
                    page_size=200,
                )
        _log_write_result(True, "backtest_results", len(rows))
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_write_result(False, "backtest_results", 0, exc)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def upsert_metrics_daily(run_date: Any, summary_metrics_dict: Mapping[str, Any] | None) -> None:
    if not summary_metrics_dict:
        return
    conn = _conn_or_none()
    if conn is None:
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
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO metrics_daily (
                        run_date, total_trades, win_rate, net_pnl, expectancy,
                        profit_factor, max_drawdown, sharpe, sortino
                    )
                    VALUES (
                        %(run_date)s, %(total_trades)s, %(win_rate)s, %(net_pnl)s, %(expectancy)s,
                        %(profit_factor)s, %(max_drawdown)s, %(sharpe)s, %(sortino)s
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
                    """,
                    payload,
                )
        _log_write_result(True, "metrics_daily", 1)
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_write_result(False, "metrics_daily", 0, exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def insert_executed_trade(row_dict: Mapping[str, Any] | None) -> bool:
    if not row_dict:
        return False
    conn = _conn_or_none()
    event_label = (row_dict.get("event_type") or row_dict.get("status") or row_dict.get("order_status") or "").upper()
    order_id = row_dict.get("order_id")
    if conn is None:
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
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO executed_trades (
                        symbol, qty, entry_time, entry_price, exit_time, exit_price,
                        pnl, net_pnl, order_id, status, raw
                    )
                    VALUES (
                        %(symbol)s, %(qty)s, %(entry_time)s, %(entry_price)s, %(exit_time)s, %(exit_price)s,
                        %(pnl)s, %(net_pnl)s, %(order_id)s, %(status)s, CAST(%(raw)s AS JSONB)
                    )
                    """,
                    payload,
                )
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
    finally:
        try:
            conn.close()
        except Exception:
            pass


def insert_order_event(
    event: Mapping[str, Any] | None = None,
    *,
    engine: Optional[PGConnection] = None,
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

    event_label = (payload.get("event_type") or "").upper()
    order_id_value = payload.get("order_id")
    with _maybe_conn(engine) as conn:
        if conn is None:
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
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO order_events (
                            symbol, qty, order_id, status, event_type, event_time, raw
                        )
                        VALUES (
                            %(symbol)s, %(qty)s, %(order_id)s, %(status)s, %(event_type)s,
                            %(event_time)s, CAST(%(raw)s AS JSONB)
                        )
                        """,
                        stmt_payload,
                    )
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


def get_open_trades(engine: Optional[PGConnection] = None, limit: int = 200) -> list[dict[str, Any]]:
    limit = max(1, int(limit or 0))
    with _maybe_conn(engine) as conn:
        if conn is None:
            return []
        try:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT trade_id, symbol, qty, entry_time, entry_price, entry_order_id
                    FROM trades
                    WHERE status='OPEN'
                    ORDER BY entry_time DESC NULLS LAST, trade_id DESC
                    LIMIT %(limit)s
                    """,
                    {"limit": limit},
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[WARN] DB_TRADE_FETCH_FAILED err=%s", exc)
            return []


def get_closed_trades_missing_exit(
    engine: Optional[PGConnection], updated_after: datetime, limit: int = 200
) -> list[dict[str, Any]]:
    limit = max(1, int(limit or 0))
    with _maybe_conn(engine) as conn:
        if conn is None:
            return []
        try:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT trade_id, symbol, qty, entry_price, exit_price, exit_order_id, exit_time, realized_pnl
                    FROM trades
                    WHERE status='CLOSED'
                      AND (exit_price IS NULL OR exit_order_id IS NULL OR realized_pnl IS NULL)
                      AND updated_at >= %(updated_after)s
                    ORDER BY updated_at DESC NULLS LAST, trade_id DESC
                    LIMIT %(limit)s
                    """,
                    {"updated_after": updated_after, "limit": limit},
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[WARN] DB_TRADE_FETCH_FAILED err=%s", exc)
            return []


def close_trade(
    engine: Optional[PGConnection],
    trade_id: Any,
    exit_order_id: str | None,
    exit_time: Any,
    exit_price: Any,
    exit_reason: str | None,
) -> bool:
    with _maybe_conn(engine) as conn:
        if conn is None:
            logger.warning(
                "[WARN] DB_TRADE_CLOSE_FAILED trade_id=%s exit_order_id=%s err=%s",
                trade_id,
                exit_order_id or "",
                "disabled",
            )
            return False

        normalized_exit_time = normalize_ts(exit_time, field="exit_time") or datetime.now(timezone.utc)
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT status, qty, entry_price
                        FROM trades
                        WHERE trade_id=%(trade_id)s
                        """,
                        {"trade_id": trade_id},
                    )
                    row = cursor.fetchone()
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
                    cursor.execute(
                        """
                        UPDATE trades
                        SET exit_order_id=%(exit_order_id)s,
                            exit_time=%(exit_time)s,
                            exit_price=%(exit_price)s,
                            realized_pnl=%(realized_pnl)s,
                            exit_reason=%(exit_reason)s,
                            status='CLOSED',
                            updated_at=now()
                        WHERE trade_id=%(trade_id)s
                        """,
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
    engine: Optional[PGConnection],
    trade_id: Any,
    *,
    exit_order_id: str | None = None,
    exit_time: Any | None = None,
    exit_price: Any | None = None,
    exit_reason: str | None = None,
) -> bool:
    with _maybe_conn(engine) as conn:
        if conn is None:
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
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT status, qty, entry_price, exit_price, realized_pnl
                        FROM trades
                        WHERE trade_id=%(trade_id)s
                        """,
                        {"trade_id": trade_id},
                    )
                    row = cursor.fetchone()
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

                    cursor.execute(
                        """
                        UPDATE trades
                        SET exit_order_id=COALESCE(%(exit_order_id)s, exit_order_id),
                            exit_time=COALESCE(%(exit_time)s, exit_time),
                            exit_price=COALESCE(%(exit_price)s, exit_price),
                            realized_pnl=%(realized_pnl)s,
                            exit_reason=COALESCE(%(exit_reason)s, exit_reason),
                            status='CLOSED',
                            updated_at=now()
                        WHERE trade_id=%(trade_id)s
                        """,
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


def decorate_trade_exit(
    engine: Optional[PGConnection],
    trade_id: Any,
    *,
    exit_order_id: str | None,
    exit_time: Any,
    exit_price: Any,
    exit_reason: str | None,
    realized_pnl: Any | None = None,
) -> bool:
    with _maybe_conn(engine) as conn:
        if conn is None:
            logger.warning(
                "[WARN] DB_TRADE_EXIT_DECORATE_FAILED trade_id=%s exit_order_id=%s err=%s",
                trade_id,
                exit_order_id or "",
                "disabled",
            )
            return False

        normalized_exit_time = normalize_ts(exit_time, field="exit_time") or datetime.now(timezone.utc)
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT qty, entry_price
                        FROM trades
                        WHERE trade_id=%(trade_id)s
                        """,
                        {"trade_id": trade_id},
                    )
                    row = cursor.fetchone()
                    if row is None:
                        logger.warning(
                            "[WARN] DB_TRADE_EXIT_DECORATE_FAILED trade_id=%s exit_order_id=%s err=%s",
                            trade_id,
                            exit_order_id or "",
                            "trade_not_found",
                        )
                        return False

                    qty_value = row[0]
                    entry_price_value = row[1]
                    computed_realized = realized_pnl
                    try:
                        if exit_price is not None and entry_price_value is not None:
                            computed_realized = float(exit_price) - float(entry_price_value)
                            if qty_value is not None:
                                computed_realized *= float(qty_value)
                    except Exception:
                        pass

                    cursor.execute(
                        """
                        UPDATE trades
                        SET exit_order_id=%(exit_order_id)s,
                            exit_time=%(exit_time)s,
                            exit_price=%(exit_price)s,
                            realized_pnl=%(realized_pnl)s,
                            exit_reason=%(exit_reason)s,
                            status='CLOSED',
                            updated_at=now()
                        WHERE trade_id=%(trade_id)s
                        """,
                        {
                            "exit_order_id": exit_order_id,
                            "exit_time": normalized_exit_time,
                            "exit_price": exit_price,
                            "realized_pnl": computed_realized,
                            "exit_reason": exit_reason,
                            "trade_id": trade_id,
                        },
                    )
            _log_write_result(True, "trades", 1)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "[WARN] DB_TRADE_EXIT_DECORATE_FAILED trade_id=%s exit_order_id=%s err=%s",
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
    conn = _conn_or_none()
    if conn is None:
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
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO trades (
                        symbol, qty, entry_order_id, entry_time, entry_price, status, updated_at
                    )
                    VALUES (
                        %(symbol)s, %(qty)s, %(entry_order_id)s, %(entry_time)s, %(entry_price)s, 'OPEN', now()
                    )
                    ON CONFLICT (entry_order_id) DO UPDATE SET
                        symbol=EXCLUDED.symbol,
                        qty=EXCLUDED.qty,
                        entry_time=EXCLUDED.entry_time,
                        entry_price=EXCLUDED.entry_price,
                        status='OPEN',
                        updated_at=now()
                    """,
                    payload,
                )
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
    finally:
        try:
            conn.close()
        except Exception:
            pass


def close_trade_on_sell_fill(
    symbol: str,
    exit_order_id: str | None,
    exit_time: Any,
    exit_price: Any,
    exit_reason: str | None = None,
) -> bool:
    conn = _conn_or_none()
    if conn is None:
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
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT trade_id, qty, entry_price
                    FROM trades
                    WHERE symbol=%(symbol)s AND status='OPEN'
                    ORDER BY entry_time DESC NULLS LAST, trade_id DESC
                    LIMIT 1
                    """,
                    {"symbol": symbol_value},
                )
                row = cursor.fetchone()
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
                cursor.execute(
                    """
                    UPDATE trades
                    SET exit_order_id=%(exit_order_id)s,
                        exit_time=%(exit_time)s,
                        exit_price=%(exit_price)s,
                        realized_pnl=%(realized_pnl)s,
                        exit_reason=%(exit_reason)s,
                        status='CLOSED',
                        updated_at=now()
                    WHERE trade_id=%(trade_id)s
                    """,
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
    finally:
        try:
            conn.close()
        except Exception:
            pass


def count_open_trades(symbol: str | None = None) -> int:
    conn = _conn_or_none()
    if conn is None:
        return 0
    try:
        query = """
            SELECT COUNT(*) FROM trades
            WHERE status='OPEN'
        """
        params: dict[str, Any] = {}
        if symbol:
            query += " AND symbol=%(symbol)s"
            params["symbol"] = symbol.upper()
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            result = cursor.fetchone()
            return int(result[0] if result else 0)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_TRADE_COUNT_FAILED symbol=%s err=%s", symbol or "", exc)
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass
