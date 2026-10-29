from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from scripts import db

BASE_DIR = Path(os.environ.get("JBRAVO_HOME", Path(__file__).resolve().parents[1])).expanduser()
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

logger = logging.getLogger(__name__)


def _db_conn() -> Optional[Any]:
    if not db.db_enabled():
        logger.warning("DASH_DB_READ_FALLBACK table=trades err=db_disabled")
        return None
    conn = db.get_db_conn()
    if conn is None:
        logger.warning("DASH_DB_READ_FALLBACK table=trades err=db_conn_none")
    return conn


def _fetch_dataframe(conn, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    with conn.cursor() as cursor:
        cursor.execute(query, params or {})
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def load_trades_db(limit: int = 500) -> pd.DataFrame:
    """Return trades from the database ordered by lifecycle recency."""

    conn = _db_conn()
    if conn is None:
        return pd.DataFrame()

    try:
        df = _fetch_dataframe(
            conn,
            """
            select trade_id, symbol, qty, status, entry_time, entry_price,
                   exit_time, exit_price, realized_pnl, exit_reason, created_at, updated_at
            from trades
            order by coalesce(exit_time, entry_time) desc
            limit %(limit)s
            """,
            {"limit": int(limit)},
        )
        logger.info("DASH_DB_READ_OK table=trades rows=%d", len(df))
        return df
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("DASH_DB_READ_FALLBACK table=trades err=%s", exc)
        return pd.DataFrame()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_open_trades_db() -> pd.DataFrame:
    """Return OPEN trades ordered by entry_time desc."""

    conn = _db_conn()
    if conn is None:
        return pd.DataFrame()

    try:
        df = _fetch_dataframe(
            conn,
            """
            select trade_id, symbol, qty, status, entry_time, entry_price,
                   exit_time, exit_price, realized_pnl, exit_reason, created_at, updated_at
            from trades
            where status = 'OPEN'
            order by entry_time desc
            """,
        )
        logger.info("DASH_DB_READ_OK table=trades rows=%d", len(df))
        return df
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("DASH_DB_READ_FALLBACK table=trades err=%s", exc)
        return pd.DataFrame()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _read_json_safe(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in {"1", "true", "yes", "y", "ok", "up", "healthy"}:
            return True
        if text in {"0", "false", "no", "n", "fail", "down", "bad"}:
            return False
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            first = next(handle, None)
            if first is None:
                return 0
            return max(0, sum(1 for _ in handle))
    except FileNotFoundError:
        return 0
    except Exception:
        return 0


def _parse_pipeline_log(base_dir: Path) -> Dict[str, Any]:
    log_path = Path(base_dir) / "logs" / "pipeline.log"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

    result: Dict[str, Any] = {}
    for line in reversed(lines):
        if "PIPELINE_END" in line and "rc=" in line and "pipeline_rc" not in result:
            for part in line.split():
                if part.startswith("rc="):
                    result["pipeline_rc"] = _coerce_int(part.split("=", 1)[1])
                    break
        if "PIPELINE_SUMMARY" in line and "source=" in line and "latest_source" not in result:
            for part in line.split():
                if part.startswith("source="):
                    result["latest_source"] = part.split("=", 1)[1]
                    break
        if "latest_source" in result and "pipeline_rc" in result:
            break

    return result


def _mtime_iso(path: Path) -> Optional[str]:
    try:
        ts = path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return None
    return dt.datetime.utcfromtimestamp(ts).replace(tzinfo=dt.timezone.utc).isoformat()


def _health_history_path() -> Path:
    return REPORTS_DIR / "health_history.json"


def _load_health_history() -> list[dict[str, Any]]:
    path = _health_history_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except Exception:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    return []


def _update_coverage_history(snapshot: Dict[str, Any]) -> Dict[str, Optional[int]]:
    history = _load_health_history()
    latest_entry = history[-1] if history else None

    coverage_value = _coerce_int(snapshot.get("symbols_with_any_bars"))
    if coverage_value is None:
        coverage_value = _coerce_int(snapshot.get("symbols_with_bars_fetch"))
    timestamp = snapshot.get("last_run_utc")
    if not timestamp:
        timestamp = _mtime_iso(DATA_DIR / "top_candidates.csv")
    if not timestamp:
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

    if latest_entry and latest_entry.get("timestamp") == timestamp:
        latest_cov = latest_entry.get("symbols_with_bars_fetch")
        if latest_cov == coverage_value:
            return {"coverage_drift": latest_entry.get("coverage_drift")}

    previous_value: Optional[int] = None
    if latest_entry is not None:
        prev_cov = latest_entry.get("symbols_with_bars_fetch")
        if isinstance(prev_cov, int):
            previous_value = prev_cov

    drift: Optional[int] = None
    if coverage_value is not None and previous_value is not None:
        drift = coverage_value - previous_value

    entry = {
        "timestamp": timestamp,
        "symbols_with_bars_fetch": coverage_value,
        "coverage_drift": drift,
    }
    history.append(entry)
    history = history[-7:]

    path = _health_history_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except Exception:
        pass

    return {"coverage_drift": drift}


def _freshness(last_run_utc: Optional[str]) -> Dict[str, Any]:
    age_seconds: Optional[int] = None
    level = "gray"
    if last_run_utc:
        try:
            parsed = dt.datetime.fromisoformat(last_run_utc.replace("Z", "+00:00"))
            delta = dt.datetime.now(dt.timezone.utc) - parsed
            age_seconds = int(delta.total_seconds())
            if age_seconds < 2 * 3600:
                level = "green"
            elif age_seconds < 12 * 3600:
                level = "amber"
            else:
                level = "gray"
        except Exception:
            age_seconds = None
    return {"age_seconds": age_seconds, "freshness_level": level}


def _run_type_hint() -> str:
    marker = DATA_DIR / "last_premarket_run.json"
    try:
        mtime = marker.stat().st_mtime
    except (FileNotFoundError, OSError):
        return "nightly"
    marker_dt = dt.datetime.fromtimestamp(mtime, tz=dt.timezone.utc)
    age_seconds = (dt.datetime.now(dt.timezone.utc) - marker_dt).total_seconds()
    return "pre-market" if age_seconds <= 12 * 3600 else "nightly"


def _connection_timestamp(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Normalize producer aliases without borrowing freshness or inventing a time."""
    stamps = []
    for key in ("timestamp", "ts_utc", "last_run_utc", "last_run"):
        value = payload.get(key)
        if value is None:
            continue
        if (not isinstance(value, str) or len(value) > 128
                or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})", value) is None):
            return None, "timestamp_invalid"
        # fromisoformat normalizes overflowing offset minutes (e.g. +00:60).
        # Reject their lexical form before parsing so it cannot create freshness.
        if value[-1] != "Z" and (int(value[-5:-3]) > 23 or int(value[-2:]) > 59):
            return None, "timestamp_invalid"
        try:
            stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
            if stamp.utcoffset() is None:
                return None, "timestamp_invalid"
            stamps.append(stamp.astimezone(dt.timezone.utc))
        except (ValueError, TypeError, OverflowError):
            return None, "timestamp_invalid"
    if not stamps:
        return None, None
    if any(stamp != stamps[0] for stamp in stamps[1:]):
        return None, "timestamp_conflict"
    return stamps[0].isoformat(), None


def load_connection_health(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    base = Path(base_dir) if base_dir else DATA_DIR
    primary = base / "connection_health.json"
    fallback = base / "connectivity.json"

    payload = _read_json_safe(primary)
    source = "connection_health.json" if payload else None
    if not payload:
        payload = _read_json_safe(fallback)
        source = "connectivity.json" if payload else None

    timestamp, timestamp_error = _connection_timestamp(payload)

    normalized = {
        "source": source,
        "trading_ok": _coerce_bool(payload.get("trading_ok")),
        "trading_status": _coerce_int(payload.get("trading_status")),
        "data_ok": _coerce_bool(payload.get("data_ok")),
        "data_status": _coerce_int(payload.get("data_status")),
        "feed": _normalize_feed(payload.get("feed")),
        "timestamp": timestamp,
        "timestamp_error": timestamp_error,
        "buying_power": _coerce_float(payload.get("buying_power")),
    }

    for key in (
        "trading_ok",
        "trading_status",
        "data_ok",
        "data_status",
        "feed",
        "timestamp",
        "buying_power",
    ):
        normalized.setdefault(key, None)

    return normalized


def load_screener_metrics(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    base = Path(base_dir) if base_dir else DATA_DIR
    metrics_path = base / "screener_metrics.json"
    payload = _read_json_safe(metrics_path)

    symbols_with_any_bars = _coerce_int(
        payload.get("symbols_with_any_bars") or payload.get("symbols_with_bars_any")
    )
    symbols_with_bars_fetch = _coerce_int(payload.get("symbols_with_bars_fetch"))
    symbols_attempted_fetch = _coerce_int(payload.get("symbols_attempted_fetch"))
    if symbols_with_any_bars is not None:
        symbols_with_bars_fetch = symbols_with_any_bars
    elif symbols_with_bars_fetch is None:
        symbols_with_bars_fetch = _coerce_int(
            payload.get("symbols_with_required_bars")
        ) or _coerce_int(payload.get("symbols_with_bars"))
    attempted_fetch = symbols_attempted_fetch or symbols_with_bars_fetch
    symbols_with_required_bars = (
        _coerce_int(payload.get("symbols_with_required_bars")) or symbols_with_bars_fetch
    )
    if symbols_with_any_bars is None:
        symbols_with_any_bars = symbols_with_required_bars
    bars_rows_total_fetch = _coerce_int(payload.get("bars_rows_total_fetch")) or _coerce_int(
        payload.get("bars_rows_total")
    )

    rows_final = _coerce_int(payload.get("rows_final"))
    if rows_final is None:
        rows_final = _count_csv_rows(base / "top_candidates.csv")
    else:
        rows_final = max(0, rows_final)

    latest_source = payload.get("latest_source") or payload.get("source")

    pipeline_rc_value = payload.get("pipeline_rc")
    if pipeline_rc_value is None and "pipeline_rc" not in payload:
        pipeline_rc_value = payload.get("rc")

    normalized: Dict[str, Any] = {
        "last_run_utc": payload.get("last_run_utc") or payload.get("last_run"),
        "symbols_in": _coerce_int(payload.get("symbols_in")),
        "symbols_with_bars_fetch": symbols_with_bars_fetch,
        "symbols_attempted_fetch": attempted_fetch,
        "symbols_with_any_bars": symbols_with_any_bars,
        "symbols_with_required_bars": symbols_with_required_bars,
        "symbols_with_bars_post": _coerce_int(payload.get("symbols_with_bars_post")),
        "bars_rows_total_fetch": bars_rows_total_fetch,
        "bars_rows_total_post": _coerce_int(payload.get("bars_rows_total_post")),
        "rows_premetrics": _coerce_int(payload.get("rows_premetrics"))
        or _coerce_int(payload.get("rows")),
        "rows_final": rows_final,
        "latest_source": latest_source,
        "pipeline_rc": _coerce_int(pipeline_rc_value),
    }

    if 'provisional_context' in payload:
        normalized.update(provisional_context=payload['provisional_context'], qualified_daily=False,
                          entry_ready=False, run_type='PROVISIONAL functional test: not qualified daily')
    inferred = _parse_pipeline_log(base.parent)
    for key in ("latest_source", "pipeline_rc"):
        if normalized.get(key) in (None, "") and key in inferred:
            normalized[key] = inferred[key]

    return normalized


def _normalize_feed(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip().lower()
    if text in {"iex", "sip"}:
        return text
    return None


def connection_check_snapshot(connection: Dict[str, Any], *, now=None) -> Dict[str, Any]:
    """Describe an existing stored check; never perform or refresh a broker call."""
    source = connection.get("source")
    result = {"source": source if source in {"connection_health.json", "connectivity.json"} else None,
              "status": "unknown", "reason": "source_missing", "checked_at": None,
              "trading_ok": connection.get("trading_ok") if type(connection.get("trading_ok")) is bool else None,
              "freshness": {"age_seconds": None, "freshness_level": "gray"}}
    if result["source"] is None:
        return result
    timestamp_error = connection.get("timestamp_error")
    if timestamp_error in {"timestamp_invalid", "timestamp_conflict"}:
        result["reason"] = timestamp_error
        return result
    text = connection.get("timestamp")
    if not isinstance(text, str) or not text or len(text) > 128:
        result["reason"] = "timestamp_missing"
        return result
    try:
        normalized, error = _connection_timestamp({"timestamp": text})
        if error is not None or normalized is None:
            raise ValueError("invalid timestamp")
        stamp = dt.datetime.fromisoformat(normalized)
        if stamp.utcoffset() is None:
            raise ValueError("naive timestamp")
        stamp = stamp.astimezone(dt.timezone.utc)
        age = ((now or dt.datetime.now(dt.timezone.utc)) - stamp).total_seconds()
    except (ValueError, TypeError, OverflowError):
        result["reason"] = "timestamp_invalid"
        return result
    result["checked_at"] = stamp.isoformat()
    if age < 0:
        result["reason"] = "timestamp_future"
        return result
    result["freshness"] = {"age_seconds": int(age),
        "freshness_level": "green" if age < 7200 else "amber" if age < 43200 else "gray"}
    if result["trading_ok"] is None:
        result["reason"] = "result_unknown"
    else:
        result.update(status="fresh" if age < 7200 else "stale", reason="fresh" if age < 7200 else "stale")
    return result


def screener_health(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Return a resilient snapshot for the Screener Health view."""

    base = Path(base_dir) if base_dir else DATA_DIR
    metrics = load_screener_metrics(base)
    connection = load_connection_health(base)

    # Prefer artifact timestamps; fall back to file mtime for awareness
    last_run_utc = metrics.get("last_run_utc") or _mtime_iso(base / "top_candidates.csv")

    feed = connection.get("feed") or _normalize_feed(os.getenv("ALPACA_DATA_FEED"))

    rows_premetrics = metrics.get("rows_premetrics") or metrics.get("rows_final")
    if rows_premetrics is None:
        rows_premetrics = 0
    rows_final = metrics.get("rows_final") or 0
    if rows_premetrics < rows_final:
        rows_premetrics = rows_final

    snapshot: Dict[str, Any] = {
        "last_run_utc": last_run_utc,
        "symbols_in": metrics.get("symbols_in"),
        "symbols_with_bars_fetch": metrics.get("symbols_with_bars_fetch"),
        "symbols_with_any_bars": metrics.get("symbols_with_any_bars"),
        "symbols_with_required_bars": metrics.get("symbols_with_required_bars"),
        "bars_rows_total_fetch": metrics.get("bars_rows_total_fetch"),
        "symbols_with_bars_post": metrics.get("symbols_with_bars_post"),
        "bars_rows_total_post": metrics.get("bars_rows_total_post"),
        "rows_final": rows_final,
        "rows_premetrics": rows_premetrics,
        "trading_ok": connection.get("trading_ok"),
        "trading_status": connection.get("trading_status"),
        "data_ok": connection.get("data_ok"),
        "data_status": connection.get("data_status"),
        "feed": feed,
        "latest_source": metrics.get("latest_source"),
        "pipeline_rc": metrics.get("pipeline_rc"),
        "freshness": _freshness(last_run_utc),
        "connection_check": connection_check_snapshot(connection),
        "run_type": _run_type_hint(),
        "buying_power": connection.get("buying_power"),
    }

    if 'provisional_context' in metrics:
        snapshot.update(provisional_context=metrics['provisional_context'], qualified_daily=False,
                        entry_ready=False, run_type='PROVISIONAL functional test: not qualified daily')
    # Legacy aliases for existing callers
    snapshot["symbols_with_bars"] = (
        snapshot["symbols_with_required_bars"] or snapshot["symbols_with_bars_fetch"]
    )
    snapshot["bars_rows_total"] = snapshot["bars_rows_total_fetch"]
    snapshot["rows"] = snapshot["rows_premetrics"]
    snapshot["source"] = snapshot.get("latest_source")

    snapshot.update(_update_coverage_history(snapshot))
    return snapshot


def screener_table() -> Tuple[pd.DataFrame, str, str]:
    """Return (DataFrame, iso timestamp, file source) for the screener table."""

    from scripts.strategy_pipeline import enabled as strategy_enabled
    if strategy_enabled():
        from scripts.strategy_dashboard import snapshot
        current = snapshot()
        return (pd.DataFrame(current['rows']), current['run_ts_utc'] or '',
                current['source_detail'])

    def _db_candidates() -> tuple[pd.DataFrame, Optional[str]]:
        if not db.db_enabled():
            return pd.DataFrame(), None
        conn = db.get_db_conn()
        if conn is None:
            return pd.DataFrame(), None
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT MAX(run_date) FROM screener_candidates")
                row = cursor.fetchone()
                latest_run_date = row[0] if row else None
            if not latest_run_date:
                logger.info("DASH_DB_READ_OK table=screener_candidates rows=0")
                return pd.DataFrame(), None
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        run_date, timestamp, symbol, score, exchange, close, volume,
                        universe_count, score_breakdown, entry_price, adv20, atrp, source
                    FROM screener_candidates
                    WHERE run_date = %(run_date)s
                    ORDER BY score DESC NULLS LAST
                    """,
                    {"run_date": latest_run_date},
                )
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
            df_db = pd.DataFrame(rows, columns=columns)
            run_label = str(latest_run_date)
            logger.info(
                "DASH_DB_READ_OK table=screener_candidates rows=%s run_date=%s",
                len(df_db),
                run_label,
            )
            return df_db, run_label
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.info("DASH_DB_READ_FALLBACK reason=%s", exc)
            return pd.DataFrame(), None
        finally:
            try:
                conn.close()
            except Exception:
                pass

    df, run_label = _db_candidates()
    updated = run_label or ""
    source_file = "screener_candidates (postgres)" if run_label is not None else "postgres unavailable or empty; no file fallback"

    df = df.copy()
    for column in ("score", "win_rate", "net_pnl", "close", "adv20", "atrp"):
        if column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            except Exception:
                continue

    if 'source' in df.columns and df['source'].astype('string').str.startswith('PROVISIONAL:').fillna(False).any():
        source_file = 'PROVISIONAL manual test - NOT qualified daily: ' + source_file
        df['qualified_daily'] = False
    return df, (updated or ""), source_file


def paper_gross_summary_snapshot() -> Dict[str, Any]:
    """Separate gross namespace; every read validates current DB/code revisions."""
    try:
        value = db.validated_paper_gross_report()
        report = value['report']
        return {'validation_status': value['validation_status'],
            'pnl_basis': report['pnl_basis'], 'gross_pnl': report['gross_pnl'],
            'total_trades': report['total_trades'], 'rows': report['rows'],
            'generated_at': value['generated_at'], 'validated_at': value['validated_at'],
            'cache_revision': value['cache_revision'], 'evidence': report['evidence'],
            'fees': None, 'fees_status': 'UNKNOWN', 'net_pnl': None, 'net_status': 'UNKNOWN',
            'net_eligible': False, 'accounting_complete': False,
            'activity_ingestion_complete': False, 'recovery_complete': False}
    except Exception as exc:
        logger.warning('PAPER_GROSS_REPORT_HELD error_type=%s', type(exc).__name__)
        return {'validation_status': 'HELD_PAPER_GROSS', 'gross_pnl': None,
            'total_trades': None, 'rows': [], 'generated_at': None, 'validated_at': None,
            'cache_revision': None, 'fees': None, 'fees_status': 'UNKNOWN',
            'net_pnl': None, 'net_status': 'UNKNOWN', 'net_eligible': False,
            'accounting_complete': False, 'activity_ingestion_complete': False,
            'recovery_complete': False}


def metrics_summary_snapshot() -> Dict[str, Any]:
    """Validated scoped summary, or explicit held values; never a legacy CSV fallback."""
    try:
        return db.validated_metrics_summary()
    except Exception as exc:
        logger.warning('VALIDATED_PERFORMANCE_HELD error_type=%s', type(exc).__name__)
        return dict.fromkeys(('profit_factor','expectancy','win_rate','net_pnl',
            'max_drawdown','sharpe','sortino','total_trades','last_run_utc'), None) | {
            'validation_status':'HELD_UNVALIDATED_PERFORMANCE', 'recovery_complete':False,
            'accounting_complete':False, 'activity_ingestion_complete':False,
            **db.performance_hold_visibility()}


def diagnostics() -> Dict[str, Any]:
    """Return a simple diagnostic payload used in dashboards."""

    health = screener_health()
    table_df, updated, source_file = screener_table()
    diagnostics_payload = {
        "health": health,
        "table_rows": int(table_df.shape[0]),
        "table_cols": list(table_df.columns),
        "table_updated": updated,
        "table_source": source_file,
    }
    diagnostics_payload.update(
        {
            "symbols_in": health.get("symbols_in"),
            "symbols_with_bars_fetch": health.get("symbols_with_bars_fetch"),
            "bars_rows_total_fetch": health.get("bars_rows_total_fetch"),
            "rows_final": health.get("rows_final"),
            "trading_ok": health.get("trading_ok"),
            "data_ok": health.get("data_ok"),
        }
    )
    return diagnostics_payload


def health_payload_for_api() -> Dict[str, Any]:
    """Return the canonical health payload used by ``/api/health``."""

    return screener_health()
