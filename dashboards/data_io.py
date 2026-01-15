from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from scripts import db

BASE_DIR = Path(
    os.environ.get("JBRAVO_HOME", Path(__file__).resolve().parents[1])
).expanduser()
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
    return (
        dt.datetime.utcfromtimestamp(ts)
        .replace(tzinfo=dt.timezone.utc)
        .isoformat()
    )


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


def load_connection_health(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    base = Path(base_dir) if base_dir else DATA_DIR
    primary = base / "connection_health.json"
    fallback = base / "connectivity.json"

    payload = _read_json_safe(primary)
    if not payload:
        payload = _read_json_safe(fallback)

    timestamp = payload.get("timestamp") or payload.get("last_run_utc") or payload.get("last_run")
    if timestamp is not None:
        timestamp = str(timestamp)

    normalized = {
        "trading_ok": _coerce_bool(payload.get("trading_ok")),
        "trading_status": _coerce_int(payload.get("trading_status")),
        "data_ok": _coerce_bool(payload.get("data_ok")),
        "data_status": _coerce_int(payload.get("data_status")),
        "feed": _normalize_feed(payload.get("feed")),
        "timestamp": timestamp,
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
        symbols_with_bars_fetch = _coerce_int(payload.get("symbols_with_required_bars")) or _coerce_int(
            payload.get("symbols_with_bars")
        )
    attempted_fetch = symbols_attempted_fetch or symbols_with_bars_fetch
    symbols_with_required_bars = _coerce_int(
        payload.get("symbols_with_required_bars")
    ) or symbols_with_bars_fetch
    if symbols_with_any_bars is None:
        symbols_with_any_bars = symbols_with_required_bars
    bars_rows_total_fetch = _coerce_int(
        payload.get("bars_rows_total_fetch")
    ) or _coerce_int(payload.get("bars_rows_total"))

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
        "run_type": _run_type_hint(),
        "buying_power": connection.get("buying_power"),
    }

    # Legacy aliases for existing callers
    snapshot["symbols_with_bars"] = snapshot["symbols_with_required_bars"] or snapshot["symbols_with_bars_fetch"]
    snapshot["bars_rows_total"] = snapshot["bars_rows_total_fetch"]
    snapshot["rows"] = snapshot["rows_premetrics"]
    snapshot["source"] = snapshot.get("latest_source")

    snapshot.update(_update_coverage_history(snapshot))
    return snapshot


def screener_table() -> Tuple[pd.DataFrame, str, str]:
    """Return (DataFrame, iso timestamp, file source) for the screener table."""

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

    db_df, run_label = _db_candidates()
    if not db_df.empty or run_label is not None:
        df = db_df
        updated = run_label or ""
        source_file = "screener_candidates (db)"
    else:
        logger.info("DASH_DB_READ_FALLBACK reason=csv_fallback")
        top_path = DATA_DIR / "top_candidates.csv"
        latest_path = DATA_DIR / "latest_candidates.csv"

        df = _read_csv_safe(top_path)
        updated = _mtime_iso(top_path)
        source_file = "top_candidates.csv"

        if df.empty:
            df = _read_csv_safe(latest_path)
            source_file = "latest_candidates.csv"
            if not updated:
                updated = _mtime_iso(latest_path)

    df = df.copy()
    for column in ("score", "win_rate", "net_pnl", "close", "adv20", "atrp"):
        if column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            except Exception:
                continue

    return df, (updated or ""), source_file


def metrics_summary_snapshot() -> Dict[str, Any]:
    """Return the latest metrics summary row from ``data/metrics_summary.csv``."""

    if db.db_enabled():
        conn = db.get_db_conn()
        if conn is not None:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT MAX(run_date) FROM metrics_daily")
                    row = cursor.fetchone()
                    latest_run_date = row[0] if row else None
                if latest_run_date:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT run_date, total_trades, win_rate, net_pnl, expectancy,
                                   profit_factor, max_drawdown, sharpe, sortino
                            FROM metrics_daily
                            WHERE run_date = %(run_date)s
                            LIMIT 1
                            """,
                            {"run_date": latest_run_date},
                        )
                        row = cursor.fetchone()
                        if row:
                            logger.info("DASH_DB_READ_OK table=metrics_daily run_date=%s", latest_run_date)
                            return {
                                "profit_factor": row[5],
                                "expectancy": row[4],
                                "win_rate": row[2],
                                "net_pnl": row[3],
                                "max_drawdown": row[6],
                                "sharpe": row[7],
                                "sortino": row[8],
                                "last_run_utc": str(row[0]),
                            }
                logger.info("DASH_DB_READ_OK table=metrics_daily rows=0")
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.info("DASH_DB_READ_FALLBACK reason=%s", exc)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        else:
            logger.info("DASH_DB_READ_FALLBACK reason=db_disabled")
    else:
        logger.info("DASH_DB_READ_FALLBACK reason=db_disabled")

    path = DATA_DIR / "metrics_summary.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    if df.empty:
        return {}
    row = df.tail(1).to_dict("records")[0]
    fields = [
        "profit_factor",
        "expectancy",
        "win_rate",
        "net_pnl",
        "max_drawdown",
        "sharpe",
        "sortino",
        "last_run_utc",
    ]
    return {field: row.get(field) for field in fields}


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
