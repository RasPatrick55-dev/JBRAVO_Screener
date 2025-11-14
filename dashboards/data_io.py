from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

BASE_DIR = Path(
    os.environ.get("JBRAVO_HOME", Path(__file__).resolve().parents[1])
).expanduser()
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"


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


def _safe_csv_rows(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        df = pd.read_csv(path)
    except Exception:
        return 0
    return int(df.shape[0])


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


def _read_health_json() -> Dict[str, Any]:
    return _read_json_safe(DATA_DIR / "connection_health.json")


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


def _parse_health_from_logs(log_path: Path) -> Dict[str, Any]:
    try:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    pattern = re.compile(
        r"trading_ok=(True|False).*data_ok=(True|False).*trading_status=(\d+).*data_status=(\d+)"
    )
    for raw in reversed(tail.splitlines()[-800:]):
        if "HEALTH" not in raw:
            continue
        match = pattern.search(raw)
        if match:
            return {
                "trading_ok": match.group(1) == "True",
                "data_ok": match.group(2) == "True",
                "trading_status": int(match.group(3)),
                "data_status": int(match.group(4)),
            }
    return {}


def _parse_latest_pipeline_end_rc(log_path: Path) -> Optional[int]:
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in reversed(lines[-400:]):
        if "PIPELINE_END" not in line:
            continue
        match = re.search(r"PIPELINE_END rc=(\d+)", line)
        if match:
            return int(match.group(1))
        break
    return None


def _parse_latest_source(log_path: Path) -> str:
    """Return 'screener', 'fallback', or 'unknown' based on log hints."""

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return "unknown"

    for line in reversed(lines[-800:]):
        if "PIPELINE_SUMMARY" in line and "source=" in line:
            match = re.search(r"source=([a-zA-Z0-9_]+)", line)
            if match:
                return match.group(1)
    for line in reversed(lines[-800:]):
        if "FALLBACK_CHECK" in line:
            return "fallback"
    return "screener"


def screener_health() -> Dict[str, Any]:
    """Return a resilient snapshot for the Screener Health view."""

    metrics = _read_json_safe(DATA_DIR / "screener_metrics.json")
    top_path = DATA_DIR / "top_candidates.csv"
    log_path = LOGS_DIR / "pipeline.log"

    symbols_in = _coerce_int(metrics.get("symbols_in")) or 0
    with_bars_fetch = (
        _coerce_int(metrics.get("symbols_with_bars_fetch"))
        or _coerce_int(metrics.get("symbols_with_bars_raw"))
        or _coerce_int(metrics.get("symbols_with_bars"))
        or 0
    )
    with_bars_post = _coerce_int(metrics.get("symbols_with_bars_post"))
    bars_rows_fetch = (
        _coerce_int(metrics.get("bars_rows_total_fetch"))
        or _coerce_int(metrics.get("bars_rows_total"))
        or 0
    )
    bars_rows_post = _coerce_int(metrics.get("bars_rows_total_post"))
    rows_metric = _coerce_int(metrics.get("rows")) or 0
    top_rows = _safe_csv_rows(top_path)
    rows_final = top_rows or rows_metric
    rows_premetrics = rows_metric or rows_final
    last_run_utc = metrics.get("last_run_utc") or _mtime_iso(top_path)
    if not last_run_utc:
        last_run_utc = metrics.get("last_run")

    source = (
        metrics.get("latest_source")
        or metrics.get("source")
        or _parse_latest_source(log_path)
    )
    pipeline_rc = _parse_latest_pipeline_end_rc(log_path)
    if pipeline_rc is None:
        pipeline_rc = _coerce_int(metrics.get("pipeline_rc") or metrics.get("rc"))
    conn = _read_health_json()
    if not conn:
        conn = _parse_health_from_logs(log_path)
    freshness = _freshness(last_run_utc)
    run_type = _run_type_hint()
    feed = (conn.get("feed") or os.getenv("ALPACA_DATA_FEED") or "iex").lower()

    snapshot = {
        "symbols_in": symbols_in,
        "symbols_with_bars_fetch": int(with_bars_fetch),
        "symbols_with_bars_post": int(with_bars_post) if with_bars_post is not None else None,
        "bars_rows_total_fetch": int(bars_rows_fetch),
        "bars_rows_total_post": int(bars_rows_post) if bars_rows_post is not None else None,
        "rows_premetrics": int(rows_premetrics),
        "rows_final": int(rows_final),
        "last_run_utc": last_run_utc,
        "source": source,
        "pipeline_rc": pipeline_rc,
        "trading_ok": bool(conn.get("trading_ok")),
        "data_ok": bool(conn.get("data_ok")),
        "trading_status": conn.get("trading_status"),
        "data_status": conn.get("data_status"),
        "feed": feed,
        "freshness": freshness,
        "run_type": run_type,
    }
    # Legacy aliases for existing callers
    snapshot["symbols_with_bars"] = snapshot["symbols_with_bars_fetch"]
    snapshot["bars_rows_total"] = snapshot["bars_rows_total_fetch"]
    snapshot["rows"] = snapshot["rows_premetrics"]
    return snapshot


def screener_table() -> Tuple[pd.DataFrame, str, str]:
    """Return (DataFrame, iso timestamp, file source) for the screener table."""

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
