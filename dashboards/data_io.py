from __future__ import annotations

import datetime as dt
import json
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
    symbols_in = int(metrics.get("symbols_in") or 0)
    symbols_with_bars = int(metrics.get("symbols_with_bars") or 0)
    bars_rows_total = int(metrics.get("bars_rows_total") or 0)
    rows_premetrics = int(metrics.get("rows") or 0)
    last_run_utc = metrics.get("last_run_utc")

    top_path = DATA_DIR / "top_candidates.csv"
    latest_path = DATA_DIR / "latest_candidates.csv"
    top_df = _read_csv_safe(top_path)
    rows_final = int(top_df.shape[0]) if not top_df.empty else int(_read_csv_safe(latest_path).shape[0])

    if not last_run_utc:
        last_run_utc = _mtime_iso(top_path) or _mtime_iso(latest_path)

    log_path = LOGS_DIR / "pipeline.log"
    source = _parse_latest_source(log_path)
    pipeline_rc = _parse_latest_pipeline_end_rc(log_path)

    return {
        "symbols_in": symbols_in,
        "symbols_with_bars": symbols_with_bars,
        "bars_rows_total": bars_rows_total,
        "rows_premetrics": rows_premetrics,
        "rows_final": rows_final,
        "last_run_utc": last_run_utc,
        "source": source,
        "pipeline_rc": pipeline_rc,
    }


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


def diagnostics() -> Dict[str, Any]:
    """Return a simple diagnostic payload used in dashboards."""

    health = screener_health()
    table_df, updated, source_file = screener_table()
    return {
        "health": health,
        "table_rows": int(table_df.shape[0]),
        "table_cols": list(table_df.columns),
        "table_updated": updated,
        "table_source": source_file,
    }
