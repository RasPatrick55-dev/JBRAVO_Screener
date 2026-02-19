# Complete Integrated Dashboard (dashboard_app.py)

import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.dash_table.Format import Format, Scheme
from datetime import date, datetime, timezone, timedelta
import subprocess
import json
import math
from decimal import Decimal
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import logging
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import time
import inspect
import pytz
import re
import urllib.error
import urllib.request
from urllib.parse import quote, urlencode
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from flask import Response, abort, jsonify, send_from_directory, request
from plotly.subplots import make_subplots
from scripts import db
from scripts.utils.env import load_env as _load_runtime_env

logger = logging.getLogger(__name__)

from dashboards.screener_health import build_layout as build_screener_health
from dashboards.screener_health import register_callbacks as register_screener_health
from dashboards.screener_health import load_kpis as _load_screener_health_kpis
from dashboards.data_io import (
    screener_health as load_screener_health,
    screener_table,
    metrics_summary_snapshot,
    load_trades_db,
    load_open_trades_db,
)
from dashboards.utils import coerce_kpi_types, parse_pipeline_summary
from dashboards.overview import overview_layout, render_timeline_table
from dashboards.pipeline_tab import pipeline_layout
from dashboards.ml_tab import build_predictions_table, ml_layout
from scripts.run_pipeline import write_complete_screener_metrics
from scripts.trade_performance import (
    CACHE_PATH as TRADE_PERFORMANCE_CACHE_PATH,
    evaluate_sold_too_soon_flags,
    summarize_by_window,
)
from scripts.indicators import macd as _macd, rsi as _rsi, adx as _adx, obv as _obv
from dashboards.db_client import db_query_df

# Base directory of the project (parent of this file)
DEFAULT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_HOME = os.environ.get("JBRAVO_HOME")
if not ENV_HOME or not os.path.exists(ENV_HOME):
    ENV_HOME = DEFAULT_HOME
    os.environ["JBRAVO_HOME"] = ENV_HOME
BASE_DIR = ENV_HOME

DATA_EXPORT_ALLOWLIST: set[str] = set()

LOG_EXPORT_ALLOWLIST = {
    "pipeline.log",
    "execute_trades.log",
    "monitor.log",
}

_DAILY_BARS_CACHE: dict[str, Any] = {"mtime": None, "df": None}
TASK_LOG_PATH = os.environ.get("PIPELINE_TASK_LOG_PATH") or os.path.join(
    BASE_DIR, "logs", "pipeline_task.log"
)
EXECUTE_TASK_LOG_PATH = os.environ.get("EXECUTE_TASK_LOG_PATH") or os.path.join(
    BASE_DIR, "logs", "execute_task.log"
)


def _pythonanywhere_token() -> Optional[str]:
    return os.environ.get("PYTHONANYWHERE_API_TOKEN") or os.environ.get("API_TOKEN")


def _pythonanywhere_username() -> Optional[str]:
    return os.environ.get("PYTHONANYWHERE_USERNAME") or os.environ.get("PYTHONANYWHERE_USER")


def _pythonanywhere_api_base() -> str:
    return (
        os.environ.get("PYTHONANYWHERE_API_BASE_URL")
        or "https://www.pythonanywhere.com/api/v0/user"
    ).rstrip("/")


def _pythonanywhere_api_get_json(path: str) -> Optional[Any]:
    token = _pythonanywhere_token()
    username = _pythonanywhere_username()
    if not token or not username:
        return None
    cleaned = path.lstrip("/")
    url = f"{_pythonanywhere_api_base()}/{username}/{cleaned}"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        return None


def _pythonanywhere_usage_snapshot() -> Optional[dict[str, Any]]:
    remote_url = os.environ.get("PYTHONANYWHERE_USAGE_URL")
    remote_path = os.environ.get("PYTHONANYWHERE_USAGE_PATH")
    if not remote_url and not remote_path:
        username = _pythonanywhere_username()
        if username:
            remote_path = f"/home/{username}/JBRAVO_Screener/data/pythonanywhere_usage.json"
    if not remote_url and not remote_path:
        return None
    payload = _fetch_pythonanywhere_file(remote_url, remote_path)
    if not payload:
        return None
    text, source = payload
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    parsed["_source"] = source
    return parsed


def _percent_used(used: Optional[float], limit: Optional[float]) -> Optional[float]:
    if used is None or limit is None or limit <= 0:
        return None
    return max(0.0, min(100.0, (used / limit) * 100))


def _pythonanywhere_cpu_usage() -> Optional[dict[str, Any]]:
    payload = _pythonanywhere_api_get_json("cpu/")
    cpu_seconds = None
    cpu_limit = None
    next_reset = None
    if isinstance(payload, dict):
        cpu_seconds = _to_float(payload.get("daily_cpu_total_usage_seconds"))
        cpu_limit = _to_float(payload.get("daily_cpu_limit_seconds"))
        if cpu_seconds is None:
            cpu_seconds = _to_float(payload.get("cpu_seconds"))
        if cpu_limit is None:
            cpu_limit = _to_float(payload.get("cpu_limit"))
        next_reset = payload.get("next_reset_time")

    percent = _percent_used(cpu_seconds, cpu_limit)
    if percent is not None:
        return {
            "label": "CPU Usage",
            "value": math.floor(percent),
            "used": cpu_seconds,
            "limit": cpu_limit,
            "unit": "sec",
            "next_reset_time": next_reset,
            "source": "pythonanywhere",
        }

    snapshot = _pythonanywhere_usage_snapshot()
    if snapshot:
        cpu = snapshot.get("cpu") or snapshot.get("cpu_usage")
        if isinstance(cpu, dict):
            cpu_seconds = _to_float(cpu.get("used_seconds") or cpu.get("used"))
            cpu_limit = _to_float(cpu.get("limit_seconds") or cpu.get("limit"))
            percent = _to_float(cpu.get("percent"))
            if percent is None:
                percent = _percent_used(cpu_seconds, cpu_limit)
            if percent is not None:
                return {
                    "label": "CPU Usage",
                    "value": math.floor(percent),
                    "used": cpu_seconds,
                    "limit": cpu_limit,
                    "unit": "sec",
                    "next_reset_time": cpu.get("next_reset_time"),
                    "source": "snapshot",
                }

    return None


def _pythonanywhere_file_storage_usage() -> Optional[dict[str, Any]]:
    snapshot = _pythonanywhere_usage_snapshot()
    if snapshot:
        storage = snapshot.get("file_storage") or snapshot.get("storage")
        if isinstance(storage, dict):
            percent = _to_float(storage.get("percent"))
            used_bytes = _to_float(storage.get("used_bytes") or storage.get("used"))
            limit_bytes = _to_float(storage.get("limit_bytes") or storage.get("limit"))
            if percent is None:
                percent = _percent_used(used_bytes, limit_bytes)
            if percent is not None:
                return {
                    "label": "File Storage",
                    "value": math.floor(percent),
                    "used": used_bytes,
                    "limit": limit_bytes,
                    "unit": "bytes",
                    "source": snapshot.get("_source"),
                }

    direct_pct = _to_float(os.environ.get("PYTHONANYWHERE_FILE_STORAGE_PCT"))
    if direct_pct is not None:
        return {
            "label": "File Storage",
            "value": round(max(0.0, min(100.0, direct_pct))),
            "source": "env",
        }
    return None


def _pythonanywhere_postgres_usage() -> Optional[dict[str, Any]]:
    snapshot = _pythonanywhere_usage_snapshot()
    if snapshot:
        storage = snapshot.get("postgres_storage") or snapshot.get("postgres")
        if isinstance(storage, dict):
            percent = _to_float(storage.get("pg_storage_percent"))
            if percent is None:
                percent = _to_float(storage.get("percent"))
            used_bytes = _to_float(storage.get("used_bytes") or storage.get("used"))
            limit_bytes = _to_float(storage.get("limit_bytes") or storage.get("limit"))
            if percent is None:
                percent = _percent_used(used_bytes, limit_bytes)
            if percent is not None:
                return {
                    "label": "Postgres Storage",
                    "value": math.floor(percent),
                    "used": used_bytes,
                    "limit": limit_bytes,
                    "unit": "bytes",
                    "source": snapshot.get("_source"),
                }

    direct_pct = _to_float(os.environ.get("PYTHONANYWHERE_POSTGRES_PCT"))
    if direct_pct is not None:
        return {
            "label": "Postgres Storage",
            "value": round(max(0.0, min(100.0, direct_pct))),
            "source": "env",
        }
    return None


def _pythonanywhere_schedule_tasks() -> list[dict[str, Any]]:
    payload = _pythonanywhere_api_get_json("schedule/")
    if not isinstance(payload, list):
        return []
    return [task for task in payload if isinstance(task, dict)]


def _pythonanywhere_always_on_tasks() -> list[dict[str, Any]]:
    payload = _pythonanywhere_api_get_json("always_on/")
    if not isinstance(payload, list):
        return []
    return [task for task in payload if isinstance(task, dict)]


def _task_name_from_command(command: str) -> Optional[str]:
    normalized = command.lower()
    if "run_pipeline" in normalized:
        return "Run Pipeline"
    if "backtest" in normalized:
        return "Screener Backtest"
    if "metrics" in normalized:
        return "Metrics Update"
    if "execute_trades" in normalized:
        return "Execute Trades"
    return None


def _format_task_time(task: Mapping[str, Any]) -> str:
    printable = str(task.get("printable_time") or "").strip()
    interval = str(task.get("interval") or "").lower()
    minute = task.get("minute")
    hour = task.get("hour")
    tz_label = os.environ.get("PYTHONANYWHERE_TASK_TZ") or "UTC"

    if printable:
        return f"{printable} {tz_label}".strip()

    if interval == "hourly":
        try:
            minute_int = int(minute)
            return f"**:{minute_int:02d} {tz_label}"
        except Exception:
            return f"**:00 {tz_label}"

    if interval in {"daily", "weekly", "monthly"}:
        try:
            hour_int = int(hour)
            minute_int = int(minute)
            return f"{hour_int:02d}:{minute_int:02d} {tz_label}"
        except Exception:
            return f"--:-- {tz_label}"

    return "--:--"


def _normalize_schedule_task(task: Mapping[str, Any]) -> dict[str, Any]:
    description = str(task.get("description") or "").strip()
    command = str(task.get("command") or "").strip()
    name = description or _task_name_from_command(command) or "Scheduled Task"
    interval = str(task.get("interval") or "").lower()
    frequency = interval.capitalize() if interval else "Schedule"
    if interval == "daily":
        frequency = "Daily"
    elif interval == "hourly":
        frequency = "Hourly"
    elif interval == "weekly":
        frequency = "Weekly"
    elif interval == "monthly":
        frequency = "Monthly"
    return {
        "name": name,
        "frequency": frequency,
        "time": _format_task_time(task),
    }


def _tail_text_lines(text: str, limit: int = 160) -> str:
    if not text:
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(lines[-limit:])


def _task_log_source_label(task: Mapping[str, Any], logfile: str | None = None) -> str:
    command = str(task.get("command") or task.get("script") or "").strip()
    description = str(task.get("description") or "").strip()
    if command:
        return command
    if description:
        return description
    if logfile:
        return str(logfile)
    return "task"


def _task_logfile_from_detail(kind: str, task_id: Any) -> Optional[str]:
    if task_id is None:
        return None
    try:
        task_id_str = str(task_id)
    except Exception:
        return None
    detail = _pythonanywhere_api_get_json(f"{kind}/{task_id_str}/")
    if isinstance(detail, dict):
        logfile = detail.get("logfile") or detail.get("log_file")
        if logfile:
            return str(logfile)
    return None


def _pythonanywhere_log_sources() -> list[dict[str, str]]:
    tasks: list[tuple[dict[str, Any], str]] = []
    tasks.extend([(task, "schedule") for task in _pythonanywhere_schedule_tasks()])
    tasks.extend([(task, "always_on") for task in _pythonanywhere_always_on_tasks()])

    seen_logs: set[str] = set()
    sources: list[dict[str, str]] = []
    for task, kind in tasks:
        logfile = task.get("logfile") or task.get("log_file")
        if not logfile and kind in {"schedule", "always_on"}:
            logfile = _task_logfile_from_detail(kind, task.get("id"))
        if not logfile:
            continue
        logfile_str = str(logfile)
        if logfile_str in seen_logs:
            continue
        seen_logs.add(logfile_str)

        payload = _fetch_pythonanywhere_file(None, logfile_str)
        if not payload:
            continue
        text, _ = payload
        trimmed = _tail_text_lines(text)
        if not trimmed:
            continue
        sources.append(
            {
                "source": _task_log_source_label(task, logfile_str),
                "text": trimmed,
            }
        )
    return sources


def _fetch_pythonanywhere_file(
    remote_url: Optional[str], remote_path: Optional[str]
) -> Optional[tuple[str, str]]:
    token = _pythonanywhere_token()
    username = _pythonanywhere_username()

    if not token:
        return None
    if not remote_url and not (username and remote_path):
        return None

    if not remote_url:
        if remote_path and username and remote_path.startswith(f"/user/{username}/files"):
            remote_path = remote_path.replace(f"/user/{username}/files", "", 1)
        if remote_path and not remote_path.startswith("/"):
            remote_path = f"/{remote_path}"
        remote_url = (
            f"https://www.pythonanywhere.com/api/v0/user/{username}/files/path{remote_path}"
        )

    try:
        req = urllib.request.Request(remote_url, headers={"Authorization": f"Token {token}"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode("utf-8", errors="ignore"), remote_url
    except Exception:
        return None


def _fetch_pythonanywhere_task_log_for(
    *,
    task_id: Optional[str],
    remote_path: Optional[str],
    remote_url: Optional[str],
    match_keywords: list[str],
) -> Optional[tuple[str, str]]:
    token = _pythonanywhere_token()
    username = _pythonanywhere_username()

    if not token:
        return None
    if not remote_url and not (username and (remote_path or task_id or match_keywords)):
        return None

    if not remote_url:
        if task_id:
            try:
                schedule_url = (
                    f"https://www.pythonanywhere.com/api/v0/user/{username}/schedule/{task_id}/"
                )
                req = urllib.request.Request(
                    schedule_url, headers={"Authorization": f"Token {token}"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
                remote_path = payload.get("logfile")
            except Exception:
                remote_path = None
        elif username and match_keywords:
            try:
                schedule_url = f"https://www.pythonanywhere.com/api/v0/user/{username}/schedule/"
                req = urllib.request.Request(
                    schedule_url, headers={"Authorization": f"Token {token}"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    tasks = json.loads(resp.read().decode("utf-8", errors="ignore"))
                best_task = None
                best_rank = len(match_keywords) + 1
                for task in tasks:
                    description = str(task.get("description") or "")
                    command = str(task.get("command") or "")
                    match_source = f"{description} {command}".lower()
                    for idx, keyword in enumerate(match_keywords):
                        if keyword in match_source and idx < best_rank:
                            best_rank = idx
                            best_task = task
                            break
                if best_task:
                    remote_path = best_task.get("logfile")
            except Exception:
                remote_path = None

        if not remote_path:
            return None

    return _fetch_pythonanywhere_file(remote_url, remote_path)


def _fetch_pythonanywhere_task_log() -> Optional[tuple[str, str]]:
    return _fetch_pythonanywhere_task_log_for(
        task_id=os.environ.get("PYTHONANYWHERE_PIPELINE_TASK_ID"),
        remote_path=os.environ.get("PYTHONANYWHERE_TASK_LOG_PATH"),
        remote_url=os.environ.get("PYTHONANYWHERE_TASK_LOG_URL"),
        match_keywords=["run pipeline", "run_pipeline"],
    )


def _fetch_pythonanywhere_execute_task_log() -> Optional[tuple[str, str]]:
    return _fetch_pythonanywhere_task_log_for(
        task_id=os.environ.get("PYTHONANYWHERE_EXECUTE_TASK_ID"),
        remote_path=os.environ.get("PYTHONANYWHERE_EXECUTE_TASK_LOG_PATH"),
        remote_url=os.environ.get("PYTHONANYWHERE_EXECUTE_TASK_LOG_URL"),
        match_keywords=[
            "pre-market executor",
            "pre-market",
            "premarket",
            "execute trades",
            "execute_trades",
            "execute_trades.py",
        ],
    )


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_daily_bars_cache() -> Optional[pd.DataFrame]:
    path = Path(BASE_DIR) / "data" / "daily_bars.csv"
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None

    cached_df = _DAILY_BARS_CACHE.get("df")
    cached_mtime = _DAILY_BARS_CACHE.get("mtime")
    if cached_df is None or cached_mtime != mtime:
        try:
            df = pd.read_csv(path, usecols=["symbol", "timestamp", "close"])
        except Exception:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["symbol", "timestamp", "close"])
        df["symbol"] = df["symbol"].astype(str)
        df.sort_values(["symbol", "timestamp"], inplace=True)
        _DAILY_BARS_CACHE["df"] = df
        _DAILY_BARS_CACHE["mtime"] = mtime
        cached_df = df

    return cached_df


def _logo_url_for_symbol(symbol: str) -> Optional[str]:
    if not symbol:
        return None
    safe_symbol = quote(symbol.upper(), safe=".-")
    return f"/api/logos/{safe_symbol}.png"


DEFAULT_TRAIL_PERCENT = 3.0

_MONITOR_LOG_LINE_PATTERNS = (
    re.compile(
        r"^\[(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,\d+)?\]\s*(?:\[(\w+)\])?\s*:?\s*(.*)$"
    ),
    re.compile(
        r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:,\d+)?(?:\s+-\s+(\w+)\s+-\s+)?\s*(?:\[(\w+)\])?\s*(.*)$"
    ),
)


def _normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def _days_held_from_timestamp(entry_time: Any) -> int:
    if entry_time in (None, ""):
        return 0
    parsed = pd.to_datetime(entry_time, utc=True, errors="coerce")
    if parsed is None or pd.isna(parsed):
        return 0
    try:
        entry_dt = parsed.to_pydatetime()
    except Exception:
        return 0
    if entry_dt.tzinfo is None:
        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - entry_dt.astimezone(timezone.utc)
    if delta.total_seconds() <= 0:
        return 0
    return max(1, int(delta.total_seconds() // 86400))


def _positions_summary(positions: list[dict[str, Any]]) -> dict[str, float]:
    if not positions:
        return {
            "totalShares": 0.0,
            "totalOpenPL": 0.0,
            "avgDaysHeld": 0.0,
            "totalCapturedPL": 0.0,
        }

    total_shares = 0.0
    total_open_pl = 0.0
    total_days = 0.0
    total_captured_pl = 0.0

    for position in positions:
        shares = _to_float(position.get("qty"))
        open_pl = _to_float(position.get("dollarPL"))
        days_held = _to_float(position.get("daysHeld"))
        captured_pl = _to_float(position.get("capturedPL"))
        if shares is not None:
            total_shares += shares
        if open_pl is not None:
            total_open_pl += open_pl
        if days_held is not None:
            total_days += days_held
        if captured_pl is not None:
            total_captured_pl += captured_pl

    avg_days = total_days / len(positions) if positions else 0.0
    return {
        "totalShares": total_shares,
        "totalOpenPL": total_open_pl,
        "avgDaysHeld": avg_days,
        "totalCapturedPL": total_captured_pl,
    }


def _position_db_metrics(symbols: list[str] | None = None) -> dict[str, dict[str, Any]]:
    if not db.db_enabled():
        return {}

    normalized_symbols = sorted(
        {_normalize_symbol(symbol) for symbol in (symbols or []) if _normalize_symbol(symbol)}
    )
    metrics: dict[str, dict[str, Any]] = {}

    symbol_filter_sql = "AND symbol = ANY(%(symbols)s)" if normalized_symbols else ""
    params = {"symbols": normalized_symbols} if normalized_symbols else None

    open_rows = _db_fetch_all(
        f"""
        SELECT symbol, MIN(entry_time) AS earliest_entry
        FROM trades
        WHERE status = 'OPEN'
          {symbol_filter_sql}
        GROUP BY symbol
        """,
        params,
    )
    for row in open_rows:
        symbol = _normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        bucket = metrics.setdefault(symbol, {})
        bucket["daysHeld"] = _days_held_from_timestamp(row.get("earliest_entry"))

    closed_rows = _db_fetch_all(
        f"""
        SELECT symbol, SUM(COALESCE(realized_pnl, 0)) AS captured_pl
        FROM trades
        WHERE status = 'CLOSED'
          {symbol_filter_sql}
        GROUP BY symbol
        """,
        params,
    )
    for row in closed_rows:
        symbol = _normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        captured_pl = _to_float(row.get("captured_pl"))
        if captured_pl is None:
            continue
        bucket = metrics.setdefault(symbol, {})
        bucket["capturedPL"] = captured_pl

    trail_rows = _db_fetch_all(
        f"""
        SELECT DISTINCT ON (symbol)
            symbol,
            event_time,
            COALESCE(
                NULLIF(COALESCE(raw, '{{}}'::jsonb)->>'trail_percent', ''),
                NULLIF(COALESCE(raw, '{{}}'::jsonb)->>'trail_pct', ''),
                NULLIF(COALESCE(raw, '{{}}'::jsonb)->>'trailPercent', '')
            ) AS trail_percent,
            COALESCE(
                NULLIF(COALESCE(raw, '{{}}'::jsonb)->>'stop_price', ''),
                NULLIF(COALESCE(raw, '{{}}'::jsonb)->>'stopPrice', ''),
                NULLIF(COALESCE(raw, '{{}}'::jsonb)->>'trail_price', '')
            ) AS stop_price
        FROM order_events
        WHERE (
            COALESCE(raw, '{{}}'::jsonb) ? 'trail_percent'
            OR COALESCE(raw, '{{}}'::jsonb) ? 'trail_pct'
            OR COALESCE(raw, '{{}}'::jsonb) ? 'trailPercent'
            OR COALESCE(raw, '{{}}'::jsonb) ? 'stop_price'
            OR COALESCE(raw, '{{}}'::jsonb) ? 'stopPrice'
            OR COALESCE(raw, '{{}}'::jsonb) ? 'trail_price'
            OR event_type ILIKE 'TRAIL%%'
            OR event_type ILIKE '%%STOP%%'
        )
        {"AND symbol = ANY(%(symbols)s)" if normalized_symbols else ""}
        ORDER BY symbol, event_time DESC NULLS LAST, event_id DESC
        """,
        params,
    )
    for row in trail_rows:
        symbol = _normalize_symbol(row.get("symbol"))
        if not symbol:
            continue
        bucket = metrics.setdefault(symbol, {})
        trail_percent = _to_float(row.get("trail_percent"))
        stop_price = _to_float(row.get("stop_price"))
        if trail_percent is not None:
            bucket["trailPercent"] = trail_percent
        if stop_price is not None:
            bucket["trailingStop"] = stop_price

    return metrics


def _enrich_positions_with_db_metrics(
    positions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    symbols = [_normalize_symbol(position.get("symbol")) for position in positions]
    metrics_by_symbol = _position_db_metrics(symbols)

    enriched: list[dict[str, Any]] = []
    for position in positions:
        symbol = _normalize_symbol(position.get("symbol"))
        bucket = metrics_by_symbol.get(symbol, {})
        qty = _to_float(position.get("qty"))
        current_price = _to_float(position.get("currentPrice"))
        entry_price = _to_float(position.get("entryPrice"))
        base_price = current_price if current_price is not None else entry_price

        existing_trail_percent = _to_float(position.get("trailPercent"))
        existing_trailing_stop = _to_float(position.get("trailingStop"))

        trail_percent = existing_trail_percent
        if trail_percent is None:
            trail_percent = _to_float(bucket.get("trailPercent"))
        if trail_percent is None or trail_percent <= 0:
            trail_percent = DEFAULT_TRAIL_PERCENT

        trailing_stop = existing_trailing_stop
        if trailing_stop is None:
            trailing_stop = _to_float(bucket.get("trailingStop"))
        if trailing_stop is None and base_price is not None:
            trailing_stop = base_price * (1 - (trail_percent / 100.0))

        # Captured P/L represents what this trade would realize if the trailing
        # stop is triggered at the current stop price.
        captured_pl = None
        if trailing_stop is not None and entry_price is not None and qty is not None:
            captured_pl = (trailing_stop - entry_price) * qty
        if captured_pl is None:
            captured_pl = _to_float(position.get("capturedPL"))
        if captured_pl is None:
            captured_pl = _to_float(bucket.get("capturedPL"))
        if captured_pl is None:
            captured_pl = 0.0

        existing_days_held = _to_float(position.get("daysHeld"))
        if existing_days_held is not None:
            days_held = max(0, int(round(existing_days_held)))
        else:
            days_held = int(bucket.get("daysHeld") or 0)

        enriched_row = dict(position)
        enriched_row["qty"] = qty if qty is not None else position.get("qty")
        enriched_row["daysHeld"] = days_held
        enriched_row["trailingStop"] = trailing_stop
        enriched_row["capturedPL"] = captured_pl
        enriched_row["trailPercent"] = trail_percent
        enriched.append(enriched_row)

    return enriched, _positions_summary(enriched)


def _parse_monitor_log_line(line: str) -> Optional[dict[str, str]]:
    for index, pattern in enumerate(_MONITOR_LOG_LINE_PATTERNS):
        match = pattern.match(line)
        if not match:
            continue
        if index == 0:
            date, time_text, level, message = match.groups()
        else:
            date, time_text, level_a, level_b, message = match.groups()
            level = level_a or level_b
        return {
            "date": date,
            "time": time_text,
            "level": (level or "").strip(),
            "message": (message or "").strip(),
        }
    return None


def _format_monitor_timestamp(date_text: str, time_text: str) -> str:
    try:
        parsed = datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        )
        hour = parsed.strftime("%I").lstrip("0") or "0"
        return f"{parsed.month}/{parsed.day}/{parsed.year}, {hour}:{parsed.strftime('%M')} {parsed.strftime('%p')} UTC"
    except Exception:
        return f"{date_text} {time_text} UTC"


def _format_utc_datetime_for_monitor(value: datetime) -> str:
    dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    hour = dt.strftime("%I").lstrip("0") or "0"
    return f"{dt.month}/{dt.day}/{dt.year}, {hour}:{dt.strftime('%M')} {dt.strftime('%p')} UTC"


def _coerce_datetime_utc(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if parsed is None or pd.isna(parsed):
        return None
    try:
        converted = parsed.to_pydatetime()
    except Exception:
        return None
    if converted.tzinfo is None:
        return converted.replace(tzinfo=timezone.utc)
    return converted.astimezone(timezone.utc)


def _float_text(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _monitor_log_type(level: str, message: str) -> str:
    level_upper = (level or "").upper()
    lower = message.lower()
    if (
        level_upper.startswith("WARN")
        or level_upper.startswith("ERR")
        or "warning" in lower
        or "threshold" in lower
        or "approach" in lower
        or "fail" in lower
        or "error" in lower
    ):
        return "warning"
    if (
        level_upper == "SUCCESS"
        or "opened" in lower
        or "updated" in lower
        or "adjusted" in lower
        or "submitted" in lower
        or "attach" in lower
    ):
        return "success"
    return "info"


def _is_position_log_message(message: str) -> bool:
    lower = (message or "").lower()
    return any(
        keyword in lower
        for keyword in (
            "trailing stop",
            "trail_",
            "trail ",
            "position ",
            "open position",
            "stop_attach",
            "alert",
            "db_event_ok",
            "db_event_fail",
        )
    )


def _monitor_log_entries_from_text(text: str | None, limit: int = 80) -> list[dict[str, str]]:
    if not text:
        return []
    entries: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = _parse_monitor_log_line(line)
        if not parsed:
            continue
        message = parsed["message"]
        if not message or not _is_position_log_message(message):
            continue
        sort_dt = _coerce_datetime_utc(f"{parsed['date']}T{parsed['time']}Z")
        entries.append(
            {
                "timestamp": _format_monitor_timestamp(parsed["date"], parsed["time"]),
                "type": _monitor_log_type(parsed.get("level", ""), message),
                "message": message,
                "_ts": sort_dt.isoformat() if sort_dt else "",
            }
        )
    if not entries:
        return []
    return list(reversed(entries[-max(1, int(limit)) :]))


def _fetch_pythonanywhere_monitor_log_text() -> Optional[str]:
    remote_payload = _fetch_pythonanywhere_file(
        os.environ.get("PYTHONANYWHERE_MONITOR_LOG_URL"),
        os.environ.get("PYTHONANYWHERE_MONITOR_LOG_PATH"),
    )
    if remote_payload:
        remote_text, _ = remote_payload
        return remote_text

    sources = _pythonanywhere_log_sources()
    if not sources:
        return None

    preferred_chunks: list[str] = []
    fallback_chunks: list[str] = []
    for source in sources:
        source_label = str(source.get("source") or "")
        text = str(source.get("text") or "")
        if not text.strip():
            continue
        source_lower = source_label.lower()
        text_lower = text.lower()
        if (
            "monitor" in source_lower
            or "trailing" in source_lower
            or "monitor" in text_lower
            or "trailing stop" in text_lower
            or "trail_" in text_lower
        ):
            preferred_chunks.append(text)
        else:
            fallback_chunks.append(text)

    selected = preferred_chunks if preferred_chunks else fallback_chunks[:1]
    if not selected:
        return None
    return "\n".join(selected)


def _alpaca_trailing_logs(limit: int = 80, lookback_hours: int = 96) -> list[dict[str, str]]:
    if trading_client is None:
        return []

    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(hours=max(1, int(lookback_hours)))
    try:
        order_request = GetOrdersRequest(
            status="all",
            after=since_utc,
            until=now_utc,
            direction="desc",
            limit=500,
        )
        orders = trading_client.get_orders(filter=order_request) or []
    except Exception as exc:
        logger.warning("Failed to fetch Alpaca trailing orders for monitoring logs: %s", exc)
        return []

    entries: list[dict[str, str]] = []
    for order in orders:
        order_type = getattr(order, "type", None)
        order_type_text = (
            (order_type.value if hasattr(order_type, "value") else str(order_type or ""))
            .strip()
            .lower()
        )

        raw_trail_pct = _to_float(getattr(order, "trail_percent", None))
        raw_trail_price = _to_float(getattr(order, "trail_price", None))
        is_trailing = (
            "trailing" in order_type_text
            or raw_trail_pct is not None
            or raw_trail_price is not None
        )
        if not is_trailing:
            continue

        symbol = _normalize_symbol(getattr(order, "symbol", ""))
        if not symbol:
            continue

        status_value = getattr(order, "status", None)
        status_text = (
            status_value.value if hasattr(status_value, "value") else str(status_value or "")
        ).strip().lower() or "unknown"

        qty = _to_float(getattr(order, "qty", None))
        filled_qty = _to_float(getattr(order, "filled_qty", None))
        effective_qty = filled_qty if filled_qty is not None and filled_qty > 0 else qty
        avg_fill = _to_float(getattr(order, "filled_avg_price", None))

        event_dt = (
            _coerce_datetime_utc(getattr(order, "filled_at", None))
            or _coerce_datetime_utc(getattr(order, "updated_at", None))
            or _coerce_datetime_utc(getattr(order, "submitted_at", None))
            or _coerce_datetime_utc(getattr(order, "created_at", None))
        )
        if event_dt is None:
            continue

        detail_bits: list[str] = []
        if effective_qty is not None:
            detail_bits.append(f"qty={_float_text(effective_qty, 0)}")
        if raw_trail_pct is not None:
            detail_bits.append(f"trail_pct={_float_text(raw_trail_pct)}%")
        if raw_trail_price is not None:
            detail_bits.append(f"trail_price=${_float_text(raw_trail_price)}")
        if avg_fill is not None:
            detail_bits.append(f"fill=${_float_text(avg_fill)}")
        details = ", ".join(detail_bits)
        detail_suffix = f" ({details})" if details else ""

        message = f"Alpaca trailing stop {status_text} for {symbol}{detail_suffix}"
        tone = "info"
        if status_text in {"filled", "partially_filled"}:
            tone = "success"
        elif any(
            token in status_text
            for token in ("canceled", "expired", "rejected", "suspended", "stopped")
        ):
            tone = "warning"

        entries.append(
            {
                "timestamp": _format_utc_datetime_for_monitor(event_dt),
                "type": tone,
                "message": message,
                "_ts": event_dt.isoformat(),
            }
        )

    if not entries:
        return []

    entries.sort(key=lambda item: str(item.get("_ts") or ""), reverse=True)
    return entries[: max(1, int(limit))]


def _positions_logs_live(limit: int = 80) -> tuple[list[dict[str, str]], str]:
    text = _fetch_pythonanywhere_monitor_log_text()
    logs = _monitor_log_entries_from_text(text, limit=limit)
    alpaca_logs = _alpaca_trailing_logs(limit=limit)

    merged = logs + alpaca_logs
    if merged:
        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in sorted(merged, key=lambda item: str(item.get("_ts") or ""), reverse=True):
            key = (str(row.get("_ts") or ""), str(row.get("message") or ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "timestamp": str(row.get("timestamp") or "--"),
                    "type": str(row.get("type") or "info"),
                    "message": str(row.get("message") or ""),
                }
            )
            if len(deduped) >= max(1, int(limit)):
                break
        source = (
            "pythonanywhere+alpaca"
            if logs and alpaca_logs
            else ("pythonanywhere" if logs else "alpaca")
        )
        return deduped, source

    local_path = Path(BASE_DIR) / "logs" / "monitor.log"
    if local_path.exists():
        try:
            local_text = local_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            local_text = ""
        local_logs = _monitor_log_entries_from_text(local_text, limit=limit)
        merged_local = local_logs + alpaca_logs
        if merged_local:
            deduped: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for row in sorted(
                merged_local, key=lambda item: str(item.get("_ts") or ""), reverse=True
            ):
                key = (str(row.get("_ts") or ""), str(row.get("message") or ""))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(
                    {
                        "timestamp": str(row.get("timestamp") or "--"),
                        "type": str(row.get("type") or "info"),
                        "message": str(row.get("message") or ""),
                    }
                )
                if len(deduped) >= max(1, int(limit)):
                    break
            source = (
                "local-fallback+alpaca"
                if local_logs and alpaca_logs
                else ("local-fallback" if local_logs else "alpaca")
            )
            return deduped, source

    if alpaca_logs:
        return [
            {
                "timestamp": str(row.get("timestamp") or "--"),
                "type": str(row.get("type") or "info"),
                "message": str(row.get("message") or ""),
            }
            for row in alpaca_logs[: max(1, int(limit))]
        ], "alpaca"

    return [], "none"


def _alpaca_feed() -> Optional[DataFeed]:
    feed_env = (os.getenv("ALPACA_DATA_FEED") or "IEX").strip().upper()
    if feed_env == "IEX":
        return DataFeed.IEX
    if feed_env == "SIP":
        return DataFeed.SIP
    return None


def _fetch_alpaca_sparklines(symbols: list[str], points: int = 12) -> dict[str, list[float]]:
    if data_client is None or not symbols:
        return {}

    now_utc = datetime.now(timezone.utc)
    start = now_utc - timedelta(days=45)
    feed = _alpaca_feed()
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=now_utc,
            feed=feed,
        )
        bars = data_client.get_stock_bars(request)
    except Exception:
        return {}

    df = getattr(bars, "df", None)
    if df is None or df.empty:
        return {}

    sparkline_map: dict[str, list[float]] = {}
    if isinstance(df.index, pd.MultiIndex) and "symbol" in df.index.names:
        for symbol in symbols:
            if symbol not in df.index.get_level_values("symbol"):
                continue
            symbol_df = df.xs(symbol, level="symbol")
            closes = symbol_df["close"].tail(points).tolist()
            sparkline_map[symbol] = [float(value) for value in closes if value is not None]
        return sparkline_map

    if "symbol" in df.columns:
        for symbol, group in df.groupby("symbol"):
            closes = group["close"].tail(points).tolist()
            sparkline_map[str(symbol)] = [float(value) for value in closes if value is not None]
        return sparkline_map

    return {}


def _positions_from_alpaca() -> list[dict[str, Any]]:
    if trading_client is None:
        return []
    try:
        positions = trading_client.get_all_positions() or []
    except Exception:
        return []

    output: list[dict[str, Any]] = []
    for position in positions:
        symbol = str(getattr(position, "symbol", "") or "").strip()
        if not symbol:
            continue
        qty = _to_float(getattr(position, "qty", None))
        entry_price = _to_float(getattr(position, "avg_entry_price", None))
        current_price = _to_float(getattr(position, "current_price", None))
        dollar_pl = _to_float(getattr(position, "unrealized_pl", None))
        percent_pl = _to_float(getattr(position, "unrealized_plpc", None))
        if percent_pl is not None:
            percent_pl = percent_pl * 100
        output.append(
            {
                "symbol": symbol,
                "qty": qty,
                "entry_price": entry_price,
                "current_price": current_price,
                "dollar_pl": dollar_pl,
                "percent_pl": percent_pl,
            }
        )
    return output


def _active_trailing_stops_from_alpaca(symbols: list[str]) -> dict[str, dict[str, float]]:
    if trading_client is None:
        return {}

    normalized_symbols = {
        _normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)
    }
    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(days=7)

    try:
        req = GetOrdersRequest(
            status="all",
            after=since_utc,
            until=now_utc,
            direction="desc",
            limit=500,
        )
        orders = trading_client.get_orders(filter=req) or []
    except Exception:
        return {}

    inactive_statuses = {
        "canceled",
        "cancelled",
        "expired",
        "rejected",
        "filled",
        "done_for_day",
        "stopped",
    }
    by_symbol: dict[str, dict[str, float]] = {}
    by_symbol_ts: dict[str, datetime] = {}

    for order in orders:
        symbol = _normalize_symbol(getattr(order, "symbol", ""))
        if not symbol or (normalized_symbols and symbol not in normalized_symbols):
            continue

        side_obj = getattr(order, "side", None)
        side_text = (
            (side_obj.value if hasattr(side_obj, "value") else str(side_obj or "")).strip().lower()
        )
        if side_text and side_text != "sell":
            continue

        order_type_obj = getattr(order, "type", None)
        order_type = (
            (
                order_type_obj.value
                if hasattr(order_type_obj, "value")
                else str(order_type_obj or "")
            )
            .strip()
            .lower()
        )
        trail_percent = _to_float(getattr(order, "trail_percent", None))
        trail_price = _to_float(getattr(order, "trail_price", None))
        if "trailing" not in order_type and trail_percent is None and trail_price is None:
            continue

        status_obj = getattr(order, "status", None)
        status_text = (
            (status_obj.value if hasattr(status_obj, "value") else str(status_obj or ""))
            .strip()
            .lower()
        )
        if status_text in inactive_statuses:
            continue

        event_dt = (
            _coerce_datetime_utc(getattr(order, "updated_at", None))
            or _coerce_datetime_utc(getattr(order, "submitted_at", None))
            or _coerce_datetime_utc(getattr(order, "created_at", None))
            or _coerce_datetime_utc(getattr(order, "filled_at", None))
            or datetime.now(timezone.utc)
        )

        stop_price = _to_float(getattr(order, "stop_price", None))
        if stop_price is None:
            # Fallback when the broker only returns high-water mark + trail percent.
            hwm = _to_float(getattr(order, "hwm", None))
            if hwm is not None and trail_percent is not None:
                stop_price = hwm * (1 - (trail_percent / 100.0))

        previous_ts = by_symbol_ts.get(symbol)
        if previous_ts is not None and event_dt <= previous_ts:
            continue

        payload: dict[str, float] = {}
        if stop_price is not None:
            payload["trailingStop"] = stop_price
        if trail_percent is not None:
            payload["trailPercent"] = trail_percent
        if not payload:
            continue

        by_symbol[symbol] = payload
        by_symbol_ts[symbol] = event_dt

    return by_symbol


def _fetch_alpaca_fill_activities(
    *,
    symbols: set[str],
    lookback_days: int = 120,
    page_size: int = 100,
    max_pages: int = 24,
) -> list[dict[str, Any]]:
    if not API_KEY or not API_SECRET:
        return []

    base_url = (os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
    now_utc = datetime.now(timezone.utc)
    after_utc = now_utc - timedelta(days=max(1, int(lookback_days)))
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
        "Accept": "application/json",
    }

    events: list[dict[str, Any]] = []
    page_token: Optional[str] = None
    for _ in range(max(1, int(max_pages))):
        params: dict[str, Any] = {
            "activity_types": "FILL",
            "after": after_utc.isoformat(),
            "until": now_utc.isoformat(),
            "page_size": min(max(1, int(page_size)), 100),
        }
        if page_token:
            params["page_token"] = page_token
        url = f"{base_url}/v2/account/activities?{urlencode(params)}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    break
                raw_text = resp.read().decode("utf-8", errors="ignore")
                try:
                    payload = json.loads(raw_text)
                except Exception:
                    payload = []
                header_page_token = resp.headers.get("Next-Page-Token") or resp.headers.get(
                    "next-page-token"
                )
        except Exception:
            break

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("activities"), list):
            records = payload.get("activities", [])
        else:
            records = []

        for item in records:
            if not isinstance(item, Mapping):
                continue
            symbol = _normalize_symbol(item.get("symbol"))
            if not symbol or (symbols and symbol not in symbols):
                continue
            ts = _coerce_datetime_utc(
                item.get("transaction_time")
                or item.get("processed_at")
                or item.get("date")
                or item.get("timestamp")
            )
            if ts is None:
                continue
            side = str(item.get("side") or "").strip().lower()
            qty = _to_float(item.get("qty") or item.get("quantity"))
            if qty is None or qty <= 0:
                continue
            events.append({"symbol": symbol, "side": side, "qty": qty, "timestamp": ts})

        payload_token = (
            str(payload.get("next_page_token"))
            if isinstance(payload, dict) and payload.get("next_page_token")
            else None
        )
        page_token = header_page_token or payload_token
        if not page_token:
            break

    return events


def _days_held_map_from_alpaca_activities(alpaca_positions: list[dict[str, Any]]) -> dict[str, int]:
    symbol_qty: dict[str, float] = {}
    for position in alpaca_positions:
        symbol = _normalize_symbol(position.get("symbol"))
        qty = _to_float(position.get("qty"))
        if not symbol or qty is None or qty == 0:
            continue
        symbol_qty[symbol] = abs(qty)
    if not symbol_qty:
        return {}

    fills = _fetch_alpaca_fill_activities(symbols=set(symbol_qty.keys()))
    if not fills:
        return {}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for fill in fills:
        symbol = _normalize_symbol(fill.get("symbol"))
        if symbol not in symbol_qty:
            continue
        grouped.setdefault(symbol, []).append(fill)

    days_map: dict[str, int] = {}
    for symbol, open_qty in symbol_qty.items():
        rows = grouped.get(symbol, [])
        if not rows:
            continue
        rows.sort(
            key=lambda item: item.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        remaining = float(open_qty)
        earliest_entry: Optional[datetime] = None

        for row in rows:
            side = str(row.get("side") or "").lower()
            qty = _to_float(row.get("qty")) or 0.0
            ts = row.get("timestamp")
            if qty <= 0 or not isinstance(ts, datetime):
                continue

            if side == "sell":
                remaining += qty
                continue
            if side != "buy":
                continue
            if remaining <= 0:
                break

            consumed = min(remaining, qty)
            if consumed > 0:
                earliest_entry = ts if earliest_entry is None else min(earliest_entry, ts)
                remaining -= consumed

        if earliest_entry is not None:
            days_map[symbol] = _days_held_from_timestamp(earliest_entry)

    return days_map


def _fetch_latest_prices(symbols: list[str]) -> dict[str, float]:
    if data_client is None or not symbols:
        return {}
    feed = _alpaca_feed()
    try:
        request = StockLatestTradeRequest(symbol_or_symbols=symbols, feed=feed)
        latest_trades = data_client.get_stock_latest_trade(request) or {}
    except Exception:
        return {}

    prices: dict[str, float] = {}
    for symbol, trade in latest_trades.items():
        price = getattr(trade, "price", None)
        if price is None:
            continue
        try:
            prices[str(symbol)] = float(price)
        except (TypeError, ValueError):
            continue
    return prices


def _parse_task_log(
    path: Path, fetcher: Optional[Callable[[], Optional[tuple[str, str]]]] = None
) -> dict[str, Any]:
    lines: list[str] | None = None
    source: Optional[str] = None
    remote_payload = (fetcher or _fetch_pythonanywhere_task_log)()
    if remote_payload:
        remote_text, remote_source = remote_payload
        lines = remote_text.splitlines()
        source = remote_source

    if not lines and path.exists():
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            source = str(path)
        except Exception:
            lines = None

    if not lines:
        return {"ok": False}

    start_ts = None
    end_ts = None
    duration_seconds = None
    rc = None

    def _parse_timestamp(line: str) -> Optional[str]:
        match = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", line)
        if not match:
            return None
        date, time = match.groups()
        return f"{date}T{time}Z"

    end_index = None
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if "Completed task" in line:
            end_index = idx
            end_ts = _parse_timestamp(line)
            duration_match = re.search(r"took\s+([0-9.]+)\s+seconds", line)
            if duration_match:
                try:
                    duration_seconds = float(duration_match.group(1))
                except ValueError:
                    duration_seconds = None
            rc_match = re.search(r"return code was\s+(-?\d+)", line)
            if rc_match:
                try:
                    rc = int(rc_match.group(1))
                except ValueError:
                    rc = None
            break

    if end_index is not None:
        for idx in range(end_index - 1, -1, -1):
            line = lines[idx]
            if "Started task" in line:
                start_ts = _parse_timestamp(line)
                break

    return {
        "ok": bool(start_ts or end_ts),
        "started_utc": start_ts,
        "finished_utc": end_ts,
        "duration_seconds": duration_seconds,
        "rc": rc,
        "source": source,
    }


def _db_fetch_one(sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    conn = db.get_db_conn()
    if conn is None:
        return None
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or {})
            row = cursor.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_QUERY_FAIL err=%s", exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _db_fetch_all(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    conn = db.get_db_conn()
    if conn is None:
        return []
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or {})
            rows = cursor.fetchall()
            if not rows:
                return []
            columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_QUERY_FAIL err=%s", exc)
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _metrics_summary_db() -> dict[str, Any]:
    row = _db_fetch_one(
        """
        SELECT run_date, total_trades, win_rate, net_pnl, profit_factor
        FROM metrics_daily
        ORDER BY run_date DESC
        LIMIT 1
        """
    )
    if not row:
        return {}
    return {
        "last_run_utc": _serialize_record(row.get("run_date")),
        "total_trades": _serialize_record(row.get("total_trades")),
        "win_rate": _serialize_record(row.get("win_rate")),
        "net_pnl": _serialize_record(row.get("net_pnl")),
        "profit_factor": _serialize_record(row.get("profit_factor")),
    }


_TRADES_RANGE_ORDER = ["d", "w", "m", "y", "all"]
_TRADES_RANGE_LABELS = {
    "d": "DAILY",
    "w": "WEEKLY",
    "m": "MONTHLY",
    "y": "YEARLY",
    "all": "ALL",
}
_TRADES_RANGE_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}


def _parse_trades_range(value: Any, *, default: str = "all") -> str:
    normalized = str(value or default).strip().lower()
    mapping = {
        "d": "d",
        "day": "d",
        "daily": "d",
        "w": "w",
        "week": "w",
        "weekly": "w",
        "m": "m",
        "month": "m",
        "monthly": "m",
        "y": "y",
        "year": "y",
        "yearly": "y",
        "a": "all",
        "all": "all",
    }
    return mapping.get(normalized, default)


def _parse_positive_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _trades_api_allow_file_fallback() -> bool:
    value = str(os.getenv("TRADES_API_ALLOW_FILE_FALLBACK") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _account_api_allow_live_fallback() -> bool:
    value = str(os.getenv("ACCOUNT_API_ALLOW_LIVE_FALLBACK") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _db_connection_available() -> bool:
    if not db.db_enabled():
        return False
    conn = db.get_db_conn()
    if conn is None:
        return False
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            row = cursor.fetchone()
        return bool(row and row[0] == 1)
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _trades_api_db_ready(source: str, source_detail: str) -> bool:
    if source != "postgres":
        return False
    detail = str(source_detail or "")
    return not detail.startswith("trades-db-")


def _load_trades_for_api_from_db(max_rows: int = 100_000) -> tuple[pd.DataFrame, str]:
    conn = db.get_db_conn()
    if conn is None:
        return pd.DataFrame(), "unavailable"

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.trades')")
            row = cursor.fetchone()
            table_name = row[0] if row else None
            if table_name is None:
                return pd.DataFrame(), "missing"
            cursor.execute(
                """
                SELECT
                    symbol,
                    qty,
                    status,
                    entry_time,
                    entry_price,
                    exit_time,
                    exit_price,
                    realized_pnl,
                    exit_reason
                FROM trades
                ORDER BY COALESCE(exit_time, entry_time) DESC NULLS LAST
                LIMIT %(limit)s
                """,
                {"limit": int(max_rows)},
            )
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns), "ok"
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] TRADES_API_DB_LOAD_FAIL err=%s", exc)
        return pd.DataFrame(), "error"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _load_trades_for_api_from_files() -> tuple[pd.DataFrame, str]:
    sources: list[str] = []
    frames: list[pd.DataFrame] = []

    for path, label in (
        (executed_trades_path, "Executed trades"),
        (trades_log_real_path, "Real trades"),
        (trades_log_path, "Paper trades"),
    ):
        if not os.path.exists(path):
            continue
        df, alert = load_csv(path)
        if alert is not None or df is None or df.empty:
            continue
        sources.append(label)
        frames.append(df.copy())

    if not frames:
        return pd.DataFrame(), "files"

    if len(frames) == 1:
        return frames[0], sources[0]

    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_columns = [
        column
        for column in (
            "trade_id",
            "entry_order_id",
            "order_id",
            "symbol",
            "entry_time",
            "exit_time",
            "qty",
        )
        if column in combined.columns
    ]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns, keep="first")

    source_label = " + ".join(list(dict.fromkeys(sources)))
    return combined, source_label


def _series_from_alias(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    for alias in aliases:
        if alias in df.columns:
            return df[alias]
    return pd.Series([None] * len(df), index=df.index)


def _normalize_trades_api_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "qty",
                "status",
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                "realized_pnl",
                "exit_reason",
                "sort_ts",
            ]
        )

    work = frame.copy()
    work["symbol"] = (
        _series_from_alias(work, ["symbol", "ticker"]).astype(str).str.upper().str.strip()
    )
    work["qty"] = pd.to_numeric(
        _series_from_alias(work, ["qty", "filled_qty", "quantity", "shares"]), errors="coerce"
    )
    work["status"] = (
        _series_from_alias(work, ["status", "order_status"]).astype(str).str.upper().str.strip()
    )
    work["entry_time"] = pd.to_datetime(
        _series_from_alias(work, ["entry_time", "buy_date", "entry_date"]),
        utc=True,
        errors="coerce",
    )
    work["entry_price"] = pd.to_numeric(
        _series_from_alias(work, ["entry_price", "avg_entry_price", "avg_entry"]), errors="coerce"
    )
    work["exit_time"] = pd.to_datetime(
        _series_from_alias(work, ["exit_time", "sell_date", "exit_date"]), utc=True, errors="coerce"
    )
    work["exit_price"] = pd.to_numeric(
        _series_from_alias(work, ["exit_price", "price_sold", "sell_price"]), errors="coerce"
    )
    work["realized_pnl"] = pd.to_numeric(
        _series_from_alias(work, ["realized_pnl", "net_pnl", "pnl", "total_pl", "profit_loss_usd"]),
        errors="coerce",
    )
    work["exit_reason"] = _series_from_alias(work, ["exit_reason"]).astype(str).replace({"nan": ""})

    computed_pnl = (work["exit_price"] - work["entry_price"]) * work["qty"]
    work["realized_pnl"] = work["realized_pnl"].fillna(computed_pnl).fillna(0.0)

    is_closed = work["exit_time"].notna() | work["status"].eq("CLOSED")
    work = work[is_closed].copy()
    work = work[work["symbol"].str.len() > 0]
    work["sort_ts"] = work["exit_time"].fillna(work["entry_time"])
    work.sort_values("sort_ts", ascending=False, na_position="last", inplace=True)
    work.reset_index(drop=True, inplace=True)
    return work


def _load_trades_analytics_frame(max_rows: int = 100_000) -> tuple[pd.DataFrame, str, str]:
    db_frame, db_status = _load_trades_for_api_from_db(max_rows=max_rows)
    if db_status == "ok":
        normalized_db = _normalize_trades_api_frame(db_frame)
        if normalized_db.empty and _trades_api_allow_file_fallback():
            file_frame, file_source = _load_trades_for_api_from_files()
            normalized_file = _normalize_trades_api_frame(file_frame)
            if not normalized_file.empty:
                detail = f"{file_source};db=ok-empty;fallback=enabled"
                return normalized_file, "fallback", detail
        return normalized_db, "postgres", ("trades-empty" if normalized_db.empty else "trades")

    if _trades_api_allow_file_fallback():
        file_frame, file_source = _load_trades_for_api_from_files()
        normalized_file = _normalize_trades_api_frame(file_frame)
        if not normalized_file.empty:
            detail = f"{file_source};db={db_status};fallback=enabled"
            return normalized_file, "fallback", detail

    empty = _normalize_trades_api_frame(pd.DataFrame())
    return empty, "postgres", f"trades-db-{db_status}"


def _filter_trades_by_range(frame: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if range_key == "all":
        return frame
    days = _TRADES_RANGE_DAYS.get(range_key)
    if days is None:
        return frame
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return frame[(frame["sort_ts"].notna()) & (frame["sort_ts"] >= cutoff)]


def _build_range_metrics(frame: pd.DataFrame, range_key: str) -> dict[str, Any]:
    scoped = _filter_trades_by_range(frame, range_key)
    if scoped.empty:
        return {
            "key": range_key,
            "label": _TRADES_RANGE_LABELS[range_key],
            "winRatePct": 0.0,
            "totalPL": 0.0,
            "topTrade": {"symbol": "--", "pl": 0.0},
            "worstLoss": {"symbol": "--", "pl": 0.0},
            "tradesCount": 0,
        }

    total_trades = int(len(scoped))
    wins = int((scoped["realized_pnl"] > 0).sum())
    win_rate_pct = (wins / total_trades) * 100 if total_trades > 0 else 0.0
    total_pl = float(scoped["realized_pnl"].sum())

    top_row = scoped.loc[scoped["realized_pnl"].idxmax()]
    worst_row = scoped.loc[scoped["realized_pnl"].idxmin()]

    return {
        "key": range_key,
        "label": _TRADES_RANGE_LABELS[range_key],
        "winRatePct": float(win_rate_pct),
        "totalPL": total_pl,
        "topTrade": {
            "symbol": str(top_row.get("symbol") or "--"),
            "pl": float(top_row.get("realized_pnl") or 0.0),
        },
        "worstLoss": {
            "symbol": str(worst_row.get("symbol") or "--"),
            "pl": float(worst_row.get("realized_pnl") or 0.0),
        },
        "tradesCount": total_trades,
    }


def _build_leaderboard_rows(
    frame: pd.DataFrame, range_key: str, mode: str, limit: int
) -> list[dict[str, Any]]:
    scoped = _filter_trades_by_range(frame, range_key)
    if scoped.empty:
        return []

    grouped = (
        scoped.groupby("symbol", dropna=False)
        .agg(pl=("realized_pnl", "sum"), latest_sort_ts=("sort_ts", "max"))
        .reset_index()
    )

    if mode == "losers":
        grouped = grouped[grouped["pl"] < 0].sort_values(
            ["pl", "latest_sort_ts"], ascending=[True, False], na_position="last"
        )
    else:
        grouped = grouped[grouped["pl"] > 0].sort_values(
            ["pl", "latest_sort_ts"], ascending=[False, False], na_position="last"
        )

    grouped = grouped.head(limit).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for index, row in grouped.iterrows():
        rows.append(
            {
                "rank": index + 1,
                "symbol": str(row.get("symbol") or "--"),
                "pl": float(row.get("pl") or 0.0),
            }
        )
    return rows


def _build_latest_trades_rows(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    scoped = frame.sort_values("sort_ts", ascending=False, na_position="last").head(limit)
    rows: list[dict[str, Any]] = []
    for _, row in scoped.iterrows():
        entry_time = row.get("entry_time")
        exit_time = row.get("exit_time")
        hold_days = 0
        if isinstance(entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
            hold_days = max(0, int((exit_time - entry_time).total_seconds() // 86400))

        qty = row.get("qty")
        try:
            qty_value = int(float(qty)) if qty is not None else 0
        except (TypeError, ValueError):
            qty_value = 0

        rows.append(
            {
                "symbol": str(row.get("symbol") or "--"),
                "buyDate": _serialize_record(entry_time),
                "sellDate": _serialize_record(exit_time),
                "totalDays": hold_days,
                "totalShares": qty_value,
                "avgEntryPrice": float(row.get("entry_price") or 0.0),
                "priceSold": float(row.get("exit_price") or 0.0),
                "totalPL": float(row.get("realized_pnl") or 0.0),
            }
        )
    return rows


def _record_trades_api_request(
    *, endpoint: str, params: Mapping[str, Any], source: str, rows_returned: int
) -> bool:
    conn = db.get_db_conn()
    if conn is None:
        return False

    payload = json.dumps({key: _serialize_record(value) for key, value in params.items()})
    try:
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO trades_api_requests (endpoint, params, source, rows_returned)
                    VALUES (%(endpoint)s, CAST(%(params)s AS JSONB), %(source)s, %(rows_returned)s)
                    """,
                    {
                        "endpoint": endpoint,
                        "params": payload,
                        "source": source,
                        "rows_returned": int(rows_returned),
                    },
                )
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] TRADES_API_RECORD_FAIL endpoint=%s err=%s", endpoint, exc)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _account_latest_db() -> dict[str, Any]:
    for sql, source in (
        (_ACCOUNT_LATEST_SQL, "v_account_latest"),
        (_ACCOUNT_LATEST_SQL_NO_PORTFOLIO, "v_account_latest"),
        (_ACCOUNT_RECENT_SQL, "alpaca_account_snapshots"),
    ):
        row = _db_fetch_one(sql)
        if row:
            row["source"] = source
            return row
    return {}


def _account_latest_alpaca() -> dict[str, Any]:
    if trading_client is None:
        return {}
    try:
        account = trading_client.get_account()
    except Exception as exc:
        logger.warning("[WARN] ALPACA_ACCOUNT_FETCH_FAILED err=%s", exc)
        return {}
    return {
        "taken_at": datetime.now(timezone.utc).isoformat(),
        "account_id": getattr(account, "id", None) or getattr(account, "account_id", None),
        "status": getattr(account, "status", None),
        "equity": _to_float(getattr(account, "equity", None)),
        "cash": _to_float(getattr(account, "cash", None)),
        "buying_power": _to_float(getattr(account, "buying_power", None)),
        "portfolio_value": _to_float(getattr(account, "portfolio_value", None)),
        "source": "alpaca",
    }


_ACCOUNT_PERFORMANCE_RANGE_ORDER = ("d", "w", "m", "y")
_ACCOUNT_PERFORMANCE_LABELS = {
    "d": "Daily",
    "w": "Weekly",
    "m": "Monthly",
    "y": "Yearly",
}
_ACCOUNT_PERFORMANCE_WINDOW_DAYS = {"w": 7, "m": 30, "y": 365}
_ACCOUNT_OPEN_ORDER_STATUSES = {
    "accepted",
    "accepted_for_bidding",
    "calculated",
    "held",
    "new",
    "partially_filled",
    "pending_cancel",
    "pending_new",
    "pending_replace",
    "stopped",
}
_EXECUTE_LOG_LINE_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+\[(?P<level>[A-Z]+)\]\s*(?P<message>.*)$")


def _alpaca_base_url() -> str:
    return (os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")


def _account_api_is_paper_mode() -> bool:
    return "paper-api" in _alpaca_base_url().lower()


def _alpaca_rest_headers() -> dict[str, str]:
    if not API_KEY or not API_SECRET:
        return {}
    return {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
        "Accept": "application/json",
    }


def _alpaca_rest_get_json(
    path: str, *, params: Mapping[str, Any] | None = None, timeout: int = 15
) -> tuple[Any, str]:
    headers = _alpaca_rest_headers()
    if not headers:
        return {}, "missing_credentials"

    cleaned_path = path if path.startswith("/") else f"/{path}"
    query_params: dict[str, str] = {}
    for key, value in (params or {}).items():
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            query_params[str(key)] = "true" if value else "false"
        else:
            query_params[str(key)] = str(value)
    query = urlencode(query_params)
    url = f"{_alpaca_base_url()}{cleaned_path}"
    if query:
        url = f"{url}?{query}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_text = resp.read().decode("utf-8", errors="ignore")
        payload = json.loads(raw_text) if raw_text else {}
        return payload, "ok"
    except urllib.error.HTTPError as exc:
        logger.warning("[WARN] ALPACA_HTTP_ERROR path=%s code=%s", path, exc.code)
        return {}, f"http_{exc.code}"
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] ALPACA_HTTP_FETCH_FAIL path=%s err=%s", path, exc)
        return {}, "request_failed"


def _parse_account_range(value: Any, *, default: str = "all") -> str:
    normalized = str(value or default).strip().lower()
    mapping = {
        "d": "d",
        "day": "d",
        "daily": "d",
        "w": "w",
        "week": "w",
        "weekly": "w",
        "m": "m",
        "month": "m",
        "monthly": "m",
        "y": "y",
        "year": "y",
        "yearly": "y",
        "a": "all",
        "all": "all",
    }
    return mapping.get(normalized, default)


def _normalize_account_history_period(value: Any) -> tuple[str, str]:
    normalized = str(value or "1Y").strip().upper()
    if normalized in {"ALL", "A", "MAX"}:
        return "ALL", "all"
    if normalized in {"1A", "1Y", "YEAR", "YEARLY", "Y"}:
        return "1Y", "1A"
    return "1Y", "1A"


def _normalize_account_history_timeframe(value: Any) -> str:
    normalized = str(value or "1D").strip().upper()
    aliases = {
        "DAY": "1D",
        "1DAY": "1D",
        "1D": "1D",
        "HOUR": "1H",
        "1HOUR": "1H",
        "1H": "1H",
        "MIN": "1Min",
        "1MIN": "1Min",
        "5MIN": "5Min",
        "15MIN": "15Min",
    }
    return aliases.get(normalized, "1D")


def _coerce_timestamp_utc(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            epoch = float(value)
        except (TypeError, ValueError):
            return None
        if abs(epoch) > 10_000_000_000:
            epoch = epoch / 1000.0
        try:
            return datetime.fromtimestamp(epoch, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
            return _coerce_timestamp_utc(float(stripped))
        return _coerce_datetime_utc(stripped)
    return _coerce_datetime_utc(value)


def _portfolio_history_points(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    timestamps = payload.get("timestamp")
    equities = payload.get("equity")
    if not isinstance(timestamps, list) or not isinstance(equities, list):
        return []

    points: list[dict[str, Any]] = []
    for raw_ts, raw_equity in zip(timestamps, equities):
        ts = _coerce_timestamp_utc(raw_ts)
        equity = _to_float(raw_equity)
        if ts is None or equity is None:
            continue
        points.append({"t": ts.isoformat(), "equity": float(equity), "_dt": ts})
    points.sort(key=lambda item: item.get("_dt") or datetime.fromtimestamp(0, tz=timezone.utc))
    return points


def _fetch_account_portfolio_points(
    *, period: str, timeframe: str
) -> tuple[list[dict[str, Any]], str]:
    payload, detail = _alpaca_rest_get_json(
        "/v2/account/portfolio/history",
        params={
            "period": period,
            "timeframe": timeframe,
            "intraday_reporting": "market_hours",
        },
        timeout=20,
    )
    if detail != "ok":
        return [], detail
    points = _portfolio_history_points(payload)
    if not points:
        return [], "empty_history"
    return points, "ok"


def _account_delta_for_range(
    points: list[dict[str, Any]], range_key: str
) -> tuple[float, float, str]:
    if not points:
        return 0.0, 0.0, "insufficient_history"
    latest_point = points[-1]
    latest_dt = latest_point.get("_dt")
    latest_equity = _to_float(latest_point.get("equity"))
    if latest_dt is None or latest_equity is None:
        return 0.0, 0.0, "insufficient_history"

    baseline: Optional[dict[str, Any]] = None
    status = "ok"
    if range_key == "all":
        if len(points) < 2:
            return 0.0, 0.0, "insufficient_history"
        baseline = points[0]
    elif range_key == "d":
        if len(points) < 2:
            return 0.0, 0.0, "insufficient_history"
        baseline = points[-2]
    else:
        window_days = _ACCOUNT_PERFORMANCE_WINDOW_DAYS.get(range_key)
        if window_days is None:
            return 0.0, 0.0, "invalid_range"
        cutoff = latest_dt - timedelta(days=window_days)
        oldest_dt = points[0].get("_dt")
        if oldest_dt is None:
            return 0.0, 0.0, "insufficient_history"
        if oldest_dt > cutoff:
            baseline = points[0]
            status = "partial_history"
        else:
            baseline = points[0]
            for point in points:
                point_dt = point.get("_dt")
                if point_dt is None:
                    continue
                if point_dt >= cutoff:
                    baseline = point
                    break

    if baseline is None:
        return 0.0, 0.0, "insufficient_history"
    baseline_index = points.index(baseline)
    baseline_equity = _to_float(baseline.get("equity"))
    if baseline_equity is None:
        return 0.0, 0.0, "insufficient_history"

    # Some Alpaca histories begin with zero-equity placeholders; for returns,
    # use the first non-zero point at or after baseline if available.
    if baseline_equity == 0:
        for candidate in points[baseline_index:]:
            candidate_equity = _to_float(candidate.get("equity"))
            if candidate_equity is None or candidate_equity == 0:
                continue
            baseline_equity = candidate_equity
            status = "partial_history" if status == "ok" else status
            break
        if baseline_equity == 0:
            return 0.0, 0.0, "insufficient_history"

    net_change_usd = float(latest_equity - baseline_equity)
    if baseline_equity == 0:
        return 0.0, net_change_usd, status
    net_change_pct = (net_change_usd / baseline_equity) * 100.0
    return float(net_change_pct), net_change_usd, status


def _coerce_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except Exception:
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _order_log_level(event_type: str, status: str, message: str, raw_level: str = "") -> str:
    normalized_raw = str(raw_level or "").strip().lower()
    if normalized_raw in {"success", "info", "warning"}:
        return normalized_raw
    if normalized_raw in {"error", "warn"}:
        return "warning"

    text = f"{event_type} {status} {message}".lower()
    if any(token in text for token in ("reject", "cancel", "error", "fail", "expired")):
        return "warning"
    if any(token in text for token in ("fill", "submit", "confirmed", "accepted", "new")):
        return "success"
    return "info"


def _order_event_message(row: Mapping[str, Any], raw: Mapping[str, Any]) -> str:
    for key in ("message", "msg", "detail", "description", "event"):
        candidate = raw.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    event_type = str(row.get("event_type") or "EVENT").strip().upper()
    symbol = str(row.get("symbol") or "").strip().upper()
    qty = _to_float(row.get("qty"))
    status = str(row.get("status") or "").strip().lower()

    parts = [event_type]
    if symbol:
        parts.append(symbol)
    if qty is not None:
        parts.append(f"qty={_float_text(qty, 0)}")
    if status:
        parts.append(f"status={status}")
    return " ".join(parts)


def _account_order_logs_from_db(limit: int) -> list[dict[str, Any]]:
    day_start_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = _db_fetch_all(
        """
        SELECT event_time, event_type, symbol, qty, order_id, status, raw
        FROM order_events
        WHERE event_time >= %(start_utc)s
        ORDER BY event_time DESC NULLS LAST
        LIMIT %(limit)s
        """,
        {"start_utc": day_start_utc, "limit": int(limit)},
    )

    parsed: list[dict[str, Any]] = []
    for row in rows:
        ts = _coerce_datetime_utc(row.get("event_time"))
        if ts is None:
            continue
        raw_payload = _coerce_json_mapping(row.get("raw"))
        message = _order_event_message(row, raw_payload)
        level = _order_log_level(
            str(row.get("event_type") or ""),
            str(row.get("status") or ""),
            message,
            str(raw_payload.get("level") or raw_payload.get("severity") or ""),
        )
        parsed.append({"ts": ts.isoformat(), "level": level, "message": message, "_ts": ts})

    parsed.sort(
        key=lambda item: item.get("_ts") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return parsed[: max(1, int(limit))]


def _clean_execute_log_message(message: str) -> str:
    cleaned = (message or "").strip()
    while cleaned:
        next_cleaned = re.sub(
            r"^\[(?:INFO|WARNING|WARN|ERROR|SUCCESS)\]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    return cleaned


def _account_order_logs_from_file(limit: int) -> list[dict[str, Any]]:
    lines = tail_log(execute_trades_log_path, limit=max(200, int(limit) * 8))
    if not lines:
        return []

    day_start_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    parsed: list[dict[str, Any]] = []

    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line:
            continue
        match = _EXECUTE_LOG_LINE_RE.match(line)
        if not match:
            continue
        ts = _coerce_datetime_utc(match.group("ts"))
        if ts is None or ts < day_start_utc:
            continue
        level_text = match.group("level")
        message = _clean_execute_log_message(match.group("message"))
        parsed.append(
            {
                "ts": ts.isoformat(),
                "level": _order_log_level(level_text, "", message, level_text),
                "message": message,
                "_ts": ts,
            }
        )
        if len(parsed) >= max(1, int(limit)):
            break

    return parsed


def _account_order_logs(limit: int) -> tuple[list[dict[str, str]], str]:
    db_rows = _account_order_logs_from_db(limit)
    if db_rows:
        return [
            {
                "ts": str(row.get("ts") or ""),
                "level": str(row.get("level") or "info"),
                "message": str(row.get("message") or ""),
            }
            for row in db_rows[: max(1, int(limit))]
        ], "postgres:order_events"

    file_rows = _account_order_logs_from_file(limit)
    if file_rows:
        return [
            {
                "ts": str(row.get("ts") or ""),
                "level": str(row.get("level") or "info"),
                "message": str(row.get("message") or ""),
            }
            for row in file_rows[: max(1, int(limit))]
        ], "log-fallback:execute_trades.log"

    return [], "none"


def _open_positions_value_from_alpaca() -> Optional[float]:
    positions = _positions_from_alpaca()
    if not positions:
        return None
    total_value = 0.0
    for position in positions:
        qty = _to_float(position.get("qty"))
        current_price = _to_float(position.get("current_price"))
        entry_price = _to_float(position.get("entry_price"))
        ref_price = current_price if current_price is not None else entry_price
        if qty is None or ref_price is None:
            continue
        total_value += abs(qty) * ref_price
    return float(total_value)


def _order_price_or_stop(raw: Mapping[str, Any]) -> Optional[float]:
    for field in ("limit_price", "stop_price", "trail_price"):
        value = _to_float(raw.get(field))
        if value is not None:
            return value
    return None


def _account_open_orders_from_db(limit: int) -> tuple[list[dict[str, Any]], str]:
    scan_limit = max(250, int(limit) * 8)
    rows = _db_fetch_all(
        """
        SELECT DISTINCT ON (
            COALESCE(
                NULLIF(order_id, ''),
                symbol || ':' || COALESCE(event_time::text, '')
            )
        )
            symbol,
            qty,
            order_id,
            status,
            event_type,
            event_time,
            raw
        FROM order_events
        ORDER BY
            COALESCE(
                NULLIF(order_id, ''),
                symbol || ':' || COALESCE(event_time::text, '')
            ),
            event_time DESC NULLS LAST
        LIMIT %(scan_limit)s
        """,
        {"scan_limit": scan_limit},
    )
    if not rows:
        return [], "postgres:order_events_empty"

    output: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "").strip().lower()
        if status and status not in _ACCOUNT_OPEN_ORDER_STATUSES:
            continue

        raw_payload = _coerce_json_mapping(row.get("raw"))
        qty = _to_float(row.get("qty"))
        submitted_at_dt = _coerce_datetime_utc(row.get("event_time"))
        submitted_at = (
            submitted_at_dt.isoformat()
            if submitted_at_dt is not None
            else str(row.get("event_time") or "")
        )

        side_value = (
            str(raw_payload.get("side") or raw_payload.get("order_side") or "buy").strip().lower()
        )
        type_value = (
            str(
                raw_payload.get("type")
                or raw_payload.get("order_type")
                or row.get("event_type")
                or "market"
            )
            .strip()
            .lower()
        )
        if type_value.endswith("_submit"):
            type_value = type_value.replace("_submit", "")

        output.append(
            {
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "type": type_value,
                "side": side_value,
                "qty": float(qty) if qty is not None else 0.0,
                "price_or_stop": _order_price_or_stop(raw_payload),
                "submitted_at": submitted_at,
            }
        )

    output.sort(key=lambda entry: str(entry.get("submitted_at") or ""), reverse=True)
    return output[: max(1, int(limit))], "postgres:order_events"


def _account_summary_from_db() -> tuple[dict[str, Any] | None, str]:
    row = _account_latest_db()
    if not row:
        return None, "db_unavailable"

    equity = _to_float(row.get("equity")) or 0.0
    cash = _to_float(row.get("cash")) or 0.0
    buying_power = _to_float(row.get("buying_power")) or 0.0
    portfolio_value = _to_float(row.get("portfolio_value"))
    open_positions_value = (
        (portfolio_value - cash) if portfolio_value is not None else (equity - cash)
    )
    if open_positions_value < 0:
        open_positions_value = 0.0
    ratio = float(cash / open_positions_value) if open_positions_value > 0 else None

    taken_value = _serialize_record(row.get("taken_at"))
    taken_at_utc = str(taken_value) if isinstance(taken_value, str) and taken_value else ""
    if not taken_at_utc:
        taken_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return (
        {
            "equity": float(equity),
            "cash": float(cash),
            "buying_power": float(buying_power),
            "open_positions_value": float(open_positions_value),
            "cash_to_positions_ratio": ratio,
            "taken_at_utc": taken_at_utc,
        },
        f"postgres:{row.get('source') or 'account'}",
    )


def _account_portfolio_points_from_db(
    *, period: str, timeframe: str
) -> tuple[list[dict[str, Any]], str]:
    params: dict[str, Any] = {"limit": 5000}
    where_clause = ""
    if str(period).upper() == "1Y":
        params["start_utc"] = datetime.now(timezone.utc) - timedelta(days=366)
        where_clause = "AND taken_at >= %(start_utc)s"

    rows = _db_fetch_all(
        f"""
        SELECT taken_at, equity
        FROM alpaca_account_snapshots
        WHERE equity IS NOT NULL
          {where_clause}
        ORDER BY taken_at ASC
        LIMIT %(limit)s
        """,
        params,
    )
    if not rows:
        return [], "db_empty"

    points: list[dict[str, Any]] = []
    for row in rows:
        ts = _coerce_datetime_utc(row.get("taken_at"))
        equity = _to_float(row.get("equity"))
        if ts is None or equity is None:
            continue
        points.append({"t": ts.isoformat(), "equity": float(equity), "_dt": ts})

    if not points:
        return [], "db_invalid_rows"

    if str(timeframe).upper() == "1D":
        by_day: dict[str, dict[str, Any]] = {}
        for point in points:
            dt = point.get("_dt")
            if dt is None:
                continue
            day_key = dt.date().isoformat()
            # points are ASC; overwrite to keep end-of-day snapshot
            by_day[day_key] = point
        points = list(by_day.values())

    points.sort(key=lambda item: item.get("_dt") or datetime.fromtimestamp(0, tz=timezone.utc))
    return points, "ok"


def _open_positions_value_from_csv() -> Optional[float]:
    path = Path(open_positions_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    if "market_value" in df.columns:
        values = pd.to_numeric(df["market_value"], errors="coerce").dropna()
        if not values.empty:
            return float(values.sum())

    qty_series = pd.to_numeric(df.get("qty"), errors="coerce")
    current_series = pd.to_numeric(df.get("current_price"), errors="coerce")
    if qty_series is not None and current_series is not None:
        combined = (qty_series.fillna(0) * current_series.fillna(0)).abs()
        if not combined.empty:
            return float(combined.sum())
    return None


def _account_summary_from_csv() -> tuple[dict[str, Any] | None, str]:
    path = Path(BASE_DIR) / "data" / "account_equity.csv"
    if not path.exists():
        return None, "csv_missing"
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, "csv_unreadable"
    if df is None or df.empty:
        return None, "csv_empty"

    required_cols = {"timestamp", "equity"}
    if not required_cols.issubset(set(df.columns)):
        return None, "csv_missing_columns"

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    if "cash" in frame.columns:
        frame["cash"] = pd.to_numeric(frame["cash"], errors="coerce")
    else:
        frame["cash"] = np.nan
    if "buying_power" in frame.columns:
        frame["buying_power"] = pd.to_numeric(frame["buying_power"], errors="coerce")
    else:
        frame["buying_power"] = np.nan

    frame = frame.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if frame.empty:
        return None, "csv_no_valid_rows"

    latest = frame.iloc[-1]
    equity = _to_float(latest.get("equity")) or 0.0
    cash = _to_float(latest.get("cash"))
    if cash is None:
        cash = equity
    buying_power = _to_float(latest.get("buying_power"))
    if buying_power is None:
        buying_power = 0.0

    open_positions_value = _open_positions_value_from_csv()
    if open_positions_value is None:
        inferred = equity - cash
        open_positions_value = inferred if inferred > 0 else 0.0
    ratio = float(cash / open_positions_value) if open_positions_value > 0 else None

    taken_at = latest.get("timestamp")
    taken_at_utc = ""
    if isinstance(taken_at, pd.Timestamp):
        taken_at_utc = taken_at.to_pydatetime().isoformat()
    elif taken_at is not None and not pd.isna(taken_at):
        parsed = _coerce_datetime_utc(taken_at)
        taken_at_utc = parsed.isoformat() if parsed else ""
    if not taken_at_utc:
        taken_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return (
        {
            "equity": float(equity),
            "cash": float(cash),
            "buying_power": float(buying_power),
            "open_positions_value": float(open_positions_value),
            "cash_to_positions_ratio": ratio,
            "taken_at_utc": taken_at_utc,
        },
        "csv:account_equity",
    )


def _account_portfolio_points_from_csv(
    *, period: str, timeframe: str
) -> tuple[list[dict[str, Any]], str]:
    path = Path(BASE_DIR) / "data" / "account_equity.csv"
    if not path.exists():
        return [], "csv_missing"
    try:
        df = pd.read_csv(path)
    except Exception:
        return [], "csv_unreadable"
    if df is None or df.empty:
        return [], "csv_empty"
    if "timestamp" not in df.columns or "equity" not in df.columns:
        return [], "csv_missing_columns"

    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if frame.empty:
        return [], "csv_no_valid_rows"

    if str(period).upper() == "1Y":
        cutoff = datetime.now(timezone.utc) - timedelta(days=366)
        frame = frame[frame["timestamp"] >= cutoff]
        if frame.empty:
            return [], "csv_period_empty"

    points: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        ts_raw = row.get("timestamp")
        ts = None
        if isinstance(ts_raw, pd.Timestamp):
            ts = ts_raw.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
        else:
            ts = _coerce_datetime_utc(ts_raw)
        equity = _to_float(row.get("equity"))
        if ts is None or equity is None:
            continue
        points.append({"t": ts.isoformat(), "equity": float(equity), "_dt": ts})

    if not points:
        return [], "csv_no_points"

    if str(timeframe).upper() == "1D":
        by_day: dict[str, dict[str, Any]] = {}
        for point in points:
            dt = point.get("_dt")
            if dt is None:
                continue
            by_day[dt.date().isoformat()] = point
        points = list(by_day.values())

    points.sort(key=lambda item: item.get("_dt") or datetime.fromtimestamp(0, tz=timezone.utc))
    return points, "ok"


def _serialize_record(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


# React build output for the new Dashboard UI
REACT_BUILD_DIR = Path(BASE_DIR) / "frontend" / "dist"
REACT_ASSET_DIR = REACT_BUILD_DIR / "assets"
BRAND_ASSET_DIR = Path(BASE_DIR) / "assets"

# Absolute paths to CSV data files used throughout the dashboard
trades_log_path = os.path.join(BASE_DIR, "data", "trades_log.csv")
trades_log_real_path = os.path.join(BASE_DIR, "data", "trades_log_real.csv")
open_positions_path = os.path.join(BASE_DIR, "data", "open_positions.csv")
top_candidates_path = os.path.join(BASE_DIR, "data", "top_candidates.csv")
screener_metrics_path = os.path.join(BASE_DIR, "data", "screener_metrics.json")
latest_candidates_path = os.path.join(BASE_DIR, "data", "latest_candidates.csv")

TOP_CANDIDATES = Path(top_candidates_path)
LATEST_CANDIDATES = Path(latest_candidates_path)
TRADES_LOG_PATH = Path(trades_log_path)

# Additional datasets introduced for monitoring
metrics_summary_path = os.path.join(BASE_DIR, "data", "metrics_summary.csv")
executed_trades_path = os.path.join(BASE_DIR, "data", "executed_trades.csv")
historical_candidates_path = os.path.join(BASE_DIR, "data", "historical_candidates.csv")
execute_metrics_path = os.path.join(BASE_DIR, "data", "execute_metrics.json")

# Absolute paths to log files for the Screener tab
screener_log_dir = os.path.join(BASE_DIR, "logs")
pipeline_log_path = os.path.join(screener_log_dir, "pipeline.log")
monitor_log_path = os.path.join(screener_log_dir, "monitor.log")
# Additional logs
screener_log_path = os.path.join(screener_log_dir, "screener.log")
backtest_log_path = os.path.join(screener_log_dir, "backtest.log")
execute_trades_log_path = os.path.join(screener_log_dir, "execute_trades.log")
error_log_path = os.path.join(screener_log_dir, "error.log")
metrics_log_path = os.path.join(screener_log_dir, "metrics.log")
pipeline_status_json_path = os.path.join(BASE_DIR, "data", "pipeline_status.json")

# Threshold in minutes to consider a log stale
STALE_THRESHOLD_MINUTES = 1440  # 24 hours
ERROR_RETENTION_DAYS = 1

LOG_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")
LOG_LEVEL_RE = re.compile(r"\b(INFO|ERROR)\b")
LEVEL_HINTS = ("[INFO]", "[ERROR]", " - INFO - ", " - ERROR - ")

# Displayed configuration values
MAX_OPEN_TRADES = 10


def _normalize_pnl(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return ``df`` with normalized ``pnl`` / ``net_pnl`` columns."""

    if df is None:
        return None
    normalized = df.copy()
    for alias in ("net_pnl", "netPnL", "net_pnl_usd"):
        if alias in normalized.columns and "net_pnl" not in normalized.columns:
            normalized["net_pnl"] = normalized[alias]
            break

    if "net_pnl" not in normalized.columns and "pnl" in normalized.columns:
        normalized["net_pnl"] = normalized["pnl"]
    if "pnl" not in normalized.columns and "net_pnl" in normalized.columns:
        normalized["pnl"] = normalized["net_pnl"]
    return normalized


def _coerce_int_value(value: Any) -> int:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return 0
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_paper_mode() -> bool:
    """Return True when the Alpaca base URL indicates paper trading."""

    return "paper-api" in (os.getenv("APCA_API_BASE_URL", "") or "").lower()


TRADES_LOG_REAL = Path(trades_log_real_path)
TRADES_LOG_PAPER = Path(trades_log_path)
TRADES_LOG_FOR_SYMBOLS = TRADES_LOG_PAPER if _is_paper_mode() else TRADES_LOG_REAL
TRADE_PERFORMANCE_CACHE = Path(TRADE_PERFORMANCE_CACHE_PATH)
TRADE_PERF_TABLE_FIELDS = [
    "symbol",
    "entry_time",
    "exit_time",
    "qty",
    "entry_price",
    "exit_price",
    "pnl",
    "return_pct",
    "hold_days",
    "exit_reason",
    "mfe_pct",
    "mae_pct",
    "peak_price",
    "trough_price",
    "missed_profit_pct",
    "exit_efficiency_pct",
    "is_trailing_stop_exit",
    "rebound_window_days",
    "rebound_pct",
    "rebounded",
    "post_exit_high",
]


def tail_log(log_path: str, limit: int = 10) -> list[str]:
    """Return up to ``limit`` most recent non-empty lines from ``log_path``."""

    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return []
    trimmed = [line.rstrip() for line in lines if line.strip()]
    return trimmed[-limit:]


def _line_contains_level(line: str) -> bool:
    return bool(LOG_LEVEL_RE.search(line)) or any(hint in line for hint in LEVEL_HINTS)


def _parse_log_timestamp(line: str) -> Optional[datetime]:
    match = LOG_TS_RE.match(line.strip())
    if not match:
        return None
    ts_str = match.group("ts")
    ts_str = ts_str.replace("T", " ")
    try:
        parsed = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return parsed


def _log_activity_badge(lines: list[str], *, path: str | None = None) -> Optional[dbc.Badge]:
    last_event: Optional[datetime] = None
    for raw_line in reversed(lines):
        if not _line_contains_level(raw_line):
            continue
        candidate = _parse_log_timestamp(raw_line)
        if candidate:
            last_event = candidate
            break
    if last_event is None and path:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        if mtime:
            last_event = datetime.utcfromtimestamp(mtime)
    if not last_event:
        return None
    age = (_utcnow() - last_event).total_seconds() / 3600
    if age < 1:
        color = "success"
    elif age < 24:
        color = "warning"
    else:
        color = "danger"
    label = last_event.strftime("Last event: %Y-%m-%d %H:%M UTC")
    return dbc.Badge(label, color=color, pill=True, className="ms-2 small")


def _freshness_badge(path: str) -> Optional[dbc.Badge]:
    mtime = get_file_mtime(path)
    if not mtime:
        return None
    dt = datetime.utcfromtimestamp(mtime)
    age_minutes = int((_utcnow() - dt).total_seconds() / 60)
    if age_minutes < 60:
        color = "success"
    elif age_minutes < 60 * 6:
        color = "warning"
    else:
        color = "danger"
    label = dt.strftime("Data freshness: %Y-%m-%d %H:%M UTC")
    return dbc.Badge(label, color=color, pill=True, className="ms-2 small")


def _score_breakdown_badges(raw: Any) -> str:
    if isinstance(raw, str):
        try:
            data = json.loads(raw) if raw.strip() else {}
        except Exception:
            data = {}
    elif isinstance(raw, dict):
        data = raw
    else:
        data = {}
    if not data:
        return ""
    fragments: list[str] = []
    for key, value in sorted(data.items()):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            color = "success"
        elif numeric < 0:
            color = "danger"
        else:
            color = "secondary"
        fragments.append(f'<span class="badge bg-{color} me-1">{key}: {numeric:+.2f}</span>')
    return "".join(fragments)


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def is_log_stale(path, max_age_hours: int = 24) -> bool:
    """Return True if the latest log activity is older than ``max_age_hours``.

    Matches logger_utils format (" - INFO - " / " - ERROR - ") and falls back to
    file modification time when timestamps are not present.
    """

    try:
        log_path = Path(path)
    except TypeError:
        return True

    try:
        if not log_path.exists():
            return True

        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            lines = []

        last_ts = None
        for line in reversed(lines):
            if not _line_contains_level(line):
                continue
            candidate = _parse_log_timestamp(line)
            if candidate:
                last_ts = candidate
                break

        if last_ts is None:
            mtime = datetime.utcfromtimestamp(os.path.getmtime(log_path))
            last_ts = mtime

        return (_utcnow() - last_ts).total_seconds() > max_age_hours * 3600
    except Exception:
        return True


# Load env the same way our scripts do (user config + repo .env), then ensure
# repo-local .env is also considered for dashboard runtime.
_load_runtime_env(required_keys=())
load_dotenv(os.path.join(BASE_DIR, ".env"))
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
if not API_KEY or not API_SECRET:
    trading_client = None
    logger.warning("Missing Alpaca credentials; Alpaca API features disabled")
    data_client = None
else:
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    try:
        data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    except Exception:
        data_client = None
logger = logging.getLogger(__name__)


def _environment_state() -> tuple[bool, str]:
    base_url = (os.getenv("APCA_API_BASE_URL") or "").lower()
    paper = "paper-api" in base_url
    if not paper:
        exec_flag = (os.getenv("JBR_EXEC_PAPER") or "").strip().lower()
        paper = exec_flag in {"1", "true", "yes", "on"}
    feed = (os.getenv("ALPACA_DATA_FEED") or "").strip().upper() or "?"
    return paper, feed


def _detect_paper_mode() -> bool:
    """Return True when Alpaca is configured for paper trading."""

    paper, _ = _environment_state()
    return paper


PAPER_TRADING_MODE = _detect_paper_mode()


def _paper_badge_component() -> dbc.Badge:
    """Return a subtle badge indicating execution environment."""

    paper, feed = _environment_state()
    label = f"{'Paper' if paper else 'Live'} ({feed})"
    if paper:
        badge_kwargs = dict(color="info", text_color="dark")
        style = {
            "backgroundColor": "#cfe2ff",
            "color": "#084298",
        }
    else:
        badge_kwargs = dict(color="success", text_color="dark")
        style = {
            "backgroundColor": "#d1e7dd",
            "color": "#0f5132",
        }
    return dbc.Badge(
        label,
        className="me-2",
        style={
            "fontSize": "0.75rem",
            "letterSpacing": "0.04em",
            "padding": "0.25rem 0.5rem",
            "fontWeight": 600,
            **style,
        },
        **badge_kwargs,
    )


def connection_badge_color(health_data: Mapping[str, Any]) -> Optional[str]:
    """Return the Bootstrap color for the Alpaca connection badge."""

    trading_ok = health_data.get("trading_ok")
    data_ok = health_data.get("data_ok")
    if trading_ok is None and data_ok is None:
        return None
    if trading_ok is None or data_ok is None:
        return "secondary"
    return "success" if bool(trading_ok) and bool(data_ok) else "danger"


def fetch_positions_api():
    """Fetch open positions from Alpaca for fallback."""
    if trading_client is None:
        logger.warning(
            "Alpaca credentials missing; skipping API position fallback",
        )
        return pd.DataFrame()

    try:
        positions = trading_client.get_all_positions()
        return pd.DataFrame(
            [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "unrealized_pl": p.unrealized_pl,
                    "entry_time": getattr(p, "created_at", datetime.utcnow()).isoformat(),
                }
                for p in positions
            ]
        )
    except Exception as e:
        logger.error("open_positions.csv empty; failed Alpaca fallback: %s", e)
        return pd.DataFrame()


def load_csv(csv_path, required_columns=None, alert_prefix=""):
    """Load a CSV file and validate required columns."""
    required_columns = required_columns or []
    prefix = f"{alert_prefix}: " if alert_prefix else ""
    if not os.path.exists(csv_path):
        return None, dbc.Alert(
            f"{prefix}No data yet. Expected file {csv_path} was not found.",
            color="info",
        )
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", csv_path, exc)
        return None, dbc.Alert(
            f"{prefix}Unable to read {csv_path}. See logs for details.",
            color="danger",
        )
    df = _normalize_pnl(df)
    if df is None or df.empty:
        return None, dbc.Alert(f"{prefix}No data yet in {csv_path}.", color="info")
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return None, dbc.Alert(
            f"{prefix}Missing required columns: {missing_cols}",
            color="danger",
        )
    return df, None


def load_screener_kpis() -> tuple[dict[str, Any], dbc.Alert | None]:
    """Load screener KPIs with a pipeline.log fallback.

    The function never raises; on failure it returns default KPI values with a
    human-friendly alert component describing the issue.
    """

    defaults = {
        "last_run_utc": None,
        "symbols_in": None,
        "symbols_with_bars": None,
        "bars_rows_total": None,
        "rows": None,
    }
    alert: dbc.Alert | None = None

    try:
        kpis = dict(_load_screener_health_kpis())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load screener KPIs directly: %s", exc)
        kpis = {}

    payload = defaults | kpis

    missing = [
        key
        for key in ("symbols_in", "symbols_with_bars", "bars_rows_total", "rows")
        if not isinstance(payload.get(key), int)
    ]
    if missing:
        pipeline_fallback = coerce_kpi_types(parse_pipeline_summary(Path(pipeline_log_path)))
        for key, value in pipeline_fallback.items():
            if key in defaults and payload.get(key) is None and value is not None:
                payload[key] = value
        still_missing = [
            key
            for key in ("symbols_in", "symbols_with_bars", "bars_rows_total", "rows")
            if not isinstance(payload.get(key), int)
        ]
        if still_missing:
            alert = dbc.Alert(
                "Screener metrics unavailable; showing defaults. Check logs/pipeline.log.",
                color="warning",
                className="mb-2",
            )

    return payload, alert


def load_latest_candidates():
    """Load latest screener candidates (DB-first with CSV fallback)."""

    canonical_header = [
        "timestamp",
        "symbol",
        "score",
        "exchange",
        "close",
        "volume",
        "universe_count",
        "score_breakdown",
        "entry_price",
        "adv20",
        "atrp",
        "source",
        "sma9",
        "ema20",
        "sma180",
        "rsi14",
        "passed_gates",
        "gate_fail_reason",
    ]
    df, updated, source_file = screener_table()
    source_label = source_file or "unknown"
    if df is None or df.empty:
        return None, dbc.Alert(
            "No candidates yet (DB view empty/unreachable and CSV fallback unavailable).",
            color="info",
        )

    work = df.copy()
    if "timestamp" not in work.columns:
        if "run_date" in work.columns:
            try:
                ts = pd.to_datetime(work["run_date"], errors="coerce", utc=True)
                work["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                work["timestamp"] = pd.NA
        else:
            work["timestamp"] = pd.NA

    if "source" not in work.columns:
        work["source"] = source_label
    else:
        source_series = work["source"].astype("string").fillna("")
        work["source"] = source_series.where(source_series.str.strip() != "", source_label)

    required_columns = {"symbol", "score"}
    missing_required = [column for column in sorted(required_columns) if column not in work.columns]
    if missing_required:
        return None, dbc.Alert(
            f"Candidates missing required columns from {source_label}: {', '.join(missing_required)}",
            color="warning",
        )

    for column in canonical_header:
        if column not in work.columns:
            work[column] = pd.NA
    work = work[canonical_header]
    logger.info(
        "Loaded latest candidates source=%s updated=%s rows=%s",
        source_label,
        updated,
        len(work.index),
    )
    return work, None


def _safe_csv_with_message(path: Path, *, required: Optional[list[str]] = None):
    required = required or []
    if not path.exists():
        return pd.DataFrame(), dbc.Alert(
            f"No data available yet: {path.name} is missing.", color="info", className="mb-2"
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(), dbc.Alert(
            f"Unable to read {path.name}: {exc}", color="danger", className="mb-2"
        )
    missing = [col for col in required if col not in df.columns]
    if missing:
        return pd.DataFrame(), dbc.Alert(
            f"Missing required columns in {path.name}: {', '.join(missing)}",
            color="warning",
            className="mb-2",
        )
    return df, None


def load_execute_metrics() -> tuple[dict[str, Any], dbc.Alert | None, list[dict[str, Any]]]:
    """Load executor metrics with skip-reason normalization."""

    defaults: dict[str, Any] = {
        "last_run_utc": None,
        "orders_submitted": 0,
        "orders_filled": 0,
        "orders_canceled": 0,
        "trailing_attached": 0,
        "api_retries": 0,
        "api_failures": 0,
        "latency_secs": {"p50": 0.0, "p95": 0.0},
        "skip_reasons": {},
    }
    alert: dbc.Alert | None = None
    skip_rows: list[dict[str, Any]] = []

    if not os.path.exists(execute_metrics_path):
        return (
            defaults,
            dbc.Alert(
                "Execution has not produced metrics yet (execute_metrics.json missing).",
                color="info",
                className="mb-2",
            ),
            skip_rows,
        )

    try:
        with open(execute_metrics_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return (
            defaults,
            dbc.Alert(
                f"Unable to read execute_metrics.json: {exc}", color="danger", className="mb-2"
            ),
            skip_rows,
        )

    if isinstance(payload, dict):
        metrics = defaults | payload
    else:
        metrics = defaults
        alert = dbc.Alert(
            "execute_metrics.json contained an unexpected format.",
            color="warning",
            className="mb-2",
        )

    latency = metrics.get("latency_secs") or {}
    metrics["latency_secs"] = {
        "p50": (latency.get("p50") or 0),
        "p95": (latency.get("p95") or 0),
    }

    raw_skip = (
        metrics.get("skip_reasons")
        or metrics.get("skip_reason_counts")
        or metrics.get("skip_reason")
        or {}
    )
    if isinstance(raw_skip, list):
        skip_dict = {str(entry): 1 for entry in raw_skip}
    elif isinstance(raw_skip, dict):
        skip_dict = {str(k): _coerce_int_value(v) for k, v in raw_skip.items()}
    else:
        skip_dict = {}
    skip_rows = [{"reason": reason, "count": count} for reason, count in sorted(skip_dict.items())]
    metrics["skip_reasons"] = skip_dict

    return metrics, alert, skip_rows


def load_open_positions() -> tuple[pd.DataFrame, dbc.Alert | None]:
    return _safe_csv_with_message(Path(open_positions_path), required=["symbol", "qty"])


def load_executed_trades() -> tuple[pd.DataFrame, dbc.Alert | None]:
    return _safe_csv_with_message(Path(executed_trades_path))


_FILLS_RECENT_SQL = """
SELECT activity_type, transaction_time, symbol, side, qty, price, order_id
FROM v_fills_recent
WHERE transaction_time >= now() - interval '7 days'
ORDER BY transaction_time DESC
LIMIT 200
"""

_TRADE_PNL_RECENT_SQL = """
SELECT exit_time, symbol, trade_side, profit_loss_usd
FROM v_trade_pnl
ORDER BY exit_time DESC
LIMIT 50;
"""

_TRADE_PNL_DAILY_SQL = """
SELECT
  (exit_time AT TIME ZONE 'America/New_York')::date AS trade_day,
  SUM(profit_loss_usd) AS daily_pnl
FROM v_trade_pnl
WHERE exit_time >= now() - interval '30 days'
GROUP BY trade_day
ORDER BY trade_day;
"""

_TRADE_PNL_KPI_SQL = """
SELECT
  COUNT(*) AS trades,
  AVG((profit_loss_usd > 0)::int)::numeric AS win_rate,
  AVG(CASE WHEN profit_loss_usd > 0 THEN profit_loss_usd END) AS avg_win,
  AVG(CASE WHEN profit_loss_usd < 0 THEN profit_loss_usd END) AS avg_loss,
  (
    AVG(CASE WHEN profit_loss_usd > 0 THEN profit_loss_usd END)
    * AVG((profit_loss_usd > 0)::int)::numeric
  ) +
  (
    AVG(CASE WHEN profit_loss_usd < 0 THEN profit_loss_usd END)
    * (1 - AVG((profit_loss_usd > 0)::int)::numeric)
  ) AS expectancy
FROM v_trade_pnl
WHERE exit_time >= now() - interval '30 days';
"""


def load_recent_fills() -> tuple[pd.DataFrame, list[dbc.Alert]]:
    df, alert = db_query_df(_FILLS_RECENT_SQL)
    alerts = [alert] if alert else []
    if df is None:
        return pd.DataFrame(), alerts

    frame = df.copy()
    if "transaction_time" in frame.columns:
        frame["transaction_time"] = pd.to_datetime(
            frame["transaction_time"], utc=True, errors="coerce"
        )
        frame = frame.dropna(subset=["transaction_time"])
        frame = frame.sort_values("transaction_time", ascending=False)
    for col in ("qty", "price"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if "side" in frame.columns:
        frame["side"] = frame["side"].astype(str).str.upper()
    if "order_id" in frame.columns:
        frame["order_id"] = frame["order_id"].astype(str)
    if "order_id" in frame.columns and "order_id_short" not in frame.columns:
        frame["order_id_short"] = frame["order_id"].str.slice(0, 8)
    return frame, alerts


def _load_recent_trade_pnl() -> tuple[pd.DataFrame | None, list[dbc.Alert]]:
    df, alert = db_query_df(_TRADE_PNL_RECENT_SQL)
    alerts = [alert] if alert else []
    if df is None:
        return None, alerts

    frame = df.copy()
    if frame.empty:
        return frame, alerts

    if "exit_time" in frame.columns:
        frame["exit_time"] = pd.to_datetime(frame["exit_time"], errors="coerce", utc=True)
        frame = frame.dropna(subset=["exit_time"])
        frame = frame.sort_values("exit_time", ascending=False)
        frame["exit_time"] = frame["exit_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if "trade_side" in frame.columns:
        frame["trade_side"] = frame["trade_side"].astype(str).str.upper()
    if "profit_loss_usd" in frame.columns:
        frame["profit_loss_usd"] = pd.to_numeric(frame["profit_loss_usd"], errors="coerce")

    return frame, alerts


def _load_daily_trade_pnl() -> tuple[pd.DataFrame | None, list[dbc.Alert]]:
    df, alert = db_query_df(_TRADE_PNL_DAILY_SQL)
    alerts = [alert] if alert else []
    if df is None:
        return None, alerts

    frame = df.copy()
    if frame.empty:
        return frame, alerts

    if "trade_day" in frame.columns:
        frame["trade_day"] = pd.to_datetime(frame["trade_day"], errors="coerce").dt.date
    if "daily_pnl" in frame.columns:
        frame["daily_pnl"] = pd.to_numeric(frame["daily_pnl"], errors="coerce")

    return frame, alerts


def _load_trade_pnl_kpis() -> tuple[dict[str, Any] | None, list[dbc.Alert]]:
    df, alert = db_query_df(_TRADE_PNL_KPI_SQL)
    alerts = [alert] if alert else []
    if df is None:
        return None, alerts

    defaults: dict[str, Any] = {
        "trades": 0,
        "win_rate": 0.0,
        "avg_win": None,
        "avg_loss": None,
        "expectancy": None,
    }

    if df.empty:
        return defaults, alerts

    row = df.iloc[0]

    def _safe_float(value: Any) -> float | None:
        try:
            result = float(value)
            if math.isnan(result):
                return None
            return result
        except (TypeError, ValueError):
            return None

    def _safe_int(value: Any) -> int:
        try:
            if value is None:
                return 0
            if isinstance(value, float) and math.isnan(value):
                return 0
            return int(value)
        except (TypeError, ValueError):
            return 0

    metrics = {
        "trades": _safe_int(row.get("trades")),
        "win_rate": _safe_float(row.get("win_rate")) or 0.0,
        "avg_win": _safe_float(row.get("avg_win")),
        "avg_loss": _safe_float(row.get("avg_loss")),
        "expectancy": _safe_float(row.get("expectancy")),
    }

    return metrics, alerts


_ACCOUNT_LATEST_SQL = """
SELECT taken_at, account_id, status, equity, cash, buying_power, portfolio_value
FROM v_account_latest
LIMIT 1
"""

_ACCOUNT_LATEST_SQL_NO_PORTFOLIO = """
SELECT taken_at, account_id, status, equity, cash, buying_power
FROM v_account_latest
LIMIT 1
"""

_ACCOUNT_SERIES_SQL = """
SELECT taken_at, equity, cash, buying_power
FROM alpaca_account_snapshots
WHERE taken_at >= (NOW() AT TIME ZONE 'utc') - INTERVAL '7 days'
ORDER BY taken_at DESC
LIMIT 500
"""

_ACCOUNT_RECENT_SQL = """
SELECT taken_at, equity, cash, buying_power
FROM alpaca_account_snapshots
ORDER BY taken_at DESC
LIMIT 25
"""


def _account_query_with_fallback(
    primary_sql: str, fallback_sql: str | None = None
) -> tuple[pd.DataFrame, list[dbc.Alert]]:
    alerts: list[dbc.Alert] = []
    df, alert = db_query_df(primary_sql)
    if alert:
        alerts.append(alert)
    if df is not None:
        return df, alerts
    if fallback_sql:
        fallback_df, fallback_alert = db_query_df(fallback_sql)
        if fallback_alert:
            alerts.append(fallback_alert)
        if fallback_df is not None:
            alerts.append(
                dbc.Alert(
                    "Using alpaca_account_snapshots as fallback for account history.",
                    color="info",
                    className="mb-3",
                )
            )
            return fallback_df, alerts
    return pd.DataFrame(), alerts


def _account_missing_columns_alert(context: str, missing: set[str]) -> dbc.Alert:
    missing_list = ", ".join(sorted(missing))
    logger.warning("[WARN] ACCOUNT_MISSING_COLUMNS context=%s missing=%s", context, sorted(missing))
    return dbc.Alert(
        f"Account data is missing expected column(s) ({missing_list}) for {context}.",
        color="warning",
        className="mb-3",
    )


def _normalize_account_frame(
    df: pd.DataFrame | None, *, context: str, required: set[str]
) -> tuple[pd.DataFrame, list[dbc.Alert]]:
    alerts: list[dbc.Alert] = []
    if df is None or df.empty:
        return pd.DataFrame(), alerts

    frame = df.copy()
    missing = {col for col in required if col not in frame.columns}
    if missing:
        alerts.append(_account_missing_columns_alert(context, missing))
        return pd.DataFrame(), alerts

    frame["taken_at"] = pd.to_datetime(frame["taken_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["taken_at"])

    numeric_cols = ["equity", "cash", "buying_power", "portfolio_value"]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "status" in frame.columns:
        frame["status"] = frame["status"].astype(str)
    return frame, alerts


def _load_account_latest() -> tuple[pd.DataFrame, list[dbc.Alert]]:
    latest_df, alerts = _account_query_with_fallback(_ACCOUNT_LATEST_SQL)
    normalized, normalize_alerts = _normalize_account_frame(
        latest_df,
        context="latest account snapshot",
        required={"taken_at", "equity", "cash", "buying_power"},
    )
    alerts.extend(normalize_alerts)

    if normalized.empty and latest_df is None:
        fallback_df, fallback_alerts = _account_query_with_fallback(
            _ACCOUNT_LATEST_SQL_NO_PORTFOLIO
        )
        alerts.extend(fallback_alerts)
        normalized, normalize_alerts = _normalize_account_frame(
            fallback_df,
            context="latest account snapshot",
            required={"taken_at", "equity", "cash", "buying_power"},
        )
        alerts.extend(normalize_alerts)

    return normalized.head(1), alerts


def _load_account_series() -> tuple[pd.DataFrame, list[dbc.Alert]]:
    series_df, alerts = _account_query_with_fallback(_ACCOUNT_SERIES_SQL)
    normalized, normalize_alerts = _normalize_account_frame(
        series_df,
        context="account history",
        required={"taken_at", "equity", "cash", "buying_power"},
    )
    alerts.extend(normalize_alerts)
    normalized = normalized.sort_values("taken_at")
    return normalized, alerts


def _load_account_recent() -> tuple[pd.DataFrame, list[dbc.Alert]]:
    recent_df, alert = db_query_df(_ACCOUNT_RECENT_SQL)
    alerts = [alert] if alert else []
    normalized, normalize_alerts = _normalize_account_frame(
        recent_df,
        context="recent account snapshots",
        required={"taken_at", "equity", "cash", "buying_power"},
    )
    alerts.extend(normalize_alerts)
    return normalized, alerts


def _format_currency(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"${numeric:,.2f}"


def _timestamp_badges(taken_at: pd.Timestamp | None) -> list[dbc.Badge]:
    if taken_at is None or not isinstance(taken_at, pd.Timestamp):
        return []
    ny_tz = pytz.timezone("America/New_York")
    ny_time = taken_at.tz_convert(ny_tz)
    return [
        dbc.Badge(
            f"UTC: {taken_at.strftime('%Y-%m-%d %H:%M:%S')}", color="secondary", className="me-2"
        ),
        dbc.Badge(f"NY: {ny_time.strftime('%Y-%m-%d %H:%M:%S')}", color="info"),
    ]


def _account_kpi_card(title: str, value: Any, color: str = "secondary") -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(title),
                dbc.CardBody(html.H4(_format_currency(value), className="card-title")),
            ],
            className="mb-3",
            color=color,
            inverse=False,
        ),
        md=2,
    )


def _account_status_badge(status: str | None) -> dbc.Badge | None:
    if not status:
        return None
    status_text = str(status).upper()
    color = "success" if status_text == "ACTIVE" else "warning"
    return dbc.Badge(f"Status: {status_text}", color=color, className="me-2")


def _account_timeseries_fig(df: pd.DataFrame, y: str, title: str) -> dcc.Graph:
    if df.empty or y not in df.columns:
        return dcc.Graph(
            figure=go.Figure(layout=go.Layout(template="plotly_dark", title=f"No {title} data"))
        )
    fig = px.line(df, x="taken_at", y=y, title=title, template="plotly_dark")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return dcc.Graph(figure=fig)


def _account_drawdown_fig(df: pd.DataFrame) -> dcc.Graph:
    if df.empty or "equity" not in df.columns:
        return dcc.Graph(
            figure=go.Figure(layout=go.Layout(template="plotly_dark", title="Drawdown (equity)"))
        )
    equity = df["equity"].fillna(method="ffill")
    running_max = equity.cummax()
    with pd.option_context("mode.use_inf_as_na", True):
        drawdown = (equity - running_max) / running_max * 100
    fig = px.area(
        pd.DataFrame({"taken_at": df["taken_at"], "drawdown_pct": drawdown}),
        x="taken_at",
        y="drawdown_pct",
        title="Drawdown (from equity)",
        template="plotly_dark",
    )
    fig.update_yaxes(ticksuffix="%")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return dcc.Graph(figure=fig)


def _account_table(df: pd.DataFrame) -> dash_table.DataTable:
    required = {"taken_at", "equity", "cash", "buying_power"}
    if df is None or df.empty or not required.issubset(df.columns):
        return dbc.Alert("No recent account snapshots to display.", color="info", className="mb-3")

    display_df = df.copy()
    display_df["taken_at"] = display_df["taken_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    columns = [
        {"name": "Taken At (UTC)", "id": "taken_at"},
        {
            "name": "Equity",
            "id": "equity",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "Cash",
            "id": "cash",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
        {
            "name": "Buying Power",
            "id": "buying_power",
            "type": "numeric",
            "format": Format(precision=2, scheme=Scheme.fixed),
        },
    ]
    if "status" in display_df.columns:
        columns.append({"name": "Status", "id": "status"})
    return dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=columns,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#343a40", "color": "#fff"},
        style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
    )


account_table = _account_table


def render_account_tab() -> dbc.Container:
    try:
        alerts: list[dbc.Alert] = []
        latest_df, latest_alerts = _load_account_latest()
        series_df, series_alerts = _load_account_series()
        recent_df, recent_alerts = _load_account_recent()

        alerts.extend(latest_alerts)
        alerts.extend(series_alerts)
        alerts.extend(recent_alerts)

        if alerts:
            deduped = []
            seen: set[str] = set()
            for alert in alerts:
                content = getattr(alert, "children", "")
                key = str(content)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(alert)
            alerts = deduped

        if series_df.empty and recent_df.empty and latest_df.empty:
            empty_state = dbc.Alert(
                "No account snapshots yet. Run python -m scripts.fetch_account_snapshot or wait for the scheduled task.",
                color="info",
                className="mb-3",
            )
            return dbc.Container(alerts + [empty_state], fluid=True)

        latest_row = latest_df.iloc[0] if not latest_df.empty else None
        taken_at = latest_row["taken_at"] if latest_row is not None else None

        kpi_cards: list[dbc.Col] = []
        if latest_row is not None:
            kpi_cards.extend(
                [
                    _account_kpi_card("Equity", latest_row.get("equity"), color="primary"),
                    _account_kpi_card("Cash", latest_row.get("cash"), color="info"),
                    _account_kpi_card(
                        "Buying Power", latest_row.get("buying_power"), color="secondary"
                    ),
                ]
            )
            if "portfolio_value" in latest_row and pd.notna(latest_row["portfolio_value"]):
                kpi_cards.append(
                    _account_kpi_card(
                        "Portfolio Value", latest_row.get("portfolio_value"), color="success"
                    )
                )

        status_badge = _account_status_badge(
            latest_row.get("status") if latest_row is not None else None
        )
        timestamp_badges = _timestamp_badges(
            taken_at if isinstance(taken_at, pd.Timestamp) else None
        )

        series_section: list[Any] = []
        if not series_df.empty:
            logger.info("[INFO] Account tab loaded %s points", len(series_df))
            series_section = [
                dbc.Row(
                    [
                        dbc.Col(
                            _account_timeseries_fig(series_df, "equity", "Equity (last 7 days)"),
                            md=4,
                        ),
                        dbc.Col(
                            _account_timeseries_fig(series_df, "cash", "Cash (last 7 days)"), md=4
                        ),
                        dbc.Col(
                            _account_timeseries_fig(
                                series_df, "buying_power", "Buying Power (last 7 days)"
                            ),
                            md=4,
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row([dbc.Col(_account_drawdown_fig(series_df), md=12)], className="mb-3"),
            ]

        table_section = []
        if not recent_df.empty:
            table_section.append(html.H5("Recent Snapshots (latest 25)", className="mt-3"))
            table_section.append(_account_table(recent_df))

        header = dbc.Row(
            [
                dbc.Col(html.H4("Account Overview", className="mb-2 text-light"), width="auto"),
                dbc.Col(
                    html.Div(
                        [badge for badge in [status_badge, *timestamp_badges] if badge],
                        className="d-flex align-items-center",
                    ),
                    width=True,
                ),
            ],
            className="mb-2",
        )

        kpi_row = dbc.Row(kpi_cards, className="mb-2") if kpi_cards else html.Div()

        return dbc.Container(
            alerts
            + [
                header,
                kpi_row,
                *series_section,
                *table_section,
            ],
            fluid=True,
        )
    except Exception as exc:  # pragma: no cover - defensive guard for runtime issues
        logger.exception("Failed to render account tab")
        message = f"Account tab failed: {type(exc).__name__}: {exc}"
        return dbc.Container(
            [dbc.Alert(message, color="danger", className="mb-3")],
            fluid=True,
        )


def load_last_premarket_run() -> tuple[dict[str, Any], dbc.Alert | None]:
    marker = Path(BASE_DIR) / "data" / "last_premarket_run.json"
    if not marker.exists():
        return {}, dbc.Alert(
            "No pre-market marker found yet (last_premarket_run.json).", color="info"
        )
    try:
        with marker.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, dict):
                return payload, None
    except Exception as exc:
        return {}, dbc.Alert(f"Unable to read last_premarket_run.json: {exc}", color="danger")
    return {}, dbc.Alert("Malformed last_premarket_run.json payload.", color="warning")


def load_top_or_latest_candidates(required_columns: Optional[set[str] | list[str]] = None):
    """Prefer post-metrics top_candidates.csv with graceful fallback."""

    if required_columns is None:
        req = {"symbol", "score"}
    else:
        req = set(required_columns)

    df, updated, source_file = screener_table()
    if df is None or df.empty:
        return None, dbc.Alert(
            "No candidates available yet (top or latest).",
            color="info",
        )

    missing = [col for col in sorted(req) if col not in df.columns]
    if missing:
        return None, dbc.Alert(
            f"Missing required columns: {missing}",
            color="danger",
        )

    work = df.copy()
    work["__source"] = source_file
    work["__updated"] = updated
    return work, None


def load_symbol_perf_df() -> pd.DataFrame | dbc.Alert:
    """Return the trades dataframe (or alert) for the Symbol Performance tab."""

    df, alert = load_csv(
        str(TRADES_LOG_FOR_SYMBOLS),
        ["symbol"],
        alert_prefix="Symbol Performance",
    )
    if alert:
        return alert
    return df


def _compute_trade_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a trades dataframe with hold_days/return_pct convenience columns."""

    normalized = _normalize_pnl(df)
    if normalized is None or normalized.empty:
        return pd.DataFrame()

    work = normalized.copy()
    if "status" in work.columns:
        work["status"] = work["status"].astype(str).str.upper()
    entry_times = pd.to_datetime(work.get("entry_time"), utc=True, errors="coerce")
    exit_times = pd.to_datetime(work.get("exit_time"), utc=True, errors="coerce")
    now = pd.Timestamp.utcnow()
    if "entry_time" in work.columns:
        effective_exit = (
            exit_times.fillna(now)
            if "exit_time" in work.columns
            else pd.Series(now, index=work.index)
        )
        work["hold_days"] = (effective_exit - entry_times).dt.days
        work["entry_time"] = entry_times
        if "exit_time" in work.columns:
            work["exit_time"] = exit_times

    if "realized_pnl" in work.columns and "net_pnl" not in work.columns:
        work["net_pnl"] = work["realized_pnl"]
    if "pnl" not in work.columns and "net_pnl" in work.columns:
        work["pnl"] = work["net_pnl"]

    if {"entry_price", "exit_price"}.issubset(work.columns):
        entry_price = pd.to_numeric(work["entry_price"], errors="coerce")
        exit_price = pd.to_numeric(work["exit_price"], errors="coerce")
        with pd.option_context("mode.use_inf_as_na", True):
            return_pct = ((exit_price - entry_price) / entry_price) * 100
        work["return_pct"] = return_pct
        if "exit_pct" not in work.columns:
            work["exit_pct"] = return_pct

    return work


def load_trades_for_exits() -> tuple[pd.DataFrame, pd.DataFrame, dbc.Alert | None, str]:
    """Load trades for exit analytics with DB-first fallback to CSV."""

    db_enabled = db.db_enabled()
    db_ready = db.get_engine() is not None if db_enabled else False

    db_trades = load_trades_db()
    open_db_trades = load_open_trades_db() if not db_trades.empty else pd.DataFrame()
    if not db_trades.empty:
        recent_df = _compute_trade_columns(db_trades)
        open_df = _compute_trade_columns(open_db_trades)
        if open_df.empty and not recent_df.empty and "status" in recent_df.columns:
            open_df = recent_df[recent_df["status"] == "OPEN"].copy()
        if "entry_time" in open_df.columns:
            open_df.sort_values("entry_time", ascending=False, inplace=True)
        if "exit_time" in recent_df.columns:
            recent_df.sort_values(by="exit_time", ascending=False, na_position="last", inplace=True)
        return recent_df, open_df, None, "db"

    csv_df, alert = load_csv(
        str(TRADES_LOG_PATH),
        required_columns=["symbol", "entry_time"],
        alert_prefix="Trades log",
    )
    if alert:
        if db_enabled and db_ready is False:
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                dbc.Alert("Trades unavailable.", color="warning"),
                "unavailable",
            )
        if not db_enabled:
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                dbc.Alert("No trades yet (paper).", color="info"),
                "csv",
            )
        return pd.DataFrame(), pd.DataFrame(), alert, "unavailable"

    if not isinstance(csv_df, pd.DataFrame) or csv_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            dbc.Alert("No trades yet (paper).", color="info"),
            "csv",
        )

    csv_df = _compute_trade_columns(csv_df)
    open_df = pd.DataFrame()
    if "status" in csv_df.columns:
        open_df = csv_df[csv_df["status"] == "OPEN"].copy()
    elif "exit_time" in csv_df.columns:
        open_df = csv_df[csv_df["exit_time"].isna()].copy()

    if "entry_time" in open_df.columns:
        open_df.sort_values("entry_time", ascending=False, inplace=True)
    sorted_csv = (
        csv_df.sort_values(by="exit_time", ascending=False, na_position="last")
        if "exit_time" in csv_df.columns
        else csv_df
    )

    return sorted_csv, open_df, None, "csv"


def make_trades_exits_layout():
    """Build the Trades / Exits analytics layout."""

    trades_df, open_trades_df, alert, source_label = load_trades_for_exits()

    if alert:
        return html.Div([alert])

    if trades_df.empty and open_trades_df.empty:
        return dbc.Alert("No trades yet (paper).", color="info", className="m-2")

    has_exit_reason = "exit_reason" in trades_df.columns
    has_exit_eff = "exit_efficiency" in trades_df.columns

    open_table: Any
    if not open_trades_df.empty:
        open_columns = [
            col
            for col in (
                "symbol",
                "qty",
                "entry_time",
                "entry_price",
                "hold_days",
                "return_pct",
                "status",
            )
            if col in open_trades_df.columns
        ]
        if not open_columns:
            open_columns = list(open_trades_df.columns)
        open_table = _styled_table(
            open_trades_df[open_columns],
            table_id="open-trades-table",
            page_size=10,
        )
    else:
        open_table = dbc.Alert("No trades yet (paper).", color="info")

    preferred_recent_columns = [
        "symbol",
        "status",
        "entry_time",
        "exit_time",
        "qty",
        "entry_price",
        "exit_price",
        "return_pct",
        "net_pnl",
        "exit_reason",
        "hold_days",
    ]
    recent_columns = [col for col in preferred_recent_columns if col in trades_df.columns] or list(
        trades_df.columns
    )
    recent_table = _styled_table(
        trades_df[recent_columns],
        table_id="recent-trades-table",
        page_size=20,
    )

    summary_cards: list[Any] = []
    if not trades_df.empty:
        pnl_col = "net_pnl" if "net_pnl" in trades_df.columns else "pnl"
        total_trades = len(trades_df)
        open_count = len(open_trades_df) if not open_trades_df.empty else 0
        if "status" in trades_df.columns:
            closed_count = len(trades_df[trades_df["status"] == "CLOSED"])
        else:
            closed_count = max(0, total_trades - open_count)
        realized_pnl = float(trades_df[pnl_col].sum()) if pnl_col in trades_df.columns else 0.0
        avg_hold = (
            float(pd.to_numeric(trades_df["hold_days"], errors="coerce").mean())
            if "hold_days" in trades_df.columns
            else 0.0
        )
        kpi_data = [
            ("Trades", total_trades),
            ("Open", open_count),
            ("Closed", closed_count),
            ("Realized PnL", f"${realized_pnl:,.2f}"),
            ("Avg Hold (days)", f"{avg_hold:.1f}"),
        ]
        for label, value in kpi_data:
            summary_cards.append(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(label, className="text-muted small"),
                                html.H5(value, className="mb-0"),
                            ]
                        ),
                        className="h-100",
                    ),
                    md=2,
                    sm=4,
                    xs=6,
                )
            )

    table_columns = [
        {"name": "Symbol", "id": "symbol"},
        {"name": "Entry Time", "id": "entry_time"},
        {"name": "Exit Time", "id": "exit_time"},
        {"name": "Exit %", "id": "exit_pct"} if "exit_pct" in trades_df.columns else None,
        {"name": "Net PnL", "id": "net_pnl"} if "net_pnl" in trades_df.columns else None,
    ]
    if has_exit_reason:
        table_columns.append({"name": "Exit Reason", "id": "exit_reason"})
    if has_exit_eff:
        table_columns.append({"name": "Exit Efficiency", "id": "exit_efficiency"})

    table_columns = [column for column in table_columns if column is not None]

    trades_table = dash_table.DataTable(
        id="trades-exits-table",
        columns=table_columns,
        data=trades_df.to_dict("records"),
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
    )

    if has_exit_reason:
        exit_counts = trades_df["exit_reason"].value_counts().reset_index()
        exit_counts.columns = ["exit_reason", "count"]
        counts_bar = dcc.Graph(
            id="exit-reason-counts",
            figure={
                "data": [
                    {
                        "type": "bar",
                        "x": exit_counts["exit_reason"],
                        "y": exit_counts["count"],
                    }
                ],
                "layout": {
                    "title": "Exit count by reason",
                    "margin": {"l": 40, "r": 10, "t": 40, "b": 80},
                },
            },
        )
    else:
        counts_bar = html.Div("No exit_reason column available yet.")

    if has_exit_reason and has_exit_eff:
        eff_by_reason = (
            trades_df.groupby("exit_reason")["exit_efficiency"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        eff_bar = dcc.Graph(
            id="exit-efficiency-by-reason",
            figure={
                "data": [
                    {
                        "type": "bar",
                        "x": eff_by_reason["exit_reason"],
                        "y": eff_by_reason["exit_efficiency"],
                    }
                ],
                "layout": {
                    "title": "Avg exit efficiency by reason",
                    "margin": {"l": 40, "r": 10, "t": 40, "b": 80},
                    "yaxis": {"range": [0.9, 1.0]},
                },
            },
        )
    else:
        eff_bar = html.Div("No exit_efficiency data available yet.")

    def _trade_pnl_section() -> Any:
        kpi_metrics, kpi_alerts = _load_trade_pnl_kpis()
        table_df, table_alerts = _load_recent_trade_pnl()
        daily_df, daily_alerts = _load_daily_trade_pnl()
        alerts = kpi_alerts + table_alerts + daily_alerts

        if table_df is None and daily_df is None and alerts:
            return html.Div(alerts)

        def _format_pct(value: Any) -> str:
            try:
                return f"{float(value) * 100:.1f}%"
            except Exception:
                return "0.0%"

        def _format_currency(value: Any) -> str:
            try:
                number = float(value)
                if math.isnan(number):
                    raise ValueError
                return f"${number:,.2f}"
            except Exception:
                return "N/A"

        def _trade_pnl_kpi_cards(metrics: Mapping[str, Any] | None) -> Any:
            if metrics is None:
                return html.Div()

            trades = metrics.get("trades", 0)
            win_rate = metrics.get("win_rate", 0.0)
            avg_win = metrics.get("avg_win")
            avg_loss = metrics.get("avg_loss")
            expectancy = metrics.get("expectancy")

            card_data = [
                ("Trades", f"{_coerce_int_value(trades)}"),
                ("Win rate", _format_pct(win_rate)),
                ("Avg win", _format_currency(avg_win)),
                ("Avg loss", _format_currency(avg_loss)),
                ("Expectancy", _format_currency(expectancy)),
            ]

            return html.Div(
                [
                    html.H5("Last 30 Days", className="mb-3"),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader(label, className="fw-semibold"),
                                        dbc.CardBody(
                                            html.H4(value, className="card-title mb-0"),
                                            className="py-3",
                                        ),
                                    ],
                                    className="h-100",
                                    color="dark",
                                    outline=True,
                                    style={"borderColor": "var(--bs-primary)"},
                                ),
                                md=2,
                                sm=6,
                                className="mb-3",
                            )
                            for label, value in card_data
                        ],
                        className="g-3",
                        justify="start",
                    ),
                ],
                className="mb-4",
            )

        if table_df is None or table_df.empty:
            table_component: Any = dbc.Alert("No completed trades yet.", color="info")
        else:
            table_component = dash_table.DataTable(
                id="trade-pnl-recent-table",
                columns=[
                    {"name": "Exit Time", "id": "exit_time"},
                    {"name": "Symbol", "id": "symbol"},
                    {"name": "Side", "id": "trade_side"},
                    {
                        "name": "Profit / Loss (USD)",
                        "id": "profit_loss_usd",
                        "type": "numeric",
                        "format": Format(precision=2, scheme=Scheme.fixed),
                    },
                ],
                data=table_df.to_dict("records"),
                sort_action="native",
                sort_by=[{"column_id": "exit_time", "direction": "desc"}],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#1e1e1e", "color": "#fff"},
                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": "{profit_loss_usd} > 0",
                            "column_id": "profit_loss_usd",
                        },
                        "color": "#28a745",
                    },
                    {
                        "if": {
                            "filter_query": "{profit_loss_usd} < 0",
                            "column_id": "profit_loss_usd",
                        },
                        "color": "#dc3545",
                    },
                ],
            )

        if daily_df is None or daily_df.empty:
            chart_component: Any = dbc.Alert("No completed trades yet.", color="info")
        else:
            colors = [
                "#28a745" if (val or 0) >= 0 else "#dc3545"
                for val in daily_df["daily_pnl"].tolist()
            ]
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=daily_df["trade_day"],
                        y=daily_df["daily_pnl"],
                        marker_color=colors,
                    )
                ]
            )
            fig.update_layout(
                title="30-Day Daily P&L",
                xaxis_title="Trade Day (ET)",
                yaxis_title="Daily P&L (USD)",
                template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=40),
            )
            chart_component = dcc.Graph(figure=fig, id="trade-pnl-daily-chart")

        return dbc.Card(
            [
                dbc.CardHeader("Trade Performance (Realized P/L)", className="fw-bold"),
                dbc.CardBody(
                    [
                        html.Div(alerts),
                        _trade_pnl_kpi_cards(kpi_metrics),
                        dbc.Row(
                            [
                                dbc.Col(table_component, md=5, className="mb-3"),
                                dbc.Col(chart_component, md=7, className="mb-3"),
                            ],
                            className="g-3",
                        ),
                    ]
                ),
            ],
            className="bg-dark text-light mb-4",
        )

    return html.Div(
        [
            html.H3("Trades & Exit Analytics"),
            html.P(
                "Review how well each exit rule captures profit, using exit_reason and exit_efficiency from trades_log.csv."
            ),
            (dbc.Row(summary_cards, className="g-3 mb-3") if summary_cards else html.Div()),
            _trade_pnl_section(),
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.Span("Open Trades"),
                            dbc.Badge(source_label.upper(), color="secondary", className="ms-2"),
                        ],
                        className="d-flex justify-content-between align-items-center",
                    ),
                    dbc.CardBody(open_table),
                ],
                className="mb-3 bg-dark text-light",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Recent Trades"),
                    dbc.CardBody(recent_table),
                ],
                className="mb-4 bg-dark text-light",
            ),
            html.Div(trades_table, style={"marginBottom": "2rem"}),
            html.Div(
                [
                    html.Div(counts_bar, style={"flex": 1, "minWidth": "300px"}),
                    html.Div(eff_bar, style={"flex": 1, "minWidth": "300px"}),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "1rem"},
            ),
        ]
    )


def _load_trade_perf_cache() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "summary": {},
        "trades": [],
        "written_at": None,
        "trades_total": 0,
    }
    try:
        with open(TRADE_PERFORMANCE_CACHE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        payload = {}
    if not isinstance(payload, Mapping):
        payload = {}
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    trades = payload.get("trades", [])
    if not isinstance(trades, list):
        trades = []
    written_at = payload.get("written_at")
    trades_total = payload.get("trades_total", len(trades))
    try:
        trades_total = int(trades_total)
    except Exception:
        trades_total = len(trades)
    defaults.update(
        {
            "summary": summary,
            "trades": trades,
            "written_at": written_at,
            "trades_total": trades_total,
        }
    )
    return defaults


def _build_trade_perf_kpis(
    summary: Mapping[str, Mapping[str, Any]], windows: tuple[str, ...] = ("30D", "ALL")
) -> list[Any]:
    def _metric(label: str, value: str) -> html.Div:
        return html.Div(
            [
                html.Span(label, className="text-muted me-2"),
                html.Strong(value),
            ],
            className="d-flex justify-content-between",
        )

    def _format_pct(value: Any) -> str:
        try:
            return f"{float(value):.1f}%"
        except Exception:
            return "0.0%"

    cards: list[Any] = []
    for window in windows:
        metrics = summary.get(window, {}) if isinstance(summary, Mapping) else {}
        trades = int(metrics.get("trades", 0) or 0)
        total_pnl = float(metrics.get("net_pnl", 0.0) or 0.0)
        win_rate = metrics.get("win_rate_pct", metrics.get("win_rate", 0.0))
        avg_return = metrics.get("avg_return_pct", 0.0)
        stop_exits = _coerce_int_value(metrics.get("stop_exits", 0))
        rebounds = _coerce_int_value(metrics.get("rebounds", 0))
        rebound_rate = float(metrics.get("rebound_rate", 0.0) or 0.0) * 100.0
        avg_rebound_pct = float(metrics.get("avg_rebound_pct", 0.0) or 0.0)
        cards.append(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(f"{window} KPIs"),
                        dbc.CardBody(
                            [
                                _metric("Trades", f"{trades}"),
                                _metric("Net P&L", f"{total_pnl:,.2f}"),
                                _metric("Win rate", _format_pct(win_rate)),
                                _metric("Avg return", _format_pct(avg_return)),
                                _metric("Stop exits", f"{stop_exits}"),
                                _metric("Rebounds", f"{rebounds}"),
                                _metric("Rebound rate", _format_pct(rebound_rate)),
                                _metric("Avg rebound %", f"{avg_rebound_pct:.2f}%"),
                            ]
                        ),
                    ],
                    color="dark",
                    outline=True,
                    style={"borderColor": "var(--bs-info)"},
                ),
                md=3,
                className="mb-3",
            )
        )
    return cards


def _build_window_net_pnl_bar(summary: Mapping[str, Mapping[str, Any]]) -> go.Figure:
    windows = []
    pnl_values = []
    for window in ("7D", "30D", "365D", "ALL"):
        windows.append(window)
        metrics = summary.get(window, {}) if isinstance(summary, Mapping) else {}
        pnl_values.append(float(metrics.get("net_pnl", 0.0) or 0.0))
    fig = px.bar(
        x=windows,
        y=pnl_values,
        labels={"x": "Window", "y": "Net P&L"},
        title="Net P&L by window",
    )
    fig.update_layout(template="plotly_dark")
    return fig


def render_trade_performance_panel() -> html.Div:
    payload = _load_trade_perf_cache()
    store_data = {
        "trades": payload.get("trades", []),
        "summary": payload.get("summary", {}),
        "written_at": payload.get("written_at"),
        "trades_total": payload.get("trades_total", 0),
    }
    status_badges: list[Any] = []
    if store_data.get("written_at"):
        status_badges.append(
            dbc.Badge(
                f"Cache updated {store_data['written_at']}",
                color="secondary",
                className="me-2",
            )
        )
    if store_data.get("summary"):
        metrics = store_data["summary"].get("ALL", {}) or {}
        status_badges.append(
            dbc.Badge(
                f"Trades: {metrics.get('trades', 0)}",
                color="info",
                className="me-2",
            )
        )
    alerts: list[Any] = []
    if not store_data.get("trades"):
        alerts.append(
            dbc.Alert(
                "Trade performance cache is empty. Run the refresh script to populate results.",
                color="info",
                className="mb-3",
            )
        )
    trade_perf_columns = [
        {"name": "Symbol", "id": "symbol"},
        {"name": "Entry Time", "id": "entry_time"},
        {"name": "Exit Time", "id": "exit_time"},
        {"name": "Qty", "id": "qty", "type": "numeric", "format": Format(precision=2, scheme="f")},
        {
            "name": "Entry Price",
            "id": "entry_price",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Exit Price",
            "id": "exit_price",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {"name": "PnL", "id": "pnl", "type": "numeric", "format": Format(precision=2, scheme="f")},
        {
            "name": "Return %",
            "id": "return_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Hold Days",
            "id": "hold_days",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {"name": "Exit Reason", "id": "exit_reason"},
        {
            "name": "MFE %",
            "id": "mfe_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "MAE %",
            "id": "mae_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Peak Price",
            "id": "peak_price",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Trough Price",
            "id": "trough_price",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Missed Profit %",
            "id": "missed_profit_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Exit Efficiency %",
            "id": "exit_efficiency_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {"name": "Trailing Stop Exit", "id": "is_trailing_stop_exit"},
        {
            "name": "Rebound Window (days)",
            "id": "rebound_window_days",
            "type": "numeric",
            "format": Format(precision=0, scheme="f"),
        },
        {
            "name": "Post-exit High",
            "id": "post_exit_high",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Rebound %",
            "id": "rebound_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {"name": "Rebounded", "id": "rebounded"},
    ]

    trade_pnl_columns = [
        {"name": "Exit Time", "id": "exit_time"},
        {"name": "Symbol", "id": "symbol"},
        {"name": "Side", "id": "side"},
        {"name": "Qty", "id": "qty", "type": "numeric", "format": Format(precision=2, scheme="f")},
        {
            "name": "Entry Price",
            "id": "entry_price",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Exit Price",
            "id": "exit_price",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Entry Value",
            "id": "entry_value",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Exit Value",
            "id": "exit_value",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {"name": "PnL", "id": "pnl", "type": "numeric", "format": Format(precision=2, scheme="f")},
        {
            "name": "Return %",
            "id": "return_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Hold Days",
            "id": "hold_days",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {"name": "Order Type", "id": "order_type"},
        {"name": "Exit Reason", "id": "exit_reason"},
    ]

    sold_too_columns = [
        {"name": "Exit Time", "id": "exit_time"},
        {"name": "Symbol", "id": "symbol"},
        {"name": "Order Type", "id": "order_type"},
        {
            "name": "Return %",
            "id": "return_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Exit Efficiency %",
            "id": "exit_efficiency_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Missed Profit %",
            "id": "missed_profit_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Rebound %",
            "id": "rebound_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {"name": "Rebounded", "id": "rebounded"},
        {
            "name": "Hold Days",
            "id": "hold_days",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {"name": "PnL", "id": "pnl", "type": "numeric", "format": Format(precision=2, scheme="f")},
    ]
    sold_too_columns = [
        {"name": "Exit Time", "id": "exit_time"},
        {"name": "Symbol", "id": "symbol"},
        {"name": "Order Type", "id": "order_type"},
        {
            "name": "Return %",
            "id": "return_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Exit Efficiency %",
            "id": "exit_efficiency_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Missed Profit %",
            "id": "missed_profit_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {
            "name": "Rebound %",
            "id": "rebound_pct",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {"name": "Rebounded", "id": "rebounded"},
        {
            "name": "Hold Days",
            "id": "hold_days",
            "type": "numeric",
            "format": Format(precision=2, scheme="f"),
        },
        {"name": "PnL", "id": "pnl", "type": "numeric", "format": Format(precision=2, scheme="f")},
    ]

    return html.Div(
        [
            dcc.Store(id="trade-perf-store", data=store_data),
            html.Div(alerts),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Trade Performance", className="text-light mb-0"),
                            html.P(
                                "Aggregate P&L, exit quality, and excursion metrics across closed trades.",
                                className="text-muted",
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        dbc.RadioItems(
                            id="trade-perf-range",
                            options=[
                                {"label": "7D", "value": "7D"},
                                {"label": "30D", "value": "30D"},
                                {"label": "365D", "value": "365D"},
                                {"label": "ALL", "value": "ALL"},
                            ],
                            value="30D",
                            inline=True,
                            labelClassName="me-3",
                        ),
                        md=4,
                        className="d-flex align-items-center justify-content-end",
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                _build_trade_perf_kpis(store_data.get("summary", {})),
                id="trade-perf-kpis",
                className="g-3 mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Sold Too Soon Controls", className="fw-bold"),
                    # Sold Too Soon flagged-trades section
                    html.H4("Sold Too Soon (Flagged Trades)", className="mt-5 text-light"),
                    html.Div(id="sold-too-summary-chips", className="mb-2 d-flex flex-wrap gap-2"),
                    dash_table.DataTable(
                        id="sold-too-table",
                        columns=sold_too_columns,
                        data=[],
                        sort_action="native",
                        filter_action="native",
                        sort_by=[{"column_id": "exit_time", "direction": "desc"}],
                        page_size=20,
                        style_table={"overflowX": "auto"},
                        style_cell={"backgroundColor": "#1e1e1e", "color": "#fff"},
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Flag mode"),
                                            dbc.RadioItems(
                                                id="sold-too-mode",
                                                options=[
                                                    {"label": "Either", "value": "either"},
                                                    {
                                                        "label": "Efficiency only",
                                                        "value": "efficiency",
                                                    },
                                                    {
                                                        "label": "Missed-profit only",
                                                        "value": "missed",
                                                    },
                                                ],
                                                value="either",
                                                inline=True,
                                                labelClassName="me-3",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Exit efficiency cutoff (%)"),
                                            dcc.Slider(
                                                id="sold-too-eff-cutoff",
                                                min=0,
                                                max=100,
                                                step=1,
                                                value=40,
                                                marks=None,
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Missed profit cutoff (%)"),
                                            dcc.Slider(
                                                id="sold-too-missed-cutoff",
                                                min=0,
                                                max=20,
                                                step=0.5,
                                                value=3,
                                                marks=None,
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                        ],
                                        md=4,
                                    ),
                                ],
                                className="gy-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Rebound threshold (%)"),
                                            dcc.Slider(
                                                id="sold-too-rebound-threshold",
                                                min=0,
                                                max=20,
                                                step=0.5,
                                                value=3,
                                                marks=None,
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label("Rebound window (days)"),
                                            dcc.Slider(
                                                id="sold-too-rebound-window",
                                                min=1,
                                                max=30,
                                                step=1,
                                                value=5,
                                                marks=None,
                                                tooltip={
                                                    "placement": "bottom",
                                                    "always_visible": False,
                                                },
                                            ),
                                        ],
                                        md=6,
                                    ),
                                ],
                                className="gy-3 mt-2",
                            ),
                        ]
                    ),
                ],
                className="bg-dark text-light mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.Span("Trade Performance Charts"),
                            html.Div(status_badges, id="trade-perf-status", className="mt-2"),
                        ],
                        className="d-flex justify-content-between align-items-center flex-wrap gap-2",
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                dbc.Col(
                                    dcc.Graph(
                                        id="trade-perf-window-bar",
                                        figure=_empty_trade_perf_fig("Net P&L by window"),
                                    ),
                                    width=12,
                                ),
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="trade-perf-scatter",
                                            figure=_empty_trade_perf_fig("MFE % vs Return %"),
                                        ),
                                        md=6,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="trade-perf-eff-hist",
                                            figure=_empty_trade_perf_fig("Exit efficiency %"),
                                        ),
                                        md=3,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="trade-perf-reason-bar",
                                            figure=_empty_trade_perf_fig("Exit reasons"),
                                        ),
                                        md=3,
                                    ),
                                ],
                                className="g-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="trade-perf-rebound-scatter",
                                            figure=_empty_trade_perf_fig(
                                                "Exit Efficiency % vs Rebound %"
                                            ),
                                        ),
                                        md=6,
                                    ),
                                    dbc.Col(
                                        dcc.Graph(
                                            id="trade-perf-rebound-hist",
                                            figure=_empty_trade_perf_fig("Rebound %"),
                                        ),
                                        md=6,
                                    ),
                                ],
                                className="g-3",
                            ),
                            html.H4("Closed Trades — Cash P&L", className="mt-4"),
                            html.Div(
                                id="trade-pnl-summary-chips",
                                className="mb-2 d-flex flex-wrap gap-2",
                            ),
                            dash_table.DataTable(
                                id="trade-pnl-table",
                                columns=trade_pnl_columns,
                                data=[],
                                sort_action="native",
                                filter_action="native",
                                sort_by=[{"column_id": "exit_time", "direction": "desc"}],
                                page_size=20,
                                style_table={"overflowX": "auto"},
                                style_cell={"backgroundColor": "#1e1e1e", "color": "#fff"},
                            ),
                            html.H4("Per-trade details", className="mt-4"),
                            dash_table.DataTable(
                                id="trade-perf-table",
                                columns=trade_perf_columns,
                                data=store_data.get("trades", []),
                                sort_action="native",
                                filter_action="native",
                                filter_query='{exit_reason} = "TrailingStop" && {rebounded} = true',
                                page_size=20,
                                style_table={"overflowX": "auto"},
                                style_cell={"backgroundColor": "#1e1e1e", "color": "#fff"},
                            ),
                        ]
                    ),
                ],
                className="bg-dark text-light mb-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Sold Too Soon (Flagged Trades)", className="fw-bold"),
                    dbc.CardBody(
                        [
                            html.Div(
                                id="sold-too-soon-summary", className="mb-2 d-flex flex-wrap gap-2"
                            ),
                            dash_table.DataTable(
                                id="sold-too-soon-table",
                                columns=sold_too_columns,
                                data=[],
                                sort_action="native",
                                filter_action="native",
                                sort_by=[{"column_id": "exit_time", "direction": "desc"}],
                                page_size=20,
                                style_table={"overflowX": "auto"},
                                style_cell={"backgroundColor": "#1e1e1e", "color": "#fff"},
                            ),
                        ]
                    ),
                ],
                className="bg-dark text-light",
            ),
        ]
    )


def _resolve_trades_dataframe() -> tuple[pd.DataFrame | None, str, str, list[Any]]:
    """Return the most suitable trades dataframe and metadata."""

    alerts: list[Any] = []
    if os.path.exists(executed_trades_path):
        df, alert = load_csv(executed_trades_path)
        if alert is None and df is not None and not df.empty:
            return df, executed_trades_path, "Executed trades", alerts
        if alert:
            alerts.append(alert)

    if os.path.exists(trades_log_path):
        df, alert = load_csv(trades_log_path)
        if alert is None and df is not None and not df.empty:
            return df, trades_log_path, "Paper trades", alerts
        if alert:
            alerts.append(alert)

    return None, "", "", alerts


def load_prediction_history(limit: int = 7) -> list[tuple[str, pd.DataFrame]]:
    directory = os.path.join(BASE_DIR, "data", "predictions")
    if not os.path.isdir(directory):
        return []
    frames: list[tuple[str, pd.DataFrame]] = []
    for path in sorted(Path(directory).glob("*.csv")):
        if path.name == "latest.csv":
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            rename_map = {}
            for col in df.columns:
                lower = str(col).lower()
                if lower == "gappen":
                    rename_map[col] = "gap_pen"
                elif lower == "liqpen":
                    rename_map[col] = "liq_pen"
                else:
                    rename_map[col] = lower
            df = df.rename(columns=rename_map)
        frames.append((path.stem, df))
    return frames[-limit:]


def read_recent_lines(filepath, num_lines=50, since_days=None):
    """Return recent lines from ``filepath``.

    ``since_days`` filters out log entries older than the given number of days
    based on the leading timestamp of each line.
    """
    filename = os.path.basename(filepath)
    if not os.path.exists(filepath):
        return [f"{filename} not found.\n"]

    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
        if since_days is not None:
            cutoff = datetime.utcnow() - timedelta(days=since_days)
            filtered = []
            for line in lines:
                try:
                    ts = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                    if ts >= cutoff:
                        filtered.append(line)
                except Exception:
                    filtered.append(line)
            lines = filtered
        lines = lines[-num_lines:]
        return lines if lines else [f"No entries in {filename}.\n"]
    except Exception as e:
        return [f"Error reading {filename}: {e}\n"]


def _tail_with_timestamp(path: str, limit: int = 100) -> tuple[list[str], str]:
    lines = read_recent_lines(path, num_lines=limit)
    last_ts: Optional[datetime] = None
    for line in reversed(lines):
        ts = _parse_log_timestamp(line)
        if ts:
            last_ts = ts
            break
    if last_ts is None and os.path.exists(path):
        try:
            last_ts = datetime.utcfromtimestamp(os.path.getmtime(path))
        except OSError:
            last_ts = None
    label = (
        last_ts.strftime("Last log line at: %Y-%m-%d %H:%M UTC")
        if last_ts
        else "Last log line: n/a"
    )
    return lines, label


def read_error_log(path: str = error_log_path) -> pd.DataFrame:
    """Return a DataFrame of error log entries from the last day."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "level", "message"])
    try:
        records = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 2)
                if len(parts) < 3:
                    continue
                ts = f"{parts[0]} {parts[1]}"
                level_msg = parts[2].split(" ", 1)
                level = level_msg[0]
                msg = level_msg[1] if len(level_msg) > 1 else ""
                records.append({"timestamp": ts, "level": level, "message": msg})
        errors_df = pd.DataFrame(records)
        if not errors_df.empty:
            errors_df = errors_df[
                pd.to_datetime(errors_df["timestamp"]) >= datetime.now() - timedelta(days=1)
            ]
        return errors_df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "level", "message"])


def format_log_lines(lines):
    """Return a list of HTML spans with coloring for ERROR and WARNING lines."""
    formatted = []
    for line in lines:
        if "[ERROR]" in line or "ERROR" in line:
            formatted.append(html.Span(line, style={"color": "#E57373"}))
        elif "[WARNING]" in line or "WARNING" in line:
            formatted.append(html.Span(line, style={"color": "#FFB74D"}))
        else:
            formatted.append(html.Span(line))
    return formatted


def add_days_in_trade(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``days_in_trade`` column based on ``entry_time``."""
    entry_times = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df["days_in_trade"] = (pd.Timestamp.utcnow() - entry_times).dt.days
    df["entry_time"] = entry_times.dt.strftime("%Y-%m-%d %H:%M") + " UTC"
    return df


def explain_breakdown(json_str: str) -> str:
    import json as _json

    try:
        data = _json.loads(json_str) if isinstance(json_str, str) else {}
    except Exception:
        return "n/a"

    labels = {
        "TS": "Trend",
        "MS": "MA Stack",
        "BP": "Near 20D High",
        "PT": "Pullback Tight",
        "RSI": "RSI",
        "MH": "MACD Hist",
        "ADX": "ADX",
        "AROON": "Aroon",
        "VCP": "VCP",
        "VOLexp": "Vol Exp",
        "GAPpen": "Gap Pen",
        "LIQpen": "Liq Pen",
    }
    items = []
    for key, value in (data or {}).items():
        base_key = key.replace("_z", "")
        if base_key not in labels:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        items.append((labels[base_key], numeric))
    if not items:
        return "n/a"
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    items = items[:4]

    def _fmt(name: str, val: float) -> str:
        arrow = "▲" if val >= 0 else "▼"
        return f"{name} {arrow} {abs(val):.2f}"

    return " • ".join(_fmt(name, val) for name, val in items)


def create_open_positions_chart(df: pd.DataFrame) -> dcc.Graph:
    """Return a bar chart showing open position P/L."""
    pnl_col = "unrealized_pl" if "unrealized_pl" in df.columns else "pnl"
    fig = px.bar(
        df,
        x="symbol",
        y=pnl_col,
        color=df[pnl_col] > 0,
        color_discrete_map={True: "#4DB6AC", False: "#E57373"},
        template="plotly_dark",
        title="Open Positions P/L",
    )
    return dcc.Graph(figure=fig)


def create_open_positions_table(df: pd.DataFrame) -> dash_table.DataTable:
    """Return a styled Dash DataTable for open positions."""
    pnl_col = "unrealized_pl" if "unrealized_pl" in df.columns else "pnl"
    columns = [{"name": c.replace("_", " ").title(), "id": c} for c in df.columns]
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=columns,
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
        style_data_conditional=[
            {
                "if": {"filter_query": f"{{{pnl_col}}} < 0", "column_id": pnl_col},
                "color": "#E57373",
            },
            {
                "if": {"filter_query": f"{{{pnl_col}}} > 0", "column_id": pnl_col},
                "color": "#4DB6AC",
            },
        ],
    )


def _styled_table(
    df: pd.DataFrame, table_id: str | None = None, page_size: int = 25
) -> dash_table.DataTable:
    columns = [{"name": c.replace("_", " ").title(), "id": c} for c in df.columns]
    return dash_table.DataTable(
        id=table_id,
        data=df.to_dict("records"),
        columns=columns,
        style_table={"overflowX": "auto", "maxHeight": "450px", "overflowY": "auto"},
        style_cell={"backgroundColor": "#212529", "color": "#E0E0E0", "fontSize": "0.85rem"},
        page_size=page_size,
    )


def log_box(
    title: str,
    lines: list[str],
    element_id: str,
    log_path: str | None = None,
    subtitle: str | None = None,
) -> html.Div:
    """Return a styled log display box."""
    header_children: list[Any] = [html.Span(title, className="text-light")]
    if subtitle:
        header_children.append(html.Span(subtitle, className="text-muted small"))
    badge = _log_activity_badge(lines, path=log_path)
    if badge:
        header_children.append(badge)
    return html.Div(
        [
            html.Div(
                header_children,
                className="d-flex align-items-center justify-content-between",
            ),
            html.Pre(
                format_log_lines(lines),
                id=element_id,
                style={
                    "maxHeight": "200px",
                    "overflowY": "auto",
                    "backgroundColor": "#272B30",
                    "color": "#E0E0E0",
                    "padding": "0.5rem",
                },
            ),
        ],
        className="mb-3",
    )


def _render_pipeline_summary_panel(summary: Mapping[str, Any] | None) -> html.Div:
    if not summary:
        return dbc.Alert(
            "No PIPELINE_SUMMARY found in logs yet.",
            color="secondary",
            className="mb-2",
        )
    fields = [
        ("Symbols In", "symbols_in"),
        ("With Bars", "with_bars"),
        ("Rows", "rows"),
        ("Fetch secs", "fetch_secs"),
        ("Feature secs", "feature_secs"),
        ("Rank secs", "rank_secs"),
        ("Gate secs", "gate_secs"),
    ]
    rows = []
    for label, key in fields:
        value = summary.get(key, "n/a") if isinstance(summary, Mapping) else "n/a"
        rows.append(html.Tr([html.Th(label, className="text-muted"), html.Td(value)]))
    return dbc.Table(rows, bordered=True, size="sm", className="mb-3")


def stale_warning(paths: list[str], threshold_minutes: int = 30) -> html.Div:
    """Return a warning banner if the newest of ``paths`` is older than the threshold."""
    latest_update = 0.0
    for path in paths:
        mtime = get_file_mtime(path)
        if mtime:
            latest_update = max(latest_update, mtime)

    if not latest_update:
        return html.Div(
            "Warning: Monitoring service is stale!",
            style={"color": "red"},
        )

    age_minutes = (
        datetime.now(timezone.utc) - datetime.fromtimestamp(latest_update, timezone.utc)
    ).total_seconds() / 60
    if age_minutes > threshold_minutes:
        return html.Div(
            "Warning: Monitoring service is stale!",
            style={"color": "red"},
        )
    return html.Div()


def get_version_string():
    """Return short git commit hash or file mtime for version banner."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR)
            .decode("utf-8")
            .strip()
        )
        ts = datetime.utcfromtimestamp(os.path.getmtime(__file__)).strftime("%Y-%m-%d %H:%M UTC")
        return f"Dashboard version: {commit} (updated {ts})"
    except Exception:
        ts = datetime.utcfromtimestamp(os.path.getmtime(__file__)).strftime("%Y-%m-%d %H:%M UTC")
        return f"Dashboard version: {ts}"


def file_timestamp(path):
    """Return modification time of ``path`` formatted for display."""
    if not os.path.exists(path):
        return "N/A"
    ts = datetime.utcfromtimestamp(os.path.getmtime(path))
    return ts.strftime("%Y-%m-%d %H:%M:%S") + " UTC"


def get_file_mtime(path):
    """Return the modification time of ``path`` or ``None`` if unavailable."""
    try:
        return os.path.getmtime(path)
    except Exception as e:
        logger.error(f"Failed to get mtime for {path}: {e}")
        return None


def format_time(ts):
    """Return ``ts`` converted to CST/CDT (America/Chicago)."""
    if not ts:
        return "N/A"
    utc_time = datetime.fromtimestamp(ts, pytz.utc)
    local_time = utc_time.astimezone(pytz.timezone("America/Chicago"))
    return local_time.strftime("%Y-%m-%d %H:%M:%S %Z")


def pipeline_status_component():
    """Create status indicators for Screener -> Backtest -> Execution."""
    steps = [
        ("Screener", screener_log_path),
        ("Backtest", backtest_log_path),
        ("Execution", execute_trades_log_path),
    ]
    now = datetime.utcnow()
    items = []
    status_data = {}
    if os.path.exists(pipeline_status_json_path):
        try:
            status_data = json.load(open(pipeline_status_json_path))
        except Exception:
            status_data = {}
    for name, path in steps:
        if os.path.exists(path):
            mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
            stale = is_log_stale(path)
            if stale:
                color = "warning"
                status = "Stale"
            else:
                color = "success"
                status = "Completed"
            timestamp = mtime.strftime("%Y-%m-%d %H:%M") + " UTC"
        else:
            ts = status_data.get(name)
            if ts:
                mtime = datetime.utcfromtimestamp(ts)
                timestamp = mtime.strftime("%Y-%m-%d %H:%M") + " UTC"
                stale = (datetime.utcnow() - mtime).total_seconds() > (STALE_THRESHOLD_MINUTES * 60)
                if stale:
                    color = "warning"
                    status = "Stale"
                else:
                    color = "success"
                    status = "Completed"
            else:
                color = "danger"
                status = "Missing"
                timestamp = "N/A"
        items.append(dbc.ListGroupItem(f"{name}: {status} ({timestamp})", color=color))
    return dbc.ListGroup(items, className="mb-3")


def data_freshness_alert(path, name, threshold_minutes=60):
    """Return a Dash alert if ``path`` was not modified within ``threshold_minutes``."""
    if not os.path.exists(path):
        return dbc.Alert(f"{name} not found", color="danger", className="m-2")
    mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
    age = (datetime.utcnow() - mtime).total_seconds() / 60
    if age > threshold_minutes:
        return dbc.Alert(
            f"{name} has not updated for {int(age)} minutes",
            color="warning",
            className="m-2",
        )
    return None


DASH_BASE_PATH = "/v2/"

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V2.1.0/dbc.min.css",
    ],
    url_base_pathname=DASH_BASE_PATH,
    suppress_callback_exceptions=True,
    title="JBRAVO Trading Bot",
)

server = app.server


@server.after_request
def _api_cors_headers(response: Response) -> Response:
    path = str(getattr(request, "path", "") or "")
    if path.startswith("/api/"):
        response.headers.setdefault("Access-Control-Allow-Origin", "*")
        response.headers.setdefault(
            "Access-Control-Allow-Headers", "Content-Type, Authorization, Accept"
        )
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, OPTIONS")
    return response


@server.route("/")
def react_root():
    """Serve the React build at the root instead of the Dash UI."""

    index_path = REACT_BUILD_DIR / "index.html"
    if not index_path.exists():
        message = "React build not found. Run the frontend build and place output in frontend/dist."
        return message, 503
    return send_from_directory(REACT_BUILD_DIR, "index.html")


@server.route("/ui-assets/<path:filename>")
def react_assets(filename: str):
    if not REACT_BUILD_DIR.exists():
        abort(404)
    file_path = REACT_BUILD_DIR / filename
    if not file_path.exists():
        abort(404)
    return send_from_directory(REACT_BUILD_DIR, filename)


@server.route("/assets/<path:filename>")
def react_asset_bundle(filename: str):
    file_path = REACT_ASSET_DIR / filename if REACT_ASSET_DIR.exists() else None
    if file_path and file_path.exists():
        return send_from_directory(REACT_ASSET_DIR, filename)
    brand_path = BRAND_ASSET_DIR / filename
    if brand_path.exists():
        return send_from_directory(BRAND_ASSET_DIR, filename)
    abort(404)


@server.route("/health/overview")
def health_overview():
    """Return a lightweight JSON payload describing dashboard health."""

    summary: dict[str, Any] = {
        "metrics_summary_present": False,
        "trades_log_present": False,
        "trades_log_rows": None,
        "kpis": {},
    }
    metrics = _metrics_summary_db()
    if metrics:
        summary["metrics_summary_present"] = True
        summary["kpis"] = metrics

    trades_row = _db_fetch_one("SELECT COUNT(*) AS count FROM trades")
    if trades_row and trades_row.get("count") is not None:
        summary["trades_log_present"] = True
        summary["trades_log_rows"] = int(trades_row["count"])

    return jsonify({"ok": bool(metrics or trades_row), **summary})


@server.route("/api/time")
def api_time():
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    payload = {"utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store, max-age=1, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


def _json_no_store(payload: Mapping[str, Any] | list[Any]) -> Response:
    response = jsonify(payload)
    response.headers["Cache-Control"] = "no-store, max-age=1, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@server.route("/api/pythonanywhere/resources")
def api_pythonanywhere_resources():
    resources: list[dict[str, Any]] = []
    cpu = _pythonanywhere_cpu_usage()
    if cpu:
        resources.append(cpu)
    file_usage = _pythonanywhere_file_storage_usage()
    if file_usage:
        resources.append(file_usage)
    postgres_usage = _pythonanywhere_postgres_usage()
    if postgres_usage:
        resources.append(postgres_usage)
    return jsonify({"ok": bool(resources), "resources": resources})


@server.route("/api/pythonanywhere/tasks")
def api_pythonanywhere_tasks():
    tasks = _pythonanywhere_schedule_tasks()
    normalized = [_normalize_schedule_task(task) for task in tasks]
    return jsonify({"ok": bool(normalized), "tasks": normalized})


@server.route("/api/pythonanywhere/logs")
def api_pythonanywhere_logs():
    sources = _pythonanywhere_log_sources()
    return jsonify({"ok": bool(sources), "sources": sources, "source": "pythonanywhere"})


@server.route("/api/positions/logs")
def api_positions_logs():
    limit_raw = request.args.get("limit")
    try:
        limit = int(limit_raw) if limit_raw is not None else 80
    except (TypeError, ValueError):
        limit = 80
    limit = max(1, min(limit, 200))

    logs, source = _positions_logs_live(limit=limit)
    return jsonify({"ok": bool(logs), "logs": logs, "source": source})


@server.route("/api/logos/<symbol>")
def api_logo(symbol: str):
    safe_symbol = quote(symbol.upper(), safe=".-")
    url = f"https://storage.googleapis.com/iex/api/logos/{safe_symbol}.png"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
            content_type = resp.headers.get_content_type() or "image/png"
    except Exception:
        return Response(status=404)
    response = Response(data, content_type=content_type)
    response.headers["Cache-Control"] = "public, max-age=86400"
    return response


@server.route("/api/account/overview")
def api_account_overview():
    row = _account_latest_db()
    if row:
        snapshot = {key: _serialize_record(value) for key, value in row.items()}
        return _json_no_store(
            {
                "ok": True,
                "snapshot": snapshot,
                "source": "postgres",
                "source_detail": str(row.get("source") or "db"),
                "db_source_of_truth": True,
                "db_ready": True,
            }
        )

    if _account_api_allow_live_fallback():
        live_row = _account_latest_alpaca()
        if live_row:
            snapshot = {key: _serialize_record(value) for key, value in live_row.items()}
            return _json_no_store(
                {
                    "ok": True,
                    "snapshot": snapshot,
                    "source": "alpaca-fallback",
                    "source_detail": "live",
                    "db_source_of_truth": True,
                    "db_ready": False,
                }
            )

    return _json_no_store(
        {
            "ok": False,
            "snapshot": {},
            "source": "postgres",
            "source_detail": "unavailable",
            "db_source_of_truth": True,
            "db_ready": False,
        }
    )


@server.route("/api/account/summary")
def api_account_summary():
    paper_mode = _account_api_is_paper_mode()
    alpaca_detail = "paper_mode_required" if not paper_mode else "skipped"

    if paper_mode:
        payload, detail = _alpaca_rest_get_json("/v2/account", timeout=12)
        alpaca_detail = detail
        if detail == "ok" and isinstance(payload, Mapping):
            equity = _to_float(payload.get("equity")) or 0.0
            cash = _to_float(payload.get("cash")) or 0.0
            buying_power = _to_float(payload.get("buying_power")) or 0.0

            open_positions_value = _open_positions_value_from_alpaca()
            if open_positions_value is None or open_positions_value <= 0:
                inferred_value = equity - cash
                open_positions_value = inferred_value if inferred_value > 0 else 0.0

            ratio = None
            if open_positions_value > 0:
                ratio = float(cash / open_positions_value)

            return _json_no_store(
                {
                    "ok": True,
                    "source": "alpaca",
                    "equity": float(equity),
                    "cash": float(cash),
                    "buying_power": float(buying_power),
                    "open_positions_value": float(open_positions_value),
                    "cash_to_positions_ratio": ratio,
                    "taken_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                }
            )

    db_summary, db_detail = _account_summary_from_db()
    if db_summary:
        return _json_no_store(
            {
                "ok": True,
                "source": "postgres",
                **db_summary,
                "source_detail": f"alpaca:{alpaca_detail};{db_detail}",
            }
        )

    csv_summary, csv_detail = _account_summary_from_csv()
    if csv_summary:
        return _json_no_store(
            {
                "ok": True,
                "source": "csv",
                **csv_summary,
                "source_detail": f"alpaca:{alpaca_detail};{csv_detail}",
            }
        )

    return _json_no_store(
        {
            "ok": False,
            "source": "alpaca",
            "equity": 0.0,
            "cash": 0.0,
            "buying_power": 0.0,
            "open_positions_value": 0.0,
            "cash_to_positions_ratio": None,
            "taken_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_detail": f"alpaca:{alpaca_detail};db:unavailable;csv:unavailable",
        }
    )


@server.route("/api/account/performance")
def api_account_performance():
    requested_range = _parse_account_range(request.args.get("range"), default="all")
    paper_mode = _account_api_is_paper_mode()
    points: list[dict[str, Any]] = []
    source_detail = "alpaca:skipped"

    if paper_mode:
        points, history_detail = _fetch_account_portfolio_points(period="all", timeframe="1D")
        source_detail = f"alpaca:{history_detail}"
    else:
        history_detail = "paper_mode_required"
        source_detail = f"alpaca:{history_detail}"

    if not points:
        db_points, db_detail = _account_portfolio_points_from_db(period="ALL", timeframe="1D")
        if db_points:
            points = db_points
            source_detail = f"{source_detail};postgres:{db_detail}"
        else:
            source_detail = f"{source_detail};postgres:{db_detail}"
    if not points:
        csv_points, csv_detail = _account_portfolio_points_from_csv(period="ALL", timeframe="1D")
        if csv_points:
            points = csv_points
        source_detail = f"{source_detail};csv:{csv_detail}"

    latest_equity = _to_float(points[-1].get("equity")) if points else 0.0
    if latest_equity is None:
        latest_equity = 0.0

    rows: list[dict[str, Any]] = []
    detail_tokens: list[str] = []
    calculable_statuses = {"ok", "partial_history"}
    for key in _ACCOUNT_PERFORMANCE_RANGE_ORDER:
        pct, usd, status = _account_delta_for_range(points, key)
        if status not in calculable_statuses:
            pct = 0.0
            usd = 0.0
            detail_tokens.append(f"{key}:{status}")
        elif status != "ok":
            detail_tokens.append(f"{key}:{status}")
        rows.append(
            {
                "period": _ACCOUNT_PERFORMANCE_LABELS[key],
                "netChangePct": float(pct),
                "netChangeUsd": float(usd),
            }
        )

    total_pct, total_usd, total_status = _account_delta_for_range(points, requested_range)
    if total_status not in calculable_statuses:
        total_pct = 0.0
        total_usd = 0.0
        detail_tokens.append(f"total:{total_status}")
    elif total_status != "ok":
        detail_tokens.append(f"total:{total_status}")

    if history_detail != "ok":
        detail_tokens.append(f"history:{history_detail}")

    detail_parts = [source_detail]
    if detail_tokens:
        detail_parts.append(";".join(dict.fromkeys(detail_tokens)))
    source_detail = ";".join([part for part in detail_parts if part])

    return _json_no_store(
        {
            "ok": bool(points),
            "range": requested_range,
            "rows": rows,
            "accountTotal": {
                "equity": float(latest_equity),
                "netChangePct": float(total_pct),
                "netChangeUsd": float(total_usd),
            },
            "source_detail": source_detail,
        }
    )


@server.route("/api/account/portfolio_history")
def api_account_portfolio_history():
    period_label, alpaca_period = _normalize_account_history_period(request.args.get("period"))
    timeframe = _normalize_account_history_timeframe(request.args.get("timeframe"))
    paper_mode = _account_api_is_paper_mode()

    points: list[dict[str, Any]] = []
    detail = "paper_mode_required" if not paper_mode else "skipped"
    if paper_mode:
        points, detail = _fetch_account_portfolio_points(period=alpaca_period, timeframe=timeframe)
    source_detail = f"alpaca:{detail}"

    if not points:
        db_points, db_detail = _account_portfolio_points_from_db(
            period=period_label, timeframe=timeframe
        )
        if db_points:
            points = db_points
        source_detail = f"{source_detail};postgres:{db_detail}"
    if not points:
        csv_points, csv_detail = _account_portfolio_points_from_csv(
            period=period_label, timeframe=timeframe
        )
        if csv_points:
            points = csv_points
        source_detail = f"{source_detail};csv:{csv_detail}"

    serialized_points = [
        {"t": str(point.get("t") or ""), "equity": float(point.get("equity") or 0.0)}
        for point in points
    ]
    start = serialized_points[0]["t"] if serialized_points else None
    end = serialized_points[-1]["t"] if serialized_points else None
    return _json_no_store(
        {
            "ok": bool(serialized_points),
            "points": serialized_points,
            "start": start,
            "end": end,
            "timeframe": timeframe,
            "period": period_label,
            "source_detail": source_detail,
        }
    )


@server.route("/api/account/open_orders")
def api_account_open_orders():
    limit = _parse_positive_int(request.args.get("limit"), default=50, minimum=1, maximum=200)
    paper_mode = _account_api_is_paper_mode()
    if paper_mode:
        payload, detail = _alpaca_rest_get_json(
            "/v2/orders",
            params={"status": "open", "limit": int(limit), "direction": "desc"},
            timeout=20,
        )
        if detail == "ok":
            if isinstance(payload, list):
                raw_orders = payload
            elif isinstance(payload, Mapping) and isinstance(payload.get("orders"), list):
                raw_orders = payload.get("orders", [])
            else:
                raw_orders = []

            rows: list[dict[str, Any]] = []
            for raw in raw_orders:
                if not isinstance(raw, Mapping):
                    continue
                status = str(raw.get("status") or "").strip().lower()
                if status and status not in _ACCOUNT_OPEN_ORDER_STATUSES:
                    continue
                submitted_at_dt = _coerce_datetime_utc(
                    raw.get("submitted_at") or raw.get("created_at") or raw.get("updated_at")
                )
                submitted_at = (
                    submitted_at_dt.isoformat()
                    if submitted_at_dt is not None
                    else str(raw.get("submitted_at") or raw.get("created_at") or "")
                )

                qty = _to_float(raw.get("qty"))
                rows.append(
                    {
                        "symbol": str(raw.get("symbol") or "").strip().upper(),
                        "type": str(raw.get("type") or "market").strip().lower(),
                        "side": str(raw.get("side") or "buy").strip().lower(),
                        "qty": float(qty) if qty is not None else 0.0,
                        "price_or_stop": _order_price_or_stop(raw),
                        "submitted_at": submitted_at,
                    }
                )

            rows.sort(key=lambda row: str(row.get("submitted_at") or ""), reverse=True)
            return _json_no_store({"ok": True, "rows": rows[: int(limit)]})
        alpaca_detail = detail
    else:
        alpaca_detail = "paper_mode_required"

    db_rows, db_detail = _account_open_orders_from_db(limit=limit)
    if db_rows:
        return _json_no_store(
            {
                "ok": True,
                "rows": db_rows,
                "source_detail": f"alpaca:{alpaca_detail};{db_detail}",
            }
        )
    return _json_no_store(
        {
            "ok": False,
            "rows": [],
            "source_detail": f"alpaca:{alpaca_detail};{db_detail}",
        }
    )


@server.route("/api/account/order_logs")
def api_account_order_logs():
    limit = _parse_positive_int(request.args.get("limit"), default=100, minimum=1, maximum=300)
    rows, source_detail = _account_order_logs(limit=limit)
    return _json_no_store(
        {
            "ok": bool(rows),
            "rows": rows,
            "source_detail": source_detail,
        }
    )


@server.route("/api/trades/overview")
def api_trades_overview():
    metrics = _metrics_summary_db()
    trades_df = load_trades_db(limit=200)
    trade_records = []
    if not trades_df.empty:
        for record in trades_df.to_dict(orient="records"):
            trade_records.append({key: _serialize_record(value) for key, value in record.items()})

    live_positions = _positions_from_alpaca()
    open_source = "alpaca"
    open_count = int(len(live_positions))
    open_pnl = None
    if live_positions:
        open_pnl = float(sum(_to_float(row.get("dollar_pl")) or 0.0 for row in live_positions))
    else:
        open_df = load_open_trades_db()
        open_source = "db-fallback"
        open_count = int(len(open_df)) if open_df is not None else 0
        if (
            isinstance(open_df, pd.DataFrame)
            and not open_df.empty
            and "realized_pnl" in open_df.columns
        ):
            open_pnl = float(open_df["realized_pnl"].fillna(0).sum())

    payload = {
        "ok": bool(metrics or trade_records),
        "db_source_of_truth": True,
        "metrics": metrics,
        "trades": trade_records,
        "open_positions": {"count": open_count, "realized_pnl": open_pnl, "source": open_source},
    }
    return jsonify(payload)


@server.route("/api/trades/stats")
def api_trades_stats():
    requested_range = _parse_trades_range(request.args.get("range"), default="all")
    trades_frame, source, source_detail = _load_trades_analytics_frame()
    db_ready = _trades_api_db_ready(source, source_detail)

    range_keys = _TRADES_RANGE_ORDER if requested_range == "all" else [requested_range]
    rows = [_build_range_metrics(trades_frame, key) for key in range_keys]

    recorded = _record_trades_api_request(
        endpoint="/api/trades/stats",
        params={"range": requested_range, "source_detail": source_detail},
        source=source,
        rows_returned=len(rows),
    )
    return _json_no_store(
        {
            "ok": bool(db_ready),
            "source": source,
            "source_detail": source_detail,
            "db_source_of_truth": True,
            "db_ready": bool(db_ready),
            "range": requested_range,
            "rows": rows,
            "recorded_to_db": recorded,
        }
    )


@server.route("/api/trades/leaderboard")
def api_trades_leaderboard():
    range_key = str(request.args.get("range", "all")).strip().lower()
    mode = str(request.args.get("mode", "winners")).strip().lower()
    limit = request.args.get("limit", 10, type=int)
    if limit is None:
        limit = 10
    limit = max(1, min(50, limit))
    range_key = _parse_trades_range(range_key, default="all")
    if mode not in {"winners", "losers"}:
        response = _json_no_store(
            {
                "ok": False,
                "error": "invalid_mode",
                "message": "mode must be one of: winners, losers",
                "range": range_key,
                "mode": mode,
                "limit": limit,
            }
        )
        response.status_code = 400
        return response

    trades_frame, source, source_detail = _load_trades_analytics_frame()
    db_ready = _trades_api_db_ready(source, source_detail)
    rows = _build_leaderboard_rows(trades_frame, range_key=range_key, mode=mode, limit=limit)

    recorded = _record_trades_api_request(
        endpoint="/api/trades/leaderboard",
        params={"range": range_key, "mode": mode, "limit": limit, "source_detail": source_detail},
        source=source,
        rows_returned=len(rows),
    )
    return _json_no_store(
        {
            "ok": bool(db_ready),
            "source": source,
            "source_detail": source_detail,
            "db_source_of_truth": True,
            "db_ready": bool(db_ready),
            "range": range_key,
            "mode": mode,
            "limit": limit,
            "rows": rows,
            "recorded_to_db": recorded,
        }
    )


@server.route("/api/trades/latest")
def api_trades_latest():
    limit = _parse_positive_int(request.args.get("limit"), default=25, minimum=1, maximum=200)
    trades_frame, source, source_detail = _load_trades_analytics_frame()
    db_ready = _trades_api_db_ready(source, source_detail)
    rows = _build_latest_trades_rows(trades_frame, limit=limit)

    recorded = _record_trades_api_request(
        endpoint="/api/trades/latest",
        params={"limit": limit, "source_detail": source_detail},
        source=source,
        rows_returned=len(rows),
    )
    return _json_no_store(
        {
            "ok": bool(db_ready),
            "source": source,
            "source_detail": source_detail,
            "db_source_of_truth": True,
            "db_ready": bool(db_ready),
            "limit": limit,
            "rows": rows,
            "recorded_to_db": recorded,
        }
    )


@server.route("/api/pipeline/task")
def api_pipeline_task():
    payload = _parse_task_log(Path(TASK_LOG_PATH))
    return jsonify(payload)


@server.route("/api/execute/task")
def api_execute_task():
    payload = _parse_task_log(
        Path(EXECUTE_TASK_LOG_PATH), fetcher=_fetch_pythonanywhere_execute_task_log
    )
    return jsonify(payload)


@server.route("/api/positions/monitoring")
def api_positions_monitoring():
    include_debug = str(request.args.get("debug", "")).lower() in {"1", "true", "yes", "on"}
    db_ready = _db_connection_available()

    def _sparkline_from_daily_bars(symbols: list[str], points: int = 12) -> dict[str, list[float]]:
        bars_df = _load_daily_bars_cache()
        if bars_df is None or bars_df.empty:
            return {}
        sparkline_map: dict[str, list[float]] = {}
        for symbol in symbols:
            symbol_df = bars_df[bars_df["symbol"] == symbol]
            if symbol_df.empty:
                continue
            closes = symbol_df["close"].tail(points).tolist()
            sparkline_map[symbol] = [float(value) for value in closes if value is not None]
        return sparkline_map

    sparkline_points = 12
    open_df = load_open_trades_db()
    aggregated: dict[str, dict[str, Any]] = {}
    calculation_source = "alpaca"
    db_source_of_truth = False

    # Open positions are live-source first (Alpaca API).
    for record in _positions_from_alpaca():
        symbol = _normalize_symbol(record.get("symbol"))
        if not symbol:
            continue

        qty = _to_float(record.get("qty"))
        entry_price = _to_float(record.get("entry_price"))
        if qty is None or entry_price is None or qty == 0:
            continue

        dollar_pl = _to_float(record.get("dollar_pl"))
        percent_pl = _to_float(record.get("percent_pl"))

        bucket = aggregated.setdefault(
            symbol,
            {
                "symbol": symbol,
                "qty": 0.0,
                "entry_value": 0.0,
                "realized_pnl": 0.0,
                "has_realized": False,
                "dollar_pl": 0.0,
                "percent_pl": None,
            },
        )
        bucket["qty"] += qty
        bucket["entry_value"] += qty * entry_price
        if dollar_pl is not None:
            bucket["dollar_pl"] = _to_float(bucket.get("dollar_pl")) or 0.0
            bucket["dollar_pl"] += dollar_pl
        if percent_pl is not None and bucket.get("percent_pl") is None:
            bucket["percent_pl"] = percent_pl

    # Fallback to DB OPEN rows only when live API returns no positions.
    if not aggregated:
        if open_df is not None and not open_df.empty and "symbol" not in open_df.columns:
            return _json_no_store(
                {
                    "ok": False,
                    "positions": [],
                    "summary": _positions_summary([]),
                    "source": "db-fallback",
                    "calculationSource": "postgres",
                    "db_source_of_truth": True,
                    "db_ready": bool(db_ready),
                }
            )

        if open_df is not None and not open_df.empty:
            calculation_source = "postgres"
            db_source_of_truth = True
            for record in open_df.to_dict(orient="records"):
                symbol = str(record.get("symbol") or "").strip()
                if not symbol:
                    continue
                qty = _to_float(record.get("qty"))
                entry_price = _to_float(record.get("entry_price"))
                realized_pnl = _to_float(record.get("realized_pnl"))

                bucket = aggregated.setdefault(
                    symbol,
                    {
                        "symbol": symbol,
                        "qty": 0.0,
                        "entry_value": 0.0,
                        "realized_pnl": 0.0,
                        "has_realized": False,
                        "dollar_pl": None,
                        "percent_pl": None,
                    },
                )
                if qty is not None and entry_price is not None:
                    bucket["qty"] += qty
                    bucket["entry_value"] += qty * entry_price
                if realized_pnl is not None:
                    bucket["realized_pnl"] += realized_pnl
                    bucket["has_realized"] = True

    if not aggregated:
        return _json_no_store(
            {
                "ok": bool(db_ready),
                "positions": [],
                "summary": _positions_summary([]),
                "source": "alpaca",
                "calculationSource": "alpaca",
                "db_source_of_truth": False,
                "db_ready": bool(db_ready),
                "reconciliation_required": False,
            }
        )

    symbols = list(aggregated.keys())
    sparkline_map = _fetch_alpaca_sparklines(symbols, points=sparkline_points)
    alpaca_sparkline = bool(sparkline_map)
    if not sparkline_map:
        sparkline_map = _sparkline_from_daily_bars(symbols, points=sparkline_points)

    latest_prices = _fetch_latest_prices(symbols)
    trailing_stop_map = _active_trailing_stops_from_alpaca(symbols)
    activity_seed = [
        {"symbol": symbol, "qty": bucket.get("qty")} for symbol, bucket in aggregated.items()
    ]
    days_held_map = _days_held_map_from_alpaca_activities(activity_seed)
    using_alpaca_overlay = bool(
        alpaca_sparkline or latest_prices or trailing_stop_map or days_held_map
    )

    positions = []

    for symbol, bucket in aggregated.items():
        raw_qty = _to_float(bucket.get("qty"))
        qty = raw_qty if raw_qty is not None and raw_qty != 0 else None
        entry_price = (
            bucket["entry_value"] / raw_qty if raw_qty is not None and raw_qty != 0 else None
        )
        cost_basis = bucket["entry_value"] if bucket["entry_value"] else None
        sparkline = sparkline_map.get(symbol, [])
        latest_price = latest_prices.get(symbol)
        current_price = (
            latest_price
            if latest_price is not None
            else (sparkline[-1] if sparkline else entry_price)
        )

        dollar_pl = None
        if current_price is not None and entry_price is not None and qty is not None:
            dollar_pl = (current_price - entry_price) * qty
        if dollar_pl is None:
            dollar_pl = _to_float(bucket.get("dollar_pl"))

        percent_pl = None
        if dollar_pl is not None and entry_price is not None and qty is not None and qty != 0:
            cost_basis = entry_price * qty
            if cost_basis != 0:
                percent_pl = (dollar_pl / cost_basis) * 100
        if percent_pl is None:
            percent_pl = _to_float(bucket.get("percent_pl"))

        live_trailing = trailing_stop_map.get(_normalize_symbol(symbol), {})
        trailing_stop = _to_float(live_trailing.get("trailingStop"))
        trail_percent = _to_float(live_trailing.get("trailPercent"))

        positions.append(
            {
                "symbol": symbol,
                "qty": qty,
                "entryPrice": entry_price,
                "currentPrice": current_price,
                "sparklineData": sparkline,
                "percentPL": percent_pl,
                "dollarPL": dollar_pl,
                "costBasis": cost_basis,
                "logoUrl": _logo_url_for_symbol(symbol),
                "daysHeld": days_held_map.get(_normalize_symbol(symbol)),
                "trailingStop": trailing_stop,
                "trailPercent": trail_percent,
                **(
                    {
                        "_debug": {
                            "qty": qty,
                            "entryPrice": entry_price,
                            "costBasis": cost_basis,
                            "currentPrice": current_price,
                            "latestTradePrice": latest_price,
                            "sparklineTail": sparkline[-3:] if sparkline else [],
                            "daysHeldFromActivities": days_held_map.get(_normalize_symbol(symbol)),
                            "liveTrailingStop": trailing_stop,
                            "liveTrailPercent": trail_percent,
                        }
                    }
                    if include_debug
                    else {}
                ),
            }
        )

    positions, summary = _enrich_positions_with_db_metrics(positions)
    payload = {
        "ok": bool(db_ready),
        "positions": positions,
        "summary": summary,
        "source": "alpaca"
        if calculation_source == "alpaca"
        else ("db+alpaca" if using_alpaca_overlay else "db-fallback"),
        "calculationSource": calculation_source,
        "db_source_of_truth": bool(db_source_of_truth),
        "db_ready": bool(db_ready),
        "reconciliation_required": bool(positions) and not bool(db_source_of_truth),
    }
    if include_debug:
        payload["_debug"] = {
            "symbols": symbols,
            "feed": os.getenv("ALPACA_DATA_FEED"),
            "latestPrices": latest_prices,
            "trailingStops": trailing_stop_map,
            "daysHeldFromActivities": days_held_map,
            "alpacaOverlayUsed": using_alpaca_overlay,
        }
    return _json_no_store(payload)


@server.route("/api/execute/overview")
def api_execute_overview():
    now_utc = datetime.now(timezone.utc)
    ny_tz = pytz.timezone("America/New_York")
    ny_now = now_utc.astimezone(ny_tz)
    window_start = ny_now.replace(hour=7, minute=0, second=0, microsecond=0)
    window_end = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)
    in_window = window_start <= ny_now <= window_end

    account = _account_latest_db()
    buying_power = account.get("buying_power") if account else None

    trades_df = load_trades_db(limit=500)
    submitted = int(len(trades_df)) if not trades_df.empty else 0
    filled = 0
    rejected = 0
    if not trades_df.empty and "status" in trades_df.columns:
        for status in trades_df["status"].fillna("").astype(str):
            status_lower = status.lower()
            if "reject" in status_lower:
                rejected += 1
            elif "fill" in status_lower or status_lower in {"closed", "filled"}:
                filled += 1

    live_positions = _positions_from_alpaca()
    open_count = int(len(live_positions))
    open_positions_source = "alpaca"
    if open_count == 0:
        open_df = load_open_trades_db()
        open_count = int(len(open_df)) if open_df is not None else 0
        open_positions_source = "db-fallback"

    skip_counts: dict[str, int] = {}
    lines = tail_log(execute_trades_log_path, limit=400)
    last_exec_ts = None
    for line in lines:
        match = re.search(r"EXECUTE_SKIP reason=([A-Z_]+)", line)
        if match:
            reason = match.group(1)
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
        if "Starting pre-market trade execution" in line or "Starting trade execution" in line:
            candidate = _parse_log_timestamp(line)
            if candidate:
                last_exec_ts = candidate
    payload = {
        "ok": True,
        "in_window": in_window,
        "buying_power": buying_power,
        "open_positions": open_count,
        "orders_submitted": submitted,
        "orders_filled": filled,
        "orders_rejected": rejected,
        "skip_counts": skip_counts,
        "last_execution": last_exec_ts.isoformat() if last_exec_ts else None,
        "ny_now": ny_now.isoformat(),
        "open_positions_source": open_positions_source,
    }
    return jsonify(payload)


@server.route("/api/execute/orders-summary")
def api_execute_orders_summary():
    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(hours=24)
    if trading_client is None:
        return jsonify(
            {
                "ok": False,
                "orders_filled": 0,
                "total_value": 0,
                "since_utc": since_utc.isoformat(),
                "until_utc": now_utc.isoformat(),
                "source": "alpaca",
            }
        )
    try:
        request = GetOrdersRequest(
            status="closed",
            after=since_utc,
            until=now_utc,
            direction="desc",
            limit=500,
        )
        orders = trading_client.get_orders(filter=request) or []
        filled_count = 0
        filled_value = 0.0
        for order in orders:
            filled_at = getattr(order, "filled_at", None)
            if filled_at is None:
                continue
            if filled_at < since_utc:
                continue
            side = getattr(order, "side", None)
            side_value = side.value if hasattr(side, "value") else str(side or "").lower()
            if side_value and side_value != "buy":
                continue
            qty = float(getattr(order, "filled_qty", 0) or 0)
            price = float(getattr(order, "filled_avg_price", 0) or 0)
            if qty <= 0 or price <= 0:
                continue
            filled_count += 1
            filled_value += qty * price
        return jsonify(
            {
                "ok": True,
                "orders_filled": filled_count,
                "total_value": filled_value,
                "since_utc": since_utc.isoformat(),
                "until_utc": now_utc.isoformat(),
                "source": "alpaca",
            }
        )
    except Exception as exc:
        logger.error("Failed to fetch Alpaca order summary: %s", exc)
        return jsonify(
            {
                "ok": False,
                "orders_filled": 0,
                "total_value": 0,
                "since_utc": since_utc.isoformat(),
                "until_utc": now_utc.isoformat(),
                "source": "alpaca",
            }
        )


_EXECUTE_STATUS_SCOPES = {"all", "open", "closed"}
_EXECUTE_LSX_SCOPES = {"all", "l", "s", "e", "x"}
_EXECUTE_LOG_STAGES = {"execute", "monitor", "pipeline"}
_EXECUTE_LOG_LEVELS = {"all", "errors", "warnings"}
_EXECUTE_STAGE_FILES = {
    "execute": "execute_trades.log",
    "monitor": "monitor.log",
    "pipeline": "pipeline.log",
}
_EXECUTE_MONITOR_EVENT_TYPES = {
    "SELL_SUBMIT",
    "SELL_FILL",
    "SELL_CANCELLED",
    "SELL_REJECTED",
    "SELL_EXPIRED",
    "TRAIL_ADJUST",
    "TRAIL_CANCEL",
}


def _execute_parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _execute_parse_scope(value: Any, *, allowed: set[str], default: str) -> str:
    normalized = str(value or default).strip().lower()
    if normalized in allowed:
        return normalized
    return default


def _execute_now_in_window() -> bool:
    now_utc = datetime.now(timezone.utc)
    ny_tz = pytz.timezone("America/New_York")
    ny_now = now_utc.astimezone(ny_tz)
    window_start = ny_now.replace(hour=7, minute=0, second=0, microsecond=0)
    window_end = ny_now.replace(hour=9, minute=30, second=0, microsecond=0)
    return window_start <= ny_now <= window_end


def _execute_iso_utc(value: Any) -> str | None:
    parsed = _coerce_datetime_utc(value)
    if parsed is None:
        return None
    return parsed.isoformat()


def _execute_pick_value(payload: Mapping[str, Any], candidates: list[str]) -> Any:
    for key in candidates:
        if key in payload and payload.get(key) not in (None, ""):
            return payload.get(key)
    return None


def _execute_to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _execute_latest_candidates_count() -> int:
    count_row = _db_fetch_one("SELECT COUNT(*) AS count FROM latest_screener_candidates")
    if count_row and count_row.get("count") is not None:
        return _execute_to_int(count_row.get("count"), 0)

    latest_run = _db_fetch_one("SELECT MAX(run_date) AS run_date FROM screener_candidates")
    run_date = latest_run.get("run_date") if latest_run else None
    if run_date is not None:
        row = _db_fetch_one(
            "SELECT COUNT(*) AS count FROM screener_candidates WHERE run_date = %(run_date)s",
            {"run_date": run_date},
        )
        if row and row.get("count") is not None:
            return _execute_to_int(row.get("count"), 0)

    latest_candidates_file = Path(BASE_DIR) / "data" / "latest_candidates.csv"
    if latest_candidates_file.exists():
        try:
            frame = pd.read_csv(latest_candidates_file)
            return int(len(frame.index))
        except Exception:
            return 0
    return 0


def _execute_last_run_from_logs() -> str | None:
    lines = tail_log(execute_trades_log_path, limit=800)
    if not lines:
        return None

    for line in reversed(lines):
        text = str(line or "").strip()
        if not text:
            continue
        if (
            "starting pre-market trade execution" in text.lower()
            or "starting trade execution" in text.lower()
            or "execute_summary" in text.lower()
            or "metrics_updated" in text.lower()
        ):
            parsed = _parse_log_timestamp(text)
            if parsed is not None:
                return parsed.replace(tzinfo=timezone.utc).isoformat()
            parsed = _coerce_datetime_utc(text)
            if parsed is not None:
                return parsed.isoformat()
    return None


def _execute_order_status_bucket(status: str, event_type: str = "") -> str:
    text = f"{status} {event_type}".strip().lower()
    if any(
        token in text for token in ("reject", "cancel", "cancelled", "expired", "fail", "error")
    ):
        return "REJECTED"
    if "partial" in text:
        return "PARTIAL"
    if any(token in text for token in ("fill", "closed", "done_for_day")):
        return "FILLED"
    if any(
        token in text
        for token in (
            "new",
            "accepted",
            "pending",
            "submit",
            "open",
            "held",
            "replace",
            "calculated",
            "stopped",
        )
    ):
        return "PENDING"
    normalized = str(status or "").strip().upper()
    return normalized or "PENDING"


def _execute_trailing_status_bucket(status: str, event_type: str = "") -> str:
    text = f"{status} {event_type}".strip().lower()
    if any(token in text for token in ("trigger", "fill", "closed", "cancel", "reject", "expired")):
        return "TRIGGERED"
    return "ACTIVE"


def _execute_status_scope_matches(status_bucket: str, status_scope: str) -> bool:
    scope = _execute_parse_scope(status_scope, allowed=_EXECUTE_STATUS_SCOPES, default="all")
    normalized = str(status_bucket or "").strip().upper()
    if scope == "all":
        return True
    if scope == "open":
        return normalized in {"PENDING", "PARTIAL", "ACTIVE"}
    return normalized in {"FILLED", "REJECTED", "TRIGGERED"}


def _execute_lsx_matches(
    lsx: str,
    *,
    side: str = "",
    message: str = "",
    notes: str = "",
    event_type: str = "",
    status: str = "",
) -> bool:
    scope = _execute_parse_scope(lsx, allowed=_EXECUTE_LSX_SCOPES, default="all")
    if scope == "all":
        return True

    side_text = str(side or "").strip().lower()
    haystack = " ".join(
        [
            side_text,
            str(message or "").lower(),
            str(notes or "").lower(),
            str(event_type or "").lower(),
            str(status or "").lower(),
        ]
    )
    if scope == "l":
        return side_text == "buy" or " long" in f" {haystack}" or "entry" in haystack
    if scope == "s":
        return side_text == "sell" or " short" in f" {haystack}" or "trail" in haystack
    if scope == "e":
        return any(token in haystack for token in ("entry", "submit", "open", "buy"))
    if scope == "x":
        return any(token in haystack for token in ("exit", "close", "sell", "trigger", "filled"))
    return True


def _execute_query_matches(query: str, values: list[Any]) -> bool:
    normalized_query = str(query or "").strip().lower()
    if not normalized_query:
        return True
    haystack = " ".join(str(value or "") for value in values).lower()
    return normalized_query in haystack


def _execute_summary_from_metrics_table() -> tuple[dict[str, Any] | None, str]:
    table_candidates = ("executor_runs", "execute_metrics", "execute_runs")
    for table in table_candidates:
        columns = _db_table_columns(table)
        if not columns:
            continue
        ts_col = _first_existing_column(
            columns,
            [
                "run_finished_utc",
                "last_run_utc",
                "finished_utc",
                "updated_at",
                "created_at",
                "timestamp",
            ],
        )
        if ts_col is None:
            continue
        row = _db_fetch_one(f"SELECT * FROM {table} ORDER BY {ts_col} DESC NULLS LAST LIMIT 1")
        if not row:
            continue

        in_window_raw = _execute_pick_value(row, ["in_window", "market_in_window", "within_window"])
        in_window = _execute_parse_bool(in_window_raw)
        payload = {
            "last_run_utc": _execute_iso_utc(
                _execute_pick_value(
                    row,
                    [
                        "last_run_utc",
                        "run_finished_utc",
                        "finished_utc",
                        "timestamp",
                        ts_col,
                    ],
                )
            ),
            "in_window": _execute_now_in_window() if in_window is None else bool(in_window),
            "candidates": _execute_to_int(
                _execute_pick_value(
                    row, ["candidates", "candidates_in", "symbols_in", "symbols", "rows"]
                ),
                0,
            ),
            "submitted": _execute_to_int(
                _execute_pick_value(row, ["orders_submitted", "submitted", "orders_total"]),
                0,
            ),
            "filled": _execute_to_int(
                _execute_pick_value(row, ["orders_filled", "filled", "fills"]),
                0,
            ),
            "rejected": _execute_to_int(
                _execute_pick_value(
                    row, ["orders_rejected", "rejected", "orders_canceled", "canceled", "cancelled"]
                ),
                0,
            ),
            "result_pl_usd": _to_float(
                _execute_pick_value(
                    row,
                    ["result_pl_usd", "net_pnl", "pnl", "realized_pnl", "profit_loss_usd"],
                )
            ),
        }
        if not payload["last_run_utc"]:
            payload["last_run_utc"] = _execute_last_run_from_logs()
        return payload, f"postgres:{table}"
    return None, "postgres:none"


def _execute_summary_from_metrics_file() -> tuple[dict[str, Any] | None, str]:
    path = Path(execute_metrics_path)
    if not path.exists():
        return None, "file:missing"
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "file:read_error"
    if not isinstance(loaded, Mapping):
        return None, "file:invalid_format"

    payload = dict(loaded)
    in_window = _execute_parse_bool(payload.get("in_window"))
    skip_block = payload.get("skips") or payload.get("skip_reasons") or {}
    if in_window is None and isinstance(skip_block, Mapping):
        in_window = _execute_to_int(skip_block.get("TIME_WINDOW"), 0) <= 0

    candidates = _execute_to_int(
        _execute_pick_value(
            payload, ["candidates", "candidates_in", "symbols_in", "symbols", "rows"]
        ),
        0,
    )
    if candidates <= 0:
        candidates = _execute_latest_candidates_count()

    normalized = {
        "last_run_utc": _execute_iso_utc(
            _execute_pick_value(
                payload,
                [
                    "last_run_utc",
                    "run_finished_utc",
                    "finished_utc",
                    "timestamp",
                    "run_utc",
                ],
            )
        ),
        "in_window": _execute_now_in_window() if in_window is None else bool(in_window),
        "candidates": candidates,
        "submitted": _execute_to_int(
            _execute_pick_value(payload, ["orders_submitted", "submitted", "orders_total"]),
            0,
        ),
        "filled": _execute_to_int(
            _execute_pick_value(payload, ["orders_filled", "filled", "fills"]),
            0,
        ),
        "rejected": _execute_to_int(
            _execute_pick_value(
                payload, ["orders_rejected", "rejected", "orders_canceled", "canceled", "cancelled"]
            ),
            0,
        ),
        "result_pl_usd": _to_float(
            _execute_pick_value(
                payload,
                ["result_pl_usd", "net_pnl", "pnl", "realized_pnl", "profit_loss_usd"],
            )
        ),
    }
    if not normalized["last_run_utc"]:
        normalized["last_run_utc"] = _execute_last_run_from_logs()
    return normalized, f"file:{path}"


def _execute_result_pl_from_db(last_run_dt: datetime | None) -> float | None:
    row = _db_fetch_one(
        """
        SELECT SUM(COALESCE(realized_pnl, 0)) AS result_pl_usd
        FROM trades
        WHERE status = 'CLOSED'
          AND (%(since)s IS NULL OR exit_time >= %(since)s)
        """,
        {"since": last_run_dt},
    )
    if row and row.get("result_pl_usd") is not None:
        value = _to_float(row.get("result_pl_usd"))
        if value is not None:
            return value

    fallback_row = _db_fetch_one(
        """
        SELECT SUM(COALESCE(net_pnl, pnl, 0)) AS result_pl_usd
        FROM executed_trades
        WHERE (%(since)s IS NULL OR COALESCE(exit_time, entry_time, created_at) >= %(since)s)
        """,
        {"since": last_run_dt},
    )
    if fallback_row and fallback_row.get("result_pl_usd") is not None:
        return _to_float(fallback_row.get("result_pl_usd"))
    return None


def _execute_side_label(side_value: Any, event_type: str = "") -> str:
    side = str(side_value or "").strip().lower()
    if not side:
        event_upper = str(event_type or "").strip().upper()
        if "SELL" in event_upper:
            side = "sell"
        elif "BUY" in event_upper:
            side = "buy"
    if side in {"buy", "long"}:
        return "BUY"
    if side in {"sell", "short"}:
        return "SELL"
    return side.upper()


def _execute_type_label(
    type_value: Any, event_type: str = "", raw_payload: Mapping[str, Any] | None = None
) -> str:
    text = str(type_value or "").strip().lower()
    raw = raw_payload or {}
    if not text:
        event_upper = str(event_type or "").strip().upper()
        if event_upper.startswith("TRAIL"):
            text = "trailing_stop"
        elif event_upper.startswith("BUY"):
            text = "limit"
        elif event_upper.startswith("SELL"):
            text = "market"
    if not text and (
        raw.get("trail_percent") not in (None, "")
        or raw.get("trail_price") not in (None, "")
        or raw.get("trail") not in (None, "")
    ):
        text = "trailing_stop"
    if not text:
        return "MARKET"
    text = text.replace("-", "_")
    mapping = {
        "trailing_stop": "TRAILING",
        "trailing": "TRAILING",
        "stop_limit": "STOP",
        "stop_loss": "STOP",
    }
    return mapping.get(text, text.upper())


def _execute_fmt_price(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "--"
    return f"${numeric:,.2f}"


def _execute_limit_stop_trail(raw_payload: Mapping[str, Any]) -> str:
    trail_percent = _to_float(
        raw_payload.get("trail_percent")
        or raw_payload.get("trail_pct")
        or raw_payload.get("trailPercent")
    )
    trail_price = _to_float(raw_payload.get("trail_price") or raw_payload.get("trailPrice"))
    stop_price = _to_float(raw_payload.get("stop_price") or raw_payload.get("stopPrice"))
    limit_price = _to_float(raw_payload.get("limit_price") or raw_payload.get("limitPrice"))

    if trail_percent is not None:
        return f"{trail_percent:.2f}%"
    if trail_price is not None:
        return _execute_fmt_price(trail_price)
    if stop_price is not None:
        return _execute_fmt_price(stop_price)
    if limit_price is not None:
        return _execute_fmt_price(limit_price)
    return "--"


def _execute_notes_from_row(row: Mapping[str, Any], raw_payload: Mapping[str, Any]) -> str:
    for key in ("reason", "note", "notes", "message", "msg", "detail", "description"):
        value = raw_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    event_type = str(row.get("event_type") or "").strip()
    if event_type:
        return event_type.replace("_", " ").title()
    return ""


def _execute_trail_text(raw_payload: Mapping[str, Any]) -> str:
    trail_percent = _to_float(
        raw_payload.get("trail_percent")
        or raw_payload.get("trail_pct")
        or raw_payload.get("trailPercent")
        or raw_payload.get("trail")
    )
    if trail_percent is not None:
        return f"{trail_percent:.2f}%"
    trail_price = _to_float(raw_payload.get("trail_price") or raw_payload.get("trailPrice"))
    if trail_price is not None:
        return _execute_fmt_price(trail_price)
    return "--"


def _execute_stop_price_from_raw(raw_payload: Mapping[str, Any]) -> float | None:
    stop_price = _to_float(raw_payload.get("stop_price") or raw_payload.get("stopPrice"))
    if stop_price is not None:
        return stop_price
    hwm = _to_float(raw_payload.get("hwm"))
    trail_percent = _to_float(
        raw_payload.get("trail_percent")
        or raw_payload.get("trail_pct")
        or raw_payload.get("trailPercent")
        or raw_payload.get("trail")
    )
    if hwm is not None and trail_percent is not None:
        return hwm * (1 - (trail_percent / 100.0))
    return None


def _execute_parent_leg(raw_payload: Mapping[str, Any], order_id: str) -> str:
    for key in ("parent_leg", "parent", "parent_order_id", "parent_id", "leg", "leg_id"):
        value = raw_payload.get(key)
        if value not in (None, ""):
            return str(value)
    if order_id:
        return order_id
    return "--"


def _execute_orders_from_alpaca(
    *, fetch_limit: int, status_scope: str
) -> tuple[list[dict[str, Any]], str]:
    if not _account_api_is_paper_mode():
        return [], "paper_mode_required"

    query_status = status_scope if status_scope in {"open", "closed"} else "all"
    payload, detail = _alpaca_rest_get_json(
        "/v2/orders",
        params={
            "status": query_status,
            "direction": "desc",
            "limit": int(fetch_limit),
            "nested": True,
        },
        timeout=20,
    )
    if detail != "ok":
        return [], f"alpaca:{detail}"

    if isinstance(payload, list):
        raw_orders = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("orders"), list):
        raw_orders = payload.get("orders", [])
    else:
        raw_orders = []

    rows: list[dict[str, Any]] = []
    for raw in raw_orders:
        if not isinstance(raw, Mapping):
            continue
        ts_raw = (
            raw.get("updated_at")
            or raw.get("filled_at")
            or raw.get("submitted_at")
            or raw.get("created_at")
        )
        ts_dt = _coerce_datetime_utc(ts_raw)
        ts_iso = ts_dt.isoformat() if ts_dt else str(ts_raw or "")

        side = _execute_side_label(raw.get("side"), str(raw.get("type") or ""))
        order_type = _execute_type_label(raw.get("type"), "", raw)
        status_bucket = _execute_order_status_bucket(str(raw.get("status") or ""), "")
        notes = str(raw.get("client_order_id") or "").strip()

        row = {
            "ts_utc": ts_iso,
            "symbol": str(raw.get("symbol") or "").strip().upper(),
            "side": side,
            "type": order_type,
            "qty": _to_float(raw.get("qty") or raw.get("filled_qty")),
            "limit_stop_trail": _execute_limit_stop_trail(raw),
            "status": status_bucket,
            "filled_avg": _to_float(raw.get("filled_avg_price")),
            "order_id": str(raw.get("id") or raw.get("order_id") or "").strip(),
            "notes": notes,
            "_ts": ts_dt,
            "_event_type": str(raw.get("type") or ""),
        }
        rows.append(row)

    rows.sort(
        key=lambda item: item.get("_ts") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return rows, "alpaca:/v2/orders"


def _execute_orders_from_db(fetch_limit: int) -> tuple[list[dict[str, Any]], str]:
    rows = _db_fetch_all(
        """
        SELECT DISTINCT ON (
            COALESCE(
                NULLIF(order_id, ''),
                symbol || ':' || COALESCE(event_time::text, '')
            )
        )
            event_time,
            symbol,
            qty,
            order_id,
            status,
            event_type,
            raw
        FROM order_events
        ORDER BY
            COALESCE(
                NULLIF(order_id, ''),
                symbol || ':' || COALESCE(event_time::text, '')
            ),
            event_time DESC NULLS LAST,
            event_id DESC
        LIMIT %(limit)s
        """,
        {"limit": int(fetch_limit)},
    )
    if not rows:
        return [], "postgres:order_events_empty"

    output: list[dict[str, Any]] = []
    for row in rows:
        raw_payload = _coerce_json_mapping(row.get("raw"))
        event_type = str(row.get("event_type") or "").strip().upper()
        ts_dt = _coerce_datetime_utc(row.get("event_time"))
        ts_iso = ts_dt.isoformat() if ts_dt else str(row.get("event_time") or "")

        side = _execute_side_label(
            raw_payload.get("side") or raw_payload.get("order_side"),
            event_type,
        )
        type_value = _execute_type_label(
            raw_payload.get("type") or raw_payload.get("order_type"),
            event_type,
            raw_payload,
        )
        status_bucket = _execute_order_status_bucket(str(row.get("status") or ""), event_type)
        notes = _execute_notes_from_row(row, raw_payload)

        output.append(
            {
                "ts_utc": ts_iso,
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "side": side,
                "type": type_value,
                "qty": _to_float(row.get("qty") or raw_payload.get("qty")),
                "limit_stop_trail": _execute_limit_stop_trail(raw_payload),
                "status": status_bucket,
                "filled_avg": _to_float(
                    raw_payload.get("filled_avg_price")
                    or raw_payload.get("avg_fill_price")
                    or raw_payload.get("filled_avg")
                ),
                "order_id": str(row.get("order_id") or "").strip(),
                "notes": notes,
                "_ts": ts_dt,
                "_event_type": event_type,
            }
        )

    output.sort(
        key=lambda item: item.get("_ts") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return output, "postgres:order_events"


def _execute_merge_order_rows(
    db_rows: list[dict[str, Any]], alpaca_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    sequence: list[str] = []

    for row in db_rows:
        key = str(row.get("order_id") or f"db:{row.get('symbol')}:{row.get('ts_utc')}")
        merged[key] = dict(row)
        sequence.append(key)

    for row in alpaca_rows:
        key = str(row.get("order_id") or f"alpaca:{row.get('symbol')}:{row.get('ts_utc')}")
        if key not in merged:
            merged[key] = dict(row)
            sequence.append(key)
            continue

        combined = dict(merged[key])
        for field in (
            "ts_utc",
            "symbol",
            "side",
            "type",
            "qty",
            "limit_stop_trail",
            "status",
            "filled_avg",
            "notes",
            "_ts",
            "_event_type",
        ):
            value = row.get(field)
            if value not in (None, "", "--"):
                combined[field] = value
        merged[key] = combined

    output = [merged[key] for key in sequence]
    output.sort(
        key=lambda item: item.get("_ts") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return output


def _execute_enrich_alpaca_orders_with_db(
    alpaca_rows: list[dict[str, Any]], db_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    db_by_id: dict[str, dict[str, Any]] = {}
    for row in db_rows:
        order_id = str(row.get("order_id") or "").strip()
        if not order_id:
            continue
        current = db_by_id.get(order_id)
        if current is None:
            db_by_id[order_id] = row
            continue
        current_ts = current.get("_ts")
        candidate_ts = row.get("_ts")
        if current_ts is None and candidate_ts is not None:
            db_by_id[order_id] = row
            continue
        if current_ts is not None and candidate_ts is not None and candidate_ts > current_ts:
            db_by_id[order_id] = row

    output: list[dict[str, Any]] = []
    for base in alpaca_rows:
        combined = dict(base)
        order_id = str(base.get("order_id") or "").strip()
        db_row = db_by_id.get(order_id)
        if db_row:
            if combined.get("notes") in (None, ""):
                combined["notes"] = db_row.get("notes") or ""
            if combined.get("limit_stop_trail") in (None, "", "--"):
                combined["limit_stop_trail"] = db_row.get("limit_stop_trail") or "--"
            if combined.get("filled_avg") in (None, "") and db_row.get("filled_avg") not in (
                None,
                "",
            ):
                combined["filled_avg"] = db_row.get("filled_avg")
            if combined.get("qty") in (None, "") and db_row.get("qty") not in (None, ""):
                combined["qty"] = db_row.get("qty")
            if combined.get("symbol") in (None, "") and db_row.get("symbol") not in (None, ""):
                combined["symbol"] = db_row.get("symbol")
            if combined.get("side") in (None, "") and db_row.get("side") not in (None, ""):
                combined["side"] = db_row.get("side")
            if combined.get("type") in (None, "") and db_row.get("type") not in (None, ""):
                combined["type"] = db_row.get("type")
            if combined.get("_event_type") in (None, "") and db_row.get("_event_type") not in (
                None,
                "",
            ):
                combined["_event_type"] = db_row.get("_event_type")
            if combined.get("ts_utc") in (None, "") and db_row.get("ts_utc") not in (None, ""):
                combined["ts_utc"] = db_row.get("ts_utc")
            if combined.get("_ts") is None and db_row.get("_ts") is not None:
                combined["_ts"] = db_row.get("_ts")
        output.append(combined)

    output.sort(
        key=lambda item: item.get("_ts") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return output


def _execute_filter_order_rows(
    rows: list[dict[str, Any]],
    *,
    status_scope: str,
    query: str,
    lsx: str,
    limit: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        status_bucket = str(row.get("status") or "")
        if not _execute_status_scope_matches(status_bucket, status_scope):
            continue
        if not _execute_query_matches(
            query,
            [
                row.get("ts_utc"),
                row.get("symbol"),
                row.get("side"),
                row.get("type"),
                row.get("qty"),
                row.get("limit_stop_trail"),
                row.get("status"),
                row.get("filled_avg"),
                row.get("order_id"),
                row.get("notes"),
            ],
        ):
            continue
        if not _execute_lsx_matches(
            lsx,
            side=str(row.get("side") or "").lower(),
            message=str(row.get("notes") or ""),
            notes=str(row.get("notes") or ""),
            event_type=str(row.get("_event_type") or ""),
            status=status_bucket,
        ):
            continue
        output.append(
            {
                "ts_utc": row.get("ts_utc"),
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "type": row.get("type"),
                "qty": row.get("qty"),
                "limit_stop_trail": row.get("limit_stop_trail") or "--",
                "status": row.get("status"),
                "filled_avg": row.get("filled_avg"),
                "order_id": row.get("order_id"),
                "notes": row.get("notes") or "",
            }
        )
        if len(output) >= max(1, int(limit)):
            break
    return output


def _execute_trailing_from_db(fetch_limit: int) -> tuple[list[dict[str, Any]], str]:
    rows = _db_fetch_all(
        """
        SELECT DISTINCT ON (
            COALESCE(
                NULLIF(order_id, ''),
                symbol || ':' || COALESCE(event_time::text, '')
            )
        )
            symbol,
            qty,
            order_id,
            status,
            event_type,
            event_time,
            raw
        FROM order_events
        WHERE (
            event_type ILIKE 'TRAIL%%'
            OR COALESCE(raw, '{}'::jsonb) ? 'trail_percent'
            OR COALESCE(raw, '{}'::jsonb) ? 'trail_pct'
            OR COALESCE(raw, '{}'::jsonb) ? 'trailPercent'
            OR COALESCE(raw, '{}'::jsonb) ? 'trail_price'
            OR COALESCE(raw, '{}'::jsonb) ? 'trailPrice'
            OR COALESCE(raw, '{}'::jsonb) ? 'stop_price'
            OR COALESCE(raw, '{}'::jsonb) ? 'stopPrice'
        )
        ORDER BY
            COALESCE(
                NULLIF(order_id, ''),
                symbol || ':' || COALESCE(event_time::text, '')
            ),
            event_time DESC NULLS LAST,
            event_id DESC
        LIMIT %(limit)s
        """,
        {"limit": int(fetch_limit)},
    )
    if not rows:
        return [], "postgres:order_events_empty"

    output: list[dict[str, Any]] = []
    for row in rows:
        raw_payload = _coerce_json_mapping(row.get("raw"))
        event_type = str(row.get("event_type") or "").strip().upper()
        status_bucket = _execute_trailing_status_bucket(str(row.get("status") or ""), event_type)
        ts_dt = _coerce_datetime_utc(row.get("event_time"))

        output.append(
            {
                "symbol": str(row.get("symbol") or "").strip().upper(),
                "qty": _to_float(row.get("qty") or raw_payload.get("qty")),
                "trail": _execute_trail_text(raw_payload),
                "stop_price": _execute_stop_price_from_raw(raw_payload),
                "status": status_bucket,
                "parent_leg": _execute_parent_leg(
                    raw_payload, str(row.get("order_id") or "").strip()
                ),
                "_ts": ts_dt,
                "_side": _execute_side_label(raw_payload.get("side"), event_type),
                "_event_type": event_type,
            }
        )

    output.sort(
        key=lambda item: item.get("_ts") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return output, "postgres:order_events"


def _execute_trailing_from_alpaca(
    *, fetch_limit: int, status_scope: str
) -> tuple[list[dict[str, Any]], str]:
    if not _account_api_is_paper_mode():
        return [], "paper_mode_required"

    query_status = status_scope if status_scope in {"open", "closed"} else "all"
    payload, detail = _alpaca_rest_get_json(
        "/v2/orders",
        params={
            "status": query_status,
            "direction": "desc",
            "limit": int(fetch_limit),
            "nested": True,
        },
        timeout=20,
    )
    if detail != "ok":
        return [], f"alpaca:{detail}"

    if isinstance(payload, list):
        raw_orders = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("orders"), list):
        raw_orders = payload.get("orders", [])
    else:
        raw_orders = []

    rows: list[dict[str, Any]] = []
    for raw in raw_orders:
        if not isinstance(raw, Mapping):
            continue
        type_text = str(raw.get("type") or "").strip().lower()
        has_trail_fields = (
            raw.get("trail_percent") not in (None, "")
            or raw.get("trail_pct") not in (None, "")
            or raw.get("trail_price") not in (None, "")
        )
        if "trailing" not in type_text and not has_trail_fields:
            continue

        ts_raw = (
            raw.get("updated_at")
            or raw.get("submitted_at")
            or raw.get("created_at")
            or raw.get("filled_at")
        )
        ts_dt = _coerce_datetime_utc(ts_raw)
        event_type = _execute_type_label(raw.get("type"), "TRAIL_SUBMIT", raw)
        status_bucket = _execute_trailing_status_bucket(str(raw.get("status") or ""), event_type)

        rows.append(
            {
                "symbol": str(raw.get("symbol") or "").strip().upper(),
                "qty": _to_float(raw.get("qty") or raw.get("filled_qty")),
                "trail": _execute_trail_text(raw),
                "stop_price": _execute_stop_price_from_raw(raw),
                "status": status_bucket,
                "parent_leg": _execute_parent_leg(raw, str(raw.get("id") or "").strip()),
                "_ts": ts_dt,
                "_side": _execute_side_label(raw.get("side"), event_type),
                "_event_type": event_type,
            }
        )

    rows.sort(
        key=lambda item: item.get("_ts") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return rows, "alpaca:/v2/orders"


def _execute_filter_trailing_rows(
    rows: list[dict[str, Any]],
    *,
    status_scope: str,
    query: str,
    lsx: str,
    limit: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        status_bucket = str(row.get("status") or "")
        if not _execute_status_scope_matches(status_bucket, status_scope):
            continue
        if not _execute_query_matches(
            query,
            [
                row.get("symbol"),
                row.get("qty"),
                row.get("trail"),
                row.get("stop_price"),
                row.get("status"),
                row.get("parent_leg"),
            ],
        ):
            continue
        if not _execute_lsx_matches(
            lsx,
            side=str(row.get("_side") or "").lower(),
            message=str(row.get("parent_leg") or ""),
            notes=str(row.get("parent_leg") or ""),
            event_type=str(row.get("_event_type") or ""),
            status=status_bucket,
        ):
            continue
        output.append(
            {
                "symbol": row.get("symbol"),
                "qty": row.get("qty"),
                "trail": row.get("trail") or "--",
                "stop_price": row.get("stop_price"),
                "status": row.get("status"),
                "parent_leg": row.get("parent_leg") or "--",
            }
        )
        if len(output) >= max(1, int(limit)):
            break
    return output


def _execute_summary_counts_from_alpaca(last_run_dt: datetime | None) -> tuple[dict[str, int], str]:
    if not _account_api_is_paper_mode():
        return {"submitted": 0, "filled": 0, "rejected": 0}, "paper_mode_required"

    params: dict[str, Any] = {"status": "all", "direction": "desc", "limit": 500, "nested": True}
    if last_run_dt is not None:
        params["after"] = last_run_dt.isoformat()
    payload, detail = _alpaca_rest_get_json("/v2/orders", params=params, timeout=20)
    if detail != "ok":
        return {"submitted": 0, "filled": 0, "rejected": 0}, f"alpaca:{detail}"

    if isinstance(payload, list):
        raw_orders = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("orders"), list):
        raw_orders = payload.get("orders", [])
    else:
        raw_orders = []

    submitted = 0
    filled = 0
    rejected = 0
    for raw in raw_orders:
        if not isinstance(raw, Mapping):
            continue
        submitted += 1
        status_bucket = _execute_order_status_bucket(
            str(raw.get("status") or ""), str(raw.get("type") or "")
        )
        if status_bucket == "FILLED":
            filled += 1
        if status_bucket == "REJECTED":
            rejected += 1
    return {"submitted": submitted, "filled": filled, "rejected": rejected}, "alpaca:/v2/orders"


def _execute_log_level(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"WARN", "WARNING"}:
        return "WARNING"
    if text in {"ERR", "ERROR"}:
        return "ERROR"
    if text == "SUCCESS":
        return "SUCCESS"
    if text == "INFO":
        return "INFO"
    normalized = _screener_normalize_log_level(text)
    if normalized == "WARN":
        return "WARNING"
    return normalized


def _execute_event_stage(event_type: str, raw_payload: Mapping[str, Any]) -> str:
    source = str(raw_payload.get("source") or "").strip().lower()
    event_upper = str(event_type or "").strip().upper()
    if source == "monitor_positions":
        return "monitor"
    if event_upper in _EXECUTE_MONITOR_EVENT_TYPES:
        return "monitor"
    return "execute"


def _execute_logs_from_order_events(
    stage: str, fetch_limit: int
) -> tuple[list[dict[str, Any]], str]:
    rows = _db_fetch_all(
        """
        SELECT event_time, event_type, symbol, qty, order_id, status, raw
        FROM order_events
        ORDER BY event_time DESC NULLS LAST
        LIMIT %(limit)s
        """,
        {"limit": int(fetch_limit)},
    )
    if not rows:
        return [], "postgres:order_events_empty"

    output: list[dict[str, Any]] = []
    for row in rows:
        raw_payload = _coerce_json_mapping(row.get("raw"))
        event_type = str(row.get("event_type") or "").strip().upper()
        event_stage = _execute_event_stage(event_type, raw_payload)
        if event_stage != stage:
            continue

        ts_dt = _coerce_datetime_utc(row.get("event_time"))
        message = _order_event_message(row, raw_payload)
        level_hint = raw_payload.get("level") or raw_payload.get("severity")
        level = _execute_log_level(
            _order_log_level(
                str(row.get("event_type") or ""),
                str(row.get("status") or ""),
                message,
                str(level_hint or ""),
            )
        )
        side = _execute_side_label(
            raw_payload.get("side") or raw_payload.get("order_side"),
            event_type,
        )
        output.append(
            {
                "ts_utc": _execute_iso_utc(ts_dt),
                "level": level,
                "message": message,
                "_ts": ts_dt,
                "_side": side.lower(),
            }
        )

    output.sort(
        key=lambda item: item.get("_ts") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return output, "postgres:order_events"


def _execute_log_file_payload(stage: str) -> tuple[str | None, str]:
    filename = _EXECUTE_STAGE_FILES.get(stage)
    if not filename:
        return None, "files:none"

    env_map = {
        "execute": ("PYTHONANYWHERE_EXECUTE_LOG_URL", "PYTHONANYWHERE_EXECUTE_LOG_PATH"),
        "monitor": ("PYTHONANYWHERE_MONITOR_LOG_URL", "PYTHONANYWHERE_MONITOR_LOG_PATH"),
        "pipeline": ("PYTHONANYWHERE_PIPELINE_LOG_URL", "PYTHONANYWHERE_PIPELINE_LOG_PATH"),
    }
    url_env, path_env = env_map.get(stage, ("", ""))
    remote_payload = _fetch_pythonanywhere_file(
        os.environ.get(url_env) if url_env else None,
        os.environ.get(path_env) if path_env else None,
    )
    if remote_payload:
        text, source = remote_payload
        return text, f"pythonanywhere:{source}"

    local_path = Path(BASE_DIR) / "logs" / filename
    if not local_path.exists():
        return None, f"files:missing:{local_path}"
    try:
        text = local_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, f"files:read_error:{local_path}"
    return text, f"local:{local_path}"


def _execute_logs_from_files(stage: str, fetch_limit: int) -> tuple[list[dict[str, Any]], str]:
    text, source = _execute_log_file_payload(stage)
    if not text:
        return [], source

    lines = [line for line in str(text).splitlines() if line.strip()]
    tail_limit = max(fetch_limit * 10, 400)
    parsed_rows: list[dict[str, Any]] = []
    for line in lines[-tail_limit:]:
        parsed = _screener_parse_log_line(line)
        if parsed:
            ts_dt = _coerce_datetime_utc(parsed.get("_ts") or parsed.get("ts_utc"))
            level = _execute_log_level(parsed.get("level"))
            message = str(parsed.get("message") or "").strip() or str(line).strip()
        else:
            ts_dt = _coerce_datetime_utc(_parse_log_timestamp(line))
            upper = str(line).upper()
            if "ERROR" in upper:
                level = "ERROR"
            elif "WARN" in upper:
                level = "WARNING"
            elif "SUCCESS" in upper:
                level = "SUCCESS"
            else:
                level = "INFO"
            message = str(line).strip()
        parsed_rows.append(
            {
                "ts_utc": _execute_iso_utc(ts_dt),
                "level": level,
                "message": message,
                "_ts": ts_dt,
                "_side": "",
            }
        )

    parsed_rows.sort(
        key=lambda item: item.get("_ts") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return parsed_rows, source


def _execute_filter_log_rows(
    rows: list[dict[str, Any]],
    *,
    level_filter: str,
    today_only: bool,
    query: str,
    lsx: str,
    limit: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    today = datetime.now(timezone.utc).date()
    for row in rows:
        level = _execute_log_level(row.get("level"))
        if level_filter == "errors" and level != "ERROR":
            continue
        if level_filter == "warnings" and level != "WARNING":
            continue

        ts_dt = row.get("_ts")
        if today_only:
            if ts_dt is None or ts_dt.date() != today:
                continue

        if not _execute_query_matches(
            query,
            [row.get("ts_utc"), level, row.get("message")],
        ):
            continue

        if not _execute_lsx_matches(
            lsx,
            side=str(row.get("_side") or "").lower(),
            message=str(row.get("message") or ""),
            notes=str(row.get("message") or ""),
            event_type="",
            status=level,
        ):
            continue

        output.append(
            {
                "ts_utc": row.get("ts_utc"),
                "level": level,
                "message": str(row.get("message") or "").strip(),
            }
        )
        if len(output) >= max(1, int(limit)):
            break
    return output


def _execute_summary_payload() -> dict[str, Any]:
    source = "none"
    source_detail_parts: list[str] = []

    summary_payload, summary_detail = _execute_summary_from_metrics_table()
    if summary_payload:
        source = "postgres"
        source_detail_parts.append(summary_detail)
    else:
        file_payload, file_detail = _execute_summary_from_metrics_file()
        summary_payload = file_payload
        source = "file-fallback" if file_payload else "none"
        source_detail_parts.append(f"{summary_detail};{file_detail}")

    if not summary_payload:
        summary_payload = {
            "last_run_utc": _execute_last_run_from_logs(),
            "in_window": _execute_now_in_window(),
            "candidates": _execute_latest_candidates_count(),
            "submitted": 0,
            "filled": 0,
            "rejected": 0,
            "result_pl_usd": None,
        }

    last_run_dt = _coerce_datetime_utc(summary_payload.get("last_run_utc"))
    alpaca_counts, alpaca_detail = _execute_summary_counts_from_alpaca(last_run_dt)
    source_detail_parts.append(alpaca_detail)

    submitted = _execute_to_int(summary_payload.get("submitted"), 0)
    filled = _execute_to_int(summary_payload.get("filled"), 0)
    rejected = _execute_to_int(summary_payload.get("rejected"), 0)
    if submitted <= 0:
        submitted = _execute_to_int(alpaca_counts.get("submitted"), 0)
    if filled <= 0:
        filled = _execute_to_int(alpaca_counts.get("filled"), 0)
    if rejected <= 0:
        rejected = _execute_to_int(alpaca_counts.get("rejected"), 0)

    candidates = _execute_to_int(summary_payload.get("candidates"), 0)
    if candidates <= 0:
        candidates = max(submitted, _execute_latest_candidates_count())

    result_pl = _to_float(summary_payload.get("result_pl_usd"))
    if result_pl is None:
        result_pl = _execute_result_pl_from_db(last_run_dt)

    in_window_value = _execute_parse_bool(summary_payload.get("in_window"))
    in_window = _execute_now_in_window() if in_window_value is None else bool(in_window_value)
    last_run_utc = (
        _execute_iso_utc(summary_payload.get("last_run_utc")) or _execute_last_run_from_logs()
    )

    if source == "none":
        source = "alpaca-fallback" if alpaca_counts.get("submitted", 0) > 0 else "local-fallback"

    return {
        "ok": bool(last_run_utc or submitted or filled or rejected or candidates),
        "last_run_utc": last_run_utc,
        "in_window": bool(in_window),
        "candidates": int(candidates),
        "submitted": int(submitted),
        "filled": int(filled),
        "rejected": int(rejected),
        "result_pl_usd": result_pl if result_pl is not None else 0.0,
        "source": source,
        "source_detail": ";".join(part for part in source_detail_parts if part),
    }


def _execute_orders_payload(
    *, status_scope: str, limit: int, query: str, lsx: str
) -> dict[str, Any]:
    fetch_limit = max(int(limit) * 8, 320)

    db_rows, db_detail = _execute_orders_from_db(fetch_limit)
    alpaca_rows, alpaca_detail = _execute_orders_from_alpaca(
        fetch_limit=fetch_limit, status_scope=status_scope
    )

    if alpaca_rows:
        merged_rows = _execute_enrich_alpaca_orders_with_db(alpaca_rows, db_rows)
        source = "alpaca+postgres_enriched"
        source_detail = f"{alpaca_detail};{db_detail}"
    elif db_rows:
        merged_rows = db_rows
        source = "postgres-fallback"
        source_detail = f"{alpaca_detail};{db_detail}"
    else:
        merged_rows = []
        source = "none"
        source_detail = f"{alpaca_detail};{db_detail}"

    rows = _execute_filter_order_rows(
        merged_rows,
        status_scope=status_scope,
        query=query,
        lsx=lsx,
        limit=limit,
    )

    return {
        "ok": bool(rows),
        "rows": rows,
        "source": source,
        "source_detail": source_detail,
    }


def _execute_trailing_stops_payload(
    *, status_scope: str, limit: int, query: str, lsx: str
) -> dict[str, Any]:
    fetch_limit = max(int(limit) * 8, 320)

    db_rows, db_detail = _execute_trailing_from_db(fetch_limit)
    if db_rows:
        source = "postgres"
        source_detail = db_detail
        staged_rows = db_rows
    else:
        alpaca_rows, alpaca_detail = _execute_trailing_from_alpaca(
            fetch_limit=fetch_limit,
            status_scope=status_scope,
        )
        staged_rows = alpaca_rows
        source = "alpaca" if alpaca_rows else "none"
        source_detail = f"{db_detail};{alpaca_detail}"

    rows = _execute_filter_trailing_rows(
        staged_rows,
        status_scope=status_scope,
        query=query,
        lsx=lsx,
        limit=limit,
    )

    return {
        "ok": bool(rows),
        "rows": rows,
        "source": source,
        "source_detail": source_detail,
    }


def _execute_logs_payload(
    *,
    stage: str,
    limit: int,
    level_filter: str,
    today_only: bool,
    query: str,
    lsx: str,
) -> dict[str, Any]:
    fetch_limit = max(int(limit) * 6, 240)

    if stage == "pipeline":
        db_rows_raw, db_detail = _screener_logs_from_db("pipeline", fetch_limit)
        db_rows = [
            {
                "ts_utc": row.get("ts_utc"),
                "level": _execute_log_level(row.get("level")),
                "message": str(row.get("message") or "").strip(),
                "_ts": row.get("_ts"),
                "_side": "",
            }
            for row in db_rows_raw
        ]
    else:
        db_rows, db_detail = _execute_logs_from_order_events(stage, fetch_limit)

    if db_rows:
        staged_rows = db_rows
        source = "postgres"
        source_detail = db_detail
    else:
        file_rows, file_detail = _execute_logs_from_files(stage, fetch_limit)
        staged_rows = file_rows
        source = (
            "pythonanywhere" if file_detail.startswith("pythonanywhere:") else "local-log-fallback"
        )
        if not staged_rows:
            source = "none"
        source_detail = f"{db_detail};{file_detail}"

    rows = _execute_filter_log_rows(
        staged_rows,
        level_filter=level_filter,
        today_only=today_only,
        query=query,
        lsx=lsx,
        limit=limit,
    )

    return {
        "ok": bool(rows),
        "stage": stage,
        "rows": rows,
        "source": source,
        "source_detail": source_detail,
    }


def _execute_audit_null_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    missing = 0
    for row in rows:
        value = row.get(key)
        if value is None:
            missing += 1
            continue
        if isinstance(value, str) and not value.strip():
            missing += 1
    return round(missing / max(1, len(rows)), 4)


def _execute_audit_timestamp_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    parsed: list[datetime] = []
    invalid = 0
    for row in rows:
        ts = _coerce_datetime_utc(row.get(key))
        if ts is None:
            invalid += 1
            continue
        parsed.append(ts)
    is_newest_first = all(parsed[idx] >= parsed[idx + 1] for idx in range(len(parsed) - 1))
    return {
        "invalid_ts_rows": invalid,
        "is_newest_first": bool(is_newest_first),
        "newest_ts_utc": parsed[0].isoformat() if parsed else None,
        "oldest_ts_utc": parsed[-1].isoformat() if parsed else None,
    }


def _execute_audit_payload(*, limit: int) -> dict[str, Any]:
    bounded_limit = max(50, min(500, int(limit)))
    now_utc = datetime.now(timezone.utc)

    summary_payload = _execute_summary_payload()
    orders_all_payload = _execute_orders_payload(
        status_scope="all", limit=bounded_limit, query="", lsx="all"
    )
    orders_open_payload = _execute_orders_payload(
        status_scope="open", limit=bounded_limit, query="", lsx="all"
    )
    orders_closed_payload = _execute_orders_payload(
        status_scope="closed", limit=bounded_limit, query="", lsx="all"
    )
    trailing_all_payload = _execute_trailing_stops_payload(
        status_scope="all", limit=bounded_limit, query="", lsx="all"
    )
    trailing_open_payload = _execute_trailing_stops_payload(
        status_scope="open", limit=bounded_limit, query="", lsx="all"
    )
    trailing_closed_payload = _execute_trailing_stops_payload(
        status_scope="closed", limit=bounded_limit, query="", lsx="all"
    )
    logs_execute_payload = _execute_logs_payload(
        stage="execute",
        limit=bounded_limit,
        level_filter="all",
        today_only=False,
        query="",
        lsx="all",
    )
    logs_monitor_payload = _execute_logs_payload(
        stage="monitor",
        limit=bounded_limit,
        level_filter="all",
        today_only=False,
        query="",
        lsx="all",
    )
    logs_pipeline_payload = _execute_logs_payload(
        stage="pipeline",
        limit=bounded_limit,
        level_filter="all",
        today_only=False,
        query="",
        lsx="all",
    )

    orders_all = list(orders_all_payload.get("rows") or [])
    orders_open = list(orders_open_payload.get("rows") or [])
    orders_closed = list(orders_closed_payload.get("rows") or [])
    trailing_all = list(trailing_all_payload.get("rows") or [])
    trailing_open = list(trailing_open_payload.get("rows") or [])
    trailing_closed = list(trailing_closed_payload.get("rows") or [])
    logs_execute = list(logs_execute_payload.get("rows") or [])
    logs_monitor = list(logs_monitor_payload.get("rows") or [])
    logs_pipeline = list(logs_pipeline_payload.get("rows") or [])

    def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            label = str(row.get(key) or "").strip().upper() or "UNKNOWN"
            counts[label] = counts.get(label, 0) + 1
        return counts

    orders_all_ids = {
        str(row.get("order_id") or "").strip()
        for row in orders_all
        if str(row.get("order_id") or "").strip()
    }
    orders_open_ids = {
        str(row.get("order_id") or "").strip()
        for row in orders_open
        if str(row.get("order_id") or "").strip()
    }
    orders_closed_ids = {
        str(row.get("order_id") or "").strip()
        for row in orders_closed
        if str(row.get("order_id") or "").strip()
    }
    overlap_count = len(orders_open_ids.intersection(orders_closed_ids))
    open_not_in_all = len(orders_open_ids - orders_all_ids)
    closed_not_in_all = len(orders_closed_ids - orders_all_ids)
    all_scope_limited = len(orders_all) >= bounded_limit

    last_run_dt = _coerce_datetime_utc(summary_payload.get("last_run_utc"))
    orders_since_last_run: list[dict[str, Any]] = []
    for row in orders_all:
        ts = _coerce_datetime_utc(row.get("ts_utc"))
        if ts is None:
            continue
        if last_run_dt is None or ts >= last_run_dt:
            orders_since_last_run.append(row)

    orders_since_status = _counter(orders_since_last_run, "status")
    submitted_since = len(orders_since_last_run)
    filled_since = int(orders_since_status.get("FILLED", 0))
    rejected_since = int(orders_since_status.get("REJECTED", 0))

    orders_ts_metrics = _execute_audit_timestamp_metrics(orders_all, "ts_utc")
    logs_execute_ts_metrics = _execute_audit_timestamp_metrics(logs_execute, "ts_utc")
    logs_monitor_ts_metrics = _execute_audit_timestamp_metrics(logs_monitor, "ts_utc")
    logs_pipeline_ts_metrics = _execute_audit_timestamp_metrics(logs_pipeline, "ts_utc")

    newest_orders_ts = _coerce_datetime_utc(orders_ts_metrics.get("newest_ts_utc"))
    newest_execute_log_ts = _coerce_datetime_utc(logs_execute_ts_metrics.get("newest_ts_utc"))

    quality = {
        "orders_filled_avg_null_rate": _execute_audit_null_rate(orders_all, "filled_avg"),
        "orders_notes_null_rate": _execute_audit_null_rate(orders_all, "notes"),
        "trailing_stop_price_null_rate": _execute_audit_null_rate(trailing_all, "stop_price"),
        "logs_execute_message_null_rate": _execute_audit_null_rate(logs_execute, "message"),
    }

    findings: list[dict[str, Any]] = []
    if overlap_count > 0:
        findings.append(
            {
                "severity": "high",
                "code": "orders_scope_overlap",
                "message": "Order IDs overlap between open and closed result sets.",
                "details": "Open/closed buckets should be disjoint when Alpaca status is authoritative.",
                "value": overlap_count,
            }
        )
    if (open_not_in_all > 0 or closed_not_in_all > 0) and not all_scope_limited:
        findings.append(
            {
                "severity": "warning",
                "code": "orders_scope_not_subset_all",
                "message": "Open/closed sets are not strict subsets of the all-scope set.",
                "details": f"open_not_in_all={open_not_in_all};closed_not_in_all={closed_not_in_all}",
                "value": open_not_in_all + closed_not_in_all,
            }
        )
    if quality["trailing_stop_price_null_rate"] >= 0.95 and len(trailing_all) > 0:
        findings.append(
            {
                "severity": "warning",
                "code": "trailing_stop_price_missing",
                "message": "Trailing stop price is frequently missing.",
                "details": "Active threshold may be unavailable from the current source payload.",
                "value": quality["trailing_stop_price_null_rate"],
            }
        )
    if quality["orders_filled_avg_null_rate"] >= 0.95 and len(orders_all) > 0:
        findings.append(
            {
                "severity": "info",
                "code": "orders_filled_avg_sparse",
                "message": "Filled average price is mostly null.",
                "details": "Expected when most orders are pending/rejected.",
                "value": quality["orders_filled_avg_null_rate"],
            }
        )
    if bool(summary_payload.get("submitted") or 0) == 0 and len(orders_all) > 0:
        findings.append(
            {
                "severity": "info",
                "code": "summary_zero_submitted_with_history",
                "message": "Summary submitted count is zero while historical orders exist.",
                "details": "May be valid when orders predate last_run_utc.",
                "value": len(orders_all),
            }
        )

    endpoint_health = {
        "summary": {"ok": bool(summary_payload.get("ok")), "source": summary_payload.get("source")},
        "orders_all": {
            "ok": bool(orders_all_payload.get("ok")),
            "source": orders_all_payload.get("source"),
        },
        "orders_open": {
            "ok": bool(orders_open_payload.get("ok")),
            "source": orders_open_payload.get("source"),
        },
        "orders_closed": {
            "ok": bool(orders_closed_payload.get("ok")),
            "source": orders_closed_payload.get("source"),
        },
        "trailing_all": {
            "ok": bool(trailing_all_payload.get("ok")),
            "source": trailing_all_payload.get("source"),
        },
        "logs_execute": {
            "ok": bool(logs_execute_payload.get("ok")),
            "source": logs_execute_payload.get("source"),
        },
        "logs_monitor": {
            "ok": bool(logs_monitor_payload.get("ok")),
            "source": logs_monitor_payload.get("source"),
        },
        "logs_pipeline": {
            "ok": bool(logs_pipeline_payload.get("ok")),
            "source": logs_pipeline_payload.get("source"),
        },
    }
    severity_counts = {
        "high": sum(1 for item in findings if str(item.get("severity") or "") == "high"),
        "warning": sum(1 for item in findings if str(item.get("severity") or "") == "warning"),
        "info": sum(1 for item in findings if str(item.get("severity") or "") == "info"),
    }
    overall_ok = severity_counts["high"] == 0

    return {
        "ok": bool(overall_ok),
        "fetched_at_utc": now_utc.isoformat(),
        "limit": bounded_limit,
        "endpoint_health": endpoint_health,
        "row_counts": {
            "orders_all": len(orders_all),
            "orders_open": len(orders_open),
            "orders_closed": len(orders_closed),
            "trailing_all": len(trailing_all),
            "trailing_open": len(trailing_open),
            "trailing_closed": len(trailing_closed),
            "logs_execute": len(logs_execute),
            "logs_monitor": len(logs_monitor),
            "logs_pipeline": len(logs_pipeline),
        },
        "summary": {
            "last_run_utc": summary_payload.get("last_run_utc"),
            "candidates": summary_payload.get("candidates"),
            "submitted": summary_payload.get("submitted"),
            "filled": summary_payload.get("filled"),
            "rejected": summary_payload.get("rejected"),
            "source": summary_payload.get("source"),
            "source_detail": summary_payload.get("source_detail"),
        },
        "status_breakdowns": {
            "orders_all": _counter(orders_all, "status"),
            "orders_open": _counter(orders_open, "status"),
            "orders_closed": _counter(orders_closed, "status"),
            "trailing_all": _counter(trailing_all, "status"),
            "logs_execute": _counter(logs_execute, "level"),
            "logs_monitor": _counter(logs_monitor, "level"),
            "logs_pipeline": _counter(logs_pipeline, "level"),
        },
        "timestamp_checks": {
            "orders_all": orders_ts_metrics,
            "logs_execute": logs_execute_ts_metrics,
            "logs_monitor": logs_monitor_ts_metrics,
            "logs_pipeline": logs_pipeline_ts_metrics,
        },
        "quality": quality,
        "cross_checks": {
            "orders_subset": {
                "all_scope_limited": bool(all_scope_limited),
                "open_subset_of_all": (
                    None if all_scope_limited else bool(orders_open_ids.issubset(orders_all_ids))
                ),
                "closed_subset_of_all": (
                    None if all_scope_limited else bool(orders_closed_ids.issubset(orders_all_ids))
                ),
                "open_closed_intersection_count": overlap_count,
                "open_not_in_all": open_not_in_all,
                "closed_not_in_all": closed_not_in_all,
            },
            "summary_vs_orders_since_last_run": {
                "summary_last_run_utc": summary_payload.get("last_run_utc"),
                "orders_since_last_run": submitted_since,
                "orders_since_last_run_filled": filled_since,
                "orders_since_last_run_rejected": rejected_since,
                "summary_submitted": summary_payload.get("submitted"),
                "summary_filled": summary_payload.get("filled"),
                "summary_rejected": summary_payload.get("rejected"),
            },
            "freshness": {
                "orders_newest_age_seconds": (
                    None
                    if newest_orders_ts is None
                    else int((now_utc - newest_orders_ts).total_seconds())
                ),
                "execute_logs_newest_age_seconds": (
                    None
                    if newest_execute_log_ts is None
                    else int((now_utc - newest_execute_log_ts).total_seconds())
                ),
            },
        },
        "severity_counts": severity_counts,
        "findings": findings,
    }


def _execute_sse_response(
    payload_factory: Callable[[], Mapping[str, Any] | list[Any]], *, interval_seconds: float
) -> Response:
    interval = max(0.5, float(interval_seconds))

    def _stream():
        last_payload_text = ""
        while True:
            try:
                payload = payload_factory()
                payload_text = json.dumps(payload, separators=(",", ":"), default=str)
            except GeneratorExit:
                raise
            except Exception as exc:
                logger.exception("Execute SSE payload failure: %s", exc)
                payload_text = json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    separators=(",", ":"),
                    default=str,
                )
            if payload_text != last_payload_text:
                last_payload_text = payload_text
                yield f"data: {payload_text}\n\n"
            else:
                yield f": keepalive {int(time.time())}\n\n"
            time.sleep(interval)

    response = Response(_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@server.route("/api/execute/summary")
def api_execute_summary():
    return _json_no_store(_execute_summary_payload())


@server.route("/api/execute/summary/stream")
def api_execute_summary_stream():
    return _execute_sse_response(_execute_summary_payload, interval_seconds=6.0)


@server.route("/api/execute/orders")
def api_execute_orders():
    status_scope = _execute_parse_scope(
        request.args.get("status"), allowed=_EXECUTE_STATUS_SCOPES, default="all"
    )
    limit = _parse_positive_int(request.args.get("limit"), default=200, minimum=1, maximum=200)
    query = str(request.args.get("q") or "").strip()
    lsx = _execute_parse_scope(request.args.get("lsx"), allowed=_EXECUTE_LSX_SCOPES, default="all")
    return _json_no_store(
        _execute_orders_payload(
            status_scope=status_scope,
            limit=int(limit),
            query=query,
            lsx=lsx,
        )
    )


@server.route("/api/execute/orders/stream")
def api_execute_orders_stream():
    status_scope = _execute_parse_scope(
        request.args.get("status"), allowed=_EXECUTE_STATUS_SCOPES, default="all"
    )
    limit = _parse_positive_int(request.args.get("limit"), default=200, minimum=1, maximum=200)
    query = str(request.args.get("q") or "").strip()
    lsx = _execute_parse_scope(request.args.get("lsx"), allowed=_EXECUTE_LSX_SCOPES, default="all")
    return _execute_sse_response(
        lambda: _execute_orders_payload(
            status_scope=status_scope,
            limit=int(limit),
            query=query,
            lsx=lsx,
        ),
        interval_seconds=4.0,
    )


@server.route("/api/execute/trailing_stops")
def api_execute_trailing_stops():
    status_scope = _execute_parse_scope(
        request.args.get("status"), allowed=_EXECUTE_STATUS_SCOPES, default="all"
    )
    limit = _parse_positive_int(request.args.get("limit"), default=200, minimum=1, maximum=200)
    query = str(request.args.get("q") or "").strip()
    lsx = _execute_parse_scope(request.args.get("lsx"), allowed=_EXECUTE_LSX_SCOPES, default="all")
    return _json_no_store(
        _execute_trailing_stops_payload(
            status_scope=status_scope,
            limit=int(limit),
            query=query,
            lsx=lsx,
        )
    )


@server.route("/api/execute/trailing_stops/stream")
def api_execute_trailing_stops_stream():
    status_scope = _execute_parse_scope(
        request.args.get("status"), allowed=_EXECUTE_STATUS_SCOPES, default="all"
    )
    limit = _parse_positive_int(request.args.get("limit"), default=200, minimum=1, maximum=200)
    query = str(request.args.get("q") or "").strip()
    lsx = _execute_parse_scope(request.args.get("lsx"), allowed=_EXECUTE_LSX_SCOPES, default="all")
    return _execute_sse_response(
        lambda: _execute_trailing_stops_payload(
            status_scope=status_scope,
            limit=int(limit),
            query=query,
            lsx=lsx,
        ),
        interval_seconds=5.0,
    )


@server.route("/api/execute/logs")
def api_execute_logs():
    stage = _execute_parse_scope(
        request.args.get("stage"), allowed=_EXECUTE_LOG_STAGES, default="execute"
    )
    limit = _parse_positive_int(request.args.get("limit"), default=200, minimum=1, maximum=300)
    level_filter = _execute_parse_scope(
        request.args.get("level"), allowed=_EXECUTE_LOG_LEVELS, default="all"
    )
    today_only = str(request.args.get("today") or "0").strip().lower() in {"1", "true", "yes", "on"}
    query = str(request.args.get("q") or "").strip()
    lsx = _execute_parse_scope(request.args.get("lsx"), allowed=_EXECUTE_LSX_SCOPES, default="all")
    return _json_no_store(
        _execute_logs_payload(
            stage=stage,
            limit=int(limit),
            level_filter=level_filter,
            today_only=today_only,
            query=query,
            lsx=lsx,
        )
    )


@server.route("/api/execute/logs/stream")
def api_execute_logs_stream():
    stage = _execute_parse_scope(
        request.args.get("stage"), allowed=_EXECUTE_LOG_STAGES, default="execute"
    )
    limit = _parse_positive_int(request.args.get("limit"), default=200, minimum=1, maximum=300)
    level_filter = _execute_parse_scope(
        request.args.get("level"), allowed=_EXECUTE_LOG_LEVELS, default="all"
    )
    today_only = str(request.args.get("today") or "0").strip().lower() in {"1", "true", "yes", "on"}
    query = str(request.args.get("q") or "").strip()
    lsx = _execute_parse_scope(request.args.get("lsx"), allowed=_EXECUTE_LSX_SCOPES, default="all")
    return _execute_sse_response(
        lambda: _execute_logs_payload(
            stage=stage,
            limit=int(limit),
            level_filter=level_filter,
            today_only=today_only,
            query=query,
            lsx=lsx,
        ),
        interval_seconds=3.0,
    )


def _execute_state_payload() -> dict[str, Any]:
    """Aggregate Execute tab data for a single SSE connection.

    PythonAnywhere WSGI apps have limited worker capacity; multiple long-lived SSE
    streams can starve the app. Keeping this as a single stream avoids that.
    """

    summary = _execute_summary_payload()
    orders = _execute_orders_payload(status_scope="all", limit=200, query="", lsx="all")
    trailing = _execute_trailing_stops_payload(status_scope="all", limit=200, query="", lsx="all")
    logs_execute = _execute_logs_payload(
        stage="execute",
        limit=200,
        level_filter="all",
        today_only=False,
        query="",
        lsx="all",
    )
    logs_monitor = _execute_logs_payload(
        stage="monitor",
        limit=200,
        level_filter="all",
        today_only=False,
        query="",
        lsx="all",
    )
    logs_pipeline = _execute_logs_payload(
        stage="pipeline",
        limit=200,
        level_filter="all",
        today_only=False,
        query="",
        lsx="all",
    )

    return {
        "ok": bool(summary.get("ok") or orders.get("ok") or trailing.get("ok")),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "orders": orders,
        "trailing_stops": trailing,
        "logs": {
            "execute": logs_execute,
            "monitor": logs_monitor,
            "pipeline": logs_pipeline,
        },
    }


@server.route("/api/execute/state")
def api_execute_state():
    return _json_no_store(_execute_state_payload())


@server.route("/api/execute/state/stream")
def api_execute_state_stream():
    return _execute_sse_response(_execute_state_payload, interval_seconds=5.0)


@server.route("/api/execute/audit")
def api_execute_audit():
    limit = _parse_positive_int(request.args.get("limit"), default=200, minimum=50, maximum=500)
    return _json_no_store(_execute_audit_payload(limit=int(limit)))


@server.route("/api/execute/audit/stream")
def api_execute_audit_stream():
    limit = _parse_positive_int(request.args.get("limit"), default=200, minimum=50, maximum=500)
    return _execute_sse_response(
        lambda: _execute_audit_payload(limit=int(limit)),
        interval_seconds=6.0,
    )


_DB_TABLE_COLUMNS_CACHE: dict[str, tuple[float, set[str]]] = {}
_DB_TABLE_COLUMNS_CACHE_TTL_SECONDS = 300.0
_SCREENER_RUN_SCOPE_CACHE: tuple[float, dict[str, Any]] | None = None
_SCREENER_RUN_SCOPE_CACHE_TTL_SECONDS = 20.0


def _db_table_columns(table_name: str) -> set[str]:
    cache_key = str(table_name).strip().lower()
    now = time.time()
    cached = _DB_TABLE_COLUMNS_CACHE.get(cache_key)
    if cached and (now - cached[0]) <= _DB_TABLE_COLUMNS_CACHE_TTL_SECONDS:
        return set(cached[1])

    rows = _db_fetch_all(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %(table_name)s
        """,
        {"table_name": cache_key},
    )
    columns = {
        str(row.get("column_name") or "").strip().lower() for row in rows if row.get("column_name")
    }
    _DB_TABLE_COLUMNS_CACHE[cache_key] = (now, set(columns))
    return columns


def _first_existing_column(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        lowered = str(candidate).strip().lower()
        if lowered in columns:
            return lowered
    return None


def _screener_parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "pass", "passed"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "fail", "failed"}:
        return False
    return None


def _screener_normalize_pct(value: Any) -> Optional[float]:
    numeric = _to_float(value)
    if numeric is None:
        return None
    if abs(numeric) <= 1:
        return float(numeric * 100.0)
    return float(numeric)


def _screener_iso_utc(value: Any) -> Optional[str]:
    dt_value = _coerce_datetime_utc(value)
    if dt_value is None:
        return None
    return dt_value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _screener_sort_rows_with_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -(_to_float(row.get("final_score")) or float("-inf")),
            str(row.get("symbol") or ""),
        ),
    )
    for index, row in enumerate(sorted_rows, start=1):
        row["rank"] = index
    return sorted_rows


def _screener_apply_pick_filter(
    rows: list[dict[str, Any]],
    *,
    filter_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        passed_gates = _screener_parse_bool(row.get("_passed_gates"))
        fail_reason = str(row.get("_gate_fail_reason") or "").strip()
        has_error = bool(fail_reason) or passed_gates is False
        if filter_key == "top10":
            if (row.get("rank") or 0) > 10:
                continue
        elif filter_key == "passed":
            if passed_gates is not True or has_error:
                continue
        elif filter_key == "errors":
            if not has_error:
                continue
        filtered.append(row)

    trimmed = filtered[:limit]
    for row in trimmed:
        row.pop("_passed_gates", None)
        row.pop("_gate_fail_reason", None)
    return trimmed


def _screener_latest_run_scope() -> dict[str, Any]:
    global _SCREENER_RUN_SCOPE_CACHE

    now = time.time()
    if _SCREENER_RUN_SCOPE_CACHE is not None:
        cached_at, cached_scope = _SCREENER_RUN_SCOPE_CACHE
        if (now - cached_at) <= _SCREENER_RUN_SCOPE_CACHE_TTL_SECONDS:
            return dict(cached_scope)

    scope: dict[str, Any] = {
        "run_date": None,
        "run_ts_utc": None,
        "status": "COMPLETE",
        "source_detail": "unavailable",
    }
    if not db.db_enabled():
        scope["status"] = "ERROR"
        scope["source_detail"] = "db_disabled"
        _SCREENER_RUN_SCOPE_CACHE = (now, dict(scope))
        return scope

    conn = db.get_db_conn()
    if conn is None:
        scope["status"] = "ERROR"
        scope["source_detail"] = "db_connect_failed"
        _SCREENER_RUN_SCOPE_CACHE = (now, dict(scope))
        return scope

    run_date = None
    run_ts_utc = None
    rc = None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT MAX(run_date) AS run_date FROM screener_candidates")
            latest_date = cursor.fetchone()
            run_date = latest_date[0] if latest_date else None
            if run_date is None:
                scope["status"] = "EMPTY"
                scope["source_detail"] = "screener_candidates:empty"
                _SCREENER_RUN_SCOPE_CACHE = (now, dict(scope))
                return scope

            try:
                cursor.execute(
                    """
                    SELECT run_ts_utc
                    FROM pipeline_health_app
                    WHERE mode = 'screener' AND run_date = %(run_date)s
                    ORDER BY run_ts_utc DESC NULLS LAST
                    LIMIT 1
                    """,
                    {"run_date": run_date},
                )
                run_ts_row = cursor.fetchone()
                run_ts_utc = run_ts_row[0] if run_ts_row else None
            except Exception:
                run_ts_utc = None
            if run_ts_utc is None:
                try:
                    cursor.execute(
                        """
                        SELECT MAX(run_ts_utc) AS run_ts_utc
                        FROM screener_run_map_app
                        """
                    )
                    run_ts_row = cursor.fetchone()
                    run_ts_utc = run_ts_row[0] if run_ts_row else None
                except Exception:
                    run_ts_utc = None

            try:
                cursor.execute(
                    """
                    SELECT rc
                    FROM pipeline_runs
                    WHERE run_date = %(run_date)s
                    ORDER BY created_at DESC NULLS LAST
                    LIMIT 1
                    """,
                    {"run_date": run_date},
                )
                pipeline_row = cursor.fetchone()
                rc = pipeline_row[0] if pipeline_row else None
            except Exception:
                rc = None
    except Exception as exc:
        logger.warning("[WARN] DB_QUERY_FAIL err=%s", exc)
        scope["status"] = "ERROR"
        scope["source_detail"] = f"db_query_fail:{exc}"
        _SCREENER_RUN_SCOPE_CACHE = (now, dict(scope))
        return scope
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if run_date is None:
        scope["status"] = "EMPTY"
        scope["source_detail"] = "screener_candidates:empty"
        _SCREENER_RUN_SCOPE_CACHE = (now, dict(scope))
        return scope
    scope["run_date"] = run_date

    scope["run_ts_utc"] = run_ts_utc

    if rc not in (None, 0):
        scope["status"] = "ERROR"

    scope["source_detail"] = f"run_date={_serialize_record(run_date)}"
    _SCREENER_RUN_SCOPE_CACHE = (now, dict(scope))
    return scope


def _screener_pick_rows_from_db(
    *,
    run_date: Any,
    run_ts_utc: Any,
    filter_key: str,
    limit: int,
    query: str,
) -> tuple[list[dict[str, Any]], str]:
    columns = _db_table_columns("screener_candidates")
    if not columns:
        return [], "screener_candidates:missing"

    score_col = _first_existing_column(columns, ["final_score", "score"])
    exchange_col = _first_existing_column(columns, ["exchange"])
    timestamp_col = _first_existing_column(columns, ["timestamp", "screened_at_utc", "created_at"])
    volume_col = _first_existing_column(columns, ["volume"])
    price_col = _first_existing_column(columns, ["close", "price", "last"])
    entry_col = _first_existing_column(columns, ["entry_price"])
    adv20_col = _first_existing_column(columns, ["adv20"])
    atrp_col = _first_existing_column(columns, ["atrp"])
    passed_col = _first_existing_column(columns, ["passed_gates"])
    fail_reason_col = _first_existing_column(columns, ["gate_fail_reason"])
    sma_ema_pct_col = _first_existing_column(columns, ["sma_ema_pct"])
    sma9_col = _first_existing_column(columns, ["sma9"])
    ema20_col = _first_existing_column(columns, ["ema20"])

    score_expr = f"c.{score_col}" if score_col else "NULL"
    exchange_expr = f"c.{exchange_col}" if exchange_col else "NULL"
    screened_expr = f"c.{timestamp_col}" if timestamp_col else "NULL"
    volume_expr = f"c.{volume_col}" if volume_col else "NULL"
    price_expr = f"c.{price_col}" if price_col else "NULL"
    entry_expr = f"c.{entry_col}" if entry_col else "NULL"
    adv20_expr = f"c.{adv20_col}" if adv20_col else "NULL"
    atrp_expr = f"c.{atrp_col}" if atrp_col else "NULL"
    passed_expr = f"c.{passed_col}" if passed_col else "NULL"
    fail_expr = f"c.{fail_reason_col}" if fail_reason_col else "NULL"

    if sma_ema_pct_col:
        sma_ema_expr = f"c.{sma_ema_pct_col}"
    elif sma9_col and ema20_col:
        sma_ema_expr = (
            f"CASE WHEN c.{ema20_col} IS NULL OR c.{ema20_col} = 0 THEN NULL "
            f"ELSE ((c.{sma9_col} - c.{ema20_col}) / c.{ema20_col}) * 100 END"
        )
    else:
        sma_ema_expr = "NULL"

    q_like = f"%{query}%"
    q_parts: list[str] = []
    if query:
        q_parts.append("COALESCE(c.symbol, '') ILIKE %(q_like)s")
        if exchange_col:
            q_parts.append(f"COALESCE(c.{exchange_col}::text, '') ILIKE %(q_like)s")
    q_sql = f" AND ({' OR '.join(q_parts)})" if q_parts else ""

    scope_sql = ""
    map_columns = _db_table_columns("screener_run_map_app")
    if run_ts_utc and {"run_ts_utc", "symbol"}.issubset(map_columns):
        scope_sql = (
            " AND EXISTS ("
            "SELECT 1 FROM screener_run_map_app m "
            "WHERE m.symbol = c.symbol AND m.run_ts_utc = %(run_ts_utc)s"
            ")"
        )

    order_sql = "c.symbol ASC"
    if score_col:
        order_sql = f"c.{score_col} DESC NULLS LAST, c.symbol ASC"

    fetch_limit = max(limit * 6, 120)
    params: dict[str, Any] = {
        "run_date": run_date,
        "run_ts_utc": run_ts_utc,
        "fetch_limit": fetch_limit,
        "q_like": q_like,
    }

    def _fetch_raw_rows(scope_fragment: str) -> list[dict[str, Any]]:
        return _db_fetch_all(
            f"""
            SELECT
                c.symbol AS symbol,
                {exchange_expr} AS exchange,
                {screened_expr} AS screened_at_utc,
                {score_expr} AS final_score,
                {volume_expr} AS volume,
                {price_expr} AS price,
                {entry_expr} AS entry_price,
                {adv20_expr} AS adv20,
                {atrp_expr} AS atrp,
                {sma_ema_expr} AS sma_ema_pct,
                {passed_expr} AS passed_gates,
                {fail_expr} AS gate_fail_reason
            FROM screener_candidates c
            WHERE c.run_date = %(run_date)s
            {scope_fragment}
            {q_sql}
            ORDER BY {order_sql}
            LIMIT %(fetch_limit)s
            """,
            params,
        )

    raw_rows = _fetch_raw_rows(scope_sql)
    scope_note = "run_ts_scope=exact" if scope_sql else "run_ts_scope=none"
    if not raw_rows and scope_sql:
        raw_rows = _fetch_raw_rows("")
        scope_note = "run_ts_scope=fallback_unscoped"

    staged_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        price = _to_float(row.get("price"))
        volume = _to_float(row.get("volume"))
        dollar_volume = None
        if price is not None and volume is not None:
            dollar_volume = float(price * volume)
        staged_rows.append(
            {
                "rank": None,
                "symbol": symbol,
                "exchange": str(row.get("exchange") or "").strip().upper(),
                "screened_at_utc": _screener_iso_utc(row.get("screened_at_utc")),
                "final_score": _to_float(row.get("final_score")),
                "volume": volume,
                "dollar_volume": dollar_volume,
                "price": price,
                "sma_ema_pct": _screener_normalize_pct(row.get("sma_ema_pct")),
                "entry_price": _to_float(row.get("entry_price")),
                "adv20": _to_float(row.get("adv20")),
                "atrp": _screener_normalize_pct(row.get("atrp")),
                "_passed_gates": row.get("passed_gates"),
                "_gate_fail_reason": row.get("gate_fail_reason"),
            }
        )

    ranked_rows = _screener_sort_rows_with_rank(staged_rows)
    return _screener_apply_pick_filter(ranked_rows, filter_key=filter_key, limit=limit), (
        f"postgres:screener_candidates run_date={_serialize_record(run_date)} {scope_note}"
    )


def _screener_pick_rows_from_file(
    *,
    filter_key: str,
    limit: int,
    query: str,
) -> tuple[list[dict[str, Any]], str, Optional[str]]:
    def _pick_rows_from_frame(
        frame: pd.DataFrame,
    ) -> tuple[list[dict[str, Any]], Optional[str], Optional[str]]:
        if frame.empty:
            return [], None, "empty"

        rename_candidates = {
            "symbol": ["symbol"],
            "exchange": ["exchange"],
            "run_date": ["run_date", "date"],
            "screened_at_utc": ["timestamp", "screened_at_utc", "created_at"],
            "rank": ["rank"],
            "final_score": ["final_score", "score"],
            "volume": ["volume"],
            "price": ["close", "price", "last", "price_close"],
            "entry_price": ["entry_price", "entry"],
            "adv20": ["adv20"],
            "atrp": ["atrp", "atr_pct", "atr_percent"],
            "passed_gates": ["passed_gates"],
            "gate_fail_reason": ["gate_fail_reason"],
            "sma_ema_pct": ["sma_ema_pct"],
            "sma9": ["sma9"],
            "ema20": ["ema20"],
        }

        def _find_column(possible_names: list[str]) -> Optional[str]:
            lowered_map = {str(column).strip().lower(): column for column in frame.columns}
            for name in possible_names:
                hit = lowered_map.get(name)
                if hit is not None:
                    return str(hit)
            return None

        symbol_col = _find_column(rename_candidates["symbol"])
        if symbol_col is None:
            return [], None, "missing_symbol"

        exchange_col = _find_column(rename_candidates["exchange"])
        run_date_col = _find_column(rename_candidates["run_date"])
        screened_col = _find_column(rename_candidates["screened_at_utc"])
        rank_col = _find_column(rename_candidates["rank"])
        score_col = _find_column(rename_candidates["final_score"])
        volume_col = _find_column(rename_candidates["volume"])
        price_col = _find_column(rename_candidates["price"])
        entry_col = _find_column(rename_candidates["entry_price"])
        adv20_col = _find_column(rename_candidates["adv20"])
        atrp_col = _find_column(rename_candidates["atrp"])
        passed_col = _find_column(rename_candidates["passed_gates"])
        fail_col = _find_column(rename_candidates["gate_fail_reason"])
        sma_ema_col = _find_column(rename_candidates["sma_ema_pct"])
        sma9_col = _find_column(rename_candidates["sma9"])
        ema20_col = _find_column(rename_candidates["ema20"])

        q_lower = query.lower()
        staged_rows: list[dict[str, Any]] = []
        for _, source_row in frame.iterrows():
            symbol = str(source_row.get(symbol_col) or "").strip().upper()
            if not symbol:
                continue
            exchange = (
                str(source_row.get(exchange_col) or "").strip().upper() if exchange_col else ""
            )
            if q_lower and q_lower not in symbol.lower() and q_lower not in exchange.lower():
                continue

            price = _to_float(source_row.get(price_col)) if price_col else None
            volume = _to_float(source_row.get(volume_col)) if volume_col else None
            sma_ema_pct = _to_float(source_row.get(sma_ema_col)) if sma_ema_col else None
            if sma_ema_pct is None and sma9_col and ema20_col:
                sma9 = _to_float(source_row.get(sma9_col))
                ema20 = _to_float(source_row.get(ema20_col))
                if sma9 is not None and ema20 not in (None, 0):
                    sma_ema_pct = ((sma9 - ema20) / ema20) * 100.0

            source_rank = _to_float(source_row.get(rank_col)) if rank_col else None
            if source_rank is not None:
                source_rank = max(1.0, source_rank)

            screened_value: Any = None
            if screened_col:
                screened_value = source_row.get(screened_col)
            elif run_date_col:
                screened_value = source_row.get(run_date_col)

            dollar_volume = None
            if price is not None and volume is not None:
                dollar_volume = float(price * volume)

            staged_rows.append(
                {
                    "rank": None,
                    "_source_rank": source_rank,
                    "symbol": symbol,
                    "exchange": exchange,
                    "screened_at_utc": _screener_iso_utc(screened_value),
                    "final_score": _to_float(source_row.get(score_col)) if score_col else None,
                    "volume": volume,
                    "dollar_volume": dollar_volume,
                    "price": price,
                    "sma_ema_pct": _screener_normalize_pct(sma_ema_pct),
                    "entry_price": _to_float(source_row.get(entry_col)) if entry_col else None,
                    "adv20": _to_float(source_row.get(adv20_col)) if adv20_col else None,
                    "atrp": _screener_normalize_pct(source_row.get(atrp_col)) if atrp_col else None,
                    "_passed_gates": source_row.get(passed_col) if passed_col else None,
                    "_gate_fail_reason": source_row.get(fail_col) if fail_col else None,
                }
            )

        if any(row.get("_source_rank") is not None for row in staged_rows):
            ranked_rows = sorted(
                staged_rows,
                key=lambda row: (
                    _to_float(row.get("_source_rank")) or float("inf"),
                    str(row.get("symbol") or ""),
                ),
            )
            for index, row in enumerate(ranked_rows, start=1):
                row["rank"] = index
        else:
            ranked_rows = _screener_sort_rows_with_rank(staged_rows)

        for row in ranked_rows:
            row.pop("_source_rank", None)

        picked_rows = _screener_apply_pick_filter(ranked_rows, filter_key=filter_key, limit=limit)
        run_ts_utc = None
        for candidate_column in [screened_col, run_date_col]:
            if not candidate_column:
                continue
            try:
                run_ts_candidate = pd.to_datetime(
                    frame[candidate_column], errors="coerce", utc=True
                ).max()
                if run_ts_candidate is not None and not pd.isna(run_ts_candidate):
                    run_ts_utc = (
                        run_ts_candidate.to_pydatetime()
                        .replace(microsecond=0)
                        .isoformat()
                        .replace("+00:00", "Z")
                    )
                    break
            except Exception:
                continue
        return picked_rows, run_ts_utc, None

    candidate_paths: list[Path] = [Path(BASE_DIR) / "data" / "latest_candidates.csv"]
    predictions_dir = Path(BASE_DIR) / "data" / "predictions"
    candidate_paths.append(predictions_dir / "latest.csv")
    if predictions_dir.exists():
        dated_prediction_files = list(predictions_dir.glob("????-??-??.csv"))
        dated_prediction_files.sort(key=lambda value: value.name, reverse=True)
        candidate_paths.extend(dated_prediction_files)

    seen_paths: set[str] = set()
    diagnostic_details: list[str] = []
    for path in candidate_paths:
        path_key = str(path.resolve()) if path.exists() else str(path)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)

        if not path.exists():
            diagnostic_details.append(f"{path.name}:missing")
            continue

        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            diagnostic_details.append(f"{path.name}:read_error:{exc}")
            continue

        parsed_rows, run_ts_utc, parse_error = _pick_rows_from_frame(frame)
        if parse_error is None:
            if run_ts_utc is None:
                stem_match = re.match(r"^(\d{4}-\d{2}-\d{2})$", path.stem)
                if stem_match:
                    run_ts_utc = f"{stem_match.group(1)}T00:00:00Z"
            return parsed_rows, f"file:{path}", run_ts_utc

        diagnostic_details.append(f"{path.name}:{parse_error}")

    if diagnostic_details:
        return [], ";".join(diagnostic_details[:8]), None
    return [], "picks_file_fallback:missing", None


def _screener_resolve_run_date_from_run_ts(
    raw_value: str | None,
) -> tuple[Optional[Any], Optional[Any]]:
    if not raw_value:
        return None, None
    parsed_ts = _coerce_datetime_utc(raw_value)
    if parsed_ts is None:
        return None, None

    health_row = _db_fetch_one(
        """
        SELECT run_date, run_ts_utc
        FROM pipeline_health_app
        WHERE run_ts_utc <= %(run_ts_utc)s
        ORDER BY run_ts_utc DESC NULLS LAST
        LIMIT 1
        """,
        {"run_ts_utc": parsed_ts},
    )
    if health_row and health_row.get("run_date") is not None:
        return health_row.get("run_date"), health_row.get("run_ts_utc")
    return parsed_ts.date(), parsed_ts


def _screener_latest_run_ts_for_date(run_date: Any) -> Optional[str]:
    if run_date is None:
        return None
    row = _db_fetch_one(
        """
        SELECT run_ts_utc
        FROM pipeline_health_app
        WHERE run_date = %(run_date)s
        ORDER BY run_ts_utc DESC NULLS LAST
        LIMIT 1
        """,
        {"run_date": run_date},
    )
    value = row.get("run_ts_utc") if row else None
    return _screener_iso_utc(value)


def _screener_backtest_rows_from_db(
    *,
    run_date: Any,
    run_ts_utc: Any,
    window: str,
    limit: int,
    query: str,
) -> tuple[list[dict[str, Any]], str]:
    columns = _db_table_columns("backtest_results")
    if not columns:
        return [], "backtest_results:missing"

    symbol_col = _first_existing_column(columns, ["symbol"])
    if symbol_col is None:
        return [], "backtest_results:missing_symbol"

    run_date_col = _first_existing_column(columns, ["run_date"])
    window_col = _first_existing_column(columns, ["window", "range", "period"])
    trades_col = _first_existing_column(columns, ["trades"])
    win_rate_col = _first_existing_column(columns, ["win_rate", "win_rate_pct"])
    avg_return_col = _first_existing_column(
        columns, ["avg_return_pct", "avg_return", "expectancy", "return_pct"]
    )
    pl_ratio_col = _first_existing_column(columns, ["pl_ratio", "profit_factor"])
    max_dd_col = _first_existing_column(columns, ["max_dd_pct", "max_drawdown", "max_drawdown_pct"])
    avg_hold_col = _first_existing_column(columns, ["avg_hold_days", "hold_days", "avg_days_held"])
    total_pl_col = _first_existing_column(columns, ["total_pl_usd", "net_pnl", "total_pl", "pnl"])

    def _expr(column: Optional[str]) -> str:
        return f"b.{column}" if column else "NULL"

    where_parts = ["1=1"]
    params: dict[str, Any] = {
        "limit": max(limit, 1),
        "window": window,
        "q_like": f"%{query}%",
        "run_ts_utc": run_ts_utc,
    }
    if run_date is not None and run_date_col:
        where_parts.append(f"b.{run_date_col} = %(run_date)s")
        params["run_date"] = run_date
    if window != "ALL" and window_col:
        where_parts.append(f"UPPER(COALESCE(b.{window_col}::text, '')) = %(window)s")
    if query:
        where_parts.append(f"COALESCE(b.{symbol_col}::text, '') ILIKE %(q_like)s")

    scope_sql = ""
    map_columns = _db_table_columns("screener_run_map_app")
    if run_ts_utc and {"run_ts_utc", "symbol"}.issubset(map_columns):
        scope_sql = (
            " AND EXISTS ("
            "SELECT 1 FROM screener_run_map_app m "
            "WHERE m.run_ts_utc = %(run_ts_utc)s "
            f"AND UPPER(COALESCE(m.symbol::text, '')) = UPPER(COALESCE(b.{symbol_col}::text, ''))"
            ")"
        )
    else:
        candidate_columns = _db_table_columns("screener_candidates")
        candidate_symbol_col = _first_existing_column(candidate_columns, ["symbol"])
        candidate_run_date_col = _first_existing_column(candidate_columns, ["run_date"])
        if candidate_symbol_col:
            candidate_where = [
                f"UPPER(COALESCE(c.{candidate_symbol_col}::text, '')) = UPPER(COALESCE(b.{symbol_col}::text, ''))"
            ]
            if run_date is not None and candidate_run_date_col:
                candidate_where.append(f"c.{candidate_run_date_col} = %(run_date)s")
            scope_sql = f" AND EXISTS (SELECT 1 FROM screener_candidates c WHERE {' AND '.join(candidate_where)})"

    order_sql = "b.symbol ASC"
    if total_pl_col:
        order_sql = f"b.{total_pl_col} DESC NULLS LAST, b.{symbol_col} ASC"

    rows = _db_fetch_all(
        f"""
        SELECT
            b.{symbol_col} AS symbol,
            {_expr(window_col)} AS window,
            {_expr(trades_col)} AS trades,
            {_expr(win_rate_col)} AS win_rate_pct,
            {_expr(avg_return_col)} AS avg_return_pct,
            {_expr(pl_ratio_col)} AS pl_ratio,
            {_expr(max_dd_col)} AS max_dd_pct,
            {_expr(avg_hold_col)} AS avg_hold_days,
            {_expr(total_pl_col)} AS total_pl_usd
        FROM backtest_results b
        WHERE {" AND ".join(where_parts)}
          {scope_sql}
        ORDER BY {order_sql}
        LIMIT %(limit)s
        """,
        params,
    )

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        resolved_window = str(row.get("window") or window).strip().upper() or window
        if resolved_window not in {"3M", "6M", "1Y", "ALL"}:
            resolved_window = window
        normalized_rows.append(
            {
                "symbol": symbol,
                "window": resolved_window,
                "trades": _to_float(row.get("trades")),
                "win_rate_pct": _screener_normalize_pct(row.get("win_rate_pct")),
                "avg_return_pct": _screener_normalize_pct(row.get("avg_return_pct")),
                "pl_ratio": _to_float(row.get("pl_ratio")),
                "max_dd_pct": _screener_normalize_pct(row.get("max_dd_pct")),
                "avg_hold_days": _to_float(row.get("avg_hold_days")),
                "total_pl_usd": _to_float(row.get("total_pl_usd")),
            }
        )
    return normalized_rows[:limit], (
        f"postgres:backtest_results run_date={_serialize_record(run_date)} window={window}"
    )


def _screener_backtest_rows_from_file(
    *,
    window: str,
    limit: int,
    query: str,
) -> tuple[list[dict[str, Any]], str]:
    path = Path(BASE_DIR) / "data" / "backtest_results.csv"
    if not path.exists():
        return [], "backtest_results.csv:missing"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return [], f"backtest_results.csv:read_error:{exc}"
    if frame.empty:
        return [], "backtest_results.csv:empty"

    lowered_map = {str(column).strip().lower(): str(column) for column in frame.columns}

    def _col(*names: str) -> Optional[str]:
        for name in names:
            if name in lowered_map:
                return lowered_map[name]
        return None

    symbol_col = _col("symbol")
    if symbol_col is None:
        return [], "backtest_results.csv:missing_symbol"

    window_col = _col("window", "range", "period")
    trades_col = _col("trades")
    win_rate_col = _col("win_rate", "win_rate_pct")
    avg_return_col = _col("avg_return_pct", "avg_return", "expectancy", "return_pct")
    pl_ratio_col = _col("pl_ratio", "profit_factor")
    max_dd_col = _col("max_dd_pct", "max_drawdown", "max_drawdown_pct")
    avg_hold_col = _col("avg_hold_days", "hold_days", "avg_days_held")
    total_pl_col = _col("total_pl_usd", "net_pnl", "total_pl", "pnl")

    query_lower = query.lower()
    rows: list[dict[str, Any]] = []
    for _, source_row in frame.iterrows():
        symbol = str(source_row.get(symbol_col) or "").strip().upper()
        if not symbol:
            continue
        if query_lower and query_lower not in symbol.lower():
            continue
        resolved_window = (
            str(source_row.get(window_col) or window).strip().upper() if window_col else window
        )
        if resolved_window not in {"3M", "6M", "1Y", "ALL"}:
            resolved_window = window
        if window != "ALL" and resolved_window != window:
            continue
        rows.append(
            {
                "symbol": symbol,
                "window": resolved_window,
                "trades": _to_float(source_row.get(trades_col)) if trades_col else None,
                "win_rate_pct": _screener_normalize_pct(source_row.get(win_rate_col))
                if win_rate_col
                else None,
                "avg_return_pct": _screener_normalize_pct(source_row.get(avg_return_col))
                if avg_return_col
                else None,
                "pl_ratio": _to_float(source_row.get(pl_ratio_col)) if pl_ratio_col else None,
                "max_dd_pct": _screener_normalize_pct(source_row.get(max_dd_col))
                if max_dd_col
                else None,
                "avg_hold_days": _to_float(source_row.get(avg_hold_col)) if avg_hold_col else None,
                "total_pl_usd": _to_float(source_row.get(total_pl_col)) if total_pl_col else None,
            }
        )

    rows.sort(key=lambda item: _to_float(item.get("total_pl_usd")) or float("-inf"), reverse=True)
    return rows[:limit], f"file:{path}"


def _screener_score_breakdown_short(value: Any) -> str:
    if value in (None, ""):
        return "--"

    parsed: Any = None
    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return "--"
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if parsed is None:
            token_matches = re.findall(
                r"([A-Za-z][A-Za-z0-9_]+)\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+)",
                text,
            )
            if token_matches:
                parsed = {key: float(score) for key, score in token_matches}
            else:
                return text[:72]

    if not isinstance(parsed, dict):
        return str(value)[:72]

    pretty_names: dict[str, str] = {
        "ms": "Momentum",
        "momentum": "Momentum",
        "macd": "Momentum",
        "rsi": "Momentum",
        "vol": "Vol",
        "liquidity": "Vol",
        "liq": "Vol",
        "trend": "Trend",
        "ma_stack": "Trend",
        "adx": "Trend",
    }
    parts: list[tuple[str, float]] = []
    for raw_key, raw_score in parsed.items():
        score = _to_float(raw_score)
        if score is None:
            continue
        key = str(raw_key).strip().lower()
        key = key.replace("_z", "")
        label = pretty_names.get(key)
        if label is None:
            label = str(raw_key).replace("_", " ").strip().title()
        parts.append((label, score))
    if not parts:
        return "--"

    parts.sort(key=lambda item: abs(item[1]), reverse=True)
    snippets: list[str] = []
    for label, score in parts[:3]:
        snippets.append(f"{label}:{score:.0f}")
    return " ".join(snippets)


def _screener_infer_metric_gates(
    *,
    passed_gates: Any,
    gate_fail_reason: Any,
) -> tuple[str, str, str]:
    passed_value = _screener_parse_bool(passed_gates)
    reason = str(gate_fail_reason or "").strip().lower()
    liquidity = "PASS"
    volatility = "PASS"
    trend = "PASS"

    if passed_value is False and not reason:
        liquidity = "FAIL"
        volatility = "FAIL"
        trend = "FAIL"
        return liquidity, volatility, trend

    if any(token in reason for token in ("liq", "liquid", "adv", "volume", "dollar")):
        liquidity = "FAIL"
    if any(token in reason for token in ("vol", "atr", "sigma", "range")):
        volatility = "FAIL"
    if any(token in reason for token in ("trend", "ema", "sma", "ma stack", "moving average")):
        trend = "FAIL"

    if passed_value is False and liquidity == "PASS" and volatility == "PASS" and trend == "PASS":
        trend = "FAIL"
    return liquidity, volatility, trend


def _screener_metrics_rows_from_db(
    *,
    run_date: Any,
    run_ts_utc: Any,
    filter_key: str,
    limit: int,
    query: str,
) -> tuple[list[dict[str, Any]], str]:
    columns = _db_table_columns("screener_candidates")
    if not columns:
        return [], "screener_candidates:missing"

    score_breakdown_col = _first_existing_column(columns, ["score_breakdown"])
    final_score_col = _first_existing_column(columns, ["final_score", "score"])
    passed_col = _first_existing_column(columns, ["passed_gates"])
    fail_col = _first_existing_column(columns, ["gate_fail_reason"])
    source_col = _first_existing_column(columns, ["source"])
    entry_col = _first_existing_column(columns, ["entry_price"])
    adv20_col = _first_existing_column(columns, ["adv20"])
    atrp_col = _first_existing_column(columns, ["atrp"])
    exchange_col = _first_existing_column(columns, ["exchange"])

    def _expr(column: Optional[str]) -> str:
        return f"c.{column}" if column else "NULL"

    q_parts: list[str] = []
    if query:
        q_parts.append("COALESCE(c.symbol, '') ILIKE %(q_like)s")
        if source_col:
            q_parts.append(f"COALESCE(c.{source_col}::text, '') ILIKE %(q_like)s")
        if score_breakdown_col:
            q_parts.append(f"COALESCE(c.{score_breakdown_col}::text, '') ILIKE %(q_like)s")
    q_sql = f" AND ({' OR '.join(q_parts)})" if q_parts else ""

    scope_sql = ""
    map_columns = _db_table_columns("screener_run_map_app")
    if run_ts_utc and {"run_ts_utc", "symbol"}.issubset(map_columns):
        scope_sql = (
            " AND EXISTS ("
            "SELECT 1 FROM screener_run_map_app m "
            "WHERE m.symbol = c.symbol AND m.run_ts_utc = %(run_ts_utc)s"
            ")"
        )

    fetch_limit = max(limit * 6, 120)
    raw_rows = _db_fetch_all(
        f"""
        SELECT
            c.symbol AS symbol,
            {_expr(score_breakdown_col)} AS score_breakdown,
            {_expr(final_score_col)} AS final_score,
            {_expr(passed_col)} AS passed_gates,
            {_expr(fail_col)} AS gate_fail_reason,
            {_expr(source_col)} AS source_label,
            {_expr(entry_col)} AS entry_price,
            {_expr(adv20_col)} AS adv20,
            {_expr(atrp_col)} AS atrp,
            {_expr(exchange_col)} AS exchange
        FROM screener_candidates c
        WHERE c.run_date = %(run_date)s
        {scope_sql}
        {q_sql}
        ORDER BY c.symbol ASC
        LIMIT %(fetch_limit)s
        """,
        {
            "run_date": run_date,
            "run_ts_utc": run_ts_utc,
            "q_like": f"%{query}%",
            "fetch_limit": fetch_limit,
        },
    )

    staged: list[dict[str, Any]] = []
    for row in raw_rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        final_score = _to_float(row.get("final_score"))
        liquidity_gate, volatility_gate, trend_gate = _screener_infer_metric_gates(
            passed_gates=row.get("passed_gates"),
            gate_fail_reason=row.get("gate_fail_reason"),
        )
        bars_complete = "YES"
        if (
            _to_float(row.get("entry_price")) in (None, 0)
            or _to_float(row.get("adv20")) in (None, 0)
            or _to_float(row.get("atrp")) in (None, 0)
            or not str(row.get("exchange") or "").strip()
        ):
            bars_complete = "NO"

        confidence = "Low"
        if final_score is not None and final_score >= 80:
            confidence = "High"
        elif final_score is not None and final_score >= 60:
            confidence = "Medium"

        source_label = str(row.get("source_label") or "").strip()
        if not source_label:
            source_label = "DB"

        record = {
            "symbol": symbol,
            "score_breakdown_short": _screener_score_breakdown_short(row.get("score_breakdown")),
            "liquidity_gate": liquidity_gate,
            "volatility_gate": volatility_gate,
            "trend_gate": trend_gate,
            "bars_complete": bars_complete,
            "confidence": confidence,
            "source_label": source_label,
            "_gate_fail_reason": str(row.get("gate_fail_reason") or "").strip().lower(),
        }

        if filter_key == "gate_failures":
            if all(
                record[key] == "PASS" for key in ("liquidity_gate", "volatility_gate", "trend_gate")
            ):
                continue
        elif filter_key == "data_issues":
            has_data_issue = (
                record["bars_complete"] == "NO"
                or "data" in record["_gate_fail_reason"]
                or "missing" in record["_gate_fail_reason"]
            )
            if not has_data_issue:
                continue
        elif filter_key == "high_confidence" and record["confidence"] != "High":
            continue

        staged.append(record)

    for row in staged:
        row.pop("_gate_fail_reason", None)
    return staged[:limit], f"postgres:screener_candidates run_date={_serialize_record(run_date)}"


def _screener_metrics_rows_from_file(
    *,
    filter_key: str,
    limit: int,
    query: str,
) -> tuple[list[dict[str, Any]], str]:
    path = Path(BASE_DIR) / "data" / "latest_candidates.csv"
    if not path.exists():
        return [], "latest_candidates.csv:missing"
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return [], f"latest_candidates.csv:read_error:{exc}"
    if frame.empty:
        return [], "latest_candidates.csv:empty"

    lowered = {str(column).strip().lower(): str(column) for column in frame.columns}

    def _col(*names: str) -> Optional[str]:
        for name in names:
            if name in lowered:
                return lowered[name]
        return None

    symbol_col = _col("symbol")
    if symbol_col is None:
        return [], "latest_candidates.csv:missing_symbol"

    score_breakdown_col = _col("score_breakdown")
    score_col = _col("final_score", "score")
    passed_col = _col("passed_gates")
    fail_col = _col("gate_fail_reason")
    source_col = _col("source")
    entry_col = _col("entry_price")
    adv20_col = _col("adv20")
    atrp_col = _col("atrp")
    exchange_col = _col("exchange")

    query_lower = query.lower()
    rows: list[dict[str, Any]] = []
    for _, source_row in frame.iterrows():
        symbol = str(source_row.get(symbol_col) or "").strip().upper()
        if not symbol:
            continue
        score_breakdown = source_row.get(score_breakdown_col) if score_breakdown_col else None
        source_label = str(source_row.get(source_col) or "").strip() if source_col else "DB"
        if query_lower:
            joined = " ".join(
                [
                    symbol,
                    str(score_breakdown or ""),
                    str(source_label),
                ]
            ).lower()
            if query_lower not in joined:
                continue

        liquidity_gate, volatility_gate, trend_gate = _screener_infer_metric_gates(
            passed_gates=source_row.get(passed_col) if passed_col else None,
            gate_fail_reason=source_row.get(fail_col) if fail_col else None,
        )
        bars_complete = "YES"
        if (
            (_to_float(source_row.get(entry_col)) in (None, 0) if entry_col else True)
            or ((_to_float(source_row.get(adv20_col)) in (None, 0)) if adv20_col else True)
            or ((_to_float(source_row.get(atrp_col)) in (None, 0)) if atrp_col else True)
            or (not str(source_row.get(exchange_col) or "").strip() if exchange_col else True)
        ):
            bars_complete = "NO"

        final_score = _to_float(source_row.get(score_col)) if score_col else None
        confidence = "Low"
        if final_score is not None and final_score >= 80:
            confidence = "High"
        elif final_score is not None and final_score >= 60:
            confidence = "Medium"

        record = {
            "symbol": symbol,
            "score_breakdown_short": _screener_score_breakdown_short(score_breakdown),
            "liquidity_gate": liquidity_gate,
            "volatility_gate": volatility_gate,
            "trend_gate": trend_gate,
            "bars_complete": bars_complete,
            "confidence": confidence,
            "source_label": source_label or "DB",
            "_gate_fail_reason": str(source_row.get(fail_col) or "").strip().lower()
            if fail_col
            else "",
        }

        if filter_key == "gate_failures":
            if all(
                record[key] == "PASS" for key in ("liquidity_gate", "volatility_gate", "trend_gate")
            ):
                continue
        elif filter_key == "data_issues":
            has_data_issue = (
                record["bars_complete"] == "NO"
                or "data" in record["_gate_fail_reason"]
                or "missing" in record["_gate_fail_reason"]
            )
            if not has_data_issue:
                continue
        elif filter_key == "high_confidence" and record["confidence"] != "High":
            continue

        rows.append(record)

    rows.sort(key=lambda item: item.get("symbol") or "")
    for row in rows:
        row.pop("_gate_fail_reason", None)
    return rows[:limit], f"file:{path}"


_SCREENER_LOG_TS_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2}:\d{2})(?:[.,](?P<frac>\d{1,6}))?"
)
_SCREENER_LOG_LEVEL_RE = re.compile(
    r"\b(ERROR|ERR|WARN|WARNING|INFO|SUCCESS|OK|DEBUG)\b", re.IGNORECASE
)


def _screener_normalize_log_level(level: str) -> str:
    normalized = str(level or "").strip().upper()
    if normalized.startswith("ERR"):
        return "ERROR"
    if normalized.startswith("WARN"):
        return "WARN"
    if normalized in {"SUCCESS", "OK"}:
        return "SUCCESS"
    return "INFO"


def _screener_parse_log_line(line: str) -> Optional[dict[str, Any]]:
    text = str(line or "").strip()
    if not text:
        return None

    timestamp = None
    match = _SCREENER_LOG_TS_RE.search(text)
    if match:
        date_text = match.group("date")
        time_text = match.group("time")
        frac_text = (match.group("frac") or "").ljust(6, "0")[:6]
        dt_text = f"{date_text} {time_text}"
        try:
            timestamp = datetime.strptime(dt_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            if frac_text:
                timestamp = timestamp.replace(microsecond=int(frac_text))
        except Exception:
            timestamp = None

    level_match = _SCREENER_LOG_LEVEL_RE.search(text)
    level = _screener_normalize_log_level(level_match.group(1) if level_match else "")

    message = re.sub(
        r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\s*",
        "",
        text,
    ).strip()
    if not message:
        message = text

    return {
        "ts_utc": _screener_iso_utc(timestamp),
        "level": level,
        "message": message,
        "_ts": timestamp,
    }


def _screener_filter_log_rows(
    rows: list[dict[str, Any]],
    *,
    level_filter: str,
    today_only: bool,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    now_utc = datetime.now(timezone.utc).date()
    query_lower = query.lower()
    filtered: list[dict[str, Any]] = []

    for row in rows:
        level = _screener_normalize_log_level(row.get("level", ""))
        if level_filter == "errors" and level != "ERROR":
            continue
        if level_filter == "warnings" and level != "WARN":
            continue

        ts_value = row.get("_ts")
        if today_only:
            if ts_value is None or ts_value.date() != now_utc:
                continue

        if query_lower:
            joined = " ".join(
                [
                    str(row.get("ts_utc") or ""),
                    str(level),
                    str(row.get("message") or ""),
                ]
            ).lower()
            if query_lower not in joined:
                continue

        filtered.append(
            {
                "ts_utc": row.get("ts_utc"),
                "level": level,
                "message": str(row.get("message") or "").strip(),
                "_ts": ts_value,
            }
        )

    filtered.sort(
        key=lambda item: item.get("_ts") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )

    trimmed = filtered[:limit]
    for row in trimmed:
        row.pop("_ts", None)
    return trimmed


def _screener_logs_from_db(stage: str, fetch_limit: int) -> tuple[list[dict[str, Any]], str]:
    stage = str(stage).strip().lower()
    candidate_tables = [
        f"{stage}_logs",
        "pipeline_logs",
        "log_events",
        "app_logs",
    ]
    seen: set[str] = set()
    for table in candidate_tables:
        if table in seen:
            continue
        seen.add(table)

        columns = _db_table_columns(table)
        if not columns:
            continue

        ts_col = _first_existing_column(
            columns,
            ["ts_utc", "timestamp", "created_at", "event_ts", "event_time", "ts", "time_utc"],
        )
        message_col = _first_existing_column(
            columns,
            ["message", "msg", "log_text", "text", "event", "line", "payload"],
        )
        if ts_col is None or message_col is None:
            continue

        level_col = _first_existing_column(columns, ["level", "severity", "log_level", "lvl"])
        stage_col = _first_existing_column(
            columns, ["stage", "component", "source", "module", "task"]
        )

        level_expr = f"{level_col}" if level_col else "'INFO'"
        params: dict[str, Any] = {"limit": int(fetch_limit), "stage_like": f"%{stage}%"}
        stage_sql = ""
        if stage_col:
            stage_sql = f"AND COALESCE({stage_col}::text, '') ILIKE %(stage_like)s"

        raw_rows = _db_fetch_all(
            f"""
            SELECT
                {ts_col} AS ts_utc,
                {level_expr} AS level,
                {message_col} AS message
            FROM {table}
            WHERE {message_col} IS NOT NULL
              {stage_sql}
            ORDER BY {ts_col} DESC NULLS LAST
            LIMIT %(limit)s
            """,
            params,
        )
        if not raw_rows:
            continue

        parsed_rows: list[dict[str, Any]] = []
        for row in raw_rows:
            parsed_ts = _coerce_datetime_utc(row.get("ts_utc"))
            parsed_rows.append(
                {
                    "ts_utc": _screener_iso_utc(parsed_ts),
                    "level": _screener_normalize_log_level(str(row.get("level") or "")),
                    "message": str(row.get("message") or "").strip(),
                    "_ts": parsed_ts,
                }
            )
        if parsed_rows:
            return parsed_rows, f"postgres:{table}"
    return [], "postgres:none"


def _screener_remote_log_text(filename: str) -> Optional[tuple[str, str]]:
    username = _pythonanywhere_username()
    token = _pythonanywhere_token()
    if not username or not token:
        return None
    candidate_paths = [
        f"/home/{username}/JBRAVO_Screener/logs/{filename}",
        f"/home/{username}/jbravo_screener/logs/{filename}",
    ]
    for remote_path in candidate_paths:
        payload = _fetch_pythonanywhere_file(None, remote_path)
        if payload:
            text, _ = payload
            return text, f"pythonanywhere:{remote_path}"
    return None


def _screener_local_log_text(filename: str) -> Optional[tuple[str, str]]:
    path = Path(BASE_DIR) / "logs" / filename
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    return text, f"local:{path}"


def _screener_logs_from_files(stage: str, fetch_limit: int) -> tuple[list[dict[str, Any]], str]:
    files_by_stage = {
        "screener": ["screener.log", "pipeline.log"],
        "backtest": ["step.backtest.out", "backtest.log", "pipeline.log"],
        "metrics": ["step.metrics.out", "metrics.log", "pipeline.log"],
    }
    file_list = files_by_stage.get(stage, [])
    if not file_list:
        return [], "files:none"

    parsed_rows: list[dict[str, Any]] = []
    sources: list[str] = []
    tail_limit = max(fetch_limit * 12, 400)

    for filename in file_list:
        payload = _screener_remote_log_text(filename) or _screener_local_log_text(filename)
        if not payload:
            continue
        text, source = payload
        sources.append(source)
        lines = [line for line in text.splitlines() if line.strip()]
        for line in lines[-tail_limit:]:
            parsed = _screener_parse_log_line(line)
            if not parsed:
                continue
            parsed_rows.append(parsed)

    if not parsed_rows:
        return [], "files:none"
    return parsed_rows, ";".join(sources)


@server.route("/api/screener/picks")
def api_screener_picks():
    limit = _parse_positive_int(request.args.get("limit"), default=50, minimum=1, maximum=200)
    query = str(request.args.get("q") or "").strip()
    filter_key = str(request.args.get("filter") or "all").strip().lower()
    if filter_key not in {"all", "top10", "passed", "errors"}:
        filter_key = "all"

    scope = _screener_latest_run_scope()
    run_date = scope.get("run_date")
    run_ts_utc = scope.get("run_ts_utc")
    status = str(scope.get("status") or "COMPLETE").upper()
    source = "postgres"
    source_detail = str(scope.get("source_detail") or "")

    rows: list[dict[str, Any]] = []
    if run_date is not None and db.db_enabled():
        rows, detail = _screener_pick_rows_from_db(
            run_date=run_date,
            run_ts_utc=run_ts_utc,
            filter_key=filter_key,
            limit=limit,
            query=query,
        )
        source_detail = detail

    if not rows:
        fallback_rows, fallback_detail, fallback_run_ts = _screener_pick_rows_from_file(
            filter_key=filter_key,
            limit=limit,
            query=query,
        )
        if fallback_detail.startswith("file:"):
            rows = fallback_rows
            source = "file-fallback"
            source_detail = fallback_detail
            run_ts_utc = fallback_run_ts or run_ts_utc
            status = "COMPLETE"
        elif status != "ERROR":
            source_detail = fallback_detail
            status = "EMPTY"

    return _json_no_store(
        {
            "ok": status != "ERROR",
            "run_ts_utc": _screener_iso_utc(run_ts_utc),
            "status": status,
            "source": source,
            "source_detail": source_detail,
            "rows": rows,
        }
    )


@server.route("/api/screener/backtest")
def api_screener_backtest():
    limit = _parse_positive_int(request.args.get("limit"), default=50, minimum=1, maximum=200)
    query = str(request.args.get("q") or "").strip()
    window = str(request.args.get("window") or "6M").strip().upper()
    if window not in {"3M", "6M", "1Y", "ALL"}:
        window = "6M"

    requested_run_ts = str(request.args.get("run_ts_utc") or "").strip() or None
    run_date, resolved_run_ts = _screener_resolve_run_date_from_run_ts(requested_run_ts)
    if run_date is None:
        latest_scope = _screener_latest_run_scope()
        run_date = latest_scope.get("run_date")
        resolved_run_ts = resolved_run_ts or latest_scope.get("run_ts_utc")
    if resolved_run_ts is None:
        resolved_run_ts = _screener_latest_run_ts_for_date(run_date)

    source = "postgres"
    rows: list[dict[str, Any]] = []
    if run_date is not None and db.db_enabled():
        rows, _ = _screener_backtest_rows_from_db(
            run_date=run_date,
            run_ts_utc=resolved_run_ts,
            window=window,
            limit=limit,
            query=query,
        )

    if not rows:
        fallback_rows, _ = _screener_backtest_rows_from_file(
            window=window, limit=limit, query=query
        )
        if fallback_rows:
            rows = fallback_rows
            source = "file-fallback"

    return _json_no_store(
        {
            "ok": True,
            "run_ts_utc": _screener_iso_utc(resolved_run_ts),
            "window": window,
            "source": source,
            "rows": rows,
        }
    )


@server.route("/api/screener/metrics")
def api_screener_metrics():
    limit = _parse_positive_int(request.args.get("limit"), default=50, minimum=1, maximum=200)
    query = str(request.args.get("q") or "").strip()
    filter_key = str(request.args.get("filter") or "all").strip().lower()
    if filter_key not in {"all", "gate_failures", "data_issues", "high_confidence"}:
        filter_key = "all"

    requested_run_ts = str(request.args.get("run_ts_utc") or "").strip() or None
    run_date, resolved_run_ts = _screener_resolve_run_date_from_run_ts(requested_run_ts)
    if run_date is None:
        latest_scope = _screener_latest_run_scope()
        run_date = latest_scope.get("run_date")
        resolved_run_ts = resolved_run_ts or latest_scope.get("run_ts_utc")

    rows: list[dict[str, Any]] = []
    source = "postgres"
    if run_date is not None and db.db_enabled():
        rows, _ = _screener_metrics_rows_from_db(
            run_date=run_date,
            run_ts_utc=resolved_run_ts,
            filter_key=filter_key,
            limit=limit,
            query=query,
        )

    if not rows:
        fallback_rows, _ = _screener_metrics_rows_from_file(
            filter_key=filter_key,
            limit=limit,
            query=query,
        )
        if fallback_rows:
            rows = fallback_rows
            source = "file-fallback"

    return _json_no_store(
        {
            "ok": True,
            "run_ts_utc": _screener_iso_utc(resolved_run_ts),
            "source": source,
            "rows": rows,
        }
    )


@server.route("/api/screener/logs")
def api_screener_logs():
    stage = str(request.args.get("stage") or "screener").strip().lower()
    if stage not in {"screener", "backtest", "metrics"}:
        stage = "screener"
    limit = _parse_positive_int(request.args.get("limit"), default=200, minimum=1, maximum=500)
    level_filter = str(request.args.get("level") or "all").strip().lower()
    if level_filter not in {"all", "errors", "warnings"}:
        level_filter = "all"
    today_only = str(request.args.get("today") or "0").strip().lower() in {"1", "true", "yes", "on"}
    query = str(request.args.get("q") or "").strip()

    fetch_limit = max(limit * 6, 300)
    file_rows, file_source_detail = _screener_logs_from_files(stage, fetch_limit)
    staged_rows = file_rows
    source_detail = file_source_detail
    if staged_rows:
        source = (
            "pythonanywhere" if "pythonanywhere:" in file_source_detail else "local-log-fallback"
        )
    else:
        db_rows, db_source_detail = _screener_logs_from_db(stage, fetch_limit)
        staged_rows = db_rows
        source = "postgres" if db_rows else "none"
        source_detail = db_source_detail if db_rows else f"{file_source_detail};{db_source_detail}"

    rows = _screener_filter_log_rows(
        staged_rows,
        level_filter=level_filter,
        today_only=today_only,
        query=query,
        limit=limit,
    )

    return _json_no_store(
        {
            "ok": True,
            "stage": stage,
            "source": source,
            "source_detail": source_detail,
            "rows": rows,
        }
    )


@server.route("/api/screener/candidates")
def api_screener_candidates():
    if not db.db_enabled():
        return jsonify({"ok": False, "rows": [], "rows_final": 0, "run_date": None})

    latest = _db_fetch_one("SELECT MAX(run_date) AS run_date FROM screener_candidates")
    run_date = latest.get("run_date") if latest else None
    if not run_date:
        return jsonify({"ok": True, "rows": [], "rows_final": 0, "run_date": None})

    rows = _db_fetch_all(
        """
        SELECT
            run_date, timestamp, symbol, score, exchange, close, volume,
            universe_count, score_breakdown, entry_price, adv20, atrp, source
        FROM screener_candidates
        WHERE run_date = %(run_date)s
        ORDER BY score DESC NULLS LAST
        LIMIT 50
        """,
        {"run_date": run_date},
    )
    serialized = [{key: _serialize_record(value) for key, value in row.items()} for row in rows]
    return jsonify(
        {
            "ok": True,
            "rows": serialized,
            "rows_final": len(serialized),
            "run_date": _serialize_record(run_date),
        }
    )


@server.route("/data/<path:filename>")
def data_exports(filename: str):
    if filename not in DATA_EXPORT_ALLOWLIST:
        abort(404)
    base = Path(BASE_DIR) / "data"
    if not (base / filename).exists():
        abort(404)
    return send_from_directory(base, filename)


@server.route("/logs/<path:filename>")
@server.route("/api/logs/<path:filename>")
def log_exports(filename: str):
    if filename not in LOG_EXPORT_ALLOWLIST:
        abort(404)

    def _no_cache(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    if filename == "pipeline.log":
        remote_payload = _fetch_pythonanywhere_file(
            os.environ.get("PYTHONANYWHERE_PIPELINE_LOG_URL"),
            os.environ.get("PYTHONANYWHERE_PIPELINE_LOG_PATH"),
        )
        if remote_payload:
            remote_text, _ = remote_payload
            return _no_cache(Response(remote_text, mimetype="text/plain"))
    if filename == "execute_trades.log":
        remote_payload = _fetch_pythonanywhere_file(
            os.environ.get("PYTHONANYWHERE_EXECUTE_LOG_URL"),
            os.environ.get("PYTHONANYWHERE_EXECUTE_LOG_PATH"),
        )
        if remote_payload:
            remote_text, _ = remote_payload
            return _no_cache(Response(remote_text, mimetype="text/plain"))
    if filename == "monitor.log":
        remote_payload = _fetch_pythonanywhere_file(
            os.environ.get("PYTHONANYWHERE_MONITOR_LOG_URL"),
            os.environ.get("PYTHONANYWHERE_MONITOR_LOG_PATH"),
        )
        if remote_payload:
            remote_text, _ = remote_payload
            return _no_cache(Response(remote_text, mimetype="text/plain"))
    base = Path(BASE_DIR) / "logs"
    if not (base / filename).exists():
        abort(404)
    return _no_cache(send_from_directory(base, filename))


@app.server.route("/api/health")
def api_health():
    payload = load_screener_health()
    return jsonify(payload)


@app.server.route("/api/candidates")
def api_candidates():
    try:
        df, _, source_file = screener_table()
    except Exception as exc:  # pragma: no cover - defensive read
        return jsonify({"columns": [], "rows": [], "rows_final": 0, "error": str(exc)}), 500
    if df is None or df.empty:
        return jsonify(
            {"columns": [], "rows": [], "rows_final": 0, "source": source_file or "none"}
        )
    rows_final = int(df.shape[0])
    source = source_file or "unknown"
    db_source_of_truth = "db" in str(source).lower()
    return jsonify(
        {
            "columns": [str(col) for col in df.columns],
            "rows": df.to_dict(orient="records"),
            "rows_final": rows_final,
            "source": source,
            "db_source_of_truth": db_source_of_truth,
        }
    )


DEFAULT_ACTIVE_TAB = "tab-overview"


def build_tabs(active_tab: str = DEFAULT_ACTIVE_TAB) -> dbc.Tabs:
    return dbc.Tabs(
        id="tabs",
        active_tab=active_tab,
        class_name="mb-3",
        children=[
            dbc.Tab(
                label="Overview",
                tab_id="tab-overview",
                id="tab-overview",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="Pipeline",
                tab_id="tab-pipeline",
                id="tab-pipeline",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="ML",
                tab_id="tab-ml",
                id="tab-ml",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="Screener Health",
                tab_id="tab-screener-health",
                id="tab-screener-health",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="Screener",
                tab_id="tab-screener",
                id="tab-screener",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="Execute Trades",
                tab_id="tab-execute",
                id="tab-execute",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="Activities",
                tab_id="tab-activities",
                id="tab-activities",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="Account",
                tab_id="tab-account",
                id="tab-account",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="Symbol Performance",
                tab_id="tab-symbol-performance",
                id="tab-symbol-performance",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="Monitoring Positions",
                tab_id="tab-monitor",
                id="tab-monitor",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="Trades / Exits",
                tab_id="tab-trades",
                id="tab-trades",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
            dbc.Tab(
                label="Trade Performance",
                tab_id="tab-trade-performance",
                id="tab-trade-performance",
                tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                active_tab_style={"backgroundColor": "#22c55e", "color": "#fff"},
                className="custom-tab",
            ),
        ],
    )


# Layout with Tabs and Modals
app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(
            id="active-tab-store", storage_type="memory", data={"active_tab": DEFAULT_ACTIVE_TAB}
        ),
        dbc.Row(
            dbc.Col(
                html.H1(
                    "JBRAVO Trading Bot",
                    className="text-center my-4 text-light",
                )
            )
        ),
        html.Div(
            [
                build_tabs(),
                html.Button(
                    "Refresh Now",
                    id="refresh-button",
                    className="btn btn-secondary mb-2",
                ),
                dcc.Store(id="refresh-ts"),
                dcc.Loading(
                    id="loading",
                    children=html.Div(id="tabs-content", className="mt-4"),
                    type="default",
                ),
            ],
        ),
        # Refresh dashboards once per day; the app is reloaded by the wrapper after runs
        dcc.Interval(id="interval-update", interval=24 * 60 * 60 * 1000, n_intervals=0),
        dcc.Interval(id="log-interval", interval=24 * 60 * 60 * 1000, n_intervals=0),
        dcc.Interval(id="interval-trades", interval=24 * 60 * 60 * 1000, n_intervals=0),
        dcc.Store(id="predictions-store"),
        dbc.Modal(
            id="detail-modal",
            is_open=False,
            size="lg",
            children=[
                dbc.ModalHeader(dbc.ModalTitle("Details")),
                dbc.ModalBody(id="modal-content"),
                dbc.ModalFooter(dbc.Button("Close", id="close-modal", className="ms-auto")),
            ],
        ),
        html.Div(get_version_string(), id="version-banner", className="text-muted mt-2"),
    ],
    fluid=True,
)


# Register Screener Health callbacks
register_screener_health(app)


# Callbacks for tabs content
_TAB_HASH_MAP = {
    "overview": "tab-overview",
    "tab-overview": "tab-overview",
    "pipeline": "tab-pipeline",
    "tab-pipeline": "tab-pipeline",
    "ml": "tab-ml",
    "tab-ml": "tab-ml",
    "screener-health": "tab-screener-health",
    "tab-screener-health": "tab-screener-health",
    "screener": "tab-screener",
    "tab-screener": "tab-screener",
    "execute": "tab-execute",
    "tab-execute": "tab-execute",
    "activities": "tab-activities",
    "tab-activities": "tab-activities",
    "account": "tab-account",
    "tab-account": "tab-account",
    "symbol-performance": "tab-symbol-performance",
    "tab-symbol-performance": "tab-symbol-performance",
    "monitor": "tab-monitor",
    "tab-monitor": "tab-monitor",
    "trades": "tab-trades",
    "tab-trades": "tab-trades",
    "trade-performance": "tab-trade-performance",
    "tab-trade-performance": "tab-trade-performance",
}
_TAB_IDS = {
    "tab-overview",
    "tab-pipeline",
    "tab-ml",
    "tab-screener-health",
    "tab-screener",
    "tab-execute",
    "tab-activities",
    "tab-account",
    "tab-symbol-performance",
    "tab-monitor",
    "tab-trades",
    "tab-trade-performance",
}


@app.callback(
    Output("active-tab-store", "data"),
    [
        Input("tabs", "active_tab"),
        Input("url", "hash"),
    ],
    State("active-tab-store", "data"),
)
def _update_active_tab_store(
    active_tab_value: str | None,
    url_hash: str | None,
    store_data,
):
    """
    Update the active tab state based on the Tabs component or URL hash.

    Dash Bootstrap Components Tabs no longer expose ``n_clicks`` on individual tabs,
    so we listen to ``tabs.active_tab`` instead and keep the hash fallback for
    direct links.
    """
    ctx = dash.callback_context
    triggered = getattr(ctx, "triggered_id", None)
    tab_id = None
    if triggered == "url":
        tab_id = _TAB_HASH_MAP.get((url_hash or "").lstrip("#"))
    else:
        tab_id = active_tab_value
    if tab_id in _TAB_IDS:
        return {"active_tab": tab_id}
    if isinstance(store_data, Mapping):
        existing = store_data.get("active_tab")
        if existing in _TAB_IDS:
            return {"active_tab": existing}
    return {"active_tab": DEFAULT_ACTIVE_TAB}


@app.callback(Output("tabs", "active_tab"), Input("active-tab-store", "data"))
def _sync_tabs_to_store(store_data: Mapping[str, Any] | None):
    if isinstance(store_data, Mapping) and store_data.get("active_tab") in _TAB_IDS:
        return store_data["active_tab"]
    return DEFAULT_ACTIVE_TAB


@app.callback(Output("url", "hash"), Input("active-tab-store", "data"))
def _sync_hash_from_store(store_data: Mapping[str, Any] | None):
    if isinstance(store_data, Mapping) and store_data.get("active_tab"):
        return f"#{store_data['active_tab']}"
    return f"#{DEFAULT_ACTIVE_TAB}"


def _render_tab(tab, _n_intervals, _n_log_intervals, _refresh_clicks):
    app.logger.info("Rendering tab %s", tab)
    if tab == "tab-screener":
        metrics_data: dict = {}
        health_snapshot = load_screener_health()
        health_data = health_snapshot or {}
        metrics_summary_row = metrics_summary_snapshot()
        metrics_alert = None
        backfill_banner = None
        metrics_freshness_chip = None
        if os.path.exists(screener_metrics_path):
            try:
                with open(screener_metrics_path, "r", encoding="utf-8") as handle:
                    metrics_data = json.load(handle) or {}
            except Exception as exc:
                metrics_alert = dbc.Alert(
                    f"Failed to read screener metrics: {exc}",
                    color="danger",
                )
        else:
            metrics_alert = dbc.Alert(
                "Screener metrics not available yet.",
                color="warning",
            )

        if metrics_data:

            def _needs_backfill(payload: dict[str, Any]) -> bool:
                for key in ("symbols_in", "symbols_with_bars", "bars_rows_total", "candidates_out"):
                    value = payload.get(key)
                    if value in (None, "", []):
                        return True
                    if isinstance(value, float) and math.isnan(value):
                        return True
                return False

            if _needs_backfill(metrics_data):
                try:
                    recovered = write_complete_screener_metrics(Path(BASE_DIR))
                except Exception:
                    logger.error("Unable to backfill screener metrics", exc_info=True)
                    if metrics_alert is None:
                        metrics_alert = dbc.Alert(
                            "Unable to backfill screener metrics. Using last known metrics.",
                            color="warning",
                        )
                else:
                    if isinstance(recovered, dict):
                        metrics_data.update(recovered)
                    backfill_banner = dbc.Alert(
                        "Backfilled from PIPELINE_SUMMARY to fill missing KPIs.",
                        color="info",
                        className="mb-3",
                    )

        metrics_data = metrics_data or {}
        if health_snapshot:
            if health_data.get("symbols_in") not in (None, ""):
                metrics_data["symbols_in"] = health_data.get("symbols_in")
            bars_fetch = health_data.get("bars_rows_total_fetch")
            if bars_fetch not in (None, ""):
                metrics_data["bars_rows_total_fetch"] = bars_fetch
                metrics_data["bars_rows_total"] = bars_fetch
            with_bars_fetch = health_data.get("symbols_with_bars_fetch")
            if with_bars_fetch not in (None, ""):
                metrics_data["symbols_with_bars_fetch"] = with_bars_fetch
                metrics_data["symbols_with_bars"] = with_bars_fetch
            rows_pre = health_data.get("rows_premetrics")
            if rows_pre not in (None, ""):
                metrics_data["rows"] = rows_pre
            if health_data.get("rows_final") not in (None, ""):
                metrics_data["rows_final"] = health_data.get("rows_final")
            if not metrics_data.get("last_run_utc") and health_data.get("last_run_utc"):
                metrics_data["last_run_utc"] = health_data.get("last_run_utc")

        run_type_label = health_data.get("run_type") or "nightly"

        def _build_freshness_chip() -> Optional[dbc.Badge]:
            freshness_info = (health_snapshot or {}).get("freshness") or {}
            level = freshness_info.get("freshness_level")
            age_seconds = freshness_info.get("age_seconds")
            if not level:
                return None
            color_map = {"green": "success", "amber": "warning", "gray": "secondary"}
            if isinstance(age_seconds, (int, float)) and age_seconds >= 0:
                if age_seconds < 3600:
                    label = f"Freshness: {int(age_seconds // 60)}m ago"
                else:
                    hours = age_seconds / 3600
                    label = f"Freshness: {hours:.1f}h ago"
            else:
                label = "Freshness: unknown"
            return dbc.Badge(
                label, color=color_map.get(level, "secondary"), className="badge-small"
            )

        metrics_freshness_chip = _build_freshness_chip()

        execute_metrics: dict = {}
        execute_alert = None
        if os.path.exists(execute_metrics_path):
            try:
                with open(execute_metrics_path, "r", encoding="utf-8") as handle:
                    loaded_execute = json.load(handle) or {}
                if isinstance(loaded_execute, dict):
                    execute_metrics = loaded_execute
                else:
                    execute_alert = dbc.Alert(
                        "Execution metrics are in an unexpected format.",
                        color="warning",
                    )
            except Exception as exc:
                execute_alert = dbc.Alert(
                    f"Failed to read execution metrics: {exc}",
                    color="danger",
                )

        candidates_df, candidates_alert = load_top_or_latest_candidates()
        fallback_count = 0
        table_updated = None
        table_source_file = None
        table_df = None
        if candidates_df is not None and not candidates_df.empty:
            table_source_file = (
                candidates_df["__source"].iloc[0] if "__source" in candidates_df.columns else None
            )
            table_updated = (
                candidates_df["__updated"].iloc[0] if "__updated" in candidates_df.columns else None
            )
            work_df = candidates_df.drop(columns=["__source", "__updated"], errors="ignore")
            for column in ("source", "origin"):
                if column in work_df.columns:
                    source_series = work_df[column].astype("string")
                    fallback_count = int(
                        source_series.str.contains("fallback", case=False, na=False).sum()
                    )
                    if fallback_count:
                        break
            table_df = work_df

        latest_notice = None
        if os.path.exists(latest_candidates_path):
            try:
                latest_preview = pd.read_csv(latest_candidates_path, nrows=1)
            except Exception:
                latest_preview = pd.DataFrame()
            if latest_preview.empty:
                latest_notice = dbc.Alert(
                    [
                        "No candidates today; fallback may populate shortly. ",
                        html.A(
                            "View pipeline panel",
                            href="#sh-pipeline-panel",
                            className="alert-link",
                        ),
                    ],
                    color="info",
                    className="mb-3",
                )

        alerts = [a for a in (metrics_alert, candidates_alert, execute_alert) if a]

        def _safe_int(value) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        def _safe_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _format_value(value) -> str:
            if value is None:
                return "—"
            if isinstance(value, str):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return value
            else:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return str(value)
            if abs(numeric - int(numeric)) < 1e-6:
                return f"{int(numeric):,}"
            return f"{numeric:,.2f}"

        def _format_iso_display(value):
            if isinstance(value, str):
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    return parsed.strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    return value
            return value

        def _format_kpi_tile_value(
            value: Any, *, fmt: str = "{:.2f}", prefix: str = "", suffix: str = ""
        ) -> str:
            if value in (None, "", [], {}):
                return "—"
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return str(value)
            if math.isnan(numeric):
                return "—"
            return f"{prefix}{fmt.format(numeric)}{suffix}"

        def _build_metrics_summary_tiles(summary_values: dict[str, Any]) -> dbc.Row:
            specs = [
                ("Profit Factor", "profit_factor", "{:.2f}", "", ""),
                ("Expectancy", "expectancy", "{:.2f}", "$", ""),
                ("Win-rate", "win_rate", "{:.2f}", "", "%"),
                ("Net PnL", "net_pnl", "{:,.2f}", "$", ""),
                ("Max DD", "max_drawdown", "{:,.2f}", "$", ""),
                ("Sharpe", "sharpe", "{:.2f}", "", ""),
                ("Sortino", "sortino", "{:.2f}", "", ""),
            ]
            columns: list[Any] = []
            for label, key, fmt_pattern, prefix, suffix in specs:
                value = summary_values.get(key)
                rendered = _format_kpi_tile_value(
                    value, fmt=fmt_pattern, prefix=prefix, suffix=suffix
                )
                card = dbc.Card(
                    [
                        dbc.CardHeader(label),
                        dbc.CardBody(html.H4(rendered, className="mb-0")),
                    ],
                    className="bg-dark text-light h-100",
                )
                columns.append(dbc.Col(card, lg=2, md=4, sm=6, className="mb-3"))
            return dbc.Row(columns, className="g-3 mb-4")

        last_run_display = _format_iso_display(metrics_data.get("last_run_utc"))

        pre_candidates = (
            health_data.get("rows_premetrics")
            or health_data.get("rows_final")
            or metrics_data.get("rows")
            or metrics_data.get("candidates_out")
        )
        final_candidates = health_data.get("rows_final") or metrics_data.get("rows_final")
        with_bars_value = (
            health_data.get("symbols_with_bars_fetch")
            or metrics_data.get("symbols_with_bars_fetch")
            or metrics_data.get("symbols_with_bars")
        )
        with_bars_post = health_data.get("symbols_with_bars_post")
        bars_rows_value = (
            health_data.get("bars_rows_total_fetch")
            or metrics_data.get("bars_rows_total_fetch")
            or metrics_data.get("bars_rows_total")
        )
        bars_rows_post = health_data.get("bars_rows_total_post")
        final_candidates_value = final_candidates or health_data.get("rows_final")
        health_columns = []
        first_card_body = [
            html.Div("Last Run (UTC)", className="card-metric-label"),
            html.Div(_format_value(last_run_display), className="card-metric-value"),
            html.Div(f"Run type: {run_type_label}", className="small text-muted mt-1"),
        ]
        first_card = dbc.Card(
            dbc.CardBody(first_card_body),
            className="bg-dark text-light h-100",
        )
        health_columns.append(dbc.Col(first_card, md=4, sm=6))
        counter_items = [
            ("Symbols In", health_data.get("symbols_in") or metrics_data.get("symbols_in"), None),
            (
                "With Bars (fetch)",
                with_bars_value,
                f"post-filter: {_format_value(with_bars_post)}"
                if with_bars_post not in (None, "")
                else None,
            ),
            (
                "Bars Rows (total)",
                bars_rows_value,
                f"post-filter: {_format_value(bars_rows_post)}"
                if bars_rows_post not in (None, "")
                else None,
            ),
            (
                "Candidates (final)",
                final_candidates_value,
                f"pre-metrics: {_format_value(pre_candidates)}"
                if pre_candidates not in (None, "")
                else None,
            ),
        ]
        for label, value, sub_text in counter_items:
            card_body = [
                html.Div(label, className="card-metric-label"),
                html.Div(_format_value(value), className="card-metric-value"),
            ]
            if sub_text:
                card_body.append(html.Div(sub_text, className="small text-muted mt-1"))
            card = dbc.Card(
                dbc.CardBody(card_body),
                className="bg-dark text-light h-100",
            )
            health_columns.append(dbc.Col(card, md=2, sm=6))
        health_cards = dbc.Row(health_columns, className="g-3 mb-4")
        kpi_tiles = _build_metrics_summary_tiles(metrics_summary_row or {})

        timings = metrics_data.get("timings", {}) or {}

        execution_card_body = [html.Div("Execution", className="card-metric-label")]
        if execute_metrics:
            exec_last_run = _format_iso_display(execute_metrics.get("last_run_utc"))
            submitted = _safe_int(execute_metrics.get("orders_submitted"))
            trailing = _safe_int(execute_metrics.get("trailing_attached"))
            skips_payload = execute_metrics.get("skips")
            skip_text = "None"
            if isinstance(skips_payload, dict):
                parts = []
                for key, value in sorted(skips_payload.items()):
                    count = _safe_int(value)
                    if count <= 0:
                        continue
                    parts.append(f"{key}:{count}")
                skip_text = ", ".join(parts) if parts else "None"
            elif skips_payload not in (None, ""):
                skip_text = str(skips_payload)
            execution_card_body.extend(
                [
                    html.Div(f"Last Run: {exec_last_run or '—'}", className="small text-muted"),
                    html.Div(f"Submitted: {submitted}", className="small text-muted"),
                    html.Div(f"Trailing: {trailing}", className="small text-muted"),
                    html.Div(f"Skips: {skip_text}", className="small text-muted"),
                ]
            )
        else:
            execution_card_body.append(
                html.Div(
                    "No execution metrics yet (will appear after execute step).",
                    className="small text-muted",
                )
            )
        execution_card = dbc.Card(
            dbc.CardBody(execution_card_body),
            className="bg-dark text-light h-100",
        )
        timing_labels = [
            ("fetch_secs", "Fetch"),
            ("normalize_secs", "Normalize"),
            ("feature_secs", "Features"),
            ("rank_secs", "Rank"),
            ("gates_secs", "Gates"),
        ]
        timing_badges = []
        for key, label in timing_labels:
            value = _safe_float(timings.get(key))
            text_value = f"{value:.2f}s" if value is not None else "—"
            timing_badges.append(dbc.Badge(f"{label}: {text_value}", className="badge-small me-1"))
        timing_card = dbc.Card(
            dbc.CardBody(
                [
                    html.Div("Timings", className="card-metric-label mb-2"),
                    html.Div(timing_badges),
                ]
            ),
            className="bg-dark text-light h-100",
        )

        http_badges = [
            dbc.Badge(
                f"429: {_safe_int(metrics_data.get('rate_limited'))}", className="badge-small me-1"
            ),
            dbc.Badge(
                f"404: {_safe_int(metrics_data.get('http_404_batches'))}",
                className="badge-small me-1",
            ),
            dbc.Badge(
                f"Empty: {_safe_int(metrics_data.get('http_empty_batches'))}",
                className="badge-small",
            ),
        ]
        http_card = dbc.Card(
            dbc.CardBody(
                [
                    html.Div("HTTP Status", className="card-metric-label mb-2"),
                    html.Div(http_badges),
                ]
            ),
            className="bg-dark text-light h-100",
        )

        cache_badges = [
            dbc.Badge(
                f"Cache hits: {_safe_int(metrics_data.get('cache_hits'))}",
                className="badge-small me-1",
            ),
            dbc.Badge(
                f"Parsed rows: {_safe_int(metrics_data.get('parsed_rows_count'))}",
                className="badge-small",
            ),
        ]
        cache_card = dbc.Card(
            dbc.CardBody(
                [
                    html.Div("Cache", className="card-metric-label mb-2"),
                    html.Div(cache_badges),
                ]
            ),
            className="bg-dark text-light h-100",
        )

        info_row = dbc.Row(
            [dbc.Col(timing_card, md=4), dbc.Col(http_card, md=4), dbc.Col(cache_card, md=4)],
            className="g-3 mb-4",
        )

        gate_counts = metrics_data.get("gate_fail_counts", {}) or {}
        gate_items = []
        for key, value in gate_counts.items():
            if not str(key).startswith("failed_"):
                continue
            count = _safe_int(value)
            if count <= 0:
                continue
            label = str(key).replace("failed_", "").replace("_", " ").title()
            gate_items.append({"reason": label, "count": count})
        if gate_items:
            gate_df = pd.DataFrame(gate_items).sort_values("count", ascending=False)
            gate_fig = px.bar(
                gate_df,
                x="reason",
                y="count",
                text="count",
                template="plotly_dark",
            )
            gate_fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
                xaxis_title="Gate",
                yaxis_title="Failures",
            )
            gate_fig.update_traces(marker_color="#4DB6AC", textposition="outside")
        else:
            gate_fig = go.Figure()
            gate_fig.add_annotation(
                text="No gate failures recorded",
                showarrow=False,
                font=dict(color="#adb5bd"),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
            gate_fig.update_layout(
                template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), height=320
            )

        with_bars = _safe_int(metrics_data.get("symbols_with_bars"))
        without_bars = _safe_int(metrics_data.get("symbols_no_bars"))
        if with_bars + without_bars > 0:
            coverage_fig = px.pie(
                names=["With Bars", "No Bars"],
                values=[with_bars, without_bars],
                hole=0.5,
                template="plotly_dark",
            )
            coverage_fig.update_traces(textinfo="label+percent")
            coverage_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
        else:
            coverage_fig = go.Figure()
            coverage_fig.add_annotation(
                text="No universe coverage data",
                showarrow=False,
                font=dict(color="#adb5bd"),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
            coverage_fig.update_layout(
                template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), height=320
            )

        charts_row = dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=gate_fig), md=8),
                dbc.Col(dcc.Graph(figure=coverage_fig), md=4),
            ],
            className="g-3 mb-4",
        )

        top_table_component = dbc.Alert(
            "No candidates available yet.", color="warning", className="m-2"
        )
        if table_df is not None and not table_df.empty:
            display_df = table_df.copy()
            display_df.columns = [str(col) for col in display_df.columns]
            columns = [
                {"name": col.replace("_", " ").title(), "id": col} for col in display_df.columns
            ]
            style_data_conditional: list[dict[str, Any]] = []
            for column in ("source", "origin"):
                if column in display_df.columns:
                    style_data_conditional.append(
                        {
                            "if": {
                                "column_id": column,
                                "filter_query": "{" + column + "} contains 'fallback'",
                            },
                            "backgroundColor": "#2b223a",
                            "color": "#f4d9ff",
                            "fontWeight": "600",
                        }
                    )
            top_table_component = dash_table.DataTable(
                id="screener-top-table",
                data=display_df.to_dict("records"),
                columns=columns,
                page_size=20,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#1b1e21", "fontWeight": "600"},
                style_cell={
                    "backgroundColor": "#212529",
                    "color": "#E0E0E0",
                    "fontSize": "0.9rem",
                    "textAlign": "left",
                },
                style_data_conditional=style_data_conditional,
            )

        raw_candidates_component: Any = dbc.Alert(
            "Raw screener candidates are not available yet.",
            color="secondary",
            className="m-2",
        )
        latest_df, latest_alert = load_latest_candidates()
        if latest_alert:
            raw_candidates_component = latest_alert
        if latest_df is not None and not latest_df.empty:
            raw_display = latest_df.copy()
            if "score" in raw_display.columns:
                raw_display = raw_display.sort_values("score", ascending=False)
            raw_columns = [
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Symbol", "id": "symbol"},
                {"name": "Score", "id": "score", "type": "numeric", "format": {"specifier": ".3f"}},
                {"name": "Exchange", "id": "exchange"},
                {"name": "Close", "id": "close", "type": "numeric", "format": {"specifier": ".2f"}},
                {
                    "name": "Volume",
                    "id": "volume",
                    "type": "numeric",
                    "format": {"specifier": ".0f"},
                },
                {"name": "Universe Count", "id": "universe_count"},
                {
                    "name": "Entry Price",
                    "id": "entry_price",
                    "type": "numeric",
                    "format": {"specifier": ".2f"},
                },
                {"name": "ADV20", "id": "adv20", "type": "numeric", "format": {"specifier": ".0f"}},
                {"name": "ATR%", "id": "atrp", "type": "numeric", "format": {"specifier": ".2%"}},
                {"name": "Source", "id": "source"},
            ]
            raw_candidates_component = dash_table.DataTable(
                id="screener-latest-table",
                data=raw_display.to_dict("records"),
                columns=raw_columns,
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#1b1e21", "fontWeight": "600"},
                style_cell={
                    "backgroundColor": "#212529",
                    "color": "#E0E0E0",
                    "fontSize": "0.9rem",
                    "textAlign": "left",
                },
            )

        feature_summary = metrics_data.get("feature_summary") or {}
        history_frames = load_prediction_history(limit=7)

        def _feature_column(name: str) -> str:
            mapping = {"VOLexp": "volexp", "GAPpen": "gap_pen", "LIQpen": "liq_pen"}
            return mapping.get(name, name.lower())

        feature_cards = []
        if feature_summary:
            sorted_features = sorted(
                feature_summary.items(),
                key=lambda kv: abs(kv[1].get("mean") or 0.0),
                reverse=True,
            )
            for feature, stats in sorted_features[:6]:
                mean_val = stats.get("mean")
                std_val = stats.get("std")
                history_values = []
                for date_str, frame in history_frames:
                    column_name = _feature_column(feature)
                    if column_name not in frame.columns:
                        continue
                    series = pd.to_numeric(frame[column_name], errors="coerce")
                    if series.dropna().empty:
                        continue
                    try:
                        display_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        continue
                    history_values.append((display_date.strftime("%m-%d"), float(series.mean())))
                history_values.sort(key=lambda item: item[0])
                fig = go.Figure()
                if history_values:
                    fig.add_trace(
                        go.Scatter(
                            x=[item[0] for item in history_values],
                            y=[item[1] for item in history_values],
                            mode="lines+markers",
                            line=dict(color="#4DB6AC"),
                            marker=dict(size=4),
                        )
                    )
                else:
                    fig.add_annotation(
                        text="No trend data",
                        showarrow=False,
                        font=dict(color="#adb5bd"),
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                    )
                fig.update_layout(
                    template="plotly_dark",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=110,
                    xaxis=dict(showgrid=False, tickfont=dict(size=9)),
                    yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=9)),
                )
                mean_display = "—" if mean_val is None else f"{mean_val:.2f}"
                std_display = "—" if std_val is None else f"{std_val:.2f}"
                subtitle = f"μ {mean_display} | σ {std_display}"
                feature_cards.append(
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(feature, className="sparkline-title"),
                                html.Div(subtitle, className="sparkline-subtitle"),
                                dcc.Graph(
                                    figure=fig,
                                    config={"displayModeBar": False},
                                    style={"height": "90px"},
                                ),
                            ],
                            className="sparkline-card",
                        ),
                        md=2,
                        sm=6,
                    )
                )
        feature_section = (
            dbc.Row(feature_cards, className="g-3 mb-4")
            if feature_cards
            else dbc.Alert(
                "No feature diagnostics available yet.",
                color="info",
                className="m-2",
            )
        )

        pipeline_lines, pipeline_label = _tail_with_timestamp(pipeline_log_path, limit=120)
        screener_lines, screener_label = _tail_with_timestamp(screener_log_path, limit=120)
        backtest_lines = read_recent_lines(backtest_log_path)[::-1]
        pipeline_summary_panel = _render_pipeline_summary_panel(
            parse_pipeline_summary(Path(pipeline_log_path))
        )

        logs_row = dbc.Row(
            [
                dbc.Col(
                    log_box(
                        "Pipeline Log",
                        pipeline_lines,
                        "pipeline-log",
                        log_path=pipeline_log_path,
                        subtitle=pipeline_label,
                    ),
                    md=4,
                ),
                dbc.Col(
                    log_box(
                        "Screener Log",
                        screener_lines,
                        "screener-log",
                        log_path=screener_log_path,
                        subtitle=screener_label,
                    ),
                    md=4,
                ),
                dbc.Col(
                    log_box(
                        "Backtest Log",
                        backtest_lines,
                        "backtest-log",
                        log_path=backtest_log_path,
                    ),
                    md=4,
                ),
            ],
            className="g-3 mb-4",
        )
        logs_stack = html.Div([logs_row, pipeline_summary_panel], className="mb-3")

        components = [html.Div(_paper_badge_component(), className="mb-2")]
        if latest_notice:
            components.append(latest_notice)
        if backfill_banner:
            components.append(backfill_banner)
        components.extend(alerts)

        source_label = str(health_data.get("source") or "unknown")
        source_color = (
            "success"
            if source_label == "screener"
            else ("warning" if source_label == "fallback" else "secondary")
        )
        source_badge = dbc.Badge(
            f"Source: {source_label}",
            color=source_color,
            className="me-2",
        )
        pipeline_rc = health_data.get("pipeline_rc")
        rc_text = f"rc={pipeline_rc}" if pipeline_rc is not None else "rc=n/a"
        rc_color = "success" if pipeline_rc == 0 else ("danger" if pipeline_rc else "secondary")
        rc_badge = dbc.Badge(rc_text, color=rc_color, className="me-2")
        run_type_badge = dbc.Badge(
            f"Run: {run_type_label}",
            color="info",
            className="me-2",
        )
        trading_ok_value = health_data.get("trading_ok")
        data_ok_value = health_data.get("data_ok")
        alpaca_badge = None
        alpaca_color = connection_badge_color(health_data)
        if alpaca_color:
            trading_status = health_data.get("trading_status")
            data_status = health_data.get("data_status")
            feed_label = str(health_data.get("feed") or "").upper()

            def _badge_icon(value: Any) -> str:
                if value is None:
                    return "—"
                return "✅" if bool(value) else "❌"

            trade_label = f"Trading {_badge_icon(trading_ok_value)}"
            if trading_status:
                trade_label = f"{trade_label} ({trading_status})"
            data_label = f"Data {_badge_icon(data_ok_value)}"
            if data_status:
                data_label = f"{data_label} ({data_status})"

            badge_children = [html.Span("Alpaca", className="me-2")]
            if feed_label:
                badge_children.append(html.Span(feed_label, className="me-2 text-uppercase"))
            badge_children.extend([html.Span(trade_label, className="me-2"), html.Span(data_label)])
            alpaca_badge = dbc.Badge(badge_children, color=alpaca_color, className="me-2")
        status_badges = [source_badge, rc_badge, run_type_badge]
        if alpaca_badge:
            status_badges.append(alpaca_badge)
        if metrics_freshness_chip:
            status_badges.append(metrics_freshness_chip)
        status_badges = [badge for badge in status_badges if badge]
        if status_badges:
            components.append(
                html.Div(
                    status_badges,
                    className="mb-3 d-flex flex-wrap align-items-center gap-2",
                )
            )
        table_updated_display = _format_iso_display(table_updated) if table_updated else "unknown"
        table_source_note = html.Span(
            f"Candidates file: {table_source_file or 'unknown'}",
            className="text-muted me-3",
        )
        table_updated_note = html.Span(
            f"Updated: {table_updated_display}",
            className="text-muted",
        )
        components.append(
            html.Div(
                [table_source_note, table_updated_note],
                className="mb-3 text-muted small d-flex flex-wrap gap-3",
            )
        )

        metrics_sections: list[Any] = []
        if health_data or metrics_data:
            metrics_sections.append(health_cards)
        metrics_sections.append(kpi_tiles)
        if execution_card is not None:
            metrics_sections.append(
                dbc.Row([dbc.Col(execution_card, md=4, sm=6)], className="g-3 mb-4")
            )
        if metrics_data:
            metrics_sections.extend([info_row, charts_row])
        if metrics_sections:
            components.extend(metrics_sections)
        fallback_badge = None
        if fallback_count:
            fallback_badge = dbc.Badge(
                f"Fallback ({fallback_count})",
                color="warning",
                text_color="dark",
                className="ms-2",
                style={
                    "fontSize": "0.65rem",
                    "letterSpacing": "0.04em",
                    "fontWeight": 600,
                    "padding": "0.2rem 0.45rem",
                },
            )
        title_children: list[Any] = ["Top Candidates"]
        if fallback_badge is not None:
            title_children.append(fallback_badge)
        components.append(
            html.H4(
                title_children,
                className="text-light",
                style={"display": "flex", "alignItems": "center", "gap": "8px"},
            )
        )
        components.append(top_table_component)
        components.append(html.Hr())
        components.append(
            html.H4(
                "Today's Candidates (raw)",
                className="text-light",
                style={"display": "flex", "alignItems": "center", "gap": "8px"},
            )
        )
        components.append(raw_candidates_component)
        if metrics_data:
            components.extend(
                [html.Hr(), html.H4("Diagnostics", className="text-light"), feature_section]
            )
        components.extend([html.Hr(), logs_stack])
        return dbc.Container(components, fluid=True)

    elif tab == "tab-execute":
        trades_df, trade_source_path, trade_source_label, trade_alerts = _resolve_trades_dataframe()
        executed_exists = os.path.exists(executed_trades_path)
        summary_section = None

        if trades_df is not None:
            sort_column = "entry_time" if "entry_time" in trades_df.columns else None
            if sort_column:
                trades_df.sort_values(sort_column, ascending=False, inplace=True)

            exit_reason_cards: list[dbc.Col] = []
            if "exit_reason" in trades_df.columns:
                exit_reason_counts = trades_df["exit_reason"].value_counts()
                if not exit_reason_counts.empty:
                    counts_df = exit_reason_counts.reset_index()
                    counts_df.columns = ["exit_reason", "trades"]
                    counts_table = dash_table.DataTable(
                        data=counts_df.to_dict("records"),
                        columns=[
                            {"name": "Exit Reason", "id": "exit_reason"},
                            {"name": "Trades", "id": "trades", "type": "numeric"},
                        ],
                        style_table={"overflowX": "auto"},
                        style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
                        page_size=10,
                    )
                    exit_reason_cards.append(
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Trades by exit_reason"),
                                    dbc.CardBody(counts_table),
                                ],
                                className="h-100",
                            ),
                            md=6,
                        )
                    )

            efficiency_fig = None
            if {"exit_reason", "exit_efficiency"}.issubset(trades_df.columns):
                efficiency_df = trades_df.copy()
                efficiency_df["exit_efficiency"] = pd.to_numeric(
                    efficiency_df["exit_efficiency"], errors="coerce"
                )
                efficiency_df.dropna(subset=["exit_reason", "exit_efficiency"], inplace=True)
                if not efficiency_df.empty:
                    avg_efficiency = efficiency_df.groupby("exit_reason")["exit_efficiency"].mean()
                    avg_efficiency_df = avg_efficiency.reset_index()
                    avg_efficiency_df.columns = ["exit_reason", "avg_exit_efficiency"]
                    efficiency_table = dash_table.DataTable(
                        data=avg_efficiency_df.to_dict("records"),
                        columns=[
                            {"name": "Exit Reason", "id": "exit_reason"},
                            {
                                "name": "Avg Exit Efficiency",
                                "id": "avg_exit_efficiency",
                                "type": "numeric",
                                "format": Format(precision=2, scheme=Scheme.percentage),
                            },
                        ],
                        style_table={"overflowX": "auto"},
                        style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
                        page_size=10,
                    )
                    exit_reason_cards.append(
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader("Avg exit_efficiency by exit_reason"),
                                    dbc.CardBody(efficiency_table),
                                ],
                                className="h-100",
                            ),
                            md=6,
                        )
                    )

                    efficiency_fig = px.bar(
                        avg_efficiency_df,
                        x="exit_reason",
                        y="avg_exit_efficiency",
                        template="plotly_dark",
                        title="Average Exit Efficiency",
                    )
                    efficiency_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

            preferred_columns = [
                "symbol",
                "entry_time",
                "exit_time",
                "side",
                "qty",
                "entry_price",
                "exit_price",
                "exit_pct",
                "exit_reason",
                "exit_efficiency",
                "mfe_pct",
                "net_pnl",
            ]
            columns_order = [col for col in preferred_columns if col in trades_df.columns]
            columns_order.extend([col for col in trades_df.columns if col not in columns_order])
            column_defs: list[dict[str, Any]] = []
            for col in columns_order:
                col_def: dict[str, Any] = {"name": col.replace("_", " ").title(), "id": col}
                if col in {"exit_efficiency", "exit_pct", "mfe_pct"}:
                    col_def["type"] = "numeric"
                    col_def["format"] = Format(precision=2, scheme=Scheme.percentage)
                column_defs.append(col_def)

            table = dash_table.DataTable(
                id="executed-trades-table",
                columns=column_defs,
                data=trades_df[columns_order].to_dict("records"),
                page_size=15,
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
                style_data_conditional=(
                    [
                        {
                            "if": {"filter_query": "{net_pnl} < 0", "column_id": "net_pnl"},
                            "color": "#E57373",
                        },
                        {
                            "if": {"filter_query": "{net_pnl} > 0", "column_id": "net_pnl"},
                            "color": "#4DB6AC",
                        },
                    ]
                    if "net_pnl" in trades_df.columns
                    else []
                ),
            )

            summary_children: list[Any] = []
            if exit_reason_cards:
                summary_children.append(dbc.Row(exit_reason_cards, className="g-3 mb-3"))
            if efficiency_fig is not None:
                summary_children.append(
                    dbc.Card(
                        dcc.Graph(figure=efficiency_fig, style={"height": "320px"}),
                        className="mb-4",
                    )
                )
            if summary_children:
                summary_section = html.Div(
                    [html.H5("Trade Exit Summary", className="text-light"), *summary_children],
                    className="mb-3",
                )
        else:
            hint = dbc.Alert(
                "No trades yet (paper).",
                color="info",
                className="mb-2",
            )
            table = html.Div([hint, *trade_alerts]) if trade_alerts else hint

        metrics_data, metrics_alert, skip_rows = load_execute_metrics()
        metrics_file_exists = os.path.exists(execute_metrics_path)

        api_error_alert = (
            html.Div(
                f"Recent API Errors: {metrics_data['api_failures']}",
                style={"color": "orange"},
            )
            if metrics_data["api_failures"] > 0
            else None
        )

        skip_table = None
        if skip_rows:
            skip_table = dash_table.DataTable(
                data=skip_rows,
                columns=[{"name": "Reason", "id": "reason"}, {"name": "Count", "id": "count"}],
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
            )

        positions_df, _ = load_csv(open_positions_path)
        position_count = len(positions_df) if positions_df is not None else 0
        trade_limit_alert = (
            html.Div(
                f"Max open trades limit reached ({MAX_OPEN_TRADES}).",
                style={"color": "orange"},
            )
            if position_count >= MAX_OPEN_TRADES
            else None
        )

        skipped_warning = None
        if metrics_data.get("skip_reasons"):
            summary_parts = [
                f"{reason}={count}" for reason, count in metrics_data["skip_reasons"].items()
            ]
            skipped_warning = dbc.Alert(
                "Skips: " + ", ".join(summary_parts),
                color="warning",
                className="mb-3",
            )

        orders_fig = go.Figure(
            data=[
                go.Bar(
                    x=["Submitted", "Filled", "Canceled"],
                    y=[
                        metrics_data.get("orders_submitted", 0),
                        metrics_data.get("orders_filled", 0),
                        metrics_data.get("orders_canceled", 0),
                    ],
                    marker_color=["#1f77b4", "#2ca02c", "#d62728"],
                )
            ]
        )
        orders_fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=40),
            title="Order Lifecycle (Last Run)",
            yaxis_title="Count",
        )

        kpi_cards = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6("Trailing Stops Attached", className="card-title"),
                                html.H3(
                                    f"{metrics_data.get('trailing_attached', 0)}",
                                    className="card-text",
                                ),
                            ]
                        ),
                        className="mb-3",
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6("API Retries", className="card-title"),
                                html.H3(
                                    f"{metrics_data.get('api_retries', 0)}", className="card-text"
                                ),
                            ]
                        ),
                        className="mb-3",
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6("Latency p95 (s)", className="card-title"),
                                html.H3(
                                    f"{metrics_data.get('latency_secs', {}).get('p95', 0.0):.2f}",
                                    className="card-text",
                                ),
                            ]
                        ),
                        className="mb-3",
                    ),
                    md=4,
                ),
            ]
        )

        recent_events = tail_log(execute_trades_log_path, limit=10)
        events_block = html.Div(
            [
                html.H5("Recent Executor Events"),
                html.Pre(
                    "\n".join(recent_events) if recent_events else "No executor activity logged.",
                    style={
                        "maxHeight": "300px",
                        "overflowY": "auto",
                        "backgroundColor": "#272B30",
                        "color": "#E0E0E0",
                        "padding": "10px",
                        "borderRadius": "4px",
                    },
                    id="execute-trades-log",
                ),
            ]
        )

        metrics_view = html.Div(
            [
                html.H5("Execute Trades Metrics"),
                kpi_cards,
                dcc.Graph(figure=orders_fig, style={"height": "320px"}),
                events_block,
            ]
        )

        download = None
        if trade_source_path:
            download = html.Div(
                [
                    html.Button("Download CSV", id="btn-download-trades"),
                    dcc.Download(id="download-executed-trades"),
                ],
                className="mb-2",
            )

        last_updated_path = (
            trade_source_path
            if trade_source_path
            else (executed_trades_path if executed_exists else trades_log_path)
        )
        last_updated = format_time(get_file_mtime(last_updated_path))
        header_children: list[Any] = [_paper_badge_component()]
        header_children.append(
            html.Span(
                f"Last Updated: {last_updated}",
                className="text-muted",
            )
        )
        if trade_source_label:
            header_children.append(
                html.Span(
                    f"Source: {trade_source_label}",
                    className="text-muted small",
                )
            )
        header = html.Div(
            header_children,
            style={"display": "flex", "gap": "0.75rem", "flexWrap": "wrap", "alignItems": "center"},
            className="mb-2",
        )

        components: list[Any] = [header]
        if metrics_alert:
            components.append(metrics_alert)
        if download is not None:
            components.append(download)
        if summary_section is not None:
            components.append(summary_section)
        components.append(table)
        components.append(html.Hr())
        paper_notice = None
        if not metrics_file_exists:
            paper_notice = dbc.Alert(
                "No execution yet today (paper).",
                color="info",
                className="mb-3",
            )
        if paper_notice:
            components.append(paper_notice)
        if trade_limit_alert:
            components.append(trade_limit_alert)
        if skipped_warning:
            components.append(skipped_warning)
        if api_error_alert:
            components.append(api_error_alert)
        if skip_table is not None:
            components.append(
                html.Div(
                    [html.H6("Skip reasons"), skip_table],
                    className="mb-4",
                )
            )
        components.append(metrics_view)
        return dbc.Container(components, fluid=True)

    elif tab == "tab-activities":
        fills_df, fills_alerts = load_recent_fills()
        components: list[Any] = [
            html.H4("Recent Fills (Alpaca Activities)", className="text-light"),
            html.Div(
                "Latest fills from v_fills_recent (7 days, max 200).",
                className="text-muted small mb-2",
            ),
        ]
        if fills_alerts:
            components.extend(fills_alerts)

        last_updated_value = None
        if not fills_df.empty and "transaction_time" in fills_df.columns:
            last_updated_value = fills_df["transaction_time"].max()
        if last_updated_value is None:
            last_updated_value = datetime.now(timezone.utc)
        last_updated_display = last_updated_value.strftime("%Y-%m-%d %H:%M:%S UTC")
        components.append(
            html.Div(f"Last updated: {last_updated_display}", className="text-muted mb-3")
        )

        if fills_df.empty:
            components.append(
                dbc.Alert(
                    "No Alpaca fills in the last 7 days. Recent bot activity will appear here.",
                    color="info",
                    className="mb-0",
                )
            )
            return dbc.Container(components, fluid=True)

        display_df = fills_df.copy()
        if "transaction_time" in display_df.columns:
            display_df["transaction_time"] = display_df["transaction_time"].dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        order_column = "order_id_short" if "order_id_short" in display_df.columns else "order_id"
        table_columns = [
            {"name": "Time (UTC)", "id": "transaction_time"},
            {"name": "Symbol", "id": "symbol"},
            {"name": "Side", "id": "side"},
            {"name": "Qty", "id": "qty", "type": "numeric"},
            {
                "name": "Price",
                "id": "price",
                "type": "numeric",
                "format": Format(precision=2, scheme=Scheme.fixed),
            },
        ]
        if order_column in display_df.columns:
            table_columns.append({"name": "Order ID", "id": order_column})
        display_df = display_df[
            [col["id"] for col in table_columns if col["id"] in display_df.columns]
        ]

        table = dash_table.DataTable(
            data=display_df.to_dict("records"),
            columns=table_columns,
            page_size=25,
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
            style_header={"fontWeight": "bold"},
        )
        components.append(table)
        return dbc.Container(components, fluid=True)

    elif tab == "tab-trades":
        return make_trades_exits_layout()

    elif tab == "tab-symbol-performance":
        paper_mode = _is_paper_mode()
        symbol_source_path = TRADES_LOG_FOR_SYMBOLS
        data_or_alert = load_symbol_perf_df()
        if isinstance(data_or_alert, dbc.Alert):
            # Friendly message for paper mode with no trades yet.
            alert_color = None
            if hasattr(data_or_alert, "props"):
                alert_color = getattr(data_or_alert, "props", {}).get("color")
            if alert_color is None and hasattr(data_or_alert, "kwargs"):
                alert_color = getattr(data_or_alert, "kwargs", {}).get("color")
            if (
                paper_mode
                and symbol_source_path == TRADES_LOG_PAPER
                and (alert_color in {None, "info"})
            ):
                return dbc.Alert(
                    "Paper mode: no paper trades yet. Run the pre-market executor to populate data/trades_log.csv.",
                    color="info",
                    className="m-2",
                )
            return data_or_alert

        trades_df = data_or_alert
        if trades_df is None or trades_df.empty:
            if paper_mode and symbol_source_path == TRADES_LOG_PAPER:
                return dbc.Alert(
                    "Paper mode: no paper trades yet. Run the pre-market executor to populate data/trades_log.csv.",
                    color="info",
                    className="m-2",
                )
            return dbc.Alert(
                f"No trade data available in {symbol_source_path.name}.",
                color="warning",
                className="m-2",
            )

        if "net_pnl" not in trades_df.columns:
            for candidate in ("pnl", "profit", "netPnL", "net_pnl_usd"):
                if candidate in trades_df.columns:
                    trades_df = trades_df.copy()
                    trades_df["net_pnl"] = trades_df[candidate]
                    break
        if "net_pnl" not in trades_df.columns:
            return dbc.Alert(
                "Trades log is missing a P&L column (pnl/net_pnl).",
                color="warning",
                className="m-2",
            )

        app.logger.info(
            "Loaded %d trades for symbol performance from %s",
            len(trades_df),
            symbol_source_path,
        )

        grouped = trades_df.groupby("symbol")["net_pnl"]
        avg_pnl = grouped.mean()
        total_pnl = grouped.sum()
        trade_count = grouped.count()
        symbol_perf = pd.DataFrame(
            {
                "Symbol": avg_pnl.index,
                "Average P/L": avg_pnl.values,
                "Total P/L": total_pnl.values,
                "Trades": trade_count.values,
            }
        )
        symbol_fig = px.bar(
            symbol_perf,
            x="Symbol",
            y="Total P/L",
            color="Total P/L",
            template="plotly_dark",
            title="Performance by Symbol",
        )
        columns = [{"name": c, "id": c} for c in symbol_perf.columns]
        table = dash_table.DataTable(
            data=symbol_perf.to_dict("records"),
            columns=columns,
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
            style_data_conditional=[
                {
                    "if": {"filter_query": "{Total P/L} < 0", "column_id": "Total P/L"},
                    "color": "#E57373",
                },
                {
                    "if": {"filter_query": "{Total P/L} > 0", "column_id": "Total P/L"},
                    "color": "#4DB6AC",
                },
            ],
        )
        last_updated = format_time(get_file_mtime(str(symbol_source_path)))
        timestamp = html.Div(
            f"Last Updated: {last_updated}",
            className="text-muted mb-2",
        )
        freshness = data_freshness_alert(str(symbol_source_path), "Trades log")
        components = [timestamp]
        if freshness:
            components.append(freshness)
        components.extend([dcc.Graph(figure=symbol_fig), table])
        return dbc.Container(components, fluid=True)

    elif tab == "tab-account":
        return render_account_tab()

    elif tab == "tab-monitor":
        # Load open positions
        positions_df, freshness_alert = load_csv(
            open_positions_path,
            ["symbol", "qty", "net_pnl"],
            alert_prefix="Real positions",
        )

        if positions_df is not None and not positions_df.empty:
            positions_df["entry_time"] = pd.to_datetime(
                positions_df["entry_time"], utc=True, errors="coerce"
            )
            positions_df.sort_values("entry_time", ascending=False, inplace=True)
            positions_df = add_days_in_trade(positions_df)
            positions_chart = create_open_positions_chart(positions_df)
            positions_table = create_open_positions_table(positions_df)
        else:
            positions_chart = html.Div("No open positions available.")
            positions_table = html.Div()

        monitor_lines = read_recent_lines(monitor_log_path, num_lines=100)[::-1]
        exec_lines = read_recent_lines(execute_trades_log_path, num_lines=100)[::-1]
        error_lines = read_recent_lines(
            error_log_path, num_lines=100, since_days=ERROR_RETENTION_DAYS
        )[::-1]

        monitor_log_box = log_box(
            "Monitor Log", monitor_lines, "monitor-log", log_path=monitor_log_path
        )
        exec_log_box = log_box(
            "Execution Log",
            exec_lines,
            "exec-log",
            log_path=execute_trades_log_path,
        )
        error_log_box = log_box("Errors", error_lines, "error-log", log_path=error_log_path)

        stale_warning_banner = stale_warning(
            [monitor_log_path, open_positions_path], threshold_minutes=10
        )

        last_updated = format_time(get_file_mtime(open_positions_path))
        timestamp = html.Div(
            f"Last Updated: {last_updated}",
            className="text-muted mb-2",
        )

        return html.Div(
            [
                freshness_alert if freshness_alert else html.Div(),
                stale_warning_banner if stale_warning_banner else html.Div(),
                timestamp,
                positions_chart,
                positions_table,
                html.Hr(),
                monitor_log_box,
                exec_log_box,
                error_log_box,
            ]
        )
    else:
        return dbc.Alert("Unknown tab requested.", color="danger")


def account_layout():
    return _render_tab("tab-account", 0, 0, None)


def symbol_performance_layout():
    return _render_tab("tab-symbol-performance", 0, 0, None)


def monitor_positions_layout():
    return _render_tab("tab-monitor", 0, 0, None)


def execute_trades_layout():
    return _render_tab("tab-execute", 0, 0, None)


def screener_layout():
    return _render_tab("tab-screener", 0, 0, None)


@app.callback(Output("refresh-ts", "data"), Input("refresh-button", "n_clicks"))
def _refresh_ts(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return datetime.utcnow().isoformat()


def _empty_trade_perf_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", title=title)
    return fig


def _prepare_trade_perf_frame(store_data: Mapping[str, Any], window: str) -> pd.DataFrame:
    if not store_data or not store_data.get("trades"):
        return pd.DataFrame()
    frame = pd.DataFrame(store_data.get("trades"))
    for col in ("entry_time", "exit_time"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    numeric_cols = [
        "qty",
        "entry_price",
        "exit_price",
        "pnl",
        "return_pct",
        "hold_days",
        "mfe_pct",
        "mae_pct",
        "peak_price",
        "trough_price",
        "missed_profit_pct",
        "exit_efficiency_pct",
        "rebound_pct",
        "rebound_window_days",
        "post_exit_high",
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "exit_efficiency_pct" not in frame.columns:
        frame["exit_efficiency_pct"] = np.nan
    if "rebounded" in frame.columns:
        frame["rebounded"] = frame["rebounded"].fillna(False)
    if "is_trailing_stop_exit" in frame.columns:
        frame["is_trailing_stop_exit"] = frame["is_trailing_stop_exit"].fillna(False)
    if "exit_reason" in frame.columns:
        frame["exit_reason"] = frame["exit_reason"].fillna("Unknown")
    if window != "ALL" and "exit_time" in frame.columns:
        now = datetime.now(timezone.utc)
        days = 365 if window == "365D" else int(window.replace("D", ""))
        cutoff = now - timedelta(days=days)
        frame = frame[frame["exit_time"] >= cutoff]
    return frame


@app.callback(
    [
        Output("trade-perf-kpis", "children"),
        Output("trade-perf-window-bar", "figure"),
        Output("trade-perf-scatter", "figure"),
        Output("trade-perf-eff-hist", "figure"),
        Output("trade-perf-reason-bar", "figure"),
        Output("trade-perf-rebound-scatter", "figure"),
        Output("trade-perf-rebound-hist", "figure"),
        Output("trade-perf-table", "data"),
        Output("trade-perf-status", "children"),
    ],
    [Input("trade-perf-range", "value"), Input("trade-perf-store", "data")],
)
def update_trade_performance_tab(range_value: str, store_data: Mapping[str, Any]):
    window = range_value or "30D"
    store_mapping = store_data if isinstance(store_data, Mapping) else {}
    frame = _prepare_trade_perf_frame(store_mapping, window)
    summary = store_mapping.get("summary", {}) if isinstance(store_mapping, Mapping) else {}
    if not summary:
        summary = summarize_by_window(frame)
    metrics = summary.get(window) or summarize_by_window(frame).get(window, {})
    summary_bar = _build_window_net_pnl_bar(summary)
    kpi_cards = _build_trade_perf_kpis(summary)
    status_children: list[Any] = []
    if store_mapping.get("written_at"):
        status_children.append(
            dbc.Badge(
                f"Cache updated {store_mapping['written_at']}", color="secondary", className="me-2"
            )
        )
    if metrics:
        status_children.append(
            dbc.Badge(
                f"Sold-too-soon flags: {metrics.get('sold_too_soon', 0)}",
                color="info",
            )
        )
        status_children.append(
            dbc.Badge(
                f"Trailing-stop exits: {metrics.get('stop_exits', 0)}; rebounds: {metrics.get('rebounds', 0)}",
                color="secondary",
                className="ms-2",
            )
        )

    if frame.empty:
        empty_data = store_mapping.get("trades", []) if isinstance(store_mapping, Mapping) else []
        return (
            kpi_cards,
            summary_bar,
            _empty_trade_perf_fig("MFE % vs Return %"),
            _empty_trade_perf_fig("Exit efficiency %"),
            _empty_trade_perf_fig("Exit reasons"),
            _empty_trade_perf_fig("Exit Efficiency vs Rebound %"),
            _empty_trade_perf_fig("Rebound %"),
            empty_data,
            status_children,
        )

    if "exit_reason" not in frame.columns:
        frame["exit_reason"] = "Unknown"

    scatter_fig = px.scatter(
        frame,
        x="mfe_pct",
        y="return_pct",
        color="exit_reason",
        hover_data=["symbol", "exit_time", "return_pct", "exit_efficiency_pct"],
        title="MFE % vs Return %",
    )
    scatter_fig.update_layout(template="plotly_dark")

    eff_hist = px.histogram(
        frame,
        x="exit_efficiency_pct",
        nbins=20,
        title="Exit efficiency %",
    )
    eff_hist.update_layout(template="plotly_dark")

    reason_counts = frame["exit_reason"].value_counts()
    reason_bar = go.Figure(
        data=[go.Bar(x=reason_counts.index, y=reason_counts.values)],
        layout=go.Layout(title="Exit reasons", template="plotly_dark"),
    )
    trailing_mask = frame.get("is_trailing_stop_exit")
    if trailing_mask is None:
        trailing_mask = pd.Series(False, index=frame.index)
    trailing_frame = frame[trailing_mask.fillna(False)]
    valid_rebounds = (
        trailing_frame.dropna(subset=["rebound_pct"])
        if (not trailing_frame.empty and "rebound_pct" in trailing_frame.columns)
        else pd.DataFrame()
    )
    rebound_scatter = _empty_trade_perf_fig("Exit Efficiency % vs Rebound %")
    rebound_hist = _empty_trade_perf_fig("Rebound %")
    if not valid_rebounds.empty:
        rebound_scatter = px.scatter(
            valid_rebounds,
            x="exit_efficiency_pct",
            y="rebound_pct",
            color="rebounded",
            hover_data=["symbol", "exit_time", "exit_price", "rebound_pct"],
            title="Exit Efficiency % vs Rebound %",
        )
        rebound_scatter.update_layout(template="plotly_dark")
        rebound_hist = px.histogram(
            valid_rebounds,
            x="rebound_pct",
            nbins=15,
            title="Rebound %",
        )
        rebound_hist.update_layout(template="plotly_dark")

    table_frame = frame.copy()
    if "exit_time" in table_frame.columns:
        table_frame = table_frame.sort_values("exit_time", ascending=False, na_position="last")
    elif "entry_time" in table_frame.columns:
        table_frame = table_frame.sort_values("entry_time", ascending=False, na_position="last")
    table_frame = table_frame.head(50)
    for col in ("entry_time", "exit_time"):
        if col in table_frame.columns:
            table_frame[col] = table_frame[col].dt.strftime("%Y-%m-%d %H:%M")
    table_frame = table_frame.reindex(columns=TRADE_PERF_TABLE_FIELDS)
    table_data = table_frame.fillna("").to_dict("records")

    return (
        kpi_cards,
        summary_bar,
        scatter_fig,
        eff_hist,
        reason_bar,
        rebound_scatter,
        rebound_hist,
        table_data,
        status_children,
    )


def _build_trade_pnl_summary(frame: pd.DataFrame) -> list[Any]:
    if frame is None or frame.empty:
        return [
            _format_chip("Trades", "0"),
            _format_chip("Total P&L", "0.00"),
            _format_chip("Avg P&L", "0.00"),
            _format_chip("Median P&L", "0.00"),
            _format_chip("Win rate", "0.0%"),
            _format_chip("Gross profit", "0.00"),
            _format_chip("Gross loss", "0.00"),
            _format_chip("Profit factor", "n/a"),
            _format_chip("Best/Worst trade", "n/a"),
        ]

    pnl = pd.to_numeric(frame.get("pnl"), errors="coerce")
    pnl = pnl.fillna(0.0)
    trades = len(pnl.index)
    total_pnl = float(pnl.sum())
    avg_pnl = float(pnl.mean()) if trades else 0.0
    median_pnl = float(pnl.median()) if trades else 0.0
    wins = pnl[pnl > 0].count()
    win_rate = (wins / trades) * 100.0 if trades else 0.0
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(pnl[pnl < 0].sum())
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss) if abs(gross_loss) > 0 else float("inf")
        profit_factor_display = f"{profit_factor:.2f}" if math.isfinite(profit_factor) else "∞"
    else:
        profit_factor_display = "∞" if gross_profit > 0 else "n/a"

    best = float(pnl.max()) if not pnl.empty else 0.0
    worst = float(pnl.min()) if not pnl.empty else 0.0
    best_worst = f"{best:,.2f} / {worst:,.2f}" if trades else "n/a"

    return [
        _format_chip("Trades", f"{trades}"),
        _format_chip("Total P&L", f"{total_pnl:,.2f}"),
        _format_chip("Avg P&L", f"{avg_pnl:,.2f}"),
        _format_chip("Median P&L", f"{median_pnl:,.2f}"),
        _format_chip("Win rate", f"{win_rate:.1f}%"),
        _format_chip("Gross profit", f"{gross_profit:,.2f}"),
        _format_chip("Gross loss", f"{gross_loss:,.2f}"),
        _format_chip("Profit factor", profit_factor_display),
        _format_chip("Best/Worst trade", best_worst),
    ]


@app.callback(
    [
        Output("trade-pnl-table", "data"),
        Output("trade-pnl-summary-chips", "children"),
    ],
    [
        Input("active-tab-store", "data"),
        Input("refresh-ts", "data"),
        Input("trade-perf-range", "value"),
    ],
)
def update_trade_pnl_table(
    active_tab: Mapping[str, Any] | None, _refresh_ts: str | None, window: str | None
):
    if isinstance(active_tab, Mapping) and active_tab.get("active_tab") not in (
        None,
        "tab-trade-performance",
    ):
        return dash.no_update, dash.no_update

    window = window or "30D"
    payload = _load_trade_perf_cache()
    trades = payload.get("trades", [])
    if not trades:
        empty_frame = pd.DataFrame(columns=["exit_time"])
        return [], _build_trade_pnl_summary(empty_frame)

    frame = pd.DataFrame(trades)
    if "exit_time" in frame.columns:
        frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    if "entry_time" in frame.columns:
        frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")

    for col in ("qty", "entry_price", "exit_price", "pnl", "return_pct", "hold_days"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = np.nan

    frame["qty"] = frame["qty"].fillna(0.0)
    frame["entry_price"] = frame["entry_price"].fillna(0.0)
    frame["exit_price"] = frame["exit_price"].fillna(0.0)
    frame["entry_value"] = frame["qty"] * frame["entry_price"]
    frame["exit_value"] = frame["qty"] * frame["exit_price"]

    pnl_fallback = frame["exit_value"] - frame["entry_value"]
    if "pnl" not in frame.columns:
        frame["pnl"] = pnl_fallback
    else:
        frame["pnl"] = frame["pnl"].fillna(pnl_fallback)
        missing_mask = pd.to_numeric(frame["pnl"], errors="coerce").isna()
        frame.loc[missing_mask, "pnl"] = pnl_fallback[missing_mask]

    if "hold_days" not in frame.columns:
        frame["hold_days"] = np.nan
    if (
        frame["hold_days"].isna().any()
        and "entry_time" in frame.columns
        and "exit_time" in frame.columns
    ):
        frame.loc[frame["hold_days"].isna(), "hold_days"] = (
            frame["exit_time"] - frame["entry_time"]
        ).dt.total_seconds() / 86400.0

    frame["side"] = frame["side"] if "side" in frame.columns else "long"
    frame["side"] = frame["side"].fillna("long")
    if "order_type" not in frame.columns:
        frame["order_type"] = ""
    if "exit_reason" not in frame.columns:
        frame["exit_reason"] = ""

    if window != "ALL" and "exit_time" in frame.columns:
        now = datetime.now(timezone.utc)
        days = 365 if window == "365D" else int(window.replace("D", "") or 0)
        cutoff = now - timedelta(days=days)
        frame = frame[frame["exit_time"] >= cutoff]

    if "exit_time" in frame.columns:
        frame = frame.sort_values("exit_time", ascending=False, na_position="last")
        frame["exit_time"] = frame["exit_time"].dt.strftime("%Y-%m-%d %H:%M")

    table_columns = [
        "exit_time",
        "symbol",
        "side",
        "qty",
        "entry_price",
        "exit_price",
        "entry_value",
        "exit_value",
        "pnl",
        "return_pct",
        "hold_days",
        "order_type",
        "exit_reason",
    ]
    table_frame = frame.reindex(columns=table_columns).fillna("")
    summary_chips = _build_trade_pnl_summary(frame)

    return table_frame.to_dict("records"), summary_chips


def _format_chip(label: str, value: str | float) -> dbc.Badge:
    return dbc.Badge(f"{label}: {value}", color="secondary", className="me-2")


@app.callback(
    [
        Output("sold-too-soon-table", "data"),
        Output("sold-too-soon-summary", "children"),
    ],
    [
        Input("active-tab-store", "data"),
        Input("refresh-ts", "data"),
        Input("sold-too-mode", "value"),
        Input("sold-too-eff-cutoff", "value"),
        Input("sold-too-missed-cutoff", "value"),
        Input("sold-too-rebound-threshold", "value"),
        Input("sold-too-rebound-window", "value"),
    ],
)
def update_sold_too_soon_table(
    active_tab: Mapping[str, Any] | None,
    _refresh_ts: str | None,
    mode: str,
    eff_cutoff: float,
    missed_cutoff: float,
    rebound_threshold: float,
    rebound_window_days: int,
):
    if isinstance(active_tab, Mapping) and active_tab.get("active_tab") not in (
        None,
        "tab-trade-performance",
    ):
        return dash.no_update, dash.no_update

    payload = _load_trade_perf_cache()
    trades = payload.get("trades", [])
    if not trades:
        return [], []

    try:
        evaluated = evaluate_sold_too_soon_flags(
            trades,
            efficiency_cutoff_pct=eff_cutoff or 0.0,
            missed_profit_cutoff_pct=missed_cutoff or 0.0,
            mode=mode or "either",
            rebound_threshold_pct=rebound_threshold,
            rebound_window_days=rebound_window_days,
        )
    except Exception:
        return [], []

    flagged = evaluated[evaluated["sold_too_soon_flag"] == True]  # noqa: E712
    total_trades = payload.get("trades_total", len(evaluated.index))
    try:
        total_trades = int(total_trades)
    except Exception:
        total_trades = len(evaluated.index)
    flagged_count = len(flagged.index)

    def _safe_mean(series: pd.Series) -> float:
        if series is None or series.empty:
            return 0.0
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return 0.0
        return float(numeric.mean())

    avg_return = _safe_mean(flagged.get("return_pct", pd.Series(dtype=float)))
    avg_eff = _safe_mean(flagged.get("exit_efficiency_pct", pd.Series(dtype=float)))
    total_pnl = (
        float(pd.to_numeric(flagged.get("pnl", pd.Series(dtype=float)), errors="coerce").sum())
        if flagged_count
        else 0.0
    )

    summary = [
        _format_chip("Flagged", f"{flagged_count} of {total_trades}"),
        _format_chip("Avg return (flagged)", f"{avg_return:.2f}%"),
        _format_chip("Avg exit efficiency", f"{avg_eff:.2f}%"),
        _format_chip("Total P&L", f"{total_pnl:,.2f}"),
    ]

    table_frame = flagged.copy()
    for col in ("entry_time", "exit_time"):
        if col in table_frame.columns:
            table_frame[col] = pd.to_datetime(table_frame[col], utc=True, errors="coerce")
    if "exit_time" in table_frame.columns:
        table_frame = table_frame.sort_values("exit_time", ascending=False, na_position="last")
        table_frame["exit_time"] = table_frame["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
    desired_columns = [
        "exit_time",
        "symbol",
        "order_type",
        "return_pct",
        "exit_efficiency_pct",
        "missed_profit_pct",
        "rebound_pct",
        "rebounded",
        "hold_days",
        "pnl",
    ]
    table_frame = table_frame.reindex(columns=desired_columns).fillna("")

    return table_frame.to_dict("records"), summary


@app.callback(
    Output("ml-predictions-table", "children"),
    [Input("predictions-dropdown", "value"), Input("refresh-ts", "data")],
)
def update_ml_predictions_table(prediction_path, refresh_ts=None):
    _ = refresh_ts
    return build_predictions_table(prediction_path)


@app.callback(
    Output("tabs-content", "children"),
    [Input("active-tab-store", "data"), Input("refresh-ts", "data")],
)
def render_tab(store_data, refresh_ts=None):
    _ = refresh_ts
    tab = DEFAULT_ACTIVE_TAB
    if isinstance(store_data, Mapping) and store_data.get("active_tab"):
        tab = store_data["active_tab"]
    if tab == "tab-overview":
        logger.info("Rendering content for tab: %s", tab)
        return overview_layout()
    if tab == "tab-pipeline":
        logger.info("Rendering content for tab: %s", tab)
        return pipeline_layout()
    if tab == "tab-ml":
        logger.info("Rendering content for tab: %s", tab)
        return ml_layout()
    if tab == "tab-screener-health":
        logger.info("Rendering content for tab: %s", tab)
        return build_screener_health()
    elif tab == "tab-account":
        logger.info("Rendering content for tab: %s", tab)
        return account_layout()
    elif tab == "tab-symbol-performance":
        logger.info("Rendering content for tab: %s", tab)
        return symbol_performance_layout()
    elif tab == "tab-monitor":
        logger.info("Rendering content for tab: %s", tab)
        return monitor_positions_layout()
    elif tab == "tab-execute":
        logger.info("Rendering content for tab: %s", tab)
        return execute_trades_layout()
    elif tab == "tab-trades":
        logger.info("Rendering content for tab: %s", tab)
        return make_trades_exits_layout()
    elif tab == "tab-trade-performance":
        logger.info("Rendering content for tab: %s", tab)
        return render_trade_performance_panel()
    elif tab == "tab-screener":
        logger.info("Rendering content for tab: %s", tab)
        return screener_layout()
    else:
        return dbc.Alert("Tab not implemented yet.", color="secondary")


_render_tab_signature = inspect.signature(render_tab)
_render_tab_param_count = len(_render_tab_signature.parameters)
if _render_tab_param_count != 2:
    message = (
        "render_tab callback signature mismatch: expected 2 parameters (store_data, refresh_ts) "
        f"but found {_render_tab_param_count}."
    )
    logger.error(message)
    raise RuntimeError(message)


@app.callback(
    Output("today-timeline-table", "children"),
    [
        Input("timeline-events-store", "data"),
        Input("timeline-source-filter", "value"),
        Input("timeline-severity-filter", "value"),
    ],
)
def _update_timeline_table(events, source_filter, severity_filter):
    if events is None:
        raise PreventUpdate
    tz_label = None
    try:
        if isinstance(events, list) and events:
            tz_label = events[0].get("tz_label")
    except Exception:
        tz_label = None
    return render_timeline_table(
        events if isinstance(events, list) else [],
        source_filter or "all",
        severity_filter or "all",
        tz_label,
    )


# Callback for modal interaction
@app.callback(
    [Output("detail-modal", "is_open"), Output("modal-content", "children")],
    [
        Input("screener-table", "active_cell"),
        Input("screener-table", "data"),
        Input("close-modal", "n_clicks"),
    ],
    [State("detail-modal", "is_open")],
)
def toggle_modal(active_cell, table_data, close_click, is_open):
    if active_cell and not is_open and table_data:
        row = table_data[active_cell["row"]]
        symbol = row.get("symbol")
        path = os.path.join(BASE_DIR, "data", "history_cache", f"{symbol}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                return True, dbc.Alert(f"Error loading data for {symbol}: {exc}", color="danger")
            df = df.dropna(subset=["timestamp", "close", "open", "high", "low"])
            if df.empty:
                return True, dbc.Alert("Insufficient historical data for chart.", color="warning")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            close_series = pd.to_numeric(df["close"], errors="coerce")
            df["SMA9"] = close_series.rolling(9).mean()
            df["EMA20"] = close_series.ewm(span=20, adjust=False).mean()
            df["SMA180"] = close_series.rolling(180).mean()
            mid = close_series.rolling(20).mean()
            std = close_series.rolling(20).std(ddof=0)
            df["BB_MID"] = mid
            df["BB_UPPER"] = mid + 2 * std
            df["BB_LOWER"] = mid - 2 * std
            macd_line, macd_signal, macd_hist = _macd(close_series)
            df["MACD_LINE"] = macd_line
            df["MACD_SIGNAL"] = macd_signal
            df["MACD_HIST"] = macd_hist
            df["RSI"] = _rsi(close_series)
            df["ADX"] = _adx(df, period=14)
            df["OBV"] = _obv(df)
            volume_series = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0)

            specs = [
                [{"secondary_y": False}],
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
            ]
            fig = make_subplots(
                rows=5,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.45, 0.2, 0.15, 0.1, 0.1],
                specs=specs,
            )
            fig.add_trace(
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=close_series,
                    name="Price",
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=df["SMA9"], name="SMA9", line=dict(color="#feca57")
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=df["EMA20"], name="EMA20", line=dict(color="#54a0ff")
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=df["SMA180"], name="SMA180", line=dict(color="#ff6b6b")
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["BB_UPPER"],
                    name="Bollinger Upper",
                    line=dict(color="#8395a7", width=1),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["BB_LOWER"],
                    name="Bollinger Lower",
                    line=dict(color="#8395a7", width=1),
                    fill="tonexty",
                    fillcolor="rgba(131,149,167,0.15)",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(x=df["timestamp"], y=volume_series, name="Volume", marker_color="#576574"),
                row=2,
                col=1,
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["OBV"], name="OBV", line=dict(color="#1dd1a1")),
                row=2,
                col=1,
                secondary_y=True,
            )
            fig.add_trace(
                go.Bar(
                    x=df["timestamp"], y=df["MACD_HIST"], name="MACD Hist", marker_color="#10ac84"
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"], y=df["MACD_LINE"], name="MACD", line=dict(color="#ff9f43")
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["MACD_SIGNAL"],
                    name="Signal",
                    line=dict(color="#54a0ff"),
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["RSI"], name="RSI", line=dict(color="#c8d6e5")),
                row=4,
                col=1,
            )
            fig.add_hline(y=70, line=dict(color="#ff6b6b", width=1, dash="dash"), row=4, col=1)
            fig.add_hline(y=30, line=dict(color="#54a0ff", width=1, dash="dash"), row=4, col=1)
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["ADX"], name="ADX", line=dict(color="#ff9ff3")),
                row=5,
                col=1,
            )
            fig.update_layout(
                template="plotly_dark",
                title=f"{symbol} Daily Technical Overview",
                height=900,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=False)
            fig.update_yaxes(title_text="OBV", row=2, col=1, secondary_y=True)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="RSI", row=4, col=1, range=[0, 100])
            fig.update_yaxes(title_text="ADX", row=5, col=1)
            return True, dcc.Graph(figure=fig)
        return True, html.Div(f"No data for {symbol}")
    if close_click and is_open:
        return False, ""
    return is_open, ""


# Periodically refresh screener table
@app.callback(Output("screener-table", "data"), Input("interval-update", "n_intervals"))
def update_screener_table(n):
    df, alert = load_latest_candidates()
    if alert:
        return []
    if df is None or df.empty:
        logger.info("Screener table update skipped; no rows available.")
        return []
    df = df.sort_values("score", ascending=False)
    payload = df.to_dict("records")
    logger.info(
        "Screener table updated successfully with %d records (source=latest_candidates).",
        len(payload),
    )
    return payload


# Periodically refresh metrics log display
@app.callback(
    Output("metrics-logs", "children"),
    Input("interval-update", "n_intervals"),
)
def update_metrics_logs(n):
    """Read the metrics log file and return the most recent entries."""
    try:
        with open(metrics_log_path, "r") as file:
            log_lines = file.readlines()
        recent_log_lines = log_lines[-50:][::-1]
        return "".join(recent_log_lines)
    except FileNotFoundError:
        return "Metrics log file not found."


# Periodically refresh executed trades table
@app.callback(
    Output("executed-trades-table", "data"),
    Input("interval-trades", "n_intervals"),
)
def refresh_trades_table(n):
    df, _, _, _ = _resolve_trades_dataframe()
    if df is None or df.empty:
        return []
    try:
        df = df.sort_values("entry_time", ascending=False)
    except Exception:
        df = df.copy()
    return df.to_dict("records")


# Periodically update execution logs
@app.callback(
    Output("execute-trades-log", "children"),
    Input("log-interval", "n_intervals"),
)
def update_execution_logs(n):
    if os.path.exists(execute_trades_log_path):
        with open(execute_trades_log_path, "r") as file:
            lines = file.readlines()
        recent_lines = lines[-50:][::-1]
        return "".join(recent_lines)
    return ""


# Download executed trades CSV
@app.callback(
    Output("download-executed-trades", "data"),
    Input("btn-download-trades", "n_clicks"),
    prevent_initial_call=True,
)
def download_trades(n_clicks):
    if not n_clicks:
        return dash.no_update
    _, path, _, _ = _resolve_trades_dataframe()
    candidate = path or (
        executed_trades_path if os.path.exists(executed_trades_path) else trades_log_path
    )
    if candidate and os.path.exists(candidate):
        return dcc.send_file(candidate)
    return dash.no_update


if __name__ == "__main__":
    app.run(debug=False)
