# Complete Integrated Dashboard (dashboard_app.py)

import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.dash_table import Format
from datetime import datetime, timezone, timedelta
import subprocess
import json
import math
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import logging
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import pytz
import re
from pathlib import Path
from typing import Any, Mapping, Optional
from flask import jsonify
from plotly.subplots import make_subplots

os.environ.setdefault("JBRAVO_HOME", "/home/oai/jbravo_screener")

logger = logging.getLogger(__name__)

from dashboards.screener_health import build_layout as build_screener_health
from dashboards.screener_health import register_callbacks as register_screener_health
from dashboards.screener_health import load_kpis as _load_screener_health_kpis
from dashboards.data_io import (
    screener_health as load_screener_health,
    screener_table,
    metrics_summary_snapshot,
)
from dashboards.utils import coerce_kpi_types, parse_pipeline_summary
from dashboards.overview import overview_layout
from dashboards.pipeline_tab import pipeline_layout
from dashboards.ml_tab import ml_layout
from scripts.run_pipeline import write_complete_screener_metrics
from scripts.indicators import macd as _macd, rsi as _rsi, adx as _adx, obv as _obv

# Base directory of the project (parent of this file)
BASE_DIR = os.environ.get(
    "JBRAVO_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

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
account_equity_path = os.path.join(BASE_DIR, "data", "account_equity.csv")

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
    """Return ``df`` with common P&L columns normalized to ``net_pnl``."""

    if df is None:
        return None
    for candidate in ("net_pnl", "pnl", "netPnL", "net_pnl_usd"):
        if candidate in df.columns:
            if candidate != "net_pnl":
                df = df.rename(columns={candidate: "net_pnl"})
            break
    return df


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
        fragments.append(
            f'<span class="badge bg-{color} me-1">{key}: {numeric:+.2f}</span>'
        )
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

load_dotenv(os.path.join(BASE_DIR, ".env"))
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
if not API_KEY or not API_SECRET:
    trading_client = None
    logger.warning("Missing Alpaca credentials; Alpaca API features disabled")
else:
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
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
                    "entry_time": getattr(
                        p, "created_at", datetime.utcnow()
                    ).isoformat(),
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
        return None, dbc.Alert(
            f"{prefix}No data yet in {csv_path}.", color="info"
        )
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
        key for key in ("symbols_in", "symbols_with_bars", "bars_rows_total", "rows")
        if not isinstance(payload.get(key), int)
    ]
    if missing:
        pipeline_fallback = coerce_kpi_types(parse_pipeline_summary(Path(pipeline_log_path)))
        for key, value in pipeline_fallback.items():
            if key in defaults and payload.get(key) is None and value is not None:
                payload[key] = value
        still_missing = [
            key for key in ("symbols_in", "symbols_with_bars", "bars_rows_total", "rows")
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
    """Load canonical latest_candidates.csv with header validation."""

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
    ]
    path = latest_candidates_path
    if not os.path.exists(path):
        return None, dbc.Alert(
            "No candidates yet. latest_candidates.csv not found.", color="info"
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return None, dbc.Alert(
            f"Unable to read latest_candidates.csv: {exc}", color="danger"
        )

    header = list(df.columns)
    if header != canonical_header:
        return None, dbc.Alert(
            [
                html.Div("latest_candidates.csv header mismatch."),
                html.Div(f"Expected: {', '.join(canonical_header)}"),
                html.Div(f"Found: {', '.join(header)}"),
            ],
            color="warning",
        )

    if df.empty:
        return None, dbc.Alert(
            "No candidates today (fallback may populate).", color="info"
        )
    return df, None


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
        return defaults, dbc.Alert(
            "Execution has not produced metrics yet (execute_metrics.json missing).",
            color="info",
            className="mb-2",
        ), skip_rows

    try:
        with open(execute_metrics_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return defaults, dbc.Alert(
            f"Unable to read execute_metrics.json: {exc}", color="danger", className="mb-2"
        ), skip_rows

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
    skip_rows = [
        {"reason": reason, "count": count} for reason, count in sorted(skip_dict.items())
    ]
    metrics["skip_reasons"] = skip_dict

    return metrics, alert, skip_rows


def load_open_positions() -> tuple[pd.DataFrame, dbc.Alert | None]:
    return _safe_csv_with_message(Path(open_positions_path), required=["symbol", "qty"])


def load_executed_trades() -> tuple[pd.DataFrame, dbc.Alert | None]:
    return _safe_csv_with_message(Path(executed_trades_path))


def load_account_equity() -> tuple[pd.DataFrame, dbc.Alert | None]:
    df, alert = _safe_csv_with_message(Path(account_equity_path), required=["timestamp", "equity"])
    if alert is None and not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
    return df, alert


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
        return {}, dbc.Alert(
            f"Unable to read last_premarket_run.json: {exc}", color="danger"
        )
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


def load_trades_for_exits() -> pd.DataFrame | dbc.Alert:
    """Load trades for exit analytics with graceful fallback."""

    df, alert = load_csv(
        str(TRADES_LOG_PATH),
        required_columns=["symbol", "entry_time", "exit_time", "pnl"],
        alert_prefix="Trades log",
    )
    if alert:
        return alert

    if not isinstance(df, pd.DataFrame):
        return dbc.Alert("Trades log is unavailable.", color="warning")

    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "exit_time" in df.columns:
        df = df.sort_values(by="exit_time", ascending=False, na_position="last")

    return df


def make_trades_exits_layout():
    """Build the Trades / Exits analytics layout."""

    df_trades = load_trades_for_exits()

    if not isinstance(df_trades, pd.DataFrame):
        return html.Div([df_trades])

    has_exit_reason = "exit_reason" in df_trades.columns
    has_exit_eff = "exit_efficiency" in df_trades.columns

    table_columns = [
        {"name": "Symbol", "id": "symbol"},
        {"name": "Entry Time", "id": "entry_time"},
        {"name": "Exit Time", "id": "exit_time"},
        {"name": "Exit %", "id": "exit_pct"}
        if "exit_pct" in df_trades.columns
        else None,
        {"name": "Net PnL", "id": "net_pnl"}
        if "net_pnl" in df_trades.columns
        else None,
    ]
    if has_exit_reason:
        table_columns.append({"name": "Exit Reason", "id": "exit_reason"})
    if has_exit_eff:
        table_columns.append({"name": "Exit Efficiency", "id": "exit_efficiency"})

    table_columns = [column for column in table_columns if column is not None]

    trades_table = dash_table.DataTable(
        id="trades-exits-table",
        columns=table_columns,
        data=df_trades.to_dict("records"),
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
    )

    if has_exit_reason:
        exit_counts = df_trades["exit_reason"].value_counts().reset_index()
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
            df_trades.groupby("exit_reason")["exit_efficiency"]
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

    return html.Div(
        [
            html.H3("Trades & Exit Analytics"),
            html.P(
                "Review how well each exit rule captures profit, using exit_reason and exit_efficiency from trades_log.csv."
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
    label = last_ts.strftime("Last log line at: %Y-%m-%d %H:%M UTC") if last_ts else "Last log line: n/a"
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
            errors_df = errors_df[pd.to_datetime(errors_df["timestamp"]) >= datetime.now() - timedelta(days=1)]
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


def _styled_table(df: pd.DataFrame, table_id: str | None = None, page_size: int = 25) -> dash_table.DataTable:
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
        ts = datetime.utcfromtimestamp(os.path.getmtime(__file__)).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        return f"Dashboard version: {commit} (updated {ts})"
    except Exception:
        ts = datetime.utcfromtimestamp(os.path.getmtime(__file__)).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
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


app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V2.1.0/dbc.min.css",
    ],
    suppress_callback_exceptions=True,
)

server = app.server


@server.route("/health/overview")
def health_overview():
    """Return a lightweight JSON payload describing dashboard health."""

    base = Path(BASE_DIR) / "data"
    ms_path = base / "metrics_summary.csv"
    tl_path = base / "trades_log.csv"
    summary: dict[str, Any] = {
        "metrics_summary_present": ms_path.exists(),
        "trades_log_present": tl_path.exists(),
        "trades_log_rows": None,
        "kpis": {},
    }

    try:
        if summary["metrics_summary_present"]:
            df = pd.read_csv(ms_path)
            if not df.empty:
                summary["kpis"] = df.iloc[-1].to_dict()
    except Exception as exc:  # pragma: no cover - telemetry helper
        summary["kpis_error"] = str(exc)

    try:
        if summary["trades_log_present"]:
            with open(tl_path, "r", encoding="utf-8", errors="ignore") as handle:
                row_count = sum(1 for _ in handle)
            summary["trades_log_rows"] = max(0, row_count - 1)
    except Exception:
        pass

    return jsonify({"ok": True, **summary})


@app.server.route("/api/health")
def api_health():
    payload = load_screener_health()
    return jsonify(payload)


@app.server.route("/api/candidates")
def api_candidates():
    candidates_path = Path(BASE_DIR) / "data" / "top_candidates.csv"
    if not candidates_path.exists():
        return jsonify({"columns": [], "rows": [], "rows_final": 0})
    try:
        df = pd.read_csv(candidates_path)
    except Exception as exc:  # pragma: no cover - defensive read
        return (
            jsonify({"columns": [], "rows": [], "rows_final": 0, "error": str(exc)}),
            500,
        )
    rows_final = int(df.shape[0])
    return jsonify(
        {
            "columns": [str(col) for col in df.columns],
            "rows": df.to_dict(orient="records"),
            "rows_final": rows_final,
        }
    )

# Layout with Tabs and Modals
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1(
                    "JBravo Swing Trading Dashboard",
                    className="text-center my-4 text-light",
                )
            )
        ),
        html.Div(
            [
                dbc.Tabs(
                    id="tabs",
                    active_tab="tab-screener-health",
                    class_name="mb-3",
                    children=[
                        dbc.Tab(
                            label="Overview",
                            tab_id="tab-overview",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="Pipeline",
                            tab_id="tab-pipeline",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="ML",
                            tab_id="tab-ml",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="Screener Health",
                            tab_id="tab-screener-health",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="Screener",
                            tab_id="tab-screener",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="Execute Trades",
                            tab_id="tab-execute",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="Account",
                            tab_id="tab-account",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="Symbol Performance",
                            tab_id="tab-symbol-performance",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="Monitoring Positions",
                            tab_id="tab-monitor",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="Trades / Exits",
                            tab_id="tab-trades",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                    ],
                ),
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
        dcc.Interval(
            id="interval-update", interval=24 * 60 * 60 * 1000, n_intervals=0
        ),
        dcc.Interval(
            id="log-interval", interval=24 * 60 * 60 * 1000, n_intervals=0
        ),
dcc.Interval(
            id="interval-trades", interval=24 * 60 * 60 * 1000, n_intervals=0
        ),
        dcc.Store(id="predictions-store"),
        dbc.Modal(
            id="detail-modal",
            is_open=False,
            size="lg",
            children=[
                dbc.ModalHeader(dbc.ModalTitle("Details")),
                dbc.ModalBody(id="modal-content"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-modal", className="ms-auto")
                ),
            ],
        ),
        html.Div(get_version_string(), id="version-banner", className="text-muted mt-2"),
    ],
    fluid=True,
)


# Register Screener Health callbacks
register_screener_health(app)


# Callbacks for tabs content
def _render_tab(tab, n_intervals, n_log_intervals, refresh_clicks):
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
                except Exception as exc:
                    logger.error(
                        "Unable to backfill screener metrics", exc_info=True
                    )
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
            return dbc.Badge(label, color=color_map.get(level, "secondary"), className="badge-small")

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
                candidates_df["__source"].iloc[0]
                if "__source" in candidates_df.columns
                else None
            )
            table_updated = (
                candidates_df["__updated"].iloc[0]
                if "__updated" in candidates_df.columns
                else None
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

        def _format_kpi_tile_value(value: Any, *, fmt: str = "{:.2f}", prefix: str = "", suffix: str = "") -> str:
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
                rendered = _format_kpi_tile_value(value, fmt=fmt_pattern, prefix=prefix, suffix=suffix)
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
                f"post-filter: {_format_value(with_bars_post)}" if with_bars_post not in (None, "") else None,
            ),
            (
                "Bars Rows (total)",
                bars_rows_value,
                f"post-filter: {_format_value(bars_rows_post)}" if bars_rows_post not in (None, "") else None,
            ),
            (
                "Candidates (final)",
                final_candidates_value,
                f"pre-metrics: {_format_value(pre_candidates)}" if pre_candidates not in (None, "") else None,
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
            dbc.Badge(f"429: {_safe_int(metrics_data.get('rate_limited'))}", className="badge-small me-1"),
            dbc.Badge(f"404: {_safe_int(metrics_data.get('http_404_batches'))}", className="badge-small me-1"),
            dbc.Badge(f"Empty: {_safe_int(metrics_data.get('http_empty_batches'))}", className="badge-small"),
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
            dbc.Badge(f"Cache hits: {_safe_int(metrics_data.get('cache_hits'))}", className="badge-small me-1"),
            dbc.Badge(f"Parsed rows: {_safe_int(metrics_data.get('parsed_rows_count'))}", className="badge-small"),
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
            gate_fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), height=320)

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
            coverage_fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), height=320)

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
                {"name": col.replace("_", " ").title(), "id": col}
                for col in display_df.columns
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
                {"name": "Volume", "id": "volume", "type": "numeric", "format": {"specifier": ".0f"}},
                {"name": "Universe Count", "id": "universe_count"},
                {"name": "Entry Price", "id": "entry_price", "type": "numeric", "format": {"specifier": ".2f"}},
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
                                dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "90px"}),
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
        screener_lines, screener_label = _tail_with_timestamp(
            screener_log_path, limit=120
        )
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
            badge_children.extend(
                [html.Span(trade_label, className="me-2"), html.Span(data_label)]
            )
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
        table_updated_display = (
            _format_iso_display(table_updated) if table_updated else "unknown"
        )
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
            components.extend([html.Hr(), html.H4("Diagnostics", className="text-light"), feature_section])
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
                    avg_efficiency = (
                        efficiency_df.groupby("exit_reason")["exit_efficiency"].mean()
                    )
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
                                "format": Format(precision=2, scheme=Format.Scheme.percentage),
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
                    col_def["format"] = Format(
                        precision=2, scheme=Format.Scheme.percentage
                    )
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
                summary_children.append(
                    dbc.Row(exit_reason_cards, className="g-3 mb-3")
                )
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
                                html.H3(f"{metrics_data.get('trailing_attached', 0)}", className="card-text"),
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
                                html.H3(f"{metrics_data.get('api_retries', 0)}", className="card-text"),
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

        last_updated_path = trade_source_path if trade_source_path else (
            executed_trades_path if executed_exists else trades_log_path
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
        paper_mode = _is_paper_mode()
        components: list[Any] = []
        alerts: list[Any] = []

        trades_df: pd.DataFrame | None = None
        if not paper_mode:
            trades_df, trade_alert = load_csv(
                trades_log_real_path,
                ["symbol", "entry_price", "exit_price", "qty", "net_pnl"],
                alert_prefix="Real trades",
            )
            if trade_alert:
                alerts.append(trade_alert)

        equity_df, equity_alert = load_account_equity()
        if equity_alert:
            alerts.append(equity_alert)

        if alerts:
            components.extend(alerts)

        latest_cash = None
        latest_bp = None
        if equity_df is not None and not equity_df.empty:
            if "timestamp" not in equity_df.columns and "date" in equity_df.columns:
                equity_df["timestamp"] = equity_df["date"]
            if "cash" in equity_df.columns:
                latest_cash = pd.to_numeric(equity_df["cash"], errors="coerce").dropna()
                latest_cash = latest_cash.iloc[-1] if not latest_cash.empty else None
            if "buying_power" in equity_df.columns:
                latest_bp = pd.to_numeric(
                    equity_df["buying_power"], errors="coerce"
                ).dropna()
                latest_bp = latest_bp.iloc[-1] if not latest_bp.empty else None
            equity_fig = px.line(
                equity_df,
                x="timestamp",
                y="equity",
                title="Account Equity Over Time",
                template="plotly_dark",
            )
            last_updated = format_time(get_file_mtime(account_equity_path))
            graphs: list[dbc.Col] = [dbc.Col(dcc.Graph(figure=equity_fig), md=6)]
        else:
            graphs = []
            last_updated = None

        if latest_cash is not None or latest_bp is not None:
            kpi_cards = []
            if latest_cash is not None:
                kpi_cards.append(
                    dbc.Card(
                        [
                            dbc.CardHeader("Cash"),
                            dbc.CardBody(f"${latest_cash:,.2f}"),
                        ],
                        className="mb-3",
                    )
                )
            if latest_bp is not None:
                kpi_cards.append(
                    dbc.Card(
                        [
                            dbc.CardHeader("Buying Power"),
                            dbc.CardBody(f"${latest_bp:,.2f}"),
                        ],
                        className="mb-3",
                    )
                )
            graphs.insert(0, dbc.Col(kpi_cards, md=3))

        monthly_fig = None
        top_trades_table = None
        if not paper_mode and trades_df is not None and not trades_df.empty:
            exit_times = None
            if "exit_time" in trades_df.columns:
                exit_times = pd.to_datetime(trades_df["exit_time"], errors="coerce")
                trades_df = trades_df.assign(exit_time=exit_times)
            trades_df["pct_profit"] = (
                trades_df["net_pnl"]
                / (trades_df["entry_price"] * trades_df["qty"])
            ) * 100
            trades_df["pct_profit"].replace([np.inf, -np.inf], 0, inplace=True)
            top_trades = trades_df[trades_df["net_pnl"] > 0].nlargest(
                10, "pct_profit"
            )

            if exit_times is not None:
                monthly_series = trades_df.dropna(subset=["exit_time"]).copy()
                if not monthly_series.empty:
                    monthly_series["month"] = (
                        monthly_series["exit_time"].dt.to_period("M").astype(str)
                    )
                    monthly_pnl = (
                        monthly_series.groupby("month")["net_pnl"].sum().reset_index()
                    )
                    if not monthly_pnl.empty:
                        monthly_fig = px.bar(
                            monthly_pnl,
                            x="month",
                            y="net_pnl",
                            title="Monthly Profit/Loss",
                            template="plotly_dark",
                        )

            top_trades_table = dash_table.DataTable(
                data=top_trades.to_dict("records"),
                columns=[
                    {"name": i.title(), "id": i}
                    for i in [
                        "symbol",
                        "entry_price",
                        "exit_price",
                        "pct_profit",
                        "net_pnl",
                    ]
                ],
                style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
                style_data_conditional=[
                    {"if": {"filter_query": "{net_pnl} > 0"}, "color": "green"},
                    {"if": {"filter_query": "{net_pnl} < 0"}, "color": "red"},
                ],
            )

        if last_updated:
            components.append(html.Div(f"Last Updated: {last_updated}"))
        if graphs:
            if monthly_fig is not None:
                graphs.append(dbc.Col(dcc.Graph(figure=monthly_fig), md=6))
            components.append(dbc.Row(graphs))

        if paper_mode:
            components.append(
                dbc.Alert(
                    "Paper mode: real-trades feed is intentionally disabled.",
                    color="info",
                    className="mb-3",
                )
            )
        elif top_trades_table is not None:
            components.append(dbc.Row([dbc.Col(top_trades_table, width=12)]))

        return dbc.Container(components, fluid=True)

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


@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs", "active_tab"), Input("refresh-ts", "data"), Input("predictions-dropdown", "value")],
)
def render_tab(tab, refresh_ts=None, prediction_path=None):
    if tab == "tab-overview":
        logger.info("Rendering content for tab: %s", tab)
        return overview_layout()
    if tab == "tab-pipeline":
        logger.info("Rendering content for tab: %s", tab)
        return pipeline_layout()
    if tab == "tab-ml":
        logger.info("Rendering content for tab: %s", tab)
        return ml_layout(prediction_path)
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
    elif tab == "tab-screener":
        logger.info("Rendering content for tab: %s", tab)
        return screener_layout()
    else:
        return dbc.Alert("Tab not found.", color="danger")


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
                go.Scatter(x=df["timestamp"], y=df["SMA9"], name="SMA9", line=dict(color="#feca57")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["EMA20"], name="EMA20", line=dict(color="#54a0ff")),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["SMA180"], name="SMA180", line=dict(color="#ff6b6b")),
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
                go.Bar(x=df["timestamp"], y=df["MACD_HIST"], name="MACD Hist", marker_color="#10ac84"),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["MACD_LINE"], name="MACD", line=dict(color="#ff9f43")),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#54a0ff")),
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
    candidate = path or (executed_trades_path if os.path.exists(executed_trades_path) else trades_log_path)
    if candidate and os.path.exists(candidate):
        return dcc.send_file(candidate)
    return dash.no_update



if __name__ == "__main__":
    app.run(debug=False)
