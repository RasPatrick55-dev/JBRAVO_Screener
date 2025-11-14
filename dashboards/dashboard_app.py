# Complete Integrated Dashboard (dashboard_app.py)

import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
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
from typing import Any, Optional
from flask import jsonify
from plotly.subplots import make_subplots

os.environ.setdefault("JBRAVO_HOME", "/home/oai/jbravo_screener")

from dashboards.screener_health import build_layout as build_screener_health
from dashboards.screener_health import register_callbacks as register_screener_health
from dashboards.data_io import screener_health as load_screener_health, screener_table
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
predictions_dir_path = os.path.join(BASE_DIR, "data", "predictions")
ranker_eval_dir_path = os.path.join(BASE_DIR, "data", "ranker_eval")
latest_candidates_path = os.path.join(BASE_DIR, "data", "latest_candidates.csv")

TOP_CANDIDATES = Path(top_candidates_path)
LATEST_CANDIDATES = Path(latest_candidates_path)
RANKER_EVAL_DIR = Path(ranker_eval_dir_path)

# Additional datasets introduced for monitoring
metrics_summary_path = os.path.join(BASE_DIR, "data", "metrics_summary.csv")
executed_trades_path = os.path.join(BASE_DIR, "data", "executed_trades.csv")
historical_candidates_path = os.path.join(BASE_DIR, "data", "historical_candidates.csv")
execute_metrics_path = os.path.join(BASE_DIR, "data", "execute_metrics.json")
account_equity_path = os.path.join(BASE_DIR, "data", "account_equity.csv")
health_connectivity_path = os.path.join(BASE_DIR, "data", "health", "connectivity.json")

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
    """Return True if the most recent INFO/ERROR entry is older than ``max_age_hours``."""

    try:
        log_path = Path(path)
    except TypeError:
        return True

    try:
        last_ts = None
        with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not _line_contains_level(line):
                    continue
                candidate = _parse_log_timestamp(line)
                if candidate:
                    last_ts = candidate
        if not last_ts:
            mtime = datetime.utcfromtimestamp(os.path.getmtime(log_path))
            return (_utcnow() - mtime).total_seconds() > max_age_hours * 3600
        return (_utcnow() - last_ts).total_seconds() > max_age_hours * 3600
    except FileNotFoundError:
        return True
    except Exception:
        return True

load_dotenv(os.path.join(BASE_DIR, ".env"))
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
if not API_KEY or not API_SECRET:
    raise ValueError("Missing Alpaca credentials")
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


def fetch_positions_api():
    """Fetch open positions from Alpaca for fallback."""
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


def _latest_prediction_snapshot() -> tuple[Path, pd.DataFrame] | None:
    directory = Path(predictions_dir_path)
    if not directory.exists():
        return None
    candidates: list[tuple[datetime, Path]] = []
    for path in directory.glob("*.csv"):
        if path.name == "latest.csv":
            continue
        try:
            snapshot_date = datetime.strptime(path.stem, "%Y-%m-%d")
        except ValueError:
            snapshot_date = datetime.utcfromtimestamp(path.stat().st_mtime)
        candidates.append((snapshot_date, path))
    if not candidates:
        return None
    _, latest_path = max(candidates, key=lambda item: item[0])
    try:
        df = pd.read_csv(latest_path)
    except Exception:
        return None
    return latest_path, df


def _read_ranker_eval_summary() -> tuple[Path, dict] | None:
    summary_path = RANKER_EVAL_DIR / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return summary_path, payload


def _load_deciles(as_of: str) -> pd.DataFrame | None:
    decile_path = RANKER_EVAL_DIR / f"deciles_{as_of}.csv"
    if not decile_path.exists():
        return None
    try:
        return pd.read_csv(decile_path)
    except Exception:
        return None


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


def log_box(title: str, lines: list[str], element_id: str, log_path: str | None = None) -> html.Div:
    """Return a styled log display box."""
    header_children: list[Any] = [html.Span(title, className="text-light")]
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
                html.Div("Ops", className="text-uppercase small text-muted mb-2 fw-bold"),
                dbc.Button(
                    "Pipeline Log",
                    id="ops-btn-pipeline",
                    color="secondary",
                    size="sm",
                    className="mb-2",
                ),
                dbc.Button(
                    "Execute Log",
                    id="ops-btn-exec",
                    color="secondary",
                    size="sm",
                    className="mb-2",
                ),
                dbc.Button(
                    "Latest Candidates",
                    id="ops-btn-candidates",
                    color="secondary",
                    size="sm",
                    className="mb-2",
                ),
                dbc.Button(
                    "Screener Metrics",
                    id="ops-btn-metrics",
                    color="secondary",
                    size="sm",
                ),
            ],
            id="ops-sidebar",
            className="p-3 bg-dark rounded d-none d-md-block",
            style={
                "position": "fixed",
                "top": "120px",
                "left": "20px",
                "width": "180px",
                "zIndex": 1050,
                "boxShadow": "0 0 10px rgba(0,0,0,0.4)",
            },
        ),
        html.Div(
            [
                dbc.Tabs(
                    id="tabs",
                    active_tab="tab-overview",
                    class_name="mb-3",
                    children=[
                        dbc.Tab(
                            label="Screener Health",
                            tab_id="tab-screener-health",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="Overview",
                            tab_id="tab-overview",
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
                            label="Predictions",
                            tab_id="tab-predictions",
                            tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                            active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                            className="custom-tab",
                        ),
                        dbc.Tab(
                            label="Ranker Eval",
                            tab_id="tab-ranker-eval",
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
                    ],
                ),
                html.Button(
                    "Refresh Now",
                    id="refresh-button",
                    className="btn btn-secondary mb-2",
                ),
                dcc.Loading(
                    id="loading",
                    children=html.Div(id="tabs-content", className="mt-4"),
                    type="default",
                ),
            ],
            style={"marginLeft": "220px"},
        ),
        # Refresh dashboards roughly every 15 seconds to reflect new logs
        dcc.Interval(id="interval-update", interval=15000, n_intervals=0),
        dcc.Interval(id="log-interval", interval=10000, n_intervals=0),
        dcc.Interval(id="interval-trades", interval=30 * 1000, n_intervals=0),
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
        dbc.Modal(
            id="ops-modal",
            is_open=False,
            size="lg",
            children=[
                dbc.ModalHeader(dbc.ModalTitle("Operations Snapshot")),
                dbc.ModalBody(id="ops-modal-body"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="ops-modal-close", className="ms-auto")
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
    if tab == "tab-overview":
        overview_sources: list[tuple[str, str]] = []
        if not _is_paper_mode():
            overview_sources.append(("Real trades", trades_log_real_path))
        overview_sources.append(("Trades log", trades_log_path))
        overview_sources.append(("Executed trades", executed_trades_path))
        alerts = []
        trades_df = None
        source_label = ""
        source_path = ""

        for label, path in overview_sources:
            df, df_alert = load_csv(
                path,
                ["net_pnl", "entry_time"],
                alert_prefix=label,
            )
            if df_alert:
                alerts.append(df_alert)
                continue
            trades_df = df
            source_label = label
            source_path = path
            break

        if trades_df is None:
            info_alert = dbc.Alert(
                "No trade history available yet. Run the execution or monitoring jobs to populate trade logs.",
                color="info",
                className="m-3",
            )
            return dbc.Container([info_alert, *alerts], fluid=True)

        freshness = data_freshness_alert(source_path, f"{source_label} log")
        app.logger.info(
            "Loaded %d trades for overview from %s",
            len(trades_df),
            source_label,
        )

        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["pnl"] = trades_df["net_pnl"]
        trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
        trades_df["cummax"] = trades_df["cumulative_pnl"].cummax()
        trades_df["drawdown"] = trades_df["cumulative_pnl"] - trades_df["cummax"]

        equity_fig = px.line(
            trades_df,
            x="entry_time",
            y="cumulative_pnl",
            template="plotly_dark",
            title="Equity Curve",
        )
        hist_fig = px.histogram(
            trades_df,
            x="pnl",
            nbins=20,
            template="plotly_dark",
            title="Distribution of Trade PnL",
        )
        pie_fig = px.pie(
            trades_df,
            names=trades_df["pnl"] > 0,
            title="Win vs Loss",
            template="plotly_dark",
        )

        daily = (
            trades_df.groupby(trades_df["entry_time"].dt.date)["pnl"]
            .sum()
            .reset_index()
        )
        daily_fig = px.bar(
            daily, x="entry_time", y="pnl", template="plotly_dark", title="Daily PnL"
        )
        monthly = (
            trades_df.groupby(trades_df["entry_time"].dt.to_period("M"))["pnl"]
            .sum()
            .reset_index()
        )
        monthly["entry_time"] = monthly["entry_time"].astype(str)
        monthly_fig = px.bar(
            monthly,
            x="entry_time",
            y="pnl",
            template="plotly_dark",
            title="Monthly PnL",
        )
        dd_fig = px.area(
            trades_df,
            x="entry_time",
            y="drawdown",
            template="plotly_dark",
            title="Drawdown Over Time",
        )

        try:
            metrics_df, alert_metrics = load_csv(
                metrics_summary_path,
                required_columns=[
                    "total_trades",
                    "net_pnl",
                    "win_rate",
                    "expectancy",
                    "profit_factor",
                    "max_drawdown",
                ],
                alert_prefix="Metrics",
            )
            if alert_metrics:
                return alert_metrics
        except Exception as e:
            return dbc.Alert(f"Metrics Load Error: {str(e)}", color="danger", className="mt-3")

        if metrics_df is not None and not metrics_df.empty:
            latest_metrics = metrics_df.iloc[-1]
            total_trades = latest_metrics.get("total_trades", len(trades_df))
            total_pnl = latest_metrics.get("net_pnl", trades_df["pnl"].sum())
            win_rate_val = latest_metrics.get("win_rate", (trades_df["pnl"] > 0).mean() * 100)
            expectancy = latest_metrics.get("expectancy", trades_df["pnl"].mean())
            profit_factor_val = latest_metrics.get("profit_factor")
            max_drawdown_val = latest_metrics.get("max_drawdown")
            sharpe_val = latest_metrics.get("sharpe")
        else:
            total_trades = len(trades_df)
            total_pnl = trades_df["pnl"].sum()
            win_rate_val = (trades_df["pnl"] > 0).mean() * 100
            expectancy = trades_df["pnl"].mean()
            wins = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
            losses = trades_df[trades_df["pnl"] < 0]["pnl"].sum()
            profit_factor_val = wins / abs(losses) if losses else float("nan")
            max_drawdown_val = trades_df["drawdown"].min() if "drawdown" in trades_df.columns else float("nan")
            pnl_std = trades_df["pnl"].std(ddof=0)
            sharpe_val = (expectancy / pnl_std) if pnl_std else float("nan")

        def _overview_card(label: str, value: Any, fmt: str = "{}", tooltip: str | None = None, prefix: str = "") -> dbc.Col:
            card_id = f"overview-{label.lower().replace(' ', '-')}-card"
            try:
                if value in (None, ""):
                    rendered = "n/a"
                elif isinstance(value, float) and math.isnan(value):
                    rendered = "n/a"
                else:
                    rendered = fmt.format(value)
            except Exception:
                rendered = str(value)
            card = dbc.Card(
                [
                    dbc.CardHeader(label, id=f"{card_id}-header"),
                    dbc.CardBody(html.H4(f"{prefix}{rendered}", className="mb-0")),
                ],
                className="h-100",
            )
            if tooltip:
                tooltip_component = dbc.Tooltip(tooltip, target=f"{card_id}-header", placement="top")
                return dbc.Col(html.Div([card, tooltip_component]), md=2, className="mb-3")
            return dbc.Col(card, md=2, className="mb-3")

        kpis = dbc.Row(
            [
                _overview_card("Total Trades", total_trades, "{:.0f}"),
                _overview_card("Total Net PnL", total_pnl, "{:.2f}", prefix="$"),
                _overview_card("Win Rate", win_rate_val, "{:.2f}%"),
                _overview_card(
                    "Expectancy",
                    expectancy,
                    "{:.2f}",
                    tooltip="Average per-trade profit after costs.",
                    prefix="$",
                ),
                _overview_card(
                    "Profit Factor",
                    profit_factor_val,
                    "{:.2f}",
                    tooltip="Gross profits divided by gross losses.",
                ),
                _overview_card(
                    "Max Drawdown",
                    max_drawdown_val,
                    "{:.2f}",
                    tooltip="Largest peak-to-trough equity decline.",
                    prefix="$",
                ),
                _overview_card(
                    "Sharpe",
                    sharpe_val,
                    "{:.2f}",
                    tooltip="Per-trade Sharpe ratio (mean divided by volatility).",
                ),
            ],
            className="mb-4",
        )

        graphs = dbc.Row(
            [
                dbc.Col(dcc.Graph(figure=equity_fig), md=6),
                dbc.Col(dcc.Graph(figure=hist_fig), md=6),
                dbc.Col(dcc.Graph(figure=pie_fig), md=4),
                dbc.Col(dcc.Graph(figure=daily_fig), md=4),
                dbc.Col(dcc.Graph(figure=monthly_fig), md=4),
                dbc.Col(dcc.Graph(figure=dd_fig), md=12),
            ]
        )
        latest_file = source_path
        latest_mtime = get_file_mtime(latest_file)
        for contender in {trades_log_real_path, executed_trades_path}:
            contender_mtime = get_file_mtime(contender)
            if contender_mtime and (
                latest_mtime is None or contender_mtime > latest_mtime
            ):
                latest_file = contender
                latest_mtime = contender_mtime

        last_updated = format_time(latest_mtime)
        timestamp = html.Div(
            f"Last Updated: {last_updated}",
            className="text-muted mb-2",
        )

        source_note = html.Div(
            f"Overview metrics use {source_label.lower()} data.",
            className="text-muted small mb-2",
        )

        components = [timestamp, source_note]
        if freshness:
            components.append(freshness)
        components.extend(alerts)
        components.extend([kpis, graphs])
        return dbc.Container(components, fluid=True)


    elif tab == "tab-screener":
        metrics_data: dict = {}
        health_snapshot = load_screener_health()
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
            metrics_freshness_chip = _freshness_badge(screener_metrics_path)

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
                    if metrics_alert is None:
                        metrics_alert = dbc.Alert(
                            f"Unable to backfill screener metrics: {exc}",
                            color="danger",
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
            for field, source_key in (
                ("symbols_in", "symbols_in"),
                ("symbols_with_bars", "symbols_with_bars"),
                ("bars_rows_total", "bars_rows_total"),
            ):
                value = health_snapshot.get(source_key)
                if value not in (None, ""):
                    metrics_data[field] = value
            rows_pre = health_snapshot.get("rows_premetrics")
            if rows_pre not in (None, ""):
                metrics_data["rows"] = rows_pre
            metrics_data["rows_final"] = health_snapshot.get("rows_final")
            if not metrics_data.get("last_run_utc"):
                metrics_data["last_run_utc"] = health_snapshot.get("last_run_utc")

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

        last_run_display = _format_iso_display(metrics_data.get("last_run_utc"))

        pre_candidates = metrics_data.get("rows") or metrics_data.get("candidates_out")
        health_items = [
            ("Last Run (UTC)", last_run_display),
            ("Symbols In", metrics_data.get("symbols_in")),
            ("Symbols With Bars", metrics_data.get("symbols_with_bars")),
            ("Bars Rows", metrics_data.get("bars_rows_total")),
            ("Candidates (pre-metrics)", pre_candidates),
            ("Candidates (final)", metrics_data.get("rows_final")),
        ]
        health_columns = []
        for idx, (label, value) in enumerate(health_items):
            card = dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(label, className="card-metric-label"),
                        html.Div(_format_value(value), className="card-metric-value"),
                    ]
                ),
                className="bg-dark text-light h-100",
            )
            width = 4 if idx == 0 else 2
            health_columns.append(dbc.Col(card, md=width, sm=6))
        health_cards = dbc.Row(health_columns, className="g-3 mb-4")

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

        pipeline_lines = read_recent_lines(pipeline_log_path)[::-1]
        screener_lines = read_recent_lines(screener_log_path, num_lines=20)[::-1]
        backtest_lines = read_recent_lines(backtest_log_path)[::-1]

        logs_row = dbc.Row(
            [
                dbc.Col(
                    log_box(
                        "Pipeline Log",
                        pipeline_lines,
                        "pipeline-log",
                        log_path=pipeline_log_path,
                    ),
                    md=4,
                ),
                dbc.Col(
                    log_box(
                        "Screener Log",
                        screener_lines,
                        "screener-log",
                        log_path=screener_log_path,
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

        connectivity_chip = None
        if os.path.exists(health_connectivity_path):
            try:
                with open(health_connectivity_path, "r", encoding="utf-8") as handle:
                    connectivity_payload = json.load(handle) or {}
                trading = connectivity_payload.get("trading", {}) if isinstance(connectivity_payload, dict) else {}
                data_status = connectivity_payload.get("data", {}) if isinstance(connectivity_payload, dict) else {}
                trading_ok = bool(trading.get("ok"))
                data_ok = bool(data_status.get("ok"))
                status_color = "success" if trading_ok and data_ok else ("warning" if trading_ok or data_ok else "danger")
                connectivity_chip = dbc.Badge(
                    [
                        html.Span("Alpaca Connectivity", className="me-2"),
                        html.Span(f"Trading {'✅' if trading_ok else '❌'}", className="me-2"),
                        html.Span(f"Data {'✅' if data_ok else '❌'}"),
                    ],
                    color=status_color,
                    className="me-2",
                )
            except Exception:
                connectivity_chip = None

        components = [html.Div(_paper_badge_component(), className="mb-2")]
        if connectivity_chip:
            components.append(html.Div(connectivity_chip, className="mb-3"))
        if latest_notice:
            components.append(latest_notice)
        if metrics_freshness_chip:
            components.append(html.Div(metrics_freshness_chip, className="mb-3"))
        if backfill_banner:
            components.append(backfill_banner)
        components.extend(alerts)

        source_label = str((health_snapshot or {}).get("source") or "unknown")
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
        pipeline_rc = (health_snapshot or {}).get("pipeline_rc")
        rc_text = f"rc={pipeline_rc}" if pipeline_rc is not None else "rc=n/a"
        rc_color = "success" if pipeline_rc == 0 else ("danger" if pipeline_rc else "secondary")
        rc_badge = dbc.Badge(rc_text, color=rc_color, className="me-2")
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
                [source_badge, rc_badge, table_source_note, table_updated_note],
                className="mb-3",
            )
        )

        metrics_sections = []
        if metrics_data:
            metrics_sections.append(health_cards)
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
        if metrics_data:
            components.extend([html.Hr(), html.H4("Diagnostics", className="text-light"), feature_section])
        components.extend([html.Hr(), logs_row])
        return dbc.Container(components, fluid=True)

    elif tab == "tab-execute":
        trades_df, trade_source_path, trade_source_label, trade_alerts = _resolve_trades_dataframe()
        executed_exists = os.path.exists(executed_trades_path)

        if trades_df is not None:
            sort_column = "entry_time" if "entry_time" in trades_df.columns else None
            if sort_column:
                trades_df.sort_values(sort_column, ascending=False, inplace=True)
            table = dash_table.DataTable(
                id="executed-trades-table",
                columns=[
                    {"name": col.replace("_", " ").title(), "id": col}
                    for col in trades_df.columns
                ],
                data=trades_df.to_dict("records"),
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
        else:
            hint = dbc.Alert(
                "No trades yet (paper).",
                color="info",
                className="mb-2",
            )
            table = html.Div([hint, *trade_alerts]) if trade_alerts else hint

        metrics_defaults = {
            "last_run_utc": "",
            "symbols_in": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_canceled": 0,
            "trailing_attached": 0,
            "api_retries": 0,
            "api_failures": 0,
            "latency_secs": {"p50": 0.0, "p95": 0.0},
        }
        metrics_data = metrics_defaults.copy()
        metrics_file_exists = os.path.exists(execute_metrics_path)
        if metrics_file_exists:
            try:
                with open(execute_metrics_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    for key, default_value in metrics_defaults.items():
                        if key == "latency_secs":
                            latency = loaded.get("latency_secs", {}) or {}
                            metrics_data["latency_secs"] = {
                                "p50": latency.get("p50", 0.0),
                                "p95": latency.get("p95", 0.0),
                            }
                        else:
                            metrics_data[key] = loaded.get(key, default_value)
            except Exception as exc:  # pragma: no cover - log file errors
                logger.error("Failed to load execution metrics: %s", exc)

        api_error_alert = (
            html.Div(
                f"Recent API Errors: {metrics_data['api_failures']}",
                style={"color": "orange"},
            )
            if metrics_data["api_failures"] > 0
            else None
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
        if download is not None:
            components.append(download)
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
        components.append(metrics_view)
        return dbc.Container(components, fluid=True)


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

        equity_df, equity_alert = load_csv(
            account_equity_path,
            ["timestamp", "equity"],
            alert_prefix="Account equity",
        )
        if equity_alert:
            alerts.append(equity_alert)

        if alerts:
            components.extend(alerts)

        if equity_df is not None:
            if "timestamp" not in equity_df.columns and "date" in equity_df.columns:
                equity_df["timestamp"] = equity_df["date"]
            equity_df["timestamp"] = pd.to_datetime(
                equity_df["timestamp"], utc=True, errors="coerce"
            )
            equity_df = equity_df.dropna(subset=["timestamp"])
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


def overview_layout():
    return _render_tab("tab-overview", 0, 0, None)


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


def predictions_layout():
    snapshot = _latest_prediction_snapshot()
    if snapshot is None:
        return dbc.Alert("Prediction snapshot not available yet.", color="secondary")

    path, df = snapshot
    if df.empty:
        return dbc.Alert(
            f"{path.name} does not contain any rows yet.",
            color="secondary",
        )

    display_df = df.copy()
    display_df.columns = [str(col) for col in display_df.columns]

    def _find_column(candidates: set[str]) -> Optional[str]:
        for col in display_df.columns:
            if str(col).strip().lower() in candidates:
                return col
        return None

    score_col = _find_column({"score", "total_score"})
    relvol_col = _find_column({"rel_volume", "relvol", "rel_vol"})
    atr_col = _find_column({"atr_percent", "atrp", "atr_pct"})
    breakdown_col = _find_column({"score_breakdown", "score_breakdown_json", "breakdown"})

    work = pd.DataFrame()
    work["symbol"] = display_df.get("symbol", pd.Series(dtype="string"))
    if score_col:
        work["score"] = pd.to_numeric(display_df[score_col], errors="coerce")
    else:
        work["score"] = pd.Series(dtype="float64")
    if relvol_col:
        work["rel_volume"] = pd.to_numeric(display_df[relvol_col], errors="coerce")
    else:
        work["rel_volume"] = pd.Series(dtype="float64")
    if atr_col:
        work["atr_percent"] = pd.to_numeric(display_df[atr_col], errors="coerce")
    else:
        work["atr_percent"] = pd.Series(dtype="float64")
    if breakdown_col:
        work["breakdown_raw"] = display_df[breakdown_col]
    else:
        work["breakdown_raw"] = "{}"

    work.fillna({"rel_volume": 0.0, "atr_percent": 0.0}, inplace=True)
    work["breakdown_html"] = work["breakdown_raw"].apply(_score_breakdown_badges)

    store_payload = work.to_dict("records")

    updated = format_time(get_file_mtime(str(path)))
    meta = dbc.Row(
        [
            dbc.Col(html.Div(f"Snapshot: {path.name}"), md=6),
            dbc.Col(html.Div(f"Last updated: {updated}"), md=6),
        ],
        className="mb-3 text-light",
    )

    controls = dbc.Row(
        [
            dbc.Col(
                dbc.Input(
                    id="predictions-search",
                    placeholder="Search symbol",
                    type="text",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Input(
                    id="predictions-min-score",
                    placeholder="Min score",
                    type="number",
                    step="0.05",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Input(
                    id="predictions-min-relvol",
                    placeholder="Min rel-vol",
                    type="number",
                    step="0.1",
                ),
                md=3,
            ),
            dbc.Col(
                dbc.Input(
                    id="predictions-max-atrp",
                    placeholder="Max ATR %",
                    type="number",
                    step="0.1",
                ),
                md=3,
            ),
        ],
        className="g-2 mb-3",
    )

    table = dash_table.DataTable(
        id="predictions-table",
        columns=[
            {"name": "Symbol", "id": "symbol"},
            {"name": "Score", "id": "score", "type": "numeric", "format": {"specifier": ".4f"}},
            {"name": "Rel Vol", "id": "rel_volume", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "ATR %", "id": "atr_percent", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "Score Breakdown", "id": "breakdown_html"},
        ],
        data=[],
        sort_action="native",
        page_size=25,
        style_table={"overflowX": "auto", "maxHeight": "520px", "overflowY": "auto"},
        style_cell={"backgroundColor": "#212529", "color": "#E0E0E0", "fontSize": "0.85rem"},
        style_header={"backgroundColor": "#343a40", "color": "#FFFFFF", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"column_id": "score"}, "fontWeight": "bold"},
        ],
        dangerously_allow_html=True,
    )

    return html.Div(
        [
            meta,
            controls,
            dcc.Store(id="predictions-store", data=store_payload),
            table,
        ]
    )


def ranker_eval_layout():
    summary_payload = _read_ranker_eval_summary()
    if summary_payload is None:
        return dbc.Alert("Ranker evaluation artefacts are not available yet.", color="secondary")

    summary_path, summary = summary_payload
    metrics = summary.get("metrics", {}) or {}
    as_of = summary.get("as_of", "?")

    def _ensure_numeric(value: Any) -> float:
        if isinstance(value, (int, float)) and not pd.isna(value):
            return float(value)
        if isinstance(value, (int, float)) and math.isinf(value):
            return float(value)
        return float("nan")

    def _format_value(value: float, fmt: str, *, infinite_label: str = "∞") -> str:
        if math.isnan(value):
            return "n/a"
        if math.isinf(value):
            return infinite_label if value > 0 else f"-{infinite_label}"
        return fmt.format(value)

    def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
        renamed = {col: str(col).strip().lower() for col in frame.columns}
        return frame.rename(columns=renamed)

    deciles_df = _load_deciles(as_of)
    if deciles_df is None:
        latest_deciles = RANKER_EVAL_DIR / "latest_deciles.csv"
        if latest_deciles.exists():
            try:
                deciles_df = pd.read_csv(latest_deciles)
            except Exception:
                deciles_df = None
    decile_chart = None
    if deciles_df is not None and not deciles_df.empty:
        deciles_df = _normalise_columns(deciles_df)
        if "decile" in deciles_df.columns:
            working = deciles_df.copy()
            working.sort_values("decile", inplace=True)
            if "avg_return" in working.columns:
                working["avg_return"] = working["avg_return"].astype(float)
            decile_chart = px.bar(
                working,
                x=working["decile"].astype(str),
                y="avg_return",
                title="Decile Next-Day Returns",
                template="plotly_dark",
            )
            decile_chart.update_yaxes(tickformat=".2%")
            decile_chart.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    calibration_records = summary.get("calibration", []) or []
    calibration_chart = None
    if calibration_records:
        calibration_df = _normalise_columns(pd.DataFrame(calibration_records))
        if {"avg_score", "hit_rate"}.issubset(calibration_df.columns):
            calibration_df.sort_values("avg_score", inplace=True)
            calibration_chart = px.line(
                calibration_df,
                x="avg_score",
                y="hit_rate",
                markers=True,
                title="Score Calibration",
                template="plotly_dark",
            )
            calibration_chart.update_yaxes(tickformat=".2%")

    history_records = summary.get("history") or summary.get("sessions") or []
    if isinstance(history_records, dict):
        history_records = history_records.get("data") or list(history_records.values())
    history_df = pd.DataFrame(history_records)
    if history_df.empty:
        candidate_row = {k: metrics.get(k) for k in ("expectancy", "profit_factor", "sharpe")}
        candidate_row["as_of"] = as_of
        history_df = pd.DataFrame([candidate_row])
    history_chart = None
    if not history_df.empty:
        history_df = _normalise_columns(history_df)
        if "as_of" in history_df.columns:
            history_df["as_of"] = pd.to_datetime(history_df["as_of"], errors="coerce")
            history_df = history_df.dropna(subset=["as_of"])
        history_df = history_df.tail(20)
        value_columns = [col for col in ("expectancy", "profit_factor", "sharpe") if col in history_df.columns]
        if value_columns and not history_df.empty:
            melted = history_df.melt(id_vars=[col for col in ("as_of", "session") if col in history_df.columns], value_vars=value_columns, var_name="metric", value_name="value")
            x_axis = "as_of" if "as_of" in history_df.columns else "session"
            history_chart = px.line(
                melted,
                x=x_axis,
                y="value",
                color="metric",
                title="Trailing KPI Trend (last 20 sessions)",
                template="plotly_dark",
            )

    def _metric_card(title: str, value: Any, tooltip: str, fmt: str = "{}") -> dbc.Col:
        number = _ensure_numeric(value)
        rendered = _format_value(number, fmt)
        card_id = f"ranker-kpi-{title.lower().replace(' ', '-') }"
        card = dbc.Card(
            [
                dbc.CardHeader(title, id=f"{card_id}-header"),
                dbc.CardBody(html.H4(rendered, className="mb-0")),
            ]
        )
        tooltip_component = dbc.Tooltip(tooltip, target=f"{card_id}-header", placement="top")
        return dbc.Col(html.Div([card, tooltip_component]), md=3, className="mb-3")

    cards = dbc.Row(
        [
            _metric_card("Expectancy", metrics.get("expectancy"), "Average next-day return per trade.", fmt="{:.4f}"),
            _metric_card("Profit Factor", metrics.get("profit_factor"), "Gross wins divided by gross losses.", fmt="{:.2f}"),
            _metric_card("Sharpe", metrics.get("sharpe"), "Annualised Sharpe ratio over the evaluation window.", fmt="{:.2f}"),
            _metric_card("Hit Rate", metrics.get("hit_rate"), "Share of trades ending positive.", fmt="{:.2%}"),
        ]
    )

    secondary_cards = dbc.Row(
        [
            _metric_card("Max Drawdown", metrics.get("max_drawdown"), "Worst peak-to-trough decline during the window.", fmt="{:.2%}"),
            _metric_card("Return Volatility", metrics.get("expectancy_std"), "Standard deviation of realised returns.", fmt="{:.4f}"),
        ]
    )

    updated = format_time(get_file_mtime(str(summary_path)))

    components: list[Any] = [
        html.Div(f"Evaluation as of {as_of}", className="text-light"),
        html.Div(f"Summary generated: {updated}", className="text-light mb-3"),
        cards,
        secondary_cards,
    ]

    if decile_chart is not None:
        components.extend(
            [
                html.H4("Decile Performance", className="text-light"),
                dcc.Graph(figure=decile_chart),
            ]
        )
    else:
        components.append(dbc.Alert("Decile breakdown not available.", color="secondary"))

    if calibration_chart is not None:
        components.extend(
            [
                html.Hr(),
                html.H4("Calibration", className="text-light"),
                dcc.Graph(figure=calibration_chart),
            ]
        )
    else:
        components.append(dbc.Alert("Calibration data not available.", color="secondary"))

    if history_chart is not None:
        components.extend(
            [
                html.Hr(),
                html.H4("Summary KPIs (20 sessions)", className="text-light"),
                dcc.Graph(figure=history_chart),
            ]
        )
    else:
        components.append(dbc.Alert("Historical KPI trend not available yet.", color="secondary"))

    return html.Div(components)


@app.callback(Output("tabs-content", "children"), [Input("tabs", "active_tab")])
def render_tab(tab):
    if tab == "tab-screener-health":
        logger.info("Rendering content for tab: %s", tab)
        return build_screener_health()
    if tab == "tab-overview":
        logger.info("Rendering content for tab: %s", tab)
        return overview_layout()
    elif tab == "tab-account":
        logger.info("Rendering content for tab: %s", tab)
        return account_layout()
    elif tab == "tab-symbol-performance":
        logger.info("Rendering content for tab: %s", tab)
        return symbol_performance_layout()
    elif tab == "tab-monitor":
        logger.info("Rendering content for tab: %s", tab)
        return monitor_positions_layout()
    elif tab == "tab-predictions":
        logger.info("Rendering content for tab: %s", tab)
        return predictions_layout()
    elif tab == "tab-ranker-eval":
        logger.info("Rendering content for tab: %s", tab)
        return ranker_eval_layout()
    elif tab == "tab-execute":
        logger.info("Rendering content for tab: %s", tab)
        return execute_trades_layout()
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


@app.callback(
    [Output("ops-modal", "is_open"), Output("ops-modal-body", "children")],
    [
        Input("ops-btn-pipeline", "n_clicks"),
        Input("ops-btn-exec", "n_clicks"),
        Input("ops-btn-candidates", "n_clicks"),
        Input("ops-btn-metrics", "n_clicks"),
        Input("ops-modal-close", "n_clicks"),
    ],
    State("ops-modal", "is_open"),
)
def toggle_ops_modal(pipeline_click, exec_click, candidates_click, metrics_click, close_click, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "ops-modal-close":
        return False, dash.no_update

    def _log_content(path: str, label: str) -> Any:
        lines = tail_log(path, limit=40)
        if not lines:
            return dbc.Alert(f"No recent entries for {label}.", color="warning")
        return html.Pre(
            format_log_lines(lines),
            style={
                "maxHeight": "400px",
                "overflowY": "auto",
                "backgroundColor": "#272B30",
                "color": "#E0E0E0",
                "padding": "0.75rem",
            },
        )

    if trigger == "ops-btn-pipeline":
        content = _log_content(pipeline_log_path, "pipeline.log")
    elif trigger == "ops-btn-exec":
        content = _log_content(execute_trades_log_path, "execute_trades.log")
    elif trigger == "ops-btn-candidates":
        if not os.path.exists(latest_candidates_path):
            content = dbc.Alert("latest_candidates.csv not found.", color="warning")
        else:
            try:
                preview = pd.read_csv(latest_candidates_path, nrows=2)
            except Exception as exc:
                content = dbc.Alert(f"Failed to read latest candidates: {exc}", color="danger")
            else:
                if preview.empty:
                    content = dbc.Alert("File exists but no rows are available yet.", color="info")
                else:
                    preview.columns = [str(c) for c in preview.columns]
                    content = dash_table.DataTable(
                        data=preview.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in preview.columns],
                        style_cell={
                            "backgroundColor": "#212529",
                            "color": "#E0E0E0",
                            "fontSize": "0.85rem",
                        },
                        style_header={"backgroundColor": "#343a40", "color": "#FFFFFF"},
                        page_size=2,
                    )
    elif trigger == "ops-btn-metrics":
        if not os.path.exists(screener_metrics_path):
            content = dbc.Alert("screener_metrics.json not found.", color="warning")
        else:
            try:
                with open(screener_metrics_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle) or {}
            except Exception as exc:
                content = dbc.Alert(f"Failed to load screener metrics: {exc}", color="danger")
            else:
                rows: list[dict[str, Any]] = []
                for key, value in sorted(payload.items()):
                    if isinstance(value, dict):
                        for sub_key, sub_value in sorted(value.items()):
                            rows.append({"Key": f"{key}.{sub_key}", "Value": sub_value})
                    else:
                        rows.append({"Key": key, "Value": value})
                metrics_table = dash_table.DataTable(
                    data=rows,
                    columns=[{"name": c, "id": c} for c in ("Key", "Value")],
                    style_cell={
                        "backgroundColor": "#212529",
                        "color": "#E0E0E0",
                        "fontSize": "0.85rem",
                        "textAlign": "left",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },
                    page_size=10,
                )
                freshness = _freshness_badge(screener_metrics_path)
                content = html.Div(
                    [
                        html.Div(f"Metrics entries: {len(rows)}", className="text-light mb-2"),
                        html.Div(freshness) if freshness else html.Div(),
                        metrics_table,
                    ]
                )
    else:
        content = dbc.Alert("No content available.", color="warning")

    return True, content


# Periodically refresh screener table
@app.callback(Output("screener-table", "data"), Input("interval-update", "n_intervals"))
def update_screener_table(n):
    df, alert = load_top_or_latest_candidates()
    if alert:
        return []
    if df is None or df.empty:
        logger.info("Screener table update skipped; no rows available.")
        return []
    source_note = (
        df["__source"].iloc[0] if "__source" in df.columns else "unknown"
    )
    payload = df.drop(columns=["__source", "__updated"], errors="ignore")
    logger.info(
        "Screener table updated successfully with %d records (source=%s).",
        len(payload),
        source_note,
    )
    return payload.to_dict("records")


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


# Update predictions table based on filters
@app.callback(
    Output("predictions-table", "data"),
    [
        Input("predictions-store", "data"),
        Input("predictions-search", "value"),
        Input("predictions-min-score", "value"),
        Input("predictions-min-relvol", "value"),
        Input("predictions-max-atrp", "value"),
    ],
)
def filter_predictions_table(store, search, min_score, min_relvol, max_atrp):
    if not store:
        return []
    frame = pd.DataFrame(store)
    if frame.empty:
        return []
    work = frame.copy()
    if search:
        mask = work.get("symbol", pd.Series(dtype="string")).astype(str).str.contains(
            str(search), case=False, na=False
        )
        work = work[mask]
    if min_score is not None and "score" in work.columns:
        work = work[pd.to_numeric(work["score"], errors="coerce") >= float(min_score)]
    if min_relvol is not None and "rel_volume" in work.columns:
        work = work[pd.to_numeric(work["rel_volume"], errors="coerce") >= float(min_relvol)]
    if max_atrp is not None and "atr_percent" in work.columns:
        work = work[pd.to_numeric(work["atr_percent"], errors="coerce") <= float(max_atrp)]
    if "score" in work.columns:
        work = work.sort_values("score", ascending=False)
    return work.to_dict("records")


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
