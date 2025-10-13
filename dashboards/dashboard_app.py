# Complete Integrated Dashboard (dashboard_app.py)

import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from datetime import datetime, timezone, timedelta
import subprocess
import json
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import logging
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import pytz
from pathlib import Path
from typing import Optional

os.environ.setdefault("JBRAVO_HOME", "/home/oai/jbravo_screener")

from dashboards.screener_health import build_layout as build_screener_health
from dashboards.screener_health import register_callbacks as register_screener_health

# Base directory of the project (parent of this file)
BASE_DIR = os.environ.get(
    "JBRAVO_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# Absolute paths to CSV data files used throughout the dashboard
trades_log_path = os.path.join(BASE_DIR, "data", "trades_log.csv")
trades_log_real_path = os.path.join(BASE_DIR, "data", "trades_log_real.csv")
open_positions_path = os.path.join(BASE_DIR, "data", "open_positions.csv")
top_candidates_path = os.path.join(BASE_DIR, "data", "top_candidates.csv")
scored_candidates_path = os.path.join(BASE_DIR, "data", "scored_candidates.csv")
screener_metrics_path = os.path.join(BASE_DIR, "data", "screener_metrics.json")
predictions_dir_path = os.path.join(BASE_DIR, "data", "predictions")

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

# Displayed configuration values
MAX_OPEN_TRADES = 10


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


def is_log_stale(log_path):
    """Return True if the most recent INFO/ERROR entry in ``log_path`` is older than 24 hours."""
    if not os.path.exists(log_path):
        return True
    try:
        with open(log_path) as file:
            lines = file.readlines()
        info_tokens = ("[INFO]", "- INFO -")
        error_tokens = ("[ERROR]", "- ERROR -")
        for line in reversed(lines):
            is_info = any(token in line for token in info_tokens)
            is_error = any(token in line for token in error_tokens)
            if is_info or is_error:
                ts_str = line[:19]
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                return (datetime.utcnow() - ts).total_seconds() > (STALE_THRESHOLD_MINUTES * 60)
    except Exception:
        return True
    return True

load_dotenv(os.path.join(BASE_DIR, ".env"))
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
if not API_KEY or not API_SECRET:
    raise ValueError("Missing Alpaca credentials")
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
logger = logging.getLogger(__name__)


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
        return None, dbc.Alert(f"{prefix}File {csv_path} does not exist.", color="danger")
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, dbc.Alert(f"{prefix}No data available in {csv_path}.", color="warning")
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return None, dbc.Alert(
            f"{prefix}Missing required columns: {missing_cols}",
            color="danger",
        )
    return df, None


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


def log_box(title: str, lines: list[str], element_id: str) -> html.Div:
    """Return a styled log display box."""
    return html.Div(
        [
            html.H5(title, className="text-light"),
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
                # Consolidated into Monitoring Positions tab
                # dbc.Tab(
                #     label="Open Positions",
                #     tab_id="tab-positions",
                #     tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                #     active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                #     className="custom-tab",
                # ),
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
        overview_sources = [
            ("Real trades", trades_log_real_path),
            ("Executed trades", executed_trades_path),
        ]
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
            total_trades = latest_metrics["total_trades"]
            total_pnl = latest_metrics["net_pnl"]
            win_rate_val = latest_metrics["win_rate"]
            expectancy = latest_metrics["expectancy"]
            profit_factor_val = latest_metrics["profit_factor"]
            max_drawdown_val = latest_metrics["max_drawdown"]
        else:
            total_trades = len(trades_df)
            total_pnl = trades_df["pnl"].sum()
            win_rate_val = (trades_df["pnl"] > 0).mean() * 100
            expectancy = trades_df["pnl"].mean()
            profit_factor_val = (
                trades_df[trades_df["pnl"] > 0]["pnl"].sum()
                / abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
                if not trades_df[trades_df["pnl"] < 0].empty
                else 0
            )
            max_drawdown_val = trades_df["drawdown"].min() if "drawdown" in trades_df.columns else 0

        kpis = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Total Trades"),
                            dbc.CardBody(html.H4(f"{total_trades}")),
                        ]
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Total Net PnL"),
                            dbc.CardBody(html.H4(f"${total_pnl:.2f}")),
                        ]
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Win Rate"),
                            dbc.CardBody(html.H4(f"{win_rate_val:.2f}%")),
                        ]
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Expectancy"),
                            dbc.CardBody(html.H4(f"${expectancy:.2f}")),
                        ]
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Profit Factor"),
                            dbc.CardBody(html.H4(f"{profit_factor_val:.2f}")),
                        ]
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Max Drawdown"),
                            dbc.CardBody(html.H4(f"${max_drawdown_val:.2f}")),
                        ]
                    ),
                    width=2,
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
        metrics_alert = None
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

        df, alert = load_csv(top_candidates_path)
        scored_df, scored_alert = load_csv(scored_candidates_path)

        alerts = [a for a in (metrics_alert, alert, scored_alert) if a]

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

        last_run_display = metrics_data.get("last_run_utc")
        if isinstance(last_run_display, str):
            try:
                parsed = datetime.fromisoformat(last_run_display.replace("Z", "+00:00"))
                last_run_display = parsed.strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                pass

        health_items = [
            ("Last Run (UTC)", last_run_display),
            ("Symbols In", metrics_data.get("symbols_in")),
            ("Symbols With Bars", metrics_data.get("symbols_with_bars")),
            ("Bars Rows", metrics_data.get("bars_rows_total")),
            ("Candidates", metrics_data.get("candidates_out")),
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
            "No scored candidates available.", color="warning", className="m-2"
        )
        if scored_df is not None and not scored_df.empty:
            scored_work = scored_df.copy()
            scored_work["Score"] = pd.to_numeric(scored_work.get("Score"), errors="coerce")
            scored_work["symbol"] = scored_work.get("symbol", "").astype(str)
            scored_work["close"] = pd.to_numeric(scored_work.get("close"), errors="coerce")
            scored_work["ADV20"] = pd.to_numeric(scored_work.get("ADV20"), errors="coerce")
            scored_work["ATR14"] = pd.to_numeric(scored_work.get("ATR14"), errors="coerce")
            scored_work.sort_values("Score", ascending=False, inplace=True)
            atr_pct = (scored_work["ATR14"] / scored_work["close"]) * 100.0
            atr_pct = atr_pct.replace([np.inf, -np.inf], np.nan)
            breakdown_series = None
            if "score_breakdown" in scored_work.columns:
                breakdown_series = scored_work["score_breakdown"]
            elif "score_breakdown_json" in scored_work.columns:
                breakdown_series = scored_work["score_breakdown_json"]
            if breakdown_series is None:
                breakdown_series = pd.Series([None] * len(scored_work))
            scored_work["Why"] = breakdown_series.fillna("").apply(explain_breakdown)
            scored_work["ATR%"] = atr_pct
            table_df = scored_work.loc[
                :, ["symbol", "Score", "close", "ADV20", "ATR%", "Why"]
            ].head(15)
            table_df.rename(
                columns={"symbol": "Symbol", "close": "Close", "ADV20": "ADV20"}, inplace=True
            )
            table_df["Score"] = table_df["Score"].round(3)
            table_df["Close"] = table_df["Close"].round(2)
            table_df["ADV20"] = table_df["ADV20"].round(0)
            table_df["ATR%"] = table_df["ATR%"].round(1)
            table_df.replace({np.nan: None}, inplace=True)
            columns = [
                {"name": "Symbol", "id": "Symbol"},
                {"name": "Score", "id": "Score"},
                {"name": "Close", "id": "Close"},
                {"name": "ADV20", "id": "ADV20"},
                {"name": "ATR %", "id": "ATR%"},
                {"name": "Why", "id": "Why"},
            ]
            top_table_component = dash_table.DataTable(
                id="screener-top-table",
                data=table_df.to_dict("records"),
                columns=columns,
                page_size=15,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#1b1e21", "fontWeight": "600"},
                style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
                style_data_conditional=[
                    {
                        "if": {"column_id": "Score"},
                        "color": "#4DB6AC",
                        "fontWeight": "600",
                    }
                ],
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
                    html.Div(
                        [
                            html.H5("Pipeline Log", className="text-light"),
                            html.Pre(
                                format_log_lines(pipeline_lines),
                                style={
                                    "maxHeight": "220px",
                                    "overflowY": "auto",
                                    "backgroundColor": "#272B30",
                                    "color": "#E0E0E0",
                                    "padding": "0.5rem",
                                },
                            ),
                        ]
                    ),
                    md=4,
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H5("Screener Log", className="text-light"),
                            html.Pre(
                                format_log_lines(screener_lines),
                                style={
                                    "maxHeight": "220px",
                                    "overflowY": "auto",
                                    "backgroundColor": "#272B30",
                                    "color": "#E0E0E0",
                                    "padding": "0.5rem",
                                },
                            ),
                        ]
                    ),
                    md=4,
                ),
                dbc.Col(
                    html.Div(
                        [
                            html.H5("Backtest Log", className="text-light"),
                            html.Pre(
                                format_log_lines(backtest_lines),
                                style={
                                    "maxHeight": "220px",
                                    "overflowY": "auto",
                                    "backgroundColor": "#272B30",
                                    "color": "#E0E0E0",
                                    "padding": "0.5rem",
                                },
                            ),
                        ]
                    ),
                    md=4,
                ),
            ],
            className="g-3 mb-4",
        )

        components = []
        components.extend(alerts)
        if metrics_data:
            components.extend([health_cards, info_row, charts_row])
        components.append(html.H4("Top Candidates", className="text-light"))
        components.append(top_table_component)
        if metrics_data:
            components.extend([html.Hr(), html.H4("Diagnostics", className="text-light"), feature_section])
        components.extend([html.Hr(), logs_row])
        return dbc.Container(components, fluid=True)

    elif tab == "tab-execute":
        trades_df, alert = load_csv(executed_trades_path)
        if alert:
            table = alert
        else:
            trades_df.sort_values("entry_time", ascending=False, inplace=True)
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
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{net_pnl} < 0", "column_id": "net_pnl"},
                        "color": "#E57373",
                    },
                    {
                        "if": {"filter_query": "{net_pnl} > 0", "column_id": "net_pnl"},
                        "color": "#4DB6AC",
                    },
                ],
            )

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
        if os.path.exists(execute_metrics_path):
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

        download = html.Div(
            [
                html.Button("Download CSV", id="btn-download-trades"),
                dcc.Download(id="download-executed-trades"),
            ],
            className="mb-2",
        )

        last_updated = format_time(get_file_mtime(executed_trades_path))
        timestamp = html.Div(
            f"Last Updated: {last_updated}",
            className="text-muted mb-2",
        )

        components = [timestamp, download, table, html.Hr()]
        if trade_limit_alert:
            components.append(trade_limit_alert)
        if skipped_warning:
            components.append(skipped_warning)
        if api_error_alert:
            components.append(api_error_alert)
        components.append(metrics_view)
        return dbc.Container(components, fluid=True)


    elif tab == "tab-symbol-performance":
        trades_df, alert = load_csv(
            trades_log_real_path,
            ["symbol", "net_pnl"],
            alert_prefix="Real trades",
        )
        freshness = data_freshness_alert(trades_log_real_path, "Trade log")
        if alert:
            return alert
        if trades_df is None or trades_df.empty:
            return dbc.Alert(
                "No trade data available in trades_log_real.csv.",
                color="warning",
                className="m-2",
            )
        app.logger.info("Loaded %d trades for symbol performance", len(trades_df))

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
        last_updated = format_time(get_file_mtime(trades_log_real_path))
        timestamp = html.Div(
            f"Last Updated: {last_updated}",
            className="text-muted mb-2",
        )
        components = [timestamp]
        if freshness:
            components.append(freshness)
        components.extend([dcc.Graph(figure=symbol_fig), table])
        return dbc.Container(components, fluid=True)

    elif tab == "tab-account":
        trades_df, trade_alert = load_csv(
            trades_log_real_path,
            ["symbol", "entry_price", "exit_price", "qty", "net_pnl"],
            alert_prefix="Real trades",
        )
        equity_df, equity_alert = load_csv(
            "data/account_equity.csv", ["date", "equity"]
        )

        alerts = []
        if trade_alert:
            alerts.append(trade_alert)
        if equity_alert:
            alerts.append(equity_alert)
        if alerts:
            return dbc.Container(alerts, fluid=True)

        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        trades_df["pct_profit"] = (
            trades_df["net_pnl"]
            / (trades_df["entry_price"] * trades_df["qty"])
        ) * 100
        trades_df["pct_profit"].replace([np.inf, -np.inf], 0, inplace=True)
        top_trades = trades_df[trades_df["net_pnl"] > 0].nlargest(10, "pct_profit")

        equity_fig = px.line(
            equity_df,
            x="date",
            y="equity",
            title="Account Equity Over Time",
            template="plotly_dark",
        )

        monthly_pnl = (
            trades_df.groupby(
                pd.to_datetime(trades_df["exit_time"]).dt.strftime("%Y-%m")
            )["net_pnl"]
            .sum()
            .reset_index()
        )
        monthly_fig = px.bar(
            monthly_pnl,
            x="exit_time",
            y="net_pnl",
            title="Monthly Profit/Loss",
            template="plotly_dark",
        )

        top_trades_table = dash_table.DataTable(
            data=top_trades.to_dict("records"),
            columns=[
                {"name": i.title(), "id": i}
                for i in ["symbol", "entry_price", "exit_price", "pct_profit", "net_pnl"]
            ],
            style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
            style_data_conditional=[
                {"if": {"filter_query": "{net_pnl} > 0"}, "color": "green"},
                {"if": {"filter_query": "{net_pnl} < 0"}, "color": "red"},
            ],
        )

        last_updated = format_time(get_file_mtime("data/account_equity.csv"))
        return dbc.Container(
            [
                html.Div(f"Last Updated: {last_updated}"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=equity_fig), md=6),
                    dbc.Col(dcc.Graph(figure=monthly_fig), md=6),
                ]),
                dbc.Row([dbc.Col(top_trades_table, width=12)]),
            ],
            fluid=True,
        )

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

        monitor_log_box = log_box("Monitor Log", monitor_lines, "monitor-log")
        exec_log_box = log_box("Execution Log", exec_lines, "exec-log")
        error_log_box = log_box("Errors", error_lines, "error-log")

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
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name=symbol,
                )
            )
            fig.update_layout(template="plotly_dark", title=f"{symbol} Price")
            return True, dcc.Graph(figure=fig)
        return True, html.Div(f"No data for {symbol}")
    if close_click and is_open:
        return False, ""
    return is_open, ""


# Periodically refresh screener table
@app.callback(Output("screener-table", "data"), Input("interval-update", "n_intervals"))
def update_screener_table(n):
    df, alert = load_csv(top_candidates_path)
    if alert:
        return []
    logger.info(
        "Screener table updated successfully with %d records.", len(df)
    )
    return df.to_dict("records") if df is not None and not df.empty else []


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
    df, alert = load_csv(executed_trades_path)
    if alert or df is None or df.empty:
        return []
    df.sort_values("entry_time", ascending=False, inplace=True)
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
    if n_clicks:
        return dcc.send_file(executed_trades_path)



if __name__ == "__main__":
    app.run(debug=False)
