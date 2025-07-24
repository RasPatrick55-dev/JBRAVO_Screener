# Complete Integrated Dashboard (dashboard_app.py)

import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from datetime import datetime
import subprocess
import json
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
import logging
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# Base directory of the project (parent of this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Absolute paths to CSV data files used throughout the dashboard
trades_log_path = os.path.join(BASE_DIR, "data", "trades_log.csv")
open_positions_path = os.path.join(BASE_DIR, "data", "open_positions.csv")
top_candidates_path = os.path.join(BASE_DIR, "data", "top_candidates.csv")
scored_candidates_path = os.path.join(BASE_DIR, "data", "scored_candidates.csv")

# Additional datasets introduced for monitoring
metrics_summary_path = os.path.join(BASE_DIR, "data", "metrics_summary.csv")
executed_trades_path = os.path.join(BASE_DIR, "data", "executed_trades.csv")
historical_candidates_path = os.path.join(BASE_DIR, "data", "historical_candidates.csv")
execute_metrics_path = os.path.join(BASE_DIR, "data", "execute_metrics.json")

# Absolute paths to log files for the Screener tab
screener_log_dir = os.path.join(BASE_DIR, "logs")
pipeline_log_path = os.path.join(screener_log_dir, "pipeline.log")
monitor_log_path = os.path.expanduser("~/jbravo_screener/logs/monitor.log")
# Additional logs
screener_log_path = os.path.join(screener_log_dir, "screener.log")
backtest_log_path = os.path.join(screener_log_dir, "backtest.log")
execute_trades_log_path = os.path.join(screener_log_dir, "execute_trades.log")
error_log_path = os.path.join(screener_log_dir, "error.log")
metrics_log_path = os.path.join(screener_log_dir, "metrics.log")
pipeline_status_json_path = os.path.join(BASE_DIR, "data", "pipeline_status.json")

# Threshold in minutes to consider a log stale
STALE_THRESHOLD_MINUTES = 1440  # 24 hours

# Displayed configuration values
MAX_OPEN_TRADES = 10


def is_log_stale(log_path):
    """Return True if the most recent INFO/ERROR entry in ``log_path`` is older than 24 hours."""
    if not os.path.exists(log_path):
        return True
    try:
        with open(log_path) as file:
            lines = file.readlines()
        for line in reversed(lines):
            if '[INFO]' in line or '[ERROR]' in line:
                ts_str = line[:19]
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                return (datetime.utcnow() - ts).total_seconds() > (STALE_THRESHOLD_MINUTES * 60)
    except Exception:
        return True
    return True

load_dotenv(os.path.join(BASE_DIR, ".env"))
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
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


def load_csv(filepath, required_columns=None, alert_prefix: str | None = None):
    """Load a CSV file from ``filepath`` and validate required columns.

    Returns a tuple of (DataFrame, alert_component). If the file is missing or
    columns are absent, an empty DataFrame and a Dash alert component are
    returned so the UI can gracefully inform the user.
    """
    filename = os.path.basename(filepath)
    prefix = f"{alert_prefix}: " if alert_prefix else ""
    if not os.path.exists(filepath):
        alert = dbc.Alert(f"{prefix}{filename} not found.", color="warning", className="m-2")
        return pd.DataFrame(), alert

    try:
        df = pd.read_csv(filepath, sep=",", encoding="utf-8")
        df.rename(columns=lambda c: c.strip(), inplace=True)
        app.logger.info("Loaded columns: %s", df.columns.tolist())
    except pd.errors.EmptyDataError:
        app.logger.warning("No data in %s", filepath)
        alert = dbc.Alert(
            f"{prefix}No data available in {filename}.",
            color="warning",
            className="m-2",
        )
        return pd.DataFrame(), alert
    except Exception as e:
        alert = dbc.Alert(
            f"{prefix}Error reading {filename}: {e}", color="danger", className="m-2"
        )
        return pd.DataFrame(), alert

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            msg = f"{prefix}{filename} missing columns: {', '.join(missing)}"
            app.logger.error(msg)
            alert = dbc.Alert(msg, color="danger", className="m-2")
            return pd.DataFrame(), alert

    if df.empty:
        alert = dbc.Alert(
            f"{prefix}No data available in {filename}.",
            color="warning",
            className="m-2",
        )
        return df, alert

    return df, None


def read_recent_lines(filepath, num_lines=50):
    """Return the last ``num_lines`` lines of ``filepath``.
    If the file is missing or unreadable, an informative message is returned
    instead.
    """
    filename = os.path.basename(filepath)
    if not os.path.exists(filepath):
        return [f"{filename} not found.\n"]

    try:
        with open(filepath, "r") as f:
            lines = f.readlines()[-num_lines:]
        return lines if lines else [f"No entries in {filename}.\n"]
    except Exception as e:
        return [f"Error reading {filename}: {e}\n"]


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


def stale_warning(paths: list[str], threshold_minutes: int = 30) -> html.Div | None:
    """Return a warning banner if any of ``paths`` are older than the threshold."""
    for path in paths:
        if not os.path.exists(path):
            continue
        mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
        age = (datetime.utcnow() - mtime).total_seconds() / 60
        if age > threshold_minutes:
            return html.Div(
                "Warning: Monitoring service is stale or scheduled tasks have errors!",
                style={"color": "red"},
            )
    return None


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


def get_file_mtime(path: str) -> float:
    """Return the modification time of ``path`` or 0."""
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


def format_time(ts: float) -> str:
    """Format a POSIX timestamp for display."""
    if not ts:
        return "N/A"
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M UTC")


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
            id="main-tabs",
            active_tab="tab-overview",
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
                    label="Screener",
                    tab_id="tab-screener",
                    tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                    active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                    className="custom-tab",
                ),
                dbc.Tab(
                    label="Execute Trades",
                    tab_id="tab-execute-trades",
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
                    tab_id="tab-symbols",
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


# Callbacks for tabs content
@app.callback(
    Output("tabs-content", "children"),
    [
        Input("main-tabs", "active_tab"),
        Input("interval-update", "n_intervals"),
        Input("log-interval", "n_intervals"),
        Input("refresh-button", "n_clicks"),
    ],
)
def render_tab(tab, n_intervals, n_log_intervals, refresh_clicks):
    if tab == "tab-overview":
        trades_df, alert = load_csv(
            trades_log_path, required_columns=["pnl", "entry_time"]
        )
        freshness = data_freshness_alert(trades_log_path, "Trade log")
        if alert:
            return alert

        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
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

        metrics_df, _ = load_csv(metrics_summary_path)
        if not metrics_df.empty:
            total_trades = int(metrics_df["Total Trades"][0])
            total_pnl = metrics_df["Total Net PnL"][0]
        else:
            total_trades = len(trades_df)
            total_pnl = trades_df["pnl"].sum()

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
                            dbc.CardBody(
                                html.H4(f"{(trades_df['pnl'] > 0).mean()*100:.2f}%")
                            ),
                        ]
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Expectancy"),
                            dbc.CardBody(html.H4(f"${trades_df['pnl'].mean():.2f}")),
                        ]
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Profit Factor"),
                            dbc.CardBody(
                                html.H4(
                                    f"{trades_df[trades_df['pnl']>0]['pnl'].sum()/abs(trades_df[trades_df['pnl']<0]['pnl'].sum()):.2f}"
                                )
                            ),
                        ]
                    ),
                    width=2,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Max Drawdown"),
                            dbc.CardBody(
                                html.H4(f"${trades_df['drawdown'].min():.2f}")
                            ),
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
        latest_file = trades_log_path
        if os.path.exists(executed_trades_path) and os.path.getmtime(
            executed_trades_path
        ) > os.path.getmtime(trades_log_path):
            latest_file = executed_trades_path
        timestamp = html.Div(
            f"Last Updated: {format_time(get_file_mtime(latest_file))}",
            className="text-muted mb-2",
        )

        components = [timestamp]
        if freshness:
            components.append(freshness)
        components.extend([kpis, graphs])
        return dbc.Container(components, fluid=True)

    elif tab == "tab-screener":
        df, alert = load_csv(top_candidates_path)
        scored_df, scored_alert = load_csv(scored_candidates_path)

        alerts = [alert] if alert else []
        if scored_alert:
            alerts.append(scored_alert)

        if not df.empty:
            if "score" in df.columns:
                df.sort_values("score", ascending=False, na_position="last", inplace=True)
            df = df.head(15)
            columns = [
                {"name": c.replace("_", " ").title(), "id": c} for c in df.columns
            ]
            table = dash_table.DataTable(
                id="screener-table",
                data=df.to_dict("records"),
                columns=columns,
                page_size=15,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
            )
        else:
            table = dbc.Alert("No candidates to display.", color="warning", className="m-2")

        if not scored_df.empty:
            scored_columns = [
                {"name": c.replace("_", " ").title(), "id": c} for c in scored_df.columns
            ]
            scored_table = dash_table.DataTable(
                id="scored-candidates-table",
                data=scored_df.to_dict("records"),
                columns=scored_columns,
                page_size=15,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
            )
        else:
            scored_table = dbc.Alert("No scored candidates available.", color="warning", className="m-2")

        pipeline_lines = read_recent_lines(pipeline_log_path)[::-1]
        screener_lines = read_recent_lines(screener_log_path, num_lines=20)[::-1]
        backtest_lines = read_recent_lines(backtest_log_path)[::-1]

        def format_lines(lines):
            return format_log_lines(lines)

        pipeline_log = html.Div(
            [
                html.H5("Pipeline Log", className="text-light"),
                html.Pre(
                    format_lines(pipeline_lines),
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

        screener_log = html.Div(
            [
                html.H5("Screener Log", className="text-light"),
                html.Pre(
                    format_lines(screener_lines),
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

        backtest_log = html.Div(
            [
                html.H5("Backtest Log", className="text-light"),
                html.Pre(
                    format_lines(backtest_lines),
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

        metrics_log = html.Div(
            [
                html.H5("Metrics Logs", className="text-light"),
                html.Pre(
                    id="metrics-logs",
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

        status = pipeline_status_component()
        freshness = data_freshness_alert(top_candidates_path, "Top candidates")

        timestamp = html.Div(
            f"Last Updated: {format_time(get_file_mtime(top_candidates_path))}",
            className="text-muted mb-2",
        )

        components = [timestamp]
        components.extend(alerts)
        components.extend([table, scored_table, status])
        if freshness:
            components.append(freshness)
        components.extend([pipeline_log, screener_log, backtest_log, metrics_log])

        return dbc.Container(components, fluid=True)

    elif tab == "tab-execute-trades":
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

        metrics_data = {
            "symbols_processed": 0,
            "orders_submitted": 0,
            "symbols_skipped": 0,
            "api_retries": 0,
            "api_failures": 0,
        }
        if os.path.exists(execute_metrics_path):
            try:
                with open(execute_metrics_path) as f:
                    metrics_data.update(json.load(f))
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

        metrics_view = html.Div([
            html.H5("Execute Trades Metrics"),
            html.Ul([
                html.Li(f"Symbols Processed: {metrics_data['symbols_processed']}") ,
                html.Li(f"Trades Submitted: {metrics_data['orders_submitted']}"),
                html.Li(f"Trades Skipped: {metrics_data['symbols_skipped']}"),
                html.Li(f"Retries Attempted: {metrics_data['api_retries']}"),
                html.Li(f"API Failures: {metrics_data['api_failures']}")
            ]),
            html.Pre(
                id="execute-trades-log",
                style={
                    "maxHeight": "400px",
                    "overflowY": "auto",
                    "backgroundColor": "#272B30",
                    "color": "#E0E0E0",
                    "padding": "10px",
                },
            ),
        ])

        download = html.Div(
            [
                html.Button("Download CSV", id="btn-download-trades"),
                dcc.Download(id="download-executed-trades"),
            ],
            className="mb-2",
        )

        timestamp = html.Div(
            f"Last Updated: {format_time(get_file_mtime(executed_trades_path))}",
            className="text-muted mb-2",
        )

        components = [timestamp, download, table, html.Hr()]
        if api_error_alert:
            components.append(api_error_alert)
        components.append(metrics_view)
        return dbc.Container(components, fluid=True)


    elif tab == "tab-symbols":
        trades_df, alert = load_csv(trades_log_path, required_columns=["symbol", "pnl"])
        freshness = data_freshness_alert(trades_log_path, "Trade log")
        if alert:
            return alert
        symbol_perf = (
            trades_df.groupby("symbol")
            .agg({"pnl": ["count", "mean", "sum"]})
            .reset_index()
        )
        symbol_perf.columns = ["Symbol", "Trades", "Avg P/L", "Total P/L"]
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
        timestamp = html.Div(
            f"Last Updated: {format_time(get_file_mtime(trades_log_path))}",
            className="text-muted mb-2",
        )
        components = [timestamp]
        if freshness:
            components.append(freshness)
        components.extend([dcc.Graph(figure=symbol_fig), table])
        return dbc.Container(components, fluid=True)

    elif tab == "tab-monitor":
        # Load open positions
        positions_df, freshness_alert = load_csv(
            open_positions_path,
            required_columns=["symbol", "entry_time", "qty"],
            alert_prefix="Open positions",
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
        error_lines = read_recent_lines(error_log_path, num_lines=100)[::-1]

        monitor_log_box = log_box("Monitor Log", monitor_lines, "monitor-log")
        exec_log_box = log_box("Execution Log", exec_lines, "exec-log")
        error_log_box = log_box("Errors", error_lines, "error-log")

        stale_warning_banner = stale_warning(
            [monitor_log_path, pipeline_log_path], threshold_minutes=30
        )

        timestamp = html.Div(
            f"Last Updated: {format_time(get_file_mtime(open_positions_path))}",
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
            df = pd.read_csv(path)
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
    if os.path.exists(top_candidates_path):
        df = pd.read_csv(top_candidates_path)
        logger.info(
            "Screener table updated successfully with %d records.", len(df)
        )
        return df.to_dict("records")
    logger.warning("%s not found when updating screener table", top_candidates_path)
    return []


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
    if os.path.exists(executed_trades_path):
        trades_df = pd.read_csv(executed_trades_path)
        trades_df.sort_values("entry_time", ascending=False, inplace=True)
        return trades_df.to_dict("records")
    return []


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
