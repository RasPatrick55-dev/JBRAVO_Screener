# Complete Integrated Dashboard (dashboard_app.py)

import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from datetime import datetime
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

# Additional datasets introduced for monitoring
metrics_summary_path = os.path.join(BASE_DIR, "data", "metrics_summary.csv")
executed_trades_path = os.path.join(BASE_DIR, "data", "executed_trades.csv")
historical_candidates_path = os.path.join(BASE_DIR, "data", "historical_candidates.csv")

# Absolute paths to log files for the Screener tab
screener_log_dir = os.path.join(BASE_DIR, "logs")
pipeline_log_path = os.path.join(screener_log_dir, "pipeline_log.txt")
monitor_log_path = os.path.join(screener_log_dir, "monitor.log")
# Additional logs
screener_log_path = os.path.join(screener_log_dir, "screener.log")
backtest_log_path = os.path.join(screener_log_dir, "backtest.log")
execute_trades_log_path = os.path.join(screener_log_dir, "execute_trades.log")
error_log_path = os.path.join(screener_log_dir, "error.log")

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


def load_csv(filepath, required_columns=None):
    """Load a CSV file from ``filepath`` and validate required columns.

    Returns a tuple of (DataFrame, alert_component). If the file is missing or
    columns are absent, an empty DataFrame and a Dash alert component are
    returned so the UI can gracefully inform the user.
    """
    filename = os.path.basename(filepath)
    if not os.path.exists(filepath):
        alert = dbc.Alert(f"{filename} not found.", color="warning", className="m-2")
        return pd.DataFrame(), alert

    try:
        df = pd.read_csv(filepath, sep=",", encoding="utf-8")
        df.rename(columns=lambda c: c.strip(), inplace=True)
        app.logger.info("Loaded columns: %s", df.columns.tolist())
    except pd.errors.EmptyDataError:
        app.logger.warning("No data in %s", filepath)
        alert = dbc.Alert(
            f"No data available in {filename}.", color="warning", className="m-2"
        )
        return pd.DataFrame(), alert
    except Exception as e:
        alert = dbc.Alert(
            f"Error reading {filename}: {e}", color="danger", className="m-2"
        )
        return pd.DataFrame(), alert

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            msg = f"{filename} missing columns: {', '.join(missing)}"
            app.logger.error(msg)
            alert = dbc.Alert(msg, color="danger", className="m-2")
            return pd.DataFrame(), alert

    if df.empty:
        alert = dbc.Alert(
            f"No data available in {filename}.", color="warning", className="m-2"
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


def file_timestamp(path):
    """Return modification time of ``path`` formatted for display."""
    if not os.path.exists(path):
        return "N/A"
    return datetime.utcfromtimestamp(os.path.getmtime(path)).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def pipeline_status_component():
    """Create status indicators for Screener -> Backtest -> Execution."""
    steps = [
        ("Screener", screener_log_path),
        ("Backtest", backtest_log_path),
        ("Execution", execute_trades_log_path),
    ]
    now = datetime.now()
    items = []
    for name, path in steps:
        if os.path.exists(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            age = (now - mtime).total_seconds() / 60
            if age < 60:
                color = "success"
                status = "Completed"
            else:
                color = "warning"
                status = "Stale"
            timestamp = mtime.strftime("%Y-%m-%d %H:%M")
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
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    age = (datetime.now() - mtime).total_seconds() / 60
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
                    label="Pipeline Log",
                    tab_id="tab-trades",
                    tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                    active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                    className="custom-tab",
                ),
                dbc.Tab(
                    label="Open Positions",
                    tab_id="tab-positions",
                    tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                    active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                    className="custom-tab",
                ),
                dbc.Tab(
                    label="Symbol Performance",
                    tab_id="tab-symbols",
                    tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                    active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                    className="custom-tab",
                ),
                dbc.Tab(
                    label="Monitor Log",
                    tab_id="tab-monitor",
                    tab_style={"backgroundColor": "#343a40", "color": "#ccc"},
                    active_tab_style={"backgroundColor": "#17a2b8", "color": "#fff"},
                    className="custom-tab",
                ),
            ],
        ),
        html.Div(id="tabs-content", className="mt-4"),
        dcc.Interval(id="interval-update", interval=600000, n_intervals=0),
        dcc.Interval(id="log-interval", interval=10000, n_intervals=0),
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
    ],
)
def render_tab(tab, n_intervals, n_log_intervals):
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
            f"Data last refreshed: {file_timestamp(latest_file)}",
            className="text-muted mb-2",
        )

        components = [timestamp]
        if freshness:
            components.append(freshness)
        components.extend([kpis, graphs])
        return dbc.Container(components, fluid=True)

    elif tab == "tab-screener":
        df, alert = load_csv(top_candidates_path)

        alerts = [alert] if alert else []

        if not df.empty:
            if "score" in df.columns:
                df.sort_values("score", ascending=False, inplace=True)
            df = df.head(15)
            columns = [
                {"name": c.replace("_", " ").title(), "id": c} for c in df.columns
            ]
            table = dash_table.DataTable(
                id="top-candidates-table",
                data=df.to_dict("records"),
                columns=columns,
                page_size=15,
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
            )
        else:
            table = dbc.Alert("No candidates to display.", color="warning", className="m-2")

        pipeline_lines = read_recent_lines(pipeline_log_path)[::-1]
        screener_lines = read_recent_lines(screener_log_path)[::-1]
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

        status = pipeline_status_component()
        freshness = data_freshness_alert(top_candidates_path, "Top candidates")

        timestamp = html.Div(
            f"Data last refreshed: {file_timestamp(top_candidates_path)}",
            className="text-muted mb-2",
        )

        components = [timestamp]
        components.extend(alerts)
        components.extend([table, status])
        if freshness:
            components.append(freshness)
        components.extend([pipeline_log, screener_log, backtest_log])

        return dbc.Container(components, fluid=True)

    elif tab == "tab-trades":
        lines = read_recent_lines(pipeline_log_path)[::-1]
        timestamp = html.Div(
            f"Log last refreshed: {file_timestamp(pipeline_log_path)}",
            className="text-muted mb-2",
        )
        log_view = html.Div(
            [
                html.H5("Pipeline Log", className="text-light"),
                html.Pre(
                    format_log_lines(lines),
                    style={
                        "maxHeight": "400px",
                        "overflowY": "auto",
                        "backgroundColor": "#272B30",
                        "color": "#E0E0E0",
                        "padding": "0.5rem",
                    },
                ),
            ]
        )
        return dbc.Container([timestamp, log_view], fluid=True)

    elif tab == "tab-positions":
        positions_df, alert = load_csv(
            open_positions_path,
            required_columns=["symbol", "entry_price", "entry_time"],
        )
        freshness = data_freshness_alert(open_positions_path, "Open positions")
        exec_fresh = data_freshness_alert(executed_trades_path, "Executed trades")
        if positions_df.empty:
            fallback_df = fetch_positions_api()
            if not fallback_df.empty:
                positions_df = fallback_df
                alert = dbc.Alert(
                    "open_positions.csv empty; loaded positions from Alpaca API as fallback.",
                    color="warning",
                    className="m-2",
                )
                logger.info(
                    "open_positions.csv empty; loaded positions from Alpaca API as fallback."
                )
            elif alert:
                return alert
        if alert and not positions_df.empty:
            alert = None
        pnl_col = (
            "unrealized_pl"
            if "unrealized_pl" in positions_df.columns
            else "pnl" if "pnl" in positions_df.columns else None
        )
        if pnl_col is None:
            return dbc.Alert(
                "open_positions.csv missing unrealized P/L data.",
                color="warning",
                className="m-2",
            )
        positions_df["entry_time"] = pd.to_datetime(
            positions_df["entry_time"]
        ).dt.strftime("%Y-%m-%d %H:%M")
        columns = [
            {"name": c.replace("_", " ").title(), "id": c} for c in positions_df.columns
        ]
        positions_fig = px.bar(
            positions_df,
            x="symbol",
            y=pnl_col,
            color=positions_df[pnl_col] > 0,
            color_discrete_map={True: "#4DB6AC", False: "#E57373"},
            template="plotly_dark",
            title="Open Positions P/L",
        )
        table = dash_table.DataTable(
            data=positions_df.to_dict("records"),
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
        exec_df, exec_alert = load_csv(
            executed_trades_path,
            required_columns=["symbol", "order_status", "entry_time"],
        )
        if exec_alert:
            exec_table = exec_alert
        else:
            exec_df["entry_time"] = pd.to_datetime(exec_df["entry_time"]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )
            e_cols = [
                {"name": c.replace("_", " ").title(), "id": c}
                for c in ["symbol", "side", "order_status", "entry_time"]
                if c in exec_df.columns
            ]
            exec_table = dash_table.DataTable(
                data=exec_df[["symbol", "side", "order_status", "entry_time"]].to_dict(
                    "records"
                ),
                columns=e_cols,
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
            )
        latest_file = open_positions_path
        if os.path.exists(executed_trades_path) and os.path.getmtime(
            executed_trades_path
        ) > os.path.getmtime(open_positions_path):
            latest_file = executed_trades_path
        timestamp = html.Div(
            f"Data last refreshed: {file_timestamp(latest_file)}",
            className="text-muted mb-2",
        )
        components = [timestamp]
        if freshness:
            components.append(freshness)
        if alert:
            components.append(alert)
        components.extend(
            [
                dcc.Graph(figure=positions_fig),
                table,
                html.Hr(),
                html.H5("Recent Order Status", className="text-light"),
                exec_table,
            ]
        )
        if exec_fresh:
            components.append(exec_fresh)
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
            f"Data last refreshed: {file_timestamp(trades_log_path)}",
            className="text-muted mb-2",
        )
        components = [timestamp]
        if freshness:
            components.append(freshness)
        components.extend([dcc.Graph(figure=symbol_fig), table])
        return dbc.Container(components, fluid=True)

    elif tab == "tab-monitor":
        closed_df, alert = load_csv(
            trades_log_path, required_columns=["symbol", "exit_time", "pnl"]
        )
        freshness = data_freshness_alert(trades_log_path, "Trade log")
        if alert:
            closed_table = alert
        else:
            closed_df["exit_time"] = pd.to_datetime(closed_df["exit_time"])
            closed_df.sort_values("exit_time", ascending=False, inplace=True)
            recent_trades = closed_df.head(10)
            columns = [
                {"name": c.replace("_", " ").title(), "id": c}
                for c in recent_trades.columns
            ]
            closed_table = dash_table.DataTable(
                data=recent_trades.to_dict("records"),
                columns=columns,
                page_size=10,
                filter_action="native",
                sort_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"backgroundColor": "#212529", "color": "#E0E0E0"},
            )

        monitor_lines = read_recent_lines(monitor_log_path, num_lines=100)[::-1]
        exec_lines = read_recent_lines(execute_trades_log_path, num_lines=50)[::-1]
        error_lines = read_recent_lines(error_log_path, num_lines=50)[::-1]

        def log_box(title, lines):
            return html.Div(
                [
                    html.H5(title, className="text-light"),
                    html.Pre(
                        format_log_lines(lines),
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

        heartbeat = html.Div(
            [
                html.Span(
                    f"Last pipeline run: {file_timestamp(pipeline_log_path)}",
                    className="me-3",
                ),
                html.Span(f"Last monitor update: {file_timestamp(monitor_log_path)}"),
            ],
            className="text-muted",
        )

        timestamp = html.Div(
            f"Data last refreshed: {file_timestamp(trades_log_path)}",
            className="text-muted mb-2",
        )

        components = [timestamp]
        if freshness:
            components.append(freshness)
        components.extend(
            [
                html.H5("Recently Closed Positions", className="text-light"),
                closed_table,
                html.Hr(),
                log_box("Monitor Log", monitor_lines),
                log_box("Execution Log", exec_lines),
                log_box("Errors", error_lines),
                heartbeat,
            ]
        )
        return dbc.Container(components, fluid=True)


# Callback for modal interaction
@app.callback(
    [Output("detail-modal", "is_open"), Output("modal-content", "children")],
    [
        Input("top-candidates-table", "active_cell"),
        Input("top-candidates-table", "data"),
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


if __name__ == "__main__":
    app.run(debug=False)
