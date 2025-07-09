# Complete Integrated Dashboard (dashboard_app.py)

import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# Base directory of this dashboard application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute paths to CSV data files used throughout the dashboard
trades_log_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'trades_log.csv'))
open_positions_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'open_positions.csv'))
top_candidates_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'top_candidates.csv'))

# Absolute paths to log files for the Screener tab
pipeline_log_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'logs', 'pipeline_log.txt'))
monitor_log_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'logs', 'monitor.log'))

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
        df = pd.read_csv(filepath, sep=',', encoding='utf-8')
        df.rename(columns=lambda c: c.strip(), inplace=True)
        app.logger.info("Loaded columns: %s", df.columns.tolist())
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
        alert = dbc.Alert(f"No data available in {filename}.", color="warning", className="m-2")
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
        with open(filepath, 'r') as f:
            lines = f.readlines()[-num_lines:]
        return lines if lines else [f"No entries in {filename}.\n"]
    except Exception as e:
        return [f"Error reading {filename}: {e}\n"]

app = Dash(__name__, external_stylesheets=[
    dbc.themes.DARKLY,
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V2.1.0/dbc.min.css"
])

# Layout with Tabs and Modals
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1('JBravo Swing Trading Dashboard', className="text-center my-4 text-light"))),

    dbc.Tabs(
        id='main-tabs',
        active_tab='tab-overview',
        class_name='mb-3',
        children=[
            dbc.Tab(label='Overview', tab_id='tab-overview', tab_style={'backgroundColor':'#343a40','color':'#ccc'}, active_tab_style={'backgroundColor':'#17a2b8','color':'#fff'}, className='custom-tab'),
            dbc.Tab(label='Screener', tab_id='tab-screener', tab_style={'backgroundColor':'#343a40','color':'#ccc'}, active_tab_style={'backgroundColor':'#17a2b8','color':'#fff'}, className='custom-tab'),
            dbc.Tab(label='Trade Log', tab_id='tab-trades', tab_style={'backgroundColor':'#343a40','color':'#ccc'}, active_tab_style={'backgroundColor':'#17a2b8','color':'#fff'}, className='custom-tab'),
            dbc.Tab(label='Open Positions', tab_id='tab-positions', tab_style={'backgroundColor':'#343a40','color':'#ccc'}, active_tab_style={'backgroundColor':'#17a2b8','color':'#fff'}, className='custom-tab'),
            dbc.Tab(label='Symbol Performance', tab_id='tab-symbols', tab_style={'backgroundColor':'#343a40','color':'#ccc'}, active_tab_style={'backgroundColor':'#17a2b8','color':'#fff'}, className='custom-tab')
        ]
    ),

    html.Div(id='tabs-content', className="mt-4"),

    dcc.Interval(id='interval-update', interval=60000, n_intervals=0),

    dbc.Modal(id='detail-modal', is_open=False, size="lg", children=[
        dbc.ModalHeader(dbc.ModalTitle("Details")),
        dbc.ModalBody(id='modal-content'),
        dbc.ModalFooter(dbc.Button("Close", id="close-modal", className="ms-auto"))
    ])
], fluid=True)

# Callbacks for tabs content
@app.callback(
    Output('tabs-content', 'children'),
    Input('main-tabs', 'active_tab')
)
def render_tab(tab):
    if tab == 'tab-overview':
        trades_df, alert = load_csv(trades_log_path, required_columns=['pnl', 'entry_time'])
        if alert:
            return alert
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        equity_fig = px.line(trades_df, x='entry_time', y='cumulative_pnl', template='plotly_dark', title='Equity Curve')

        kpis = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Win Rate"), dbc.CardBody(html.H4(f"{(trades_df['pnl'] > 0).mean()*100:.2f}%"))]), width=3),
            dbc.Col(dbc.Card([dbc.CardHeader("Expectancy"), dbc.CardBody(html.H4(f"${trades_df['pnl'].mean():.2f}"))]), width=3),
            dbc.Col(dbc.Card([dbc.CardHeader("Profit Factor"), dbc.CardBody(html.H4(f"{trades_df[trades_df['pnl']>0]['pnl'].sum()/abs(trades_df[trades_df['pnl']<0]['pnl'].sum()):.2f}"))]), width=3),
            dbc.Col(dbc.Card([dbc.CardHeader("Max Drawdown"), dbc.CardBody(html.H4(f"${(trades_df['cumulative_pnl']-trades_df['cumulative_pnl'].cummax()).min():.2f}"))]), width=3)
        ])

        return dbc.Container([kpis, dcc.Graph(figure=equity_fig)], fluid=True)

    elif tab == 'tab-screener':
        candidates_df, alert = load_csv(top_candidates_path)
        if alert:
            table = alert
        else:
            columns = [{'name': c.replace('_',' ').title(), 'id': c} for c in candidates_df.columns]
            table = dash_table.DataTable(
                id='top-candidates-table',
                data=candidates_df.to_dict('records'),
                columns=columns,
                page_size=20,
                filter_action='native',
                sort_action='native',
                style_table={'overflowX':'auto'},
                style_cell={'backgroundColor':'#212529','color':'#E0E0E0'}
            )

        pipeline_lines = read_recent_lines(pipeline_log_path)
        monitor_lines = read_recent_lines(monitor_log_path)

        def format_lines(lines):
            return [html.Span(l, style={'color':'#E57373'} if 'ERROR' in l else {}) for l in lines]

        pipeline_log = html.Div([
            html.H5('Pipeline Log', className='text-light'),
            html.Pre(format_lines(pipeline_lines), style={'maxHeight':'200px','overflowY':'auto','backgroundColor':'#272B30','color':'#E0E0E0','padding':'0.5rem'})
        ], className='mb-3')

        monitor_log = html.Div([
            html.H5('Monitor Log', className='text-light'),
            html.Pre(format_lines(monitor_lines), style={'maxHeight':'200px','overflowY':'auto','backgroundColor':'#272B30','color':'#E0E0E0','padding':'0.5rem'})
        ])

        return dbc.Container([table, pipeline_log, monitor_log], fluid=True)

    elif tab == 'tab-trades':
        trades_df, alert = load_csv(
            trades_log_path, required_columns=['symbol', 'entry_time', 'pnl']
        )
        if alert:
            return alert
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        columns = [{'name': c.replace('_',' ').title(), 'id': c} for c in trades_df.columns]
        return dash_table.DataTable(data=trades_df.to_dict('records'), columns=columns, page_size=20, filter_action="native", sort_action="native",
                                    style_table={'overflowX':'auto'}, style_cell={'backgroundColor':'#212529','color':'#E0E0E0'})

    elif tab == 'tab-positions':
        positions_df, alert = load_csv(open_positions_path, required_columns=['symbol'])
        if alert:
            return alert
        pnl_col = 'unrealized_pl' if 'unrealized_pl' in positions_df.columns else 'pnl' if 'pnl' in positions_df.columns else None
        if pnl_col is None:
            return dbc.Alert("open_positions.csv missing unrealized P/L data.", color="warning", className="m-2")
        columns = [{'name': c.replace('_',' ').title(), 'id': c} for c in positions_df.columns]
        positions_fig = px.bar(positions_df, x='symbol', y=pnl_col, color=positions_df[pnl_col]>0, color_discrete_map={True:'#4DB6AC',False:'#E57373'}, template='plotly_dark', title='Open Positions P/L')
        return dbc.Container([dcc.Graph(figure=positions_fig), dash_table.DataTable(data=positions_df.to_dict('records'), columns=columns, style_table={'overflowX':'auto'}, style_cell={'backgroundColor':'#212529','color':'#E0E0E0'})])

    elif tab == 'tab-symbols':
        trades_df, alert = load_csv(trades_log_path, required_columns=['symbol', 'pnl'])
        if alert:
            return alert
        symbol_perf = trades_df.groupby('symbol').agg({'pnl':['count','mean','sum']}).reset_index()
        symbol_perf.columns = ['Symbol','Trades','Avg P/L','Total P/L']
        symbol_fig = px.bar(symbol_perf, x='Symbol', y='Total P/L', color='Total P/L', template='plotly_dark', title='Performance by Symbol')
        columns = [{'name': c, 'id': c} for c in symbol_perf.columns]
        return dbc.Container([dcc.Graph(figure=symbol_fig), dash_table.DataTable(data=symbol_perf.to_dict('records'), columns=columns, style_table={'overflowX':'auto'}, style_cell={'backgroundColor':'#212529','color':'#E0E0E0'})])

# Callback for modal interaction
@app.callback(
    [Output('detail-modal', 'is_open'), Output('modal-content', 'children')],
    [Input('top-candidates-table', 'active_cell'), Input('close-modal', 'n_clicks')],
    [State('detail-modal', 'is_open')]
)
def toggle_modal(active_cell, close_click, is_open):
    if active_cell and not is_open:
        return True, html.Div("Detailed information would appear here (charts, additional stats, etc.)")
    if close_click and is_open:
        return False, ""
    return is_open, ""

if __name__ == '__main__':
    app.run(debug=False)

