# Complete Integrated Dashboard (dashboard_app.py)

import dash
from dash import Dash, html, dash_table, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

# Define absolute paths
dirname = os.path.dirname(os.path.abspath(__file__))

def load_csv(filename):
    filepath = os.path.join(dirname, filename)
    return pd.read_csv(filepath) if os.path.exists(filepath) else pd.DataFrame()

app = Dash(__name__, external_stylesheets=[
    dbc.themes.DARKLY,
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V2.1.0/dbc.min.css"
])

# Layout with Tabs and Modals
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1('JBravo Swing Trading Dashboard', className="text-center my-4 text-light"))),

    dcc.Tabs(id='main-tabs', value='tab-overview', children=[
        dcc.Tab(label='Overview', value='tab-overview'),
        dcc.Tab(label='Trade Log', value='tab-trades'),
        dcc.Tab(label='Open Positions', value='tab-positions'),
        dcc.Tab(label='Symbol Performance', value='tab-symbols')
    ], className="my-2"),

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
    Input('main-tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab-overview':
        trades_df = load_csv('trades_log.csv')
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        equity_fig = px.line(trades_df, x='entry_time', y='cumulative_pnl', template='plotly_dark', title='Equity Curve')

        kpis = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Win Rate"), dbc.CardBody(html.H4(f"{(trades_df['pnl'] > 0).mean()*100:.2f}%"))]), width=3),
            dbc.Col(dbc.Card([dbc.CardHeader("Expectancy"), dbc.CardBody(html.H4(f"${trades_df['pnl'].mean():.2f}"))]), width=3),
            dbc.Col(dbc.Card([dbc.CardHeader("Profit Factor"), dbc.CardBody(html.H4(f"{trades_df[trades_df['pnl']>0]['pnl'].sum()/abs(trades_df[trades_df['pnl']<0]['pnl'].sum()):.2f}"))]), width=3),
            dbc.Col(dbc.Card([dbc.CardHeader("Max Drawdown"), dbc.CardBody(html.H4(f"${(trades_df['cumulative_pnl']-trades_df['cumulative_pnl'].cummax()).min():.2f}"))]), width=3)
        ])

        return dbc.Container([kpis, dcc.Graph(figure=equity_fig)], fluid=True)

    elif tab == 'tab-trades':
        trades_df = load_csv('trades_log.csv')
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
        columns = [{'name': c.replace('_',' ').title(), 'id': c} for c in trades_df.columns]
        return dash_table.DataTable(data=trades_df.to_dict('records'), columns=columns, page_size=20, filter_action="native", sort_action="native",
                                    style_table={'overflowX':'auto'}, style_cell={'backgroundColor':'#212529','color':'#E0E0E0'})

    elif tab == 'tab-positions':
        positions_df = load_csv('open_positions.csv')
        columns = [{'name': c.replace('_',' ').title(), 'id': c} for c in positions_df.columns]
        positions_fig = px.bar(positions_df, x='symbol', y='pnl', color=positions_df['pnl']>0, color_discrete_map={True:'#4DB6AC',False:'#E57373'}, template='plotly_dark', title='Open Positions P/L')
        return dbc.Container([dcc.Graph(figure=positions_fig), dash_table.DataTable(data=positions_df.to_dict('records'), columns=columns, style_table={'overflowX':'auto'}, style_cell={'backgroundColor':'#212529','color':'#E0E0E0'})])

    elif tab == 'tab-symbols':
        trades_df = load_csv('trades_log.csv')
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
