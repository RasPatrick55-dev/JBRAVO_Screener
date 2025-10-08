from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime
from typing import Iterable

import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dash_table import DataTable
from dash.dependencies import Input, Output

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
LOG_DIR = REPO_ROOT / "logs"

METRICS_JSON = DATA_DIR / "screener_metrics.json"
TOP_CSV = DATA_DIR / "top_candidates.csv"
SCORED_CSV = DATA_DIR / "scored_candidates.csv"
HIST_CSV = DATA_DIR / "screener_metrics_history.csv"
PRED_LATEST = DATA_DIR / "predictions" / "latest.csv"


def _safe_json(path: pathlib.Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def _safe_csv(path: pathlib.Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path, nrows=nrows)
    except Exception:
        pass
    return pd.DataFrame()


def _tail(path: pathlib.Path, lines: int = 200) -> str:
    try:
        if not path.exists():
            return "(no log)"
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= lines:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
            text = data.decode("utf-8", errors="ignore")
            return "\n".join(text.splitlines()[-lines:])
    except Exception as exc:
        return f"(log read error: {exc})"


def _why_from_breakdown(json_str: str) -> str:
    try:
        data = json.loads(json_str) if isinstance(json_str, str) else {}
    except Exception:
        data = {}
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
    items: list[tuple[str, float]] = []
    for key, value in (data or {}).items():
        base_key = str(key).replace("_z", "")
        if base_key not in labels:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        items.append((labels[base_key], numeric))
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    parts: list[str] = []
    for name, val in items[:4]:
        arrow = "▲" if val >= 0 else "▼"
        parts.append(f"{name} {arrow} {abs(val):.2f}")
    return " • ".join(parts) if parts else "n/a"


def _format_timestamp(ts: str | None) -> str:
    if not ts:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts)


def _make_stat_cards(items: Iterable[tuple[str, str]]) -> list[html.Div]:
    cards: list[html.Div] = []
    for title, value in items:
        cards.append(
            html.Div(
                [
                    html.Div(title, className="sh-kpi-title"),
                    html.Div(value, className="sh-kpi-value"),
                ],
                className="sh-kpi",
            )
        )
    return cards


def build_layout() -> html.Div:
    return html.Div(
        [
            dcc.Interval(id="sh-interval", interval=60 * 1000, n_intervals=0),
            dcc.Store(id="sh-metrics-store"),
            dcc.Store(id="sh-top-store"),
            dcc.Store(id="sh-hist-store"),
            html.Div(id="sh-kpis", className="sh-row"),
            html.Div(
                id="sh-http-stats",
                className="sh-row",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))",
                    "gap": "10px",
                    "marginBottom": "12px",
                },
            ),
            html.Div(
                [
                    dcc.Graph(id="sh-gate-pressure", style={"height": "300px"}),
                    dcc.Graph(id="sh-coverage", style={"height": "300px"}),
                    dcc.Graph(id="sh-timings", style={"height": "300px"}),
                ],
                className="sh-row",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr 1fr",
                    "gap": "14px",
                },
            ),
            dcc.Graph(id="sh-trends", style={"height": "280px", "marginBottom": "16px"}),
            html.Hr(),
            html.H3("Top Candidates"),
            DataTable(
                id="sh-top-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontSize": "13px",
                    "whiteSpace": "nowrap",
                    "textOverflow": "ellipsis",
                    "maxWidth": 220,
                },
                columns=[
                    {"name": "symbol", "id": "symbol"},
                    {"name": "Score", "id": "Score", "type": "numeric", "format": {"specifier": ".3f"}},
                    {"name": "close", "id": "close", "type": "numeric", "format": {"specifier": ".2f"}},
                    {"name": "ADV20", "id": "ADV20", "type": "numeric", "format": {"specifier": ".0f"}},
                    {"name": "ATR%", "id": "ATR_pct", "type": "numeric", "format": {"specifier": ".2%"}},
                    {"name": "Why (top contributors)", "id": "Why"},
                ],
            ),
            html.Hr(),
            html.Details(
                [
                    html.Summary("Diagnostics & Logs"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Latest Predictions (head)"),
                                    html.Pre(
                                        id="sh-preds-head",
                                        style={
                                            "maxHeight": "220px",
                                            "overflow": "auto",
                                            "background": "#0b0b0b",
                                            "color": "#cfcfcf",
                                            "padding": "8px",
                                        },
                                    ),
                                ],
                                style={"gridColumn": "1 / span 2"},
                            ),
                            html.Div(
                                [
                                    html.H4("Screener Log (tail)"),
                                    html.Pre(
                                        id="sh-screener-log",
                                        style={
                                            "maxHeight": "220px",
                                            "overflow": "auto",
                                            "background": "#0b0b0b",
                                            "color": "#cfcfcf",
                                            "padding": "8px",
                                        },
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.H4("Pipeline Log (tail)"),
                                    html.Pre(
                                        id="sh-pipeline-log",
                                        style={
                                            "maxHeight": "220px",
                                            "overflow": "auto",
                                            "background": "#0b0b0b",
                                            "color": "#cfcfcf",
                                            "padding": "8px",
                                        },
                                    ),
                                ]
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "2fr 1fr 1fr",
                            "gap": "12px",
                        },
                    ),
                ]
            ),
        ],
        style={"padding": "12px"},
    )


def register_callbacks(app) -> None:
    @app.callback(
        Output("sh-metrics-store", "data"),
        Output("sh-top-store", "data"),
        Output("sh-hist-store", "data"),
        Input("sh-interval", "n_intervals"),
    )
    def _load_artifacts(_n):
        metrics = _safe_json(METRICS_JSON)
        top = _safe_csv(TOP_CSV)
        if not top.empty:
            if "ATR14" not in top.columns:
                scored = _safe_csv(SCORED_CSV)
                if not scored.empty:
                    cols = [col for col in ["symbol", "timestamp", "ATR14", "ADV20"] if col in scored.columns]
                    if cols:
                        merge_keys = [c for c in ["symbol", "timestamp"] if c in cols]
                        if not merge_keys:
                            merge_keys = [cols[0]]
                        use = scored[cols].copy()
                        use = use.drop_duplicates(subset=merge_keys)
                        top = top.merge(use, on=merge_keys, how="left")
            if "ATR14" in top.columns and "close" in top.columns:
                with pd.option_context("mode.use_inf_as_na", True):
                    top["ATR_pct"] = (top["ATR14"].astype(float) / top["close"].astype(float)).clip(lower=0)
            else:
                top["ATR_pct"] = 0.0
            if "score_breakdown" in top.columns:
                top["Why"] = top["score_breakdown"].apply(_why_from_breakdown)
            else:
                top["Why"] = "n/a"
        history = _safe_csv(HIST_CSV)
        return (
            metrics,
            top.to_dict("records") if not top.empty else [],
            history.to_dict("records") if not history.empty else [],
        )

    @app.callback(
        Output("sh-kpis", "children"),
        Output("sh-gate-pressure", "figure"),
        Output("sh-coverage", "figure"),
        Output("sh-timings", "figure"),
        Output("sh-top-table", "data"),
        Output("sh-preds-head", "children"),
        Output("sh-screener-log", "children"),
        Output("sh-pipeline-log", "children"),
        Output("sh-http-stats", "children"),
        Output("sh-trends", "figure"),
        Input("sh-metrics-store", "data"),
        Input("sh-top-store", "data"),
        Input("sh-hist-store", "data"),
    )
    def _render(metrics, top_rows, hist_rows):
        metrics = metrics or {}

        def _card(title: str, value: str, sub: str | None = None) -> html.Div:
            return html.Div(
                [
                    html.Div(title, className="sh-kpi-title"),
                    html.Div(value, className="sh-kpi-value"),
                    html.Div(sub or "", className="sh-kpi-sub"),
                ],
                className="sh-kpi",
            )

        last_run = _format_timestamp(metrics.get("last_run_utc"))
        symbols_in = int(metrics.get("symbols_in", 0) or 0)
        symbols_with_bars = int(metrics.get("symbols_with_bars", 0) or 0)
        bars_total = int(metrics.get("bars_rows_total", 0) or 0)
        candidates = int(metrics.get("rows", 0) or 0)
        coverage_pct = (symbols_with_bars / max(symbols_in, 1)) * 100 if symbols_in else 0.0

        kpi_children = html.Div(
            [
                _card("Last Run (UTC)", last_run),
                _card("Symbols In", f"{symbols_in:,}"),
                _card("With Bars", f"{symbols_with_bars:,}", f"{coverage_pct:.1f}%"),
                _card("Bar Rows", f"{bars_total:,}"),
                _card("Candidates", f"{candidates:,}"),
            ],
            className="sh-kpi-wrap",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                "gap": "10px",
                "marginBottom": "12px",
            },
        )

        gate_failures = metrics.get("gate_fail_counts", {}) or {}
        if gate_failures:
            df_fail = pd.DataFrame({"gate": list(gate_failures.keys()), "count": list(gate_failures.values())})
            df_fail = df_fail.sort_values("count", ascending=False)
            fig_gates = px.bar(df_fail, x="gate", y="count", title="Gate Pressure (Failures by Gate)")
        else:
            fig_gates = px.bar(title="Gate Pressure (no data)")

        cov_df = pd.DataFrame(
            {
                "label": ["With Bars", "No Bars"],
                "value": [symbols_with_bars, max(symbols_in - symbols_with_bars, 0)],
            }
        )
        fig_cov = px.pie(cov_df, names="label", values="value", title="Universe Coverage", hole=0.45)

        timings = metrics.get("timings", {}) or {}
        tm_df = pd.DataFrame(
            {
                "stage": ["fetch", "features", "rank", "gates"],
                "secs": [
                    float(timings.get("fetch_secs", 0) or 0),
                    float(timings.get("feature_secs", 0) or 0),
                    float(timings.get("rank_secs", 0) or 0),
                    float(timings.get("gates_secs", 0) or 0),
                ],
            }
        )
        fig_timings = px.bar(tm_df, x="stage", y="secs", title="Stage Timings (sec)")

        top_data = top_rows or []

        preds = _safe_csv(PRED_LATEST, nrows=8)
        preds_head = preds.to_csv(index=False) if not preds.empty else "(no predictions yet)"

        screener_tail = _tail(LOG_DIR / "screener.log", 180)
        pipeline_tail = _tail(LOG_DIR / "pipeline.log", 180)

        def _safe_int(value) -> int:
            try:
                return int(value or 0)
            except Exception:
                return 0

        http_meta = (metrics.get("http") or {}) if isinstance(metrics.get("http"), dict) else {}
        cache_meta = (metrics.get("cache") or {}) if isinstance(metrics.get("cache"), dict) else {}

        http_cards = _make_stat_cards(
            [
                ("HTTP 429", f"{_safe_int(http_meta.get('429', metrics.get('rate_limited', 0))):,}"),
                ("HTTP 404", f"{_safe_int(http_meta.get('404', metrics.get('http_404_batches', 0))):,}"),
                (
                    "HTTP Empty",
                    f"{_safe_int(http_meta.get('empty_pages', metrics.get('http_empty_batches', 0))):,}",
                ),
                ("Cache Hit", f"{_safe_int(cache_meta.get('batches_hit', metrics.get('cache_hits', 0))):,}"),
                (
                    "Cache Miss",
                    f"{_safe_int(cache_meta.get('batches_miss', metrics.get('cache_misses', 0))):,}",
                ),
                ("Window Used", str(metrics.get("window_used", "n/a"))),
                ("No Bars", f"{_safe_int(metrics.get('symbols_no_bars', 0)):,}"),
            ]
        )

        hist_rows = hist_rows or []
        if hist_rows:
            hist_df = pd.DataFrame(hist_rows)
            if "run_utc" in hist_df.columns:
                hist_df["run_utc"] = pd.to_datetime(hist_df["run_utc"], errors="coerce")
            else:
                hist_df["run_utc"] = pd.NaT
            hist_df = hist_df.dropna(subset=["run_utc"]).sort_values("run_utc")
            if not hist_df.empty:
                melted = hist_df.melt(
                    id_vars="run_utc",
                    value_vars=[
                        col
                        for col in [
                            "rows",
                            "symbols_with_bars",
                            "bars_rows_total",
                        ]
                        if col in hist_df.columns
                    ],
                    var_name="metric",
                    value_name="value",
                )
                if melted.empty:
                    fig_trend = px.line(title="Screener Trends (no numeric metrics)")
                else:
                    fig_trend = px.line(
                        melted,
                        x="run_utc",
                        y="value",
                        color="metric",
                        title="Screener Trends",
                        markers=True,
                    )
            else:
                fig_trend = px.line(title="Screener Trends (no history)")
        else:
            fig_trend = px.line(title="Screener Trends (no history)")

        return (
            kpi_children,
            fig_gates,
            fig_cov,
            fig_timings,
            top_data,
            preds_head,
            screener_tail,
            pipeline_tail,
            http_cards,
            fig_trend,
        )
