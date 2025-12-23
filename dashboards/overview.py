from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping
from zoneinfo import ZoneInfo

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html

from dashboards.utils import (
    safe_read_json,
    parse_pipeline_summary,
    safe_tail_text,
    parse_timed_events_from_logs,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"


def _read_screener_metrics() -> Dict[str, Any]:
    path = DATA_DIR / "screener_metrics.json"
    payload = safe_read_json(path)
    if payload:
        payload.setdefault("timestamp", payload.get("last_run_utc"))
    else:
        payload = parse_pipeline_summary(LOG_DIR / "pipeline.log")
    return payload


def _read_execute_metrics() -> Dict[str, Any]:
    path = DATA_DIR / "execute_metrics.json"
    return safe_read_json(path)


def _read_candidates() -> tuple[int | None, List[Dict[str, Any]]]:
    path = DATA_DIR / "latest_candidates.csv"
    if not path.exists():
        return None, []
    try:
        df = pd.read_csv(path)
        rows = len(df)
        top_cols = [col for col in ("symbol", "score", "model_score_5d") if col in df.columns]
        return rows, df[top_cols].head(5).to_dict(orient="records") if top_cols else []
    except Exception:
        return None, []


def _read_nightly_ml_status() -> Dict[str, Any]:
    return safe_read_json(DATA_DIR / "nightly_ml_status.json")


def _read_ranker_eval_path() -> Path | None:
    path = DATA_DIR / "ranker_eval" / "latest.json"
    return path if path.exists() else None


def _build_card(title: str, items: List[str]) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([html.H5(title, className="card-title"), html.Ul([html.Li(i) for i in items])]),
        className="mb-3",
    )


def _format_time(ts: str | None) -> str:
    if not ts:
        return "(missing)"
    try:
        return datetime.fromisoformat(str(ts)).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _compute_alerts(metrics: Dict[str, Any], exec_metrics: Dict[str, Any], candidates_rows: int | None) -> List[dbc.Badge]:
    alerts: List[dbc.Badge] = []
    now = datetime.now(timezone.utc)
    ts_raw = metrics.get("timestamp") or metrics.get("last_run_utc")
    try:
        dt = datetime.fromisoformat(str(ts_raw)) if ts_raw else None
    except Exception:
        dt = None
    if dt and now - dt > timedelta(hours=6):
        alerts.append(dbc.Badge("Pipeline stale", color="danger", className="me-2"))
    auth_ok = exec_metrics.get("auth_ok")
    if auth_ok is False:
        alerts.append(dbc.Badge("Auth failed", color="danger", className="me-2"))
    finished = exec_metrics.get("run_finished_utc") or exec_metrics.get("timestamp")
    if finished:
        try:
            finished_dt = datetime.fromisoformat(str(finished))
            if finished_dt.date() != now.date():
                alerts.append(dbc.Badge("Execute stale", color="warning", className="me-2"))
        except Exception:
            pass
    else:
        alerts.append(dbc.Badge("Execute stale", color="warning", className="me-2"))

    skips = exec_metrics.get("skips") or exec_metrics.get("skip_reasons") or {}
    if isinstance(skips, dict) and any("OPEN ORDER" in k for k in skips):
        alerts.append(dbc.Badge("Open order skip", color="warning", className="me-2"))

    duration_sec = exec_metrics.get("duration_sec")
    started = exec_metrics.get("run_started_utc")
    if duration_sec and started and finished:
        try:
            s_dt = datetime.fromisoformat(str(started))
            f_dt = datetime.fromisoformat(str(finished))
            actual = (f_dt - s_dt).total_seconds()
            if abs(actual - float(duration_sec)) > 10:
                alerts.append(dbc.Badge("Duration mismatch", color="warning", className="me-2"))
        except Exception:
            pass

    if not candidates_rows:
        alerts.append(dbc.Badge("Candidates missing", color="danger", className="me-2"))

    rows_out = metrics.get("rows_out")
    rows = metrics.get("rows")
    try:
        if rows_out and rows and float(rows_out) > float(rows) * 50:
            alerts.append(dbc.Badge("rows_out mismatch", color="warning", className="me-2"))
    except Exception:
        pass

    return alerts


def _build_today_timeline():
    try:
        ZoneInfo("America/New_York")
        tz_label = "America/New_York"
    except Exception:
        tz_label = "UTC (tzdata unavailable)"
    try:
        today_events = parse_timed_events_from_logs(
            LOG_DIR / "pipeline.log", LOG_DIR / "execute_trades.log"
        )
    except Exception as exc:  # pragma: no cover - defensive display
        return dbc.Alert(f"Timeline unavailable: {exc}", color="warning", className="mb-3")

    if today_events and today_events[0].get("tz_label"):
        tz_label = today_events[0]["tz_label"]
    tz_note = html.Small(
        f"Timezone: {tz_label}",
        className="text-muted",
    )

    controls = dbc.Row(
        [
            dbc.Col(
                [
                    html.Label("Show", className="text-muted small"),
                    dcc.RadioItems(
                        id="timeline-source-filter",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Pipeline", "value": "pipeline"},
                            {"label": "Execute", "value": "execute"},
                        ],
                        value="all",
                        className="text-light",
                        inputClassName="me-1",
                        labelClassName="me-3",
                    ),
                ],
                md=6,
            ),
            dbc.Col(
                [
                    html.Label("Severity", className="text-muted small"),
                    dcc.RadioItems(
                        id="timeline-severity-filter",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Warn + Error", "value": "warn"},
                            {"label": "Error only", "value": "error"},
                        ],
                        value="warn",
                        className="text-light",
                        inputClassName="me-1",
                        labelClassName="me-3",
                    ),
                ],
                md=6,
            ),
        ],
        className="mb-2",
    )

    placeholder_table = render_timeline_table(today_events, "all", "warn", tz_label)

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H5("Today Timeline (NY)", className="card-title"),
                    tz_note,
                    controls,
                    dcc.Store(id="timeline-events-store", data=today_events),
                    html.Div(placeholder_table, id="today-timeline-table"),
                ]
            )
        ],
        className="mb-3",
    )


def render_timeline_table(
    events: List[Dict[str, Any]],
    source_filter: str = "all",
    severity_filter: str = "all",
    tz_label: str | None = None,
):
    if not events:
        return dbc.Alert("No events found for today.", color="secondary", className="mb-0")

    filtered = events
    if source_filter in {"pipeline", "execute"}:
        filtered = [e for e in filtered if e.get("source") == source_filter]
    if severity_filter == "warn":
        filtered = [e for e in filtered if e.get("severity") in {"warn", "error"}]
    elif severity_filter == "error":
        filtered = [e for e in filtered if e.get("severity") == "error"]

    severity_colors = {"info": "secondary", "warn": "warning", "error": "danger"}
    time_label = f"Time ({'NY' if tz_label and 'New_York' in tz_label else tz_label or 'TZ'})"

    rows: List[html.Tr] = []
    for event in filtered[:200]:
        severity = event.get("severity") or "info"
        severity_badge = dbc.Badge(
            severity.upper(),
            color=severity_colors.get(severity, "secondary"),
            pill=True,
            className="ms-1",
        )
        event_badge = dbc.Badge(
            event.get("event", ""),
            color="info",
            className="me-1",
        )
        row_class = None
        if severity == "warn":
            row_class = "table-warning"
        elif severity == "error":
            row_class = "table-danger"
        rows.append(
            html.Tr(
                [
                    html.Td(event.get("time_str")),
                    html.Td([event_badge, severity_badge]),
                    html.Td(event.get("symbol") or "-"),
                    html.Td(event.get("details") or "-"),
                ],
                className=row_class,
            )
        )

    table = html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th(time_label),
                        html.Th("Event"),
                        html.Th("Symbol"),
                        html.Th("Details"),
                    ]
                )
            ),
            html.Tbody(rows if rows else [html.Tr(html.Td("No events match filters", colSpan=4))]),
        ],
        className="table table-sm table-dark mb-0",
    )
    return html.Div(table, style={"maxHeight": "320px", "overflowY": "auto"})


def _status_badge(label: str, status: str) -> dbc.Badge:
    colors = {"ok": "success", "warn": "warning", "error": "danger"}
    return dbc.Badge(label, color=colors.get(status, "secondary"), className="me-2", pill=True)


def _extract_skip_counts(raw: Mapping[str, Any] | None) -> Dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    cleaned: Dict[str, int] = {}
    for key, value in raw.items():
        count = _as_int(value)
        if count is None:
            continue
        cleaned[str(key)] = max(count, 0)
    return cleaned


def _build_skip_chart(skips: Dict[str, int]):
    if not skips:
        return dbc.Alert("No skip data available", color="secondary", className="mb-0")

    try:
        reasons = list(skips.keys())
        counts = [skips[r] for r in reasons]
        colors = ["#f0ad4e" if reason == "TIME_WINDOW" else "#17a2b8" for reason in reasons]
        figure = {
            "data": [
                {
                    "type": "bar",
                    "x": reasons,
                    "y": counts,
                    "marker": {"color": colors},
                }
            ],
            "layout": {
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "font": {"color": "#f8f9fa"},
                "margin": {"l": 40, "r": 10, "t": 10, "b": 40},
                "height": 220,
            },
        }
        return dcc.Graph(id="ops-summary-skip-chart", figure=figure, config={"displayModeBar": False})
    except Exception as exc:  # pragma: no cover - defensive rendering
        return dbc.Alert(f"Skip chart unavailable: {exc}", color="warning", className="mb-0")


def _build_ops_summary(metrics: Dict[str, Any], exec_metrics: Dict[str, Any]) -> dbc.Card:
    pipeline_rc = metrics.get("rc")
    pipeline_status = "ok"
    if pipeline_rc not in (None, 0):
        pipeline_status = "error"
    if not metrics:
        pipeline_status = "warn"

    skip_counts = _extract_skip_counts(
        exec_metrics.get("skips") or exec_metrics.get("skip_reasons")  # type: ignore[arg-type]
    )
    exec_status_val = str(exec_metrics.get("status", "")).lower()
    executor_status = "ok"
    if exec_metrics.get("auth_ok") is False or exec_status_val in {"error", "failed", "fail"}:
        executor_status = "error"
    elif skip_counts.get("TIME_WINDOW", 0) > 0:
        executor_status = "warn"

    last_screener = _format_time(metrics.get("timestamp") or metrics.get("last_run_utc"))
    universe = metrics.get("symbols_in") or metrics.get("symbols") or metrics.get("universe_count")
    rows_out = metrics.get("rows_out") or metrics.get("rows")

    run_started = _format_time(exec_metrics.get("run_started_utc") or exec_metrics.get("started_utc"))
    run_finished = _format_time(exec_metrics.get("run_finished_utc") or exec_metrics.get("timestamp"))
    fills = exec_metrics.get("fills", exec_metrics.get("orders_filled"))

    in_window_raw = exec_metrics.get("in_window")
    if in_window_raw is None:
        time_window_badge = _status_badge("Window n/a", "warn")
    else:
        in_window = bool(in_window_raw)
        time_window_badge = _status_badge(
            "In window" if in_window else "Out of window", "ok" if in_window else "warn"
        )

    skip_items = [f"{reason}: {count}" for reason, count in sorted(skip_counts.items(), key=lambda kv: kv[1], reverse=True)]
    skip_block = html.Ul([html.Li(text) for text in skip_items]) if skip_items else html.Div("No skips")

    status_row = html.Div(
        [
            _status_badge("Pipeline OK" if pipeline_status == "ok" else "Pipeline issue", pipeline_status),
            _status_badge("Executor OK" if executor_status == "ok" else "Executor attention", executor_status),
            time_window_badge,
        ],
        id="ops-summary-status",
        className="mb-3",
    )

    pipeline_body = dbc.Card(
        dbc.CardBody(
            [
                html.H6("Pipeline", className="card-title"),
                html.Div([html.Strong("Last run: "), last_screener]),
                html.Div([html.Strong("Universe: "), str(universe) if universe is not None else "n/a"]),
                html.Div([html.Strong("Rows out: "), str(rows_out) if rows_out is not None else "n/a"]),
            ]
        ),
        className="mb-3",
    )

    executor_body = dbc.Card(
        dbc.CardBody(
            [
                html.H6("Executor", className="card-title"),
                html.Div([html.Strong("Started: "), run_started]),
                html.Div([html.Strong("Finished: "), run_finished]),
                html.Div([html.Strong("Auth OK: "), str(exec_metrics.get("auth_ok", "n/a"))]),
                html.Div([html.Strong("Orders submitted: "), str(exec_metrics.get("orders_submitted", "n/a"))]),
                html.Div([html.Strong("Fills: "), str(fills if fills is not None else "n/a")]),
                html.Div([html.Strong("Open positions: "), str(exec_metrics.get("open_positions", "n/a"))]),
                html.Div([html.Strong("Allowed new positions: "), str(exec_metrics.get("allowed_new_positions", "n/a"))]),
                html.Div([html.Strong("Exit reason: "), str(exec_metrics.get("exit_reason", "n/a"))]),
                html.Div([html.Strong("Skips: "), skip_block]),
            ]
        ),
        className="mb-3",
    )

    skip_chart = _build_skip_chart(skip_counts)

    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Ops Summary", className="card-title"),
                status_row,
                dbc.Row(
                    [
                        dbc.Col(pipeline_body, md=6),
                        dbc.Col(executor_body, md=6),
                    ]
                ),
                html.Div(
                    [html.H6("Skip breakdown", className="card-title"), skip_chart],
                    className="mt-2",
                ),
            ]
        ),
        className="mb-3",
    )


def overview_layout():
    metrics = _read_screener_metrics()
    exec_metrics = _read_execute_metrics()
    candidates_rows, top_candidates = _read_candidates()
    ml_status = _read_nightly_ml_status()
    ranker_eval_path = _read_ranker_eval_path()
    timeline = _build_today_timeline()
    ops_summary = _build_ops_summary(metrics, exec_metrics)

    pipeline_items = [
        f"Last run: {_format_time(metrics.get('timestamp') or metrics.get('last_run_utc'))}",
        f"RC: {metrics.get('rc', 'N/A')}",
        f"Symbols in: {metrics.get('symbols_in', 'N/A')}",
        f"With bars: {metrics.get('with_bars', metrics.get('symbols_with_bars', 'N/A'))}",
        f"Rows: {metrics.get('rows', 'N/A')}",
        f"Bars rows total: {metrics.get('bars_rows_total', 'N/A')}",
        f"Source: {metrics.get('source', 'screener')}",
    ]

    candidates_items = [f"Rows: {candidates_rows if candidates_rows is not None else '(missing)'}"]
    if top_candidates:
        for row in top_candidates:
            symbol = row.get("symbol")
            score = row.get("score")
            bonus = row.get("model_score_5d")
            suffix = f" (model_5d={bonus})" if bonus is not None else ""
            candidates_items.append(f"{symbol}: {score}{suffix}")

    execute_items = [
        f"Last execute: {_format_time(exec_metrics.get('run_finished_utc') or exec_metrics.get('timestamp'))}",
        f"Auth OK: {exec_metrics.get('auth_ok')}",
        f"Orders submitted: {exec_metrics.get('orders_submitted')}",
        f"Fills: {exec_metrics.get('fills')}",
        f"Trails: {exec_metrics.get('trails')}",
        f"Duration: {exec_metrics.get('duration_sec')}",
    ]
    skips = exec_metrics.get("skips") or exec_metrics.get("skip_reasons") or {}
    if isinstance(skips, dict):
        sorted_skips = sorted(skips.items(), key=lambda kv: kv[1], reverse=True)
        for reason, count in sorted_skips[:3]:
            execute_items.append(f"Skip {reason}: {count}")

    ml_items = [
        f"Written at: {_format_time(ml_status.get('written_at'))}",
    ]
    for key in ["bars", "features", "labels", "model", "predictions", "eval"]:
        if key in ml_status:
            ml_items.append(f"{key}: {_format_time(ml_status.get(key))}")
    if ranker_eval_path:
        ml_items.append(f"Ranker eval: {ranker_eval_path.name}")

    alerts = _compute_alerts(metrics, exec_metrics, candidates_rows)

    alerts_block = html.Div(
        alerts if alerts else [dbc.Badge("Healthy", color="success")],
        className="mb-3",
    )

    return html.Div(
        [
            alerts_block,
            ops_summary,
            dbc.Row(
                [
                    dbc.Col(_build_card("Pipeline", pipeline_items), md=6),
                    dbc.Col(_build_card("Candidates", candidates_items), md=6),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(_build_card("Execution", execute_items), md=6),
                    dbc.Col(_build_card("ML", ml_items), md=6),
                ]
            ),
            timeline,
            html.H5("Recent Token Events"),
            html.Pre(
                safe_tail_text(LOG_DIR / "pipeline.log", 50),
                style={"maxHeight": "300px", "overflow": "auto"},
            ),
        ]
    )
