from __future__ import annotations

import json
import logging
import os
import pathlib
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dash_table import DataTable
from dash.dependencies import Input, Output

from dashboards.utils import (
    coerce_kpi_types,
    parse_pipeline_summary,
    safe_read_json,
    safe_read_metrics_csv,
)

LOGGER = logging.getLogger(__name__)

_WARNED_PATHS: set[tuple[str, pathlib.Path]] = set()


def _environment_state() -> tuple[bool, str]:
    base_url = (os.getenv("APCA_API_BASE_URL") or "").lower()
    paper = "paper-api" in base_url
    if not paper:
        flag = (os.getenv("JBR_EXEC_PAPER") or "").strip().lower()
        paper = flag in {"1", "true", "yes", "on"}
    feed = (os.getenv("ALPACA_DATA_FEED") or "").strip().upper() or "?"
    return paper, feed


def _detect_paper_mode() -> bool:
    paper, _ = _environment_state()
    return paper


PAPER_TRADING_MODE = _detect_paper_mode()


def _paper_badge_component() -> html.Span:
    paper, feed = _environment_state()
    label = f"{'Paper' if paper else 'Live'} ({feed})"
    bg_color = "#cfe2ff" if paper else "#d1e7dd"
    text_color = "#084298" if paper else "#0f5132"
    return html.Span(
        label,
        style={
            "display": "inline-block",
            "background": bg_color,
            "color": text_color,
            "padding": "2px 10px",
            "borderRadius": "999px",
            "fontSize": "0.7rem",
            "fontWeight": 600,
            "letterSpacing": "0.04em",
        },
    )


def _warn_once(kind: str, path: pathlib.Path, message: str, *args) -> None:
    """Emit a warning once per (kind, path) to avoid log spam."""

    key = (kind, path)
    if key in _WARNED_PATHS:
        return
    _WARNED_PATHS.add(key)
    LOGGER.warning(message, *args)


def _clear_warnings(path: pathlib.Path, *kinds: str) -> None:
    for kind in kinds:
        _WARNED_PATHS.discard((kind, path))


def _resolve_repo_root() -> pathlib.Path:
    """Resolve the repository root while tolerating bad env overrides."""

    default_root = pathlib.Path(__file__).resolve().parents[1]
    env_home = os.environ.get("JBRAVO_HOME")
    if env_home:
        try:
            candidate = pathlib.Path(env_home).expanduser()
            if not candidate.exists():
                LOGGER.warning(
                    "JBRAVO_HOME=%s not found; continuing with provided path", candidate
                )
            return candidate
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning(
                "Failed to resolve JBRAVO_HOME=%s (%s); falling back to %s",
                env_home,
                exc,
                default_root,
            )
    return default_root


REPO_ROOT = _resolve_repo_root()
DATA_DIR = REPO_ROOT / "data"
LOG_DIR = REPO_ROOT / "logs"
HEALTH_JSON = DATA_DIR / "health" / "connectivity.json"

METRICS_JSON = DATA_DIR / "screener_metrics.json"
TOP_CSV = DATA_DIR / "top_candidates.csv"
SCORED_CSV = DATA_DIR / "scored_candidates.csv"
HIST_CSV = DATA_DIR / "screener_metrics_history.csv"
PRED_LATEST = DATA_DIR / "predictions" / "latest.csv"
RANKER_EVAL_LATEST = DATA_DIR / "ranker_eval" / "latest.json"
METRICS_SUMMARY_CSV = DATA_DIR / "metrics_summary.csv"
EXECUTE_METRICS_JSON = DATA_DIR / "execute_metrics.json"
LATEST_CSV = DATA_DIR / "latest_candidates.csv"
PREMARKET_JSON = DATA_DIR / "last_premarket_run.json"


_KPI_LAST_SOURCE: str | None = None


def _normalize_metrics(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "last_run_utc": None,
        "symbols_in": None,
        "symbols_with_bars": None,
        "bars_rows_total": None,
        "rows": None,
    }
    if isinstance(payload, Mapping):
        base.update(payload)
    coerced = coerce_kpi_types(base)
    base.update(coerced)
    return base


def _log_kpi_source(source: str) -> None:
    global _KPI_LAST_SOURCE
    if source != _KPI_LAST_SOURCE:
        LOGGER.info("KPI source %s", source)
        _KPI_LAST_SOURCE = source


def load_kpis() -> dict[str, Any]:
    json_metrics = _normalize_metrics(safe_read_json(METRICS_JSON))
    if json_metrics.get("symbols_in") is not None:
        json_metrics["source"] = "screener_metrics.json"
        json_metrics["_kpi_inferred_from_log"] = False
        _log_kpi_source(json_metrics["source"])
        return json_metrics

    csv_metrics = _normalize_metrics(safe_read_metrics_csv(METRICS_SUMMARY_CSV))
    if csv_metrics.get("symbols_in") is not None:
        csv_metrics["source"] = "metrics_summary.csv"
        csv_metrics["_kpi_inferred_from_log"] = False
        _log_kpi_source(csv_metrics["source"])
        return csv_metrics

    pipeline_metrics = _normalize_metrics(parse_pipeline_summary(LOG_DIR / "pipeline.log"))
    if pipeline_metrics.get("symbols_in") is not None:
        pipeline_metrics["source"] = "pipeline.log (inferred)"
        pipeline_metrics["_kpi_inferred_from_log"] = True
        _log_kpi_source(pipeline_metrics["source"])
        return pipeline_metrics

    empty = _normalize_metrics({})
    empty["source"] = "none"
    empty["_kpi_inferred_from_log"] = False
    _log_kpi_source(empty["source"])
    return empty


def _format_probe_timestamp(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "n/a"
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except Exception:  # pragma: no cover - tolerate malformed timestamps
        return raw


def _status_row(label: str, payload: Mapping[str, Any]) -> html.Div:
    info = payload if isinstance(payload, Mapping) else {}
    ok = bool(info.get("ok"))
    icon = "✅" if ok else "❌"
    status = info.get("status")
    status_text = "n/a" if status in (None, "") else str(status)
    message = str(info.get("message") or "").strip()
    return html.Div(
        [
            html.Div(
                f"{icon} {label} ({status_text})",
                style={"fontWeight": 600, "marginBottom": "4px"},
            ),
            html.Div(
                message or "(no details)",
                style={
                    "fontSize": "12px",
                    "color": "#bfc3d9" if ok else "#ffd7d5",
                    "whiteSpace": "pre-wrap",
                    "wordBreak": "break-word",
                },
            ),
        ],
        className="sh-health-row",
        style={"padding": "4px 0"},
    )


def _health_elements(health: Mapping[str, Any] | None) -> tuple[html.Div, html.Div | None]:
    trading = health.get("trading") if isinstance(health, Mapping) else {}
    data = health.get("data") if isinstance(health, Mapping) else {}
    trading = trading if isinstance(trading, Mapping) else {}
    data = data if isinstance(data, Mapping) else {}

    ts = _format_probe_timestamp(health.get("ts_utc") if isinstance(health, Mapping) else None)

    health_card = html.Div(
        [
            html.Div("Alpaca Connectivity", className="sh-kpi-title"),
            html.Div(
                [
                    _status_row("Trading", trading),
                    _status_row("Data", data),
                ],
                style={"display": "grid", "gap": "6px"},
            ),
            html.Div(f"Last probe: {ts}", className="sh-kpi-sub"),
        ],
        className="sh-kpi sh-health-card",
        style={"gridColumn": "1 / span 2", "background": "#1e2235"},
    )

    banner = None
    issues: list[str] = []
    if not trading.get("ok", False):
        trading_msg = str(trading.get("message") or "no response").strip()
        issues.append(f"Trading ({trading.get('status', 'n/a')}): {trading_msg}")
    if not data.get("ok", False):
        data_msg = str(data.get("message") or "no response").strip()
        issues.append(f"Data ({data.get('status', 'n/a')}): {data_msg}")

    if issues:
        banner = html.Div(
            [
                html.Div("Alpaca Connectivity Issue", style={"fontWeight": 600, "marginBottom": "4px"}),
                *[html.Div(issue, style={"fontSize": "13px", "marginBottom": "2px"}) for issue in issues],
            ],
            style={
                "position": "sticky",
                "top": 0,
                "zIndex": 1100,
                "padding": "10px 14px",
                "borderRadius": "6px",
                "background": "#8c1d13",
                "color": "#fff",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.35)",
                "marginBottom": "12px",
            },
            className="sh-health-alert",
        )

    return health_card, banner


def _premarket_pill(payload: Mapping[str, Any] | None) -> html.Div:
    info = payload if isinstance(payload, Mapping) else {}
    in_window = bool(info.get("in_window"))
    auth_ok = bool(info.get("auth_ok"))
    candidates_raw = info.get("candidates_in", 0)
    try:
        candidates = int(candidates_raw)
    except (TypeError, ValueError):
        candidates = 0
    started = _format_probe_timestamp(info.get("started_utc"))
    ny_now = _format_probe_timestamp(info.get("ny_now"))

    chip_style = {
        "background": "#2a3145",
        "padding": "2px 8px",
        "borderRadius": "999px",
    }
    chips = [
        html.Span(
            f"Window {'✅' if in_window else '❌'}",
            style=chip_style,
        ),
        html.Span(
            f"Auth {'✅' if auth_ok else '❌'}",
            style=chip_style,
        ),
        html.Span(
            f"Candidates {candidates:,}",
            style=chip_style,
        ),
    ]

    sub_parts: list[str] = []
    if started and started != "n/a":
        sub_parts.append(f"Started {started}")
    if ny_now and ny_now != "n/a":
        sub_parts.append(f"NY {ny_now}")
    sub_text = " • ".join(sub_parts) if sub_parts else "Awaiting latest run"

    return html.Div(
        [
            html.Div("Pre-market readiness", className="sh-kpi-title"),
            html.Div(
                chips,
                style={
                    "display": "flex",
                    "gap": "8px",
                    "flexWrap": "wrap",
                    "fontSize": "13px",
                },
            ),
            html.Div(sub_text, className="sh-kpi-sub"),
        ],
        className="sh-kpi sh-premarket-pill",
        style={
            "background": "#1e2734",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "display": "grid",
            "gap": "6px",
        },
    )


def _safe_json(path: pathlib.Path) -> dict:
    try:
        if path.exists():
            contents = json.loads(path.read_text())
            if not isinstance(contents, dict):
                _warn_once(
                    "json_type",
                    path,
                    "JSON artifact %s did not contain an object; defaulting to empty dict",
                    path,
                )
                return {}
            _clear_warnings(path, "json_missing", "json_error", "json_type")
            return contents
        _warn_once("json_missing", path, "JSON artifact missing at %s", path)
    except Exception as exc:  # pragma: no cover - defensive
        _warn_once("json_error", path, "Failed to read JSON artifact %s: %s", path, exc)
    return {}


def _safe_csv(path: pathlib.Path, nrows: int | None = None) -> pd.DataFrame:
    try:
        if path.exists():
            df = pd.read_csv(path, nrows=nrows)
            _clear_warnings(path, "csv_missing", "csv_error")
            return df
        _warn_once("csv_missing", path, "CSV artifact missing at %s", path)
    except Exception as exc:  # pragma: no cover - defensive
        _warn_once("csv_error", path, "Failed to read CSV artifact %s: %s", path, exc)
    return pd.DataFrame()


def _extract_last_line(text: str, token: str) -> str:
    if not isinstance(text, str):
        return ""
    for line in reversed(text.splitlines()):
        if token in line:
            return line.strip()
    return ""


def _parse_pipeline_summary(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        return {}
    for line in reversed(text.splitlines()):
        if "PIPELINE_SUMMARY" not in line:
            continue
        tail = line.split("PIPELINE_SUMMARY", 1)[1].strip()
        summary: dict[str, Any] = {}
        for part in tail.split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            parsed: Any = value
            try:
                parsed = json.loads(value)
            except Exception:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
            summary[key] = parsed
        return summary
    return {}


def _tail(path: pathlib.Path, lines: int = 200) -> str:
    try:
        if not path.exists():
            _warn_once("log_missing", path, "Log file missing at %s", path)
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
            _clear_warnings(path, "log_missing", "log_error")
            return "\n".join(text.splitlines()[-lines:])
    except Exception as exc:  # pragma: no cover - defensive
        _warn_once("log_error", path, "Failed to read log %s: %s", path, exc)
        return f"(log read error: {exc})"


def _placeholder_figure(title: str, message: str = "Not computed yet") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(color="#adb5bd", size=14),
    )
    return fig

def _why_from_breakdown(json_str: str) -> str:
    try:
        d = json.loads(json_str) if isinstance(json_str, str) else {}
    except Exception:
        d = {}
    labels = {"TS":"Trend","MS":"MA Stack","BP":"Near 20D High","PT":"Pullback Tight",
              "RSI":"RSI","MH":"MACD Hist","ADX":"ADX","AROON":"Aroon",
              "VCP":"VCP","VOLexp":"Vol Exp"}
    items = []
    for k, v in d.items():
        key = k.replace("_z","")
        if key in labels and isinstance(v, (int, float)):
            items.append((labels[key], float(v)))
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    parts = []
    for name, val in items[:4]:
        arrow = "▲" if val >= 0 else "▼"
        parts.append(f"{name} {arrow} {abs(val):.2f}")
    return " • ".join(parts) if parts else "n/a"

def _fmt_millions(x):
    try:
        x = float(x)
        return f"${x/1_000_000:.2f}M"
    except Exception:
        return "n/a"

def _mk_why_tooltip(row: dict) -> dict:
    """Return tooltip content for the Why column with resilient fallbacks."""

    row = row or {}
    breakdown_source = row.get("score_breakdown", {})
    if isinstance(breakdown_source, Mapping):
        breakdown_str = json.dumps(breakdown_source)
        raw_breakdown = breakdown_source
    else:
        breakdown_str = breakdown_source if isinstance(breakdown_source, str) else ""
        try:
            raw_breakdown = json.loads(breakdown_str) if breakdown_str else {}
        except Exception:
            raw_breakdown = {}

    normalized_breakdown: dict[str, float] = {}
    for key, value in (raw_breakdown or {}).items():
        base_key = str(key).replace("_z", "")
        try:
            normalized_breakdown[base_key] = float(value)
        except (TypeError, ValueError):
            continue

    contrib_str = _why_from_breakdown(breakdown_str)

    def _is_missing(value) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip() == "":
            return True
        try:
            if pd.isna(value):
                return True
        except Exception:  # pragma: no cover - pandas type quirks
            pass
        return False

    def _format_value(value, formatter):
        if _is_missing(value):
            return "n/a"
        if formatter is None:
            try:
                return str(value)
            except Exception:  # pragma: no cover - defensive
                return "n/a"
        try:
            return formatter(value)
        except Exception:
            try:
                return str(value)
            except Exception:  # pragma: no cover - defensive
                return "n/a"

    missing_keys: set[str] = set()

    def _tracked_value(value, key: str | None, formatter=None) -> str:
        if _is_missing(value):
            if key:
                missing_keys.add(key)
            return "n/a"
        return _format_value(value, formatter)

    score = row.get("Score")
    rsi14 = row.get("RSI14", row.get("RSI"))
    adx_raw = row.get("ADX_raw")
    use_fallback = _is_missing(adx_raw)
    adx = row.get("ADX") if use_fallback else adx_raw
    aup = row.get("AROON_UP")
    adn = row.get("AROON_DN")
    mh = row.get("MACD_HIST", row.get("MH"))
    atrp = row.get("ATR_pct")
    adv20 = row.get("ADV20")
    close = row.get("close")

    md: list[str] = []
    md.append(
        "**Score:** `"
        + _format_value(score, lambda v: "{:.3f}".format(float(v)))
        + "`"
    )
    if contrib_str and contrib_str != "n/a":
        md.append("**Contributions (z‑weighted):**")
        for part in contrib_str.split("•"):
            part = part.strip()
            if part:
                md.append(f"- {part}")
    else:
        md.append("_Contributions unavailable_")

    raw_lines = [
        f"- Close: `{_tracked_value(close, None, lambda v: '{:.2f}'.format(float(v)))}`",
        f"- RSI14: `{_tracked_value(rsi14, 'RSI', lambda v: '{:.2f}'.format(float(v)))}`",
        f"- ADX: `{_tracked_value(adx, 'ADX', lambda v: '{:.2f}'.format(float(v)))}`",
    ]

    up_str = _tracked_value(aup, None, lambda v: "{:.1f}".format(float(v)))
    dn_str = _tracked_value(adn, None, lambda v: "{:.1f}".format(float(v)))
    if up_str == "n/a" and dn_str == "n/a":
        raw_lines.append("- Aroon Up/Dn: `n/a`")
    else:
        raw_lines.append(f"- Aroon Up/Dn: `{up_str}` / `{dn_str}`")

    raw_lines.extend(
        [
            f"- MACD Hist: `{_tracked_value(mh, 'MH', lambda v: '{:.4f}'.format(float(v)))}`",
            f"- ATR%: `{_tracked_value(atrp, None, lambda v: '{:.2%}'.format(float(v)))}`",
            f"- ADV20: `{_tracked_value(adv20, None, _fmt_millions)}`",
        ]
    )

    has_raw_data = any(
        not _is_missing(val)
        for val in (close, rsi14, adx, aup, adn, mh, atrp, adv20)
    )
    if has_raw_data:
        md.append("**Raw indicators:**")
        md.extend(raw_lines)
    else:
        md.append("_Raw indicators unavailable_")

    fallback_labels = {
        "RSI": "RSI",
        "ADX": "ADX",
        "TS": "Trend",
        "MS": "MA Stack",
        "BP": "Near 20D High",
        "PT": "Pullback Tight",
    }
    fallback_pairs: list[str] = []
    if missing_keys or not has_raw_data:
        for key in ["RSI", "ADX", "TS", "MS", "BP", "PT"]:
            if key not in normalized_breakdown:
                continue
            value = normalized_breakdown[key]
            fallback_pairs.append(f"{fallback_labels[key]}={value:.2f}")
    if fallback_pairs:
        md.append("**Score breakdown metrics:**")
        md.append("`" + ", ".join(fallback_pairs[:6]) + "`")
    elif not has_raw_data:
        md.append("_Score breakdown metrics unavailable_")

    tooltips = {"Why": {"value": "\n".join(md), "type": "markdown"}}
    origin = str(row.get("origin") or "").strip().lower()
    if origin == "fallback":
        tooltips["origin"] = {
            "value": "Fallback candidate generated for resilience",
            "type": "markdown",
        }

    return tooltips

def build_layout():
    return html.Div(
        [
            dcc.Interval(id="sh-interval", interval=60*1000, n_intervals=0),  # auto-refresh each 60s
            dcc.Store(id="sh-metrics-store"),
            dcc.Store(id="sh-top-store"),
            dcc.Store(id="sh-hist-store"),
            dcc.Store(id="sh-eval-store"),
            dcc.Store(id="sh-health-store"),
            dcc.Store(id="sh-summary-store"),
            dcc.Store(id="sh-premarket-store"),
            html.Div(id="sh-paper-badge", className="mb-2"),
            html.Div(id="sh-health-banner"),
            html.Div(id="sh-kpis", className="sh-row"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Universe Prefix Counts"),
                            html.Pre(id="sh-prefix-counts", className="sh-metric-pre"),
                        ],
                        className="sh-metric-card",
                    ),
                    html.Div(
                        [
                            html.H4("HTTP"),
                            html.Pre(id="sh-http", className="sh-metric-pre"),
                        ],
                        className="sh-metric-card",
                    ),
                    html.Div(
                        [
                            html.H4("Cache"),
                            html.Pre(id="sh-cache", className="sh-metric-pre"),
                        ],
                        className="sh-metric-card",
                    ),
                    html.Div(
                        [
                            html.H4("Timings"),
                            html.Pre(id="sh-timings-json", className="sh-metric-pre"),
                        ],
                        className="sh-metric-card",
                    ),
                ],
                className="sh-row sh-metric-grid",
            ),
            html.Div(
                [
                    dcc.Graph(id="sh-gate-pressure", style={"height":"300px"}),
                    dcc.Graph(id="sh-coverage", style={"height":"300px"}),
                    dcc.Graph(id="sh-timings", style={"height":"300px"}),
                ],
                className="sh-row",
                style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"14px"},
            ),
            html.Hr(),
            html.H3("Top Candidates"),
            html.Div(id="sh-top-empty", className="mb-2", style={"color": "#bfc3d9", "fontSize": "13px"}),
            DataTable(
                id="sh-top-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX":"auto"},
                style_cell={"fontSize":"13px","whiteSpace":"nowrap",
                            "textOverflow":"ellipsis","maxWidth":220},
                tooltip_data=[],
                tooltip_duration=None,
                columns=[
                    {"name":"symbol","id":"symbol"},
                    {"name":"Origin","id":"origin","presentation":"markdown"},
                    {"name":"Score","id":"Score","type":"numeric","format":{"specifier":".3f"}},
                    {"name":"close","id":"close","type":"numeric","format":{"specifier":".2f"}},
                    {"name":"ADV20","id":"ADV20","type":"numeric","format":{"specifier":".0f"}},
                    {"name":"ATR%","id":"ATR_pct","type":"numeric","format":{"specifier":".2%"}},
                    {"name":"Why (top contributors)","id":"Why"},
                ],
                style_data_conditional=[
                    {
                        "if": {"column_id": "Score"},
                        "color": "#4DB6AC",
                        "fontWeight": "600",
                    },
                    {
                        "if": {"column_id": "origin", "filter_query": "{origin} contains 'fallback'"},
                        "backgroundColor": "#2b223a",
                        "color": "#f4d9ff",
                        "fontSize": "11px",
                        "fontWeight": "600",
                        "border": "1px solid rgba(244, 217, 255, 0.45)",
                        "textAlign": "center",
                        "borderRadius": "999px",
                        "padding": "0 10px",
                    },
                    {
                        "if": {"column_id": "origin"},
                        "color": "#8f9bb3",
                        "fontSize": "11px",
                        "textAlign": "center",
                    },
                ],
            ),
            html.Hr(),
            html.H3("Trends & Performance"),
            html.Div(
                [
                    dcc.Graph(id="sh-trend-rows", style={"height":"280px"}),
                    dcc.Graph(id="sh-deciles-hit", style={"height":"280px"}),
                    dcc.Graph(id="sh-deciles-ret", style={"height":"280px"}),
                ],
                style={"display":"grid","gridTemplateColumns":"2fr 1fr 1fr","gap":"14px"},
            ),
            html.Hr(),
            html.Details([
                html.Summary("Diagnostics & Logs"),
                html.Div([
                    html.Div([
                        html.H4("Latest Predictions (head)"),
                        html.Pre(id="sh-preds-head",
                                 style={"maxHeight":"220px","overflow":"auto",
                                        "background":"#0b0b0b","color":"#cfcfcf","padding":"8px"})
                    ], style={"gridColumn":"1 / span 2"}),
                    html.Div([
                        html.H4("Screener Log (tail)"),
                        html.Pre(id="sh-screener-log",
                                 style={"maxHeight":"220px","overflow":"auto",
                                        "background":"#0b0b0b","color":"#cfcfcf","padding":"8px"})
                    ]),
                    html.Div([
                        html.H4("Pipeline Log (tail)"),
                        html.Pre(id="sh-pipeline-log",
                                 style={"maxHeight":"220px","overflow":"auto",
                                        "background":"#0b0b0b","color":"#cfcfcf","padding":"8px"})
                    ]),
                    html.Div([
                        html.H4("Pipeline Summary"),
                        html.Pre(id="sh-pipeline-panel",
                                 style={"maxHeight":"220px","overflow":"auto",
                                        "background":"#0b0b0b","color":"#cfcfcf","padding":"8px"})
                    ]),
                ], style={"display":"grid","gridTemplateColumns":"2fr 1fr 1fr 1fr","gap":"12px"})
            ]),
        ],
        style={"padding":"12px"}
    )

def register_callbacks(app):
    @app.callback(
        Output("sh-metrics-store","data"),
        Output("sh-top-store","data"),
        Output("sh-hist-store","data"),
        Output("sh-eval-store","data"),
        Output("sh-health-store","data"),
        Output("sh-summary-store","data"),
        Output("sh-premarket-store","data"),
        Input("sh-interval","n_intervals")
    )
    def _load_artifacts(_n):
        m = dict(load_kpis())
        latest = _safe_csv(LATEST_CSV, nrows=1)
        fallback_active = False
        if not latest.empty:
            source_raw = str(latest.iloc[0].get("source") or "").strip().lower()
            fallback_active = source_raw.startswith("fallback")
        m["_fallback_active"] = fallback_active
        m["_latest_zero_rows"] = LATEST_CSV.exists() and latest.empty
        m["_kpi_inferred_from_log"] = m.get("source") == "pipeline.log (inferred)"

        top = _safe_csv(TOP_CSV)
        # augment top with ATR% and Why using scored CSV if ATR14/ADV20 missing
        if not top.empty:
            needed_cols = [
                "ATR14",
                "ADV20",
                "RSI14",
                "MACD_HIST",
                "AROON_UP",
                "AROON_DN",
                "ADX",
            ]
            missing = [c for c in needed_cols if c not in top.columns]
            if missing:
                scored = _safe_csv(SCORED_CSV)
                if not scored.empty:
                    join_keys = [
                        c
                        for c in ["symbol", "timestamp"]
                        if c in top.columns and c in scored.columns
                    ]
                    if not join_keys:
                        join_keys = [
                            c for c in ["symbol"] if c in top.columns and c in scored.columns
                        ]
                    indicator_cols = [c for c in needed_cols if c in scored.columns]
                    select_cols = list(dict.fromkeys(join_keys + indicator_cols))
                    if join_keys and select_cols:
                        use = scored.loc[:, select_cols].drop_duplicates(subset=join_keys)
                        if not use.empty:
                            top = top.merge(use, on=join_keys, how="left")
            if "ATR14" in top.columns and "close" in top.columns:
                atr_num = pd.to_numeric(top["ATR14"], errors="coerce")
                close_num = pd.to_numeric(top["close"], errors="coerce")
                atr_pct = atr_num / close_num
                atr_pct = atr_pct.replace([np.inf, -np.inf], pd.NA)
                top["ATR_pct"] = atr_pct.clip(lower=0)
            else:
                top["ATR_pct"] = pd.NA
            if "score_breakdown" in top.columns:
                top["Why"] = top.get("score_breakdown", "").apply(_why_from_breakdown)
            else:
                top["Why"] = "n/a"
            if "Why" in top.columns:
                top["Why"] = top["Why"].fillna("n/a")
            badge_markup = "<span class='sh-badge sh-badge-fallback'>fallback</span>"
            top["origin"] = "—"
            if "source" in top.columns:
                source_series = top["source"].astype("string").fillna("")
                mask = source_series.str.contains("fallback", case=False, na=False)
                top.loc[mask, "origin"] = badge_markup
            elif fallback_active:
                top["origin"] = badge_markup
            else:
                top["origin"] = "—"

        hist = _safe_csv(HIST_CSV)
        ev = _safe_json(RANKER_EVAL_LATEST)
        health = _safe_json(HEALTH_JSON)
        summary_df = _safe_csv(METRICS_SUMMARY_CSV, nrows=1)
        summary = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
        premarket = _safe_json(PREMARKET_JSON)

        return (
            m,
            (top.to_dict("records") if not top.empty else []),
            (hist.to_dict("records") if not hist.empty else []),
            ev,
            health,
            summary,
            premarket,
        )

    @app.callback(
        Output("sh-health-banner","children"),
        Output("sh-paper-badge","children"),
        Output("sh-kpis","children"),
        Output("sh-prefix-counts","children"),
        Output("sh-http","children"),
        Output("sh-cache","children"),
        Output("sh-timings-json","children"),
        Output("sh-gate-pressure","figure"),
        Output("sh-coverage","figure"),
        Output("sh-timings","figure"),
        Output("sh-top-table","data"),
        Output("sh-top-empty","children"),
        Output("sh-top-table","tooltip_data"),
        Output("sh-trend-rows","figure"),
        Output("sh-deciles-hit","figure"),
        Output("sh-deciles-ret","figure"),
        Output("sh-preds-head","children"),
        Output("sh-screener-log","children"),
        Output("sh-pipeline-log","children"),
        Output("sh-pipeline-panel","children"),
        Input("sh-metrics-store","data"),
        Input("sh-top-store","data"),
        Input("sh-hist-store","data"),
        Input("sh-eval-store","data"),
        Input("sh-health-store","data"),
        Input("sh-summary-store","data"),
        Input("sh-premarket-store","data"),
    )
    def _render(m, top_rows, hist_rows, ev, health, summary, premarket):
        if not isinstance(m, dict):
            m = {}
        else:
            m = dict(m)
        if not isinstance(top_rows, list):
            top_rows = []
        if not isinstance(hist_rows, list):
            hist_rows = []
        if not isinstance(ev, dict):
            ev = {}
        if not isinstance(health, dict):
            health = {}
        if not isinstance(summary, Mapping):
            summary = {}
        if not isinstance(premarket, Mapping):
            premarket = {}

        fallback_active = bool(m.get("_fallback_active"))
        env_paper, env_feed = _environment_state()
        paper_badge = _paper_badge_component()
        latest_zero = bool(m.get("_latest_zero_rows"))
        top_empty_message: Any = ""
        if latest_zero:
            top_empty_message = html.Span(
                [
                    "No candidates today; fallback may populate shortly. ",
                    html.A(
                        "View pipeline panel",
                        href="#sh-pipeline-panel",
                        style={"color": "#63b3ed"},
                    ),
                ]
            )
        exec_metrics = _safe_json(EXECUTE_METRICS_JSON)
        if not isinstance(exec_metrics, Mapping):
            exec_metrics = {}
        skip_payload: dict[str, int] = {}
        skips_view = exec_metrics.get("skips") if isinstance(exec_metrics, Mapping) else {}
        if isinstance(skips_view, Mapping):
            for key, value in skips_view.items():
                try:
                    skip_payload[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        time_window_skips = int(skip_payload.get("TIME_WINDOW", 0))

        # ---------------- KPIs ----------------
        def _card(title, value, sub=None):
            return html.Div([
                html.Div(title, className="sh-kpi-title"),
                html.Div(value, className="sh-kpi-value"),
                html.Div(sub or "", className="sh-kpi-sub")
            ], className="sh-kpi")

        health_card, health_banner = _health_elements(health)
        premarket_card = _premarket_pill(premarket)
        summary_status = str(summary.get("status") or "").strip().lower()
        auth_reason = summary.get("auth_reason") if isinstance(summary, Mapping) else ""
        missing_raw = summary.get("auth_missing") if isinstance(summary, Mapping) else ""
        if isinstance(missing_raw, str):
            missing_list = [part.strip() for part in missing_raw.replace(";", ",").split(",") if part.strip()]
        elif isinstance(missing_raw, (list, tuple, set)):
            missing_list = [str(item).strip() for item in missing_raw if str(item).strip()]
        else:
            missing_list = []
        hint_raw = summary.get("auth_hint") if isinstance(summary, Mapping) else {}
        if isinstance(hint_raw, str):
            try:
                hint_data = json.loads(hint_raw)
            except Exception:
                hint_data = {"raw": hint_raw}
        elif isinstance(hint_raw, Mapping):
            hint_data = hint_raw
        else:
            hint_data = {}
        creds_alert = None
        last_run_raw = str(m.get("last_run_utc") or "").strip()
        if not last_run_raw:
            try:
                if METRICS_JSON.exists():
                    ts = datetime.fromtimestamp(
                        METRICS_JSON.stat().st_mtime, tz=timezone.utc
                    )
                    last_run_raw = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                last_run_raw = ""
        if not last_run_raw:
            exec_last = exec_metrics.get("last_run_utc") if isinstance(exec_metrics, Mapping) else ""
            if isinstance(exec_last, str) and exec_last.strip():
                try:
                    last_run_raw = datetime.fromisoformat(
                        exec_last.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d %H:%M:%S UTC")
                except Exception:
                    last_run_raw = exec_last
        last_run = last_run_raw or "n/a"
        sym_in = int(m.get("symbols_in", 0) or 0)
        if sym_in == 0 and isinstance(exec_metrics, Mapping):
            try:
                sym_in_exec = int(exec_metrics.get("symbols_in", 0) or 0)
            except (TypeError, ValueError):
                sym_in_exec = 0
            if sym_in_exec:
                sym_in = sym_in_exec
        sym_bars = int(m.get("symbols_with_bars", 0) or 0)
        bars_tot = int(m.get("bars_rows_total", 0) or 0)
        rows     = int(m.get("rows", 0) or 0)
        candidate_reason = str(
            (m.get("candidate_reason") or m.get("status") or "")
        ).strip().upper()
        candidate_sub = ""
        if rows == 0 and candidate_reason:
            candidate_sub = f"reason: {candidate_reason}"
        if fallback_active:
            note = "fallback"
            candidate_sub = f"{candidate_sub} | {note}" if candidate_sub else note
        zero_banner = None
        if rows == 0:
            zero_banner = html.Div(
                "Zero candidates in latest screener output.",
                style={
                    "background": "#1e2734",
                    "color": "#d8dee9",
                    "padding": "8px 12px",
                    "borderRadius": "6px",
                    "fontSize": "13px",
                },
            )
        sym_in_text = f"{sym_in:,}"
        sym_bars_text = f"{sym_bars:,}"
        bars_tot_text = f"{bars_tot:,}"
        rows_text = f"{rows:,}"
        bars_pct_value = 0.0
        if sym_in > 0:
            bars_pct_value = (sym_bars / max(sym_in, 1)) * 100
        bars_pct_text = f"{bars_pct_value:.1f}%"
        kpi_cards = [
            health_card,
            premarket_card,
            _card("Last Run (UTC)", last_run),
            _card("Symbols In", sym_in_text),
            _card("With Bars", sym_bars_text, bars_pct_text),
            _card("Bar Rows", bars_tot_text),
            _card("Candidates", rows_text, candidate_sub or None),
        ]
        kpi_grid = html.Div(
            kpi_cards,
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(7, minmax(140px,1fr))",
                "gap": "10px",
            },
            className="sh-kpi-grid",
        )
        source_label = str(m.get("source") or "none")
        env_label = "Paper" if env_paper else "Live"
        caption_children: list[Any] = [
            html.Span(
                f"Source: {source_label} • {env_label}: {env_feed}",
                style={"marginRight": "4px"},
            )
        ]
        if m.get("_kpi_inferred_from_log"):
            caption_children.append(
                html.Span(
                    "(inferred from pipeline log)",
                    style={"fontStyle": "italic", "color": "#cbd5f5"},
                )
            )
        kpi_caption = html.Div(
            caption_children,
            style={
                "fontSize": "12px",
                "color": "#9aa0b8",
                "marginTop": "6px",
            },
            className="sh-kpi-caption",
        )
        kpis = html.Div(
            [kpi_grid, kpi_caption],
            className="sh-kpi-wrap",
            style={"marginBottom": "12px", "display": "grid", "gap": "4px"},
        )

        banner_parts = []
        if health_banner is not None:
            banner_parts.append(health_banner)
        if zero_banner is not None:
            banner_parts.append(zero_banner)
        if banner_parts:
            health_banner = html.Div(
                banner_parts,
                style={"display": "grid", "gap": "10px", "marginBottom": "12px"},
            )
        else:
            health_banner = html.Div()

        # ---------------- Gate pressure ----------------
        fail = m.get("gate_fail_counts", {}) or {}
        if fail:
            df_fail = pd.DataFrame({"gate": list(fail.keys()), "count": list(fail.values())})
            df_fail = df_fail.sort_values("count", ascending=False)
            fig_gates = px.bar(df_fail, x="gate", y="count", title="Gate Pressure (Failures by Gate)")
        else:
            fig_gates = px.bar(title="Gate Pressure (no data)")

        # ---------------- Coverage donut ----------------
        cov_values = [sym_bars, max(sym_in - sym_bars, 0)]
        if sum(cov_values) > 0:
            cov_df = pd.DataFrame({"label": ["With Bars", "No Bars"], "value": cov_values})
            fig_cov = px.pie(
                cov_df, names="label", values="value", title="Universe Coverage", hole=0.45
            )
        else:
            fig_cov = px.pie(title="Universe Coverage (no data)")

        # ---------------- Timings ----------------
        t = m.get("timings", {}) or {}
        tm_df = pd.DataFrame({"stage":["fetch","features","rank","gates"],
                              "secs":[t.get("fetch_secs",0), t.get("feature_secs",0),
                                      t.get("rank_secs",0), t.get("gates_secs",0)]})
        fig_tm = px.bar(tm_df, x="stage", y="secs", title="Stage Timings (sec)")

        # ---------------- Top table ----------------
        top_data = top_rows or []
        for entry in top_data:
            origin_val = entry.get("origin")
            if origin_val in (None, ""):
                entry["origin"] = "—"
        # Build tooltip_data aligned with top_data rows
        tooltips = []
        for r in (top_data or []):
            try:
                tooltips.append(_mk_why_tooltip(r or {}))
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("Failed to build tooltip markdown: %s", exc)
                tooltips.append({})

        # ---------------- Trend (Rows & With-Bars %) ----------------
        if hist_rows:
            hist = pd.DataFrame(hist_rows).copy()
            # accept run_utc either ISO or naive
            hist["run_utc"] = pd.to_datetime(hist["run_utc"], errors="coerce", utc=True)
            hist = hist.sort_values("run_utc").tail(14)
            hist["with_bars_pct"] = (hist["symbols_with_bars"] / hist["symbols_in"].replace(0, pd.NA)).astype(float)
            # Build a tidy frame for two series
            tidy = pd.DataFrame({
                "run_utc": pd.concat([hist["run_utc"], hist["run_utc"]], ignore_index=True),
                "value":   pd.concat([hist["rows"], hist["with_bars_pct"]], ignore_index=True),
                "series":  ["Rows"]*len(hist) + ["With Bars %"]*len(hist)
            })
            fig_trend = px.line(tidy, x="run_utc", y="value", color="series",
                                markers=True, title="14‑Day Trend: Candidates & With‑Bars %")
            # nicer y-axis for percent series (auto is fine, we keep simple)
        else:
            fig_trend = px.line(title="14‑Day Trend: (insufficient history)")

        # ---------------- Deciles (Hit‑rate & Avg Return) ----------------
        hit_fig = _placeholder_figure("Decile Hit‑Rate")
        ret_fig = _placeholder_figure("Decile Avg Return")
        if ev and isinstance(ev, dict):
            dec = ev.get("deciles") or {}
            # expected: {"rank_decile":[1..10], "hit_rate":[...], "avg_return":[...], "count":[...]}
            try:
                df_dec = pd.DataFrame(dec)
                if not df_dec.empty and set(["rank_decile","hit_rate","avg_return"]).issubset(df_dec.columns):
                    df_dec = df_dec.sort_values("rank_decile")
                    hit_fig = px.bar(df_dec, x="rank_decile", y="hit_rate",
                                     title=f"Decile Hit‑Rate (k={int(ev.get('population',0))})")
                    ret_fig = px.bar(df_dec, x="rank_decile", y="avg_return",
                                     title="Decile Avg Return")
                    for fig in (hit_fig, ret_fig):
                        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
            except Exception:
                pass

        # ---------------- Predictions head & logs ----------------
        preds = _safe_csv(PRED_LATEST, nrows=8)
        preds_head = preds.to_csv(index=False) if not preds.empty else "(Not computed yet)"
        s_tail = _tail(LOG_DIR / "screener.log", 180)
        p_tail = _tail(LOG_DIR / "pipeline.log", 180)
        pipeline_summary = _parse_pipeline_summary(p_tail)
        skip_line = _extract_last_line(p_tail, "EXECUTE_SKIP_NO_CANDIDATES")
        panel_lines: list[str] = []
        token_labels = [
            "PIPELINE_START",
            "PIPELINE_SUMMARY",
            "FALLBACK_CHECK",
            "PIPELINE_END",
            "DASH RELOAD",
        ]
        pipeline_tokens = [
            _extract_last_line(p_tail, label) for label in token_labels
        ]
        for label, token_line in zip(token_labels, pipeline_tokens):
            panel_lines.append(token_line or f"{label} (missing)")
        def _try_float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

        if isinstance(pipeline_summary, Mapping) and pipeline_summary:
            try:
                sym_block = (
                    f"symbols_in={int(pipeline_summary.get('symbols_in', 0) or 0)} "
                    f"with_bars={int(pipeline_summary.get('with_bars', 0) or 0)} "
                    f"rows={int(pipeline_summary.get('rows', 0) or 0)}"
                )
            except Exception:
                sym_block = f"summary={json.dumps(pipeline_summary, sort_keys=True)}"
            panel_lines.append(sym_block)
            fetch_secs = _try_float(pipeline_summary.get("fetch_secs"))
            feature_secs = _try_float(pipeline_summary.get("feature_secs"))
            rank_secs = _try_float(pipeline_summary.get("rank_secs"))
            gate_secs = _try_float(pipeline_summary.get("gate_secs"))
            panel_lines.append(
                "timings "
                f"fetch_secs={fetch_secs:.2f} "
                f"feature_secs={feature_secs:.2f} "
                f"rank_secs={rank_secs:.2f} "
                f"gate_secs={gate_secs:.2f}"
            )
            step_rcs_view = pipeline_summary.get("step_rcs")
            if isinstance(step_rcs_view, Mapping) and step_rcs_view:
                panel_lines.append("step_rcs=" + json.dumps(step_rcs_view, sort_keys=True))
            elif isinstance(step_rcs_view, str) and step_rcs_view:
                panel_lines.append("step_rcs=" + step_rcs_view)
        if skip_line:
            panel_lines.append(skip_line)
        panel_lines.append(f"TIME_WINDOW skips={time_window_skips}")
        if skip_payload:
            panel_lines.append(
                "execute_skips=" + json.dumps(skip_payload, sort_keys=True)
            )
        try:
            pipeline_panel = "\n".join(panel_lines) if panel_lines else "(no data)"
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning("Failed to compose pipeline panel markdown: %s", exc)
            pipeline_panel = "(error rendering pipeline summary)"

        unauthorized_log = "ALPACA_UNAUTHORIZED" in (p_tail or "")
        if summary_status == "auth_error" or unauthorized_log:
            detail_blocks: list[Any] = []
            if missing_list:
                detail_blocks.append(
                    html.Div(
                        "Missing variables: " + ", ".join(missing_list),
                        style={"fontSize": "13px", "marginBottom": "4px"},
                    )
                )
            if auth_reason:
                detail_blocks.append(
                    html.Div(
                        f"Reason: {auth_reason}",
                        style={"fontSize": "13px", "marginBottom": "4px"},
                    )
                )
            hint_entries: list[Any] = []
            if isinstance(hint_data, Mapping):
                key_prefix = hint_data.get("key_prefix")
                if key_prefix:
                    hint_entries.append(
                        html.Div(
                            f"Key prefix: {key_prefix}",
                            style={"fontSize": "13px", "marginBottom": "2px"},
                        )
                    )
                secret_len = hint_data.get("secret_len")
                if secret_len is not None:
                    hint_entries.append(
                        html.Div(
                            f"Secret length: {secret_len}",
                            style={"fontSize": "13px", "marginBottom": "2px"},
                        )
                    )
                base_urls = hint_data.get("base_urls")
                if isinstance(base_urls, Mapping):
                    for label, url in base_urls.items():
                        hint_entries.append(
                            html.Div(
                                f"{str(label).title()} URL: {url or 'n/a'}",
                                style={"fontSize": "13px", "marginBottom": "2px"},
                            )
                        )
                if not hint_entries and hint_data:
                    hint_entries.append(
                        html.Div(
                            f"Hint: {json.dumps(hint_data, sort_keys=True)}",
                            style={"fontSize": "13px", "marginBottom": "2px"},
                        )
                    )
            alert_children: list[Any] = [
                html.Div("Credentials invalid", style={"fontWeight": 600, "marginBottom": "4px"}),
                *detail_blocks,
                *hint_entries,
                html.A(
                    "Troubleshooting guide",
                    href="https://docs.alpaca.markets/docs/troubleshooting-authentication",
                    target="_blank",
                    style={"color": "#ffd7d5", "textDecoration": "underline"},
                ),
            ]
            creds_alert = html.Div(
                alert_children,
                style={
                    "position": "sticky",
                    "top": 0,
                    "zIndex": 1100,
                    "padding": "10px 14px",
                    "borderRadius": "6px",
                    "background": "#8c1d13",
                    "color": "#fff",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.35)",
                    "marginBottom": "12px",
                },
                className="sh-health-alert",
            )

        prefix_raw = m.get("universe_prefix_counts") if isinstance(m, dict) else {}
        prefix_counts = (
            json.dumps(prefix_raw, indent=2, sort_keys=True)
            if isinstance(prefix_raw, dict) and prefix_raw
            else "(no data)"
        )

        http_view = m.get("http") if isinstance(m, dict) else {}
        if not isinstance(http_view, dict):
            http_view = {}
        http_json = json.dumps(http_view, indent=2, sort_keys=True) if http_view else "(no data)"

        cache_view = m.get("cache") if isinstance(m, dict) else {}
        if not isinstance(cache_view, dict):
            cache_view = {}
        cache_json = json.dumps(cache_view, indent=2, sort_keys=True) if cache_view else "(no data)"

        timings_dict = m.get("timings") if isinstance(m, dict) else {}
        if not isinstance(timings_dict, dict):
            timings_dict = {}
        base_timings = {"fetch_secs": 0, "feature_secs": 0, "rank_secs": 0, "gates_secs": 0}
        timings_view = {**base_timings, **timings_dict}
        timings_json = json.dumps({k: timings_view.get(k) for k in sorted(timings_view)}, indent=2)

        banner_children: list[Any] = []
        if creds_alert:
            banner_children.append(creds_alert)
        if health_banner:
            banner_children.append(health_banner)

        return (
            banner_children if banner_children else None,
            paper_badge,
            kpis,
            prefix_counts,
            http_json,
            cache_json,
            timings_json,
            fig_gates,
            fig_cov,
            fig_tm,
            top_data,
            top_empty_message,
            tooltips,
            fig_trend,
            hit_fig,
            ret_fig,
            preds_head,
            s_tail,
            p_tail,
            pipeline_panel,
        )
