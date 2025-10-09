from __future__ import annotations
import json, os, pathlib
from datetime import datetime
import pandas as pd
import plotly.express as px
from dash import html, dcc
from dash.dash_table import DataTable
from dash.dependencies import Input, Output

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR  = REPO_ROOT / "data"
LOG_DIR   = REPO_ROOT / "logs"

METRICS_JSON = DATA_DIR / "screener_metrics.json"
TOP_CSV      = DATA_DIR / "top_candidates.csv"
SCORED_CSV   = DATA_DIR / "scored_candidates.csv"
HIST_CSV     = DATA_DIR / "screener_metrics_history.csv"
PRED_LATEST  = DATA_DIR / "predictions" / "latest.csv"
RANKER_EVAL_LATEST = DATA_DIR / "ranker_eval" / "latest.json"

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
        if not path.exists(): return "(no log)"
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b''
            while size > 0 and data.count(b'\n') <= lines:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
            text = data.decode("utf-8", errors="ignore")
            return "\n".join(text.splitlines()[-lines:])
    except Exception as e:
        return f"(log read error: {e})"

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
    """
    Build a markdown tooltip for the 'Why' cell showing:
    - Score (final composite)
    - Top contributors (z-weighted)
    - Raw indicators (RSI14, ADX, AROON, MACD_HIST, ATR%, ADV20)
    Returns a dict mapping column-id -> {'value': md, 'type': 'markdown'}
    """
    # Contributions (same parsing used by Why string)
    contrib_str = _why_from_breakdown(row.get("score_breakdown",""))
    # Raw values with safe fallbacks
    score = row.get("Score", None)
    rsi14 = row.get("RSI14", row.get("RSI"))  # try raw first, fallback to z if needed
    adx   = row.get("ADX_raw", row.get("ADX"))  # allow a raw alias if available
    aup   = row.get("AROON_UP", None)
    adn   = row.get("AROON_DN", None)
    mh    = row.get("MACD_HIST", row.get("MH"))
    atrp  = row.get("ATR_pct", None)
    adv20 = row.get("ADV20", None)
    close = row.get("close", None)

    md = []
    if score is not None:
        try:
            md.append(f"**Score:** `{float(score):.3f}`")
        except Exception:
            md.append(f"**Score:** `{score}`")
    if contrib_str and contrib_str != "n/a":
        md.append("**Contributions (z‑weighted):**")
        # convert bullets nicely (already "• name ▲ 0.00")
        for part in contrib_str.split("•"):
            part = part.strip()
            if part:
                md.append(f"- {part}")
    else:
        md.append("_Contributions unavailable_")

    md.append("**Raw indicators:**")
    raw_lines = []
    if close is not None:
        try:
            raw_lines.append(f"- Close: `{float(close):.2f}`")
        except Exception:
            raw_lines.append(f"- Close: `{close}`")
    if rsi14 is not None:
        try:
            raw_lines.append(f"- RSI14: `{float(rsi14):.2f}`")
        except Exception:
            raw_lines.append(f"- RSI14: `{rsi14}`")
    if adx is not None:
        try:
            raw_lines.append(f"- ADX: `{float(adx):.2f}`")
        except Exception:
            raw_lines.append(f"- ADX: `{adx}`")
    if aup is not None and adn is not None:
        try:
            raw_lines.append(f"- Aroon Up/Dn: `{float(aup):.1f}` / `{float(adn):.1f}`")
        except Exception:
            raw_lines.append(f"- Aroon Up/Dn: `{aup}` / `{adn}`")
    elif aup is not None:
        try:
            raw_lines.append(f"- Aroon Up: `{float(aup):.1f}`")
        except Exception:
            raw_lines.append(f"- Aroon Up: `{aup}`")
    elif adn is not None:
        try:
            raw_lines.append(f"- Aroon Dn: `{float(adn):.1f}`")
        except Exception:
            raw_lines.append(f"- Aroon Dn: `{adn}`")
    if mh is not None:
        try:
            raw_lines.append(f"- MACD Hist: `{float(mh):.4f}`")
        except Exception:
            raw_lines.append(f"- MACD Hist: `{mh}`")
    if atrp is not None:
        try:
            raw_lines.append(f"- ATR%: `{float(atrp):.2%}`")
        except Exception:
            raw_lines.append(f"- ATR%: `{atrp}`")
    if adv20 is not None:
        raw_lines.append(f"- ADV20: `{_fmt_millions(adv20)}`")
    if not raw_lines:
        raw_lines.append("_No raw indicators available_")
    md.extend(raw_lines)

    return {"Why": {"value": "\n".join(md), "type": "markdown"}}

def build_layout():
    return html.Div(
        [
            dcc.Interval(id="sh-interval", interval=60*1000, n_intervals=0),  # auto-refresh each 60s
            dcc.Store(id="sh-metrics-store"),
            dcc.Store(id="sh-top-store"),
            dcc.Store(id="sh-hist-store"),
            dcc.Store(id="sh-eval-store"),
            html.Div(id="sh-kpis", className="sh-row"),
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
                    {"name":"Score","id":"Score","type":"numeric","format":{"specifier":".3f"}},
                    {"name":"close","id":"close","type":"numeric","format":{"specifier":".2f"}},
                    {"name":"ADV20","id":"ADV20","type":"numeric","format":{"specifier":".0f"}},
                    {"name":"ATR%","id":"ATR_pct","type":"numeric","format":{"specifier":".2%"}},
                    {"name":"Why (top contributors)","id":"Why"},
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
                ], style={"display":"grid","gridTemplateColumns":"2fr 1fr 1fr","gap":"12px"})
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
        Input("sh-interval","n_intervals")
    )
    def _load_artifacts(_n):
        m = _safe_json(METRICS_JSON)
        top = _safe_csv(TOP_CSV)
        # augment top with ATR% and Why using scored CSV if ATR14/ADV20 missing
        if not top.empty:
            if any(c not in top.columns for c in ["ATR14","ADV20","RSI14","MACD_HIST","AROON_UP","AROON_DN","ADX"]):
                scored = _safe_csv(SCORED_CSV)
                if not scored.empty:
                    use_cols = [c for c in ["symbol","timestamp","ATR14","ADV20","RSI14","MACD_HIST","AROON_UP","AROON_DN","ADX"]
                                if c in scored.columns]
                    use = scored[use_cols].drop_duplicates(["symbol","timestamp"]) if use_cols else pd.DataFrame()
                    if not use.empty:
                        top = top.merge(use, on=["symbol","timestamp"], how="left")
            if "ATR14" in top.columns and "close" in top.columns:
                top["ATR_pct"] = (top["ATR14"] / top["close"]).clip(lower=0)
            else:
                top["ATR_pct"] = 0.0
            top["Why"] = top.get("score_breakdown","").apply(_why_from_breakdown) if "score_breakdown" in top.columns else "n/a"

        hist = _safe_csv(HIST_CSV)
        ev = _safe_json(RANKER_EVAL_LATEST)

        return (m,
                (top.to_dict("records") if not top.empty else []),
                (hist.to_dict("records") if not hist.empty else []),
                ev)

    @app.callback(
        Output("sh-kpis","children"),
        Output("sh-gate-pressure","figure"),
        Output("sh-coverage","figure"),
        Output("sh-timings","figure"),
        Output("sh-top-table","data"),
        Output("sh-top-table","tooltip_data"),
        Output("sh-trend-rows","figure"),
        Output("sh-deciles-hit","figure"),
        Output("sh-deciles-ret","figure"),
        Output("sh-preds-head","children"),
        Output("sh-screener-log","children"),
        Output("sh-pipeline-log","children"),
        Input("sh-metrics-store","data"),
        Input("sh-top-store","data"),
        Input("sh-hist-store","data"),
        Input("sh-eval-store","data"),
    )
    def _render(m, top_rows, hist_rows, ev):
        # ---------------- KPIs ----------------
        def _card(title, value, sub=None):
            return html.Div([
                html.Div(title, className="sh-kpi-title"),
                html.Div(value, className="sh-kpi-value"),
                html.Div(sub or "", className="sh-kpi-sub")
            ], className="sh-kpi")
        last_run = (m.get("last_run_utc") or "n/a")
        sym_in   = int(m.get("symbols_in", 0) or 0)
        sym_bars = int(m.get("symbols_with_bars", 0) or 0)
        bars_tot = int(m.get("bars_rows_total", 0) or 0)
        rows     = int(m.get("rows", 0) or 0)
        kpis = html.Div([
            _card("Last Run (UTC)", last_run),
            _card("Symbols In", f"{sym_in:,}"),
            _card("With Bars", f"{sym_bars:,}", f"{(sym_bars/max(sym_in,1))*100:.1f}%"),
            _card("Bar Rows", f"{bars_tot:,}"),
            _card("Candidates", f"{rows:,}"),
        ], className="sh-kpi-wrap",
           style={"display":"grid","gridTemplateColumns":"repeat(5,1fr)","gap":"10px","marginBottom":"12px"})

        # ---------------- Gate pressure ----------------
        fail = m.get("gate_fail_counts", {}) or {}
        if fail:
            df_fail = pd.DataFrame({"gate": list(fail.keys()), "count": list(fail.values())})
            df_fail = df_fail.sort_values("count", ascending=False)
            fig_gates = px.bar(df_fail, x="gate", y="count", title="Gate Pressure (Failures by Gate)")
        else:
            fig_gates = px.bar(title="Gate Pressure (no data)")

        # ---------------- Coverage donut ----------------
        cov_df = pd.DataFrame({"label":["With Bars","No Bars"],
                               "value":[sym_bars, max(sym_in - sym_bars, 0)]})
        fig_cov = px.pie(cov_df, names="label", values="value", title="Universe Coverage", hole=0.45)

        # ---------------- Timings ----------------
        t = m.get("timings", {}) or {}
        tm_df = pd.DataFrame({"stage":["fetch","features","rank","gates"],
                              "secs":[t.get("fetch_secs",0), t.get("feature_secs",0),
                                      t.get("rank_secs",0), t.get("gates_secs",0)]})
        fig_tm = px.bar(tm_df, x="stage", y="secs", title="Stage Timings (sec)")

        # ---------------- Top table ----------------
        top_data = top_rows or []
        # Build tooltip_data aligned with top_data rows
        tooltips = []
        for r in (top_data or []):
            tooltips.append(_mk_why_tooltip(r or {}))

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
        hit_fig = px.bar(title="Decile Hit‑Rate (n/a)")
        ret_fig = px.bar(title="Decile Avg Return (n/a)")
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
            except Exception:
                pass

        # ---------------- Predictions head & logs ----------------
        preds = _safe_csv(PRED_LATEST, nrows=8)
        preds_head = preds.to_csv(index=False) if not preds.empty else "(no predictions yet)"
        s_tail = _tail(LOG_DIR / "screener.log", 180)
        p_tail = _tail(LOG_DIR / "pipeline.log", 180)

        return (kpis, fig_gates, fig_cov, fig_tm, top_data, tooltips,
                fig_trend, hit_fig, ret_fig,
                preds_head, s_tail, p_tail)
