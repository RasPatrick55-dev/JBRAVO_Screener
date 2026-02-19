from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from dash import dash_table, html

from dashboards.utils import file_stat, parse_pipeline_tokens, safe_tail_text

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"


def _artifact_rows() -> list[Dict[str, Any]]:
    targets = [
        DATA_DIR / "latest_candidates.csv",
        DATA_DIR / "screener_metrics.json",
        DATA_DIR / "execute_metrics.json",
        DATA_DIR / "nightly_ml_status.json",
        DATA_DIR / "ranker_eval" / "latest.json",
    ]
    preds_dir = DATA_DIR / "predictions"
    if preds_dir.exists():
        newest = sorted(preds_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if newest:
            targets.append(newest[0])
    targets.extend([LOG_DIR / "pipeline.log", LOG_DIR / "execute_trades.log"])
    rows: list[Dict[str, Any]] = []
    for path in targets:
        stat = file_stat(path)
        rows.append(
            {
                "path": str(path.relative_to(BASE_DIR)),
                "exists": stat.get("exists"),
                "mtime": stat.get("mtime_iso"),
                "size_bytes": stat.get("size_bytes"),
            }
        )
    return rows


def pipeline_layout():
    tokens = parse_pipeline_tokens(LOG_DIR / "pipeline.log")
    token_table = dash_table.DataTable(
        columns=[
            {"name": col, "id": col}
            for col in [
                "start_time",
                "end_time",
                "rc",
                "duration",
                "symbols_in",
                "with_bars",
                "rows",
                "bars_rows_total",
                "source",
            ]
        ],
        data=tokens,
        style_table={"overflowX": "auto"},
        page_size=10,
    )

    artifacts = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in ["path", "exists", "mtime", "size_bytes"]],
        data=_artifact_rows(),
        style_table={"overflowX": "auto"},
        page_size=10,
    )

    log_tail = safe_tail_text(LOG_DIR / "pipeline.log", 200)

    return html.Div(
        [
            html.H5("Pipeline tokens"),
            token_table,
            html.Hr(),
            html.H5("Artifact freshness"),
            artifacts,
            html.Hr(),
            html.H5("pipeline.log tail"),
            html.Pre(log_tail or "(no log)", style={"maxHeight": "400px", "overflow": "auto"}),
        ]
    )
