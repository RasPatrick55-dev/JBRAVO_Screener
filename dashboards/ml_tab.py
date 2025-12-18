from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, dcc, html

from dashboards.utils import list_recent_files, safe_read_json

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def _ml_status_rows() -> List[Dict[str, Any]]:
    payload = safe_read_json(DATA_DIR / "nightly_ml_status.json")
    rows: List[Dict[str, Any]] = []
    if payload:
        rows.append({"key": "written_at", "value": payload.get("written_at")})
        for key in ["bars", "features", "labels", "model", "predictions", "eval"]:
            rows.append({"key": key, "value": payload.get(key)})
    return rows


def _ranker_eval_rows() -> List[Dict[str, Any]]:
    latest = DATA_DIR / "ranker_eval" / "latest.json"
    if not latest.exists():
        return []
    payload = safe_read_json(latest)
    rows = []
    for key, value in payload.items():
        rows.append({"key": str(key), "value": json.dumps(value)[:200]})
    return rows


def _prediction_options() -> List[Dict[str, Any]]:
    files = list_recent_files(DATA_DIR / "predictions", "*.csv", limit=30)
    return [
        {"label": f.name, "value": str(f)}
        for f in files
    ]


def _prediction_preview(path_str: str | None) -> List[Dict[str, Any]]:
    if not path_str:
        return []
    path = Path(path_str)
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        if "score" in df.columns:
            df = df.sort_values("score", ascending=False)
        return df.head(50).to_dict(orient="records")
    except Exception:
        return []


def ml_layout(prediction_path: str | None = None):
    status_table = dash_table.DataTable(
        columns=[{"name": "key", "id": "key"}, {"name": "value", "id": "value"}],
        data=_ml_status_rows(),
        page_size=10,
        style_table={"overflowX": "auto"},
    )

    eval_table = dash_table.DataTable(
        columns=[{"name": "key", "id": "key"}, {"name": "value", "id": "value"}],
        data=_ranker_eval_rows(),
        page_size=10,
        style_table={"overflowX": "auto"},
    )

    preview = dash_table.DataTable(
        columns=[],
        data=_prediction_preview(prediction_path),
        page_size=50,
        style_table={"overflowX": "auto", "maxHeight": "500px", "overflowY": "auto"},
    )

    return html.Div(
        [
            html.H5("Nightly ML Status"),
            status_table,
            html.Hr(),
            html.H5("Ranker Eval"),
            eval_table,
            html.Hr(),
            html.H5("Predictions Browser"),
            dcc.Dropdown(id="predictions-dropdown", options=_prediction_options(), value=prediction_path),
            html.Div(preview, id="predictions-preview"),
        ]
    )

