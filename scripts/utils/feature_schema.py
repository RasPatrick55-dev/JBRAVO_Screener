"""Shared feature schema helpers for ML train/predict/recalibrate/refresh flows.

These helpers keep feature-column inference and schema signatures consistent
across the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

FEATURES_LATEST_META_NAME = "latest_meta.json"


def compute_feature_signature(feature_columns: Sequence[str]) -> str:
    """Return a stable hash for the ordered feature schema."""

    normalized: list[str] = []
    for value in feature_columns:
        column = str(value).strip()
        if not column:
            continue
        normalized.append(column)
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def infer_feature_columns_for_ml(
    df: pd.DataFrame,
    *,
    label_column: str | None = None,
    id_columns: Sequence[str] = ("symbol", "timestamp"),
    drop_columns: Sequence[str] = ("close",),
) -> list[str]:
    """Infer model feature columns from a frame using the canonical exclusions."""

    id_set = {str(v).strip() for v in id_columns if str(v).strip()}
    drop_set = {str(v).strip() for v in drop_columns if str(v).strip()}
    label_name = str(label_column).strip() if label_column else None
    numeric_cols = set(df.select_dtypes(include=["number", "bool"]).columns)
    resolved: list[str] = []
    for column in df.columns:
        name = str(column).strip()
        if not name:
            continue
        if name not in numeric_cols:
            continue
        if name in id_set or name in drop_set:
            continue
        if label_name and name == label_name:
            continue
        if name.startswith("label_") or name.startswith("fwd_ret_"):
            continue
        resolved.append(name)
    return resolved


def parse_features_run_date(features_path: Path | None) -> date | None:
    if features_path is None:
        return None
    match = re.search(r"features_(\d{4}-\d{2}-\d{2})", features_path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except Exception:
        return None


def per_file_meta_path(features_path: Path) -> Path:
    suffix = features_path.suffix or ""
    if suffix:
        return features_path.with_suffix(".meta.json")
    return features_path.with_name(f"{features_path.name}.meta.json")


def meta_matches_features_path(meta_payload: Mapping[str, Any], features_path: Path | None) -> bool:
    if features_path is None:
        return True
    expected_name = features_path.name
    output_path = str(meta_payload.get("output_path") or "").strip()
    if output_path and Path(output_path).name == expected_name:
        return True
    file_name = str(meta_payload.get("file_name") or "").strip()
    if file_name and file_name == expected_name:
        return True
    return False


def load_features_meta_for_path(
    features_path: Path | None,
    *,
    base_dir: Path,
    prefer_db: bool = True,
) -> tuple[dict[str, Any], str]:
    """Load feature metadata for the selected features path.

    Resolution order:
    1) DB payload for matching run_date (when DB enabled and available)
    2) Latest DB payload
    3) Per-file FS sidecar (features_*.meta.json)
    4) FS latest_meta.json in features dir
    5) FS latest_meta.json in data/features
    """

    if prefer_db:
        try:
            from scripts import db  # local import avoids circular import at module load
        except Exception:
            db = None
        if db is not None and db.db_enabled():
            run_date = parse_features_run_date(features_path)
            if run_date is not None:
                payload = db.load_ml_artifact_payload("features", run_date=run_date)
                if isinstance(payload, dict) and payload:
                    if meta_matches_features_path(payload, features_path):
                        return dict(payload), "db:run_date"
                    return dict(payload), "db:run_date_mismatch"
            latest_payload = db.load_ml_artifact_payload("features")
            if isinstance(latest_payload, dict) and latest_payload:
                if meta_matches_features_path(latest_payload, features_path):
                    return dict(latest_payload), "db:latest"
                return dict(latest_payload), "db:latest_mismatch"

    candidates: list[Path] = []
    if features_path is not None:
        candidates.append(per_file_meta_path(features_path))
        candidates.append(features_path.parent / FEATURES_LATEST_META_NAME)
    candidates.append(base_dir / "data" / "features" / FEATURES_LATEST_META_NAME)
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            source = (
                "fs:sidecar"
                if candidate.name.endswith(".meta.json")
                and candidate.name != FEATURES_LATEST_META_NAME
                else "fs:latest"
            )
            return payload, source
    return {}, "missing"
