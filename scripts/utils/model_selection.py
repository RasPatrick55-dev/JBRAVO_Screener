"""Compatible model selection for DB-first ML workflows."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts.utils.feature_schema import compute_feature_signature

try:  # pragma: no cover - environment dependent
    import joblib
except Exception:  # pragma: no cover - environment dependent
    joblib = None


def _normalize_feature_set(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text or None


def _normalize_signature(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _sanitize_feature_columns(columns: Any) -> list[str]:
    if not isinstance(columns, (list, tuple)):
        return []
    seen: set[str] = set()
    resolved: list[str] = []
    for value in columns:
        column = str(value).strip()
        if not column or column in seen:
            continue
        seen.add(column)
        resolved.append(column)
    return resolved


def model_artifact_date(model_path: Path | None) -> date | None:
    if model_path is None:
        return None
    match = re.search(r"ranker_(\d{4}-\d{2}-\d{2})", model_path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except Exception:
        return None


def summary_path_for_model(model_path: Path | None) -> Path | None:
    if model_path is None:
        return None
    match = re.search(r"ranker_(\d{4}-\d{2}-\d{2})", model_path.name)
    if not match:
        return None
    return model_path.parent / f"ranker_summary_{match.group(1)}.json"


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except Exception:
        return None


def resolve_features_identity(features_meta: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(features_meta or {})
    feature_set = _normalize_feature_set(payload.get("feature_set"))
    feature_columns = _sanitize_feature_columns(payload.get("feature_columns"))
    stored_signature = _normalize_signature(payload.get("feature_signature"))
    computed_signature = compute_feature_signature(feature_columns) if feature_columns else None
    feature_signature = computed_signature or stored_signature
    return {
        "feature_set": feature_set,
        "feature_signature": feature_signature,
        "feature_count": int(len(feature_columns)),
    }


def load_model_artifact_meta(model_path: Path) -> dict[str, Any]:
    feature_set = None
    feature_signature_stored = None
    feature_columns: list[str] = []
    metadata_sources: list[str] = []
    calibration_method = "none"
    calibration_applied = False

    summary_path = summary_path_for_model(model_path)
    if summary_path is not None and summary_path.exists():
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary_payload = {}
        if isinstance(summary_payload, Mapping):
            metadata_sources.append("summary")
            feature_set = _normalize_feature_set(summary_payload.get("feature_set"))
            feature_signature_stored = _normalize_signature(summary_payload.get("feature_signature"))
            summary_columns = _sanitize_feature_columns(summary_payload.get("feature_columns"))
            if summary_columns:
                feature_columns = summary_columns
            calibration_payload = summary_payload.get("calibration")
            if isinstance(calibration_payload, Mapping):
                calibration_method = (
                    str(calibration_payload.get("method") or "none").strip().lower() or "none"
                )
                calibration_applied = bool(calibration_payload.get("applied", False))

    if joblib is not None:
        try:
            payload = joblib.load(model_path)
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            metadata_sources.append("joblib")
            payload_set = _normalize_feature_set(payload.get("feature_set"))
            if payload_set:
                feature_set = payload_set
            payload_signature = _normalize_signature(payload.get("feature_signature"))
            if payload_signature:
                feature_signature_stored = payload_signature
            payload_columns = _sanitize_feature_columns(payload.get("feature_columns"))
            if payload_columns:
                feature_columns = payload_columns
            payload_calibration_method = (
                str(payload.get("calibration_method") or "").strip().lower() or None
            )
            if payload_calibration_method:
                calibration_method = payload_calibration_method
            if bool(payload.get("calibration_applied")):
                calibration_applied = True

    computed_signature = compute_feature_signature(feature_columns) if feature_columns else None
    resolved_signature = computed_signature or feature_signature_stored
    artifact_date = model_artifact_date(model_path)
    return {
        "model_path": str(model_path),
        "artifact_date": artifact_date.isoformat() if artifact_date is not None else None,
        "model_mtime_utc": _mtime_iso(model_path),
        "feature_set": feature_set,
        "feature_signature": resolved_signature,
        "feature_signature_stored": feature_signature_stored,
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "metadata_sources": metadata_sources,
        "calibration_method": calibration_method or "none",
        "calibration_applied": bool(calibration_applied),
    }


def _compatibility_reasons(
    model_meta: Mapping[str, Any],
    *,
    features_feature_set: str | None,
    features_feature_signature: str | None,
) -> list[str]:
    reasons: list[str] = []
    model_feature_signature = _normalize_signature(model_meta.get("feature_signature"))
    model_feature_set = _normalize_feature_set(model_meta.get("feature_set"))
    if not features_feature_signature:
        reasons.append("features_signature_missing")
    if not model_feature_signature:
        reasons.append("model_signature_missing")
    if (
        features_feature_signature
        and model_feature_signature
        and model_feature_signature != features_feature_signature
    ):
        reasons.append("feature_signature_mismatch")
    if features_feature_set and model_feature_set and model_feature_set != features_feature_set:
        reasons.append("feature_set_mismatch")
    return reasons


def select_compatible_model(
    models_dir: Path,
    features_meta: Mapping[str, Any] | None,
    *,
    requested_model_path: Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    features_identity = resolve_features_identity(features_meta)
    features_feature_set = features_identity.get("feature_set")
    features_feature_signature = features_identity.get("feature_signature")
    selection_mode = "explicit" if requested_model_path is not None else "compatible_artifact_date"

    if requested_model_path is not None:
        model_paths = [Path(requested_model_path)]
    else:
        try:
            model_paths = sorted(
                models_dir.glob("ranker_*.pkl"),
                key=lambda path: (model_artifact_date(path) or date.min, path.name),
            )
        except FileNotFoundError:
            model_paths = []

    diagnostics: dict[str, Any] = {
        "selection_mode": selection_mode,
        "requested_model_path": str(requested_model_path) if requested_model_path else None,
        "features_feature_set": features_feature_set,
        "features_feature_signature": features_feature_signature,
        "candidate_count": int(len(model_paths)),
        "compatible_count": 0,
        "reason": "unknown",
        "selected_model_path": None,
        "selected_artifact_date": None,
        "selected_model_meta": None,
        "candidates": [],
    }

    if requested_model_path is not None and not requested_model_path.exists():
        diagnostics["reason"] = "explicit_model_missing"
        return None, diagnostics
    if not model_paths:
        diagnostics["reason"] = "no_models"
        return None, diagnostics

    compatible_candidates: list[dict[str, Any]] = []
    for model_path in model_paths:
        model_meta = load_model_artifact_meta(model_path)
        reasons = _compatibility_reasons(
            model_meta,
            features_feature_set=features_feature_set,
            features_feature_signature=features_feature_signature,
        )
        compatible = len(reasons) == 0
        candidate_payload = {
            "model_path": model_meta.get("model_path"),
            "artifact_date": model_meta.get("artifact_date"),
            "feature_set": model_meta.get("feature_set"),
            "feature_signature": model_meta.get("feature_signature"),
            "compatible": bool(compatible),
            "reasons": reasons,
        }
        diagnostics["candidates"].append(candidate_payload)
        if compatible:
            compatible_candidates.append({"path": model_path, "meta": model_meta})

    diagnostics["compatible_count"] = int(len(compatible_candidates))

    if compatible_candidates:
        compatible_candidates.sort(
            key=lambda candidate: (
                _normalize_signature(candidate["meta"].get("artifact_date")) or "",
                str(candidate["meta"].get("model_path") or ""),
            )
        )
        selected = compatible_candidates[-1]
        selected_path = Path(str(selected["meta"].get("model_path")))
        diagnostics["reason"] = (
            "explicit_selected" if requested_model_path is not None else "selected"
        )
        diagnostics["selected_model_path"] = str(selected_path)
        diagnostics["selected_artifact_date"] = selected["meta"].get("artifact_date")
        diagnostics["selected_model_meta"] = dict(selected["meta"])
        return selected_path, diagnostics

    if requested_model_path is not None:
        diagnostics["reason"] = "explicit_model_incompatible"
    elif not features_feature_signature:
        diagnostics["reason"] = "features_signature_missing"
    else:
        diagnostics["reason"] = "no_compatible_model"
    return None, diagnostics

