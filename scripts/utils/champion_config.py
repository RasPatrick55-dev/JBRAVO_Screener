"""Champion configuration helpers for pipeline ML overrides."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from scripts import db


_FEATURE_SET_VALUES = {"v1", "v2"}
_BARS_ADJUSTMENT_VALUES = {"raw", "split", "dividend", "all"}
_SPLIT_ADJUST_VALUES = {"off", "auto", "force"}
_CALIBRATION_VALUES = {"none", "sigmoid", "isotonic"}


def _payload_as_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except Exception:
            return {}
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _normalize_str(value: Any) -> str:
    return str(value).strip().lower()


def _normalize_bool(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = _normalize_str(value)
    if text in {"1", "true", "yes", "on"}:
        return "true"
    if text in {"0", "false", "no", "off"}:
        return "false"
    return None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def load_latest_champion(base_dir: Path, run_date: date | None = None) -> dict[str, Any] | None:
    """Load champion payload DB-first, with filesystem fallback."""

    if db.db_enabled():
        record = db.fetch_ml_artifact("ranker_champion", run_date=run_date)
        if record:
            payload = _payload_as_dict(record.get("payload"))
            if payload:
                out = dict(payload)
                out.setdefault("_champion_source", "db")
                if record.get("run_date") is not None:
                    out.setdefault("_champion_run_date", str(record.get("run_date")))
                return out

    fs_path = base_dir / "data" / "ranker_autotune" / "champion.json"
    if not fs_path.exists():
        return None
    try:
        payload = json.loads(fs_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None

    out = dict(payload)
    if run_date is not None and out.get("run_date"):
        if str(out.get("run_date")) != run_date.isoformat():
            return None
    out.setdefault("_champion_source", "fs")
    if out.get("run_date") is not None:
        out.setdefault("_champion_run_date", str(out.get("run_date")))
    return out


def champion_env_overrides(champion: dict[str, Any]) -> dict[str, str]:
    """Map champion parameters to existing ML env knobs."""

    if not isinstance(champion, Mapping):
        return {}
    params = champion.get("champion_params")
    if not isinstance(params, Mapping):
        params = champion

    overrides: dict[str, str] = {}

    feature_set = _normalize_str(params.get("feature_set"))
    if feature_set in _FEATURE_SET_VALUES:
        overrides["JBR_ML_FEATURE_SET"] = feature_set

    bars_adjustment = _normalize_str(params.get("bars_adjustment"))
    if bars_adjustment in _BARS_ADJUSTMENT_VALUES:
        overrides["JBR_BARS_ADJUSTMENT"] = bars_adjustment

    split_adjust = _normalize_str(params.get("split_adjust"))
    if split_adjust in _SPLIT_ADJUST_VALUES:
        overrides["JBR_SPLIT_ADJUST"] = split_adjust

    calibration = _normalize_str(params.get("calibration"))
    if calibration in _CALIBRATION_VALUES:
        overrides["JBR_ML_CALIBRATION"] = calibration

    strict_fwd_ret = params.get("strict_fwd_ret")
    if strict_fwd_ret is None:
        strict_fwd_ret = champion.get("strict_fwd_ret")
    strict_norm = _normalize_bool(strict_fwd_ret)
    if strict_norm is not None:
        overrides["JBR_STRICT_FWD_RET"] = strict_norm

    return overrides


def champion_execution_overrides(champion: dict[str, Any]) -> dict[str, Any]:
    """Map champion execution fields to execute_trades-compatible defaults."""

    if not isinstance(champion, Mapping):
        return {}
    execution = champion.get("execution")
    if not isinstance(execution, Mapping):
        execution = {}
    params = champion.get("champion_params")
    if isinstance(params, Mapping):
        nested = params.get("execution")
        if isinstance(nested, Mapping):
            merged = dict(execution)
            merged.update(dict(nested))
            execution = merged

    out: dict[str, Any] = {}
    min_model_score = _safe_float(execution.get("min_model_score"))
    if min_model_score is not None:
        out["min_model_score"] = max(0.0, float(min_model_score))

    require_norm = _normalize_bool(execution.get("require_model_score"))
    if require_norm is not None:
        out["require_model_score"] = require_norm == "true"

    return out
