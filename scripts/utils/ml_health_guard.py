"""Helpers for loading ranker_monitor health state and enrichment decisions."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from scripts import db

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MAX_AGE_DAYS = 7
_ACTION_ALIASES = {
    "none": "none",
    "monitor": "recalibrate",
    "recalibrate": "recalibrate",
    "run_autotune": "retrain",
    "retrain": "retrain",
}


def _payload_to_mapping(payload: Any) -> dict[str, Any]:
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


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except Exception:
        pass
    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).date()
    except Exception:
        return None


def _normalize_action(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "none"
    return _ACTION_ALIASES.get(text, text)


def _extract_health(payload: Mapping[str, Any]) -> dict[str, Any]:
    drift = payload.get("drift")
    if not isinstance(drift, Mapping):
        drift = {}
    recent_strategy = payload.get("recent_strategy")
    if not isinstance(recent_strategy, Mapping):
        recent_strategy = {}

    action = _normalize_action(payload.get("recommended_action"))

    psi_score = _safe_float(payload.get("psi_score"))
    if psi_score is None:
        psi_score = _safe_float(drift.get("max_psi"))

    recent_sharpe = _safe_float(payload.get("recent_sharpe"))
    if recent_sharpe is None:
        recent_sharpe = _safe_float(recent_strategy.get("sharpe"))

    run_date = payload.get("run_date")
    run_date_text = str(run_date) if run_date not in (None, "") else None

    return {
        "recommended_action": action,
        "psi_score": psi_score,
        "recent_sharpe": recent_sharpe,
        "run_date": run_date_text,
    }


def _resolve_source(source: str | None) -> str:
    candidate = str(source or "").strip().lower()
    if not candidate:
        candidate = str(os.getenv("JBR_ML_HEALTH_SOURCE") or "auto").strip().lower()
    if candidate not in {"auto", "db", "fs"}:
        candidate = "auto"
    return candidate


def _resolve_path(base_dir: Path, path: str | Path | None) -> Path:
    candidate = path if path not in (None, "") else os.getenv("JBR_ML_HEALTH_PATH")
    if candidate in (None, ""):
        return base_dir / "data" / "ranker_monitor" / "latest.json"
    target = Path(str(candidate))
    if target.is_absolute():
        return target
    return base_dir / target


def resolve_ml_health_max_age_days(max_age_days: int | None = None) -> int:
    if max_age_days is not None:
        try:
            value = int(max_age_days)
            if value >= 0:
                return value
        except Exception:
            pass
    raw = str(os.getenv("JBR_ML_HEALTH_MAX_AGE_DAYS") or "").strip()
    if not raw:
        return DEFAULT_MAX_AGE_DAYS
    try:
        value = int(float(raw))
    except Exception:
        return DEFAULT_MAX_AGE_DAYS
    if value < 0:
        return DEFAULT_MAX_AGE_DAYS
    return value


def load_latest_ml_health(
    *,
    base_dir: Path | None = None,
    logger: logging.Logger | None = None,
    source: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load latest ranker monitor health status with DB-first precedence."""

    log = logger or logging.getLogger("ml_health_guard")
    resolved_base = Path(base_dir) if base_dir is not None else BASE_DIR
    source_mode = _resolve_source(source)
    fs_path = _resolve_path(resolved_base, path)
    default: dict[str, Any] = {
        "present": False,
        "source": "missing",
        "recommended_action": None,
        "psi_score": None,
        "recent_sharpe": None,
        "run_date": None,
    }

    try:
        if source_mode in {"auto", "db"} and db.db_enabled():
            record = db.fetch_latest_ml_artifact("ranker_monitor")
            if record:
                payload = _payload_to_mapping(record.get("payload"))
                if payload:
                    health = _extract_health(payload)
                    run_date = health.get("run_date")
                    if run_date in (None, "") and record.get("run_date") is not None:
                        run_date = str(record.get("run_date"))
                        health["run_date"] = run_date
                    result = {**default, **health, "present": True, "source": "db"}
                    log.info(
                        "[INFO] ML_HEALTH_LOAD source=db present=true run_date=%s",
                        result.get("run_date"),
                    )
                    log.info(
                        "[INFO] ML_HEALTH_STATUS action=%s psi_score=%s recent_sharpe=%s",
                        result.get("recommended_action"),
                        result.get("psi_score"),
                        result.get("recent_sharpe"),
                    )
                    return result
    except Exception:
        log.exception("ML_HEALTH_LOAD_DB_FAILED")

    try:
        if source_mode in {"auto", "fs"} and fs_path.exists():
            payload = _payload_to_mapping(fs_path.read_text(encoding="utf-8"))
            if payload:
                health = _extract_health(payload)
                result = {**default, **health, "present": True, "source": "fs"}
                log.info(
                    "[INFO] ML_HEALTH_LOAD source=fs present=true run_date=%s",
                    result.get("run_date"),
                )
                log.info(
                    "[INFO] ML_HEALTH_STATUS action=%s psi_score=%s recent_sharpe=%s",
                    result.get("recommended_action"),
                    result.get("psi_score"),
                    result.get("recent_sharpe"),
                )
                return result
    except Exception:
        log.exception("ML_HEALTH_LOAD_FS_FAILED")

    log.info("[INFO] ML_HEALTH_LOAD source=missing present=false run_date=%s", None)
    log.warning("[WARN] ML_HEALTH_MISSING reason=no_ranker_monitor_artifact")
    return default


def decide_ml_enrichment(
    monitor_payload: Mapping[str, Any] | None,
    *,
    mode: str,
    max_age_days: int,
    pipeline_run_date: date | None,
    predictions_stale: bool | None = None,
    predictions_stale_reason: str | None = None,
) -> dict[str, Any]:
    """Return deterministic enrichment decision from monitor payload + policy."""

    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"warn", "block"}:
        normalized_mode = "warn"
    try:
        max_age = int(max_age_days)
    except Exception:
        max_age = DEFAULT_MAX_AGE_DAYS
    if max_age < 0:
        max_age = DEFAULT_MAX_AGE_DAYS

    payload = dict(monitor_payload or {})
    present = bool(payload.get("present"))
    action = _normalize_action(payload.get("recommended_action"))
    source = str(payload.get("source") or "missing")
    psi_score = _safe_float(payload.get("psi_score"))
    recent_sharpe = _safe_float(payload.get("recent_sharpe"))
    monitor_run_date = _parse_date(payload.get("run_date"))

    reasons: list[str] = []
    if not present:
        reasons.append("missing_monitor")
    if action == "recalibrate":
        reasons.append("action_recalibrate")
    elif action == "retrain":
        reasons.append("action_retrain")
    elif action != "none":
        reasons.append(f"action_{action}")
    if pipeline_run_date is not None:
        if monitor_run_date is None:
            reasons.append("stale_monitor")
        else:
            age_days = (pipeline_run_date - monitor_run_date).days
            if age_days > max_age:
                reasons.append("stale_monitor")
    if predictions_stale is True:
        reasons.append("stale_predictions")

    if reasons:
        decision = "block" if normalized_mode == "block" else "warn"
    else:
        decision = "allow"

    return {
        "decision": decision,
        "mode": normalized_mode,
        "reasons": reasons,
        "recommended_action": action,
        "psi_score": psi_score,
        "recent_sharpe": recent_sharpe,
        "monitor_run_date": monitor_run_date.isoformat() if monitor_run_date else None,
        "source": source,
        "max_age_days": max_age,
        "predictions_stale_reason": (
            str(predictions_stale_reason).strip() if predictions_stale_reason else None
        ),
    }


__all__ = [
    "decide_ml_enrichment",
    "load_latest_ml_health",
    "resolve_ml_health_max_age_days",
]
