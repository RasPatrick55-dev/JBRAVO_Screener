"""Helpers for loading ranker_monitor health state and enrichment decisions."""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
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
    """Legacy artifact-label parsing retained for existing non-pipeline callers."""
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
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except Exception:
        return None


def _strict_monitor_date(value: Any) -> date | None:
    parsed = _coverage_date(value)
    if parsed is not None:
        return parsed
    timestamp = _generation_time(value)
    return timestamp.date() if timestamp is not None else None


def _normalize_action(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "none"
    return _ACTION_ALIASES.get(text, text)


def _coverage_date(value: Any) -> date | None:
    """Coverage is a session date, not a timestamp or an artifact date."""
    if type(value) is date:
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
        return parsed if parsed.isoformat() == value else None
    except ValueError:
        return None


def _generation_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    try:
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _monitor_provenance(payload, *, now, reference_date, max_age_days):
    """Assess report age and OOS coverage independently; never substitute dates."""
    reasons = []
    input_source = payload.get("monitor_input_source")
    if input_source in (None, ""):
        reasons.append("missing_monitor_input_source")
    elif not isinstance(input_source, str):
        reasons.append("invalid_monitor_input_source")
    raw_generation = payload.get("monitor_generated_at")
    generated = _generation_time(raw_generation)
    if raw_generation in (None, ""):
        reasons.append("missing_monitor_generation_time")
    elif generated is None:
        reasons.append("invalid_monitor_generation_time")
    if generated is not None:
        if generated > now:
            reasons.append("future_monitor_generation_time")
        elif (now - generated).total_seconds() > max_age_days * 86400:
            reasons.append("stale_monitor_generation")

    raw_coverage = payload.get("monitor_input_coverage")
    dates, counts = {}, {}
    date_fields = ("dataset_start", "dataset_end", "baseline_start", "baseline_end",
                   "recent_start", "recent_end")
    count_fields = ("population_rows", "baseline_rows", "recent_rows")
    if raw_coverage is None:
        reasons.append("missing_monitor_input_coverage")
    elif not isinstance(raw_coverage, Mapping):
        reasons.append("invalid_monitor_input_coverage")
    else:
        for field in date_fields:
            raw = raw_coverage.get(field)
            dates[field] = _coverage_date(raw)
            if raw in (None, ""):
                reasons.append("missing_monitor_input_" + field)
            elif dates[field] is None:
                reasons.append("invalid_monitor_input_" + field)
        for field in count_fields:
            raw = raw_coverage.get(field)
            counts[field] = raw if type(raw) is int and raw >= 0 else None
            if raw is None:
                reasons.append("missing_monitor_input_" + field)
            elif counts[field] is None:
                reasons.append("invalid_monitor_input_" + field)
            elif raw == 0:
                reasons.append("empty_monitor_input_" + field)
        if all(dates.get(field) is not None for field in date_fields):
            start, end = dates["dataset_start"], dates["dataset_end"]
            # recent_start is the nominal rolling-window boundary and may
            # precede dataset_start for a short dataset produced by the monitor.
            if (start > end or not start <= dates["baseline_start"] <= dates["baseline_end"] <= end
                    or dates["recent_start"] > dates["recent_end"]
                    or not start <= dates["recent_end"] <= end):
                reasons.append("inconsistent_monitor_input_windows")
        population = counts.get("population_rows")
        if population is not None and any(counts.get(field) is not None and counts[field] > population
                                           for field in ("baseline_rows", "recent_rows")):
            reasons.append("inconsistent_monitor_input_row_counts")
        for field in ("dataset_end", "recent_end"):
            end = dates.get(field)
            if end is not None and reference_date is not None:
                if end > reference_date:
                    reasons.append("future_monitor_input_" + field)
                elif (reference_date - end).days > max_age_days:
                    reasons.append("stale_monitor_input_" + field)
            if generated is not None and end is not None and end > generated.date():
                reasons.append("monitor_input_after_generation")
    return {
        "monitor_generated_at": generated.isoformat() if generated else None,
        "monitor_generation_age_seconds": (now - generated).total_seconds() if generated else None,
        "monitor_input_coverage": {
            **{field: dates[field].isoformat() if dates.get(field) else None for field in date_fields},
            **{field: counts.get(field) for field in count_fields},
        },
        "monitor_input_age_days": ((reference_date - dates["dataset_end"]).days
                                   if reference_date is not None and dates.get("dataset_end") else None),
        "provenance_reasons": sorted(set(reasons)),
    }


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
        "monitor_generated_at": payload.get("run_utc"),
        "monitor_input_coverage": payload.get("windows"),
        "monitor_input_source": payload.get("data_source"),
        "run_date_source": "payload" if run_date_text is not None else "missing",
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
        "run_date_source": "missing",
        "artifact_run_date": None,
        "monitor_generated_at": None,
        "monitor_input_coverage": None,
        "monitor_input_source": None,
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
                        health["run_date_source"] = "db_record"
                    result = {**default, **health, "present": True, "source": "db",
                              "artifact_run_date": str(record["run_date"]) if record.get("run_date") is not None else None}
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
    now: datetime | None = None,
    require_monitor_provenance: bool = False,
) -> dict[str, Any]:
    """Return deterministic enrichment decision from monitor payload + policy."""

    if type(require_monitor_provenance) is not bool:
        raise ValueError("boolean_monitor_provenance_policy_required")
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"warn", "block"}:
        normalized_mode = "warn"
    try:
        max_age = int(max_age_days)
    except Exception:
        max_age = DEFAULT_MAX_AGE_DAYS
    if max_age < 0:
        max_age = DEFAULT_MAX_AGE_DAYS

    observed_at = _generation_time(datetime.now(timezone.utc) if now is None else now)
    if observed_at is None:
        raise ValueError("aware_monitor_reference_time_required")
    reference_date = _coverage_date(pipeline_run_date)

    payload = dict(monitor_payload or {})
    present = bool(payload.get("present"))
    action = _normalize_action(payload.get("recommended_action"))
    source = str(payload.get("source") or "missing")
    psi_score = _safe_float(payload.get("psi_score"))
    recent_sharpe = _safe_float(payload.get("recent_sharpe"))
    monitor_run_date = (_strict_monitor_date if require_monitor_provenance else _parse_date)(payload.get("run_date"))

    reasons: list[str] = []
    if not present:
        reasons.append("missing_monitor")
    if reference_date is None and require_monitor_provenance:
        reasons.append("missing_or_invalid_monitor_reference_date")
    if action == "recalibrate":
        reasons.append("action_recalibrate")
    elif action == "retrain":
        reasons.append("action_retrain")
    elif action != "none":
        reasons.append(f"action_{action}")
    if reference_date is not None:
        if monitor_run_date is None:
            reasons.append("stale_monitor")
            if require_monitor_provenance:
                reasons.append("missing_monitor_run_date" if payload.get("run_date") in (None, "")
                               else "invalid_monitor_run_date")
        else:
            age_days = (reference_date - monitor_run_date).days
            if age_days > max_age:
                reasons.append("stale_monitor")
            if require_monitor_provenance and monitor_run_date > observed_at.date():
                reasons.append("future_monitor_run_date")
    provenance = _monitor_provenance(payload, now=observed_at, reference_date=reference_date,
                                    max_age_days=max_age)
    provenance_reasons = provenance.pop("provenance_reasons")
    if require_monitor_provenance:
        reasons.extend(provenance_reasons)
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
        "monitor_provenance_required": require_monitor_provenance,
        "monitor_provenance_reasons": provenance_reasons,
        "monitor_observed_at": observed_at.isoformat(),
        "monitor_input_reference_date": reference_date.isoformat() if reference_date else None,
        "monitor_run_date_source": payload.get("run_date_source", "unspecified"),
        "monitor_artifact_run_date": payload.get("artifact_run_date"),
        "monitor_input_source": (payload.get("monitor_input_source")
                                 if isinstance(payload.get("monitor_input_source"), str) else None),
        **provenance,
        "predictions_stale_reason": (
            str(predictions_stale_reason).strip() if predictions_stale_reason else None
        ),
    }


__all__ = [
    "decide_ml_enrichment",
    "load_latest_ml_health",
    "resolve_ml_health_max_age_days",
]
