from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dashboards.utils import safe_tail_lines
from scripts import db
from scripts.utils.champion_config import champion_execution_overrides, load_latest_champion

_LOG_LINE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2}:\d{2})(?:,(?P<ms>\d{3}))?"
    r"(?:\s+-\s+[^-]+\s+-\s+)?\s*(?:\[(?P<level>\w+)\])?\s*(?P<message>.*)$"
)
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")
_ML_EVENT_TOKENS = [
    "AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES_DONE",
    "AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES",
    "MODEL_SCORE_OVERLAP_SAMPLE",
    "MODEL_SCORE_OVERLAP_DIAG",
    "MODEL_SCORE_COVERAGE",
    "CANDIDATES_ENRICH_SKIPPED",
    "CANDIDATES_ENRICHED",
    "PREDICTIONS_FRESHNESS",
    "ML_ENRICHMENT_DECISION",
    "RANKER_PREDICT rc=",
    "RANKER_RECALIBRATE_END",
    "AUTOREMEDIATE_END",
    "AUTOREMEDIATE_REPREDICT_END",
    "TRADE_ATTRIBUTION_END",
    "CHAMPION_LOAD",
    "CHAMPION_APPLIED",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _payload_as_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _coerce_iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except Exception:
            return text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _coerce_label_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text[:10] if len(text) >= 10 else text


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_int(value: Any) -> int | None:
    try:
        out = int(value)
    except Exception:
        return None
    return out


def _coerce_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _normalize_level(value: Any) -> str:
    text = str(value or "INFO").strip().upper()
    if text.startswith("WARN"):
        return "WARN"
    if text == "ERROR":
        return "ERROR"
    if text == "SUCCESS":
        return "SUCCESS"
    return "INFO"


def _coerce_scalar(value: str) -> Any:
    cleaned = value.strip().rstrip(",")
    if cleaned == "":
        return ""
    maybe_bool = _coerce_bool(cleaned)
    if maybe_bool is not None:
        return maybe_bool
    if re.fullmatch(r"-?\d+", cleaned):
        try:
            return int(cleaned)
        except Exception:
            return cleaned
    if re.fullmatch(r"-?\d+\.\d+", cleaned):
        try:
            return float(cleaned)
        except Exception:
            return cleaned
    return cleaned


def _parse_missing_symbols(value: str) -> list[str]:
    raw = value.strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[str] = []
    for item in parsed:
        symbol = str(item or "").strip().upper()
        if symbol:
            out.append(symbol)
    return out


def _token_name(message: str) -> str | None:
    for token in _ML_EVENT_TOKENS:
        if token in message:
            if token == "RANKER_PREDICT rc=":
                return "RANKER_PREDICT"
            return token
    return None


def _parse_log_line(line: str) -> dict[str, Any] | None:
    match = _LOG_LINE_RE.match(line.strip())
    if not match:
        return None
    ts_text = f"{match.group('date')}T{match.group('time')}"
    ms = match.group("ms")
    if ms:
        ts_text = f"{ts_text}.{ms}"
    try:
        timestamp = datetime.fromisoformat(ts_text).replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        timestamp = None
    message = str(match.group("message") or "").strip()
    token = _token_name(message)
    data: dict[str, Any] = {}
    missing_match = re.search(r"missing_symbols=(\[[^\]]*\])", message)
    if missing_match:
        data["missing_symbols"] = _parse_missing_symbols(missing_match.group(1))
    for key, value in _KV_RE.findall(message):
        if key == "missing_symbols":
            continue
        data[key] = _coerce_scalar(value)
    return {
        "timestamp": timestamp,
        "level": _normalize_level(match.group("level")),
        "message": message,
        "token": token,
        "data": data,
    }


def _latest_token(entries: list[dict[str, Any]], token: str) -> dict[str, Any] | None:
    for entry in entries:
        if entry.get("token") == token:
            return entry
    return None


def _artifact_record(
    artifact_type: str,
    *,
    base_dir: Path,
    fs_json: Path | None = None,
    fs_glob: str | None = None,
    fs_file: Path | None = None,
) -> dict[str, Any]:
    if db.db_enabled():
        record = db.fetch_latest_ml_artifact(artifact_type)
        if record:
            return {
                "present": True,
                "source": "db",
                "run_date": _coerce_label_date(record.get("run_date")),
                "created_at": _coerce_iso(record.get("created_at")),
                "rows_count": _coerce_int(record.get("rows_count")),
                "file_name": _clean_text(record.get("file_name")),
                "payload": _payload_as_dict(record.get("payload")),
            }

    target: Path | None = None
    payload: dict[str, Any] = {}
    if fs_json is not None and fs_json.exists():
        target = fs_json
        payload = _read_json(fs_json)
    elif fs_file is not None and fs_file.exists():
        target = fs_file
    elif fs_glob:
        matches = sorted(base_dir.glob(fs_glob), key=lambda candidate: candidate.stat().st_mtime)
        if matches:
            target = matches[-1]
            if target.suffix.lower() == ".json":
                payload = _read_json(target)

    if target is None:
        return {
            "present": False,
            "source": "missing",
            "run_date": None,
            "created_at": None,
            "rows_count": None,
            "file_name": None,
            "payload": {},
        }

    try:
        created_at = datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        created_at = None

    inferred_run_date = None
    if payload.get("run_date") is not None:
        inferred_run_date = _coerce_label_date(payload.get("run_date"))
    elif target.stem:
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", target.stem)
        if date_match:
            inferred_run_date = date_match.group(1)

    return {
        "present": True,
        "source": "fs",
        "run_date": inferred_run_date,
        "created_at": created_at,
        "rows_count": None,
        "file_name": target.name,
        "payload": payload,
    }


def _latest_model_snapshot(base_dir: Path) -> dict[str, Any]:
    models_dir = base_dir / "data" / "models"
    try:
        matches = sorted(models_dir.glob("ranker_*.pkl"), key=lambda path: path.stat().st_mtime)
    except Exception:
        matches = []
    if not matches:
        return {"present": False, "path": None, "timestamp": None, "run_date": None}
    latest = matches[-1]
    try:
        ts = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        ts = None
    match = re.search(r"(\d{4}-\d{2}-\d{2})", latest.name)
    return {
        "present": True,
        "path": str(latest),
        "timestamp": ts,
        "run_date": match.group(1) if match else None,
    }


def _latest_status_payload(base_dir: Path) -> dict[str, Any]:
    for path in (
        base_dir / "data" / "pipeline_status.json",
        base_dir / "data" / "screener_metrics.json",
    ):
        payload = _read_json(path)
        if payload:
            return payload
    return {}


def _normalize_action(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    aliases = {
        "monitor": "recalibrate",
        "run_autotune": "retrain",
    }
    return aliases.get(text, text)


def _status_label(
    *,
    freshness_stale: bool | None,
    recommended_action: str | None,
    coverage_pct: float | None,
    has_any_ml_data: bool,
) -> tuple[str, str]:
    if freshness_stale is True:
        return "Stale", "warning"
    if recommended_action == "retrain":
        return "Retrain", "error"
    if recommended_action == "recalibrate":
        return "Recalibrate", "warning"
    if coverage_pct is not None and coverage_pct <= 0 and has_any_ml_data:
        return "Stale", "warning"
    if has_any_ml_data:
        return "Healthy", "success"
    return "No data", "neutral"


def _stage_status(
    *,
    label: str,
    timestamp: str | None,
    detail: str | None,
    status: str,
    tone: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "timestamp": timestamp,
        "detail": detail,
        "status": status,
        "tone": tone,
    }


def _max_iso(*values: Any) -> str | None:
    candidates = []
    for value in values:
        iso = _coerce_iso(value)
        if iso:
            candidates.append(iso)
    return max(candidates) if candidates else None


def _build_recent_events(entries: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for entry in entries:
        token = _clean_text(entry.get("token"))
        if not token:
            continue
        events.append(
            {
                "timestamp": entry.get("timestamp"),
                "level": entry.get("level"),
                "token": token,
                "message": entry.get("message"),
            }
        )
        if len(events) >= limit:
            break
    return events


def build_ml_overview(base_dir: Path) -> dict[str, Any]:
    base = Path(base_dir)
    status_payload = _latest_status_payload(base)
    log_lines = safe_tail_lines(base / "logs" / "pipeline.log", max_lines=1200)
    parsed_entries = [entry for entry in (_parse_log_line(line) for line in reversed(log_lines)) if entry]

    freshness_entry = _latest_token(parsed_entries, "PREDICTIONS_FRESHNESS")
    overlap_entry = _latest_token(parsed_entries, "MODEL_SCORE_OVERLAP_DIAG")
    overlap_sample_entry = _latest_token(parsed_entries, "MODEL_SCORE_OVERLAP_SAMPLE")
    coverage_entry = _latest_token(parsed_entries, "MODEL_SCORE_COVERAGE")
    enrich_entry = _latest_token(parsed_entries, "CANDIDATES_ENRICHED")
    enrich_skip_entry = _latest_token(parsed_entries, "CANDIDATES_ENRICH_SKIPPED")
    candidate_refresh_entry = _latest_token(parsed_entries, "AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES")
    candidate_refresh_done_entry = _latest_token(
        parsed_entries, "AUTO_REFRESH_PREDICTIONS_FOR_CANDIDATES_DONE"
    )
    predict_entry = _latest_token(parsed_entries, "RANKER_PREDICT")
    ml_decision_entry = _latest_token(parsed_entries, "ML_ENRICHMENT_DECISION")
    recalibrate_event = _latest_token(parsed_entries, "RANKER_RECALIBRATE_END")
    autoremediate_event = _latest_token(parsed_entries, "AUTOREMEDIATE_END")
    autoremediate_repredict_event = _latest_token(parsed_entries, "AUTOREMEDIATE_REPREDICT_END")
    trade_attr_event = _latest_token(parsed_entries, "TRADE_ATTRIBUTION_END")

    predictions_record = _artifact_record(
        "predictions",
        base_dir=base,
        fs_json=base / "data" / "predictions" / "latest_meta.json",
    )
    labels_record = _artifact_record(
        "labels",
        base_dir=base,
        fs_glob="data/labels/labels_*.csv",
    )
    features_record = _artifact_record(
        "features",
        base_dir=base,
        fs_glob="data/features/features_*.csv",
    )
    eval_record = _artifact_record(
        "ranker_eval",
        base_dir=base,
        fs_json=base / "data" / "ranker_eval" / "latest.json",
    )
    monitor_record = _artifact_record(
        "ranker_monitor",
        base_dir=base,
        fs_json=base / "data" / "ranker_monitor" / "latest.json",
    )
    recalibrate_record = _artifact_record("ranker_recalibrate", base_dir=base)
    autoremediate_record = _artifact_record(
        "ranker_autoremediate",
        base_dir=base,
        fs_json=base / "data" / "ranker_autoremediate" / "latest.json",
    )
    trade_attr_record = _artifact_record(
        "ranker_trade_attribution",
        base_dir=base,
        fs_json=base / "data" / "ranker_trade_attribution" / "latest.json",
    )
    oos_predictions_record = _artifact_record(
        "ranker_oos_predictions",
        base_dir=base,
        fs_file=base / "data" / "ranker_walkforward" / "oos_predictions.csv",
    )

    predictions_payload = dict(predictions_record.get("payload") or {})
    monitor_payload = dict(monitor_record.get("payload") or {})
    eval_payload = dict(eval_record.get("payload") or {})
    autoremediate_payload = dict(autoremediate_record.get("payload") or {})
    trade_attr_payload = dict(trade_attr_record.get("payload") or {})

    champion_payload = load_latest_champion(base)
    champion_execution = champion_execution_overrides(champion_payload or {})
    champion_params = (
        dict(champion_payload.get("champion_params") or {})
        if isinstance(champion_payload, Mapping)
        else {}
    )

    coverage_payload = dict(status_payload.get("model_score_coverage") or {})
    if coverage_entry and not coverage_payload:
        coverage_payload = dict(coverage_entry.get("data") or {})

    coverage_pct = _coerce_float(coverage_payload.get("pct"))
    coverage_total = _coerce_int(coverage_payload.get("total"))
    coverage_non_null = _coerce_int(coverage_payload.get("non_null"))
    coverage_run_ts = _clean_text(coverage_payload.get("run_ts_utc"))
    coverage_source = _clean_text(coverage_payload.get("source"))

    status_ml_health = status_payload.get("ml_health")
    if not isinstance(status_ml_health, Mapping):
        status_ml_health = {}

    freshness_data = dict(freshness_entry.get("data") or {}) if freshness_entry else {}
    feature_compat = predictions_payload.get("feature_compat")
    if not isinstance(feature_compat, Mapping):
        feature_compat = {}
    freshness_stale = _coerce_bool(freshness_data.get("stale"))
    if freshness_stale is None:
        freshness_stale = _coerce_bool(status_ml_health.get("predictions_stale"))
    if freshness_stale is None and predictions_payload:
        compat = _coerce_bool(feature_compat.get("compatible"))
        freshness_stale = False if compat is True else None
    freshness_reason = _clean_text(freshness_data.get("reason")) or _clean_text(
        status_ml_health.get("predictions_stale_reason")
    )
    pred_compatible = _coerce_bool(freshness_data.get("pred_compatible"))
    if pred_compatible is None:
        pred_compatible = _coerce_bool(feature_compat.get("compatible"))
    latest_features_set = _clean_text(freshness_data.get("latest_features_set")) or _clean_text(
        predictions_payload.get("features_feature_set")
    )
    latest_features_signature = _clean_text(
        freshness_data.get("latest_features_signature")
    ) or _clean_text(predictions_payload.get("features_feature_signature"))
    pred_features_set = _clean_text(freshness_data.get("pred_features_set")) or _clean_text(
        predictions_payload.get("feature_set")
    )
    pred_features_signature = _clean_text(
        freshness_data.get("pred_features_signature")
    ) or _clean_text(predictions_payload.get("feature_signature"))
    pred_missing_frac = _coerce_float(freshness_data.get("pred_missing_frac"))
    if pred_missing_frac is None:
        pred_missing_frac = _coerce_float(predictions_payload.get("missing_feature_fraction"))
    pred_compat_reason = _clean_text(freshness_data.get("pred_compat_reason")) or _clean_text(
        feature_compat.get("reason")
    )

    monitor_action = _normalize_action(monitor_payload.get("recommended_action"))
    if monitor_action is None and ml_decision_entry:
        monitor_action = _normalize_action(ml_decision_entry.get("data", {}).get("action"))

    calibration_payload = monitor_payload.get("calibration")
    if not isinstance(calibration_payload, Mapping):
        calibration_payload = {}
    recent_calibration = calibration_payload.get("recent")
    if not isinstance(recent_calibration, Mapping):
        recent_calibration = {}
    drift_payload = monitor_payload.get("drift")
    if not isinstance(drift_payload, Mapping):
        drift_payload = {}
    ml_health_payload = status_ml_health

    status_label, status_tone = _status_label(
        freshness_stale=freshness_stale,
        recommended_action=monitor_action,
        coverage_pct=coverage_pct,
        has_any_ml_data=bool(
            predictions_record.get("present")
            or eval_record.get("present")
            or monitor_record.get("present")
            or champion_payload
        ),
    )

    last_predict = _max_iso(
        predictions_payload.get("generated_at_utc"),
        predictions_record.get("created_at"),
        freshness_entry.get("timestamp") if freshness_entry else None,
        predict_entry.get("timestamp") if predict_entry else None,
    )
    last_eval = _max_iso(eval_record.get("created_at"), eval_payload.get("run_utc"))
    last_recalibrate = _max_iso(
        recalibrate_record.get("created_at"),
        recalibrate_event.get("timestamp") if recalibrate_event else None,
    )
    last_monitor = _max_iso(monitor_record.get("created_at"), monitor_payload.get("run_utc"))
    last_autoremediate = _max_iso(
        autoremediate_record.get("created_at"),
        autoremediate_payload.get("run_utc"),
        autoremediate_event.get("timestamp") if autoremediate_event else None,
    )
    last_trade_attr = _max_iso(
        trade_attr_record.get("created_at"),
        trade_attr_payload.get("run_utc"),
        trade_attr_event.get("timestamp") if trade_attr_event else None,
    )
    model_snapshot = _latest_model_snapshot(base)
    last_ml_run = _max_iso(
        status_payload.get("timestamp"),
        coverage_run_ts,
        last_predict,
        last_eval,
        last_monitor,
        last_autoremediate,
        last_recalibrate,
    )

    overlap_data = dict(overlap_entry.get("data") or {}) if overlap_entry else {}
    overlap_sample_data = dict(overlap_sample_entry.get("data") or {}) if overlap_sample_entry else {}
    overlap_payload = {
        "candidates": _coerce_int(overlap_data.get("candidates")),
        "prediction_symbols": _coerce_int(overlap_data.get("prediction_symbols")),
        "overlap": _coerce_int(overlap_data.get("overlap")),
        "run_ts_utc": _clean_text(overlap_data.get("run_ts_utc")),
        "run_date": _clean_text(overlap_data.get("run_date")),
        "score_col": _clean_text(overlap_data.get("score_col")),
        "pred_ts_min": _clean_text(overlap_data.get("pred_ts_min")),
        "pred_ts_max": _clean_text(overlap_data.get("pred_ts_max")),
        "sample_reason": _clean_text(overlap_sample_data.get("reason")),
        "missing_symbols": overlap_sample_data.get("missing_symbols") or [],
    }

    features_status = "success"
    features_tone = "success"
    features_detail = "Latest features snapshot available"
    if not features_record.get("present"):
        features_status = "blocked"
        features_tone = "neutral"
        features_detail = "No features artifact available"
    elif freshness_reason in {"features_meta_missing", "features_signature_missing"}:
        features_status = "stale"
        features_tone = "warning"
        features_detail = freshness_reason.replace("_", " ")
    elif freshness_reason in {"feature_signature_mismatch", "feature_set_mismatch"}:
        features_status = "warn"
        features_tone = "warning"
        features_detail = freshness_reason.replace("_", " ")

    predict_status = "success"
    predict_tone = "success"
    predict_detail = freshness_reason or "Predictions available"
    if freshness_stale is True:
        predict_status = "stale"
        predict_tone = "warning"
    elif predict_entry and _coerce_int(predict_entry.get("data", {}).get("rc")) not in (None, 0):
        predict_status = "blocked"
        predict_tone = "error"
        predict_detail = "ranker_predict failed"
    elif not predictions_record.get("present"):
        predict_status = "blocked"
        predict_tone = "neutral"
        predict_detail = "No predictions artifact available"

    eval_signal_quality = _clean_text(eval_payload.get("signal_quality"))
    eval_status = "success" if eval_record.get("present") else "blocked"
    eval_tone = "success" if eval_record.get("present") else "neutral"
    eval_detail = eval_signal_quality or "No evaluation artifact"
    if eval_signal_quality and eval_signal_quality.lower() == "low":
        eval_status = "warn"
        eval_tone = "warning"

    monitor_status = "success"
    monitor_tone = "success"
    monitor_detail = monitor_action or "none"
    if not monitor_record.get("present"):
        monitor_status = "blocked"
        monitor_tone = "neutral"
        monitor_detail = "No monitor artifact"
    elif monitor_action == "retrain":
        monitor_status = "blocked"
        monitor_tone = "error"
    elif monitor_action == "recalibrate":
        monitor_status = "warn"
        monitor_tone = "warning"

    remediation_kind = _clean_text(autoremediate_payload.get("remediation_kind")) or "none"
    remediation_executed = _coerce_bool(autoremediate_payload.get("executed"))
    remediation_status = "skipped"
    remediation_tone = "neutral"
    remediation_detail = remediation_kind
    if autoremediate_record.get("present"):
        if remediation_executed:
            remediation_status = "success"
            remediation_tone = "success"
        elif remediation_kind in {"recalibrate", "retrain"}:
            remediation_status = "warn"
            remediation_tone = "warning"
        else:
            remediation_status = "skipped"
            remediation_tone = "neutral"
    else:
        remediation_detail = "No auto-remediation artifact"

    trade_attr_summary = trade_attr_payload.get("summary")
    if not isinstance(trade_attr_summary, Mapping):
        trade_attr_summary = {}
    trade_attr_status = "success"
    trade_attr_tone = "success"
    trade_attr_detail = _clean_text(trade_attr_payload.get("status")) or "No attribution data"
    if not trade_attr_record.get("present"):
        trade_attr_status = "skipped"
        trade_attr_tone = "neutral"
    elif str(trade_attr_payload.get("status") or "").strip().lower() not in {"ok", "success"}:
        trade_attr_status = "warn"
        trade_attr_tone = "warning"
        trade_attr_detail = _clean_text(trade_attr_payload.get("status")) or "Needs review"

    stages = [
        _stage_status(
            label="Labels",
            timestamp=labels_record.get("created_at"),
            detail="Latest labels snapshot",
            status="success" if labels_record.get("present") else "blocked",
            tone="success" if labels_record.get("present") else "neutral",
        ),
        _stage_status(
            label="Features",
            timestamp=features_record.get("created_at"),
            detail=features_detail,
            status=features_status,
            tone=features_tone,
        ),
        _stage_status(
            label="Train",
            timestamp=model_snapshot.get("timestamp"),
            detail=_clean_text(model_snapshot.get("run_date")) or "Latest ranker model",
            status="success" if model_snapshot.get("present") else "blocked",
            tone="success" if model_snapshot.get("present") else "neutral",
        ),
        _stage_status(
            label="Recalibrate",
            timestamp=last_recalibrate,
            detail=_clean_text(recalibrate_record.get("run_date")) or "No recalibration artifact",
            status="success" if recalibrate_record.get("present") else "skipped",
            tone="success" if recalibrate_record.get("present") else "neutral",
        ),
        _stage_status(
            label="Predict",
            timestamp=last_predict,
            detail=predict_detail,
            status=predict_status,
            tone=predict_tone,
        ),
        _stage_status(
            label="Eval",
            timestamp=last_eval,
            detail=eval_detail,
            status=eval_status,
            tone=eval_tone,
        ),
        _stage_status(
            label="Monitor",
            timestamp=last_monitor,
            detail=monitor_detail,
            status=monitor_status,
            tone=monitor_tone,
        ),
        _stage_status(
            label="Auto-remediate",
            timestamp=last_autoremediate,
            detail=remediation_detail,
            status=remediation_status,
            tone=remediation_tone,
        ),
        _stage_status(
            label="Trade attribution",
            timestamp=last_trade_attr,
            detail=trade_attr_detail,
            status=trade_attr_status,
            tone=trade_attr_tone,
        ),
    ]

    return {
        "ok": bool(
            predictions_record.get("present")
            or eval_record.get("present")
            or monitor_record.get("present")
            or champion_payload
            or coverage_payload
            or freshness_entry
        ),
        "updated_at": last_ml_run,
        "status": {
            "label": status_label,
            "tone": status_tone,
            "detail": freshness_reason or "ML overview",
            "last_ml_run": last_ml_run,
        },
        "freshness": {
            "label": "Stale" if freshness_stale else ("Fresh" if freshness_stale is False else "No data"),
            "stale": freshness_stale,
            "reason": freshness_reason,
            "model_path": _clean_text(freshness_data.get("model_path")) or _clean_text(
                predictions_payload.get("model_path")
            ),
            "pred_model_path": _clean_text(freshness_data.get("pred_model_path")) or _clean_text(
                predictions_payload.get("model_path")
            ),
            "latest_features_set": latest_features_set,
            "latest_features_signature": latest_features_signature,
            "pred_features_set": pred_features_set,
            "pred_features_signature": pred_features_signature,
            "pred_compatible": pred_compatible,
            "pred_missing_frac": pred_missing_frac,
            "pred_compat_reason": pred_compat_reason,
            "generated_at_utc": _clean_text(predictions_payload.get("generated_at_utc")),
            "snapshot_date": _clean_text(predictions_payload.get("snapshot_date")),
            "latest_auto_refresh": candidate_refresh_done_entry.get("data") if candidate_refresh_done_entry else None,
        },
        "coverage": {
            "total": coverage_total,
            "non_null": coverage_non_null,
            "pct": coverage_pct,
            "source": coverage_source,
            "run_ts_utc": coverage_run_ts,
        },
        "champion": {
            "present": bool(champion_payload),
            "source": _clean_text((champion_payload or {}).get("_champion_source")),
            "run_date": _clean_text((champion_payload or {}).get("_champion_run_date"))
            or _clean_text((champion_payload or {}).get("run_date")),
            "status": _clean_text((champion_payload or {}).get("champion_status")),
            "calibration": _clean_text(champion_params.get("calibration")),
            "feature_set": _clean_text(champion_params.get("feature_set")),
            "bars_adjustment": _clean_text(champion_params.get("bars_adjustment")),
            "split_adjust": _clean_text(champion_params.get("split_adjust")),
            "top_k": _coerce_int(champion_params.get("top_k")),
            "execution": {
                "min_model_score": _coerce_float(
                    champion_execution.get("min_model_score")
                    or (champion_payload or {}).get("execution", {}).get("min_model_score")
                ),
                "require_model_score": _coerce_bool(
                    champion_execution.get("require_model_score")
                    if "require_model_score" in champion_execution
                    else (champion_payload or {}).get("execution", {}).get("require_model_score")
                ),
            },
        },
        "monitor": {
            "present": bool(monitor_record.get("present")),
            "run_date": _clean_text(monitor_payload.get("run_date")) or monitor_record.get("run_date"),
            "recommended_action": monitor_action,
            "psi_score": _coerce_float(monitor_payload.get("psi_score"))
            or _coerce_float(drift_payload.get("max_psi")),
            "recent_sharpe": _coerce_float(monitor_payload.get("recent_sharpe"))
            or _coerce_float((monitor_payload.get("recent_strategy") or {}).get("sharpe")),
            "ece": _coerce_float(recent_calibration.get("ece")),
            "delta_ece": _coerce_float(calibration_payload.get("delta_ece")),
            "guard_decision": _clean_text(ml_health_payload.get("decision")),
            "guard_mode": _clean_text(ml_health_payload.get("mode")),
            "guard_reasons": [
                str(reason)
                for reason in list(ml_health_payload.get("reasons") or [])
                if str(reason).strip()
            ],
        },
        "eval": {
            "present": bool(eval_record.get("present")),
            "run_date": eval_record.get("run_date"),
            "signal_quality": eval_signal_quality,
            "sample_size": _coerce_int(eval_payload.get("sample_size")),
            "decile_lift": _coerce_float(eval_payload.get("decile_lift")),
            "top_avg_label": _coerce_float(eval_payload.get("top_avg_label")),
            "bottom_avg_label": _coerce_float(eval_payload.get("bottom_avg_label")),
        },
        "remediation": {
            "present": bool(autoremediate_record.get("present")),
            "last_action": _clean_text(
                (autoremediate_payload.get("decision") or {}).get("recommended_action")
            ),
            "last_kind": remediation_kind,
            "executed": remediation_executed,
            "run_date": autoremediate_record.get("run_date"),
            "repredict_executed": _coerce_bool(
                (autoremediate_payload.get("repredict") or {}).get("executed")
            ),
            "repredict_rc": _coerce_int((autoremediate_payload.get("repredict") or {}).get("rc")),
            "repredict_reason": _clean_text(
                (autoremediate_payload.get("repredict") or {}).get("reason")
            ),
            "features_refresh_attempted": _coerce_bool(
                (
                    (autoremediate_payload.get("repredict") or {}).get("features_refresh") or {}
                ).get("attempted")
            ),
            "last_repredict_event": autoremediate_repredict_event.get("data")
            if autoremediate_repredict_event
            else None,
        },
        "artifacts": {
            "labels": labels_record,
            "features": features_record,
            "predictions": predictions_record,
            "oos_predictions": oos_predictions_record,
            "eval": eval_record,
            "monitor": monitor_record,
            "recalibrate": recalibrate_record,
            "autoremediate": autoremediate_record,
            "trade_attribution": trade_attr_record,
            "model": model_snapshot,
        },
        "timestamps": {
            "last_predict": last_predict,
            "last_eval": last_eval,
            "last_monitor": last_monitor,
            "last_recalibrate": last_recalibrate,
            "last_autoremediate": last_autoremediate,
            "last_trade_attribution": last_trade_attr,
            "last_model": model_snapshot.get("timestamp"),
            "last_features": features_record.get("created_at"),
            "last_labels": labels_record.get("created_at"),
            "last_ml_run": last_ml_run,
        },
        "overlap": overlap_payload,
        "enrichment": {
            "latest_result": enrich_entry.get("data") if enrich_entry else None,
            "latest_skip": enrich_skip_entry.get("data") if enrich_skip_entry else None,
            "latest_decision": ml_decision_entry.get("data") if ml_decision_entry else None,
            "candidate_refresh": candidate_refresh_entry.get("data") if candidate_refresh_entry else None,
            "candidate_refresh_done": candidate_refresh_done_entry.get("data")
            if candidate_refresh_done_entry
            else None,
        },
        "pipeline_stages": stages,
        "recent_events": _build_recent_events(parsed_entries, limit=10),
        "trade_attribution": {
            "present": bool(trade_attr_record.get("present")),
            "status": _clean_text(trade_attr_payload.get("status")),
            "trades_scored": _coerce_int(trade_attr_summary.get("trades_scored")),
            "trades_total": _coerce_int(trade_attr_summary.get("trades_total")),
            "win_rate_scored": _coerce_float(trade_attr_summary.get("win_rate_scored")),
            "brier": _coerce_float(trade_attr_summary.get("brier")),
        },
    }


__all__ = ["build_ml_overview"]
