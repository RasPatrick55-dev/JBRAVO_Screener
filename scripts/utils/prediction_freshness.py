"""Prediction freshness checks for DB-first pipeline usage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _coerce_text(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _normalize_path(value: Any) -> str:
    text = _coerce_text(value)
    if not text:
        return ""
    try:
        return str(Path(text)).replace("\\", "/").lower()
    except Exception:
        return text.lower()


def _parse_ts(value: Any) -> datetime | None:
    text = _coerce_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_feature_set(value: Any) -> str:
    return _coerce_text(value).lower()


def _normalize_signature(value: Any) -> str:
    return _coerce_text(value)


def _pred_feature_set(preds: Mapping[str, Any]) -> str:
    return (
        _normalize_feature_set(preds.get("feature_set"))
        or _normalize_feature_set(preds.get("model_feature_set"))
        or _normalize_feature_set(preds.get("features_feature_set"))
    )


def _pred_feature_signature(preds: Mapping[str, Any]) -> str:
    return (
        _normalize_signature(preds.get("feature_signature"))
        or _normalize_signature(preds.get("model_feature_signature"))
        or _normalize_signature(preds.get("features_feature_signature"))
    )


def _features_feature_set(latest_features_meta: Mapping[str, Any]) -> str:
    return _normalize_feature_set(latest_features_meta.get("feature_set"))


def _features_feature_signature(latest_features_meta: Mapping[str, Any]) -> str:
    return _normalize_signature(latest_features_meta.get("feature_signature"))


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _coerce_text(value).lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def evaluate_predictions_freshness(
    latest_model_meta: Mapping[str, Any] | None,
    latest_features_meta: Mapping[str, Any] | None,
    predictions_meta: Mapping[str, Any] | None,
    *,
    strict_meta: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    """Evaluate prediction freshness against latest model + latest features metadata.

    Returns `(stale, reason, details)` where `reason` may contain comma-separated
    reason tags when multiple staleness causes are present.
    """

    latest_model = dict(latest_model_meta or {})
    latest_features = dict(latest_features_meta or {})
    preds = dict(predictions_meta or {})

    latest_path = _normalize_path(latest_model.get("model_path"))
    pred_path = _normalize_path(preds.get("model_path"))
    latest_mtime = _parse_ts(latest_model.get("model_mtime_utc"))
    pred_mtime = _parse_ts(preds.get("model_mtime_utc"))

    latest_feature_set = _features_feature_set(latest_features)
    latest_feature_signature = _features_feature_signature(latest_features)
    pred_feature_set = _pred_feature_set(preds)
    pred_feature_signature = _pred_feature_signature(preds)
    feature_compat = preds.get("feature_compat")
    if isinstance(feature_compat, Mapping):
        compat_payload = dict(feature_compat)
    else:
        compat_payload = {}
    pred_compatible = _as_bool(compat_payload.get("compatible"))
    pred_missing_frac = _as_float(compat_payload.get("missing_feature_fraction"))
    pred_compat_reason = _coerce_text(compat_payload.get("reason"))
    if not pred_compat_reason:
        pred_compat_reason = "none"

    reasons: list[str] = []
    if not latest_path and latest_mtime is None:
        reasons.append("latest_model_meta_missing")
    if not preds:
        reasons.append("predictions_meta_missing")
    elif not pred_path and pred_mtime is None:
        reasons.append("predictions_meta_incomplete")

    if latest_path and pred_path and latest_path != pred_path:
        reasons.append("model_path_mismatch")
    if latest_mtime is not None and pred_mtime is None:
        reasons.append("predictions_model_mtime_missing")
    if latest_mtime is None and pred_mtime is not None:
        reasons.append("latest_model_mtime_missing")
    if latest_mtime is not None and pred_mtime is not None and pred_mtime < latest_mtime:
        reasons.append("model_mtime_older")

    if not latest_features:
        reasons.append("features_meta_missing")
    else:
        if latest_feature_set and pred_feature_set and latest_feature_set != pred_feature_set:
            reasons.append("pred_feature_set_mismatch")
        if (
            latest_feature_signature
            and pred_feature_signature
            and latest_feature_signature != pred_feature_signature
        ):
            reasons.append("pred_feature_signature_mismatch")
    if pred_compatible is False:
        reasons.append("pred_feature_incompatible")
    if strict_meta:
        if not compat_payload:
            reasons.append("pred_feature_compat_missing")
        if not pred_feature_set:
            reasons.append("pred_feature_set_missing")
        if not pred_feature_signature:
            reasons.append("pred_feature_signature_missing")
        if not pred_path or pred_mtime is None:
            reasons.append("pred_model_meta_missing")

    stale = bool(reasons)
    reason = ",".join(reasons) if reasons else "fresh"
    details: dict[str, Any] = {
        "latest_model_path": latest_model.get("model_path"),
        "pred_model_path": preds.get("model_path"),
        "latest_model_mtime_utc": (
            latest_mtime.isoformat() if isinstance(latest_mtime, datetime) else None
        ),
        "pred_model_mtime_utc": (
            pred_mtime.isoformat() if isinstance(pred_mtime, datetime) else None
        ),
        "latest_features_feature_set": latest_feature_set or None,
        "latest_features_feature_signature": latest_feature_signature or None,
        "predictions_feature_set": pred_feature_set or None,
        "predictions_feature_signature": pred_feature_signature or None,
        "pred_compatible": pred_compatible if pred_compatible is not None else "unknown",
        "pred_missing_frac": pred_missing_frac,
        "pred_compat_reason": pred_compat_reason,
        "pred_feature_compat_present": bool(compat_payload),
        "strict_meta": bool(strict_meta),
    }
    return stale, reason, details


def is_predictions_stale(
    latest_model_meta: Mapping[str, Any] | None,
    predictions_meta: Mapping[str, Any] | None,
    latest_features_meta: Mapping[str, Any] | None = None,
    *,
    strict_meta: bool = False,
) -> tuple[bool, str]:
    """Backward-compatible wrapper returning `(stale, reason)` only."""

    stale, reason, _ = evaluate_predictions_freshness(
        latest_model_meta,
        latest_features_meta,
        predictions_meta,
        strict_meta=bool(strict_meta),
    )
    return stale, reason
