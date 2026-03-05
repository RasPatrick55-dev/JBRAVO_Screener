"""Generate nightly ranker predictions from the latest features snapshot.

This module loads a trained ranker model (produced by ``scripts.ranker_train``),
computes probabilities for the positive class, and writes a CSV containing
predictions alongside a subset of the input features for dashboard consumption.

Usage::

    python -m scripts.ranker_predict
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils.env import load_env  # noqa: E402
from scripts import db  # noqa: E402
from scripts.ranker_train import (  # noqa: E402
    FEATURE_COLUMNS,
    _extract_features_date,
    _predict_proba,
)
from scripts.utils.feature_schema import (
    compute_feature_signature,
    load_features_meta_for_path,
    meta_matches_features_path,
)

load_env()

LOG = logging.getLogger("ranker_predict")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

try:  # pragma: no cover - environment dependent
    import joblib
except Exception:  # pragma: no cover - environment dependent
    joblib = None

DEFAULT_LABEL = "label_5d_pos_300bp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_latest(directory: Path, pattern: str) -> Path | None:
    def _model_date_token(path: Path) -> date:
        match = re.search(r"ranker_(\d{4}-\d{2}-\d{2})", path.name)
        if not match:
            return date.min
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except Exception:
            return date.min

    def _sort_key(path: Path) -> tuple[float, date, str]:
        try:
            mtime = float(path.stat().st_mtime)
        except Exception:
            mtime = 0.0
        return (mtime, _model_date_token(path), path.name)

    candidates = sorted(directory.glob(pattern), key=_sort_key)
    if not candidates:
        return None
    return candidates[-1]


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except Exception:
        return None


def _write_predictions_meta(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return True
    except Exception:
        LOG.exception("PREDICTIONS_META_WRITE_FAILED path=%s", path)
        return False


def _sanitize_feature_columns(columns: Any) -> list[str]:
    if not isinstance(columns, (list, tuple)):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for value in columns:
        column = str(value).strip()
        if not column or column in seen:
            continue
        seen.add(column)
        normalized.append(column)
    return normalized


def _parse_boolish(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _normalize_features_frame(
    df: pd.DataFrame, label_column: str, feature_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_columns: set[str] = {
        "symbol",
        "timestamp",
        label_column,
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"Features input missing columns: {sorted(missing)}")

    work = df.copy()
    if "close" not in work.columns:
        work["close"] = 0.0
    missing_model_columns = [col for col in feature_columns if col not in work.columns]
    total_model_columns = int(len(feature_columns))
    missing_fraction = (
        float(len(missing_model_columns)) / float(total_model_columns)
        if total_model_columns > 0
        else 0.0
    )
    for column in missing_model_columns:
        work[column] = 0.0
    if missing_model_columns:
        LOG.warning(
            "[WARN] RANKER_PREDICT_MISSING_FEATURE_COLUMNS count=%d columns=%s",
            len(missing_model_columns),
            ",".join(missing_model_columns),
        )

    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["close"] = pd.to_numeric(work["close"], errors="coerce").fillna(0.0)
    work[label_column] = pd.to_numeric(work[label_column], errors="coerce")
    for column in feature_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work[feature_columns] = work[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work = work.dropna(subset=["timestamp", label_column]).sort_values("timestamp")
    stats = {
        "missing_columns": missing_model_columns,
        "missing_count": int(len(missing_model_columns)),
        "feature_count": int(total_model_columns),
        "missing_fraction": float(missing_fraction),
    }
    return work.reset_index(drop=True), stats


def _load_features(
    path: Path, label_column: str, feature_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to read features from {path}: {exc}") from exc

    return _normalize_features_frame(df, label_column, feature_columns)


def _load_model(path: Path):
    if joblib is None:
        raise RuntimeError("joblib is required to load the trained model")
    try:
        payload = joblib.load(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load model from {path}: {exc}") from exc

    if isinstance(payload, dict) and "model" in payload:
        model = payload.get("model")
        scaler = payload.get("scaler")
        feature_columns = _sanitize_feature_columns(payload.get("feature_columns"))
        feature_set = str(payload.get("feature_set") or "").strip().lower() or None
        feature_signature = str(payload.get("feature_signature") or "").strip() or None
        calibration_method = str(payload.get("calibration_method") or "none").strip().lower()
        calibration_applied = bool(payload.get("calibration_applied")) or calibration_method != "none"
        posthoc_calibrator = payload.get("posthoc_score_calibrator")
        if calibration_method == "none":
            calibration_method = str(payload.get("posthoc_calibration_method") or calibration_method).strip().lower()
        if posthoc_calibrator is not None:
            calibration_applied = True
    else:
        model = payload
        scaler = None
        feature_columns = []
        feature_set = None
        feature_signature = None
        calibration_method = "none"
        calibration_applied = False
        posthoc_calibrator = None
    return (
        model,
        scaler,
        feature_columns,
        feature_set,
        feature_signature,
        calibration_applied,
        calibration_method,
        posthoc_calibrator,
    )


def _summary_path_for_model(model_path: Path) -> Path | None:
    match = re.search(r"ranker_(\d{4}-\d{2}-\d{2})", model_path.name)
    if not match:
        return None
    return model_path.parent / f"ranker_summary_{match.group(1)}.json"


def _load_feature_columns_from_summary(model_path: Path) -> list[str]:
    summary_path = _summary_path_for_model(model_path)
    if summary_path is None or not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive
        return []
    return _sanitize_feature_columns(payload.get("feature_columns"))


def _load_calibration_from_summary(model_path: Path) -> tuple[bool, str]:
    summary_path = _summary_path_for_model(model_path)
    if summary_path is None or not summary_path.exists():
        return False, "none"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive
        return False, "none"
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        return False, "none"
    method = str(calibration.get("method") or "none").strip().lower()
    applied = bool(calibration.get("applied")) or method != "none"
    return applied, method


def _load_model_meta_from_summary(model_path: Path) -> tuple[str | None, str | None, list[str]]:
    summary_path = _summary_path_for_model(model_path)
    if summary_path is None or not summary_path.exists():
        return None, None, []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - defensive
        return None, None, []
    if not isinstance(payload, dict):
        return None, None, []
    feature_set = str(payload.get("feature_set") or "").strip().lower() or None
    feature_signature = str(payload.get("feature_signature") or "").strip() or None
    feature_columns = _sanitize_feature_columns(payload.get("feature_columns"))
    return feature_set, feature_signature, feature_columns


def _resolve_features_signature_from_meta(meta_payload: dict[str, Any]) -> str | None:
    meta_columns = _sanitize_feature_columns(meta_payload.get("feature_columns"))
    if meta_columns:
        return compute_feature_signature(meta_columns)
    return str(meta_payload.get("feature_signature") or "").strip() or None


def _resolve_feature_columns(model_path: Path, model_feature_columns: list[str]) -> list[str]:
    resolved = _sanitize_feature_columns(model_feature_columns)
    if resolved:
        return resolved
    summary_columns = _load_feature_columns_from_summary(model_path)
    if summary_columns:
        LOG.info("Using feature columns from model summary: %s", len(summary_columns))
        return summary_columns
    LOG.warning(
        "[WARN] RANKER_PREDICT_FEATURE_COLUMNS_FALLBACK source=default count=%d",
        len(FEATURE_COLUMNS),
    )
    return list(FEATURE_COLUMNS)


def _apply_score_calibrator(raw_scores: np.ndarray, calibrator) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=float)
    try:
        if hasattr(calibrator, "predict_proba"):
            calibrated = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        elif hasattr(calibrator, "predict"):
            calibrated = calibrator.predict(scores)
        else:
            return scores
    except Exception:
        return scores
    calibrated = np.asarray(calibrated, dtype=float)
    return calibrated


def _predict(
    df: pd.DataFrame,
    model,
    feature_columns: list[str],
    scaler=None,
    posthoc_calibrator=None,
) -> pd.Series:
    features = df[feature_columns]
    if scaler is not None:
        features = scaler.transform(features)
    probs = np.asarray(_predict_proba(model, features), dtype=float)
    if posthoc_calibrator is not None:
        probs = _apply_score_calibrator(probs, posthoc_calibrator)
    return pd.Series(probs, index=df.index)


def _build_output(
    df: pd.DataFrame, label_column: str, probs: Iterable[float], feature_columns: list[str]
) -> pd.DataFrame:
    scores = pd.Series(probs, index=df.index, name="score_5d")
    preds = (scores >= 0.5).astype(int)

    output = df.copy()
    output["score_5d"] = scores
    output["pred_5d_pos"] = preds

    columns = [
        "symbol",
        "timestamp",
        "close",
        "mom_5d",
        "mom_10d",
        "vol_raw",
        "vol_avg_10d",
        "vol_rvol_10d",
        label_column,
        "pred_5d_pos",
        "score_5d",
    ]
    extra_columns = [col for col in feature_columns if col not in columns and col in output.columns]
    return output.loc[:, [col for col in columns if col in output.columns] + extra_columns]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Path to features CSV. Defaults to latest data/features/features_*.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained model. Defaults to latest data/models/ranker_*.pkl",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=DEFAULT_LABEL,
        help="Binary label column used during training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "predictions",
        help="Directory to write predictions CSV",
    )
    parser.add_argument(
        "--strict-feature-match",
        nargs="?",
        const="true",
        default="false",
        help="Fail prediction when model/features metadata mismatch (default false).",
    )
    parser.add_argument(
        "--max-missing-feature-fraction",
        type=float,
        default=1.0,
        help="Maximum allowed missing model-feature fraction before failing (default 1.0).",
    )
    parsed = parser.parse_args(argv)
    parsed.strict_feature_match = _parse_boolish(parsed.strict_feature_match)
    try:
        parsed.max_missing_feature_fraction = float(parsed.max_missing_feature_fraction)
    except Exception:
        parsed.max_missing_feature_fraction = 1.0
    if parsed.max_missing_feature_fraction < 0.0:
        parser.error("--max-missing-feature-fraction must be >= 0.0")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    model_path = args.model_path
    if model_path is None:
        model_path = _find_latest(BASE_DIR / "data" / "models", "ranker_*.pkl")
        if model_path is None:
            LOG.error("No model files found in data/models")
            return 1
    LOG.info(
        "[INFO] RANKER_PREDICT_MODEL_SELECTED path=%s",
        model_path,
    )

    LOG.info("Loading model from %s", model_path)
    try:
        (
            model,
            scaler,
            model_feature_columns,
            model_feature_set,
            model_feature_signature,
            model_calibrated,
            model_calibration_method,
            posthoc_calibrator,
        ) = _load_model(model_path)
    except FileNotFoundError:
        LOG.error("Model file not found: %s", model_path)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("Failed to load model: %s", exc)
        return 1

    feature_columns = _resolve_feature_columns(model_path, model_feature_columns)
    LOG.info("Using %d feature columns for prediction", len(feature_columns))
    summary_feature_set, summary_feature_signature, summary_feature_columns = _load_model_meta_from_summary(
        model_path
    )
    if not model_feature_set:
        model_feature_set = summary_feature_set
    if not model_feature_signature:
        model_feature_signature = summary_feature_signature
    if not feature_columns and summary_feature_columns:
        feature_columns = summary_feature_columns
    computed_model_feature_signature = compute_feature_signature(feature_columns)
    if model_feature_signature and model_feature_signature != computed_model_feature_signature:
        LOG.warning(
            "[WARN] RANKER_PREDICT_MODEL_SIGNATURE_MISMATCH stored=%s computed=%s",
            model_feature_signature,
            computed_model_feature_signature,
        )
    resolved_model_feature_signature = computed_model_feature_signature
    summary_calibrated, summary_calibration_method = _load_calibration_from_summary(model_path)
    calibrated = bool(model_calibrated or summary_calibrated)
    calibration_method = (
        model_calibration_method
        if str(model_calibration_method or "").strip()
        else summary_calibration_method
    )
    calibration_method = str(calibration_method or "none").strip().lower()
    if calibration_method not in {"none", "sigmoid", "isotonic"}:
        calibration_method = "none"
    LOG.info(
        "[INFO] RANKER_PREDICT_SCORE_SOURCE calibrated=%s method=%s",
        str(calibrated).lower(),
        calibration_method,
    )

    features_path = args.features_path
    features_meta: dict[str, Any] | None = None
    if db.db_enabled():
        features_meta = db.fetch_latest_ml_artifact("features")
        if not features_meta:
            LOG.error("No features artifacts found in DB (ml_artifacts: features)")
            return 1
        LOG.info("Loading features from DB (ml_artifacts: features)")
        try:
            features_df, missing_stats = _normalize_features_frame(
                db.load_ml_artifact_csv("features"), args.label_column, feature_columns
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.error("Failed to load features from DB: %s", exc)
            return 1
    else:
        if features_path is None:
            features_path = _find_latest(BASE_DIR / "data" / "features", "features_*.csv")
            if features_path is None:
                LOG.error("No features files found in data/features")
                return 1
        LOG.info("Loading features from %s", features_path)
        try:
            features_df, missing_stats = _load_features(
                features_path, args.label_column, feature_columns
            )
        except FileNotFoundError:
            LOG.error("Features file not found: %s", features_path)
            return 1
        except Exception as exc:  # pragma: no cover - defensive
            LOG.error("Failed to load features: %s", exc)
            return 1

    if features_df.empty:
        LOG.error("Features input has no usable rows")
        return 1

    features_meta_payload, features_meta_source = load_features_meta_for_path(
        features_path,
        base_dir=BASE_DIR,
        prefer_db=bool(db.db_enabled()),
    )
    if features_path is not None and features_meta_payload and not meta_matches_features_path(
        features_meta_payload, features_path
    ):
        LOG.warning(
            "[WARN] RANKER_PREDICT_FEATURE_META_FILE_MISMATCH source=%s features_path=%s meta_output=%s meta_file_name=%s",
            features_meta_source,
            features_path,
            features_meta_payload.get("output_path"),
            features_meta_payload.get("file_name"),
        )
    features_feature_set = str(features_meta_payload.get("feature_set") or "").strip().lower() or None
    features_feature_signature = _resolve_features_signature_from_meta(features_meta_payload)
    resolved_feature_meta_source = features_meta_source
    if not features_feature_signature:
        fallback_signature = compute_feature_signature(feature_columns)
        features_feature_signature = fallback_signature
        resolved_feature_meta_source = "computed_fallback"
    resolved_prediction_feature_set = features_feature_set or model_feature_set
    resolved_prediction_feature_signature = features_feature_signature
    mismatch_reasons: list[str] = []
    if (
        model_feature_set
        and features_feature_set
        and model_feature_set != features_feature_set
    ):
        mismatch_reasons.append("feature_set_mismatch")
    if (
        resolved_model_feature_signature
        and features_feature_signature
        and resolved_model_feature_signature != features_feature_signature
    ):
        mismatch_reasons.append("feature_signature_mismatch")
    missing_fraction = float(missing_stats.get("missing_fraction") or 0.0)
    missing_count = int(missing_stats.get("missing_count") or 0)
    feature_count = int(missing_stats.get("feature_count") or len(feature_columns))
    if missing_fraction > float(args.max_missing_feature_fraction):
        mismatch_reasons.append("missing_feature_fraction_exceeded")
    compatible = len(mismatch_reasons) == 0
    reason_text = ",".join(mismatch_reasons) if mismatch_reasons else "none"
    LOG.info(
        "[INFO] RANKER_PREDICT_FEATURE_COMPAT strict=%s compatible=%s feature_set_model=%s feature_set_features=%s signature_model=%s signature_features=%s missing_frac=%.6f missing_count=%d feature_count=%d meta_source=%s reason=%s",
        str(bool(args.strict_feature_match)).lower(),
        str(bool(compatible)).lower(),
        model_feature_set or "unknown",
        features_feature_set or "unknown",
        resolved_model_feature_signature or "unknown",
        features_feature_signature or "unknown",
        float(missing_fraction),
        missing_count,
        feature_count,
        features_meta_source,
        reason_text,
    )
    if bool(args.strict_feature_match) and not compatible:
        LOG.error(
            "[ERROR] RANKER_PREDICT_FEATURE_MISMATCH_FATAL reason=%s strict=%s missing_frac=%.6f max_missing_feature_fraction=%.6f",
            reason_text,
            str(bool(args.strict_feature_match)).lower(),
            float(missing_fraction),
            float(args.max_missing_feature_fraction),
        )
        return 2

    probs = _predict(
        features_df,
        model,
        feature_columns,
        scaler,
        posthoc_calibrator=posthoc_calibrator,
    )
    output_df = _build_output(features_df, args.label_column, probs, feature_columns)
    if not output_df.empty and "score_5d" in output_df.columns:
        score_min = float(pd.to_numeric(output_df["score_5d"], errors="coerce").min())
        score_max = float(pd.to_numeric(output_df["score_5d"], errors="coerce").max())
        if score_min < 0.0 or score_max > 1.0:
            LOG.warning(
                "[WARN] RANKER_PREDICT_SCORE_OUT_OF_RANGE min=%s max=%s",
                score_min,
                score_max,
            )

    if db.db_enabled() and features_meta:
        snapshot_date = str(features_meta.get("run_date") or datetime.now(timezone.utc).date())
    else:
        snapshot_date = _extract_features_date(features_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{snapshot_date}.csv"
    output_df.to_csv(output_path, index=False)

    model_mtime_utc = _mtime_iso(model_path)
    predictions_meta: dict[str, Any] = {
        "model_path": str(model_path),
        "model_mtime_utc": model_mtime_utc,
        "model_feature_set": model_feature_set,
        "model_feature_signature": resolved_model_feature_signature,
        "features_feature_set": features_feature_set,
        "features_feature_signature": features_feature_signature,
        "features_meta_source": features_meta_source,
        "feature_set": resolved_prediction_feature_set,
        "feature_signature": resolved_prediction_feature_signature,
        "feature_meta_source": resolved_feature_meta_source,
        "missing_feature_fraction": missing_fraction,
        "missing_feature_count": missing_count,
        "model_calibrated": bool(calibrated),
        "calibration_method": calibration_method,
        "model_signature": f"{model_path.name}:{model_mtime_utc or 'unknown'}",
        "predictions_path": str(output_path),
        "snapshot_date": str(snapshot_date),
        "rows": int(len(output_df.index)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_compat": {
            "strict": bool(args.strict_feature_match),
            "compatible": bool(compatible),
            "reason": reason_text,
            "missing_feature_fraction": float(missing_fraction),
            "missing_count": int(missing_count),
            "feature_count": int(feature_count),
            "feature_set_model": model_feature_set,
            "feature_set_features": features_feature_set,
            "feature_signature_model": resolved_model_feature_signature,
            "feature_signature_features": features_feature_signature,
            "meta_source": features_meta_source,
        },
    }
    if not isinstance(predictions_meta.get("feature_compat"), dict):
        predictions_meta["feature_compat"] = {
            "strict": bool(args.strict_feature_match),
            "compatible": None,
            "reason": "unknown",
            "missing_feature_fraction": float(missing_fraction),
            "missing_count": int(missing_count),
            "feature_count": int(feature_count),
            "feature_set_model": model_feature_set,
            "feature_set_features": features_feature_set,
            "feature_signature_model": resolved_model_feature_signature,
            "feature_signature_features": features_feature_signature,
            "meta_source": features_meta_source,
        }

    LOG.info(
        "Predictions written to %s (rows=%d, mean_score=%.4f)",
        output_path,
        len(output_df),
        float(output_df["score_5d"].mean()) if not output_df.empty else 0.0,
    )
    meta_sidecar = output_dir / "latest_meta.json"
    if _write_predictions_meta(meta_sidecar, predictions_meta):
        LOG.info(
            "[INFO] PREDICTIONS_META_WRITTEN source=fs model_path=%s model_mtime_utc=%s calibrated=%s method=%s feature_set=%s feature_signature=%s feature_meta_source=%s compatible=%s missing_frac=%.6f compat_reason=%s",
            predictions_meta.get("model_path"),
            predictions_meta.get("model_mtime_utc"),
            str(bool(predictions_meta.get("model_calibrated"))).lower(),
            predictions_meta.get("calibration_method"),
            predictions_meta.get("feature_set"),
            predictions_meta.get("feature_signature"),
            predictions_meta.get("feature_meta_source"),
            str(bool(compatible)).lower(),
            float(missing_fraction),
            reason_text,
        )
    if db.db_enabled():
        ok = db.upsert_ml_artifact_frame(
            "predictions",
            snapshot_date,
            output_df,
            payload=predictions_meta,
            source="ranker_predict",
            file_name=output_path.name,
        )
        if ok:
            LOG.info(
                "[INFO] PREDICTIONS_DB_WRITTEN run_date=%s rows=%d",
                snapshot_date,
                len(output_df),
            )
            LOG.info(
                "[INFO] PREDICTIONS_META_WRITTEN source=db model_path=%s model_mtime_utc=%s calibrated=%s method=%s feature_set=%s feature_signature=%s feature_meta_source=%s compatible=%s missing_frac=%.6f compat_reason=%s",
                predictions_meta.get("model_path"),
                predictions_meta.get("model_mtime_utc"),
                str(bool(predictions_meta.get("model_calibrated"))).lower(),
                predictions_meta.get("calibration_method"),
                predictions_meta.get("feature_set"),
                predictions_meta.get("feature_signature"),
                predictions_meta.get("feature_meta_source"),
                str(bool(compatible)).lower(),
                float(missing_fraction),
                reason_text,
            )
        else:
            LOG.warning("[WARN] PREDICTIONS_DB_WRITE_FAILED run_date=%s", snapshot_date)
    return 0


if __name__ == "__main__":
    sys.exit(main())
