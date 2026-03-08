"""Recalibrate an already-trained ranker model on a recent time-ordered window.

This is evaluation/remediation-only plumbing. It does not change trade
execution semantics.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts import db  # noqa: E402
from scripts.ranker_train import _predict_proba  # noqa: E402
from scripts.utils.feature_schema import (
    compute_feature_signature,
    load_features_meta_for_path,
    meta_matches_features_path,
)
from utils.env import load_env  # noqa: E402

load_env()

LOG = logging.getLogger("ranker_recalibrate")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_TARGET = "label_5d_pos_300bp"
VALID_METHODS = {"sigmoid", "isotonic"}
DEFAULT_CALIBRATION_FRACTION = 0.1
DEFAULT_MIN_EMBARGO_DAYS = 5
DEFAULT_MAX_ROWS = 60_000
try:  # pragma: no cover - env dependent
    import joblib
except Exception:  # pragma: no cover - env dependent
    joblib = None

try:  # pragma: no cover - env dependent
    from sklearn.calibration import CalibratedClassifierCV
except Exception:  # pragma: no cover - env dependent
    CalibratedClassifierCV = None

try:  # pragma: no cover - env dependent
    from sklearn.isotonic import IsotonicRegression
except Exception:  # pragma: no cover - env dependent
    IsotonicRegression = None

try:  # pragma: no cover - env dependent
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - env dependent
    LogisticRegression = None


def _parse_run_date(value: str | None) -> date | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --run-date: {value}")
    return parsed.date()


def _find_latest(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _summary_path_for_model(model_path: Path) -> Path | None:
    match = re.search(r"ranker_(\d{4}-\d{2}-\d{2})", model_path.name)
    if not match:
        return None
    return model_path.parent / f"ranker_summary_{match.group(1)}.json"


def _infer_label_horizon_days(label_column: str) -> int:
    match = re.search(r"label_(\d+)d", str(label_column or ""))
    if not match:
        return DEFAULT_MIN_EMBARGO_DAYS
    try:
        return max(int(match.group(1)), 1)
    except (TypeError, ValueError):
        return DEFAULT_MIN_EMBARGO_DAYS


def _resolve_method(cli_value: str | None) -> tuple[str, str]:
    if cli_value:
        method = str(cli_value).strip().lower()
        if method in VALID_METHODS:
            return method, "cli"
    env_value = str(os.getenv("JBR_ML_CALIBRATION") or "").strip().lower()
    if env_value in VALID_METHODS:
        return env_value, "env"
    return "sigmoid", "default"


def _resolve_feature_columns(payload: dict[str, Any], summary_path: Path | None) -> list[str]:
    columns = payload.get("feature_columns")
    if isinstance(columns, (list, tuple)):
        out = [str(c).strip() for c in columns if str(c).strip()]
        if out:
            return out
    if summary_path and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        summary_cols = summary.get("feature_columns")
        if isinstance(summary_cols, (list, tuple)):
            out = [str(c).strip() for c in summary_cols if str(c).strip()]
            if out:
                return out
    raise RuntimeError("Model does not contain feature_columns metadata for recalibration.")


def _load_summary_payload(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None or not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_features_meta_payload(
    *,
    run_date: date | None,
    inputs_source: str,
    features_path_used: Path | None,
) -> dict[str, Any]:
    if db.db_enabled() and inputs_source == "db":
        payload = db.load_ml_artifact_payload("features", run_date=run_date)
        if payload:
            return payload
    payload, _ = load_features_meta_for_path(
        features_path_used,
        base_dir=BASE_DIR,
        prefer_db=bool(db.db_enabled()),
    )
    return payload


def _load_model(path: Path) -> tuple[Any, Any, list[str], dict[str, Any], Path | None]:
    if joblib is None:
        raise RuntimeError("joblib is required for ranker_recalibrate.")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError("Unsupported model payload format; expected dict with 'model'.")
    model = payload.get("model")
    scaler = payload.get("scaler")
    summary_path = _summary_path_for_model(path)
    feature_columns = _resolve_feature_columns(payload, summary_path)
    return model, scaler, feature_columns, dict(payload), summary_path


def _normalize_inputs(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    work = features_df.copy()
    if "timestamp" not in work.columns or "symbol" not in work.columns:
        raise RuntimeError("Features dataset missing required columns: symbol,timestamp")
    work["symbol"] = work["symbol"].astype("string").str.upper()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["symbol", "timestamp"]).copy()

    if target not in work.columns:
        if labels_df.empty:
            raise RuntimeError(
                f"Target column '{target}' missing from features and labels dataset unavailable."
            )
        label_work = labels_df.copy()
        if "symbol" not in label_work.columns or "timestamp" not in label_work.columns:
            raise RuntimeError("Labels dataset missing required columns: symbol,timestamp")
        if target not in label_work.columns:
            raise RuntimeError(f"Target column '{target}' missing from labels dataset.")
        label_work["symbol"] = label_work["symbol"].astype("string").str.upper()
        label_work["timestamp"] = pd.to_datetime(label_work["timestamp"], utc=True, errors="coerce")
        label_work = label_work.dropna(subset=["symbol", "timestamp"]).copy()
        work = pd.merge(
            work,
            label_work.loc[:, ["symbol", "timestamp", target]],
            on=["symbol", "timestamp"],
            how="left",
        )

    for col in feature_columns:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work[feature_columns] = work[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work = work.dropna(subset=[target]).sort_values("timestamp").reset_index(drop=True)
    if work.empty:
        raise RuntimeError("No usable rows after joining features/labels for recalibration.")
    if len(work) > DEFAULT_MAX_ROWS:
        work = work.tail(DEFAULT_MAX_ROWS).reset_index(drop=True)
    return work


def _load_inputs_from_db(run_date: date | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = db.load_ml_artifact_csv("features", run_date=run_date)
    labels = db.load_ml_artifact_csv("labels", run_date=run_date)
    if features.empty and run_date is not None:
        LOG.warning(
            "[WARN] RANKER_RECALIBRATE_RUN_DATE_FALLBACK artifact=features run_date=%s",
            run_date,
        )
        features = db.load_ml_artifact_csv("features")
    if labels.empty and run_date is not None:
        labels = db.load_ml_artifact_csv("labels")
    if features.empty:
        raise FileNotFoundError("Features not found in DB (ml_artifacts: features).")
    return features, labels


def _load_inputs_from_fs(
    features_path: Path | None,
    labels_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path | None]:
    resolved_features = features_path or _find_latest(
        BASE_DIR / "data" / "features", "features_*.csv"
    )
    if resolved_features is None:
        raise FileNotFoundError("No features files found under data/features.")
    resolved_labels = labels_path or _find_latest(BASE_DIR / "data" / "labels", "labels_*.csv")
    features = pd.read_csv(resolved_features)
    labels = pd.DataFrame()
    if resolved_labels and resolved_labels.exists():
        labels = pd.read_csv(resolved_labels)
    return features, labels, resolved_features, resolved_labels


def _build_prefit_calibrator(model: Any, method: str):
    if CalibratedClassifierCV is None:
        raise RuntimeError("CalibratedClassifierCV not available.")
    try:
        return CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    except TypeError:  # pragma: no cover - compatibility shim
        return CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")


def _fallback_score_calibrator(method: str, scores: np.ndarray, labels: np.ndarray):
    if method == "sigmoid":
        if LogisticRegression is None:
            raise RuntimeError("LogisticRegression unavailable for sigmoid fallback calibration.")
        calibrator = LogisticRegression(max_iter=200, solver="lbfgs")
        calibrator.fit(scores.reshape(-1, 1), labels)
        return calibrator
    if IsotonicRegression is None:
        raise RuntimeError("IsotonicRegression unavailable for isotonic fallback calibration.")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(scores, labels)
    return calibrator


def _apply_score_calibrator(scores: np.ndarray, calibrator: Any) -> np.ndarray:
    if hasattr(calibrator, "predict_proba"):
        return np.asarray(calibrator.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)
    if hasattr(calibrator, "predict"):
        return np.asarray(calibrator.predict(scores), dtype=float)
    return np.asarray(scores, dtype=float)


def _to_iso(value: Any) -> str | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def run_recalibrate(args: argparse.Namespace) -> dict[str, Any]:
    method, method_source = _resolve_method(args.calibrate)
    LOG.info(
        "[INFO] RANKER_RECALIBRATE_START target=%s method=%s calibration_fraction=%s",
        args.target,
        method,
        float(args.calibration_fraction),
    )
    model_path = args.model_path or _find_latest(BASE_DIR / "data" / "models", "ranker_*.pkl")
    if model_path is None:
        raise FileNotFoundError("No trained ranker model found in data/models.")
    model, scaler, feature_columns, model_payload, summary_path = _load_model(model_path)
    source_summary_payload = _load_summary_payload(summary_path)
    model_feature_set = (
        str(model_payload.get("feature_set") or source_summary_payload.get("feature_set") or "")
        .strip()
        .lower()
        or None
    )
    model_feature_signature = str(
        model_payload.get("feature_signature")
        or source_summary_payload.get("feature_signature")
        or ""
    ).strip()

    features_path_used: Path | None = None
    labels_path_used: Path | None = None
    if db.db_enabled():
        features_df, labels_df = _load_inputs_from_db(args.run_date)
        inputs_source = "db"
    else:
        features_df, labels_df, features_path_used, labels_path_used = _load_inputs_from_fs(
            args.features_path,
            args.labels_path,
        )
        inputs_source = "fs"
    features_meta_payload = _load_features_meta_payload(
        run_date=args.run_date,
        inputs_source=inputs_source,
        features_path_used=features_path_used,
    )

    frame = _normalize_inputs(
        features_df,
        labels_df,
        target=args.target,
        feature_columns=feature_columns,
    )
    if frame[args.target].nunique() < 2:
        raise RuntimeError("Calibration target contains a single class; cannot recalibrate.")

    computed_feature_signature = compute_feature_signature(feature_columns)
    features_feature_set = (
        str(features_meta_payload.get("feature_set") or "").strip().lower() or None
    )
    features_feature_signature = str(features_meta_payload.get("feature_signature") or "").strip()
    if (
        features_path_used is not None
        and features_meta_payload
        and not meta_matches_features_path(features_meta_payload, features_path_used)
    ):
        LOG.warning(
            "[WARN] RANKER_RECALIBRATE_FEATURE_META_FILE_MISMATCH features_path=%s meta_output=%s meta_file_name=%s",
            features_path_used,
            features_meta_payload.get("output_path"),
            features_meta_payload.get("file_name"),
        )
    if model_feature_signature and model_feature_signature != computed_feature_signature:
        LOG.warning(
            "[WARN] RANKER_RECALIBRATE_MODEL_FEATURE_SIGNATURE_MISMATCH model=%s computed=%s",
            model_feature_signature,
            computed_feature_signature,
        )
    if features_feature_signature and features_feature_signature != computed_feature_signature:
        LOG.warning(
            "[WARN] RANKER_RECALIBRATE_FEATURES_SIGNATURE_MISMATCH features=%s computed=%s",
            features_feature_signature,
            computed_feature_signature,
        )
    resolved_feature_set = model_feature_set or features_feature_set
    resolved_feature_signature = computed_feature_signature

    total_rows = int(frame.shape[0])
    calib_rows = max(1, int(math.ceil(total_rows * float(args.calibration_fraction))))
    if calib_rows >= total_rows:
        calib_rows = max(1, total_rows - 1)
    calibration_df = frame.tail(calib_rows).copy()
    label_horizon = _infer_label_horizon_days(args.target)
    default_embargo = max(label_horizon, DEFAULT_MIN_EMBARGO_DAYS)
    if args.embargo_days is None:
        embargo_days = default_embargo
    else:
        embargo_days = int(args.embargo_days)
        if embargo_days < 0:
            raise RuntimeError("--embargo-days must be >= 0")
    calib_start_ts = pd.Timestamp(calibration_df["timestamp"].min())
    pre_calib_cutoff = calib_start_ts - pd.Timedelta(days=embargo_days)
    reference_rows = int(frame.loc[frame["timestamp"] <= pre_calib_cutoff].shape[0])

    X_calib = calibration_df[feature_columns]
    y_calib = pd.to_numeric(calibration_df[args.target], errors="coerce").astype(int).to_numpy()
    if scaler is not None:
        X_calib_values = scaler.transform(X_calib)
    else:
        X_calib_values = X_calib.to_numpy(dtype=float)
    base_scores = np.asarray(_predict_proba(model, X_calib_values), dtype=float)
    if base_scores.ndim != 1:
        base_scores = base_scores.reshape(-1)
    base_scores = np.clip(base_scores, 0.0, 1.0)

    fit_mode = "score_mapper"
    final_model = model
    posthoc_score_calibrator = None
    try:
        prefit = _build_prefit_calibrator(model, method)
        prefit.fit(X_calib_values, y_calib)
        final_model = prefit
        fit_mode = "prefit_estimator"
    except Exception as exc:
        LOG.warning(
            "[WARN] RANKER_RECALIBRATE_PREFIT_FAILED method=%s err=%s fallback=score_mapper",
            method,
            exc,
        )
        posthoc_score_calibrator = _fallback_score_calibrator(method, base_scores, y_calib)

    if fit_mode == "prefit_estimator":
        calibrated_scores = np.asarray(_predict_proba(final_model, X_calib_values), dtype=float)
    else:
        calibrated_scores = _apply_score_calibrator(base_scores, posthoc_score_calibrator)
    calibrated_scores = np.clip(calibrated_scores, 0.0, 1.0)

    brier_before = float(np.mean((y_calib - base_scores) ** 2))
    brier_after = float(np.mean((y_calib - calibrated_scores) ** 2))

    LOG.info(
        "[INFO] RANKER_RECALIBRATE_FIT method=%s calib_rows=%d embargo_days=%d",
        method,
        int(calibration_df.shape[0]),
        int(embargo_days),
    )

    out_date = args.run_date or datetime.now(timezone.utc).date()
    out_date_str = out_date.isoformat()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = output_dir / f"ranker_{out_date_str}.pkl"
    out_summary_path = output_dir / f"ranker_summary_{out_date_str}.json"

    save_payload = {
        "model": final_model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "feature_set": resolved_feature_set,
        "feature_signature": resolved_feature_signature,
        "calibration_method": method,
        "calibration_applied": True,
        "calibration_window": {
            "start": _to_iso(calibration_df["timestamp"].min()),
            "end": _to_iso(calibration_df["timestamp"].max()),
            "rows": int(calibration_df.shape[0]),
        },
        "embargo_days_used": int(embargo_days),
        "posthoc_score_calibrator": posthoc_score_calibrator,
        "posthoc_calibration_method": method if posthoc_score_calibrator is not None else None,
        "recalibrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "recalibrate_fit_mode": fit_mode,
    }
    joblib.dump(save_payload, out_model_path)

    summary_payload: dict[str, Any] = {
        "target": args.target,
        "model_path": str(out_model_path),
        "feature_columns": feature_columns,
        "feature_count": int(len(feature_columns)),
        "feature_set": resolved_feature_set,
        "feature_signature": resolved_feature_signature,
        "source_model_path": str(model_path),
        "input_source": inputs_source,
        "inputs": {
            "features_path": str(features_path_used)
            if features_path_used is not None
            else "db://ml_artifacts/features",
            "labels_path": str(labels_path_used)
            if labels_path_used is not None
            else "db://ml_artifacts/labels",
        },
        "calibration": {
            "requested_method": method,
            "method": method,
            "method_source": method_source,
            "applied": True,
            "fit_mode": fit_mode,
            "calibration_rows": int(calibration_df.shape[0]),
            "calibration_fraction": float(args.calibration_fraction),
            "embargo_days_used": int(embargo_days),
            "calibration_start": _to_iso(calibration_df["timestamp"].min()),
            "calibration_end": _to_iso(calibration_df["timestamp"].max()),
            "reference_rows_before_calibration": int(reference_rows),
            "brier_before": brier_before,
            "brier_after": brier_after,
        },
        "rows_total": int(total_rows),
        "run_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    artifact_payload: dict[str, Any] = {
        "target": args.target,
        "method": method,
        "method_source": method_source,
        "feature_set": resolved_feature_set,
        "feature_signature": resolved_feature_signature,
        "feature_count": int(len(feature_columns)),
        "calibration_fraction": float(args.calibration_fraction),
        "embargo_days": int(embargo_days),
        "calibration_rows": int(calibration_df.shape[0]),
        "calibration_window": {
            "start": _to_iso(calibration_df["timestamp"].min()),
            "end": _to_iso(calibration_df["timestamp"].max()),
        },
        "reference_rows_before_calibration": int(reference_rows),
        "fit_mode": fit_mode,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "model_path": str(out_model_path),
        "summary_path": str(out_summary_path),
        "source_model_path": str(model_path),
        "feature_count": int(len(feature_columns)),
        "input_source": inputs_source,
        "run_utc": datetime.now(timezone.utc).isoformat(),
    }

    if db.db_enabled():
        if db.upsert_ml_artifact(
            "ranker_recalibrate",
            out_date,
            payload=artifact_payload,
            rows_count=int(calibration_df.shape[0]),
            source="ranker_recalibrate",
            file_name=out_summary_path.name,
        ):
            LOG.info(
                "[INFO] RANKER_RECALIBRATE_DB_WRITTEN artifact_type=ranker_recalibrate run_date=%s",
                out_date,
            )
    else:
        LOG.warning("[WARN] DB_DISABLED ranker_recalibrate_using_fs_fallback=true")

    LOG.info("[INFO] RANKER_RECALIBRATE_END model_path=%s", out_model_path)
    return artifact_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Binary target column used for calibration (default: label_5d_pos_300bp).",
    )
    parser.add_argument(
        "--calibrate",
        choices=sorted(VALID_METHODS),
        default=None,
        help="Calibration method (sigmoid|isotonic). Defaults to env or sigmoid.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
        help="Fraction of latest rows used for calibration (default: 0.1).",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=None,
        help="Embargo gap between older rows and calibration window. Defaults to max(label_horizon,5).",
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Optional artifact/model run date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Optional filesystem features CSV override.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help="Optional filesystem labels CSV override.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional model path override (default latest data/models/ranker_*.pkl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "models",
        help="Directory where recalibrated model+summary are written (default: data/models).",
    )
    parsed = parser.parse_args(argv)
    if not (0.0 < float(parsed.calibration_fraction) < 0.5):
        parser.error("--calibration-fraction must be >0 and <0.5")
    if parsed.embargo_days is not None and int(parsed.embargo_days) < 0:
        parser.error("--embargo-days must be >= 0")
    try:
        parsed.run_date = _parse_run_date(parsed.run_date)
    except ValueError as exc:
        parser.error(str(exc))
    return parsed


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        run_recalibrate(args)
        return 0
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("RANKER_RECALIBRATE_FAILED err=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
