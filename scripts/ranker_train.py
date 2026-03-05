"""Train a simple supervised ranker model on nightly features.

This module loads the most recent feature snapshot (unless overridden),
performs a time-based train/validation split, trains a lightweight
classifier, and writes both the fitted model and a compact metrics
summary to disk.

Usage::

    python -m scripts.ranker_train --features-path data/features/features_2024-01-01.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils.env import load_env  # noqa: E402
from scripts import db  # noqa: E402
from scripts.utils.feature_schema import (
    compute_feature_signature,
    infer_feature_columns_for_ml,
    load_features_meta_for_path,
    meta_matches_features_path,
)

load_env()

LOG = logging.getLogger("ranker_train")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_FEATURE_COLUMNS = [
    "mom_5d",
    "mom_10d",
    "vol_raw",
    "vol_avg_10d",
    "vol_rvol_10d",
]
# Backward-compatible fallback for older model artifacts without feature metadata.
FEATURE_COLUMNS = list(DEFAULT_FEATURE_COLUMNS)
ID_COLUMNS = {"symbol", "timestamp"}
DROP_FEATURE_COLUMNS = {"close"}
VALID_CALIBRATION_METHODS = {"none", "sigmoid", "isotonic"}
DEFAULT_CALIBRATION_METHOD = "none"
DEFAULT_CALIBRATION_FRACTION = 0.1
MAX_TRAIN_ROWS = 50_000


@dataclass
class TrainArgs:
    features_path: Path | None
    label_column: str
    output_dir: Path
    calibrate: str
    calibration_fraction: float


@dataclass
class TrainOutputs:
    model_path: Path
    summary_path: Path
    metrics: dict


class ModelUnavailableError(RuntimeError):
    """Raised when no supported model implementation is available."""


def _ensure_sklearn_available():
    if not SKLEARN_AVAILABLE:
        raise ModelUnavailableError(MISSING_SKLEARN_ERR)


# Lazy imports to allow graceful degradation if sklearn is not installed
try:  # pragma: no cover - environment-dependent
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    import joblib

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    RandomForestClassifier = None
    LogisticRegression = None
    accuracy_score = None
    roc_auc_score = None
    StandardScaler = None
    CalibratedClassifierCV = None
    joblib = None
    SKLEARN_AVAILABLE = False

MISSING_SKLEARN_ERR = "RANKER_TRAIN: scikit-learn not available; install requirements.txt"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _find_latest_features(features_dir: Path) -> Path | None:
    candidates = sorted(features_dir.glob("features_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _extract_features_date(path: Path) -> str:
    match = re.search(r"features_(\d{4}-\d{2}-\d{2})", path.name)
    if match:
        return match.group(1)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _infer_label_horizon_days(label_column: str) -> int:
    match = re.search(r"label_(\d+)d_", str(label_column))
    if not match:
        return 5
    try:
        return max(1, int(match.group(1)))
    except (TypeError, ValueError):
        return 5


def _resolve_calibration_method(
    cli_value: str | None,
) -> tuple[str, str]:
    if cli_value is not None:
        method = str(cli_value).strip().lower()
        if method in VALID_CALIBRATION_METHODS:
            return method, "cli"
        return DEFAULT_CALIBRATION_METHOD, "default"

    env_value = str(os.getenv("JBR_ML_CALIBRATION") or "").strip().lower()
    if env_value:
        if env_value in VALID_CALIBRATION_METHODS:
            return env_value, "env"
        LOG.warning(
            "[WARN] RANKER_TRAIN_CALIBRATION_INVALID env=JBR_ML_CALIBRATION value=%s default=%s",
            env_value,
            DEFAULT_CALIBRATION_METHOD,
        )
    return DEFAULT_CALIBRATION_METHOD, "default"


def _resolve_calibration_fraction(value: float) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        fraction = DEFAULT_CALIBRATION_FRACTION
    if fraction <= 0 or fraction >= 0.5:
        LOG.warning(
            "[WARN] RANKER_TRAIN_CALIBRATION_FRACTION_INVALID value=%s default=%s",
            value,
            DEFAULT_CALIBRATION_FRACTION,
        )
        return DEFAULT_CALIBRATION_FRACTION
    return fraction


def _build_prefit_calibrator(model, method: str):
    if CalibratedClassifierCV is None:
        raise RuntimeError("Calibration requires scikit-learn calibration support.")
    try:
        return CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    except TypeError:  # pragma: no cover - compatibility shim
        return CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")


def _normalize_meta_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except Exception:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}

def _normalize_features_frame(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, list[str]]:
    required_columns = {"symbol", "timestamp", label_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"Features input missing columns: {sorted(missing)}")

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    for column in work.columns:
        if column in ID_COLUMNS:
            continue
        work[column] = pd.to_numeric(work[column], errors="coerce")

    feature_columns = infer_feature_columns_for_ml(
        work,
        label_column=label_column,
        id_columns=tuple(ID_COLUMNS),
        drop_columns=tuple(DROP_FEATURE_COLUMNS),
    )
    if not feature_columns:
        raise RuntimeError("No numeric feature columns found after exclusions.")
    work[feature_columns] = work[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    work[label_column] = pd.to_numeric(work[label_column], errors="coerce")
    work = work.dropna(subset=["timestamp", label_column]).sort_values("timestamp")
    return work.reset_index(drop=True), feature_columns


def _infer_feature_set_from_columns(feature_columns: Iterable[str]) -> str:
    v2_markers = {
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "logret_1d",
        "volatility_10d",
        "atr14",
        "atr_pct",
        "sma_10",
        "sma_20",
        "sma_50",
        "ema_20",
        "dist_sma20",
        "dist_ema20",
        "rsi14",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_bandwidth",
        "vol_ma30",
        "rel_volume",
        "obv",
        "obv_delta",
        "candle_body_pct",
        "candle_upper_wick_pct",
        "candle_lower_wick_pct",
        "candle_doji",
        "candle_hammer",
        "candle_engulfing_bull",
        "candle_engulfing_bear",
    }
    for col in feature_columns:
        if str(col).strip() in v2_markers:
            return "v2"
    return "v1"


def _load_features(path: Path, label_column: str) -> tuple[pd.DataFrame, list[str]]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to read features from {path}: {exc}") from exc

    return _normalize_features_frame(df, label_column)


def _choose_model():
    _ensure_sklearn_available()

    if LogisticRegression is not None:
        LOG.info("Using LogisticRegression (lbfgs, max_iter=50)")
        return LogisticRegression(max_iter=50, solver="lbfgs")

    if RandomForestClassifier is not None:
        LOG.info("Using RandomForestClassifier (n_estimators=50, max_depth=4)")
        return RandomForestClassifier(
            n_estimators=50,
            max_depth=4,
            n_jobs=-1,
            random_state=42,
        )

    raise ModelUnavailableError(MISSING_SKLEARN_ERR)


def _train_model(
    df: pd.DataFrame,
    label_column: str,
    feature_columns: list[str],
    *,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
):
    if df.empty:
        raise RuntimeError("No data available to train the model.")

    if len(df) > MAX_TRAIN_ROWS:
        LOG.info(
            "Truncating training set from %s to %s rows (keeping most recent rows)",
            len(df),
            MAX_TRAIN_ROWS,
        )
        df = df.sort_values("timestamp").tail(MAX_TRAIN_ROWS).reset_index(drop=True)

    split_idx = max(1, int(len(df) * 0.8))
    if split_idx >= len(df):
        split_idx = len(df) - 1

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    horizon_days = _infer_label_horizon_days(label_column)
    requested_calibration_method = str(calibration_method or DEFAULT_CALIBRATION_METHOD).lower()
    effective_calibration_method = (
        requested_calibration_method
        if requested_calibration_method in VALID_CALIBRATION_METHODS
        else DEFAULT_CALIBRATION_METHOD
    )
    calibration_rows = 0
    calibration_start = None
    calibration_end = None
    calibration_applied = False

    train_core_df = train_df
    calibration_df = pd.DataFrame(columns=train_df.columns)
    if effective_calibration_method != "none":
        requested_rows = max(1, int(len(train_df) * float(calibration_fraction)))
        calib_start_idx = max(1, len(train_df) - requested_rows)
        calib_start_ts = pd.to_datetime(train_df.iloc[calib_start_idx]["timestamp"], errors="coerce")
        if pd.isna(calib_start_ts):
            effective_calibration_method = "none"
        else:
            core_cutoff_ts = calib_start_ts - pd.Timedelta(days=horizon_days)
            train_core_df = train_df.loc[train_df["timestamp"] <= core_cutoff_ts].copy()
            calibration_df = train_df.iloc[calib_start_idx:].copy()
            if train_core_df.empty or calibration_df.empty:
                LOG.warning(
                    "[WARN] RANKER_TRAIN_CALIBRATION_SKIPPED reason=insufficient_rows method=%s train_core=%s calib=%s",
                    requested_calibration_method,
                    len(train_core_df),
                    len(calibration_df),
                )
                effective_calibration_method = "none"
                train_core_df = train_df
                calibration_df = pd.DataFrame(columns=train_df.columns)
            else:
                y_calibration = pd.to_numeric(
                    calibration_df[label_column], errors="coerce"
                ).dropna()
                if y_calibration.nunique() < 2:
                    LOG.warning(
                        "[WARN] RANKER_TRAIN_CALIBRATION_SKIPPED reason=single_class method=%s calib_rows=%s",
                        requested_calibration_method,
                        len(calibration_df),
                    )
                    effective_calibration_method = "none"
                    train_core_df = train_df
                    calibration_df = pd.DataFrame(columns=train_df.columns)
                else:
                    calibration_rows = int(len(calibration_df))
                    calibration_start = calibration_df["timestamp"].min()
                    calibration_end = calibration_df["timestamp"].max()

    X_train = train_core_df[feature_columns]
    y_train = train_core_df[label_column]
    X_val = val_df[feature_columns]
    y_val = val_df[label_column]

    model = _choose_model()

    if StandardScaler is not None and isinstance(model, LogisticRegression):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        final_model = model
        if effective_calibration_method != "none":
            X_calib = calibration_df[feature_columns]
            y_calib = calibration_df[label_column]
            X_calib_scaled = scaler.transform(X_calib)
            calibrator = _build_prefit_calibrator(model, effective_calibration_method)
            calibrator.fit(X_calib_scaled, y_calib)
            final_model = calibrator
            calibration_applied = True
        X_val_scaled = scaler.transform(X_val)
        val_probs = _predict_proba(final_model, X_val_scaled)
        train_size = len(train_core_df)
        val_size = len(val_df)
        calibration_info = {
            "requested_method": requested_calibration_method,
            "method": effective_calibration_method,
            "applied": calibration_applied,
            "calibration_rows": int(calibration_rows if calibration_applied else 0),
            "calibration_fraction": float(calibration_fraction),
            "embargo_days_used": int(horizon_days),
            "calibration_start": calibration_start.isoformat() if calibration_start is not None else None,
            "calibration_end": calibration_end.isoformat() if calibration_end is not None else None,
            "train_total_rows": int(len(train_df)),
            "train_core_rows": int(len(train_core_df)),
        }
        return final_model, scaler, val_probs, y_val, train_size, val_size, calibration_info

    model.fit(X_train, y_train)
    final_model = model
    if effective_calibration_method != "none":
        X_calib = calibration_df[feature_columns]
        y_calib = calibration_df[label_column]
        calibrator = _build_prefit_calibrator(model, effective_calibration_method)
        calibrator.fit(X_calib, y_calib)
        final_model = calibrator
        calibration_applied = True

    val_probs = _predict_proba(final_model, X_val)
    train_size = len(train_core_df)
    val_size = len(val_df)
    calibration_info = {
        "requested_method": requested_calibration_method,
        "method": effective_calibration_method,
        "applied": calibration_applied,
        "calibration_rows": int(calibration_rows if calibration_applied else 0),
        "calibration_fraction": float(calibration_fraction),
        "embargo_days_used": int(horizon_days),
        "calibration_start": calibration_start.isoformat() if calibration_start is not None else None,
        "calibration_end": calibration_end.isoformat() if calibration_end is not None else None,
        "train_total_rows": int(len(train_df)),
        "train_core_rows": int(len(train_core_df)),
    }
    return final_model, None, val_probs, y_val, train_size, val_size, calibration_info


def _predict_proba(model, features) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, -1]
    elif hasattr(model, "decision_function"):
        from scipy.special import expit  # lazy import

        probs = expit(model.decision_function(features))
    else:  # pragma: no cover - should not happen for supported models
        preds = model.predict(features)
        probs = pd.Series(preds, dtype=float)
    return pd.Series(probs)


def _compute_metrics(probs: pd.Series, y_true: Iterable) -> dict:
    metrics: dict[str, float | list] = {}
    y_true_series = pd.Series(y_true)

    if accuracy_score is not None:
        pred_labels = (probs >= 0.5).astype(int)
        metrics["accuracy"] = float(accuracy_score(y_true_series, pred_labels))
    else:  # pragma: no cover - environment dependent
        metrics["accuracy"] = float(((probs >= 0.5) == y_true_series).mean())

    if roc_auc_score is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true_series, probs))
        except Exception:
            metrics["auc"] = None
    else:  # pragma: no cover - environment dependent
        metrics["auc"] = None

    metrics["deciles"] = _decile_stats(probs, y_true_series)
    return metrics


def _decile_stats(probs: pd.Series, labels: pd.Series) -> list[dict]:
    if probs.empty:
        return []

    df = pd.DataFrame({"prob": probs, "label": labels}).dropna()
    if df.empty:
        return []

    try:
        df["decile"] = pd.qcut(df["prob"], q=10, labels=False, duplicates="drop")
    except ValueError:
        return []

    df["decile"] = df["decile"].fillna(-1).astype(int)
    grouped = df.groupby("decile")
    stats = []
    for decile, group in grouped:
        if decile < 0:
            continue
        stats.append(
            {
                "decile": int(decile + 1),
                "count": int(len(group)),
                "positive_rate": float(group["label"].mean()),
                "mean_score": float(group["prob"].mean()),
            }
        )
    stats.sort(key=lambda x: x["decile"], reverse=True)
    return stats


def _save_outputs(
    model,
    scaler,
    metrics: dict,
    feature_columns: list[str],
    calibration_info: dict[str, object],
    feature_set: str | None,
    feature_signature: str,
    args: TrainArgs,
    snapshot_date: str,
) -> TrainOutputs:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"ranker_{snapshot_date}.pkl"
    model_path = output_dir / model_filename
    payload = {"model": model}
    if scaler is not None:
        payload["scaler"] = scaler
    payload["feature_columns"] = feature_columns
    payload["feature_set"] = feature_set
    payload["feature_signature"] = feature_signature
    payload["calibration_method"] = calibration_info.get("method", DEFAULT_CALIBRATION_METHOD)
    payload["calibration_applied"] = bool(calibration_info.get("applied", False))
    payload["calibration_window"] = {
        "start": calibration_info.get("calibration_start"),
        "end": calibration_info.get("calibration_end"),
        "rows": int(calibration_info.get("calibration_rows") or 0),
    }
    payload["embargo_days_used"] = int(calibration_info.get("embargo_days_used") or 0)

    if joblib is None:
        raise RuntimeError("joblib is required to persist the model")
    joblib.dump(payload, model_path)

    summary_path = output_dir / f"ranker_summary_{snapshot_date}.json"
    features_path_display = (
        str(args.features_path) if args.features_path is not None else "db://ml_artifacts/features"
    )
    summary = {
        "features_path": features_path_display,
        "label_column": args.label_column,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "feature_set": feature_set,
        "feature_signature": feature_signature,
        "train_size": metrics.pop("train_size", None),
        "val_size": metrics.pop("val_size", None),
        "calibration": {
            "requested_method": calibration_info.get("requested_method"),
            "method": calibration_info.get("method"),
            "applied": bool(calibration_info.get("applied", False)),
            "calibration_rows": int(calibration_info.get("calibration_rows") or 0),
            "calibration_fraction": float(
                calibration_info.get("calibration_fraction") or DEFAULT_CALIBRATION_FRACTION
            ),
            "embargo_days_used": int(calibration_info.get("embargo_days_used") or 0),
            "calibration_start": calibration_info.get("calibration_start"),
            "calibration_end": calibration_info.get("calibration_end"),
            "train_total_rows": int(calibration_info.get("train_total_rows") or 0),
            "train_core_rows": int(calibration_info.get("train_core_rows") or 0),
        },
        "metrics": metrics,
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return TrainOutputs(
        model_path=model_path, summary_path=summary_path, metrics=summary["metrics"]
    )


def parse_args(argv: list[str] | None = None) -> TrainArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Path to a features CSV. Defaults to the latest data/features/features_*.csv",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Alias for --label-column.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label_5d_pos_300bp",
        help="Binary label column to train against",
    )
    parser.add_argument(
        "--calibrate",
        choices=sorted(VALID_CALIBRATION_METHODS),
        default=None,
        help=(
            "Probability calibration method. Precedence: CLI > JBR_ML_CALIBRATION > default none."
        ),
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
        help="Fraction of training rows reserved for calibration window (default 0.1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "models",
        help="Directory to write the trained model and summary",
    )
    parsed = parser.parse_args(argv)

    calibrate, calibration_source = _resolve_calibration_method(parsed.calibrate)
    calibration_fraction = _resolve_calibration_fraction(parsed.calibration_fraction)
    label_column = str(parsed.target or parsed.label_column)
    LOG.info(
        "[INFO] RANKER_TRAIN_CALIBRATION_CONFIG method=%s source=%s calibration_fraction=%s",
        calibrate,
        calibration_source,
        calibration_fraction,
    )

    features_path = parsed.features_path
    if not db.db_enabled():
        if features_path is None:
            default_dir = BASE_DIR / "data" / "features"
            features_path = _find_latest_features(default_dir)
            if features_path is None:
                raise FileNotFoundError(f"No features files found in {default_dir}")

    return TrainArgs(
        features_path=features_path,
        label_column=label_column,
        output_dir=parsed.output_dir,
        calibrate=calibrate,
        calibration_fraction=calibration_fraction,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1

    features_meta_record = None
    features_meta_payload: dict[str, Any] = {}
    features_meta_source = "missing"
    if db.db_enabled():
        features_meta_record = db.fetch_latest_ml_artifact("features")
        if not features_meta_record:
            LOG.error("No features artifacts found in DB (ml_artifacts: features)")
            return 1
        features_meta_payload = _normalize_meta_payload(features_meta_record.get("payload"))
        features_meta_source = "db:latest"
        LOG.info("Loading features from DB (ml_artifacts: features)")
        try:
            df, feature_columns = _normalize_features_frame(
                db.load_ml_artifact_csv("features"), args.label_column
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOG.error("Failed to load features from DB: %s", exc)
            return 1
    else:
        LOG.info("Loading features from %s", args.features_path)
        features_meta_payload, features_meta_source = load_features_meta_for_path(
            args.features_path,
            base_dir=BASE_DIR,
            prefer_db=False,
        )
        try:
            df, feature_columns = _load_features(args.features_path, args.label_column)
        except FileNotFoundError:
            LOG.error("Features file not found: %s", args.features_path)
            return 1
        except Exception as exc:  # pragma: no cover - defensive
            LOG.error("Failed to load features: %s", exc)
            return 1

    if df.empty:
        LOG.error("Features input has no usable rows")
        return 1

    if not features_meta_payload:
        LOG.warning(
            "[WARN] RANKER_TRAIN_FEATURE_META_MISSING source=%s",
            features_meta_source,
        )
    elif args.features_path is not None and not meta_matches_features_path(
        features_meta_payload, args.features_path
    ):
        LOG.warning(
            "[WARN] RANKER_TRAIN_FEATURE_META_FILE_MISMATCH source=%s features_path=%s meta_output=%s meta_file_name=%s",
            features_meta_source,
            args.features_path,
            features_meta_payload.get("output_path"),
            features_meta_payload.get("file_name"),
        )

    computed_feature_signature = compute_feature_signature(feature_columns)
    feature_set_meta = str(features_meta_payload.get("feature_set") or "").strip().lower() or None
    env_feature_set = str(os.getenv("JBR_ML_FEATURE_SET") or "").strip().lower() or None
    if env_feature_set not in {"v1", "v2"}:
        env_feature_set = None
    if env_feature_set and feature_set_meta and env_feature_set != feature_set_meta:
        LOG.warning(
            "[WARN] RANKER_TRAIN_FEATURE_SET_OVERRIDE env=%s meta=%s",
            env_feature_set,
            feature_set_meta,
        )
    feature_set = env_feature_set or feature_set_meta or _infer_feature_set_from_columns(feature_columns)
    feature_signature_meta = str(features_meta_payload.get("feature_signature") or "").strip()
    if feature_signature_meta and feature_signature_meta != computed_feature_signature:
        LOG.warning(
            "[WARN] RANKER_TRAIN_FEATURE_SIGNATURE_MISMATCH meta=%s computed=%s",
            feature_signature_meta,
            computed_feature_signature,
        )
    feature_signature = computed_feature_signature

    try:
        model, scaler, val_probs, y_val, train_size, val_size, calibration_info = _train_model(
            df,
            args.label_column,
            feature_columns,
            calibration_method=args.calibrate,
            calibration_fraction=args.calibration_fraction,
        )
    except ModelUnavailableError as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("Training failed: %s", exc)
        return 1

    metrics = _compute_metrics(val_probs, y_val)
    metrics["train_size"] = train_size
    metrics["val_size"] = val_size
    LOG.info(
        "[INFO] RANKER_TRAIN_CALIBRATION method=%s calib_rows=%s embargo_days=%s",
        calibration_info.get("method", DEFAULT_CALIBRATION_METHOD),
        int(calibration_info.get("calibration_rows") or 0),
        int(calibration_info.get("embargo_days_used") or 0),
    )

    accuracy_display = metrics.get("accuracy")
    auc_display = metrics.get("auc")
    LOG.info(
        "Features=%s | FeatureCount=%s | Train=%s | Val=%s | Accuracy=%s | AUC=%s",
        args.features_path,
        len(feature_columns),
        train_size,
        val_size,
        f"{accuracy_display:.4f}" if accuracy_display is not None else "n/a",
        f"{auc_display:.4f}" if auc_display not in (None, "nan") else "n/a",
    )

    if db.db_enabled() and features_meta_record:
        snapshot_date = str(features_meta_record.get("run_date") or datetime.now(timezone.utc).date())
    else:
        snapshot_date = _extract_features_date(args.features_path)
    try:
        outputs = _save_outputs(
            model,
            scaler,
            metrics,
            feature_columns,
            calibration_info,
            feature_set,
            feature_signature,
            args,
            snapshot_date,
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("Failed to save outputs: %s", exc)
        return 1

    LOG.info("Model saved to %s", outputs.model_path)
    LOG.info("Summary written to %s", outputs.summary_path)
    LOG.info("Validation metrics: %s", json.dumps(outputs.metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
