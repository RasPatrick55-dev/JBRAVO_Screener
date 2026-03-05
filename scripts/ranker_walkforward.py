"""Walk-forward, time-series-safe evaluation for the ML ranker.

This script evaluates ranking quality and strategy-like outcomes across
multiple rolling out-of-sample folds with an embargo gap to reduce label
overlap leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts import db  # noqa: E402
from scripts.utils.fwd_return_sanity import (  # noqa: E402
    clip_forward_returns,
    log_forward_return_sanity,
)
from utils.env import load_env  # noqa: E402

load_env()

LOG = logging.getLogger("ranker_walkforward")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

try:  # pragma: no cover - environment-dependent
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    CalibratedClassifierCV = None
    LogisticRegression = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

DEFAULT_TARGET = "label_5d_pos_300bp"
DEFAULT_SCORE_COL = "score_5d"
OOS_SCORE_COL = "score_oos"


@dataclass
class WalkforwardArgs:
    target: str
    test_window_days: int
    train_window_days: int
    step_days: int
    embargo_days: int | None
    top_k: int
    score_col: str
    retrain_per_fold: bool
    calibrate: str
    max_abs_fwd_ret: float
    run_date: date | None
    output_dir: Path
    features_path: Path | None
    labels_path: Path | None
    predictions_path: Path | None


def _find_latest(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _to_iso(ts_value: Any) -> str | None:
    if ts_value is None:
        return None
    try:
        ts = pd.Timestamp(ts_value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _infer_label_horizon_days(target: str) -> int:
    match = re.search(r"label_(\d+)d", str(target or ""))
    if not match:
        return 0
    try:
        return max(int(match.group(1)), 0)
    except (TypeError, ValueError):
        return 0


def _parse_run_date(value: str | None) -> date | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --run-date: {value}")
    return parsed.date()


def _as_utc_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _require_columns(df: pd.DataFrame, required: set[str], frame_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{frame_name} missing columns: {sorted(missing)}")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, {"symbol", "timestamp"}, "features")
    out = df.copy()
    out["symbol"] = out["symbol"].astype("string").str.upper()
    out["timestamp"] = _as_utc_timestamp(out["timestamp"])
    out = out.dropna(subset=["symbol", "timestamp"]).sort_values(["symbol", "timestamp"])
    return out.reset_index(drop=True)


def _normalize_labels(df: pd.DataFrame, target: str) -> pd.DataFrame:
    _require_columns(df, {"symbol", "timestamp", target}, "labels")
    out = df.copy()
    out["symbol"] = out["symbol"].astype("string").str.upper()
    out["timestamp"] = _as_utc_timestamp(out["timestamp"])
    out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out.dropna(subset=["symbol", "timestamp", target]).sort_values(["symbol", "timestamp"])
    return out.reset_index(drop=True)


def _normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    _require_columns(df, {"symbol", "timestamp"}, "predictions")
    out = df.copy()
    out["symbol"] = out["symbol"].astype("string").str.upper()
    out["timestamp"] = _as_utc_timestamp(out["timestamp"])
    out = out.dropna(subset=["symbol", "timestamp"]).sort_values(["symbol", "timestamp"])
    return out.reset_index(drop=True)


def _select_fwd_ret_column(labels: pd.DataFrame, target: str) -> str | None:
    horizon = _infer_label_horizon_days(target)
    preferred = f"fwd_ret_{horizon}d" if horizon > 0 else None
    if preferred and preferred in labels.columns:
        return preferred
    candidates = [col for col in labels.columns if col.startswith("fwd_ret_")]
    if not candidates:
        return None
    return sorted(candidates)[0]


def _resolve_fs_paths(
    features_path: Path | None,
    labels_path: Path | None,
    predictions_path: Path | None,
) -> tuple[Path, Path, Path | None]:
    features = features_path or _find_latest(BASE_DIR / "data" / "features", "features_*.csv")
    labels = labels_path or _find_latest(BASE_DIR / "data" / "labels", "labels_*.csv")
    predictions = predictions_path or _find_latest(
        BASE_DIR / "data" / "predictions", "predictions_*.csv"
    )
    if features is None:
        raise FileNotFoundError("No features files found in data/features")
    if labels is None:
        raise FileNotFoundError("No labels files found in data/labels")
    return features, labels, predictions


def _load_inputs(
    args: WalkforwardArgs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None, date | None]:
    artifact_run_date: date | None = args.run_date
    if db.db_enabled():
        features = db.load_ml_artifact_csv("features", run_date=args.run_date)
        labels = db.load_ml_artifact_csv("labels", run_date=args.run_date)
        predictions = db.load_ml_artifact_csv("predictions", run_date=args.run_date)
        if features.empty and args.run_date is not None:
            LOG.warning(
                "[WARN] WALKFORWARD_RUN_DATE_FALLBACK artifact=features run_date=%s",
                args.run_date,
            )
            features = db.load_ml_artifact_csv("features")
        if labels.empty and args.run_date is not None:
            LOG.warning(
                "[WARN] WALKFORWARD_RUN_DATE_FALLBACK artifact=labels run_date=%s",
                args.run_date,
            )
            labels = db.load_ml_artifact_csv("labels")
        if predictions.empty and args.run_date is not None:
            predictions = db.load_ml_artifact_csv("predictions")
        if features.empty:
            raise FileNotFoundError("Features not found in DB (ml_artifacts: features)")
        if labels.empty:
            raise FileNotFoundError("Labels not found in DB (ml_artifacts: labels)")
        if artifact_run_date is None:
            features_meta = db.fetch_latest_ml_artifact("features")
            if features_meta and features_meta.get("run_date") is not None:
                artifact_run_date = pd.to_datetime(features_meta.get("run_date")).date()
    else:
        features_path, labels_path, predictions_path = _resolve_fs_paths(
            args.features_path,
            args.labels_path,
            args.predictions_path,
        )
        LOG.info("Loading features from %s", features_path)
        LOG.info("Loading labels from %s", labels_path)
        features = _load_csv(features_path)
        labels = _load_csv(labels_path)
        predictions = _load_csv(predictions_path) if predictions_path else pd.DataFrame()
        if predictions_path:
            LOG.info("Loading predictions from %s", predictions_path)

    labels_norm = _normalize_labels(labels, args.target)
    fwd_ret_col = _select_fwd_ret_column(labels_norm, args.target)
    label_columns = ["symbol", "timestamp", args.target]
    if fwd_ret_col:
        labels_norm[fwd_ret_col] = pd.to_numeric(labels_norm[fwd_ret_col], errors="coerce")
        label_columns.append(fwd_ret_col)
    labels_subset = labels_norm[label_columns].copy()

    features_norm = _normalize_features(features)
    # Labels are merged from the canonical labels artifact to keep target/forward
    # return columns consistent and unsuffixed.
    drop_from_features = [
        col
        for col in features_norm.columns
        if col == args.target or col.startswith("label_") or col.startswith("fwd_ret_")
    ]
    if drop_from_features:
        features_norm = features_norm.drop(columns=drop_from_features, errors="ignore")
    merged = features_norm.merge(labels_subset, on=["symbol", "timestamp"], how="inner")
    if merged.empty:
        raise RuntimeError("Merged features/labels is empty. Check symbol/timestamp alignment.")

    predictions_norm = _normalize_predictions(predictions)
    if not predictions_norm.empty and args.score_col in predictions_norm.columns:
        predictions_norm[args.score_col] = pd.to_numeric(
            predictions_norm[args.score_col], errors="coerce"
        )
        merged = merged.merge(
            predictions_norm[["symbol", "timestamp", args.score_col]],
            on=["symbol", "timestamp"],
            how="left",
        )

    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return merged, features_norm, labels_norm, fwd_ret_col, artifact_run_date


def _resolve_feature_columns(df: pd.DataFrame, target: str, score_col: str) -> list[str]:
    numeric_columns = list(df.select_dtypes(include=["number", "bool"]).columns)
    excluded = {
        target,
        score_col,
        "close",
    }
    resolved: list[str] = []
    for column in df.columns:
        if column not in numeric_columns:
            continue
        if column in excluded:
            continue
        if column.startswith("label_") or column.startswith("fwd_ret_"):
            continue
        if column.startswith("score_") or column.startswith("pred_"):
            continue
        resolved.append(column)
    if not resolved:
        raise RuntimeError("No usable numeric feature columns found for fold model scoring.")
    return resolved


def _safe_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    y_true = pd.to_numeric(labels, errors="coerce")
    y_score = pd.to_numeric(scores, errors="coerce")
    valid = y_true.notna() & y_score.notna()
    y_true = y_true.loc[valid]
    y_score = y_score.loc[valid]
    if y_true.empty:
        return None

    positives = int((y_true > 0.5).sum())
    negatives = int((y_true <= 0.5).sum())
    if positives == 0 or negatives == 0:
        return None

    rank = y_score.rank(method="average")
    pos_rank_sum = float(rank[y_true > 0.5].sum())
    auc = (pos_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float | None:
    try:
        corr = x.corr(y, method="spearman")
    except Exception:
        return None
    if corr is None or (isinstance(corr, float) and math.isnan(corr)):
        return None
    return float(corr)


def _predict_fold_proba(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
) -> np.ndarray:
    y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0).astype(int)
    if y_train.nunique(dropna=True) < 2:
        return np.full(test_df.shape[0], float(y_train.mean() if not y_train.empty else 0.0))

    X_train = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = test_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if SKLEARN_AVAILABLE and LogisticRegression is not None:
        if StandardScaler is not None:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.to_numpy()
            X_test_scaled = X_test.to_numpy()
        model = LogisticRegression(max_iter=100, solver="lbfgs")
        model.fit(X_train_scaled, y_train)
        return model.predict_proba(X_test_scaled)[:, -1]

    # Deterministic fallback if sklearn is unavailable.
    return np.full(test_df.shape[0], float(y_train.mean()))


def _predict_fold_proba_retrained(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    calibrate: str,
) -> tuple[np.ndarray, str, str]:
    y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0).astype(int)
    if y_train.nunique(dropna=True) < 2:
        base_prob = float(y_train.mean() if not y_train.empty else 0.0)
        return (
            np.full(test_df.shape[0], base_prob),
            "constant_baseline",
            "none",
        )

    X_train = train_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = test_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if not SKLEARN_AVAILABLE or LogisticRegression is None:
        return (
            np.full(test_df.shape[0], float(y_train.mean())),
            "constant_baseline",
            "none",
        )

    X_train_fit = X_train.to_numpy()
    X_test_fit = X_test.to_numpy()
    if StandardScaler is not None:
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train)
        X_test_fit = scaler.transform(X_test)

    base_model = LogisticRegression(max_iter=100, solver="lbfgs")
    calibrate_requested = (calibrate or "none").strip().lower()
    calibration_used = "none"
    model_type = "logistic_regression"

    if calibrate_requested in {"sigmoid", "isotonic"} and CalibratedClassifierCV is not None:
        class_counts = y_train.value_counts(dropna=True)
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0
        cv_folds = min(3, min_class_count)
        if cv_folds >= 2:
            try:
                try:
                    calibrated_model = CalibratedClassifierCV(
                        estimator=base_model,
                        method=calibrate_requested,
                        cv=cv_folds,
                    )
                except TypeError:
                    calibrated_model = CalibratedClassifierCV(
                        base_estimator=base_model,
                        method=calibrate_requested,
                        cv=cv_folds,
                    )
                calibrated_model.fit(X_train_fit, y_train)
                return (
                    calibrated_model.predict_proba(X_test_fit)[:, -1],
                    "logistic_regression_calibrated",
                    calibrate_requested,
                )
            except Exception as exc:  # pragma: no cover - defensive
                LOG.warning(
                    "[WARN] WALKFORWARD_CALIBRATION_FALLBACK method=%s reason=%s",
                    calibrate_requested,
                    exc,
                )
        else:
            LOG.warning(
                "[WARN] WALKFORWARD_CALIBRATION_SKIPPED method=%s reason=insufficient_class_counts",
                calibrate_requested,
            )

    base_model.fit(X_train_fit, y_train)
    probs = base_model.predict_proba(X_test_fit)[:, -1]
    return probs, model_type, calibration_used


def _fold_metrics(
    fold_df: pd.DataFrame,
    target: str,
    score_col: str,
    top_k: int,
    fwd_ret_col: str | None,
) -> dict[str, Any]:
    work = fold_df.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[target, score_col]).sort_values(score_col, ascending=False)
    if work.empty:
        return {
            "sample_size": 0,
            "auc": None,
            "brier": None,
            "spearman": None,
            "precision_at_k": None,
            "top_k_count": 0,
            "top_k_mean_fwd_ret": None,
            "top_k_win_rate": None,
            "test_mean_fwd_ret": None,
            "test_win_rate": None,
            "top_k_excess_mean_fwd_ret": None,
        }

    labels = work[target].astype(float)
    scores = work[score_col].astype(float)
    sample_size = int(work.shape[0])
    k = max(1, min(int(top_k), sample_size))
    top_k_df = work.head(k)

    brier = float(((labels - scores) ** 2).mean())
    precision_at_k = float(top_k_df[target].mean())
    spearman_series = scores
    if fwd_ret_col and fwd_ret_col in work.columns:
        work[fwd_ret_col] = pd.to_numeric(work[fwd_ret_col], errors="coerce")
        spearman_series = pd.to_numeric(work[fwd_ret_col], errors="coerce")
    spearman = _safe_spearman(scores, spearman_series)

    top_k_mean_ret = None
    top_k_win_rate = None
    test_mean_ret = None
    test_win_rate = None
    top_k_excess = None
    if fwd_ret_col and fwd_ret_col in work.columns:
        ret = pd.to_numeric(work[fwd_ret_col], errors="coerce")
        top_ret = pd.to_numeric(top_k_df[fwd_ret_col], errors="coerce")
        if ret.notna().any():
            test_mean_ret = float(ret.mean())
            test_win_rate = float((ret > 0).mean())
        if top_ret.notna().any():
            top_k_mean_ret = float(top_ret.mean())
            top_k_win_rate = float((top_ret > 0).mean())
        if top_k_mean_ret is not None and test_mean_ret is not None:
            top_k_excess = float(top_k_mean_ret - test_mean_ret)

    return {
        "sample_size": sample_size,
        "auc": _safe_auc(labels, scores),
        "brier": brier,
        "spearman": spearman,
        "precision_at_k": precision_at_k,
        "top_k_count": int(k),
        "top_k_mean_fwd_ret": top_k_mean_ret,
        "top_k_win_rate": top_k_win_rate,
        "test_mean_fwd_ret": test_mean_ret,
        "test_win_rate": test_win_rate,
        "top_k_excess_mean_fwd_ret": top_k_excess,
    }


def _aggregate_fold_values(folds: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values: list[float] = []
    for fold in folds:
        value = fold.get(key)
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(num):
            continue
        values.append(num)
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "median": None}
    series = pd.Series(values, dtype="float64")
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
    }


def run_walkforward(args: WalkforwardArgs) -> dict[str, Any]:
    frame, _features_frame, _labels_frame, fwd_ret_col, artifact_run_date = _load_inputs(args)
    frame["timestamp"] = _as_utc_timestamp(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(
        drop=True
    )
    if args.run_date is not None:
        cutoff_ts = pd.Timestamp(args.run_date).tz_localize("UTC") + pd.Timedelta(days=1)
        frame = frame.loc[frame["timestamp"] < cutoff_ts].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No rows available for walk-forward evaluation.")

    fwd_ret_sanity: dict[str, Any] | None = None
    clipped_rows = 0
    if fwd_ret_col and fwd_ret_col in frame.columns:
        fwd_ret_sanity = log_forward_return_sanity(
            frame[fwd_ret_col],
            column_name=fwd_ret_col,
            logger=LOG,
        )
        if float(args.max_abs_fwd_ret or 0.0) > 0:
            frame, clipped_rows = clip_forward_returns(
                frame,
                column_name=fwd_ret_col,
                max_abs=float(args.max_abs_fwd_ret),
                logger=LOG,
            )

    horizon_days = _infer_label_horizon_days(args.target)
    requested_embargo = int(args.embargo_days) if args.embargo_days is not None else max(horizon_days, 5)
    effective_embargo = max(requested_embargo, horizon_days)

    requested_score_col = str(args.score_col).strip() or DEFAULT_SCORE_COL
    has_precomputed_score = (
        bool(requested_score_col)
        and requested_score_col in frame.columns
        and pd.to_numeric(frame[requested_score_col], errors="coerce").notna().any()
    )
    retrain_active = bool(args.retrain_per_fold) or not has_precomputed_score
    score_source = "retrained_per_fold" if retrain_active else "precomputed"
    if retrain_active and not args.retrain_per_fold and not has_precomputed_score:
        LOG.warning(
            "[WARN] WALKFORWARD_SCORE_FALLBACK reason=missing_precomputed_scores score_col=%s",
            requested_score_col,
        )

    score_col = OOS_SCORE_COL if retrain_active else requested_score_col
    feature_columns: list[str] = []
    calibrate_mode = (args.calibrate or "none").strip().lower()
    if retrain_active:
        feature_columns = _resolve_feature_columns(frame, args.target, requested_score_col)
        LOG.info(
            "[INFO] WALKFORWARD_RETRAIN enabled=true model=logistic_regression calibrate=%s features=%d",
            calibrate_mode,
            len(feature_columns),
        )
    elif calibrate_mode != "none":
        LOG.warning(
            "[WARN] WALKFORWARD_CALIBRATION_IGNORED reason=precomputed_scores calibrate=%s",
            calibrate_mode,
        )

    min_ts = pd.Timestamp(frame["timestamp"].min())
    max_ts = pd.Timestamp(frame["timestamp"].max())
    first_test_start = min_ts + pd.Timedelta(days=args.train_window_days + effective_embargo)
    if first_test_start > max_ts:
        raise RuntimeError(
            "Not enough timestamp history to build any fold. "
            f"Need >= train_window_days({args.train_window_days}) + embargo({effective_embargo})."
        )

    LOG.info(
        "[INFO] WALKFORWARD_START target=%s score_col=%s score_source=%s train_window_days=%d test_window_days=%d step_days=%d embargo_days=%d",
        args.target,
        requested_score_col,
        score_source,
        args.train_window_days,
        args.test_window_days,
        args.step_days,
        effective_embargo,
    )

    folds: list[dict[str, Any]] = []
    oos_frames: list[pd.DataFrame] = []
    model_types_used: set[str] = set()
    calibrations_used: set[str] = set()
    fold_index = 0
    test_start = first_test_start
    while test_start <= max_ts:
        fold_index += 1
        test_end = min(test_start + pd.Timedelta(days=args.test_window_days - 1), max_ts)
        train_end_exclusive = test_start - pd.Timedelta(days=effective_embargo)
        train_start = train_end_exclusive - pd.Timedelta(days=args.train_window_days)

        train_df = frame.loc[
            (frame["timestamp"] >= train_start) & (frame["timestamp"] < train_end_exclusive)
        ].copy()
        test_df = frame.loc[(frame["timestamp"] >= test_start) & (frame["timestamp"] <= test_end)].copy()

        if train_df.empty or test_df.empty:
            test_start = test_start + pd.Timedelta(days=args.step_days)
            continue

        fold_model_type = "precomputed_score"
        fold_calibration = "none"
        if retrain_active:
            probs, fold_model_type, fold_calibration = _predict_fold_proba_retrained(
                train_df,
                test_df,
                feature_columns,
                args.target,
                calibrate_mode,
            )
            test_df[score_col] = probs
            model_types_used.add(fold_model_type)
            calibrations_used.add(fold_calibration)
        else:
            test_df[score_col] = pd.to_numeric(test_df[requested_score_col], errors="coerce")
            model_types_used.add("precomputed_score")
            calibrations_used.add("none")

        fold_metrics = _fold_metrics(
            test_df,
            target=args.target,
            score_col=score_col,
            top_k=args.top_k,
            fwd_ret_col=fwd_ret_col,
        )
        fold_payload = {
            "fold": fold_index,
            "train_start": _to_iso(train_df["timestamp"].min()),
            "train_end": _to_iso(train_df["timestamp"].max()),
            "test_start": _to_iso(test_df["timestamp"].min()),
            "test_end": _to_iso(test_df["timestamp"].max()),
            "score_source": score_source,
            "model_type": fold_model_type,
            "calibration": fold_calibration,
            **fold_metrics,
        }
        folds.append(fold_payload)
        LOG.info(
            "[INFO] WALKFORWARD_FOLD fold=%d train_start=%s train_end=%s test_start=%s test_end=%s auc=%s topk_mean_fwd_ret=%s topk_win_rate=%s",
            fold_payload["fold"],
            fold_payload["train_start"],
            fold_payload["train_end"],
            fold_payload["test_start"],
            fold_payload["test_end"],
            fold_payload["auc"],
            fold_payload["top_k_mean_fwd_ret"],
            fold_payload["top_k_win_rate"],
        )

        fold_oos = pd.DataFrame()
        for column in ("symbol", "timestamp", "close", args.target):
            if column in test_df.columns:
                fold_oos[column] = test_df[column]
            else:
                fold_oos[column] = np.nan
        if fwd_ret_col:
            if fwd_ret_col in test_df.columns:
                fold_oos[fwd_ret_col] = pd.to_numeric(test_df[fwd_ret_col], errors="coerce")
            else:
                fold_oos[fwd_ret_col] = np.nan
        fold_oos["fold_id"] = int(fold_index)
        fold_oos[OOS_SCORE_COL] = pd.to_numeric(test_df[score_col], errors="coerce")
        fold_oos["score_source"] = score_source
        oos_frames.append(fold_oos)

        test_start = test_start + pd.Timedelta(days=args.step_days)

    if not folds:
        raise RuntimeError("No non-empty folds generated after rolling-window splits.")

    oos_predictions = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    oos_predictions = oos_predictions.sort_values(["timestamp", "symbol", "fold_id"]).reset_index(
        drop=True
    )
    overall_oos_metrics = _fold_metrics(
        oos_predictions,
        target=args.target,
        score_col=OOS_SCORE_COL,
        top_k=args.top_k,
        fwd_ret_col=fwd_ret_col,
    )

    summary = {
        "auc": _aggregate_fold_values(folds, "auc"),
        "brier": _aggregate_fold_values(folds, "brier"),
        "spearman": _aggregate_fold_values(folds, "spearman"),
        "precision_at_k": _aggregate_fold_values(folds, "precision_at_k"),
        "top_k_mean_fwd_ret": _aggregate_fold_values(folds, "top_k_mean_fwd_ret"),
        "top_k_win_rate": _aggregate_fold_values(folds, "top_k_win_rate"),
        "top_k_excess_mean_fwd_ret": _aggregate_fold_values(folds, "top_k_excess_mean_fwd_ret"),
    }
    sample_total = int(sum(int(fold.get("sample_size", 0) or 0) for fold in folds))
    model_types_sorted = sorted(model_types_used)
    calibrations_sorted = sorted(calibrations_used)
    model_type_value = (
        model_types_sorted[0]
        if len(model_types_sorted) == 1
        else "mixed"
        if model_types_sorted
        else "unknown"
    )
    calibration_value = (
        calibrations_sorted[0]
        if len(calibrations_sorted) == 1
        else "mixed"
        if calibrations_sorted
        else "none"
    )

    payload: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "validation_scheme": "walk_forward_embargo",
        "target": args.target,
        "score_col": requested_score_col,
        "oos_score_col": OOS_SCORE_COL,
        "score_source": score_source,
        "fwd_return_column": fwd_ret_col,
        "fwd_return_sanity": fwd_ret_sanity or {},
        "max_abs_fwd_ret": float(args.max_abs_fwd_ret or 0.0),
        "fwd_ret_clipped_rows": int(clipped_rows),
        "model_type": model_type_value,
        "model_types": model_types_sorted,
        "calibration": calibrate_mode if retrain_active else "none",
        "calibration_used": calibration_value,
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "retrain_per_fold": bool(args.retrain_per_fold),
        "retrain_per_fold_effective": bool(retrain_active),
        "train_window_days": int(args.train_window_days),
        "test_window_days": int(args.test_window_days),
        "step_days": int(args.step_days),
        "requested_embargo_days": int(requested_embargo),
        "embargo_days": int(effective_embargo),
        "label_horizon_days": int(horizon_days),
        "run_date": str(args.run_date) if args.run_date else None,
        "data_start": _to_iso(frame["timestamp"].min()),
        "data_end": _to_iso(frame["timestamp"].max()),
        "population_size": int(frame.shape[0]),
        "windows": {
            "train_window_days": int(args.train_window_days),
            "test_window_days": int(args.test_window_days),
            "step_days": int(args.step_days),
            "embargo_days": int(effective_embargo),
        },
        "fold_count": int(len(folds)),
        "folds_count": int(len(folds)),
        "sample_size_total": sample_total,
        "oos_prediction_rows": int(oos_predictions.shape[0]),
        "folds": folds,
        "summary": summary,
        "overall_oos_metrics": overall_oos_metrics,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    ordered_oos_cols = ["symbol", "timestamp", "close", args.target]
    if fwd_ret_col:
        ordered_oos_cols.append(fwd_ret_col)
    ordered_oos_cols.extend(["fold_id", OOS_SCORE_COL, "score_source"])
    for column in ordered_oos_cols:
        if column not in oos_predictions.columns:
            oos_predictions[column] = np.nan
    oos_output = oos_predictions[ordered_oos_cols].copy()
    oos_output["timestamp"] = _as_utc_timestamp(oos_output["timestamp"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    oos_path = output_dir / "oos_predictions.csv"
    oos_output.to_csv(oos_path, index=False)
    LOG.info(
        "[INFO] WALKFORWARD_OOS_WRITTEN path=%s rows=%d",
        oos_path,
        int(oos_output.shape[0]),
    )

    run_date_for_artifact = artifact_run_date or args.run_date or datetime.now(timezone.utc).date()
    if db.db_enabled():
        ok = db.upsert_ml_artifact(
            "ranker_walkforward",
            run_date_for_artifact,
            payload=payload,
            rows_count=int(len(folds)),
            source="ranker_walkforward",
            file_name=output_path.name,
        )
        if ok:
            LOG.info(
                "[INFO] WALKFORWARD_DB_WRITTEN artifact_type=ranker_walkforward run_date=%s rows=%d",
                run_date_for_artifact,
                int(len(folds)),
            )
        else:
            LOG.warning(
                "[WARN] WALKFORWARD_DB_WRITE_FAILED artifact_type=ranker_walkforward run_date=%s",
                run_date_for_artifact,
            )
        ok_oos = db.upsert_ml_artifact_frame(
            "ranker_oos_predictions",
            run_date_for_artifact,
            oos_output,
            source="ranker_walkforward",
            file_name=oos_path.name,
        )
        if ok_oos:
            LOG.info(
                "[INFO] WALKFORWARD_DB_WRITTEN artifact_type=ranker_oos_predictions run_date=%s rows=%d",
                run_date_for_artifact,
                int(oos_output.shape[0]),
            )
        else:
            LOG.warning(
                "[WARN] WALKFORWARD_DB_WRITE_FAILED artifact_type=ranker_oos_predictions run_date=%s",
                run_date_for_artifact,
            )

    LOG.info(
        "[INFO] WALKFORWARD_END folds=%d sample_size_total=%d auc_mean=%s topk_mean_fwd_ret_mean=%s topk_win_rate_mean=%s output=%s",
        len(folds),
        sample_total,
        (payload["summary"]["auc"] or {}).get("mean"),
        (payload["summary"]["top_k_mean_fwd_ret"] or {}).get("mean"),
        (payload["summary"]["top_k_win_rate"] or {}).get("mean"),
        output_path,
    )
    return payload


def parse_args(argv: list[str] | None = None) -> WalkforwardArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Label target column (default: label_5d_pos_300bp)",
    )
    parser.add_argument(
        "--test-window-days",
        type=int,
        default=63,
        help="Calendar days in each test fold window (default: 63)",
    )
    parser.add_argument(
        "--train-window-days",
        type=int,
        default=504,
        help="Calendar days in each train fold window (default: 504)",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=21,
        help="Calendar day increment between test window starts (default: 21)",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=None,
        help=(
            "Optional embargo gap in days. Defaults to max(label_horizon, 5). "
            "Effective embargo is always >= inferred target horizon."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Top-k slice for quality/strategy metrics (default: 25)",
    )
    parser.add_argument(
        "--score-col",
        type=str,
        default=DEFAULT_SCORE_COL,
        help="Score column to evaluate if available (default: score_5d)",
    )
    parser.add_argument(
        "--retrain-per-fold",
        action="store_true",
        help=(
            "Train a new fold model on each train window and score test rows out-of-sample. "
            "When omitted, precomputed --score-col is used if present."
        ),
    )
    parser.add_argument(
        "--calibrate",
        type=str,
        choices=("none", "sigmoid", "isotonic"),
        default="none",
        help=(
            "Optional probability calibration for retrain-per-fold mode "
            "(none|sigmoid|isotonic; default: none)."
        ),
    )
    parser.add_argument(
        "--max-abs-fwd-ret",
        type=float,
        default=0.0,
        help=(
            "Optional forward-return clipping threshold for research runs only. "
            "0 disables clipping (default: 0)."
        ),
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Optional run date (YYYY-MM-DD) to filter artifacts and rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "ranker_walkforward",
        help="Directory where latest walk-forward JSON is written.",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Optional features CSV path (FS fallback only).",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help="Optional labels CSV path (FS fallback only).",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Optional predictions CSV path (FS fallback only).",
    )
    parsed = parser.parse_args(argv)

    if parsed.test_window_days < 1:
        parser.error("--test-window-days must be >= 1")
    if parsed.train_window_days < 1:
        parser.error("--train-window-days must be >= 1")
    if parsed.step_days < 1:
        parser.error("--step-days must be >= 1")
    if parsed.embargo_days is not None and parsed.embargo_days < 0:
        parser.error("--embargo-days must be >= 0")
    if parsed.top_k < 1:
        parser.error("--top-k must be >= 1")
    if float(parsed.max_abs_fwd_ret) < 0:
        parser.error("--max-abs-fwd-ret must be >= 0")

    try:
        run_date = _parse_run_date(parsed.run_date)
    except ValueError as exc:
        parser.error(str(exc))
        raise  # pragma: no cover - argparse exits

    return WalkforwardArgs(
        target=str(parsed.target),
        test_window_days=int(parsed.test_window_days),
        train_window_days=int(parsed.train_window_days),
        step_days=int(parsed.step_days),
        embargo_days=parsed.embargo_days,
        top_k=int(parsed.top_k),
        score_col=str(parsed.score_col),
        retrain_per_fold=bool(parsed.retrain_per_fold),
        calibrate=str(parsed.calibrate),
        max_abs_fwd_ret=float(parsed.max_abs_fwd_ret),
        run_date=run_date,
        output_dir=Path(parsed.output_dir),
        features_path=parsed.features_path,
        labels_path=parsed.labels_path,
        predictions_path=parsed.predictions_path,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
        run_walkforward(args)
        return 0
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1
    except RuntimeError as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("WALKFORWARD_FAILED err=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
