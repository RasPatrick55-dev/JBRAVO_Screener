"""Evaluate ranker predictions with a time-series-safe validation split.

The evaluator joins nightly feature labels with predicted scores, builds a
chronological holdout split (train earlier, test later), applies an optional
embargo gap to reduce overlap leakage from forward labels, and writes a JSON
summary for the Screener dashboard.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils.env import load_env  # noqa: E402
from scripts import db  # noqa: E402

load_env()

LOG = logging.getLogger("ranker_eval")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_LABEL = "label_5d_pos_300bp"
SCORE_COLUMN = "score_5d"
DEFAULT_CALIBRATION_BINS = 10
SCORE_RANGE_EPS = 1e-6


@dataclass
class EvalArgs:
    features_path: Path | None
    predictions_path: Path | None
    label_column: str
    score_column: str
    output_dir: Path
    test_size: float
    embargo_days: int | None
    top_k: int
    top_k_fraction: float
    calibration_bins: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_latest(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _normalize_features_frame(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    required = {"symbol", "timestamp", label_column}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Features input missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def _load_features(path: Path, label_column: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to read features from {path}: {exc}") from exc

    return _normalize_features_frame(df, label_column)


def _normalize_predictions_frame(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    required = {"symbol", "timestamp", score_column}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Predictions input missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def _load_predictions(path: Path, score_column: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to read predictions from {path}: {exc}") from exc

    return _normalize_predictions_frame(df, score_column)


def _merge(
    features: pd.DataFrame,
    preds: pd.DataFrame,
    label_column: str,
    score_column: str,
) -> pd.DataFrame:
    merged = pd.merge(
        preds.loc[:, ["symbol", "timestamp", score_column]],
        features.loc[:, ["symbol", "timestamp", label_column]],
        on=["symbol", "timestamp"],
        how="inner",
    )
    merged = merged.dropna(subset=[label_column, score_column])
    merged[label_column] = pd.to_numeric(merged[label_column], errors="coerce")
    merged[score_column] = pd.to_numeric(merged[score_column], errors="coerce")
    merged = merged.dropna(subset=[label_column, score_column])
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


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


def _infer_label_horizon_days(label_column: str) -> int:
    match = re.search(r"label_(\d+)d", str(label_column or ""))
    if not match:
        return 0
    try:
        return max(int(match.group(1)), 0)
    except (TypeError, ValueError):
        return 0


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


def _safe_spearman(labels: pd.Series, scores: pd.Series) -> float | None:
    try:
        corr = labels.corr(scores, method="spearman")
    except Exception:
        return None
    if corr is None or (isinstance(corr, float) and math.isnan(corr)):
        return None
    return float(corr)


def _compute_quality_metrics(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
    *,
    top_k: int,
    top_k_fraction: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "sample_size": 0,
        "auc": None,
        "mae": None,
        "rmse": None,
        "brier": None,
        "spearman": None,
        "accuracy_at_0_5": None,
        "baseline_positive_rate": None,
        "mean_score": None,
        "top_k": max(1, int(top_k)),
        "precision_at_k": None,
        "top_k_positive_rate": None,
        "top_k_lift_vs_baseline": None,
        "top_k_fraction": float(top_k_fraction),
        "top_fraction_count": None,
        "precision_at_top_fraction": None,
        "top_fraction_lift_vs_baseline": None,
    }
    if df.empty:
        return metrics

    labels = pd.to_numeric(df[label_column], errors="coerce")
    scores = pd.to_numeric(df[score_column], errors="coerce")
    valid = labels.notna() & scores.notna()
    if not valid.any():
        return metrics

    labels = labels.loc[valid].astype(float)
    scores = scores.loc[valid].astype(float)
    metric_df = pd.DataFrame({"label": labels, "score": scores}).sort_values(
        "score",
        ascending=False,
        kind="mergesort",
    )
    sample_size = int(metric_df.shape[0])
    baseline = float(metric_df["label"].mean())
    residual = metric_df["label"] - metric_df["score"]
    mse = float((residual**2).mean())

    metrics["sample_size"] = sample_size
    metrics["baseline_positive_rate"] = baseline
    metrics["mean_score"] = float(metric_df["score"].mean())
    metrics["auc"] = _safe_auc(metric_df["label"], metric_df["score"])
    metrics["mae"] = float(residual.abs().mean())
    metrics["rmse"] = float(math.sqrt(mse))
    metrics["brier"] = mse
    metrics["spearman"] = _safe_spearman(metric_df["label"], metric_df["score"])
    metrics["accuracy_at_0_5"] = float(
        ((metric_df["score"] >= 0.5).astype(int) == metric_df["label"].astype(int)).mean()
    )

    k_abs = max(1, min(int(top_k), sample_size))
    top_k_df = metric_df.head(k_abs)
    precision_at_k = float(top_k_df["label"].mean())
    metrics["top_k"] = k_abs
    metrics["precision_at_k"] = precision_at_k
    metrics["top_k_positive_rate"] = precision_at_k
    if baseline > 0:
        metrics["top_k_lift_vs_baseline"] = float(precision_at_k / baseline)

    frac_count = max(1, min(sample_size, int(math.ceil(sample_size * float(top_k_fraction)))))
    top_frac_df = metric_df.head(frac_count)
    precision_at_frac = float(top_frac_df["label"].mean())
    metrics["top_fraction_count"] = frac_count
    metrics["precision_at_top_fraction"] = precision_at_frac
    if baseline > 0:
        metrics["top_fraction_lift_vs_baseline"] = float(precision_at_frac / baseline)

    return metrics


def _compute_calibration_diagnostics(
    df: pd.DataFrame,
    *,
    label_column: str,
    score_column: str,
    bins: int,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "calibration_bins": int(max(2, int(bins))),
        "calibration_curve": [],
        "ece": None,
        "mce": None,
        "calibration_applicable": False,
        "calibration_skip_reason": None,
        "score_min": None,
        "score_max": None,
    }
    if score_column not in df.columns:
        diagnostics["calibration_skip_reason"] = "missing_score_col"
        return diagnostics

    labels = pd.to_numeric(df[label_column], errors="coerce")
    scores = pd.to_numeric(df[score_column], errors="coerce")
    valid = labels.notna() & scores.notna()
    if int(valid.sum()) < 2:
        diagnostics["calibration_skip_reason"] = "insufficient_rows"
        if valid.any():
            valid_scores = scores.loc[valid]
            diagnostics["score_min"] = float(valid_scores.min())
            diagnostics["score_max"] = float(valid_scores.max())
        return diagnostics

    labels = labels.loc[valid].astype(float)
    scores = scores.loc[valid].astype(float)
    score_min = float(scores.min())
    score_max = float(scores.max())
    diagnostics["score_min"] = score_min
    diagnostics["score_max"] = score_max
    if score_min < -SCORE_RANGE_EPS or score_max > (1.0 + SCORE_RANGE_EPS):
        diagnostics["calibration_skip_reason"] = "out_of_range"
        return diagnostics

    bin_count = int(max(2, int(bins)))
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    # Scores in [0,1] map to bin ids [0, bin_count-1].
    bin_ids = np.digitize(scores.to_numpy(dtype=float), edges[1:-1], right=False)
    n_total = int(len(scores))
    curve: list[dict[str, Any]] = []
    ece_acc = 0.0
    mce = 0.0
    for idx in range(bin_count):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        mask = bin_ids == idx
        count = int(np.sum(mask))
        if count <= 0:
            curve.append(
                {
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "count": 0,
                    "avg_pred": None,
                    "frac_pos": None,
                }
            )
            continue
        bin_scores = scores.iloc[mask]
        bin_labels = labels.iloc[mask]
        avg_pred = float(bin_scores.mean())
        frac_pos = float(bin_labels.mean())
        abs_err = abs(avg_pred - frac_pos)
        ece_acc += (count / max(n_total, 1)) * abs_err
        mce = max(mce, abs_err)
        curve.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": count,
                "avg_pred": avg_pred,
                "frac_pos": frac_pos,
            }
        )

    diagnostics["calibration_curve"] = curve
    diagnostics["ece"] = float(ece_acc)
    diagnostics["mce"] = float(mce)
    diagnostics["calibration_applicable"] = True
    diagnostics["calibration_skip_reason"] = None
    return diagnostics


def _compute_deciles(
    df: pd.DataFrame,
    label_column: str,
    score_column: str,
) -> list[dict[str, Any]]:
    if df.empty:
        return []

    labels = pd.to_numeric(df[label_column], errors="coerce")
    scores = pd.to_numeric(df[score_column], errors="coerce")
    valid = labels.notna() & scores.notna()
    if not valid.any():
        return []

    ranked = pd.DataFrame({"label": labels.loc[valid], "score": scores.loc[valid]})
    baseline = float(ranked["label"].mean())

    # Higher scores map to higher deciles; 10 is the top bucket.
    rank_pct = ranked["score"].rank(method="first", pct=True)
    ranked["decile"] = np.ceil(rank_pct * 10).clip(1, 10).astype(int)

    results: list[dict[str, Any]] = []
    for decile, group in ranked.groupby("decile"):
        avg_label = float(group["label"].mean())
        results.append(
            {
                "decile": int(decile),
                "count": int(len(group)),
                "avg_label": avg_label,
                "avg_score": float(group["score"].mean()),
                "score_min": float(group["score"].min()),
                "score_max": float(group["score"].max()),
                "name": f"D{int(decile)}",
                "lift": float(avg_label - baseline),
            }
        )

    results.sort(key=lambda x: x["decile"])
    return results


def _compute_decile_lift(
    deciles: list[dict[str, Any]], top_decile: int, bottom_decile: int
) -> tuple[float | None, float | None, float | None]:
    """Return top/bottom average labels and their lift.

    The function is defensive: it tolerates missing deciles or malformed
    values and returns ``None`` for any missing piece.
    """

    def _avg_label_for(decile_index: int) -> float | None:
        for row in deciles:
            try:
                if int(row.get("decile")) != decile_index:
                    continue
            except (TypeError, ValueError):
                continue
            try:
                return float(row.get("avg_label"))
            except (TypeError, ValueError):
                return None
        return None

    top_avg = _avg_label_for(top_decile)
    bottom_avg = _avg_label_for(bottom_decile)

    if top_avg is None or bottom_avg is None:
        return top_avg, bottom_avg, None

    return top_avg, bottom_avg, top_avg - bottom_avg


def _signal_quality(lift: float | None) -> str | None:
    if lift is None:
        return None
    if lift >= 0.05:
        return "HIGH"
    if lift >= 0.02:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _time_series_split(
    merged: pd.DataFrame,
    *,
    test_size: float,
    embargo_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    empty = merged.iloc[0:0].copy()
    if merged.empty:
        meta = {
            "validation_scheme": "time_series_holdout",
            "split_note": "empty_merged_sample",
            "embargo_days": max(int(embargo_days), 0),
            "embargo_applied": False,
            "train_start": None,
            "train_end": None,
            "test_start": None,
            "test_end": None,
        }
        return empty, empty, meta

    working = merged.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working = working.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if working.empty:
        meta = {
            "validation_scheme": "time_series_holdout",
            "split_note": "no_valid_timestamps",
            "embargo_days": max(int(embargo_days), 0),
            "embargo_applied": False,
            "train_start": None,
            "train_end": None,
            "test_start": None,
            "test_end": None,
        }
        return empty, empty, meta

    unique_ts = pd.Index(working["timestamp"].drop_duplicates().sort_values())
    if unique_ts.size < 2:
        test_only = working.copy()
        meta = {
            "validation_scheme": "time_series_holdout",
            "split_note": "insufficient_unique_timestamps",
            "embargo_days": max(int(embargo_days), 0),
            "embargo_applied": False,
            "train_start": None,
            "train_end": None,
            "test_start": _to_iso(test_only["timestamp"].min()),
            "test_end": _to_iso(test_only["timestamp"].max()),
        }
        return empty, test_only, meta

    split_idx = int(math.floor(unique_ts.size * (1.0 - float(test_size))))
    split_idx = max(1, min(split_idx, unique_ts.size - 1))
    split_ts = pd.Timestamp(unique_ts[split_idx])

    raw_train = working.loc[working["timestamp"] < split_ts].copy()
    test_df = working.loc[working["timestamp"] >= split_ts].copy()

    requested_embargo = max(int(embargo_days), 0)
    used_embargo = requested_embargo
    split_note = "ok"
    if requested_embargo > 0:
        cutoff = split_ts - pd.Timedelta(days=requested_embargo)
        train_df = working.loc[working["timestamp"] < cutoff].copy()
        if train_df.empty and not raw_train.empty:
            train_df = raw_train.copy()
            used_embargo = 0
            split_note = "embargo_relaxed_for_min_train"
    else:
        train_df = raw_train.copy()

    if test_df.empty:
        test_df = working.tail(max(1, int(math.ceil(len(working) * float(test_size))))).copy()
        split_note = "fallback_tail_test"

    validation_scheme = "time_series_holdout_embargo" if used_embargo > 0 else "time_series_holdout"
    meta = {
        "validation_scheme": validation_scheme,
        "split_note": split_note,
        "split_timestamp": _to_iso(split_ts),
        "requested_test_size": float(test_size),
        "requested_embargo_days": requested_embargo,
        "embargo_days": used_embargo,
        "embargo_applied": bool(used_embargo > 0),
        "train_start": _to_iso(train_df["timestamp"].min()) if not train_df.empty else None,
        "train_end": _to_iso(train_df["timestamp"].max()) if not train_df.empty else None,
        "test_start": _to_iso(test_df["timestamp"].min()) if not test_df.empty else None,
        "test_end": _to_iso(test_df["timestamp"].max()) if not test_df.empty else None,
    }
    return train_df, test_df, meta


def evaluate(args: EvalArgs) -> dict[str, Any]:
    if db.db_enabled():
        features_raw = db.load_ml_artifact_csv("features")
        preds_raw = db.load_ml_artifact_csv("predictions")
        if features_raw.empty:
            raise FileNotFoundError("Features not found in DB (ml_artifacts: features)")
        if preds_raw.empty:
            raise FileNotFoundError("Predictions not found in DB (ml_artifacts: predictions)")
        features = _normalize_features_frame(features_raw, args.label_column)
        preds = _normalize_predictions_frame(preds_raw, args.score_column)
    else:
        features = _load_features(args.features_path, args.label_column)
        preds = _load_predictions(args.predictions_path, args.score_column)
    merged = _merge(features, preds, args.label_column, args.score_column)
    label_horizon_days = _infer_label_horizon_days(args.label_column)
    requested_embargo_days = (
        int(args.embargo_days) if args.embargo_days is not None else int(label_horizon_days)
    )
    requested_embargo_days = max(requested_embargo_days, 0)
    train_df, test_df, split_meta = _time_series_split(
        merged,
        test_size=args.test_size,
        embargo_days=requested_embargo_days,
    )
    eval_df = test_df.copy()
    reason: str | None = None
    if merged.empty:
        reason = "no_merged_samples"
    elif eval_df.empty:
        eval_df = merged.copy()
        reason = "no_test_samples_fallback_all_data"

    deciles = _compute_deciles(eval_df, args.label_column, args.score_column)
    top_decile_index = 10
    bottom_decile_index = 1
    top_avg_label, bottom_avg_label, decile_lift = _compute_decile_lift(
        deciles, top_decile_index, bottom_decile_index
    )
    test_metrics = _compute_quality_metrics(
        eval_df,
        args.label_column,
        args.score_column,
        top_k=args.top_k,
        top_k_fraction=args.top_k_fraction,
    )
    train_metrics = _compute_quality_metrics(
        train_df,
        args.label_column,
        args.score_column,
        top_k=args.top_k,
        top_k_fraction=args.top_k_fraction,
    )
    calibration = _compute_calibration_diagnostics(
        eval_df,
        label_column=args.label_column,
        score_column=args.score_column,
        bins=args.calibration_bins,
    )
    sample_size = int(test_metrics.get("sample_size", 0) or 0)

    payload: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "sample_size": sample_size,
        "population_size": int(len(merged)),
        "train_sample_size": int(train_metrics.get("sample_size", 0) or 0),
        "label_horizon_days": int(label_horizon_days),
        "label_column": args.label_column,
        "score_column": args.score_column,
        "validation_scheme": split_meta.get("validation_scheme"),
        "split_note": split_meta.get("split_note"),
        "test_size": float(args.test_size),
        "embargo_days": int(split_meta.get("embargo_days", requested_embargo_days) or 0),
        "embargo_applied": bool(split_meta.get("embargo_applied", False)),
        "split_timestamp": split_meta.get("split_timestamp"),
        "train_start": split_meta.get("train_start"),
        "train_end": split_meta.get("train_end"),
        "test_start": split_meta.get("test_start"),
        "test_end": split_meta.get("test_end"),
        "time_windows": {
            "train_start": split_meta.get("train_start"),
            "train_end": split_meta.get("train_end"),
            "test_start": split_meta.get("test_start"),
            "test_end": split_meta.get("test_end"),
        },
        "decile_convention": "10=top",
        "top_decile_index": top_decile_index,
        "bottom_decile_index": bottom_decile_index,
        "deciles": deciles,
        "top_avg_label": top_avg_label,
        "bottom_avg_label": bottom_avg_label,
        "decile_lift": decile_lift,
        "signal_quality": _signal_quality(decile_lift),
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "auc": test_metrics.get("auc"),
        "mae": test_metrics.get("mae"),
        "rmse": test_metrics.get("rmse"),
        "spearman": test_metrics.get("spearman"),
        "precision_at_k": test_metrics.get("precision_at_k"),
        "calibration_bins": int(calibration.get("calibration_bins", args.calibration_bins)),
        "calibration_curve": calibration.get("calibration_curve") or [],
        "ece": calibration.get("ece"),
        "mce": calibration.get("mce"),
        "calibration_applicable": bool(calibration.get("calibration_applicable", False)),
        "score_min": calibration.get("score_min"),
        "score_max": calibration.get("score_max"),
    }
    if calibration.get("calibration_skip_reason"):
        payload["calibration_skip_reason"] = calibration.get("calibration_skip_reason")
    if reason:
        payload["reason"] = reason
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> EvalArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Path to features CSV. Defaults to latest data/features/features_*.csv",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=None,
        help="Path to predictions CSV. Defaults to latest data/predictions/predictions_*.csv",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=DEFAULT_LABEL,
        help="Binary label column to evaluate",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Alias for --label-column.",
    )
    parser.add_argument(
        "--score-col",
        type=str,
        default=SCORE_COLUMN,
        help="Score column from predictions to evaluate (default: score_5d)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "ranker_eval",
        help="Directory to write evaluation JSON",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for the chronological test split (default: 0.2)",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=None,
        help=(
            "Optional gap between train and test in calendar days. "
            "Defaults to inferred label horizon."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Absolute top-k bucket for ranking metrics (default: 20)",
    )
    parser.add_argument(
        "--top-k-fraction",
        type=float,
        default=0.1,
        help="Top fraction bucket for ranking metrics (default: 0.1)",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=DEFAULT_CALIBRATION_BINS,
        help="Bin count for reliability diagnostics (ECE/MCE) on test scores (default: 10)",
    )
    parsed = parser.parse_args(argv)
    if not (0.0 < float(parsed.test_size) < 1.0):
        parser.error("--test-size must be between 0 and 1")
    if parsed.embargo_days is not None and int(parsed.embargo_days) < 0:
        parser.error("--embargo-days must be >= 0")
    if int(parsed.top_k) < 1:
        parser.error("--top-k must be >= 1")
    if not (0.0 < float(parsed.top_k_fraction) <= 1.0):
        parser.error("--top-k-fraction must be > 0 and <= 1")
    if int(parsed.calibration_bins) < 2:
        parser.error("--calibration-bins must be >= 2")

    features_path = parsed.features_path
    predictions_path = parsed.predictions_path
    if not db.db_enabled():
        if features_path is None:
            features_path = _find_latest(BASE_DIR / "data" / "features", "features_*.csv")
            if features_path is None:
                raise FileNotFoundError("No features files found in data/features")

        if predictions_path is None:
            predictions_path = _find_latest(BASE_DIR / "data" / "predictions", "predictions_*.csv")
            if predictions_path is None:
                raise FileNotFoundError("No predictions files found in data/predictions")

    return EvalArgs(
        features_path=features_path,
        predictions_path=predictions_path,
        label_column=str(parsed.target or parsed.label_column),
        score_column=str(parsed.score_col or SCORE_COLUMN),
        output_dir=parsed.output_dir,
        test_size=float(parsed.test_size),
        embargo_days=parsed.embargo_days,
        top_k=int(parsed.top_k),
        top_k_fraction=float(parsed.top_k_fraction),
        calibration_bins=int(parsed.calibration_bins),
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1

    if db.db_enabled():
        LOG.info("Loading features from DB (ml_artifacts: features)")
        LOG.info("Loading predictions from DB (ml_artifacts: predictions)")
    else:
        LOG.info("Loading features from %s", args.features_path)
        LOG.info("Loading predictions from %s", args.predictions_path)

    try:
        payload = evaluate(args)
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("Evaluation failed: %s", exc)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    LOG.info(
        "Evaluation complete: samples=%d deciles=%d output=%s",
        payload.get("sample_size", 0),
        len(payload.get("deciles") or []),
        output_path,
    )
    LOG.info(
        "Validation scheme=%s train=%s test=%s embargo_days=%s train_window=%s..%s test_window=%s..%s",
        payload.get("validation_scheme"),
        payload.get("train_sample_size", 0),
        payload.get("sample_size", 0),
        payload.get("embargo_days", 0),
        payload.get("train_start"),
        payload.get("train_end"),
        payload.get("test_start"),
        payload.get("test_end"),
    )
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    LOG.info(
        "Quality metrics: auc=%s rmse=%s mae=%s spearman=%s precision_at_k=%s",
        metrics.get("auc"),
        metrics.get("rmse"),
        metrics.get("mae"),
        metrics.get("spearman"),
        metrics.get("precision_at_k"),
    )
    calibration_applicable = bool(payload.get("calibration_applicable", False))
    if calibration_applicable:
        LOG.info(
            "[INFO] RANKER_EVAL_CALIBRATION bins=%s ece=%s mce=%s score_min=%s score_max=%s",
            int(payload.get("calibration_bins", DEFAULT_CALIBRATION_BINS) or DEFAULT_CALIBRATION_BINS),
            payload.get("ece"),
            payload.get("mce"),
            payload.get("score_min"),
            payload.get("score_max"),
        )
    else:
        LOG.warning(
            "[WARN] RANKER_EVAL_CALIBRATION_SKIPPED reason=%s score_min=%s score_max=%s",
            payload.get("calibration_skip_reason") or "insufficient_rows",
            payload.get("score_min"),
            payload.get("score_max"),
        )

    for dec in payload.get("deciles") or []:
        LOG.info(
            "Decile %d: count=%d avg_label=%.4f avg_score=%.4f",
            dec["decile"],
            dec["count"],
            dec["avg_label"],
            dec["avg_score"],
        )
    if db.db_enabled():
        meta = db.fetch_latest_ml_artifact("predictions") or db.fetch_latest_ml_artifact("features")
        run_date = meta.get("run_date") if meta else None
        if run_date is None:
            run_date = datetime.now(timezone.utc).date()
        ok = db.upsert_ml_artifact(
            "ranker_eval",
            run_date,
            payload=payload,
            rows_count=int(payload.get("sample_size", 0) or 0),
            source="ranker_eval",
            file_name=output_path.name,
        )
        if ok:
            LOG.info(
                "[INFO] RANKER_EVAL_DB_WRITTEN run_date=%s rows=%d",
                run_date,
                int(payload.get("sample_size", 0) or 0),
            )
        else:
            LOG.warning("[WARN] RANKER_EVAL_DB_WRITE_FAILED run_date=%s", run_date)

    return 0


if __name__ == "__main__":
    sys.exit(main())
