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
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils.env import load_env  # noqa: E402

load_env()

LOG = logging.getLogger("ranker_train")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

FEATURE_COLUMNS = [
    "mom_5d",
    "mom_10d",
    "vol_raw",
    "vol_avg_10d",
    "vol_rvol_10d",
]

MAX_TRAIN_ROWS = 50_000
SAMPLE_RANDOM_STATE = 42


@dataclass
class TrainArgs:
    features_path: Path
    label_column: str
    output_dir: Path


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
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    RandomForestClassifier = None
    LogisticRegression = None
    accuracy_score = None
    roc_auc_score = None
    StandardScaler = None
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
    return datetime.utcnow().strftime("%Y-%m-%d")


def _load_features(path: Path, label_column: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to read features from {path}: {exc}") from exc

    required_columns = set(["symbol", "timestamp", *FEATURE_COLUMNS, label_column])
    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"Features file {path} missing columns: {sorted(missing)}")

    df = df.dropna(subset=FEATURE_COLUMNS + [label_column])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


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


def _train_model(df: pd.DataFrame, label_column: str):
    if df.empty:
        raise RuntimeError("No data available to train the model.")

    if len(df) > MAX_TRAIN_ROWS:
        LOG.info(
            "Downsampling training set from %s to %s rows for faster execution",
            len(df),
            MAX_TRAIN_ROWS,
        )
        df = df.sample(n=MAX_TRAIN_ROWS, random_state=SAMPLE_RANDOM_STATE).sort_values(
            "timestamp"
        )

    split_idx = max(1, int(len(df) * 0.8))
    if split_idx >= len(df):
        split_idx = len(df) - 1

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[label_column]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[label_column]

    model = _choose_model()

    if StandardScaler is not None and isinstance(model, LogisticRegression):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model.fit(X_train_scaled, y_train)
        val_probs = _predict_proba(model, X_val_scaled)
        train_size = len(train_df)
        val_size = len(val_df)
        return model, scaler, val_probs, y_val, train_size, val_size

    model.fit(X_train, y_train)
    val_probs = _predict_proba(model, X_val)
    train_size = len(train_df)
    val_size = len(val_df)
    return model, None, val_probs, y_val, train_size, val_size


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

    if joblib is None:
        raise RuntimeError("joblib is required to persist the model")
    joblib.dump(payload, model_path)

    summary_path = output_dir / f"ranker_summary_{snapshot_date}.json"
    summary = {
        "features_path": str(args.features_path),
        "label_column": args.label_column,
        "train_size": metrics.pop("train_size", None),
        "val_size": metrics.pop("val_size", None),
        "metrics": metrics,
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return TrainOutputs(model_path=model_path, summary_path=summary_path, metrics=summary["metrics"])


def parse_args(argv: list[str] | None = None) -> TrainArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        type=Path,
        default=None,
        help="Path to a features CSV. Defaults to the latest data/features/features_*.csv",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label_5d_pos_300bp",
        help="Binary label column to train against",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "models",
        help="Directory to write the trained model and summary",
    )
    parsed = parser.parse_args(argv)

    features_path = parsed.features_path
    if features_path is None:
        default_dir = BASE_DIR / "data" / "features"
        features_path = _find_latest_features(default_dir)
        if features_path is None:
            raise FileNotFoundError(f"No features files found in {default_dir}")

    return TrainArgs(features_path=features_path, label_column=parsed.label_column, output_dir=parsed.output_dir)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1

    LOG.info("Loading features from %s", args.features_path)
    try:
        df = _load_features(args.features_path, args.label_column)
    except FileNotFoundError:
        LOG.error("Features file not found: %s", args.features_path)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("Failed to load features: %s", exc)
        return 1

    if df.empty:
        LOG.error("Features file %s has no usable rows", args.features_path)
        return 1

    try:
        model, scaler, val_probs, y_val, train_size, val_size = _train_model(df, args.label_column)
    except ModelUnavailableError as exc:
        LOG.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("Training failed: %s", exc)
        return 1

    metrics = _compute_metrics(val_probs, y_val)
    metrics["train_size"] = train_size
    metrics["val_size"] = val_size

    accuracy_display = metrics.get("accuracy")
    auc_display = metrics.get("auc")
    LOG.info(
        "Features=%s | Train=%s | Val=%s | Accuracy=%s | AUC=%s",
        args.features_path,
        train_size,
        val_size,
        f"{accuracy_display:.4f}" if accuracy_display is not None else "n/a",
        f"{auc_display:.4f}" if auc_display not in (None, "nan") else "n/a",
    )

    snapshot_date = _extract_features_date(args.features_path)
    try:
        outputs = _save_outputs(model, scaler, metrics, args, snapshot_date)
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("Failed to save outputs: %s", exc)
        return 1

    LOG.info("Model saved to %s", outputs.model_path)
    LOG.info("Summary written to %s", outputs.summary_path)
    LOG.info("Validation metrics: %s", json.dumps(outputs.metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
