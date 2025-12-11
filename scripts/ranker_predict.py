"""Generate nightly ranker predictions from the latest features snapshot.

This module loads a trained ranker model (produced by ``scripts.ranker_train``),
computes probabilities for the positive class, and writes a CSV containing
predictions alongside a subset of the input features for dashboard consumption.

Usage::

    python -m scripts.ranker_predict
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils.env import load_env  # noqa: E402
from scripts.ranker_train import (  # noqa: E402
    FEATURE_COLUMNS,
    _extract_features_date,
    _predict_proba,
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
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _load_features(path: Path, label_column: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to read features from {path}: {exc}") from exc

    required_columns: set[str] = {
        "symbol",
        "timestamp",
        "close",
        *FEATURE_COLUMNS,
        label_column,
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"Features file {path} missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required_columns))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


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
    else:
        model = payload
        scaler = None
    return model, scaler


def _predict(df: pd.DataFrame, model, scaler=None) -> pd.Series:
    features = df[FEATURE_COLUMNS]
    if scaler is not None:
        features = scaler.transform(features)
    return _predict_proba(model, features)


def _build_output(
    df: pd.DataFrame, label_column: str, probs: Iterable[float]
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
    return output.loc[:, columns]


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    features_path = args.features_path
    if features_path is None:
        features_path = _find_latest(BASE_DIR / "data" / "features", "features_*.csv")
        if features_path is None:
            LOG.error("No features files found in data/features")
            return 1

    model_path = args.model_path
    if model_path is None:
        model_path = _find_latest(BASE_DIR / "data" / "models", "ranker_*.pkl")
        if model_path is None:
            LOG.error("No model files found in data/models")
            return 1

    LOG.info("Loading features from %s", features_path)
    try:
        features_df = _load_features(features_path, args.label_column)
    except FileNotFoundError:
        LOG.error("Features file not found: %s", features_path)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("Failed to load features: %s", exc)
        return 1

    if features_df.empty:
        LOG.error("Features file %s has no usable rows", features_path)
        return 1

    LOG.info("Loading model from %s", model_path)
    try:
        model, scaler = _load_model(model_path)
    except FileNotFoundError:
        LOG.error("Model file not found: %s", model_path)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.error("Failed to load model: %s", exc)
        return 1

    probs = _predict(features_df, model, scaler)
    output_df = _build_output(features_df, args.label_column, probs)

    snapshot_date = _extract_features_date(features_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{snapshot_date}.csv"
    output_df.to_csv(output_path, index=False)

    LOG.info(
        "Predictions written to %s (rows=%d, mean_score=%.4f)",
        output_path,
        len(output_df),
        float(output_df["score_5d"].mean()) if not output_df.empty else 0.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
