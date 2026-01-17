"""Apply a trained ML rank model to screener candidates."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils.env import load_env  # noqa: E402

try:  # pragma: no cover - environment dependent
    import joblib
except Exception:  # pragma: no cover - environment dependent
    joblib = None


LOGGER = logging.getLogger("apply_rank_model")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_MODEL_PATH = Path("data") / "models" / "rank_model.joblib"

FEATURE_COLUMNS = [
    "score",
    "sma9",
    "ema20",
    "sma180",
    "rsi14",
    "adv20",
    "atrp",
    "ma_spread_9_20",
    "ma_spread_20_180",
]


def _series_by_alias(frame: pd.DataFrame, *aliases: str) -> pd.Series:
    for alias in aliases:
        if alias in frame.columns:
            return pd.to_numeric(frame[alias], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _build_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    features["score"] = _series_by_alias(frame, "score", "Score")
    features["sma9"] = _series_by_alias(frame, "sma9", "SMA9")
    features["ema20"] = _series_by_alias(frame, "ema20", "EMA20")
    features["sma180"] = _series_by_alias(frame, "sma180", "SMA180")
    features["rsi14"] = _series_by_alias(frame, "rsi14", "RSI14")
    features["adv20"] = _series_by_alias(frame, "adv20", "ADV20")
    features["atrp"] = _series_by_alias(frame, "atrp", "ATR_pct", "ATR%")
    features["ma_spread_9_20"] = features["sma9"] - features["ema20"]
    features["ma_spread_20_180"] = features["ema20"] - features["sma180"]
    return features


def _load_model(path: Path) -> tuple[object, object | None, list[str]]:
    if joblib is None:
        raise RuntimeError("joblib is required to load the rank model")
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload.get("model")
        scaler = payload.get("scaler")
        features = payload.get("features") or FEATURE_COLUMNS
    else:
        model = payload
        scaler = None
        features = FEATURE_COLUMNS
    return model, scaler, list(features)


def _predict_proba(model, features: np.ndarray | pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, -1]
    if hasattr(model, "decision_function"):
        logits = model.decision_function(features)
        return 1 / (1 + np.exp(-logits))
    raise RuntimeError("Model does not support probability predictions.")


def apply_rank_model(frame: pd.DataFrame, model_path: Path | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame

    model_path = model_path or Path(os.getenv("RANK_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.exists():
        LOGGER.info("Rank model not found at %s; skipping ML ranking", model_path)
        return frame

    try:
        model, scaler, feature_names = _load_model(model_path)
    except Exception as exc:
        LOGGER.warning("Rank model load failed (%s); skipping ML ranking", exc)
        return frame
    features = _build_features(frame).reindex(columns=feature_names)
    valid_mask = features.notna().all(axis=1)

    prob_up = pd.Series(np.nan, index=frame.index, dtype="float64")
    if valid_mask.any():
        X = features.loc[valid_mask]
        if scaler is not None:
            X = scaler.transform(X)
        prob_up.loc[valid_mask] = _predict_proba(model, X)

    output = frame.copy()
    output["prob_up"] = prob_up

    score_series = pd.to_numeric(
        output.get("Score", output.get("score", pd.Series(np.nan, index=output.index))),
        errors="coerce",
    )
    adjusted = score_series * 0.7 + prob_up * 0.3
    adjusted.loc[prob_up.isna()] = score_series.loc[prob_up.isna()]
    output["ml_adjusted_score"] = adjusted

    prob_non_null = int(prob_up.notna().sum())
    prob_mean = float(prob_up.mean()) if prob_non_null else 0.0
    adj_mean = float(adjusted.mean()) if adjusted.notna().any() else 0.0
    LOGGER.info(
        "[INFO] ML_RANK applied rows=%d prob_up_non_null=%d prob_up_mean=%.4f ml_score_mean=%.4f model=%s",
        int(output.shape[0]),
        prob_non_null,
        prob_mean,
        adj_mean,
        model_path,
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "scored_candidates.csv",
        help="Path to scored candidates CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "scored_candidates_ml.csv",
        help="Path to write scored candidates with ML columns",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained model (default: data/models/rank_model.joblib)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_env()
    args = parse_args(argv or sys.argv[1:])

    try:
        frame = pd.read_csv(args.input)
    except FileNotFoundError:
        LOGGER.error("Input file not found: %s", args.input)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Failed to read %s: %s", args.input, exc)
        return 1

    output = apply_rank_model(frame, model_path=args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    LOGGER.info("Wrote ML-ranked candidates to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
