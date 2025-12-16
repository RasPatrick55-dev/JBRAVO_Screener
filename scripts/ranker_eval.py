"""Evaluate ranker predictions via decile analysis for the Screener dashboard.

This script joins nightly feature labels with predicted scores, splits the
sample into deciles, and writes a compact JSON summary for the Screener
"Health" tab. It is intended as a manual/experimental utility and does not
modify existing automated tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from utils.env import load_env  # noqa: E402

load_env()

LOG = logging.getLogger("ranker_eval")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_LABEL = "label_5d_pos_300bp"
SCORE_COLUMN = "score_5d"


@dataclass
class EvalArgs:
    features_path: Path
    predictions_path: Path
    label_column: str
    output_dir: Path


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

    required = {"symbol", "timestamp", label_column}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Features file {path} missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def _load_predictions(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to read predictions from {path}: {exc}") from exc

    required = {"symbol", "timestamp", SCORE_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Predictions file {path} missing columns: {sorted(missing)}")

    df = df.dropna(subset=list(required))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


def _merge(features: pd.DataFrame, preds: pd.DataFrame, label_column: str) -> pd.DataFrame:
    merged = pd.merge(
        preds.loc[:, ["symbol", "timestamp", SCORE_COLUMN]],
        features.loc[:, ["symbol", "timestamp", label_column]],
        on=["symbol", "timestamp"],
        how="inner",
    )
    merged = merged.dropna(subset=[label_column, SCORE_COLUMN])
    merged[label_column] = pd.to_numeric(merged[label_column], errors="coerce")
    merged[SCORE_COLUMN] = pd.to_numeric(merged[SCORE_COLUMN], errors="coerce")
    merged = merged.dropna(subset=[label_column, SCORE_COLUMN])
    return merged.reset_index(drop=True)


def _compute_deciles(df: pd.DataFrame, label_column: str) -> list[dict[str, Any]]:
    if df.empty:
        return []

    scores = pd.to_numeric(df[SCORE_COLUMN], errors="coerce").dropna()
    if scores.empty:
        return []

    df = df.loc[scores.index].copy()
    # Higher scores map to higher deciles; 10 is the top bucket.
    rank_pct = scores.rank(method="first", pct=True)
    df["decile"] = np.ceil(rank_pct * 10).clip(1, 10).astype(int)

    results: list[dict[str, Any]] = []
    for decile, group in df.groupby("decile"):
        results.append(
            {
                "decile": int(decile),
                "count": int(len(group)),
                "avg_label": float(group[label_column].mean()),
                "avg_score": float(group[SCORE_COLUMN].mean()),
                "score_min": float(group[SCORE_COLUMN].min()),
                "score_max": float(group[SCORE_COLUMN].max()),
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


def evaluate(args: EvalArgs) -> dict[str, Any]:
    features = _load_features(args.features_path, args.label_column)
    preds = _load_predictions(args.predictions_path)
    merged = _merge(features, preds, args.label_column)

    deciles = _compute_deciles(merged, args.label_column)
    top_decile_index = 10
    bottom_decile_index = 1
    top_avg_label, bottom_avg_label, decile_lift = _compute_decile_lift(
        deciles, top_decile_index, bottom_decile_index
    )
    payload: dict[str, Any] = {
        "sample_size": int(len(merged)),
        "label_column": args.label_column,
        "score_column": SCORE_COLUMN,
        "decile_convention": "10=top",
        "top_decile_index": top_decile_index,
        "bottom_decile_index": bottom_decile_index,
        "deciles": deciles,
        "top_avg_label": top_avg_label,
        "bottom_avg_label": bottom_avg_label,
        "decile_lift": decile_lift,
        "signal_quality": _signal_quality(decile_lift),
    }
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
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "ranker_eval",
        help="Directory to write evaluation JSON",
    )
    parsed = parser.parse_args(argv)

    features_path = parsed.features_path
    if features_path is None:
        features_path = _find_latest(BASE_DIR / "data" / "features", "features_*.csv")
        if features_path is None:
            raise FileNotFoundError("No features files found in data/features")

    predictions_path = parsed.predictions_path
    if predictions_path is None:
        predictions_path = _find_latest(BASE_DIR / "data" / "predictions", "predictions_*.csv")
        if predictions_path is None:
            raise FileNotFoundError("No predictions files found in data/predictions")

    return EvalArgs(
        features_path=features_path,
        predictions_path=predictions_path,
        label_column=parsed.label_column,
        output_dir=parsed.output_dir,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv or sys.argv[1:])
    except FileNotFoundError as exc:
        LOG.error("%s", exc)
        return 1

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

    for dec in payload.get("deciles") or []:
        LOG.info(
            "Decile %d: count=%d avg_label=%.4f avg_score=%.4f",
            dec["decile"],
            dec["count"],
            dec["avg_label"],
            dec["avg_score"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
