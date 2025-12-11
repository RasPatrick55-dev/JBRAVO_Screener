"""Generate interpretable ML features from nightly bars and labels.

The script ingests a daily bars CSV and the latest labels artifact, merges
records that share (symbol, timestamp), and emits a feature table suitable for
model training.

Example usage (from repo root):
    python scripts/feature_generator.py \
        --bars-path data/daily_bars.csv \
        --output-dir data/features

Using an explicit labels file:
    python scripts/feature_generator.py \
        --bars-path data/daily_bars.csv \
        --labels-path data/labels/labels_20240131.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


REQUIRED_BAR_COLUMNS = {"symbol", "timestamp", "close", "volume"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ML features by combining nightly bars with labels."
    )
    parser.add_argument(
        "--bars-path",
        type=Path,
        default=Path("data/daily_bars.csv"),
        help="Path to the daily bars CSV (expects symbol, timestamp, close, volume).",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to the labels CSV. "
            "If omitted, the newest labels_*.csv under data/labels is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory where the generated features CSV will be written.",
    )
    return parser.parse_args()


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        columns = ", ".join(sorted(missing))
        raise ValueError(
            f"Input data is missing required columns: {columns}. "
            f"Required columns are: {', '.join(sorted(required))}."
        )


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _find_latest_labels(labels_dir: Path) -> Path:
    candidates = list(labels_dir.glob("labels_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No labels files found under {labels_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _determine_as_of_date(df: pd.DataFrame) -> str:
    latest_timestamp = df["timestamp"].max()
    if pd.isna(latest_timestamp):
        raise ValueError("Labels file contains no timestamps; cannot derive as-of date.")
    return latest_timestamp.date().strftime("%Y%m%d")


def _compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    sorted_bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    grouped_close = sorted_bars.groupby("symbol")["close"]
    grouped_volume = sorted_bars.groupby("symbol")["volume"]

    sorted_bars["mom_5d"] = grouped_close.transform(lambda s: s / s.shift(5) - 1)
    sorted_bars["mom_10d"] = grouped_close.transform(lambda s: s / s.shift(10) - 1)

    daily_returns = grouped_close.pct_change()
    sorted_bars["vol_10d"] = daily_returns.groupby(sorted_bars["symbol"]).transform(
        lambda s: s.rolling(window=10, min_periods=1).std()
    )

    sorted_bars["vol_avg_10d"] = grouped_volume.transform(
        lambda s: s.rolling(window=10, min_periods=1).mean()
    )
    sorted_bars["vol_rvol_10d"] = sorted_bars["volume"] / sorted_bars["vol_avg_10d"]

    sorted_bars["sma_5d"] = grouped_close.transform(
        lambda s: s.rolling(window=5, min_periods=1).mean()
    )
    sorted_bars["sma_10d"] = grouped_close.transform(
        lambda s: s.rolling(window=10, min_periods=1).mean()
    )

    feature_columns = [
        "symbol",
        "timestamp",
        "close",
        "mom_5d",
        "mom_10d",
        "vol_10d",
        "volume",
        "vol_avg_10d",
        "vol_rvol_10d",
        "sma_5d",
        "sma_10d",
    ]
    return sorted_bars[feature_columns].rename(columns={"volume": "vol_raw"})


def build_feature_set(bars_path: Path, labels_path: Path) -> pd.DataFrame:
    bars_df = _load_csv(bars_path)
    _validate_required_columns(bars_df, REQUIRED_BAR_COLUMNS)

    labels_df = _load_csv(labels_path)
    _validate_required_columns(labels_df, {"symbol", "timestamp"})

    features_df = _compute_features(bars_df)
    merged = labels_df.merge(features_df, on=["symbol", "timestamp"], how="inner")

    if merged.empty:
        raise ValueError(
            "Merged feature set is empty; ensure bars and labels share symbol/timestamp keys."
        )
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    try:
        labels_path: Optional[Path] = args.labels_path
        if labels_path is None:
            labels_path = _find_latest_labels(Path("data/labels"))
            logging.info("Using latest labels file: %s", labels_path)

        features = build_feature_set(args.bars_path, labels_path)

        as_of_date = _determine_as_of_date(features)
        output_dir: Path = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"features_{as_of_date}.csv"

        columns = [
            "symbol",
            "timestamp",
            "close",
            "mom_5d",
            "mom_10d",
            "vol_10d",
            "vol_raw",
            "vol_avg_10d",
            "vol_rvol_10d",
        ]
        label_columns = [col for col in features.columns if col.startswith("label_")]
        additional = [col for col in features.columns if col not in columns + label_columns]
        ordered_columns = columns + additional + label_columns
        features = features[ordered_columns]

        features.to_csv(output_path, index=False)
        logging.info("Features written to %s", output_path)
    except (FileNotFoundError, ValueError) as err:
        logging.error(err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
