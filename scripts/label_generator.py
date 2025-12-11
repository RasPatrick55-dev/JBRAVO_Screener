"""Utility to generate forward return labels from daily bars data.

This script ingests a CSV containing daily bars for multiple symbols and
produces forward returns and binary labels for configurable horizons.

Usage example (from repo root):
    python scripts/label_generator.py --bars-path data/bars/daily_bars.csv

Another example with custom thresholds and horizons:
    python scripts/label_generator.py \
        --bars-path data/bars/daily_bars.csv \
        --horizons 3 7 14 \
        --threshold-percent 2.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"symbol", "timestamp", "close"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate forward return labels for multiple symbols from a daily bars CSV."
        )
    )
    parser.add_argument(
        "--bars-path",
        required=True,
        type=Path,
        help="Path to the input CSV containing daily bars with symbol, timestamp, and close columns.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[5, 10],
        help=(
            "Forward return horizons in trading days. Provide one or more integers. "
            "Defaults to 5 and 10."
        ),
    )
    parser.add_argument(
        "--threshold-percent",
        type=float,
        default=3.0,
        help=(
            "Positive return threshold in percent used to generate binary labels. "
            "Defaults to 3.0 (i.e., 300 basis points)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/labels"),
        help="Directory where the output labels CSV will be written.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"Input data is missing required columns: {missing_list}. "
            "Required columns are: symbol, timestamp, close."
        )


def _format_threshold_label(threshold_percent: float) -> str:
    basis_points = int(round(threshold_percent * 100))
    return f"pos_{basis_points}bp"


def compute_forward_returns(
    df: pd.DataFrame, horizons: Iterable[int]
) -> pd.DataFrame:
    result = df.copy()
    sorted_horizons = sorted(set(horizons))

    for horizon in sorted_horizons:
        fwd_ret_col = f"fwd_ret_{horizon}d"
        result[fwd_ret_col] = (
            result.groupby("symbol")["close"].shift(-horizon) / result["close"] - 1
        )
    return result


def add_labels(
    df: pd.DataFrame, horizons: Iterable[int], threshold_percent: float
) -> pd.DataFrame:
    labeled = df.copy()
    threshold_decimal = threshold_percent / 100.0
    threshold_label = _format_threshold_label(threshold_percent)

    for horizon in sorted(set(horizons)):
        fwd_ret_col = f"fwd_ret_{horizon}d"
        if fwd_ret_col not in labeled.columns:
            raise KeyError(
                f"Missing forward return column {fwd_ret_col}. Ensure compute_forward_returns was run first."
            )
        label_col = f"label_{horizon}d_{threshold_label}"
        labeled[label_col] = (labeled[fwd_ret_col] >= threshold_decimal).astype(int)
    return labeled


def load_bars(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Bars file not found: {path}")

    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def determine_as_of_date(df: pd.DataFrame) -> str:
    latest_timestamp = df["timestamp"].max()
    if pd.isna(latest_timestamp):
        raise ValueError("Unable to determine as-of date because timestamp column is empty or invalid.")
    return latest_timestamp.date().strftime("%Y%m%d")


def main() -> None:
    args = parse_args()

    bars_df = load_bars(args.bars_path)
    validate_columns(bars_df)

    # Sort to enforce deterministic forward calculations within each symbol.
    bars_df = bars_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    with_returns = compute_forward_returns(bars_df, args.horizons)
    labeled = add_labels(with_returns, args.horizons, args.threshold_percent)

    as_of_date = determine_as_of_date(labeled)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"labels_{as_of_date}.csv"

    labeled.to_csv(output_path, index=False)
    print(f"Labels written to {output_path}")


if __name__ == "__main__":
    main()
