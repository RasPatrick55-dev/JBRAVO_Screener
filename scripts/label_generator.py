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
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd

from scripts import db

REQUIRED_COLUMNS = {"symbol", "timestamp", "close"}
RUN_TZ = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate forward return labels for multiple symbols from a daily bars CSV.")
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
        dest="threshold_pct",
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


def compute_forward_returns(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    result = df.copy()
    sorted_horizons = sorted(set(horizons))

    for horizon in sorted_horizons:
        fwd_ret_col = f"fwd_ret_{horizon}d"
        result[fwd_ret_col] = (
            result.groupby("symbol")["close"].shift(-horizon) / result["close"] - 1
        )
    return result


def add_labels(df: pd.DataFrame, horizons: Iterable[int], threshold_percent: float) -> pd.DataFrame:
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
    if db.db_enabled():
        bars_df = db.load_ml_artifact_csv("daily_bars")
        if bars_df.empty:
            raise FileNotFoundError("Bars data not found in DB (ml_artifacts: daily_bars).")
        if "timestamp" in bars_df.columns:
            bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True, errors="coerce")
        return bars_df
    if not path.exists():
        raise FileNotFoundError(f"Bars file not found: {path}")

    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _current_run_date() -> datetime.date:
    try:
        return datetime.now(RUN_TZ).date()
    except Exception:
        return datetime.now(timezone.utc).date()


def _latest_timestamp_date(df: pd.DataFrame) -> datetime.date | None:
    if "timestamp" not in df.columns or df.empty:
        return None

    latest_timestamp = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()
    if pd.isna(latest_timestamp):
        return None

    try:
        return latest_timestamp.tz_convert(RUN_TZ).date()
    except Exception:
        return latest_timestamp.date()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    bars_df = load_bars(args.bars_path)
    validate_columns(bars_df)

    # Sort to enforce deterministic forward calculations within each symbol.
    bars_df = bars_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    with_returns = compute_forward_returns(bars_df, args.horizons)
    labeled = add_labels(with_returns, args.horizons, args.threshold_pct)

    run_date = _current_run_date()
    latest_bar_date = _latest_timestamp_date(labeled)
    if latest_bar_date and latest_bar_date < run_date:
        logging.warning(
            "[WARN] LABELS_INPUT_STALE latest_bar_date=%s run_date=%s bars_path=%s",
            latest_bar_date,
            run_date,
            args.bars_path,
        )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"labels_{run_date.strftime('%Y%m%d')}.csv"

    labeled.to_csv(output_path, index=False)
    logging.info("[INFO] LABELS_WRITTEN path=%s rows=%d", output_path, int(labeled.shape[0]))
    if db.db_enabled():
        ok = db.upsert_ml_artifact_frame(
            "labels",
            run_date,
            labeled,
            source="label_generator",
            file_name=output_path.name,
        )
        if ok:
            logging.info(
                "[INFO] LABELS_DB_WRITTEN run_date=%s rows=%d", run_date, int(labeled.shape[0])
            )
        else:
            logging.warning("[WARN] LABELS_DB_WRITE_FAILED run_date=%s", run_date)


if __name__ == "__main__":
    main()
