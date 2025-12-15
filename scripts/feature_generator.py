"""Generate ML features by combining nightly bars with label artifacts.

Usage examples (from repo root with environment activated):
    python scripts/feature_generator.py
    python scripts/feature_generator.py --labels-path data/labels/labels_20240131.csv
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

BAR_COLUMNS = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
LABEL_COLUMNS = [
    "symbol",
    "timestamp",
    "label_5d_pos_300bp",
    "label_10d_pos_300bp",
]
RUN_TZ = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create ML features from nightly daily bars and label artifacts. "
            "Features are merged with labels on symbol and timestamp."
        )
    )
    parser.add_argument(
        "--bars-path",
        type=Path,
        default=Path("data/daily_bars.csv"),
        help="Path to the daily bars CSV (expects symbol, timestamp, OHLC, and volume columns).",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to labels CSV. If omitted, the newest labels_*.csv "
            "under data/labels is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory where the generated features CSV will be written.",
    )
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep rows with NaNs in the output instead of dropping them.",
    )
    return parser.parse_args()


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


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        required_list = ", ".join(sorted(required))
        raise ValueError(
            f"Input data is missing required columns: {missing_list}. "
            f"Required columns are: {required_list}."
        )


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _latest_labels_path(labels_dir: Path) -> Path:
    candidates = sorted(labels_dir.glob("labels_*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No labels files found under {labels_dir}")
    return candidates[-1]


def _prepare_bars(bars_path: Path) -> pd.DataFrame:
    bars = _load_csv(bars_path)
    _require_columns(bars, BAR_COLUMNS)

    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars["volume"] = pd.to_numeric(bars["volume"], errors="coerce")
    return bars


def _load_labels(labels_path: Path) -> pd.DataFrame:
    labels = _load_csv(labels_path)
    _require_columns(labels, LABEL_COLUMNS)

    labels = labels.copy()
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    return labels.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    grouped_close = bars.groupby("symbol")["close"]
    grouped_volume = bars.groupby("symbol")["volume"]

    features = bars.copy()
    features["ret_1d"] = grouped_close.transform(lambda s: s / s.shift(1) - 1)
    features["mom_5d"] = grouped_close.transform(lambda s: s / s.shift(5) - 1)
    features["mom_10d"] = grouped_close.transform(lambda s: s / s.shift(10) - 1)

    features["vol_10d"] = features.groupby("symbol")["ret_1d"].transform(
        lambda s: s.rolling(window=10, min_periods=5).std()
    )

    features["vol_raw"] = features["volume"]
    features["vol_avg_10d"] = grouped_volume.transform(
        lambda s: s.rolling(window=10, min_periods=5).mean()
    )
    features["vol_rvol_10d"] = features["vol_raw"] / features["vol_avg_10d"]

    selected = [
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
    return features[selected]


def build_feature_set(
    bars_path: Path, labels_path: Path, keep_na: bool
) -> Tuple[pd.DataFrame, datetime.date | None, datetime.date | None]:
    bars = _prepare_bars(bars_path)
    labels = _load_labels(labels_path)

    latest_bars_date = _latest_timestamp_date(bars)
    latest_labels_date = _latest_timestamp_date(labels)

    label_subset = labels[LABEL_COLUMNS].copy()
    features = _compute_features(bars)
    merged = features.merge(label_subset, on=["symbol", "timestamp"], how="inner")

    if merged.empty:
        raise ValueError(
            "Merged feature set is empty; ensure bars and labels share symbol/timestamp keys."
        )

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
        "label_5d_pos_300bp",
        "label_10d_pos_300bp",
    ]

    if not keep_na:
        merged = merged.dropna(subset=columns)

    return merged[columns], latest_bars_date, latest_labels_date


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    try:
        labels_path = args.labels_path
        if labels_path is None:
            labels_path = _latest_labels_path(Path("data/labels"))
            logging.info("Using latest labels file: %s", labels_path)

        merged, latest_bars_date, latest_labels_date = build_feature_set(
            args.bars_path, labels_path, keep_na=args.keep_na
        )

        run_date = _current_run_date()
        if latest_bars_date and latest_bars_date < run_date:
            logging.warning(
                "[WARN] FEATURES_BARS_STALE latest_bar_date=%s run_date=%s bars_path=%s",
                latest_bars_date,
                run_date,
                args.bars_path,
            )
        if latest_labels_date and latest_labels_date < run_date:
            logging.warning(
                "[WARN] FEATURES_LABELS_STALE latest_label_date=%s run_date=%s labels_path=%s",
                latest_labels_date,
                run_date,
                labels_path,
            )

        as_of = run_date.strftime("%Y-%m-%d")
        output_dir: Path = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"features_{as_of}.csv"

        merged.to_csv(output_path, index=False)
        logging.info("Features written to %s", output_path)
    except (FileNotFoundError, ValueError) as err:
        logging.error(err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
