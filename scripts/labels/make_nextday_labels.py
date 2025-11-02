from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from utils.io_utils import atomic_write_bytes

DATE_FORMAT = "%Y-%m-%d"


@dataclass
class LabelJob:
    predictions_path: Path
    prices_path: Path
    output_dir: Path
    horizon: int = 1
    as_of: Optional[datetime] = None

    def resolve_as_of(self) -> datetime:
        if self.as_of is not None:
            return self.as_of
        try:
            return datetime.strptime(self.predictions_path.stem, DATE_FORMAT)
        except ValueError:
            raise SystemExit(
                "Unable to infer as-of date. Pass --date explicitly when the predictions "
                "filename does not follow YYYY-MM-DD.csv."
            )


def _coerce_date(value: str) -> datetime:
    return datetime.strptime(value, DATE_FORMAT)


def _load_predictions(path: Path, as_of: datetime) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Predictions file not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise SystemExit(f"Predictions file {path} is empty")

    frame = frame.copy()
    as_of_date = as_of.date()
    frame["symbol"] = frame.get("symbol", "").astype(str).str.upper()

    if "timestamp" in frame.columns:
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["as_of_date"] = timestamp.dt.date
        frame.loc[frame["as_of_date"].isna(), "as_of_date"] = as_of_date
    else:
        frame["as_of_date"] = as_of_date

    if "Score" not in frame.columns and "score" in frame.columns:
        frame["Score"] = pd.to_numeric(frame["score"], errors="coerce")

    frame["as_of_date"] = frame["as_of_date"].astype("datetime64[ns]").dt.date
    return frame


def _load_price_history(prices_path: Path) -> pd.DataFrame:
    if not prices_path.exists():
        raise SystemExit(f"Price history not found: {prices_path}")

    def _read_csv(candidate: Path) -> pd.DataFrame:
        frame = pd.read_csv(candidate)
        frame = frame.rename(columns={c: c.lower() for c in frame.columns})
        if "symbol" not in frame.columns:
            frame["symbol"] = candidate.stem.upper()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        elif "timestamp" in frame.columns:
            frame["date"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        else:
            frame["date"] = pd.NaT
        frame = frame.dropna(subset=["date"])
        frame["date"] = frame["date"].dt.tz_localize(None)
        if "close" not in frame.columns:
            raise SystemExit(
                f"Required column 'close' missing from {candidate}. Available columns: {list(frame.columns)}"
            )
        return frame[["symbol", "date", "close"]]

    if prices_path.is_file():
        history = _read_csv(prices_path)
    else:
        frames: list[pd.DataFrame] = []
        for child in sorted(prices_path.glob("*.csv")):
            if child.name.startswith("."):
                continue
            frames.append(_read_csv(child))
        if not frames:
            raise SystemExit(f"No CSV files found in {prices_path}")
        history = pd.concat(frames, ignore_index=True)

    history["date"] = history["date"].dt.date
    history.sort_values(["symbol", "date"], inplace=True)
    return history


def _prepare_returns(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    grouped = history.groupby("symbol", group_keys=False)
    forward_close = grouped["close"].shift(-horizon)
    returns = (forward_close - history["close"]) / history["close"]
    prepared = history.assign(next_close=forward_close, nextday_ret=returns)
    prepared.rename(columns={"date": "as_of_date"}, inplace=True)
    return prepared[["symbol", "as_of_date", "close", "next_close", "nextday_ret"]]


def _merge_predictions(predictions: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(
        returns,
        how="left",
        on=["symbol", "as_of_date"],
        validate="m:1",
    )
    merged["nextday_ret"] = pd.to_numeric(merged["nextday_ret"], errors="coerce")
    merged["win_>0"] = np.where(merged["nextday_ret"].gt(0), 1, 0)
    merged.rename(columns={"close": "close_price", "next_close": "future_close"}, inplace=True)
    return merged


def _write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    atomic_write_bytes(output_path, csv_bytes)


def run(job: LabelJob) -> Path:
    as_of = job.resolve_as_of()
    predictions = _load_predictions(job.predictions_path, as_of)
    history = _load_price_history(job.prices_path)
    returns = _prepare_returns(history, job.horizon)
    merged = _merge_predictions(predictions, returns)

    missing = merged["nextday_ret"].isna().sum()
    if missing:
        symbols = ", ".join(sorted(set(merged.loc[merged["nextday_ret"].isna(), "symbol"])))
        print(
            f"Warning: {missing} rows are missing price data for forward returns. Symbols: {symbols}",
        )

    date_str = as_of.strftime(DATE_FORMAT)
    output_path = job.output_dir / f"realized_{date_str}.csv"
    _write_output(merged, output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Join predictions with realized next-day returns.")
    parser.add_argument(
        "predictions",
        type=Path,
        help="Path to the nightly predictions CSV (e.g. data/predictions/2025-01-15.csv)",
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=Path("data") / "daily_prices.csv",
        help="CSV file or directory containing daily OHLC data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "labels",
        help="Directory where the realized label CSV will be written.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forward horizon in trading days for realized return.",
    )
    parser.add_argument(
        "--date",
        type=_coerce_date,
        help="Override the as-of date when the predictions filename does not encode it.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Path:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    job = LabelJob(
        predictions_path=args.predictions,
        prices_path=args.prices,
        output_dir=args.output_dir,
        horizon=args.horizon,
        as_of=args.date,
    )
    return run(job)


if __name__ == "__main__":
    main()
