from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from utils.io_utils import atomic_write_bytes

DATE_FORMAT = "%Y-%m-%d"


@dataclass
class EvalConfig:
    labels_dir: Path
    output_dir: Path = Path("data") / "ranker_eval"
    days: int = 60
    score_column: str = "Score"
    as_of: Optional[datetime] = None


@dataclass
class EvalOutputs:
    summary_path: Path
    deciles_path: Path


class EvaluationError(RuntimeError):
    pass


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, DATE_FORMAT)


def _list_label_files(directory: Path) -> list[tuple[datetime, Path]]:
    files: list[tuple[datetime, Path]] = []
    for path in sorted(directory.glob("realized_*.csv")):
        try:
            date_value = _parse_date(path.stem.replace("realized_", ""))
        except ValueError:
            continue
        files.append((date_value, path))
    return files


def _load_window(cfg: EvalConfig) -> pd.DataFrame:
    if not cfg.labels_dir.exists():
        raise EvaluationError(f"Labels directory not found: {cfg.labels_dir}")

    files = _list_label_files(cfg.labels_dir)
    if not files:
        raise EvaluationError(f"No realized label files found in {cfg.labels_dir}")

    end_date = cfg.as_of or files[-1][0]
    start_date = end_date - timedelta(days=cfg.days - 1)

    selected = [(dt, path) for dt, path in files if start_date <= dt <= end_date]
    if not selected:
        raise EvaluationError("No realized label files fall within the requested window")

    frames: list[pd.DataFrame] = []
    for dt, path in selected:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["as_of"] = dt.date()
        frames.append(frame)

    if not frames:
        raise EvaluationError("Label files are empty for the requested window")

    combined = pd.concat(frames, ignore_index=True)
    combined["nextday_ret"] = pd.to_numeric(combined.get("nextday_ret"), errors="coerce")
    combined["win_>0"] = pd.to_numeric(combined.get("win_>0"), errors="coerce")
    combined.dropna(subset=["nextday_ret"], inplace=True)

    if combined.empty:
        raise EvaluationError("No realized returns available after dropping missing values")

    score_column = None
    for candidate in [cfg.score_column, "Score", "score"]:
        if candidate in combined.columns:
            score_column = candidate
            break
    if score_column is None:
        combined["__score"] = -combined.get("rank", pd.Series([], dtype=float))
        score_column = "__score"
    combined["__score_column"] = score_column
    combined["__score_values"] = pd.to_numeric(combined[score_column], errors="coerce")

    combined = combined.dropna(subset=["__score_values"])
    if combined.empty:
        raise EvaluationError("Score column is missing or entirely non-numeric")

    combined.rename(columns={score_column: "score"}, inplace=True)
    if score_column != "score":
        combined.drop(columns=[score_column], inplace=True, errors="ignore")
    return combined


def _profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / abs(losses)


def _max_drawdown(returns: pd.Series, dates: Optional[pd.Series] = None) -> float:
    series = returns.fillna(0.0)
    if dates is not None:
        frame = pd.DataFrame({"ret": series, "as_of": pd.to_datetime(dates, errors="coerce")})
        frame.dropna(subset=["as_of"], inplace=True)
        frame.sort_values("as_of", inplace=True)
        series = frame["ret"].fillna(0.0)
    equity = (1 + series).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1
    return float(drawdown.min())


def _decile_frame(df: pd.DataFrame) -> pd.DataFrame:
    deciles = pd.qcut(df["score"], q=10, labels=range(1, 11), duplicates="drop")
    df = df.assign(decile=deciles)
    grouped = (
        df.groupby("decile")
        .agg(
            count=("symbol", "size"),
            avg_return=("nextday_ret", "mean"),
            hit_rate=("win_>0", "mean"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )
    overall_hit_rate = df["win_>0"].mean() or np.nan
    if overall_hit_rate and np.isfinite(overall_hit_rate) and overall_hit_rate != 0:
        grouped["lift"] = grouped["hit_rate"] / overall_hit_rate
    else:
        grouped["lift"] = np.nan
    return grouped


def _calibration_points(df: pd.DataFrame, bins: int = 10) -> list[dict[str, float]]:
    binned = pd.qcut(df["score"], q=min(bins, df["score"].nunique()), duplicates="drop")
    grouped = (
        df.groupby(binned)
        .agg(avg_score=("score", "mean"), hit_rate=("win_>0", "mean"), count=("score", "size"))
        .reset_index(drop=True)
    )
    return grouped.to_dict("records")


def _stability(df: pd.DataFrame) -> list[dict[str, float]]:
    if "as_of" not in df.columns:
        return []
    df = df.copy()
    df["as_of"] = pd.to_datetime(df["as_of"], errors="coerce")
    df.dropna(subset=["as_of"], inplace=True)
    if df.empty:
        return []
    df["month"] = df["as_of"].dt.to_period("M").astype(str)
    grouped = (
        df.groupby("month")
        .agg(
            samples=("symbol", "size"),
            hit_rate=("win_>0", "mean"),
            expectancy=("nextday_ret", "mean"),
        )
        .reset_index()
    )
    grouped["profit_factor"] = df.groupby("month").apply(
        lambda frame: _profit_factor(frame["nextday_ret"])
    ).reset_index(drop=True)
    return grouped.to_dict("records")


def evaluate(cfg: EvalConfig) -> EvalOutputs:
    combined = _load_window(cfg)

    deciles = _decile_frame(combined)
    deciles_output = deciles.copy()
    deciles_output["hit_rate"] = deciles_output["hit_rate"].round(4)
    deciles_output["avg_return"] = deciles_output["avg_return"].round(6)

    returns = combined["nextday_ret"].astype(float)
    hits = combined["win_>0"].astype(float)

    expectancy = float(returns.mean())
    hit_rate = float(hits.mean())
    pf = float(_profit_factor(returns))
    sharpe = float("nan")
    std = returns.std(ddof=0)
    drawdown = float(_max_drawdown(returns, combined["as_of"]))
    if std and std > 0:
        sharpe = float(expectancy / std * np.sqrt(252))

    as_of_date = cfg.as_of or max(pd.to_datetime(combined["as_of"]))
    decile_path = cfg.output_dir / f"deciles_{as_of_date.strftime(DATE_FORMAT)}.csv"
    summary_path = cfg.output_dir / "summary.json"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    decile_bytes = deciles_output.to_csv(index=False).encode("utf-8")
    atomic_write_bytes(decile_path, decile_bytes)

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of": as_of_date.strftime(DATE_FORMAT),
        "window_days": cfg.days,
        "sample_size": int(len(combined)),
        "metrics": {
            "hit_rate": hit_rate,
            "expectancy": expectancy,
            "profit_factor": pf,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "expectancy_std": float(std) if np.isfinite(std) else None,
        },
        "calibration": _calibration_points(combined),
        "stability": _stability(combined),
    }
    summary_bytes = json.dumps(summary_payload, indent=2).encode("utf-8")
    atomic_write_bytes(summary_path, summary_bytes)

    latest_path = cfg.output_dir / "latest.json"
    try:
        tmp_link = cfg.output_dir / ".latest.json.tmp"
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        relative = os.path.relpath(summary_path, cfg.output_dir)
        tmp_link.symlink_to(relative)
        tmp_link.replace(latest_path)
    except OSError:
        atomic_write_bytes(latest_path, summary_bytes)

    latest_deciles = cfg.output_dir / "latest_deciles.csv"
    try:
        tmp_deciles = cfg.output_dir / ".latest_deciles.csv.tmp"
        if tmp_deciles.exists() or tmp_deciles.is_symlink():
            tmp_deciles.unlink()
        relative_csv = os.path.relpath(decile_path, cfg.output_dir)
        tmp_deciles.symlink_to(relative_csv)
        tmp_deciles.replace(latest_deciles)
    except OSError:
        atomic_write_bytes(latest_deciles, decile_bytes)

    return EvalOutputs(summary_path=summary_path, deciles_path=decile_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate nightly ranker performance.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data") / "labels",
        help="Directory containing realized label CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "ranker_eval",
        help="Directory for evaluation artefacts.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of days to include in the rolling evaluation window.",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="Score",
        help="Column containing the model score (defaults to 'Score').",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_date,
        help="Evaluate up to this date (inclusive). Defaults to the latest available label.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> EvalOutputs:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = EvalConfig(
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        days=args.days,
        score_column=args.score_column,
        as_of=args.as_of,
    )
    return evaluate(cfg)


if __name__ == "__main__":
    try:
        main()
    except EvaluationError as exc:
        raise SystemExit(str(exc))
