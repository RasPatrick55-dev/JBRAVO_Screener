from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from utils.io_utils import atomic_write_bytes

DATE_FORMAT = "%Y-%m-%d"
DEFAULT_PF_THRESHOLD = 1.2
DEFAULT_DRAWDOWN_CAP = 0.15


@dataclass
class AutotuneConfig:
    labels_dir: Path
    config_path: Path
    output_dir: Path
    lookback_days: int = 120
    splits: int = 4
    top_quantile: float = 0.1
    pf_threshold: float = DEFAULT_PF_THRESHOLD
    max_drawdown: float = DEFAULT_DRAWDOWN_CAP
    min_sample: int = 50
    as_of: Optional[datetime] = None
    changelog_path: Optional[Path] = None


@dataclass
class FoldMetrics:
    expectancy: float
    profit_factor: float
    max_drawdown: float
    sample_size: int


@dataclass
class CandidateResult:
    weights: Dict[str, float]
    fold_metrics: list[FoldMetrics]
    aggregate_expectancy: float
    aggregate_profit_factor: float
    aggregate_drawdown: float
    total_sample: int


class AutotuneError(RuntimeError):
    pass


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, DATE_FORMAT)


def _read_config(path: Path) -> dict:
    if not path.exists():
        raise AutotuneError(f"Config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise AutotuneError(f"Unable to read config file: {exc}") from exc

    data: dict[str, dict[str, float]] = {}
    section: Optional[str] = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":" ) and not line.startswith("-"):
            section = line[:-1]
            if section not in data:
                data[section] = {}
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                numeric = float(value)
            except ValueError:
                continue
            if section:
                data.setdefault(section, {})[key] = numeric
            else:
                data[key] = numeric
    return data


def _write_config(path: Path, weights: Dict[str, float], base: dict) -> None:
    thresholds = base.get("thresholds", {})
    version = int(base.get("version", 2))
    lines = [f"version: {version}", "weights:"]
    for key, value in weights.items():
        lines.append(f"  {key}: {value:.6f}")
    if thresholds:
        lines.append("thresholds:")
        for key, value in thresholds.items():
            lines.append(f"  {key}: {value}")
    content = "\n".join(lines) + "\n"
    atomic_write_bytes(path, content.encode("utf-8"))


def _load_labels(cfg: AutotuneConfig, feature_keys: list[str]) -> pd.DataFrame:
    files = []
    for path in sorted(cfg.labels_dir.glob("realized_*.csv")):
        try:
            dt = _parse_date(path.stem.replace("realized_", ""))
        except ValueError:
            continue
        files.append((dt, path))
    if not files:
        raise AutotuneError(f"No realized label files found in {cfg.labels_dir}")

    end_date = cfg.as_of or files[-1][0]
    start_date = end_date - timedelta(days=cfg.lookback_days - 1)
    selected = [(dt, path) for dt, path in files if start_date <= dt <= end_date]
    if not selected:
        raise AutotuneError("No realized label files fall within the requested window")

    frames: list[pd.DataFrame] = []
    for dt, path in selected:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["as_of"] = dt.date()
        frames.append(frame)

    if not frames:
        raise AutotuneError("Label files are empty for tuning window")

    combined = pd.concat(frames, ignore_index=True)
    combined["nextday_ret"] = pd.to_numeric(combined.get("nextday_ret"), errors="coerce")
    combined.dropna(subset=["nextday_ret"], inplace=True)
    if combined.empty:
        raise AutotuneError("No realized returns available for tuning")

    missing_features = [key for key in feature_keys if key not in combined.columns]
    if missing_features:
        raise AutotuneError(
            "Feature columns missing from labels: " + ", ".join(missing_features)
        )

    for key in feature_keys:
        combined[key] = pd.to_numeric(combined[key], errors="coerce")
    combined.dropna(subset=feature_keys, inplace=True)
    if combined.empty:
        raise AutotuneError("Feature columns contain only NaN values")

    combined.sort_values(["as_of"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _walk_forward_splits(dates: pd.Series, splits: int, min_train: int = 3) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.array(sorted(pd.to_datetime(dates.dropna().unique())))
    if unique_dates.size < (min_train + 1):
        return iter([])

    step = max(1, math.floor((unique_dates.size - min_train) / max(1, splits)))
    for idx in range(splits):
        train_end = min_train + idx * step
        if train_end >= unique_dates.size:
            break
        test_end = min(unique_dates.size, train_end + step)
        train_mask = dates.isin(unique_dates[:train_end])
        test_mask = dates.isin(unique_dates[train_end:test_end])
        if not test_mask.any():
            continue
        yield train_mask.to_numpy(), test_mask.to_numpy()


def _standardise(train: pd.DataFrame, test: pd.DataFrame, feature_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean = train[feature_keys].mean()
    std = train[feature_keys].std(ddof=0).replace(0, 1.0)
    train_scaled = (train[feature_keys] - mean) / std
    test_scaled = (test[feature_keys] - mean) / std
    return train_scaled, test_scaled


def _score_frame(scaled: pd.DataFrame, weights: Dict[str, float], feature_keys: list[str]) -> pd.Series:
    ordered = np.array([weights[key] for key in feature_keys])
    return scaled.to_numpy() @ ordered


def _profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / abs(losses)


def _max_drawdown(returns: pd.Series, dates: pd.Series) -> float:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "ret": returns})
    frame.sort_values("date", inplace=True)
    equity = (1 + frame["ret"].fillna(0.0)).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1
    return float(drawdown.min())


def _evaluate_candidate(
    df: pd.DataFrame,
    weights: Dict[str, float],
    feature_keys: list[str],
    cfg: AutotuneConfig,
) -> CandidateResult:
    fold_metrics: list[FoldMetrics] = []
    collected_returns: list[pd.Series] = []

    for train_mask, test_mask in _walk_forward_splits(df["as_of"], cfg.splits):
        train = df.loc[train_mask]
        test = df.loc[test_mask]
        if len(train) < len(feature_keys) or test.empty:
            continue
        train_scaled, test_scaled = _standardise(train, test, feature_keys)
        test_scores = _score_frame(test_scaled, weights, feature_keys)
        threshold = np.quantile(test_scores, 1 - cfg.top_quantile)
        selected = test_scores >= threshold
        selected_returns = test.loc[selected, "nextday_ret"].astype(float)
        if selected_returns.empty:
            continue
        pf = _profit_factor(selected_returns)
        dd = _max_drawdown(selected_returns, test.loc[selected, "as_of"])
        expectancy = float(selected_returns.mean())
        fold_metrics.append(
            FoldMetrics(
                expectancy=expectancy,
                profit_factor=float(pf),
                max_drawdown=float(dd),
                sample_size=int(selected_returns.size),
            )
        )
        collected_returns.append(selected_returns)

    if not fold_metrics:
        raise AutotuneError("Unable to evaluate candidate; insufficient folds")

    combined_returns = pd.concat(collected_returns).sort_index()
    aggregate_expectancy = float(combined_returns.mean())
    aggregate_profit_factor = float(_profit_factor(combined_returns))
    aggregate_drawdown = float(_max_drawdown(combined_returns, df.loc[combined_returns.index, "as_of"]))
    total_sample = int(combined_returns.size)

    return CandidateResult(
        weights=weights,
        fold_metrics=fold_metrics,
        aggregate_expectancy=aggregate_expectancy,
        aggregate_profit_factor=aggregate_profit_factor,
        aggregate_drawdown=aggregate_drawdown,
        total_sample=total_sample,
    )


def _generate_candidates(base_weights: Dict[str, float]) -> Iterator[Dict[str, float]]:
    keys = list(base_weights.keys())
    multipliers = [0.8, 1.0, 1.2]
    for combo in np.array(np.meshgrid(*([multipliers] * len(keys)))).T.reshape(-1, len(keys)):
        weights = {key: round(base_weights[key] * float(multiplier), 6) for key, multiplier in zip(keys, combo)}
        yield weights


def _guardrails_pass(result: CandidateResult, cfg: AutotuneConfig) -> bool:
    if result.total_sample < cfg.min_sample:
        return False
    if not math.isfinite(result.aggregate_profit_factor) or result.aggregate_profit_factor < cfg.pf_threshold:
        return False
    if result.aggregate_drawdown < -cfg.max_drawdown:
        return False
    return True


def _append_changelog(path: Path, as_of: datetime, baseline: CandidateResult, candidate: CandidateResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## {as_of.strftime('%Y-%m-%d')}",
        "", "Baseline:",
        f"- Expectancy: {baseline.aggregate_expectancy:.6f}",
        f"- Profit Factor: {baseline.aggregate_profit_factor:.3f}",
        f"- Max Drawdown: {baseline.aggregate_drawdown:.3f}",
        "", "Candidate:",
        f"- Expectancy: {candidate.aggregate_expectancy:.6f}",
        f"- Profit Factor: {candidate.aggregate_profit_factor:.3f}",
        f"- Max Drawdown: {candidate.aggregate_drawdown:.3f}",
        "", "Weights:",
    ]
    for key, value in candidate.weights.items():
        lines.append(f"- {key}: {value:.6f}")
    lines.append("")

    content = "\n".join(lines)
    if path.exists():
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(content + "\n")
    else:
        atomic_write_bytes(path, (content + "\n").encode("utf-8"))


def autotune(cfg: AutotuneConfig) -> Optional[Path]:
    base = _read_config(cfg.config_path)
    base_weights = base.get("weights")
    if not base_weights:
        raise AutotuneError("Config missing 'weights' section")

    feature_keys = list(base_weights.keys())
    df = _load_labels(cfg, feature_keys)

    baseline_result = _evaluate_candidate(df, base_weights, feature_keys, cfg)

    best_result = baseline_result
    best_weights = base_weights

    for candidate_weights in _generate_candidates(base_weights):
        result = _evaluate_candidate(df, candidate_weights, feature_keys, cfg)
        if result.aggregate_expectancy > best_result.aggregate_expectancy + 1e-6 and _guardrails_pass(result, cfg):
            best_result = result
            best_weights = candidate_weights

    if best_result is baseline_result or not _guardrails_pass(best_result, cfg):
        print("Guardrails not met or no improvement; keeping existing configuration.")
        return None

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    as_of = cfg.as_of or max(pd.to_datetime(df["as_of"]))
    version_name = f"ranker_v2_{as_of.strftime('%Y%m%d')}"
    output_path = cfg.output_dir / f"{version_name}.yml"
    _write_config(output_path, best_weights, base)

    current_link = cfg.output_dir / "ranker_v2_current.yml"
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()
    current_link.symlink_to(output_path.name)

    if cfg.changelog_path:
        _append_changelog(cfg.changelog_path, as_of, baseline_result, best_result)

    metadata = {
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of": as_of.strftime(DATE_FORMAT),
        "output": str(output_path),
        "weights": best_weights,
        "baseline_expectancy": baseline_result.aggregate_expectancy,
        "candidate_expectancy": best_result.aggregate_expectancy,
        "candidate_profit_factor": best_result.aggregate_profit_factor,
        "candidate_drawdown": best_result.aggregate_drawdown,
        "sample_size": best_result.total_sample,
    }
    atomic_write_bytes(
        (cfg.output_dir / f"{version_name}.json"),
        json.dumps(metadata, indent=2).encode("utf-8"),
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward autotuning for the ranker weights.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data") / "labels",
        help="Directory containing realized label CSVs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "ranker_v2.yml",
        help="Path to the base ranker configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs"),
        help="Directory where tuned configurations should be written.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=120,
        help="Number of days to include in the training window.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=4,
        help="Number of walk-forward splits to evaluate.",
    )
    parser.add_argument(
        "--top-quantile",
        type=float,
        default=0.1,
        help="Top quantile of scores to evaluate for expectancy.",
    )
    parser.add_argument(
        "--pf-threshold",
        type=float,
        default=DEFAULT_PF_THRESHOLD,
        help="Minimum profit factor required to accept a candidate.",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=DEFAULT_DRAWDOWN_CAP,
        help="Maximum allowable drawdown (absolute value).",
    )
    parser.add_argument(
        "--min-sample",
        type=int,
        default=50,
        help="Minimum number of validation samples required to accept a candidate.",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_date,
        help="Override the as-of date used for filtering realized labels.",
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=Path("configs") / "ranker_v2_changelog.md",
        help="Path to the changelog file that records accepted updates.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Optional[Path]:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = AutotuneConfig(
        labels_dir=args.labels_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        lookback_days=args.lookback_days,
        splits=args.splits,
        top_quantile=args.top_quantile,
        pf_threshold=args.pf_threshold,
        max_drawdown=args.max_drawdown,
        min_sample=args.min_sample,
        as_of=args.as_of,
        changelog_path=args.changelog,
    )
    return autotune(cfg)


if __name__ == "__main__":
    try:
        main()
    except AutotuneError as exc:
        raise SystemExit(str(exc))
