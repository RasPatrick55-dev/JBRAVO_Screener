"""Executor CLI and I/O contract for trade execution.

This module provides the command line interface and candidate loading logic for
future trade execution enhancements.  It validates the upstream CSV produced by
pipeline steps and prepares filtered candidates based on guardrails supplied via
CLI flags.  Later PRs will attach the actual order management logic.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "symbol",
    "close",
    "score",
    "universe_count",
    "score_breakdown",
]

class CandidateLoadError(RuntimeError):
    """Raised when candidate data cannot be loaded or validated."""


@dataclass
class ExecutorConfig:
    source: Path = Path("data/latest_candidates.csv")
    allocation_pct: float = 0.03
    max_positions: int = 4
    entry_buffer_bps: int = 75
    trailing_percent: float = 3.0
    cancel_after_min: int = 35
    extended_hours: bool = True
    dry_run: bool = False
    min_adv20: int = 2_000_000
    min_price: float = 1.0
    max_price: float = 1_000.0
    log_json: bool = False


@dataclass
class ExecutionMetrics:
    """Metrics collected during candidate validation and guard application."""

    symbols_in: int = 0
    symbols_remaining: int = 0
    skipped_by_reason: Dict[str, int] = field(default_factory=dict)
    missing_columns: List[str] = field(default_factory=list)

    def record_skip(self, reason: str) -> None:
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1

    def note_missing(self, column: str) -> None:
        if column not in self.missing_columns:
            self.missing_columns.append(column)

    def as_dict(self) -> Dict[str, object]:
        return {
            "symbols_in": self.symbols_in,
            "symbols_remaining": self.symbols_remaining,
            "skipped_by_reason": self.skipped_by_reason,
            "missing_columns": self.missing_columns,
        }

    def log_summary(self, log_json: bool = False) -> None:
        payload = self.as_dict()
        if log_json:
            LOGGER.info(json.dumps({"evt": "EXECUTE_SUMMARY", **payload}))
        else:
            LOGGER.info(
                "Execution summary: %s candidates in, %s remaining, skipped=%s",
                payload["symbols_in"],
                payload["symbols_remaining"],
                payload["skipped_by_reason"],
            )
            if payload["missing_columns"]:
                LOGGER.info("Missing optional columns: %s", payload["missing_columns"])


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute trades for pipeline candidates")
    parser.add_argument(
        "--source",
        type=Path,
        default=ExecutorConfig.source,
        help="Path to the candidate CSV file",
    )
    parser.add_argument(
        "--allocation-pct",
        type=float,
        default=ExecutorConfig.allocation_pct,
        help="Fraction of buying power allocated per position (0-1)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=ExecutorConfig.max_positions,
        help="Maximum concurrent positions the executor will open",
    )
    parser.add_argument(
        "--entry-buffer-bps",
        type=int,
        default=ExecutorConfig.entry_buffer_bps,
        help="Entry buffer in basis points added to the reference price",
    )
    parser.add_argument(
        "--trailing-percent",
        type=float,
        default=ExecutorConfig.trailing_percent,
        help="Percent trail for the protective stop order",
    )
    parser.add_argument(
        "--cancel-after-min",
        type=int,
        default=ExecutorConfig.cancel_after_min,
        help="Minutes after regular market open to cancel unfilled orders",
    )
    parser.add_argument(
        "--extended-hours",
        type=str2bool,
        default=ExecutorConfig.extended_hours,
        help="Whether to submit orders eligible for extended hours",
    )
    parser.add_argument(
        "--dry-run",
        type=str2bool,
        default=ExecutorConfig.dry_run,
        help="If true, only log intended actions without submitting orders",
    )
    parser.add_argument(
        "--min-adv20",
        type=int,
        default=ExecutorConfig.min_adv20,
        help="Minimum 20-day average dollar volume required",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=ExecutorConfig.min_price,
        help="Minimum allowed price for candidates",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=ExecutorConfig.max_price,
        help="Maximum allowed price for candidates",
    )
    parser.add_argument(
        "--log-json",
        type=str2bool,
        default=ExecutorConfig.log_json,
        help="Emit structured JSON logs in addition to human readable ones",
    )
    return parser.parse_args(argv)


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CandidateLoadError(f"Candidate file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - delegated to pandas
        raise CandidateLoadError(f"Unable to read candidate file: {exc}") from exc

    if df.empty:
        raise CandidateLoadError("Candidate file is empty")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise CandidateLoadError(
            f"Candidate file missing required columns: {', '.join(missing_columns)}"
        )

    return df


def apply_guards(df: pd.DataFrame, config: ExecutorConfig, metrics: ExecutionMetrics) -> pd.DataFrame:
    keep_indices: List[int] = []

    has_adv20 = "adv20" in df.columns
    has_entry_price = "entry_price" in df.columns

    if not has_adv20:
        metrics.note_missing("adv20")
    if "atrp" not in df.columns:
        metrics.note_missing("atrp")
    if "exchange" not in df.columns:
        metrics.note_missing("exchange")

    for idx, row in df.iterrows():
        reference_price = float(row["entry_price"]) if has_entry_price and not pd.isna(row.get("entry_price")) else float(row["close"])

        if reference_price < config.min_price:
            metrics.record_skip("PRICE_LT_MIN")
            continue
        if reference_price > config.max_price:
            metrics.record_skip("PRICE_GT_MAX")
            continue

        if has_adv20:
            adv_value = row.get("adv20")
            if pd.isna(adv_value) or adv_value < config.min_adv20:
                metrics.record_skip("ADV20_LT_MIN")
                continue

        keep_indices.append(idx)

    if not keep_indices:
        return df.iloc[0:0].copy()

    return df.loc[keep_indices].reset_index(drop=True)


def configure_logging(log_json: bool) -> None:
    if LOGGER.handlers:
        for handler in list(LOGGER.handlers):
            LOGGER.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter
    if log_json:
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def run_executor(config: ExecutorConfig) -> int:
    configure_logging(config.log_json)

    metrics = ExecutionMetrics()

    try:
        df = load_candidates(config.source)
    except CandidateLoadError as exc:
        LOGGER.error("%s", exc)
        return 1

    metrics.symbols_in = len(df)

    filtered = apply_guards(df, config, metrics)
    metrics.symbols_remaining = len(filtered)

    metrics.log_summary(config.log_json)

    if filtered.empty:
        LOGGER.info("No tradable candidates after guard application; exiting cleanly.")
        return 0

    LOGGER.info(
        "Prepared %s candidate(s) for execution (dry_run=%s)",
        metrics.symbols_remaining,
        config.dry_run,
    )
    return 0


def build_config(args: argparse.Namespace) -> ExecutorConfig:
    return ExecutorConfig(
        source=args.source,
        allocation_pct=args.allocation_pct,
        max_positions=args.max_positions,
        entry_buffer_bps=args.entry_buffer_bps,
        trailing_percent=args.trailing_percent,
        cancel_after_min=args.cancel_after_min,
        extended_hours=args.extended_hours,
        dry_run=args.dry_run,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        log_json=args.log_json,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    return run_executor(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
