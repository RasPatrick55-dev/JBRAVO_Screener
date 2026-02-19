"""Export run-scoped latest screener candidates to a canonical CSV parachute."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from scripts.db_queries import get_latest_screener_candidates
from scripts.utils.env import load_env

LOGGER = logging.getLogger("export_latest_candidates")

CANONICAL_HEADER = [
    "timestamp",
    "symbol",
    "score",
    "exchange",
    "close",
    "volume",
    "universe_count",
    "score_breakdown",
    "entry_price",
    "adv20",
    "atrp",
    "source",
]


def _resolve_timestamp(series: pd.Series, run_date: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    fallback = pd.Timestamp(f"{run_date}T20:00:00Z")
    parsed = parsed.fillna(fallback)
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def export_latest_candidates(run_date: str, out_path: Path) -> int:
    frame, latest_run_ts = get_latest_screener_candidates(run_date)
    export = frame.copy()
    for col in CANONICAL_HEADER:
        if col not in export.columns:
            export[col] = pd.NA
    export = export[CANONICAL_HEADER]
    export["timestamp"] = _resolve_timestamp(export["timestamp"], str(run_date))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(out_path, index=False)
    LOGGER.info(
        "LATEST_CANDIDATES_CSV_WRITTEN path=%s rows=%s run_date=%s latest_run_ts=%s",
        out_path,
        len(export.index),
        run_date,
        latest_run_ts,
    )
    return len(export.index)


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", required=True, help="Run date in YYYY-MM-DD")
    parser.add_argument("--out", default="data/latest_candidates.csv", help="Output CSV path")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        export_latest_candidates(args.run_date, Path(args.out))
        return 0
    except Exception as exc:  # pragma: no cover
        LOGGER.error("LATEST_CANDIDATES_CSV_FAILED err=%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
