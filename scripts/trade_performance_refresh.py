"""CLI utility to refresh trade performance cache."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from utils.env import load_env

from scripts.trade_performance import (
    BASE_DIR,
    CACHE_PATH,
    DEFAULT_LOOKBACK_DAYS,
    build_data_client,
    cache_refresh_summary_token,
    refresh_trade_performance_cache,
)

LOG = logging.getLogger("trade_performance_refresh")


def _configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh trade performance cache.")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help="Lookback horizon for trade enrichment (default: 400).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force cache rebuild even if the cache looks fresh.",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Override base directory (defaults to JBRAVO_HOME or repository root).",
    )
    parser.add_argument(
        "--cache-path",
        default=str(CACHE_PATH),
        help="Path to write the trade performance cache.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    _configure_logging()
    load_env()
    args = parse_args(argv)
    base_dir = Path(args.base_dir) if args.base_dir else BASE_DIR
    cache_path = Path(args.cache_path) if args.cache_path else CACHE_PATH

    data_client = build_data_client()
    rc = 0
    try:
        df, summary = refresh_trade_performance_cache(
            base_dir=base_dir,
            data_client=data_client,
            lookback_days=int(args.lookback_days),
            force=bool(args.force),
            cache_path=cache_path,
        )
        enrichment_flags = df.get("needs_enrichment", [])
        trades_enriched = int(pd.Series(enrichment_flags, dtype=bool).sum()) if len(df.index) else 0
        token = cache_refresh_summary_token(
            len(df.index), trades_enriched, summary, int(args.lookback_days), rc
        )
    except Exception:
        LOG.exception("TRADE_PERFORMANCE_REFRESH_FAILED")
        rc = 1
        token = cache_refresh_summary_token(0, 0, {}, int(args.lookback_days), rc)

    print(token)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
