from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

REQUIRED = [
    "timestamp",
    "symbol",
    "score",
    "exchange",
    "close",
    "volume",
    "universe_count",
    "score_breakdown",
]


def _write_latest(df: pd.DataFrame, latest: Path, reason: str) -> tuple[int, str]:
    for column in REQUIRED:
        if column not in df.columns:
            df[column] = None
    if "entry_price" not in df.columns:
        df["entry_price"] = df["close"]
    keep = REQUIRED + [
        column for column in ["entry_price", "adv20", "atrp"] if column in df.columns
    ]
    df[keep].to_csv(latest, index=False)
    return len(df), reason


def ensure_min_candidates(base_dir: Path, min_rows: int = 1) -> Tuple[int, str]:
    """Ensure ``data/latest_candidates.csv`` contains at least ``min_rows`` rows."""

    data_dir = base_dir / "data"
    latest = data_dir / "latest_candidates.csv"
    scored = data_dir / "scored_candidates.csv"
    top = data_dir / "top_candidates.csv"

    if latest.exists():
        try:
            with latest.open("r", encoding="utf-8") as handle:
                rows = sum(1 for _ in handle) - 1
            if rows >= min_rows:
                return rows, "already_populated"
        except Exception:
            pass

    if scored.exists():
        df = pd.read_csv(scored)
        if "adv20" in df.columns:
            df = df[df["adv20"] >= 2_000_000]
        df = df[(df["close"] >= 1.0) & (df["close"] <= 60.0)]
        if "exchange" in df.columns:
            df = df[df["exchange"].isin(["NASDAQ", "NYSE", "AMEX"])]
        df = df.sort_values("score", ascending=False).head(max(1, min_rows))
        if len(df) > 0:
            return _write_latest(df, latest, "scored_candidates")

    if top.exists():
        df = pd.read_csv(top).head(max(1, min_rows))
        if len(df) > 0:
            return _write_latest(df, latest, "top_candidates")

    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "symbol": "AAPL",
                "score": 0.0,
                "exchange": "NASDAQ",
                "close": 1.0,
                "volume": 1,
                "universe_count": 1,
                "score_breakdown": "fallback",
                "entry_price": 1.0,
            }
        ]
    )
    return _write_latest(df, latest, "static_fallback")
