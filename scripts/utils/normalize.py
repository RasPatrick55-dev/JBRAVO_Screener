"""Normalization helpers for bar payloads."""
from __future__ import annotations

from typing import Any

import pandas as pd

CANON = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def to_bars_df(obj: Any) -> pd.DataFrame:
    """Normalize a bars payload into a canonical DataFrame."""

    if isinstance(obj, dict) and "bars" in obj:
        obj = obj["bars"]

    df = pd.DataFrame(obj or [])
    df = df.rename(
        columns={
            "S": "symbol",
            "Symbol": "symbol",
            "t": "timestamp",
            "T": "timestamp",
            "time": "timestamp",
            "Time": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    for column in CANON:
        if column not in df.columns:
            df[column] = pd.NA

    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    return df[CANON].copy()


CANONICAL_BAR_COLUMNS = CANON
BARS_COLUMNS = CANON

__all__ = ["to_bars_df", "CANON", "CANONICAL_BAR_COLUMNS", "BARS_COLUMNS"]
