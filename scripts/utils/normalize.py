"""Normalization helpers for Alpaca bar payloads."""
from __future__ import annotations

from typing import Any

import pandas as pd

CANON = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def to_bars_df(obj: Any) -> pd.DataFrame:
    """Normalize a bars payload into a canonical DataFrame."""

    if isinstance(obj, dict) and "bars" in obj:
        obj = obj["bars"]

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
    elif hasattr(obj, "df") and isinstance(getattr(obj, "df"), pd.DataFrame):
        df = obj.df.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
    elif hasattr(obj, "data") and isinstance(getattr(obj, "data"), dict):
        rows: list[dict[str, Any]] = []
        for symbol, items in obj.data.items():
            for bar in items or []:
                rows.append(
                    {
                        "symbol": (
                            getattr(bar, "symbol", None)
                            or getattr(bar, "S", None)
                            or symbol
                        ),
                        "timestamp": getattr(bar, "timestamp", getattr(bar, "t", None)),
                        "open": getattr(bar, "open", getattr(bar, "o", None)),
                        "high": getattr(bar, "high", getattr(bar, "h", None)),
                        "low": getattr(bar, "low", getattr(bar, "l", None)),
                        "close": getattr(bar, "close", getattr(bar, "c", None)),
                        "volume": getattr(bar, "volume", getattr(bar, "v", None)),
                    }
                )
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(obj or [])
    rename_map = {
        "S": "symbol",
        "Symbol": "symbol",
        "t": "timestamp",
        "T": "timestamp",
        "time": "timestamp",
        "Time": "timestamp",
        "level_0": "symbol",
        "level_1": "timestamp",
        "index": "symbol",
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

    df = df.rename(
        columns={
            **rename_map,
        }
    )

    for column in CANON:
        if column not in df.columns:
            df[column] = pd.NA

    df["symbol"] = df["symbol"].astype("string").str.strip().str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    return df[CANON]


CANONICAL_BAR_COLUMNS = CANON
BARS_COLUMNS = CANON

__all__ = ["to_bars_df", "CANON", "CANONICAL_BAR_COLUMNS", "BARS_COLUMNS"]
