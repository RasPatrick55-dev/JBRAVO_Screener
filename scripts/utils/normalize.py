"""Normalization helpers for Alpaca bar payloads."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .frame_guards import ensure_symbol_column

CANON = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def to_bars_df(obj: Any) -> pd.DataFrame:
    """Normalize a bars payload into a canonical DataFrame."""

    if isinstance(obj, dict) and "bars" in obj:
        obj = obj["bars"]

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    elif hasattr(obj, "df") and isinstance(getattr(obj, "df"), pd.DataFrame):
        df = getattr(obj, "df").copy()
    elif hasattr(obj, "data") and isinstance(getattr(obj, "data"), dict):
        rows: list[dict[str, Any]] = []
        for symbol, items in getattr(obj, "data").items():
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
        if obj is None:
            df = pd.DataFrame([])
        else:
            df = pd.DataFrame(obj)

    # Backward tolerance: rename any variant into canon
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

    df = ensure_symbol_column(df)

    # Ensure canon columns exist
    for column in CANON:
        if column not in df.columns:
            df[column] = pd.NA

    # Dtypes
    df["symbol"] = df["symbol"].astype("string").str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    # Return *flat* canon frame
    return df[CANON].copy()


CANONICAL_BAR_COLUMNS = CANON
BARS_COLUMNS = CANON

__all__ = ["to_bars_df", "CANON", "CANONICAL_BAR_COLUMNS", "BARS_COLUMNS"]
