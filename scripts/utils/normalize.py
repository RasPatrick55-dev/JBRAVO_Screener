"""Normalization helpers for Alpaca bar payloads."""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

CANON = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in CANON:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def _coerce_symbol(df: pd.DataFrame, symbol_hint: str | None = None) -> pd.DataFrame:
    if symbol_hint:
        df["symbol"] = df["symbol"].fillna(symbol_hint)
        mask = df["symbol"].astype(str).str.strip() == ""
        if mask.any():
            df.loc[mask, "symbol"] = symbol_hint
    df["symbol"] = df["symbol"].astype("string").str.strip().str.upper()
    df = df[df["symbol"].notna()]
    df = df[df["symbol"] != ""]
    return df


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "S": "symbol",
            "Symbol": "symbol",
            "t": "timestamp",
            "T": "timestamp",
            "time": "timestamp",
            "Time": "timestamp",
            "o": "open",
            "Open": "open",
            "h": "high",
            "High": "high",
            "l": "low",
            "Low": "low",
            "c": "close",
            "Close": "close",
            "v": "volume",
            "Volume": "volume",
        }
    )


def _from_iterable(obj: Iterable[Any]) -> pd.DataFrame:
    df = pd.DataFrame(obj)
    df = _rename_columns(df)
    return df


def to_bars_df(obj: Any, symbol_hint: str | None = None) -> pd.DataFrame:
    """Normalize Alpaca Market Data bars to the canonical columns."""

    if isinstance(obj, dict) and "bars" in obj:
        obj = obj["bars"]

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        df = _rename_columns(df)
    elif isinstance(obj, (list, tuple)):
        df = _from_iterable(obj)
    elif hasattr(obj, "df") or hasattr(obj, "data"):
        try:
            if hasattr(obj, "df") and obj.df is not None:
                df = obj.df
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                df = _rename_columns(df)
            elif hasattr(obj, "data") and isinstance(obj.data, dict):
                rows: list[dict[str, Any]] = []
                for sym, items in obj.data.items():
                    for bar in items or []:
                        rows.append(
                            {
                                "symbol": str(sym).upper(),
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
                df = pd.DataFrame(columns=CANON)
        except Exception:
            df = pd.DataFrame(columns=CANON)
    else:
        df = pd.DataFrame(columns=CANON)

    df = _ensure_columns(df)
    df = _coerce_symbol(df, symbol_hint)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

    return df[CANON]


CANONICAL_BAR_COLUMNS = CANON
BARS_COLUMNS = CANON

__all__ = ["to_bars_df", "CANON", "CANONICAL_BAR_COLUMNS", "BARS_COLUMNS"]
