"""Normalization helpers for Alpaca bar payloads."""
from __future__ import annotations

import pandas as pd

CANON = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in CANON:
        if column not in df.columns:
            df[column] = pd.NA
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df[CANON].copy()


def to_bars_df(obj, symbol_hint: str | None = None) -> pd.DataFrame:
    """Normalize Alpaca Market Data bars to the canonical columns."""

    # HTTP path
    if isinstance(obj, dict) and "bars" in obj:
        obj = obj["bars"]
    if isinstance(obj, (list, tuple)):
        df = pd.DataFrame(obj)
        df = df.rename(
            columns={
                "S": "symbol",
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "Symbol": "symbol",
                "time": "timestamp",
                "Time": "timestamp",
                "T": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        if df.empty:
            return pd.DataFrame(columns=CANON)
        return _ensure_columns(df)

    # SDK: BarsSet.df
    if hasattr(obj, "df"):
        df = obj.df or pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            names = [n or f"level_{i}" for i, n in enumerate(df.index.names)]
            df.index.set_names(names, inplace=True)
            df = df.reset_index()
        df = df.rename(
            columns={
                "S": "symbol",
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "time": "timestamp",
                "T": "timestamp",
            }
        )
        if "symbol" not in df.columns and symbol_hint:
            df["symbol"] = symbol_hint.upper()
        if df.empty:
            return pd.DataFrame(columns=CANON)
        return _ensure_columns(df)

    # SDK: BarsSet.data dict[str, list[Bar]]
    if hasattr(obj, "data") and isinstance(obj.data, dict):
        rows: list[dict[str, object]] = []
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
        return pd.DataFrame(rows, columns=CANON)

    # Unknown → empty canonical DF
    return pd.DataFrame(columns=CANON)


CANONICAL_BAR_COLUMNS = CANON
BARS_COLUMNS = CANON

__all__ = ["to_bars_df", "CANON", "CANONICAL_BAR_COLUMNS", "BARS_COLUMNS"]
