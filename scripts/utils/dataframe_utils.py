"""DataFrame helpers for screener normalization."""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)

BARS_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def _empty_bars_df() -> pd.DataFrame:
    """Return an empty bars DataFrame with the expected schema."""

    return pd.DataFrame(columns=BARS_COLUMNS)


def to_bars_df(bars_obj, symbol_hint: str | None = None) -> pd.DataFrame:
    """Normalize Alpaca/HTTP bar payloads to a canonical DataFrame.

    The resulting DataFrame always contains the columns ``symbol, timestamp, open,
    high, low, close, volume``. Unknown or empty inputs yield an empty DataFrame
    with the expected schema.
    """

    required = BARS_COLUMNS

    def _index_names(index: pd.MultiIndex) -> list[str]:
        raw_names = list(index.names or [])
        names: list[str] = []
        for idx, name in enumerate(raw_names):
            if name:
                names.append(str(name))
                continue
            if idx == 0:
                names.append("symbol")
            elif idx == 1:
                names.append("timestamp")
            else:
                names.append(f"level_{idx}")
        return names

    def _standardize(df: pd.DataFrame) -> pd.DataFrame:
        rename = {
            "S": "symbol",
            "Symbol": "symbol",
            "symbol": "symbol",
            "t": "timestamp",
            "time": "timestamp",
            "Time": "timestamp",
            "Timestamp": "timestamp",
            "T": "timestamp",
            "o": "open",
            "O": "open",
            "open": "open",
            "Open": "open",
            "h": "high",
            "H": "high",
            "high": "high",
            "High": "high",
            "l": "low",
            "L": "low",
            "low": "low",
            "Low": "low",
            "c": "close",
            "C": "close",
            "close": "close",
            "Close": "close",
            "v": "volume",
            "V": "volume",
            "volume": "volume",
            "Volume": "volume",
        }
        df = df.rename(columns=rename)
        for column in required:
            if column not in df.columns:
                df[column] = pd.NA
        if symbol_hint and df.empty:
            df["symbol"] = symbol_hint.upper()
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        return df[required]

    if isinstance(bars_obj, pd.DataFrame):
        frame = bars_obj.copy()
        if isinstance(frame.index, pd.MultiIndex):
            names = _index_names(frame.index)
            frame.index.set_names(names, inplace=True)
            frame = frame.reset_index()
        elif frame.index.name in {"symbol", "timestamp"}:
            frame = frame.reset_index()
        if "symbol" not in frame.columns and symbol_hint:
            frame["symbol"] = symbol_hint.upper()
        return _standardize(frame)

    if hasattr(bars_obj, "df"):
        df = getattr(bars_obj, "df")
        if df is None or getattr(df, "empty", False):
            return _empty_bars_df()
        frame = df.copy()
        if isinstance(frame.index, pd.MultiIndex):
            names = _index_names(frame.index)
            frame.index.set_names(names, inplace=True)
            frame = frame.reset_index()
        if "symbol" not in frame.columns:
            if symbol_hint:
                frame["symbol"] = symbol_hint.upper()
            elif hasattr(bars_obj, "data") and isinstance(bars_obj.data, dict) and bars_obj.data:
                if len(bars_obj.data) == 1:
                    frame["symbol"] = str(next(iter(bars_obj.data.keys()))).upper()
        return _standardize(frame)

    if hasattr(bars_obj, "data") and isinstance(bars_obj.data, dict):
        rows: list[dict[str, object]] = []
        for symbol, items in bars_obj.data.items():
            sym = str(symbol or "").upper()
            for bar in items or []:
                rows.append(
                    {
                        "symbol": sym,
                        "timestamp": getattr(bar, "timestamp", getattr(bar, "t", None)),
                        "open": getattr(bar, "open", getattr(bar, "o", None)),
                        "high": getattr(bar, "high", getattr(bar, "h", None)),
                        "low": getattr(bar, "low", getattr(bar, "l", None)),
                        "close": getattr(bar, "close", getattr(bar, "c", None)),
                        "volume": getattr(bar, "volume", getattr(bar, "v", None)),
                    }
                )
        return _standardize(pd.DataFrame(rows)) if rows else _empty_bars_df()

    records: Iterable[dict] | None = None
    if isinstance(bars_obj, dict) and "bars" in bars_obj:
        records = bars_obj.get("bars")
    elif isinstance(bars_obj, (list, tuple)):
        records = bars_obj
    if records is not None:
        df = pd.DataFrame(list(records))
        if df.empty and symbol_hint:
            return _empty_bars_df()
        return _standardize(df)

    return _empty_bars_df()


__all__ = ["to_bars_df", "BARS_COLUMNS"]
