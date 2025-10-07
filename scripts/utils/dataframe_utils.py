"""DataFrame helpers for screener normalization."""
from __future__ import annotations

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)

BARS_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def _empty_bars_df() -> pd.DataFrame:
    """Return an empty bars DataFrame with the expected schema."""

    return pd.DataFrame(columns=BARS_COLUMNS)


def to_bars_df(bars) -> pd.DataFrame:
    """Normalize Alpaca bars/HTTP bars to a flat DataFrame.

    The resulting DataFrame always contains the columns:
    ``symbol, timestamp, open, high, low, close, volume``.
    The helper understands the following shapes:

    A) ``alpaca-py`` ``BarsSet`` with ``.df`` (MultiIndex ``[symbol, timestamp]``)
    B) ``alpaca-py`` ``BarsSet`` with ``.data`` (``dict[str, list[Bar]]``)
    C) HTTP ``list[dict]`` with keys ``'S','t','o','h','l','c','v'``
    D) HTTP ``dict`` containing ``'bars': [...]``

    Empty inputs result in an empty DataFrame with the expected schema.
    """

    # Case A: BarsSet.df (MultiIndex or regular index)
    if hasattr(bars, "df"):
        df = getattr(bars, "df")
        if df is None or getattr(df, "empty", False):
            return _empty_bars_df()
        data = df
        if isinstance(data.index, pd.MultiIndex):
            data = data.reset_index()  # symbol, timestamp become columns
        rename = {
            "time": "timestamp",
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "S": "symbol",
        }
        data = data.rename(columns=rename)
        # If 'symbol' still missing but there is a single-symbol frame, add it
        if "symbol" not in data.columns and hasattr(bars, "data") and isinstance(bars.data, dict):
            symbols = [str(sym).upper() for sym in bars.data.keys() if sym]
            if len(symbols) == 1:
                data["symbol"] = symbols[0]
        result = data.copy()
        for column in BARS_COLUMNS:
            if column not in result.columns:
                result[column] = pd.NA
        if "symbol" not in result.columns:
            LOGGER.warning(
                "Unable to locate symbol column when normalizing bars; type=%s columns=%s",
                type(bars),
                list(result.columns),
            )
        result["symbol"] = result["symbol"].astype(str).str.upper()
        return result[BARS_COLUMNS].copy()

    # Case B: BarsSet.data (dict[str, list[Bar]])
    if hasattr(bars, "data") and isinstance(bars.data, dict):
        rows: list[dict[str, object]] = []
        for sym, items in bars.data.items():
            symbol = str(sym or "").upper()
            for bar in items or []:
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": getattr(bar, "timestamp", getattr(bar, "t", None)),
                        "open": getattr(bar, "open", getattr(bar, "o", None)),
                        "high": getattr(bar, "high", getattr(bar, "h", None)),
                        "low": getattr(bar, "low", getattr(bar, "l", None)),
                        "close": getattr(bar, "close", getattr(bar, "c", None)),
                        "volume": getattr(bar, "volume", getattr(bar, "v", None)),
                    }
                )
        return pd.DataFrame(rows, columns=BARS_COLUMNS) if rows else _empty_bars_df()

    # Case C/D: HTTP JSON payloads
    records: Iterable[dict] | None = None
    if isinstance(bars, dict) and "bars" in bars:
        records = bars.get("bars")
    elif isinstance(bars, (list, tuple)):
        records = bars
    frame = pd.DataFrame(list(records or []))
    if frame.empty:
        return _empty_bars_df()
    frame = frame.rename(
        columns={
            "S": "symbol",
            "Symbol": "symbol",
            "t": "timestamp",
            "time": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    for column in BARS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    if "symbol" not in frame.columns:
        LOGGER.warning(
            "HTTP bars payload missing symbol column after normalization; type=%s columns=%s",
            type(bars),
            list(frame.columns),
        )
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame[BARS_COLUMNS].copy()


__all__ = ["to_bars_df", "BARS_COLUMNS"]
