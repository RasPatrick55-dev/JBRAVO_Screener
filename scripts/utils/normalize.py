"""Normalization helpers for Alpaca bar payloads."""
from __future__ import annotations

import pandas as pd


CANONICAL_BAR_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# Backwards compatibility for callers that previously imported ``BARS_COLUMNS``.
BARS_COLUMNS = CANONICAL_BAR_COLUMNS


def _ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the canonical schema is present on ``frame``."""

    df = frame.copy()
    rename = {
        "S": "symbol",
        "Symbol": "symbol",
        "symbol": "symbol",
        "T": "timestamp",
        "t": "timestamp",
        "time": "timestamp",
        "Time": "timestamp",
        "Timestamp": "timestamp",
        "o": "open",
        "O": "open",
        "Open": "open",
        "h": "high",
        "H": "high",
        "High": "high",
        "l": "low",
        "L": "low",
        "Low": "low",
        "c": "close",
        "C": "close",
        "Close": "close",
        "v": "volume",
        "V": "volume",
        "Volume": "volume",
    }
    df = df.rename(columns=rename)
    for column in CANONICAL_BAR_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df[CANONICAL_BAR_COLUMNS].copy()


def _normalize_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names or [])
        # Ensure the first two levels map to symbol/timestamp for Alpaca SDK payloads.
        if len(names) >= 1 and not names[0]:
            names[0] = "symbol"
        if len(names) >= 2 and not names[1]:
            names[1] = "timestamp"
        df = df.reset_index()
    elif df.index.name in {"symbol", "timestamp"}:
        df = df.reset_index()
    return df


def to_bars_df(obj, symbol_hint: str | None = None) -> pd.DataFrame:
    """Normalize HTTP or SDK responses to a canonical bars DataFrame."""

    if isinstance(obj, dict) and "bars" in obj:
        obj = obj.get("bars")

    if isinstance(obj, (list, tuple)):
        frame = pd.DataFrame(list(obj))
        if frame.empty:
            return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
        return _ensure_columns(frame)

    if hasattr(obj, "df"):
        raw_df = getattr(obj, "df")
        df = raw_df if raw_df is not None else pd.DataFrame()
        df = _normalize_multiindex(df.copy())
        if "symbol" not in df.columns and symbol_hint:
            df["symbol"] = symbol_hint
        return _ensure_columns(df)

    if hasattr(obj, "data") and isinstance(obj.data, dict):
        rows: list[dict[str, object]] = []
        for sym, items in obj.data.items():
            upper = str(sym or "").strip().upper()
            for bar in items or []:
                rows.append(
                    {
                        "symbol": upper,
                        "timestamp": getattr(bar, "timestamp", getattr(bar, "t", None)),
                        "open": getattr(bar, "open", getattr(bar, "o", None)),
                        "high": getattr(bar, "high", getattr(bar, "h", None)),
                        "low": getattr(bar, "low", getattr(bar, "l", None)),
                        "close": getattr(bar, "close", getattr(bar, "c", None)),
                        "volume": getattr(bar, "volume", getattr(bar, "v", None)),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
        return _ensure_columns(pd.DataFrame(rows))

    if isinstance(obj, pd.DataFrame):
        frame = _normalize_multiindex(obj.copy())
        if "symbol" not in frame.columns and symbol_hint:
            frame["symbol"] = symbol_hint
        return _ensure_columns(frame)

    return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)


__all__ = ["to_bars_df", "CANONICAL_BAR_COLUMNS", "BARS_COLUMNS"]

