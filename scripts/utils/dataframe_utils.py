"""DataFrame helpers for screener normalization."""
from __future__ import annotations

import pandas as pd

BARS_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


def _empty_bars_df() -> pd.DataFrame:
    """Return an empty bars DataFrame with the expected schema."""

    return pd.DataFrame(columns=BARS_COLUMNS)


def _multiindex_names(index: pd.MultiIndex) -> list[str]:
    raw = list(index.names or [])
    names: list[str] = []
    for idx, name in enumerate(raw):
        if name:
            names.append(str(name))
        elif idx == 0:
            names.append("symbol")
        elif idx == 1:
            names.append("timestamp")
        else:
            names.append(f"level_{idx}")
    return names


def _standardize_columns(df: pd.DataFrame, symbol_hint: str | None = None) -> pd.DataFrame:
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
    frame = df.rename(columns=rename)
    for column in BARS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    if "symbol" not in frame.columns:
        frame["symbol"] = pd.NA
    if symbol_hint:
        symbol_series = frame["symbol"]
        if symbol_series.isna().all():
            frame["symbol"] = symbol_hint
        else:
            cleaned = symbol_series.fillna("").astype(str).str.strip()
            if cleaned.eq("").all():
                frame["symbol"] = symbol_hint
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    return frame[BARS_COLUMNS].copy()


def to_bars_df(obj, symbol_hint: str | None = None) -> pd.DataFrame:
    """Normalize Alpaca/HTTP bar payloads to a canonical DataFrame."""

    if isinstance(obj, dict) and "bars" in obj:
        obj = obj.get("bars")

    if isinstance(obj, (list, tuple)):
        frame = pd.DataFrame(list(obj))
        if frame.empty:
            return _empty_bars_df()
        return _standardize_columns(frame, symbol_hint)

    if hasattr(obj, "df"):
        df = getattr(obj, "df")
        if df is None:
            return _empty_bars_df()
        frame = df.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame.index.set_names(_multiindex_names(frame.index), inplace=True)
            frame = frame.reset_index()
        elif frame.index.name in {"symbol", "timestamp"}:
            frame = frame.reset_index()
        if "symbol" not in frame.columns and symbol_hint:
            frame["symbol"] = symbol_hint
        return _standardize_columns(frame, symbol_hint)

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
            return _empty_bars_df()
        return _standardize_columns(pd.DataFrame(rows), symbol_hint)

    if isinstance(obj, pd.DataFrame):
        frame = obj.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame.index.set_names(_multiindex_names(frame.index), inplace=True)
            frame = frame.reset_index()
        elif frame.index.name in {"symbol", "timestamp"}:
            frame = frame.reset_index()
        if "symbol" not in frame.columns and symbol_hint:
            frame["symbol"] = symbol_hint
        return _standardize_columns(frame, symbol_hint)

    return _empty_bars_df()


__all__ = ["to_bars_df", "BARS_COLUMNS"]
