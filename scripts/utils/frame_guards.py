"""DataFrame guard utilities."""
from __future__ import annotations

import pandas as pd


def ensure_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a ``symbol`` column exists and is normalized."""

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
        return df

    for candidate in ("symbol_x", "symbol_y", "S", "Symbol"):
        if candidate in df.columns:
            df["symbol"] = df[candidate].astype(str).str.upper()
            return df

    if isinstance(df.index, pd.MultiIndex):
        if any((name or "").lower() == "symbol" for name in (df.index.names or [])):
            df = df.reset_index()
            df.rename(
                columns={name: "symbol" for name in df.columns if str(name).lower() == "symbol"},
                inplace=True,
            )
            df["symbol"] = df["symbol"].astype(str).str.upper()
            return df

    index_name = getattr(df.index, "name", None)
    if index_name and index_name.lower() == "symbol":
        df = df.reset_index().rename(columns={index_name: "symbol"})
        df["symbol"] = df["symbol"].astype(str).str.upper()
        return df

    df = df.copy()
    df["symbol"] = pd.Index(df.index).map(lambda x: str(x).upper() if x is not None else "")
    return df


__all__ = ["ensure_symbol_column"]
