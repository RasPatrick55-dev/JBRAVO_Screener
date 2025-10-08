import pandas as pd


def ensure_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a 'symbol' column exists (uppercase), even if it drifted to index or _x/_y after joins."""
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper()
        return df

    # Look for common collisions
    for cand in ("symbol_x", "symbol_y", "S", "Symbol"):
        if cand in df.columns:
            df["symbol"] = df[cand].astype(str).str.upper()
            return df

    # Index-level rescue
    if isinstance(df.index, pd.MultiIndex) and df.index.names:
        for i, name in enumerate(df.index.names):
            if (name or "").lower() == "symbol":
                df = df.reset_index()
                df.rename(columns={name: "symbol"}, inplace=True)
                df["symbol"] = df["symbol"].astype(str).str.upper()
                return df
    if getattr(df.index, "name", None) and df.index.name.lower() == "symbol":
        name = df.index.name
        df = df.reset_index().rename(columns={name: "symbol"})
        df["symbol"] = df["symbol"].astype(str).str.upper()
        return df

    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
        first_col = df.columns[0]
        df.rename(columns={first_col: "symbol"}, inplace=True)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        return df

    # Last resort: create empty uppercase string column to avoid hard failure; downstream history check will drop it
    df = df.copy()
    df["symbol"] = pd.Series(dtype="string")
    return df
