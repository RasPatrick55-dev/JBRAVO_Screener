import os
import pandas as pd
from alpaca.data.historical import StockBars


def cache_bars(symbol: str, bars_or_df, cache_dir: str = "cache") -> None:
    """Cache historical bar data for ``symbol``.

    Parameters
    ----------
    symbol: str
        Stock ticker symbol (e.g., ``"AAPL"``).
    bars_or_df: pandas.DataFrame | StockBars
        The data to cache.  Can be a DataFrame directly or an Alpaca
        ``StockBars`` object.
    cache_dir: str, optional
        Directory to store cached CSV files.
    """

    # Auto-detect input type and extract DataFrame
    if hasattr(bars_or_df, "df"):
        df = bars_or_df.df.reset_index()
    elif isinstance(bars_or_df, pd.DataFrame):
        df = bars_or_df
    else:
        raise TypeError(
            f"Unsupported type: {type(bars_or_df)} provided. Expecting DataFrame or Bars object."
        )

    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, f"{symbol}_bars.csv")
    df.to_csv(filepath, index=False)
