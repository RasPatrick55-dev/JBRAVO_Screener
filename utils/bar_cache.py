import os
import pandas as pd


def cache_bars(symbol: str, df: pd.DataFrame, cache_dir: str = "cache") -> None:
    """Cache OHLC bar data for a stock symbol to CSV.

    Parameters:
    - symbol (str): Stock ticker symbol (e.g., 'AAPL').
    - df (pd.DataFrame): DataFrame containing bar data (open, high, low, close, indicators).
    - cache_dir (str): Directory to store cached CSV files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, f"{symbol}_bars.csv")
    df.to_csv(filepath, index=False)
