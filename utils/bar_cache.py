import os
import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.models.bars import BarSet


def cache_bars(symbol: str, bars_or_df, cache_dir: str = "cache") -> None:
    """Save historical bar data for ``symbol`` to disk.

    Accepts either an Alpaca ``BarSet`` or a ``pandas.DataFrame`` and writes
    the underlying data to ``cache_dir``.
    """

    if isinstance(bars_or_df, BarSet):
        df = bars_or_df.df.reset_index()
    elif isinstance(bars_or_df, pd.DataFrame):
        df = bars_or_df
    else:
        raise TypeError(f"Expected BarSet or DataFrame, got {type(bars_or_df)}")

    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, f"{symbol}_bars.csv")
    df.to_csv(filepath, index=False)
