import os
import datetime
import pandas as pd
import logging
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
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


def fetch_bars_with_cutoff(
    symbol: str, cutoff_ts: datetime.datetime, data_client
) -> pd.DataFrame:
    """Fetch daily bars up to ``cutoff_ts`` inclusive.

    The request uses the IEX feed which includes extended trading hours data.
    The returned DataFrame is indexed by timestamp in UTC and filtered so that
    no rows exceed ``cutoff_ts``.
    """

    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=None,
            end=cutoff_ts.isoformat(),
            feed="iex",
        )
        bars = data_client.get_stock_bars(request).df
        if bars.empty:
            logging.warning("No bars available for %s using IEX feed", symbol)
            return pd.DataFrame()
    except Exception as e:  # pragma: no cover - network errors
        logging.error("Failed to fetch bars for %s via IEX: %s", symbol, e)
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        bars = (
            bars.droplevel("symbol") if "symbol" in bars.index.names else bars.droplevel(0)
        )
    bars.index = pd.to_datetime(bars.index)
    if bars.index.tzinfo is None:
        bars.index = bars.index.tz_localize("UTC")
    return bars[bars.index <= cutoff_ts]
