"""
Utilities to determine whether an open position should be closed early.

These functions encapsulate the exit rules described in the Johnny‑Bravo
trading guide and complement the time‑based and trailing stop exits
already present in ``execute_trades.py``.  Import these helpers from
``execute_trades.py`` to trigger additional liquidation when momentum
fades or a trend reverses.

Example usage in ``execute_trades.py``::

    from exit_signals import should_exit_early
    # inside daily_exit_check()
    if should_exit_early(symbol, data_client, DATA_CACHE_DIR):
        # submit market sell order

This design keeps the core trading script clean and allows the exit
logic to evolve independently of order placement.
"""

from __future__ import annotations

import logging
from typing import Optional
import pandas as pd

from indicators import rsi, macd
from utils import cache_bars


def should_exit_early(symbol: str, data_client, cache_dir: str, lookback: int = 100) -> bool:
    """Return True if the position in ``symbol`` should be exited early.

    This function evaluates several momentum‑based exit conditions:

    * Price has closed below the 20‑period EMA after previously closing
      above it (trend breakdown).
    * RSI exceeds 70, signalling overbought conditions【514888614236530†L246-L284】.
    * The MACD histogram crosses below zero, indicating a shift from
      bullish to bearish momentum.

    The function downloads up to ``lookback`` recent bars via
    ``cache_bars``, computes the necessary indicators and returns a
    boolean.  Any failure to retrieve data or compute indicators will
    result in ``False`` to avoid accidental liquidation due to data
    issues.

    Parameters
    ----------
    symbol : str
        Ticker of the open position.
    data_client : StockHistoricalDataClient
        Alpaca data client used to fetch historical bars.
    cache_dir : str
        Directory where bar data are cached.
    lookback : int, optional
        Number of recent bars to consider, by default 100.  A larger
        value yields more robust EMAs but requires more data.

    Returns
    -------
    bool
        ``True`` if exit conditions are met, ``False`` otherwise.
    """
    try:
        df = cache_bars(symbol, data_client, cache_dir)
        # Focus on the most recent ``lookback`` bars
        if len(df) < 25:
            # Not enough data to compute the indicators
            return False
        df = df.sort_index().iloc[-lookback:].copy()

        # Compute 20‑period EMA, RSI and MACD histogram
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["rsi"] = rsi(df["close"])
        macd_line, macd_signal, macd_hist = macd(df["close"])
        df["macd_hist"] = macd_hist

        last = df.iloc[-1]
        prev = df.iloc[-2]
        # Exit if price breaks the EMA20 from above
        ema_break = last["close"] < last["ema20"] and prev["close"] >= prev["ema20"]
        # Exit if RSI is overbought
        overbought = last["rsi"] > 70
        # Exit if MACD histogram turns negative
        macd_flip = last["macd_hist"] < 0 and prev["macd_hist"] >= 0
        return bool(ema_break or overbought or macd_flip)
    except Exception as exc:
        logging.error("Early exit check failed for %s: %s", symbol, exc)
        return False