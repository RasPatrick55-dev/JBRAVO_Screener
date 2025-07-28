"""Utility helpers for pipeline scripts."""
import os
import shutil
import logging
import pandas as pd
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta, timezone, time as dt_time
import pytz
import time
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError


def has_datetime_index(idx) -> bool:
    """Return True if the given index has a datetime dtype."""
    return hasattr(idx, "dtype") and pd.api.types.is_datetime64_any_dtype(idx.dtype)


def write_csv_atomic(path: str, df: pd.DataFrame) -> None:
    tmp = NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), newline="")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    shutil.move(tmp.name, path)


def fetch_bars_with_cutoff(symbol: str, start, data_client) -> pd.DataFrame:
    """Return daily bars from ``start`` using ``get_stock_bars``."""

    if not isinstance(start, datetime):
        start = pd.to_datetime(start)

    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            feed="sip",
        )
        bars = data_client.get_stock_bars(request_params).df
        if bars.empty:
            raise ValueError("No SIP data available")
    except Exception as e:
        logging.warning(
            "SIP data unavailable for %s, falling back to IEX feed: %s",
            symbol,
            e,
        )
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            feed="iex",
        )
        bars = data_client.get_stock_bars(request_params).df
    if bars.empty:
        return pd.DataFrame()
    return bars


def get_last_trading_day_end(now: datetime | None = None) -> datetime:
    """Return previous market close timestamp in UTC.

    This helps when running during weekends or outside market hours.
    """
    tz = pytz.timezone("America/New_York")
    now = now.astimezone(tz) if now else datetime.now(tz)

    # Determine the most recent trading day
    weekday = now.weekday()
    if weekday >= 5:  # Sat/Sun -> go back to Friday
        delta = weekday - 4
        last_day = (now - timedelta(days=delta)).date()
    elif now.time() < dt_time(9, 30):
        # before market open -> use previous weekday
        delta = 3 if weekday == 0 else 1
        last_day = (now - timedelta(days=delta)).date()
    else:
        # During trading hours or after close use today
        last_day = now.date() if weekday < 5 else (now - timedelta(days=weekday - 4)).date()

    close_dt = datetime.combine(last_day, dt_time(16, 0), tz)
    return close_dt.astimezone(timezone.utc)


def cache_bars(
    symbol: str,
    data_client,
    cache_dir: str,
    days: int = 1500,
    timeframe: TimeFrame = TimeFrame.Day,
) -> pd.DataFrame | None:
    """Fetch and cache bars for ``symbol``.

    For ``TimeFrame.Day`` the behaviour mirrors the original implementation
    where data are cached to disk.  When ``timeframe`` is ``TimeFrame.Minute``
    only a small intraday window is requested (most recent hour delayed by
    16 minutes) and the resulting DataFrame is returned without touching the
    cache.  ``None`` is returned when no data could be fetched.
    """

    os.makedirs(cache_dir, exist_ok=True)

    # Intraday minute bars are not cached to disk.  They are fetched for the
    # last hour with a 16 minute delay to respect Alpaca's data policies.
    if timeframe == TimeFrame.Minute:
        start = datetime.now(timezone.utc) - timedelta(hours=1, minutes=16)
        try:
            end_safe = datetime.now(timezone.utc) - timedelta(minutes=16)
            req = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Minute,
                start=start,
                end=end_safe,
            )
            df = data_client.get_stock_bars(req).df
            if df.empty:
                cutoff = datetime.now(timezone.utc) - timedelta(minutes=16)
                logging.warning(
                    "No bars returned for %s from %s to %s. Retrying with previous day's close.",
                    symbol,
                    start,
                    cutoff.isoformat(),
                )
                try:
                    prev_close = data_client.get_latest_trade(symbol).price
                    df = pd.DataFrame([{"close": prev_close}], index=[datetime.now(timezone.utc)])
                    logging.info("Using previous close price for %s: %s", symbol, prev_close)
                except Exception as exc:
                    logging.error("Error fetching latest trade for %s: %s", symbol, exc)
                    return None
            if isinstance(df.index, pd.MultiIndex):
                df = df.droplevel("symbol") if "symbol" in df.index.names else df.droplevel(0)
            df.index = pd.to_datetime(df.index)
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize("UTC")
            logging.info("Received %d bars for %s", len(df), symbol)
            return df
        except Exception as e:  # pragma: no cover - network errors
            logging.error("Market data fetch error for %s: %s", symbol, e)
            return None

    path = os.path.join(cache_dir, f"{symbol}.csv")
    df = (
        pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        if os.path.exists(path)
        else pd.DataFrame()
    )
    last = df.index.max() if not df.empty else None
    start = last + timedelta(days=1) if last is not None else datetime.now(timezone.utc) - timedelta(days=days)
    end = get_last_trading_day_end()
    if start >= end:
        return df

    try:
        new_df = fetch_bars_with_cutoff(symbol, start, data_client)
    except APIError as exc:
        if getattr(exc, "status_code", None) == 429:
            logging.error("Alpaca rate limit hit fetching %s", symbol)
        else:
            logging.error("Alpaca API error fetching %s: %s", symbol, exc)
        return df
    except Exception as exc:
        logging.error("Unexpected error fetching %s: %s", symbol, exc)
        return df

    # Drop the symbol level when returned as a MultiIndex
    if isinstance(new_df.index, pd.MultiIndex):
        if "symbol" in new_df.index.names:
            new_df = new_df.droplevel("symbol")
        else:
            new_df = new_df.droplevel(0)

    # Validate that the DataFrame has a date-based index
    if "timestamp" not in new_df.columns and not has_datetime_index(new_df.index):
        logging.warning(
            "cache_bars: %s returned invalid index type %s", symbol, new_df.index
        )
        return pd.DataFrame()

    if new_df.empty:
        logging.warning("cache_bars: %s returned empty data", symbol)
        return df

    if not new_df.empty:
        idx_list = new_df.index.tolist()
        if idx_list and isinstance(idx_list[0], tuple):
            n = len(idx_list[0])
            raw = list(idx_list)
            if n == 2:
                raw = [(y, m, 1) for (y, m) in raw]  # default day = 1
            idx_df = pd.DataFrame(raw, columns=["year", "month", "day"])
            new_df.index = pd.to_datetime(idx_df)
        else:
            new_df.index = pd.to_datetime(new_df.index)
        if new_df.index.tzinfo is None:
            new_df.index = new_df.index.tz_localize("UTC")
        df = pd.concat([df, new_df]).sort_index()
        df = df[~df.index.duplicated(keep="last")].tail(days)
        tmp = NamedTemporaryFile("w", delete=False, dir=cache_dir, newline="")
        df.reset_index().to_csv(tmp.name, index=False)
        tmp.close()
        shutil.move(tmp.name, path)
    return df


def cache_bars_batch(symbols: list[str], data_client, cache_dir: str, days: int = 1500, batch_size: int = 100, retries: int = 3) -> dict[str, pd.DataFrame]:
    """Fetch historical bars for many symbols in batches.

    Returns a mapping of symbol to DataFrame containing the updated cache.
    """

    results: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        start_map = {}
        df_map = {}
        for sym in batch:
            path = os.path.join(cache_dir, f"{sym}.csv")
            df_existing = (
                pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
                if os.path.exists(path)
                else pd.DataFrame()
            )
            df_map[sym] = df_existing
            last = df_existing.index.max() if not df_existing.empty else None
            start_map[sym] = last + timedelta(days=1) if last is not None else datetime.now(timezone.utc) - timedelta(days=days)

        end = get_last_trading_day_end()
        min_start = min(start_map.values())
        now_utc = datetime.now(timezone.utc)
        end_safe = now_utc - timedelta(minutes=16)
        request_params = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=min_start,
            end=end_safe.isoformat(),
            feed="iex",
        )

        attempt = 0
        bars_df = pd.DataFrame()
        while attempt <= retries:
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=min_start,
                    end=end_safe.isoformat(),
                    feed="sip",
                )
                bars_df = data_client.get_stock_bars(req).df
                if bars_df.empty:
                    raise ValueError("No SIP data available")
                break
            except APIError as exc:
                if getattr(exc, "status_code", None) == 429:
                    logging.warning("Rate limit hit for batch %s, retrying...", batch)
                    time.sleep(1)
                    attempt += 1
                    continue
                logging.error("Alpaca API error for batch %s: %s", batch, exc)
                break
            except Exception as exc:
                logging.warning(
                    "SIP data unavailable for %s, falling back to IEX feed: %s",
                    batch,
                    exc,
                )
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame.Day,
                        start=min_start,
                        end=end_safe.isoformat(),
                        feed="iex",
                    )
                    bars_df = data_client.get_stock_bars(req).df
                except Exception as exc2:
                    logging.error("Unexpected error for batch %s: %s", batch, exc2)
                break

        if isinstance(bars_df.index, pd.MultiIndex) and "symbol" in bars_df.index.names:
            for sym in batch:
                sym_df = bars_df.xs(sym, level="symbol") if not bars_df.empty else pd.DataFrame()
                df_map[sym] = pd.concat([df_map[sym], sym_df]).sort_index()
        else:
            for sym in batch:
                sym_df = bars_df[bars_df.get("symbol") == sym].drop(columns=["symbol"], errors="ignore") if not bars_df.empty else pd.DataFrame()
                df_map[sym] = pd.concat([df_map[sym], sym_df]).sort_index()

        for sym, df_sym in df_map.items():
            df_sym = df_sym[~df_sym.index.duplicated(keep="last")].tail(days)
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, f"{sym}.csv")
            write_csv_atomic(path, df_sym.reset_index())
            results[sym] = df_sym

    return results


def fetch_daily_bars(symbol: str, trade_date: str, data_client) -> pd.DataFrame:
    """Return the daily bar for ``symbol`` on ``trade_date`` using SIP with
    IEX fallback.

    Parameters
    ----------
    symbol : str
        The ticker to fetch.
    trade_date : str
        ISO formatted date (``YYYY-MM-DD``).
    data_client : StockHistoricalDataClient
        Alpaca data client used to request the bars.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the daily bar indexed by timestamp.
    """

    bars = fetch_bars_with_cutoff(symbol, trade_date, data_client)
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.droplevel("symbol") if "symbol" in bars.index.names else bars.droplevel(0)
    return bars


def fetch_extended_hours_bars(symbol: str, trade_date: str, data_client) -> tuple[int, pd.DataFrame]:
    """Fetch pre- and post-market minute bars and return the total volume.

    The returned DataFrame contains all minute bars between 08:00 and 17:00
    US/Eastern for ``trade_date`` retrieved from the IEX feed.
    """

    tz = pytz.timezone("America/New_York")
    day = datetime.strptime(trade_date, "%Y-%m-%d").date()
    start = tz.localize(datetime.combine(day, dt_time(8, 0)))
    end_dt = tz.localize(datetime.combine(day, dt_time(17, 0)))

    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start,
        end=end_dt,
    )
    bars = data_client.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.droplevel("symbol") if "symbol" in bars.index.names else bars.droplevel(0)
    if bars.index.tzinfo is None:
        bars.index = bars.index.tz_localize("UTC").tz_convert(tz)
    else:
        bars.index = bars.index.tz_convert(tz)

    pre = bars.between_time("08:00", "09:30")["volume"].sum()
    post = bars.between_time("16:00", "17:00")["volume"].sum()
    return int(pre + post), bars


def get_combined_daily_bar(symbol: str, trade_date: str, data_client) -> pd.DataFrame:
    """Return a daily bar adjusted with extended hours data."""

    daily = fetch_daily_bars(symbol, trade_date, data_client)
    ext_vol, ext_bars = fetch_extended_hours_bars(symbol, trade_date, data_client)

    if daily.empty:
        return daily

    idx = daily.index[0]
    daily.loc[idx, "volume"] = daily.loc[idx, "volume"] + ext_vol

    if not ext_bars.empty:
        ext_high = ext_bars["high"].max()
        ext_low = ext_bars["low"].min()
        close_series = ext_bars.between_time("16:00", "17:00")["close"]
        ext_close = close_series.iloc[-1] if not close_series.empty else daily.loc[idx, "close"]

        daily.loc[idx, "high"] = max(daily.loc[idx, "high"], ext_high)
        daily.loc[idx, "low"] = min(daily.loc[idx, "low"], ext_low)
        daily.loc[idx, "close"] = ext_close

    return daily

