"""Utility helpers for pipeline scripts."""
import os
import shutil
import pandas as pd
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta, timezone
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def write_csv_atomic(df: pd.DataFrame, dest: str) -> None:
    tmp = NamedTemporaryFile("w", delete=False, dir=os.path.dirname(dest), newline="")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    shutil.move(tmp.name, dest)


def cache_bars(symbol: str, data_client, cache_dir: str, days: int = 800) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}.csv")
    df = (
        pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        if os.path.exists(path)
        else pd.DataFrame()
    )
    last = df.index.max() if not df.empty else None
    start = last + timedelta(days=1) if last is not None else datetime.now(timezone.utc) - timedelta(days=days)
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    if start >= end:
        return df

    request_params = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start, end=end)
    try:
        new_df = data_client.get_stock_bars(request_params).df
    except Exception:
        return df

    if not new_df.empty:
        idx_list = new_df.index.tolist()
        if idx_list and isinstance(idx_list[0], tuple):
            n = len(idx_list[0])
            col_names = ["year", "month", "day"][:n]
            idx_df = pd.DataFrame(idx_list, columns=col_names)
            new_df.index = pd.to_datetime(idx_df)
        else:
            new_df.index = pd.to_datetime(new_df.index)
        df = pd.concat([df, new_df]).sort_index()
        df = df[~df.index.duplicated(keep="last")].tail(days)
        tmp = NamedTemporaryFile("w", delete=False, dir=cache_dir, newline="")
        df.reset_index().to_csv(tmp.name, index=False)
        tmp.close()
        shutil.move(tmp.name, path)
    return df

