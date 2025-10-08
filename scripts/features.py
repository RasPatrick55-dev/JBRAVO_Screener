"""Feature engineering helpers for the nightly screener."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .utils.stats import robust_z


NUMERIC_COLUMNS: Sequence[str] = ("open", "high", "low", "close", "volume")

CORE_FEATURE_COLUMNS: Sequence[str] = (
    "TS",
    "MS",
    "BP",
    "PT",
    "RSI",
    "MH",
    "ADX",
    "AROON",
    "VCP",
    "VOLexp",
)

PENALTY_COLUMNS: Sequence[str] = (
    "GAPpen",
    "LIQpen",
)

INTERMEDIATE_COLUMNS: Sequence[str] = (
    "ATR14",
    "SMA9",
    "EMA20",
    "SMA50",
    "SMA100",
    "RSI14",
    "MACD_HIST",
    "AROON_UP",
    "ADX",
)

Z_SCORE_COLUMNS: Sequence[str] = tuple(f"{col}_z" for col in (*CORE_FEATURE_COLUMNS, *PENALTY_COLUMNS))


def _dedupe(seq: Sequence[str]) -> Sequence[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


REQUIRED_FEATURE_COLUMNS: Sequence[str] = _dedupe(
    (*CORE_FEATURE_COLUMNS, *PENALTY_COLUMNS, *INTERMEDIATE_COLUMNS)
)

ALL_FEATURE_COLUMNS: Sequence[str] = _dedupe(
    (*REQUIRED_FEATURE_COLUMNS, *Z_SCORE_COLUMNS)
)


def _initialised_frame(columns: Iterable[str]) -> pd.DataFrame:
    cols = ["symbol", "timestamp", *columns]
    frame = pd.DataFrame({col: pd.Series(dtype="float64") for col in cols})
    frame = frame.assign(
        symbol=pd.Series(dtype="object"),
        timestamp=pd.Series(dtype="datetime64[ns]"),
    )
    return frame[cols]


def _get_min_history(cfg: Optional[Mapping[str, object]]) -> int:
    gates = cfg.get("gates") if isinstance(cfg, Mapping) else None
    if isinstance(gates, Mapping):
        value = gates.get("min_history", 0)
    else:
        value = 0
    try:
        return int(value) if int(value) > 0 else 0
    except (TypeError, ValueError):
        return 0


def _prepare_bars_frame(bars_df: pd.DataFrame) -> pd.DataFrame:
    if bars_df is None or bars_df.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", *NUMERIC_COLUMNS])

    df = bars_df.copy()
    df["symbol"] = df["symbol"].astype("string").str.strip().str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["symbol", "timestamp"])
    df = df[df["symbol"] != ""]
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def _compute_atr(df: pd.DataFrame, grouped: pd.core.groupby.generic.DataFrameGroupBy) -> None:
    prev_close = grouped["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_prev = (df["high"] - prev_close).abs()
    low_prev = (df["low"] - prev_close).abs()
    tr = np.maximum.reduce([
        high_low.fillna(0.0).to_numpy(),
        high_prev.fillna(0.0).to_numpy(),
        low_prev.fillna(0.0).to_numpy(),
    ])
    df["TR"] = tr
    df["ATR14"] = grouped["TR"].transform(lambda s: s.rolling(14, min_periods=14).mean())


def _compute_rsi(df: pd.DataFrame, grouped: pd.core.groupby.generic.DataFrameGroupBy) -> None:
    def _rsi(series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14.0, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14.0, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(avg_loss != 0, 100.0)
        rsi = rsi.where(avg_gain != 0, 0.0)
        both_zero = (avg_gain == 0) & (avg_loss == 0)
        rsi = rsi.where(~both_zero, 50.0)
        return rsi.clip(lower=0.0, upper=100.0)

    rsi = grouped["close"].apply(_rsi)
    df["RSI14"] = rsi.reset_index(level=0, drop=True)


def _compute_adx(df: pd.DataFrame, grouped: pd.core.groupby.generic.DataFrameGroupBy) -> None:
    high = grouped["high"]
    low = grouped["low"]

    up_move = high.diff()
    down_move = low.shift(1) - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    df["plus_dm"] = plus_dm
    df["minus_dm"] = minus_dm

    plus_smoothed = grouped["plus_dm"].transform(
        lambda s: s.ewm(alpha=1 / 14.0, adjust=False, min_periods=14).mean()
    )
    minus_smoothed = grouped["minus_dm"].transform(
        lambda s: s.ewm(alpha=1 / 14.0, adjust=False, min_periods=14).mean()
    )

    atr = df["ATR14"].replace(0.0, np.nan)
    plus_di = 100 * (plus_smoothed / atr)
    minus_di = 100 * (minus_smoothed / atr)
    di_sum = plus_di + minus_di
    dx = (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)
    df["DX"] = dx * 100
    df["ADX"] = grouped["DX"].transform(
        lambda s: s.ewm(alpha=1 / 14.0, adjust=False, min_periods=14).mean()
    )


def _compute_aroon(df: pd.DataFrame, grouped: pd.core.groupby.generic.DataFrameGroupBy) -> None:
    period = 25

    def _rolling_argmax(values: np.ndarray) -> float:
        return float(np.argmax(values))

    def _rolling_argmin(values: np.ndarray) -> float:
        return float(np.argmin(values))

    high_rank = (
        grouped["high"].rolling(window=period, min_periods=period).apply(_rolling_argmax, raw=True)
    )
    low_rank = (
        grouped["low"].rolling(window=period, min_periods=period).apply(_rolling_argmin, raw=True)
    )

    high_rank = high_rank.reset_index(level=0, drop=True)
    low_rank = low_rank.reset_index(level=0, drop=True)

    periods_since_high = (period - 1) - high_rank
    periods_since_low = (period - 1) - low_rank

    aroon_up = 100 * (period - periods_since_high) / period
    aroon_down = 100 * (period - periods_since_low) / period

    df["AROON_UP"] = aroon_up
    df["AROON_DOWN"] = aroon_down
    df["AROON"] = aroon_up - aroon_down


def _compute_moving_averages(df: pd.DataFrame, grouped: pd.core.groupby.generic.DataFrameGroupBy) -> None:
    close = grouped["close"]
    df["SMA9"] = close.transform(lambda s: s.rolling(9, min_periods=9).mean())
    df["EMA20"] = close.transform(lambda s: s.ewm(span=20, adjust=False).mean())
    df["SMA50"] = close.transform(lambda s: s.rolling(50, min_periods=50).mean())
    df["SMA100"] = close.transform(lambda s: s.rolling(100, min_periods=100).mean())


def _compute_macd(df: pd.DataFrame, grouped: pd.core.groupby.generic.DataFrameGroupBy) -> None:
    def _macd_hist(series: pd.Series) -> pd.Series:
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        return macd_line - signal

    macd_hist = grouped["close"].apply(_macd_hist)
    df["MACD_HIST"] = macd_hist.reset_index(level=0, drop=True)


def _compute_volume_features(df: pd.DataFrame, grouped: pd.core.groupby.generic.DataFrameGroupBy) -> None:
    rolling_vol = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["VOLexp"] = df["volume"] / rolling_vol

    dollar_volume = df["close"] * df["volume"]
    rolling_dollar = dollar_volume.groupby(df["symbol"]).transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    liq_pen = 1.0 / rolling_dollar.replace(0.0, np.nan)
    df["LIQpen"] = pd.Series(liq_pen, index=df.index)


def _compute_price_features(df: pd.DataFrame, grouped: pd.core.groupby.generic.DataFrameGroupBy) -> None:
    close = grouped["close"]
    high = grouped["high"]

    rolling_high = high.transform(lambda s: s.rolling(20, min_periods=20).max())

    df["TS"] = np.where(df["SMA50"].abs() > 0, df["close"] / df["SMA50"] - 1, np.nan)
    df["MS"] = close.transform(lambda s: s.pct_change(periods=5))
    df["BP"] = np.where(rolling_high > 0, df["close"] / rolling_high - 1, np.nan)
    df["PT"] = np.where(df["SMA50"].abs() > 0, df["SMA9"] / df["SMA50"] - 1, np.nan)
    df["MH"] = np.where(df["SMA100"].abs() > 0, df["SMA50"] / df["SMA100"] - 1, np.nan)

    atr_pct = (df["ATR14"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    df["VCP"] = -atr_pct

    prev_close = grouped["close"].shift(1)
    gap = (df["open"] - prev_close) / prev_close
    df["GAPpen"] = gap.replace([np.inf, -np.inf], np.nan).abs().fillna(0.0)

    df["RSI"] = df["RSI14"]


def _cleanup_columns(df: pd.DataFrame) -> None:
    for column in ["TR", "DX", "plus_dm", "minus_dm"]:
        if column in df.columns:
            df.drop(columns=column, inplace=True)


def compute_all_features(bars_df: pd.DataFrame, cfg: Optional[Mapping[str, object]]) -> pd.DataFrame:
    """Compute the full feature matrix for the screener."""

    df = _prepare_bars_frame(bars_df)
    if df.empty:
        return _initialised_frame(ALL_FEATURE_COLUMNS)

    grouped = df.groupby("symbol", group_keys=False)

    _compute_moving_averages(df, grouped)
    _compute_atr(df, grouped)
    _compute_rsi(df, grouped)
    _compute_adx(df, grouped)
    _compute_aroon(df, grouped)
    _compute_macd(df, grouped)
    _compute_volume_features(df, grouped)
    _compute_price_features(df, grouped)

    _cleanup_columns(df)

    for column in (*CORE_FEATURE_COLUMNS, *PENALTY_COLUMNS):
        if column in df.columns:
            df[f"{column}_z"] = grouped[column].transform(robust_z)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    counts = grouped.cumcount() + 1
    min_history = _get_min_history(cfg)
    if min_history > 0:
        df = df[counts >= min_history]

    df = df.dropna(subset=REQUIRED_FEATURE_COLUMNS)

    result = df[["symbol", "timestamp", *ALL_FEATURE_COLUMNS]].copy()
    result.sort_values(["symbol", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


__all__ = [
    "CORE_FEATURE_COLUMNS",
    "PENALTY_COLUMNS",
    "INTERMEDIATE_COLUMNS",
    "Z_SCORE_COLUMNS",
    "REQUIRED_FEATURE_COLUMNS",
    "ALL_FEATURE_COLUMNS",
    "compute_all_features",
]
