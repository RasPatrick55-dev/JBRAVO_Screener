"""Feature engineering helpers for the screener pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .utils.stats import robust_z, rolling_apply, rolling_max, rolling_mean, rolling_min


REQUIRED_FEATURE_COLUMNS: List[str] = [
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
    "GAPpen",
    "LIQpen",
    "ATR14",
    "SMA9",
    "EMA20",
    "SMA50",
    "SMA100",
    "H20",
    "L20",
]


def _get_min_history(cfg: Dict) -> int:
    gates = cfg.get("gates", {}) if isinstance(cfg, dict) else {}
    value = gates.get("min_history", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _initialised_frame(columns: Iterable[str]) -> pd.DataFrame:
    cols = ["symbol", "timestamp", *columns]
    frame = pd.DataFrame({col: pd.Series(dtype="float64") for col in cols})
    frame = frame.assign(
        symbol=pd.Series(dtype="object"),
        timestamp=pd.Series(dtype="datetime64[ns]"),
    )
    return frame[cols]


def compute_all_features(bars_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Compute the full feature set for the screener.

    Parameters
    ----------
    bars_df:
        Flat DataFrame containing columns ``symbol``, ``timestamp``, ``open``,
        ``high``, ``low``, ``close`` and ``volume``.
    cfg:
        Configuration mapping.  ``cfg["gates"]["min_history"]`` determines
        how many historical rows per symbol are required before emitting
        features.
    """

    if bars_df is None or bars_df.empty:
        return _initialised_frame(REQUIRED_FEATURE_COLUMNS)

    df = bars_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    grouped = df.groupby("symbol", group_keys=False)
    close_g = grouped["close"]
    high_g = grouped["high"]
    low_g = grouped["low"]
    volume_g = grouped["volume"]

    df["SMA9"] = rolling_mean(close_g, 9, min_periods=9)
    df["EMA20"] = close_g.transform(lambda s: s.ewm(span=20, adjust=False).mean())
    df["SMA50"] = rolling_mean(close_g, 50, min_periods=50)
    df["SMA100"] = rolling_mean(close_g, 100, min_periods=100)
    df["H20"] = rolling_max(high_g, 20, min_periods=20)
    df["L20"] = rolling_min(low_g, 20, min_periods=20)

    prev_close = close_g.shift(1)
    df["prev_close"] = prev_close

    tr_components = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    df["TR"] = tr_components.max(axis=1)
    df["ATR14"] = grouped["TR"].transform(lambda s: s.rolling(14, min_periods=14).mean())

    delta = close_g.diff()
    df["gain"] = delta.clip(lower=0.0)
    df["loss"] = -delta.clip(upper=0.0)
    avg_gain = grouped["gain"].transform(
        lambda s: s.ewm(alpha=1 / 14.0, adjust=False, min_periods=14).mean()
    )
    avg_loss = grouped["loss"].transform(
        lambda s: s.ewm(alpha=1 / 14.0, adjust=False, min_periods=14).mean()
    )
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    up_move = high_g.diff()
    down_move = grouped["low"].shift(1) - df["low"]
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
    atr_safe = df["ATR14"].replace(0.0, np.nan)
    plus_di = 100 * (plus_smoothed / atr_safe)
    minus_di = 100 * (minus_smoothed / atr_safe)
    di_sum = plus_di + minus_di
    dx = (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)
    df["DX"] = dx * 100
    df["ADX"] = grouped["DX"].transform(
        lambda s: s.ewm(alpha=1 / 14.0, adjust=False, min_periods=14).mean()
    )

    aroon_period = 25
    high_rank = rolling_apply(high_g, aroon_period, np.argmax, min_periods=aroon_period)
    low_rank = rolling_apply(low_g, aroon_period, np.argmin, min_periods=aroon_period)
    periods_since_high = (aroon_period - 1) - high_rank
    periods_since_low = (aroon_period - 1) - low_rank
    aroon_up = 100 * (aroon_period - periods_since_high) / aroon_period
    aroon_down = 100 * (aroon_period - periods_since_low) / aroon_period
    df["AROON"] = aroon_up - aroon_down

    df["momentum"] = close_g.pct_change(periods=5)
    df["trend_ratio"] = np.where(df["SMA50"].abs() > 0, df["close"] / df["SMA50"] - 1, np.nan)
    df["atr_ratio"] = df["ATR14"] / df["close"].replace(0.0, np.nan)

    df["TS"] = grouped["trend_ratio"].transform(robust_z)
    df["MS"] = grouped["momentum"].transform(robust_z)
    df["VCP"] = grouped["atr_ratio"].transform(robust_z)

    rolling_vol = volume_g.transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["VOLexp"] = df["volume"] / rolling_vol

    df["dollar_volume"] = df["close"] * df["volume"]
    rolling_dollar = grouped["dollar_volume"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    df["LIQpen"] = np.where(rolling_dollar > 0, 1.0 / rolling_dollar, np.nan)

    gap = (df["open"] - prev_close) / prev_close
    df["GAPpen"] = gap.replace([np.inf, -np.inf], np.nan).abs().fillna(0.0)

    df["BP"] = np.where(df["H20"] > 0, df["close"] / df["H20"] - 1, np.nan)
    df["PT"] = np.where(df["SMA50"] > 0, df["SMA9"] / df["SMA50"] - 1, np.nan)
    df["MH"] = np.where(df["SMA100"] > 0, df["SMA50"] / df["SMA100"] - 1, np.nan)

    counts = grouped.cumcount() + 1
    min_history = _get_min_history(cfg or {})
    if min_history > 0:
        df = df[counts >= min_history]

    result = df.dropna(subset=REQUIRED_FEATURE_COLUMNS).copy()
    keep_columns = ["symbol", "timestamp", *REQUIRED_FEATURE_COLUMNS]
    result = result[keep_columns]
    result.reset_index(drop=True, inplace=True)

    return result

