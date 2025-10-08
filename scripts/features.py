"""Vectorised technical feature engineering for the nightly screener."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Config & constants
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    min_history: int = 180
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    aroon_period: int = 25
    adx_period: int = 14
    sma_fast: int = 9
    ema_mid: int = 20
    sma_mid: int = 50
    sma_slow: int = 100
    breakout_lookback: int = 20
    vcp_long: int = 90
    vol_short: int = 10
    vol_long: int = 90
    robust_clip: float = 3.0  # z-score clip


DEFAULT_FEATURE_CONFIG = FeatureConfig()


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
    "H20",
    "L20",
    "ADV20",
    "+DI",
    "-DI",
    "AROON_UP",
    "AROON_DN",
    "RSI14",
    "MACD",
    "MACD_SIGNAL",
    "MACD_HIST",
    "ADX14",
    "AROON_DIFF",
)


def _dedupe(seq: Sequence[str]) -> Sequence[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


Z_SCORE_COLUMNS: Sequence[str] = tuple(f"{col}_z" for col in (*CORE_FEATURE_COLUMNS, *PENALTY_COLUMNS))

REQUIRED_FEATURE_COLUMNS: Sequence[str] = _dedupe((*CORE_FEATURE_COLUMNS, *PENALTY_COLUMNS, *INTERMEDIATE_COLUMNS))

ALL_FEATURE_COLUMNS: Sequence[str] = _dedupe((*REQUIRED_FEATURE_COLUMNS, *Z_SCORE_COLUMNS))


# --------------------------------------------------------------------------------------
# Utility / robust statistics
# --------------------------------------------------------------------------------------


def _mad(x: pd.Series, eps: float = 1e-9) -> pd.Series:
    med = x.median()
    return (x - med).abs().median() + eps


def robust_z(x: pd.Series, clip: float = 3.0) -> pd.Series:
    """
    MAD-based z-score centered at median. Clipped to [-clip, clip].
    """

    if x.empty:
        return x.copy()
    med = x.median()
    mad = _mad(x)
    if mad == 0 or not np.isfinite(mad):
        z = x - med
    else:
        z = 0.6745 * (x - med) / mad
    z = z.clip(lower=-clip, upper=clip)
    return z


# --------------------------------------------------------------------------------------
# Rolling indicator helpers (vectorized)
# NOTE: We always keep 'symbol' as a column and use groupby(..., as_index=False, group_keys=False)
# --------------------------------------------------------------------------------------


def _gb(df: pd.DataFrame):
    return df.groupby("symbol", as_index=False, group_keys=False)


def sma(df: pd.DataFrame, col: str, n: int, out: str) -> pd.DataFrame:
    df[out] = _gb(df)[col].transform(lambda s: s.rolling(n, min_periods=1).mean())
    return df


def ema(df: pd.DataFrame, col: str, n: int, out: str) -> pd.DataFrame:
    df[out] = _gb(df)[col].transform(lambda s: s.ewm(span=n, adjust=False, min_periods=1).mean())
    return df


def rsi(df: pd.DataFrame, n: int, out: str) -> pd.DataFrame:
    def _rsi(x: pd.Series) -> pd.Series:
        delta = x.diff()
        up = np.where(delta > 0, delta, 0.0)
        dn = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(up, index=x.index).ewm(alpha=1 / n, adjust=False).mean()
        roll_dn = pd.Series(dn, index=x.index).ewm(alpha=1 / n, adjust=False).mean()
        rs = roll_up / (roll_dn.replace(0, np.nan))
        return 100 - (100 / (1 + rs))

    df[out] = _gb(df)["close"].transform(_rsi)
    return df


def macd(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    signal: int,
    out_macd: str,
    out_signal: str,
    out_hist: str,
) -> pd.DataFrame:
    ema_fast = _gb(df)["close"].transform(lambda s: s.ewm(span=fast, adjust=False, min_periods=1).mean())
    ema_slow = _gb(df)["close"].transform(lambda s: s.ewm(span=slow, adjust=False, min_periods=1).mean())
    macd_line = ema_fast - ema_slow
    signal_line = _gb(pd.DataFrame({"symbol": df["symbol"], "macd": macd_line}))["macd"].transform(
        lambda s: s.ewm(span=signal, adjust=False, min_periods=1).mean()
    )
    df[out_macd] = macd_line
    df[out_signal] = signal_line
    df[out_hist] = macd_line - signal_line
    return df


def true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int, out: str) -> pd.DataFrame:
    prev_close = _gb(df)["close"].shift(1)
    tr = true_range(df["high"], df["low"], prev_close)
    df[out] = _gb(pd.DataFrame({"symbol": df["symbol"], "tr": tr}))["tr"].transform(
        lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    )
    return df


def aroon(df: pd.DataFrame, n: int, out_up: str, out_dn: str) -> pd.DataFrame:
    def _aroon_up(s: pd.Series) -> pd.Series:
        return s.rolling(n, min_periods=1).apply(
            lambda w: 100
            * (1 - (len(w) - 1 - np.argmax(w)) / (len(w) - 1 if len(w) > 1 else 1))
        )

    def _aroon_dn(s: pd.Series) -> pd.Series:
        return s.rolling(n, min_periods=1).apply(
            lambda w: 100
            * (1 - (len(w) - 1 - np.argmin(w)) / (len(w) - 1 if len(w) > 1 else 1))
        )

    df[out_up] = _gb(df)["high"].transform(_aroon_up)
    df[out_dn] = _gb(df)["low"].transform(_aroon_dn)
    return df


def dmi_adx(df: pd.DataFrame, n: int, out_adx: str, out_pdi: str, out_ndi: str) -> pd.DataFrame:
    up_move = df["high"] - _gb(df)["high"].shift(1)
    dn_move = _gb(df)["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    tr = true_range(df["high"], df["low"], _gb(df)["close"].shift(1))
    tr_ema = _gb(pd.DataFrame({"symbol": df["symbol"], "tr": tr}))["tr"].transform(
        lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    )
    pdi = 100 * _gb(pd.DataFrame({"symbol": df["symbol"], "pdm": plus_dm}))["pdm"].transform(
        lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    ) / tr_ema.replace(0, np.nan)
    ndi = 100 * _gb(pd.DataFrame({"symbol": df["symbol"], "mdm": minus_dm}))["mdm"].transform(
        lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    ) / tr_ema.replace(0, np.nan)
    dx = ((pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)) * 100
    adx = _gb(pd.DataFrame({"symbol": df["symbol"], "dx": dx}))["dx"].transform(
        lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=1).mean()
    )
    df[out_pdi] = pdi
    df[out_ndi] = ndi
    df[out_adx] = adx
    return df


def rolling_extrema(
    df: pd.DataFrame, col: str, n: int, out_max: str | None = None, out_min: str | None = None
) -> pd.DataFrame:
    if out_max:
        df[out_max] = _gb(df)[col].transform(lambda s: s.rolling(n, min_periods=1).max())
    if out_min:
        df[out_min] = _gb(df)[col].transform(lambda s: s.rolling(n, min_periods=1).min())
    return df


# --------------------------------------------------------------------------------------
# Feature computation (vectorized, flat frame)
# --------------------------------------------------------------------------------------


def compute_all_features(
    bars_df: pd.DataFrame,
    cfg: Dict | FeatureConfig | None,
    *,
    add_intermediate: bool = True,
) -> pd.DataFrame:
    """
    Compute ranking features on daily bars.

    Parameters
    ----------
    bars_df : DataFrame
        Columns: symbol, timestamp, open, high, low, close, volume (flat; 1 row per bar)
    cfg : dict | FeatureConfig | None
        Feature configuration (periods, min_history, z clipping)
    add_intermediate : bool
        If False, drop intermediate MA/ATR/etc. columns in the final output.

    Returns
    -------
    DataFrame with columns:
        symbol, timestamp,
        TS, MS, BP, PT, RSI, MH, ADX, AROON, VCP, VOLexp,
        GAPpen, LIQpen, TS_z, ..., LIQpen_z and optionally intermediate helpers.
    """

    if isinstance(cfg, dict):
        base_values = {name: getattr(DEFAULT_FEATURE_CONFIG, name) for name in FeatureConfig.__dataclass_fields__}
        for name in list(base_values):
            if name in cfg:
                base_values[name] = cfg[name]
        gates = cfg.get("gates") if isinstance(cfg.get("gates"), dict) else {}
        if "min_history" in gates:
            try:
                base_values["min_history"] = int(gates["min_history"])
            except (TypeError, ValueError):
                pass
        fc = FeatureConfig(**base_values)
    elif isinstance(cfg, FeatureConfig):
        fc = cfg
    else:
        fc = DEFAULT_FEATURE_CONFIG

    df = bars_df.copy()
    keep = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required bar columns: {', '.join(missing)}")
    df = df[keep].copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    counts = (
        df.dropna(subset=["timestamp"])
        .groupby("symbol", as_index=False)["timestamp"].size()
        .rename(columns={"size": "n"})
    )
    valid = set(counts.loc[counts["n"] >= fc.min_history, "symbol"])
    df = df[df["symbol"].isin(valid)].copy()

    if df.empty:
        columns = ["symbol", "timestamp", *ALL_FEATURE_COLUMNS]
        return pd.DataFrame(columns=columns)

    df = df.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)

    df = sma(df, "close", fc.sma_fast, "SMA9")
    df = ema(df, "close", fc.ema_mid, "EMA20")
    df = sma(df, "close", fc.sma_mid, "SMA50")
    df = sma(df, "close", fc.sma_slow, "SMA100")
    df = atr(df, fc.atr_period, "ATR14")
    df = rsi(df, fc.rsi_period, "RSI14")
    df = macd(df, fc.macd_fast, fc.macd_slow, fc.macd_signal, "MACD", "MACD_SIGNAL", "MACD_HIST")
    df = aroon(df, fc.aroon_period, "AROON_UP", "AROON_DN")
    df = dmi_adx(df, fc.adx_period, "ADX14", "+DI", "-DI")
    df = rolling_extrema(
        df,
        "close",
        fc.breakout_lookback,
        out_max="H20",
        out_min="L20",
    )

    def _linreg_slope_r2(x: pd.Series) -> float:
        y = np.log(x.replace(0, np.nan)).dropna()
        if len(y) < 5:
            return float("nan")
        t = np.arange(len(y), dtype=float)
        t_mean = t.mean()
        y_mean = y.mean()
        cov = ((t - t_mean) * (y - y_mean)).sum()
        var = ((t - t_mean) ** 2).sum()
        slope = cov / var if var != 0 else float("nan")
        ss_tot = ((y - y_mean) ** 2).sum()
        ss_res = ((y - (y_mean + slope * (t - t_mean))) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")
        return float(slope * r2 * 100.0)

    TS = _gb(df)["close"].transform(
        lambda s: s.rolling(fc.breakout_lookback, min_periods=5).apply(
            lambda w: _linreg_slope_r2(pd.Series(w)), raw=False
        )
    )
    MS = 100 * ((df["SMA9"] / df["EMA20"]) - 1.0) + 50 * ((df["EMA20"] / df["SMA50"]) - 1.0)
    BP = (df["close"] - df["H20"]) / df["ATR14"].replace(0, np.nan)
    PT = -((df["close"] - df["EMA20"]).abs() / df["ATR14"].replace(0, np.nan))
    RSI_raw = df["RSI14"]
    MH = df["MACD_HIST"] / df["ATR14"].replace(0, np.nan)
    ADX_raw = df["ADX14"]
    AROON_raw = df["AROON_UP"] - df["AROON_DN"]
    VCP = -(
        (df["ATR14"] / _gb(df)["ATR14"].transform(lambda s: s.rolling(fc.vcp_long, min_periods=1).mean()))
        - 1.0
    )
    VOLexp = (
        _gb(df)["volume"].transform(lambda s: s.rolling(fc.vol_short, min_periods=1).mean())
        / _gb(df)["volume"].transform(lambda s: s.rolling(fc.vol_long, min_periods=1).mean())
    ) - 1.0
    GAPpen = (df["open"] - _gb(df)["close"].shift(1)).abs() / df["ATR14"].replace(0, np.nan)
    adv20 = (
        _gb(df)["close"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        * _gb(df)["volume"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    )
    df["ADV20"] = adv20.astype("float64")
    LIQpen = (1_000_000 - adv20) / 1_000_000
    LIQpen = LIQpen.clip(lower=0)

    features_raw = {
        "TS": TS,
        "MS": MS,
        "BP": BP,
        "PT": PT,
        "RSI": RSI_raw,
        "MH": MH,
        "ADX": ADX_raw,
        "AROON": AROON_raw,
        "VCP": VCP,
        "VOLexp": VOLexp,
        "GAPpen": GAPpen,
        "LIQpen": LIQpen,
    }

    for name, series in features_raw.items():
        df[name] = pd.to_numeric(series, errors="coerce")

    df["AROON_DIFF"] = df["AROON"]

    z_clip = fc.robust_clip if isinstance(fc.robust_clip, (int, float)) else DEFAULT_FEATURE_CONFIG.robust_clip
    for name in (*CORE_FEATURE_COLUMNS, *PENALTY_COLUMNS):
        series = df[name]
        z = robust_z(series.dropna(), clip=z_clip)
        df[f"{name}_z"] = z.reindex(df.index).fillna(0.0)

    feature_cols = [*CORE_FEATURE_COLUMNS, *PENALTY_COLUMNS, *Z_SCORE_COLUMNS]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    float32_cols = [
        "SMA9",
        "EMA20",
        "SMA50",
        "SMA100",
        "ATR14",
        "MACD",
        "MACD_SIGNAL",
        "MACD_HIST",
        "AROON_UP",
        "AROON_DN",
        "+DI",
        "-DI",
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
    ]
    for col in float32_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    if not add_intermediate:
        cols = ["symbol", "timestamp", *feature_cols]
        return df[cols].copy()

    intermediates = [col for col in INTERMEDIATE_COLUMNS if col in df.columns]
    cols = ["symbol", "timestamp", *feature_cols, *intermediates]
    return df[cols].copy()


__all__ = [
    "FeatureConfig",
    "CORE_FEATURE_COLUMNS",
    "PENALTY_COLUMNS",
    "INTERMEDIATE_COLUMNS",
    "Z_SCORE_COLUMNS",
    "REQUIRED_FEATURE_COLUMNS",
    "ALL_FEATURE_COLUMNS",
    "robust_z",
    "compute_all_features",
]

