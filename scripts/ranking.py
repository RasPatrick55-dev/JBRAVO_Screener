"""Ranking helpers for the screener pipeline.

This module encapsulates the scoring logic for the nightly screener along with
gate checks that enforce the minimum technical requirements for a candidate.

The public surface intentionally mirrors the previous inline implementation in
``scripts.screener`` so it can be imported without pulling in the heavy
dependencies from the rest of that module.
"""

from __future__ import annotations

import json
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .features import compute_all_features


DEFAULT_COMPONENT_MAP: Mapping[str, str] = {
    "trend": "TS",
    "momentum": "MS",
    "breakout": "BP",
    "pivot_trend": "PT",
    "multi_horizon": "MH",
    "rsi": "RSI",
    "adx": "ADX",
    "aroon": "AROON",
    "vol_contraction": "VCP",
    "volume_expansion": "VOLexp",
    "gap_penalty": "GAPpen",
    "liquidity_penalty": "LIQpen",
}


DEFAULT_WEIGHTS: Mapping[str, float] = {
    "trend": 0.25,
    "momentum": 0.2,
    "breakout": 0.12,
    "pivot_trend": 0.1,
    "multi_horizon": 0.08,
    "rsi": 0.07,
    "adx": 0.06,
    "aroon": 0.04,
    "vol_contraction": 0.04,
    "volume_expansion": 0.03,
    "gap_penalty": -0.04,
    "liquidity_penalty": -0.05,
}


DEFAULT_CATEGORY_WEIGHTS: Mapping[str, float] = {
    "trend": 0.30,
    "momentum": 0.25,
    "volume": 0.20,
    "volatility": 0.15,
    "risk": 0.10,
}


DEFAULT_GATES: Mapping[str, float | bool] = {
    "min_history": 20,
    "min_rsi": None,
    "max_rsi": None,
    "rsi_tolerance": 0.5,
    "min_adx": None,
    "min_aroon": None,
    "min_macd_hist": None,
    "min_volexp": 0.0,
    "max_gap": None,
    "max_liq_penalty": 0.00001,
    "dollar_vol_min": None,
    "require_sma_stack": True,
    "min_score": None,
    "history_column": "history",
}


PRESETS: Mapping[str, Dict[str, float | int]] = {
    "strict": {
        "rsi_min": 55,
        "rsi_max": 65,
        "adx_min": 25,
        "aroon_up_min": 70,
        "macd_hist_min": 0.10,
        "cross_lookback": 2,
    },
    "standard": {
        "rsi_min": 52,
        "rsi_max": 68,
        "adx_min": 20,
        "aroon_up_min": 60,
        "macd_hist_min": 0.00,
        "cross_lookback": 3,
    },
    "mild": {
        "rsi_min": 50,
        "rsi_max": 70,
        "adx_min": 15,
        "aroon_up_min": 50,
        "macd_hist_min": -0.05,
        "cross_lookback": 5,
    },
}


FAILURE_KEYS: Tuple[str, ...] = (
    "insufficient_history",
    "failed_score",
    "nan_data",
    "failed_sma_stack",
    "failed_rsi",
    "failed_adx",
    "failed_aroon",
    "failed_macd",
    "failed_volexp",
    "failed_gap",
    "failed_liquidity",
)


REJECT_SAMPLE_LIMIT = 10

GateCountValue = Union[int, str]
GateCounts = Dict[str, GateCountValue]

FLOAT_TOL = 1e-6


def _ensure_series(
    df: pd.DataFrame, column: str, default: float | int | None = None
) -> pd.Series:
    base = pd.Series(default, index=df.index)
    if column in df.columns:
        base = df[column]
    return pd.to_numeric(base, errors="coerce")


def apply_final_gates(df: pd.DataFrame, preset: str = "standard") -> pd.DataFrame:
    """Apply preset indicator thresholds for the final gate filter."""

    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    preset_key = str(preset or "standard").strip().lower()
    preset_cfg = PRESETS.get(preset_key, PRESETS["standard"])

    rsi14 = _ensure_series(df, "RSI14")
    adx = _ensure_series(df, "ADX")
    aroon_up = _ensure_series(df, "AROON_UP")
    if "AROON_UP" not in df.columns:
        aroon_up = _ensure_series(df, "AROON")
    macd_hist = _ensure_series(df, "MACD_HIST")
    sma9 = _ensure_series(df, "SMA9")
    ema20 = _ensure_series(df, "EMA20")
    sma50 = _ensure_series(df, "SMA50")
    sma100 = _ensure_series(df, "SMA100")
    recent_cross = _ensure_series(df, "recent_cross_bars", default=np.inf).fillna(np.inf)

    rsi_mask = rsi14.between(preset_cfg["rsi_min"], preset_cfg["rsi_max"], inclusive="both")
    adx_mask = adx >= float(preset_cfg["adx_min"])
    aroon_mask = aroon_up >= float(preset_cfg["aroon_up_min"])
    macd_mask = macd_hist >= float(preset_cfg["macd_hist_min"])
    cross_mask = (sma9 > ema20) | (recent_cross <= float(preset_cfg["cross_lookback"]))
    stack_mask = cross_mask & (ema20 > sma50) & (sma50 >= sma100)

    pass_mask = rsi_mask & adx_mask & aroon_mask & macd_mask & stack_mask
    pass_mask = pass_mask.fillna(False)
    return df.loc[pass_mask].copy()


def _standardize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mask = values.notna()
    if mask.sum() == 0:
        return pd.Series(0.0, index=series.index)
    std = values[mask].std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    mean = values[mask].mean()
    standardized = (values - mean) / std
    standardized[~mask] = 0.0
    return standardized


def _normalise_config(mapping: Optional[Mapping[str, float]]) -> Dict[str, float]:
    base = dict(DEFAULT_WEIGHTS)
    if not mapping:
        return base
    for key, value in mapping.items():
        try:
            base[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return base


def _normalise_category_weights(mapping: Optional[Mapping[str, float]]) -> Dict[str, float]:
    base = dict(DEFAULT_CATEGORY_WEIGHTS)
    if not mapping:
        return base
    for key, value in mapping.items():
        try:
            base[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return base


def _config_version(cfg: Optional[Mapping[str, object]]) -> int:
    if not cfg:
        return 1
    version = cfg.get("version")
    if version is None:
        return 1
    if isinstance(version, (int, float, np.integer, np.floating)):
        return 2 if float(version) >= 2 else 1
    version_str = str(version).strip().lower()
    try:
        numeric = float(version_str)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None:
        return 2 if numeric >= 2 else 1
    if version_str.startswith("v2") or version_str.startswith("2"):
        return 2
    return 1


def _select_latest(bars_df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in bars_df.columns:
        return bars_df.copy()
    df = bars_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["symbol", "timestamp"])
    latest = df.groupby("symbol", as_index=False).tail(1)
    latest.reset_index(drop=True, inplace=True)
    return latest


def _score_universe_v2(bars_df: pd.DataFrame, cfg: Mapping[str, object]) -> pd.DataFrame:
    if bars_df is None or bars_df.empty:
        empty = pd.DataFrame(columns=["symbol", "timestamp", "Score", "score_breakdown"])
        return empty

    working = bars_df.copy()
    working["timestamp"] = pd.to_datetime(working.get("timestamp"), utc=True, errors="coerce")
    working = working.dropna(subset=["symbol", "timestamp"]).copy()
    if working.empty:
        empty = pd.DataFrame(columns=["symbol", "timestamp", "Score", "score_breakdown"])
        return empty

    working["symbol"] = working["symbol"].astype("string").str.upper()
    working.sort_values(["symbol", "timestamp"], inplace=True)

    required_numeric = [
        "SMA9",
        "EMA20",
        "SMA180",
        "WK52_PROX",
        "RSI14",
        "RSI",
        "MACD",
        "MACD_HIST",
        "ADX",
        "AROON_UP",
        "AROON_DN",
        "REL_VOLUME",
        "OBV_DELTA",
        "BB_BANDWIDTH",
        "ATR_pct",
    ]

    for column in required_numeric:
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")

    grouped = working.groupby("symbol", group_keys=False)
    working["prev_MACD_HIST"] = grouped["MACD_HIST"].shift(1)
    working["prev_AROON_UP"] = grouped["AROON_UP"].shift(1)
    working["prev_AROON_DN"] = grouped["AROON_DN"].shift(1)

    latest_idx = grouped["timestamp"].idxmax()
    latest = working.loc[latest_idx].copy()
    latest.reset_index(drop=True, inplace=True)

    thresholds_cfg = cfg.get("thresholds") if isinstance(cfg.get("thresholds"), Mapping) else {}

    rel_vol_min = _resolve_float(thresholds_cfg.get("rel_vol_min")) if thresholds_cfg else None
    if rel_vol_min is None:
        rel_vol_min = 1.5
    adx_min = _resolve_float(thresholds_cfg.get("adx_min")) if thresholds_cfg else None
    if adx_min is None:
        adx_min = 20.0
    atr_pct_max = _resolve_float(thresholds_cfg.get("atr_pct_max")) if thresholds_cfg else None
    bb_pctile = thresholds_cfg.get("bb_bw_pctile") if thresholds_cfg else None
    try:
        bb_pctile = float(bb_pctile)
    except (TypeError, ValueError):
        bb_pctile = None
    if bb_pctile is None or not (0 < bb_pctile < 1):
        bb_pctile = 0.15

    def _num(name: str) -> pd.Series:
        if name in latest.columns:
            return pd.to_numeric(latest[name], errors="coerce")
        return pd.Series(np.nan, index=latest.index)

    sma9 = _num("SMA9")
    ema20 = _num("EMA20")
    sma180 = _num("SMA180")
    wk52 = _num("WK52_PROX")
    rsi = _num("RSI") if "RSI" in latest.columns else _num("RSI14")
    macd = _num("MACD")
    macd_hist = _num("MACD_HIST")
    macd_hist_prev = _num("prev_MACD_HIST")
    rel_vol = _num("REL_VOLUME")
    obv_delta = _num("OBV_DELTA")
    bb_band = _num("BB_BANDWIDTH")
    atr_pct = _num("ATR_pct")
    aroon_up = _num("AROON_UP")
    aroon_dn = _num("AROON_DN")
    aroon_up_prev = _num("prev_AROON_UP")
    aroon_dn_prev = _num("prev_AROON_DN")
    adx = _num("ADX")

    trend_ma_align = pd.Series(
        np.where((sma9 > ema20) & (ema20 > sma180), 1.0, 0.0), index=latest.index
    )
    trend_52w = pd.Series(np.where(wk52 >= 0.90, 0.5, 0.0), index=latest.index)
    mom_rsi = pd.Series(np.where(rsi > 50, 0.5, 0.0), index=latest.index)
    mom_macd_pos = pd.Series(np.where(macd > 0, 0.5, 0.0), index=latest.index)
    mom_macd_hist_rising = pd.Series(
        np.where(macd_hist > macd_hist_prev, 0.25, 0.0), index=latest.index
    )
    vol_rel = pd.Series(np.where(rel_vol >= rel_vol_min, 0.5, 0.0), index=latest.index)
    vol_obv_up = pd.Series(np.where(obv_delta > 0, 0.5, 0.0), index=latest.index)

    squeeze_threshold = np.nan
    if bb_band.notna().any():
        try:
            squeeze_threshold = float(bb_band.dropna().quantile(bb_pctile))
        except ValueError:
            squeeze_threshold = np.nan
    vol_bb_squeeze = pd.Series(
        np.where(bb_band <= squeeze_threshold, 0.5, 0.0), index=latest.index
    )
    vol_bb_squeeze = vol_bb_squeeze.where(np.isfinite(squeeze_threshold), 0.0)

    risk_atr_penalty = pd.Series(0.0, index=latest.index)
    if atr_pct_max is not None:
        risk_atr_penalty = pd.Series(
            np.where(atr_pct > atr_pct_max, -0.5, 0.0), index=latest.index
        )

    aroon_cross = pd.Series(
        np.where(
            (aroon_up > aroon_dn) & (aroon_up_prev <= aroon_dn_prev),
            0.5,
            0.0,
        ),
        index=latest.index,
    )
    adx_trend = pd.Series(np.where(adx >= adx_min, 0.5, 0.0), index=latest.index)

    components = pd.DataFrame(
        {
            "trend_ma_align": trend_ma_align,
            "trend_52w": trend_52w,
            "mom_rsi": mom_rsi,
            "mom_macd_pos": mom_macd_pos,
            "mom_macd_hist_rising": mom_macd_hist_rising,
            "vol_rel": vol_rel,
            "vol_obv_up": vol_obv_up,
            "vol_bb_squeeze": vol_bb_squeeze,
            "risk_atr_penalty": risk_atr_penalty,
            "aroon_cross": aroon_cross,
            "adx_trend": adx_trend,
        }
    ).fillna(0.0)

    category_map = {
        "trend": ["trend_ma_align", "trend_52w", "aroon_cross", "adx_trend"],
        "momentum": ["mom_rsi", "mom_macd_pos", "mom_macd_hist_rising"],
        "volume": ["vol_rel", "vol_obv_up"],
        "volatility": ["vol_bb_squeeze"],
        "risk": ["risk_atr_penalty"],
    }

    category_scores: dict[str, pd.Series] = {}
    for name, cols in category_map.items():
        subset = components[cols] if cols else pd.DataFrame(index=latest.index)
        category_scores[name] = subset.sum(axis=1).fillna(0.0)

    weights = _normalise_category_weights(cfg.get("weights") if isinstance(cfg, Mapping) else None)
    contributions = {
        name: category_scores[name] * weights.get(name, 0.0)
        for name in category_scores
    }
    contributions_df = pd.DataFrame(contributions).fillna(0.0)
    score_series = contributions_df.sum(axis=1).fillna(0.0)

    latest["Score"] = score_series.round(4)
    for name, series in category_scores.items():
        latest[f"{name}_score"] = series.round(4)
    for name, series in contributions.items():
        latest[f"{name}_contribution"] = series.round(4)

    latest["score_breakdown"] = contributions_df.apply(
        lambda row: json.dumps({k: round(float(v), 4) for k, v in row.items()}), axis=1
    )
    latest["component_breakdown"] = components.apply(
        lambda row: json.dumps({k: round(float(v), 4) for k, v in row.items()}), axis=1
    )

    latest.sort_values("Score", ascending=False, inplace=True)
    latest.reset_index(drop=True, inplace=True)
    return latest


def score_universe(bars_df: pd.DataFrame, cfg: Optional[Mapping[str, object]] = None) -> pd.DataFrame:
    """Score every symbol in ``bars_df`` and return a ranked DataFrame.

    Parameters
    ----------
    bars_df:
        DataFrame containing the engineered features used for scoring.  When
        multiple rows per symbol are present the most recent row (by
        ``timestamp``) is used.
    cfg:
        Optional configuration mapping.  Supported keys:
            ``weights`` - overrides for component weights.
            ``components`` - mapping from component names to column names.

    Returns
    -------
    pd.DataFrame
        The input rows collapsed to one per symbol with ``Score`` and
        ``score_breakdown`` columns added.  Rows are sorted in descending score
        order.
    """

    cfg = cfg or {}
    if _config_version(cfg) >= 2:
        return _score_universe_v2(bars_df, cfg)

    latest = _select_latest(bars_df)
    if latest.empty:
        empty = latest.copy()
        empty["Score"] = pd.Series(dtype="float64")
        empty["score_breakdown"] = pd.Series(dtype="object")
        return empty

    components_map = dict(DEFAULT_COMPONENT_MAP)
    for key, value in (cfg.get("components") or {}).items():
        if not value:
            continue
        components_map[str(key)] = str(value)

    weights = _normalise_config(cfg.get("weights"))

    standardized_columns: Dict[str, pd.Series] = {}
    for comp, column in components_map.items():
        if column not in latest.columns:
            standardized_columns[comp] = pd.Series(0.0, index=latest.index)
            continue
        standardized_columns[comp] = _standardize(latest[column])

    standardized_df = pd.DataFrame(standardized_columns)

    aligned_weights = pd.Series({key: weights.get(key, 0.0) for key in standardized_df.columns})
    standardized_df = standardized_df.reindex(columns=aligned_weights.index, fill_value=0.0)

    contributions = standardized_df.multiply(aligned_weights, axis=1)
    if contributions.isna().any().any():
        contributions = contributions.fillna(0.0)
    score_series = contributions.sum(axis=1).fillna(0.0)

    breakdown_strings = standardized_df.apply(
        lambda row: json.dumps({k: round(float(v), 4) for k, v in sorted(row.items())}),
        axis=1,
    )

    result = latest.copy()
    result["Score"] = score_series.round(4)
    result["score_breakdown"] = breakdown_strings
    result.sort_values("Score", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _initialise_fail_counts() -> Dict[str, int]:
    return {key: 0 for key in FAILURE_KEYS}


def _resolve_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        if np.isnan(value):
            return None
        return float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed):
        return None
    return parsed


def apply_gates(
    df: pd.DataFrame,
    cfg: Optional[Mapping[str, object]] = None,
) -> Tuple[pd.DataFrame, GateCounts, List[Dict[str, str]]]:
    """Filter ``df`` according to the configured gate thresholds."""

    cfg = cfg or {}
    gates_cfg = dict(DEFAULT_GATES)
    gates_cfg.update({key: cfg.get("gates", {}).get(key, gates_cfg.get(key)) for key in gates_cfg})

    fail_counts = _initialise_fail_counts()
    if df is None or df.empty:
        gate_counts: GateCounts = dict(fail_counts)
        gate_counts["gate_preset"] = str((cfg or {}).get("_gate_preset", "custom"))
        gate_counts["gate_relax_mode"] = str((cfg or {}).get("_gate_relax_mode", "none"))
        gate_counts["gate_total_evaluated"] = 0
        gate_counts["gate_total_passed"] = 0
        gate_counts["gate_total_failed"] = 0
        return (
            df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(),
            gate_counts,
            [],
        )

    df = df.copy()
    gate_rows = []
    reject_samples: List[Dict[str, str]] = []
    history_col = str(gates_cfg.get("history_column") or "history")
    min_history = _resolve_float(gates_cfg.get("min_history")) or 0.0
    require_sma_stack = bool(gates_cfg.get("require_sma_stack", True))
    min_rsi = _resolve_float(gates_cfg.get("min_rsi"))
    max_rsi = _resolve_float(gates_cfg.get("max_rsi"))
    rsi_tolerance = _resolve_float(gates_cfg.get("rsi_tolerance"))
    if rsi_tolerance is None or not np.isfinite(rsi_tolerance):
        rsi_tolerance = 0.0
    else:
        rsi_tolerance = max(rsi_tolerance, 0.0)
    min_adx = _resolve_float(gates_cfg.get("min_adx"))
    min_aroon = _resolve_float(gates_cfg.get("min_aroon"))
    min_macd_hist = _resolve_float(gates_cfg.get("min_macd_hist"))
    min_volexp = _resolve_float(gates_cfg.get("min_volexp"))
    max_gap = _resolve_float(gates_cfg.get("max_gap"))
    max_liq_pen = _resolve_float(gates_cfg.get("max_liq_penalty"))
    dollar_vol_min = _resolve_float(gates_cfg.get("dollar_vol_min"))
    min_score = _resolve_float(gates_cfg.get("min_score"))

    total_symbols = int(df.shape[0])
    if dollar_vol_min is not None and dollar_vol_min > 0 and "ADV20" in df.columns:
        latest = df.copy()
        if "timestamp" in latest.columns:
            latest["timestamp"] = pd.to_datetime(latest["timestamp"], utc=True, errors="coerce")
            latest = latest.sort_values(["symbol", "timestamp"])
        if "symbol" in latest.columns:
            latest = latest.groupby("symbol", as_index=False, group_keys=False).tail(1)
        liquid_mask = pd.to_numeric(latest["ADV20"], errors="coerce") >= (dollar_vol_min - FLOAT_TOL)
        latest_liquid = latest.loc[liquid_mask.fillna(False)]
        liquid_symbols = set(latest_liquid["symbol"].astype(str).str.upper())
        failed_liquidity = max(total_symbols - len(liquid_symbols), 0)
        if failed_liquidity:
            fail_counts["failed_liquidity"] += failed_liquidity
            if "symbol" in latest.columns:
                rejected_symbols = [
                    str(sym).strip().upper()
                    for sym in latest["symbol"].astype(str)
                    if str(sym).strip().upper() not in liquid_symbols
                ]
                for sym in rejected_symbols:
                    if len(reject_samples) >= REJECT_SAMPLE_LIMIT:
                        break
                    reject_samples.append({"symbol": sym, "reason": "LOW_LIQUIDITY"})
        if liquid_symbols:
            df = df[df["symbol"].astype(str).str.upper().isin(liquid_symbols)].copy()
        else:
            df = df.iloc[0:0].copy()
    else:
        liquid_symbols = set(df.get("symbol", pd.Series(dtype="object")).astype(str).str.upper())

    for idx, row in df.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper() or "<UNKNOWN>"

        def record_failure(reason_key: str, reason_label: str) -> None:
            fail_counts[reason_key] += 1
            if len(reject_samples) < REJECT_SAMPLE_LIMIT:
                reject_samples.append({"symbol": symbol, "reason": reason_label})

        history_value = _resolve_float(row.get(history_col))
        if min_history and (history_value is None or history_value < min_history):
            record_failure("insufficient_history", "INSUFFICIENT_HISTORY")
            continue

        if min_score is not None:
            score_value = _resolve_float(row.get("Score"))
            if score_value is None:
                record_failure("nan_data", "MISSING_SCORE")
                continue
            if score_value + FLOAT_TOL < min_score:
                record_failure("failed_score", "LOW_SCORE")
                continue

        if require_sma_stack:
            sma9 = _resolve_float(row.get("SMA9"))
            ema20 = _resolve_float(row.get("EMA20"))
            sma50 = _resolve_float(row.get("SMA50"))
            sma100 = _resolve_float(row.get("SMA100"))
            values = [sma9, ema20, sma50, sma100]
            if any(v is None for v in values):
                record_failure("nan_data", "MISSING_MA")
                continue
            if not (sma9 > ema20 > sma50 > sma100):
                record_failure("failed_sma_stack", "FAILED_SMA_STACK")
                continue

        if min_rsi is not None or max_rsi is not None:
            rsi_value = _resolve_float(row.get("RSI"))
            if rsi_value is None:
                record_failure("nan_data", "MISSING_RSI")
                continue
            if min_rsi is not None and rsi_value + FLOAT_TOL < (min_rsi - rsi_tolerance):
                record_failure("failed_rsi", "LOW_RSI")
                continue
            if max_rsi is not None and rsi_value - FLOAT_TOL > (max_rsi + rsi_tolerance):
                record_failure("failed_rsi", "HIGH_RSI")
                continue

        if min_adx is not None:
            adx_value = _resolve_float(row.get("ADX"))
            if adx_value is None:
                record_failure("nan_data", "MISSING_ADX")
                continue
            if adx_value + FLOAT_TOL < min_adx:
                record_failure("failed_adx", "LOW_ADX")
                continue

        if min_aroon is not None:
            aroon_value = _resolve_float(row.get("AROON_UP"))
            if aroon_value is None:
                aroon_value = _resolve_float(row.get("AROON"))
            if aroon_value is None:
                record_failure("nan_data", "MISSING_AROON")
                continue
            if aroon_value + FLOAT_TOL < min_aroon:
                record_failure("failed_aroon", "LOW_AROON")
                continue

        if min_macd_hist is not None:
            macd_hist_value = _resolve_float(row.get("MACD_HIST"))
            if macd_hist_value is None:
                record_failure("nan_data", "MISSING_MACD")
                continue
            if macd_hist_value + FLOAT_TOL < min_macd_hist:
                record_failure("failed_macd", "LOW_MACD_HIST")
                continue

        if min_volexp is not None:
            volexp_value = _resolve_float(row.get("VOLexp"))
            if volexp_value is None:
                record_failure("nan_data", "MISSING_VOLEXP")
                continue
            if volexp_value + FLOAT_TOL < min_volexp:
                record_failure("failed_volexp", "LOW_VOLEXP")
                continue

        if max_gap is not None:
            gap_value = _resolve_float(row.get("GAPpen"))
            if gap_value is None:
                record_failure("nan_data", "MISSING_GAP")
                continue
            if gap_value - FLOAT_TOL > max_gap:
                record_failure("failed_gap", "HIGH_GAP")
                continue

        if max_liq_pen is not None:
            liq_value = _resolve_float(row.get("LIQpen"))
            if liq_value is None:
                record_failure("nan_data", "MISSING_LIQUIDITY")
                continue
            if liq_value - FLOAT_TOL > max_liq_pen:
                record_failure("failed_liquidity", "LOW_LIQUIDITY")
                continue

        gate_rows.append(idx)

    preset_key = str((cfg or {}).get("_gate_preset", "custom")).strip().lower()
    final_preset = preset_key if preset_key in PRESETS else None

    candidates_df = df.loc[gate_rows].copy()
    if final_preset and not candidates_df.empty:
        preset_cfg = PRESETS[final_preset]
        filtered_df = apply_final_gates(candidates_df, final_preset)
        if filtered_df.shape[0] != candidates_df.shape[0]:
            removed_df = candidates_df.loc[~candidates_df.index.isin(filtered_df.index)]
            for _, removed in removed_df.iterrows():
                symbol = str(removed.get("symbol", "")).strip().upper() or "<UNKNOWN>"

                def record_reason(reason_key: str, reason_label: str) -> None:
                    fail_counts[reason_key] += 1
                    if len(reject_samples) < REJECT_SAMPLE_LIMIT:
                        reject_samples.append({"symbol": symbol, "reason": reason_label})

                rsi_value = _resolve_float(removed.get("RSI14"))
                if rsi_value is None:
                    record_reason("nan_data", "MISSING_RSI14")
                    continue
                if rsi_value + FLOAT_TOL < float(preset_cfg["rsi_min"]):
                    record_reason("failed_rsi", "LOW_RSI14")
                    continue
                if rsi_value - FLOAT_TOL > float(preset_cfg["rsi_max"]):
                    record_reason("failed_rsi", "HIGH_RSI14")
                    continue

                adx_value = _resolve_float(removed.get("ADX"))
                if adx_value is None:
                    record_reason("nan_data", "MISSING_ADX")
                    continue
                if adx_value + FLOAT_TOL < float(preset_cfg["adx_min"]):
                    record_reason("failed_adx", "LOW_ADX_RAW")
                    continue

                aroon_value = _resolve_float(removed.get("AROON_UP"))
                if aroon_value is None:
                    aroon_value = _resolve_float(removed.get("AROON"))
                if aroon_value is None:
                    record_reason("nan_data", "MISSING_AROON_UP")
                    continue
                if aroon_value + FLOAT_TOL < float(preset_cfg["aroon_up_min"]):
                    record_reason("failed_aroon", "LOW_AROON_UP")
                    continue

                macd_value = _resolve_float(removed.get("MACD_HIST"))
                if macd_value is None:
                    record_reason("nan_data", "MISSING_MACD_HIST")
                    continue
                if macd_value + FLOAT_TOL < float(preset_cfg["macd_hist_min"]):
                    record_reason("failed_macd", "LOW_MACD_HIST_RAW")
                    continue

                sma9_value = _resolve_float(removed.get("SMA9"))
                ema20_value = _resolve_float(removed.get("EMA20"))
                sma50_value = _resolve_float(removed.get("SMA50"))
                sma100_value = _resolve_float(removed.get("SMA100"))
                recent_cross = _resolve_float(removed.get("recent_cross_bars"))
                if any(v is None for v in (sma9_value, ema20_value, sma50_value, sma100_value)):
                    record_reason("nan_data", "MISSING_SMA_STACK")
                    continue
                cross_ok = False
                if sma9_value is not None and ema20_value is not None and sma9_value > ema20_value:
                    cross_ok = True
                if (
                    recent_cross is not None
                    and recent_cross <= float(preset_cfg["cross_lookback"]) + FLOAT_TOL
                ):
                    cross_ok = True
                if not cross_ok:
                    record_reason("failed_sma_stack", "FAILED_SMA_OR_CROSS")
                    continue
                if ema20_value is None or sma50_value is None or ema20_value <= sma50_value:
                    record_reason("failed_sma_stack", "FAILED_EMA20_GT_SMA50")
                    continue
                if sma50_value is None or sma100_value is None or sma50_value + FLOAT_TOL < sma100_value:
                    record_reason("failed_sma_stack", "FAILED_SMA50_GTE_SMA100")
                    continue
            candidates_df = filtered_df

    candidates_df.sort_values("Score", ascending=False, inplace=True)
    candidates_df.reset_index(drop=True, inplace=True)
    gate_counts = dict(fail_counts)
    relax_mode = str((cfg or {}).get("_gate_relax_mode", "none"))
    gate_counts["gate_preset"] = preset_key
    gate_counts["gate_relax_mode"] = relax_mode
    evaluated = total_symbols
    passed = int(candidates_df.shape[0])
    gate_counts["gate_total_evaluated"] = evaluated
    gate_counts["gate_total_passed"] = passed
    gate_counts["gate_total_failed"] = max(evaluated - passed, 0)
    return candidates_df, gate_counts, reject_samples


__all__ = [
    "score_universe",
    "apply_gates",
    "apply_final_gates",
    "PRESETS",
    "DEFAULT_COMPONENT_MAP",
    "DEFAULT_WEIGHTS",
    "DEFAULT_GATES",
    "compute_all_features",
]

