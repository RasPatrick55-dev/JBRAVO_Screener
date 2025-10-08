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


DEFAULT_GATES: Mapping[str, float | bool] = {
    "min_history": 20,
    "min_rsi": None,
    "max_rsi": None,
    "rsi_tolerance": 0.5,
    "min_adx": None,
    "min_aroon": None,
    "min_volexp": 0.0,
    "max_gap": None,
    "max_liq_penalty": 0.00001,
    "dollar_vol_min": None,
    "require_sma_stack": True,
    "min_score": None,
    "history_column": "history",
}


FAILURE_KEYS: Tuple[str, ...] = (
    "insufficient_history",
    "failed_score",
    "nan_data",
    "failed_sma_stack",
    "failed_rsi",
    "failed_adx",
    "failed_aroon",
    "failed_volexp",
    "failed_gap",
    "failed_liquidity",
)


REJECT_SAMPLE_LIMIT = 10

GateCountValue = Union[int, str]
GateCounts = Dict[str, GateCountValue]

FLOAT_TOL = 1e-6


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


def _select_latest(bars_df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in bars_df.columns:
        return bars_df.copy()
    df = bars_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["symbol", "timestamp"])
    latest = df.groupby("symbol", as_index=False).tail(1)
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
            aroon_value = _resolve_float(row.get("AROON"))
            if (aroon_value is None or aroon_value + FLOAT_TOL < min_aroon) and "AROON_UP" in row:
                alt_value = _resolve_float(row.get("AROON_UP"))
                if alt_value is not None:
                    aroon_value = alt_value
            if aroon_value is None:
                record_failure("nan_data", "MISSING_AROON")
                continue
            if aroon_value + FLOAT_TOL < min_aroon:
                record_failure("failed_aroon", "LOW_AROON")
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

    candidates_df = df.loc[gate_rows].copy()
    candidates_df.sort_values("Score", ascending=False, inplace=True)
    candidates_df.reset_index(drop=True, inplace=True)
    gate_counts = dict(fail_counts)
    preset_key = str((cfg or {}).get("_gate_preset", "custom"))
    relax_mode = str((cfg or {}).get("_gate_relax_mode", "none"))
    gate_counts["gate_preset"] = preset_key
    gate_counts["gate_relax_mode"] = relax_mode
    evaluated = total_symbols
    passed = int(len(gate_rows))
    gate_counts["gate_total_evaluated"] = evaluated
    gate_counts["gate_total_passed"] = passed
    gate_counts["gate_total_failed"] = max(evaluated - passed, 0)
    return candidates_df, gate_counts, reject_samples


__all__ = [
    "score_universe",
    "apply_gates",
    "DEFAULT_COMPONENT_MAP",
    "DEFAULT_WEIGHTS",
    "DEFAULT_GATES",
    "compute_all_features",
]

