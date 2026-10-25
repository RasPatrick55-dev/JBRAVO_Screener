"""Versioned research predicates using the existing simulator indicators/scoring.

The baseline is the legacy portfolio-simulator signal, not screener v2 parity.
The challenger preserves R02 qualifying-close consumption and strict rearming.
No strategy here is approved for automatic promotion or execution.
"""
from __future__ import annotations

import math
from statistics import fmean

import numpy as np

from scripts.backtest_core import composite_score
from scripts.indicators import compute_indicators

BASELINE = "legacy_simulator_core_v1"
CHALLENGER = "trend_pullback_reclaim_qualifying_close_v2"
BENCHMARK = "spy_buy_and_hold_v1"
METHODS = (BASELINE, CHALLENGER)
DEPENDENCIES = ("ma50", "ma200", "rsi", "macd", "macd_signal", "macd_hist", "adx",
                "aroon_up", "aroon_down", "obv", "vol_avg30", "month_high", "ema20", "ATR14")
R02_CONSUMER_SHA256 = "357bed9151f3dfeb110021e3a3c666390150d790ced2c25f4a4a188324ea1e06"


def ready(features, i, warmup):
    return i + 1 >= warmup and i > 0 and all(
        features[j].get(k) is not None and math.isfinite(features[j][k])
        for j in (i - 1, i) for k in DEPENDENCIES
    )


def reclaim(features, warmup, last_reclaim=None):
    """Literal R02 pullback/reclaim predicate; no historical consumer relabelling."""
    t = len(features) - 1
    if not ready(features, t, warmup):
        return {"signal": False, "failures": ["READINESS"], "duration": None}
    c, failures = features[t], []
    if not c["close"] > c["ma50"] > c["ma200"]:
        failures.append("TREND")
    if t < 10 or not c["ma50"] > features[t - 10]["ma50"]:
        failures.append("SLOPE")
    chosen, duration_failures = None, {}
    for d in (2, 3, 4, 5):
        p, issues = t - d, []
        if p < 1 or not all(ready(features, k, warmup) for k in range(p - 1, t)):
            issues.append("WINDOW_READINESS")
        else:
            if not features[p - 1]["close"] > features[p - 1]["ema20"]:
                issues.append("START")
            for k in range(p, t):
                f = features[k]
                if not f["ATR14"] > 0:
                    issues.append("ATR")
                if not f["close"] >= f["ma50"]:
                    issues.append("FLOOR")
                if not f["close"] <= f["ema20"] + 0.5 * f["ATR14"]:
                    issues.append("UPPER_BAND")
                if not f["low"] >= f["ema20"] - 1.5 * f["ATR14"]:
                    issues.append("DEPTH")
            if not any(features[k]["close"] <= features[k]["ema20"] for k in range(p, t)):
                issues.append("NO_EMA_TOUCH")
            if last_reclaim is not None and features[p]["session"] <= last_reclaim:
                issues.append("REARM")
        duration_failures[str(d)] = issues
        if not issues and chosen is None:
            chosen = d, p
    if chosen is None:
        failures.append("NO_PULLBACK")
    if not c["ATR14"] > 0:
        failures.append("ATR")
    if not c["close"] > c["ema20"]:
        failures.append("EMA_RECLAIM")
    if not c["close"] > features[t - 1]["high"]:
        failures.append("HIGH_RECLAIM")
    if not c["close"] > c["open"]:
        failures.append("BULLISH")
    prior_volume = fmean(x["volume"] for x in features[t - 20:t])
    if prior_volume <= 0 or not c["volume"] > prior_volume:
        failures.append("PRIOR_VOLUME")
    if chosen and c["low"] < min(x["low"] for x in features[chosen[1]:t]):
        failures.append("LOWER_TROUGH")
    return {"signal": not failures, "failures": failures, "duration": chosen[0] if chosen else None,
            "pullback_start": features[chosen[1]]["session"] if chosen else None,
            "reclaim_session": c["session"], "prior_volume_mean": prior_volume,
            "duration_predicates": duration_failures}


def prepare_signals(raw, split, method, fold, warmup=250):
    if method not in METHODS + (BENCHMARK,):
        raise ValueError("Unknown versioned research strategy")
    if warmup != 250:
        raise ValueError("v1 strategy definitions require 250-session warmup")
    mask = split.index.to_series().between(fold["warmup_start"], fold["test_end"])
    source = split.loc[mask, ["open", "high", "low", "close", "volume"]].copy()
    featured = compute_indicators(source)
    featured["score"] = composite_score(featured)
    featured["session"] = featured.index.strftime("%Y-%m-%d")
    records = featured.to_dict("records")
    readiness = [ready(records, i, warmup) for i in range(len(records))]
    featured["entry_ready"] = readiness
    featured.loc[~featured["entry_ready"], "score"] = np.nan
    if method == CHALLENGER:
        details = [reclaim(records[:i + 1], warmup) for i in range(len(records))]
        featured["entry_details"] = details
        featured.loc[[not x["signal"] for x in details], "score"] = np.nan
    else:
        featured["entry_details"] = [{} for _ in records]

    # Convert price-valued risk indicators to the raw share units used for fills.
    # All current inputs have a constant split factor, verified by the loader.
    factor = raw.loc[featured.index, "close"] / source["close"]
    for column in ("ATR14", "ema20", "macd", "macd_signal"):
        featured[column] *= factor
    for column in ("open", "high", "low", "close", "volume"):
        featured[column] = raw.loc[featured.index, column]
    result = featured.loc[featured.index >= fold["test_start"]].copy()
    if result.empty or not result["entry_ready"].iloc[0]:
        raise ValueError("Insufficient fresh warmup at evaluation boundary")
    if method == BENCHMARK:
        result["score"] = np.nan
        result.iloc[0, result.columns.get_loc("score")] = 1.0
    return result
