"""Causal daily-bar pattern detectors and shared, fail-closed entry gates.

Detectors receive history ending at the signal bar. They must never inspect
future bars. Add a detector to PATTERN_DETECTORS to introduce another pattern.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternSignal:
    pattern: str
    signal_time: str
    entry: float
    resistance: float
    invalidation: float
    target: float
    metrics: dict[str, float | int] = field(default_factory=dict)
    version: int = 1
    status: str = "confirmed"


@dataclass(frozen=True)
class BullFlagConfig:
    min_pole_bars: int = 3
    max_pole_bars: int = 10
    min_flag_bars: int = 3
    max_flag_bars: int = 10
    min_pole_gain: float = 0.08
    max_retracement: float = 0.5
    max_flag_volume_ratio: float = 0.8
    breakout_volume_ratio: float = 1.5
    volume_lookback: int = 20
    breakout_buffer: float = 0.001
    max_extension: float = 0.05

    def __post_init__(self) -> None:
        for low, high in (
            (self.min_pole_bars, self.max_pole_bars),
            (self.min_flag_bars, self.max_flag_bars),
        ):
            if not isinstance(low, int) or not isinstance(high, int) or not 3 <= low <= high:
                raise ValueError("Pattern windows must be integers with 3 <= min <= max")
        if not isinstance(self.volume_lookback, int) or self.volume_lookback < 1:
            raise ValueError("volume_lookback must be a positive integer")
        ratios = (
            self.min_pole_gain,
            self.max_retracement,
            self.max_flag_volume_ratio,
            self.breakout_volume_ratio,
            self.breakout_buffer,
            self.max_extension,
        )
        if not all(np.isfinite(value) and value > 0 for value in ratios):
            raise ValueError("Pattern ratios must be finite and positive")
        if self.max_retracement >= 1 or self.max_flag_volume_ratio > 1:
            raise ValueError("Flag retracement and volume must contract")
        if self.max_extension <= self.breakout_buffer:
            raise ValueError("max_extension must exceed breakout_buffer")


def detect_bull_flag(
    bars: pd.DataFrame,
    config: BullFlagConfig = BullFlagConfig(),
) -> PatternSignal | None:
    """Find a confirmed upside breakout on the final supplied daily bar.

    Version 1 uses horizontal resistance at the highest flag high, a rising
    pole, a sideways/falling flag, contracting volume, and a volume breakout.
    Invalidation and measured-move target become fixed exit levels when the
    confirmed signal is bound to an entry order.
    """
    cfg = config
    columns = ["open", "high", "low", "close", "volume"]
    required = max(cfg.volume_lookback + 1, cfg.min_pole_bars + cfg.min_flag_bars + 1)
    if (
        not set(columns).issubset(bars.columns)
        or len(bars) < required
        or not isinstance(bars.index, pd.DatetimeIndex)
        or bars.index.hasnans
        or not bars.index.is_monotonic_increasing
        or bars.index.normalize().has_duplicates
    ):
        return None
    length = max(cfg.max_pole_bars + cfg.max_flag_bars + 1, cfg.volume_lookback + 1)
    frame = bars.iloc[-length:][columns].apply(pd.to_numeric, errors="coerce")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        return None
    if (frame.high < frame[["open", "close"]].max(axis=1)).any() or (
        frame.low > frame[["open", "close"]].min(axis=1)
    ).any():
        return None
    current = frame.iloc[-1]
    prior_volume = float(frame.volume.iloc[-cfg.volume_lookback - 1 : -1].mean())
    volume_ratio = float(current.volume / prior_volume)
    if current.close <= current.open or volume_ratio < cfg.breakout_volume_ratio:
        return None
    for flag_size in range(cfg.max_flag_bars, cfg.min_flag_bars - 1, -1):
        flag = frame.iloc[-flag_size - 1 : -1]
        if len(flag) != flag_size:
            continue
        resistance, floor = float(flag.high.max()), float(flag.low.min())
        if (
            not resistance * (1 + cfg.breakout_buffer)
            < current.close
            <= resistance * (1 + cfg.max_extension)
        ):
            continue
        # Both boundaries must slope sideways/down, rather than form an upward channel.
        if any(
            np.polyfit(np.arange(flag_size), flag[col], 1)[0] > resistance * 1e-10
            for col in ("high", "low")
        ):
            continue
        for pole_size in range(cfg.min_pole_bars, cfg.max_pole_bars + 1):
            start = len(frame) - flag_size - pole_size - 1
            if start < 0:
                continue
            pole = frame.iloc[start : start + pole_size]
            peak, base = float(pole.high.iloc[-1]), float(pole.low.min())
            gain = float(pole.close.iloc[-1] / pole.open.iloc[0] - 1)
            height = peak - base
            if (
                gain < cfg.min_pole_gain
                or height <= 0
                or peak < pole.high.max()
                or resistance > peak
                or floor <= base
            ):
                continue
            retracement = (peak - floor) / height
            if not 0 <= retracement <= cfg.max_retracement:
                continue
            if flag.volume.mean() > pole.volume.mean() * cfg.max_flag_volume_ratio:
                continue
            return PatternSignal(
                pattern="bull_flag",
                signal_time=bars.index[-1].isoformat(),
                entry=float(current.close),
                resistance=resistance,
                invalidation=floor * (1 - cfg.breakout_buffer),
                target=float(current.close + height),
                metrics={
                    "pole_bars": pole_size,
                    "flag_bars": flag_size,
                    "pole_gain": gain,
                    "retracement": float(retracement),
                    "volume_ratio": volume_ratio,
                },
            )
    return None


PATTERN_DETECTORS: dict[str, Callable[[pd.DataFrame], PatternSignal | None]] = {
    "bull_flag": detect_bull_flag,
}


def configured_patterns(value: str | None = None) -> tuple[str, ...]:
    """Empty setting preserves legacy entries; unknown names are configuration errors."""
    raw = os.getenv("JBR_ENTRY_PATTERNS", "") if value is None else value
    names = tuple(dict.fromkeys(part.strip().lower() for part in raw.split(",") if part.strip()))
    unknown = set(names) - PATTERN_DETECTORS.keys()
    if unknown:
        raise ValueError(f"Unknown entry patterns: {', '.join(sorted(unknown))}")
    return names


def detect_patterns(bars: pd.DataFrame, names: Sequence[str]) -> list[PatternSignal]:
    signals = []
    for name in names:
        signal = PATTERN_DETECTORS[name](bars)
        if signal is not None:
            signals.append(signal)
    return signals


def _payload(raw: object) -> dict:
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else {}
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        return {}


def annotate_pattern_candidates(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    names: Sequence[str],
    *,
    now: datetime,
) -> pd.DataFrame:
    """Store evidence in the existing PostgreSQL score_breakdown JSON contract.

    Ignore today's bar before 16:00 New York time, including on half days
    (conservatively waiting until 16:00). Future bars are always excluded.
    """
    if not names or candidates.empty:
        return candidates
    result = candidates.copy()
    working = completed_daily_bars(bars, now=now)
    evidence = {}
    for symbol, group in working.groupby("symbol"):
        history = group.sort_values("timestamp").set_index("timestamp")
        signals = detect_patterns(history, names)
        evidence[str(symbol).upper()] = [
            dict(asdict(signal), symbol=str(symbol).upper()) for signal in signals
        ]
    payloads = []
    for row in result.to_dict("records"):
        payload = _payload(row.get("score_breakdown"))
        payload["chart_patterns"] = evidence.get(str(row.get("symbol", "")).upper(), [])
        payloads.append(json.dumps(payload, allow_nan=False, sort_keys=True))
    result["score_breakdown"] = payloads
    return result


def completed_daily_bars(bars: pd.DataFrame, *, now: datetime) -> pd.DataFrame:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    ny_now = now.astimezone(ZoneInfo("America/New_York"))
    working = bars.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    dates = working.timestamp.dt.date
    complete = dates < ny_now.date()
    if ny_now.hour >= 16:
        complete |= dates == ny_now.date()
    return working.loc[complete].copy()


def matching_pattern(
    row: Mapping, names: Sequence[str], *, now: datetime | None = None
) -> dict | None:
    """Validate persisted confirmation; missing, malformed or stale evidence fails closed."""
    signals = _payload(row.get("score_breakdown")).get("chart_patterns", [])
    if not isinstance(signals, list):
        return None
    timestamp = _timestamp(row.get("timestamp"))
    if pd.isna(timestamp):
        return None
    if now is not None:
        ny_now = now.astimezone(ZoneInfo("America/New_York"))
        age_days = (ny_now.date() - timestamp.date()).days
        if not 0 <= age_days <= 4 or (age_days == 0 and ny_now.hour < 16):
            return None
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        if (
            signal.get("pattern") not in names
            or signal.get("status") != "confirmed"
            or signal.get("version") != 1
        ):
            continue
        if not isinstance(signal.get("symbol"), str) or not isinstance(row.get("symbol"), str):
            continue
        if not signal["symbol"] or signal["symbol"] != row["symbol"]:
            continue
        signal_time = _timestamp(signal.get("signal_time"))
        if pd.isna(signal_time) or signal_time != timestamp:
            continue
        try:
            entry, resistance, stop, target, close = (
                float(signal["entry"]),
                float(signal["resistance"]),
                float(signal["invalidation"]),
                float(signal["target"]),
                float(row["close"]),
            )
        except (KeyError, ValueError, TypeError):
            continue
        if (
            all(np.isfinite(v) for v in (entry, resistance, stop, target, close))
            and 0 < stop < resistance < entry < target
            and np.isclose(close, entry, rtol=1e-6, atol=0)
        ):
            return signal
    return None


def _timestamp(value: object) -> pd.Timestamp:
    if not isinstance(value, (str, datetime, np.datetime64)):
        return pd.NaT
    return pd.to_datetime(value, utc=True, errors="coerce")


def filter_pattern_candidates(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    if not names or frame.empty:
        return frame
    keep = [matching_pattern(row, names) is not None for row in frame.to_dict("records")]
    return frame.loc[keep].copy()


def pattern_entry_mask(bars: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    """Signals as of each close; backtesting must shift these before execution."""
    return pd.Series(
        [bool(detect_patterns(bars.iloc[:end], names)) for end in range(1, len(bars) + 1)],
        index=bars.index,
        dtype=bool,
    )


def validated_exit_plan(raw: object, symbol: str) -> dict | None:
    """Validate an original entry signal without expiring a still-open trade."""
    if not isinstance(raw, Mapping):
        return None
    plan = dict(raw)
    if (
        plan.get("symbol") != symbol
        or not isinstance(plan.get("pattern"), str)
        or plan["pattern"] not in PATTERN_DETECTORS
    ):
        return None
    # Reuse geometry/version validation, but not the entry's four-day age gate.
    row = {
        "symbol": symbol,
        "timestamp": plan.get("signal_time"),
        "close": plan.get("entry"),
        "score_breakdown": {"chart_patterns": [plan]},
    }
    if matching_pattern(row, (plan["pattern"],)) is None:
        return None
    for key in ("entry", "resistance", "invalidation", "target"):
        plan[key] = float(plan[key])
    return plan


def pattern_exit_reason(plan: Mapping, price: float) -> str | None:
    """Observed-price triggers for a long position. Stop takes priority."""
    if not np.isfinite(price) or price <= 0:
        return None
    if price <= float(plan["invalidation"]):
        return "PATTERN_INVALIDATION"
    if price >= float(plan["target"]):
        return "PATTERN_TARGET"
    return None


def pattern_bar_exit(
    plan: Mapping,
    row: Mapping,
    protective_stops: Sequence[tuple[str, float | None]] = (),
) -> tuple[str, float] | None:
    """Conservative daily-bar fills: gaps at open, ambiguous bars stop first.

    Protective stops supplied here must have existed before this bar opened.
    A target is a full exit of the remaining position.
    """
    opening, high, low = (float(row[key]) for key in ("open", "high", "low"))
    if not all(np.isfinite(value) and value > 0 for value in (opening, high, low)):
        return None
    stops = [("PATTERN_INVALIDATION", float(plan["invalidation"]))]
    stops.extend(
        (code, float(value))
        for code, value in protective_stops
        if value is not None and np.isfinite(value) and value > 0
    )
    stop_code, stop = max(stops, key=lambda item: item[1])
    target = float(plan["target"])
    if opening <= stop:
        return stop_code, opening
    if opening >= target:
        return "PATTERN_TARGET", opening
    if low <= stop:
        return stop_code, stop
    if high >= target:
        return "PATTERN_TARGET", target
    return None


def pattern_signal_series(bars: pd.DataFrame, names: Sequence[str], symbol: str) -> pd.Series:
    plans = []
    for end in range(1, len(bars) + 1):
        signals = detect_patterns(bars.iloc[:end], names)
        plans.append(dict(asdict(signals[0]), symbol=symbol) if signals else None)
    return pd.Series(plans, index=bars.index, dtype=object)
