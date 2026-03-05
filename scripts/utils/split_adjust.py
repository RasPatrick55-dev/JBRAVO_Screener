"""Split-adjustment helper for raw OHLCV time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SPLIT_MODES = {"off", "auto", "force"}
COMMON_SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 10.0, 20.0)
LOW_RATIO_THRESHOLD = 0.35
HIGH_RATIO_THRESHOLD = 2.85
RATIO_TOLERANCE = 0.05


@dataclass(frozen=True)
class _SplitEvent:
    row_index: int
    symbol: str
    timestamp: Any
    close_ratio: float
    split_ratio: float
    history_multiplier: float


def _detect_split_ratio(price_ratio: float) -> tuple[float, float] | tuple[None, None]:
    if not np.isfinite(price_ratio) or price_ratio <= 0:
        return None, None

    direction = None
    implied = None
    if price_ratio <= LOW_RATIO_THRESHOLD:
        direction = "forward"
        implied = 1.0 / price_ratio
    elif price_ratio >= HIGH_RATIO_THRESHOLD:
        direction = "reverse"
        implied = price_ratio
    if implied is None:
        return None, None

    candidates: list[tuple[float, float]] = []
    for common in COMMON_SPLIT_RATIOS:
        rel_err = abs(implied - common) / common
        if rel_err <= RATIO_TOLERANCE:
            candidates.append((rel_err, common))
    if not candidates and implied >= 20.0:
        nearest_int = float(max(int(round(implied)), 2))
        rel_err_int = abs(implied - nearest_int) / nearest_int
        if rel_err_int <= RATIO_TOLERANCE:
            candidates.append((rel_err_int, nearest_int))
    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item[0])
    split_ratio = float(candidates[0][1])
    history_multiplier = (1.0 / split_ratio) if direction == "forward" else split_ratio
    return split_ratio, history_multiplier


def _normalize_mode(mode: str | None) -> str:
    normalized = str(mode or "off").strip().lower()
    if normalized not in SPLIT_MODES:
        return "off"
    return normalized


def adjust_for_splits(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    volume_col: str = "volume",
    group_col: str = "symbol",
    time_col: str = "timestamp",
    mode: str = "off",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a frame with optional ``close_adj`` / ``volume_adj`` split fallback.

    ``mode`` semantics:
    - ``off``: passthrough, no adjustment columns added.
    - ``auto``: add adjusted columns only when split-like events are detected.
    - ``force``: always add adjusted columns (attempt adjustment even when no events).
    """

    resolved_mode = _normalize_mode(mode)
    out = df.copy()
    meta: dict[str, Any] = {
        "split_adjust_mode": resolved_mode,
        "split_adjust_applied": False,
        "split_events": 0,
        "split_symbols": 0,
        "rows_affected": 0,
        "price_col": price_col,
        "close_adj_col": None,
        "volume_adj_col": None,
        "detector": {
            "low_ratio_threshold": LOW_RATIO_THRESHOLD,
            "high_ratio_threshold": HIGH_RATIO_THRESHOLD,
            "ratio_tolerance": RATIO_TOLERANCE,
            "common_ratios": [int(v) for v in COMMON_SPLIT_RATIOS],
        },
    }
    if resolved_mode == "off":
        return out, meta

    required = {group_col, time_col, price_col}
    missing = [column for column in required if column not in out.columns]
    if missing:
        return out, meta

    work = out.copy()
    work["__orig_idx"] = np.arange(work.shape[0], dtype=np.int64)
    work[group_col] = work[group_col].astype("string")
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=[group_col, time_col]).sort_values([group_col, time_col]).reset_index(
        drop=True
    )

    split_events: list[_SplitEvent] = []
    adjust_factor = np.ones(work.shape[0], dtype="float64")
    for symbol, idx in work.groupby(group_col, sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=np.int64)
        if loc.size == 0:
            continue
        close_values = pd.to_numeric(work.loc[loc, price_col], errors="coerce").to_numpy(dtype=float)
        ratios = np.full(loc.size, np.nan, dtype=float)
        if loc.size > 1:
            prev = close_values[:-1]
            curr = close_values[1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios[1:] = curr / prev

        event_multiplier_by_pos: dict[int, tuple[float, float]] = {}
        for pos in range(1, loc.size):
            ratio = float(ratios[pos])
            split_ratio, hist_multiplier = _detect_split_ratio(ratio)
            if split_ratio is None or hist_multiplier is None:
                continue
            event_multiplier_by_pos[pos] = (split_ratio, hist_multiplier)
            split_events.append(
                _SplitEvent(
                    row_index=int(loc[pos]),
                    symbol=str(symbol),
                    timestamp=work.loc[loc[pos], time_col],
                    close_ratio=ratio,
                    split_ratio=float(split_ratio),
                    history_multiplier=float(hist_multiplier),
                )
            )

        cumulative = 1.0
        symbol_factors = np.ones(loc.size, dtype=float)
        for pos in range(loc.size - 1, -1, -1):
            symbol_factors[pos] = cumulative
            event = event_multiplier_by_pos.get(pos)
            if event is not None:
                cumulative *= float(event[1])
        adjust_factor[loc] = symbol_factors

    event_count = len(split_events)
    symbols_with_events = len({event.symbol for event in split_events})
    rows_affected = int(
        ((np.abs(adjust_factor - 1.0) > 1e-12) & work[price_col].notna().to_numpy()).sum()
    )
    meta.update(
        {
            "split_adjust_applied": bool(event_count > 0),
            "split_events": int(event_count),
            "split_symbols": int(symbols_with_events),
            "rows_affected": int(rows_affected),
            "events_sample": [
                {
                    "symbol": event.symbol,
                    "timestamp": pd.Timestamp(event.timestamp).isoformat()
                    if pd.notna(event.timestamp)
                    else None,
                    "close_ratio": float(event.close_ratio),
                    "split_ratio": float(event.split_ratio),
                    "history_multiplier": float(event.history_multiplier),
                }
                for event in split_events[:25]
            ],
        }
    )

    if resolved_mode == "auto" and event_count == 0:
        return out, meta

    work["close_adj"] = pd.to_numeric(work[price_col], errors="coerce") * adjust_factor
    meta["close_adj_col"] = "close_adj"
    if volume_col in work.columns:
        volume_values = pd.to_numeric(work[volume_col], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            work["volume_adj"] = volume_values / adjust_factor
        meta["volume_adj_col"] = "volume_adj"

    # Restore original row order and keep untouched columns.
    work = work.sort_values("__orig_idx").drop(columns=["__orig_idx"])
    return work, meta
