"""Forward-return sanity diagnostics shared by ML evaluation scripts."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def summarize_forward_returns(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            "count": 0,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "max_abs": None,
            "outlier_suspected": False,
        }

    p50 = _safe_float(values.quantile(0.50))
    p95 = _safe_float(values.quantile(0.95))
    p99 = _safe_float(values.quantile(0.99))
    vmin = _safe_float(values.min())
    vmax = _safe_float(values.max())
    max_abs = None
    if vmin is not None and vmax is not None:
        max_abs = max(abs(vmin), abs(vmax))
    outlier = bool((p99 is not None and abs(p99) > 1.0) or (max_abs is not None and max_abs > 3.0))
    return {
        "count": int(values.shape[0]),
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "min": vmin,
        "max": vmax,
        "max_abs": max_abs,
        "outlier_suspected": outlier,
    }


def log_forward_return_sanity(
    series: pd.Series,
    *,
    column_name: str,
    logger: logging.Logger,
    suggestion: str = "use_adjustment_split_or_all",
) -> dict[str, float | int | None]:
    summary = summarize_forward_returns(series)
    if summary.get("outlier_suspected"):
        logger.warning(
            "[WARN] FWD_RET_OUTLIER_SUSPECTED col=%s p99=%s max=%s suggestion=%s",
            column_name,
            summary.get("p99"),
            summary.get("max_abs"),
            suggestion,
        )
    return summary


def clip_forward_returns(
    frame: pd.DataFrame,
    *,
    column_name: str,
    max_abs: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, int]:
    if max_abs is None or float(max_abs) <= 0:
        return frame, 0
    if column_name not in frame.columns:
        return frame, 0
    out = frame.copy()
    numeric = pd.to_numeric(out[column_name], errors="coerce")
    clipped = numeric.clip(lower=-float(max_abs), upper=float(max_abs))
    changed = int((numeric.notna() & (clipped != numeric)).sum())
    out[column_name] = clipped
    if changed > 0:
        logger.warning(
            "[WARN] FWD_RET_CLIPPED max_abs=%s clipped_rows=%d",
            float(max_abs),
            changed,
        )
    return out, changed
