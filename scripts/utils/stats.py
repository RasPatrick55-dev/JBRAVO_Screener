"""Statistical helper utilities used throughout the feature pipeline."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


MAD_Z_FACTOR = 0.6744897501960817  # == scipy.stats.norm.ppf(0.75)


def robust_z(series: pd.Series, clip: float = 3.0) -> pd.Series:
    """Return a median/MAD based z-score clipped to ``[-clip, clip]``.

    The implementation mirrors the usual definition of a *robust* z-score:

    ``z = (x - median(x)) / (MAD(x) / 0.67448975)``

    ``MAD`` denotes the median absolute deviation.  When the series is
    constant (``MAD == 0``) the z-score is defined as zero to avoid
    spurious infinities.  Any NaNs in the original series are converted to
    zeros after clipping so downstream code does not need to special-case
    missing values.
    """

    if series.empty:
        return series.copy()

    median = series.median(skipna=True)
    diff = series - median
    mad = np.median(np.abs(diff.dropna()))

    if not np.isfinite(mad) or mad == 0:
        z = pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    else:
        z = MAD_Z_FACTOR * diff / mad

    z = z.clip(lower=-clip, upper=clip)
    if series.isna().any():
        z = z.where(~series.isna(), 0.0)
    return z.fillna(0.0)


def _resolve_min_periods(window: int, min_periods: Optional[int]) -> int:
    return window if min_periods is None else min_periods


def rolling_mean(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    window: int,
    *,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Return a rolling mean aligned to the original index for each group."""

    return (
        grouped.rolling(window=window, min_periods=_resolve_min_periods(window, min_periods))
        .mean()
        .reset_index(level=0, drop=True)
    )


def rolling_max(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    window: int,
    *,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Return a rolling maximum aligned to the input index for each group."""

    return (
        grouped.rolling(window=window, min_periods=_resolve_min_periods(window, min_periods))
        .max()
        .reset_index(level=0, drop=True)
    )


def rolling_min(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    window: int,
    *,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Return a rolling minimum aligned to the input index for each group."""

    return (
        grouped.rolling(window=window, min_periods=_resolve_min_periods(window, min_periods))
        .min()
        .reset_index(level=0, drop=True)
    )


def rolling_apply(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    window: int,
    func: Callable[[np.ndarray], float],
    *,
    min_periods: Optional[int] = None,
    raw: bool = True,
) -> pd.Series:
    """Apply ``func`` over a rolling window for each group.

    The return value is re-aligned to the original index similar to the
    other helpers above.  ``raw=True`` passes numpy arrays to ``func`` for
    better performance.
    """

    return (
        grouped.rolling(window=window, min_periods=_resolve_min_periods(window, min_periods))
        .apply(func, raw=raw)
        .reset_index(level=0, drop=True)
    )
