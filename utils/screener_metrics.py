"""Helpers for writing ``screener_metrics.json`` consistently.

This module centralises the logic for adding canonical KPI fields required by the
dashboard and ensures atomic writes so readers never observe partial output.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import atomic_write_bytes


DEFAULT_REQUIRED_BARS = 250


def configured_required_bars() -> int:
    env_value = os.getenv("REQUIRED_BARS")
    try:
        configured = int(env_value) if env_value not in (None, "") else DEFAULT_REQUIRED_BARS
    except Exception:
        configured = DEFAULT_REQUIRED_BARS
    return configured if configured > 0 else DEFAULT_REQUIRED_BARS


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def ensure_canonical_metrics(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``payload`` with canonical KPI fields populated.

    The dashboard expects the following keys to always be present and non-null:
    ``timestamp``, ``rows_out``, ``with_bars``, ``universe_count``, and
    ``gate_breakdown``. Existing keys are preserved.
    """

    metrics = dict(payload) if isinstance(payload, Mapping) else {}
    now_iso = datetime.now(timezone.utc).isoformat()

    timestamp = metrics.get("timestamp") or metrics.get("last_run_utc")
    metrics["timestamp"] = timestamp if isinstance(timestamp, str) and timestamp else now_iso

    configured_bars = configured_required_bars()
    metrics["required_bars"] = configured_bars

    symbols_in = _coerce_optional_int(metrics.get("symbols_in"))
    if symbols_in is None:
        symbols_in = _coerce_optional_int(metrics.get("universe_count"))
    rows_out = _coerce_optional_int(metrics.get("rows"))
    if rows_out is None:
        rows_out = _coerce_optional_int(metrics.get("rows_out"))

    required_bars_sym = _coerce_optional_int(metrics.get("symbols_with_required_bars"))
    if required_bars_sym is None:
        required_bars_sym = _coerce_optional_int(metrics.get("with_bars"))
    if required_bars_sym is None:
        required_bars_sym = _coerce_optional_int(metrics.get("symbols_with_bars"))

    any_bars_sym = _coerce_optional_int(metrics.get("symbols_with_any_bars"))
    if any_bars_sym is None:
        any_bars_sym = _coerce_optional_int(metrics.get("symbols_with_bars_any"))
    if any_bars_sym is None:
        any_bars_sym = required_bars_sym

    metrics["symbols_in"] = _coerce_int(symbols_in) if symbols_in is not None else 0
    metrics["rows"] = _coerce_int(rows_out)
    metrics["rows_out"] = metrics["rows"]
    metrics["with_bars_required"] = _coerce_int(required_bars_sym)
    metrics["with_bars_any"] = _coerce_int(any_bars_sym)
    metrics["with_bars"] = metrics["with_bars_required"]
    metrics["symbols_with_required_bars"] = metrics["with_bars_required"]
    metrics["symbols_with_any_bars"] = metrics["with_bars_any"]
    metrics["symbols_with_bars"] = _coerce_int(metrics.get("symbols_with_bars", metrics["with_bars_required"]))
    metrics["symbols_with_bars_required"] = metrics.get(
        "symbols_with_bars_required", metrics["with_bars_required"]
    )
    metrics["symbols_with_bars_any"] = metrics.get("symbols_with_bars_any", metrics["with_bars_any"])

    fetch_count = _coerce_optional_int(metrics.get("symbols_with_bars_fetch"))
    attempted_fetch = _coerce_optional_int(metrics.get("symbols_attempted_fetch"))
    if fetch_count is not None and fetch_count != metrics["symbols_with_any_bars"]:
        metrics.pop("symbols_with_bars_fetch", None)
        metrics["symbols_attempted_fetch"] = fetch_count
    elif fetch_count is not None:
        metrics["symbols_with_bars_fetch"] = fetch_count
    elif attempted_fetch is not None:
        metrics["symbols_attempted_fetch"] = attempted_fetch

    post_any = _coerce_optional_int(metrics.get("symbols_with_bars_post"))
    if post_any is None:
        post_any = _coerce_optional_int(metrics.get("symbols_with_any_bars_postprocess"))
    if post_any is not None:
        metrics["symbols_with_any_bars_postprocess"] = post_any
        metrics["symbols_with_bars_post"] = post_any

    universe_count = metrics.get("universe_count")
    if universe_count is None:
        universe_count = metrics["symbols_in"]
        if universe_count is None:
            universe_count = metrics.get("symbols_with_bars")
    metrics["universe_count"] = _coerce_int(universe_count)

    bars_total = metrics.get("bars_rows_total")
    if bars_total is None:
        bars_total = metrics.get("bars_rows") or metrics.get("bars_total") or metrics.get("bars")
    metrics["bars_rows_total"] = _coerce_int(bars_total)

    gate_breakdown = metrics.get("gate_breakdown")
    metrics["gate_breakdown"] = dict(gate_breakdown) if isinstance(gate_breakdown, Mapping) else {}

    metrics["metrics_version"] = 2

    return metrics


def write_screener_metrics_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write screener metrics atomically with canonical KPI fields.

    The payload is enriched via :func:`ensure_canonical_metrics` before being
    written to ``path``.
    """

    enriched = ensure_canonical_metrics(payload)
    serialised = json.dumps(enriched, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, serialised)


__all__ = ["configured_required_bars", "ensure_canonical_metrics", "write_screener_metrics_json"]
