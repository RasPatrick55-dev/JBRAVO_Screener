"""Helpers for writing ``screener_metrics.json`` consistently.

This module centralises the logic for adding canonical KPI fields required by the
dashboard and ensures atomic writes so readers never observe partial output.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import atomic_write_bytes


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


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

    metrics["rows_out"] = _coerce_int(metrics.get("rows_out") or metrics.get("rows", 0))
    any_bars = metrics.get("symbols_with_any_bars")
    required_bars = metrics.get("symbols_with_required_bars")
    if required_bars is None:
        required_bars = metrics.get("symbols_with_bars")
    if any_bars is None:
        any_bars = metrics.get("symbols_with_bars_any") or metrics.get("symbols_with_bars")

    metrics["with_bars_required"] = _coerce_int(required_bars)
    metrics["with_bars_any"] = _coerce_int(any_bars)
    metrics["with_bars"] = metrics["with_bars_required"]
    metrics["symbols_with_required_bars"] = metrics["with_bars_required"]
    metrics["symbols_with_any_bars"] = metrics["with_bars_any"]
    if "symbols_with_bars" not in metrics or metrics.get("symbols_with_bars") in (None, ""):
        metrics["symbols_with_bars"] = metrics["with_bars_required"]
    metrics.setdefault("symbols_with_bars_required", metrics["with_bars_required"])
    metrics.setdefault("symbols_with_bars_any", metrics["with_bars_any"])

    universe_count = metrics.get("universe_count")
    if universe_count is None:
        universe_count = metrics.get("symbols_in")
        if universe_count is None:
            universe_count = metrics.get("symbols_with_bars")
    metrics["universe_count"] = _coerce_int(universe_count)

    gate_breakdown = metrics.get("gate_breakdown")
    metrics["gate_breakdown"] = dict(gate_breakdown) if isinstance(gate_breakdown, Mapping) else {}

    return metrics


def write_screener_metrics_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write screener metrics atomically with canonical KPI fields.

    The payload is enriched via :func:`ensure_canonical_metrics` before being
    written to ``path``.
    """

    enriched = ensure_canonical_metrics(payload)
    serialised = json.dumps(enriched, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, serialised)


__all__ = ["ensure_canonical_metrics", "write_screener_metrics_json"]
