"""Account value compatibility for Alpaca REST payloads and SDK models."""

from __future__ import annotations

import math
from typing import Any, Mapping


def account_equity(account: Any) -> float | None:
    """Prefer equity; accept the equivalent legacy alias for older snapshots.

    Read the legacy SDK property only when equity is unavailable, since modern
    alpaca-py models warn on access to deprecated fields. Zero is a valid value.
    """
    for field in ("equity", "portfolio_value"):
        value = account.get(field) if isinstance(account, Mapping) else getattr(account, field, None)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            return number
    return None
