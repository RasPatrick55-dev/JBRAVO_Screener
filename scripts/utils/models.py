"""Pydantic models used by the screener and related helpers."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

try:  # pragma: no cover - import path depends on Pydantic major version
    from pydantic import BaseModel, field_validator
    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - fallback for Pydantic v1
    from pydantic import BaseModel, validator  # type: ignore

    _PYDANTIC_V2 = False


KNOWN_EQUITY = {
    "NASDAQ",
    "NYSE",
    "ARCA",
    "AMEX",
    "BATS",
    "NYSEARCA",
}

# Exchanges we consider valid for the equity screener universe. This set is
# intentionally aligned with ``KNOWN_EQUITY`` with IEX added explicitly to
# allow its symbols while still filtering out unknown/OTC venues.
ALLOWED_EQUITY_EXCHANGES = KNOWN_EQUITY | {"IEX"}


def classify_exchange(raw: Any) -> str:
    """Classify an arbitrary exchange code into EQUITY/CRYPTO/OTHER."""

    if raw is None:
        return "OTHER"
    code = str(raw).strip().upper()
    if not code:
        return "OTHER"
    if code in KNOWN_EQUITY:
        return "EQUITY"
    return "CRYPTO" if "CRYPTO" in code else "OTHER"


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_exchange(value: Any) -> str:
    return str(value or "").strip().upper()


def _parse_timestamp(value: Any) -> datetime:
    if value is None or value == "":
        raise ValueError("timestamp is required")
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _coerce_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if _PYDANTIC_V2:

    class BarData(BaseModel):
        symbol: str
        timestamp: datetime
        open: float
        high: float
        low: float
        close: float
        volume: float
        exchange: str = ""

        model_config = {"extra": "ignore"}

        @field_validator("symbol", mode="before")
        @classmethod
        def _symbol(cls, value: Any) -> str:
            return _normalize_symbol(value)

        @field_validator("exchange", mode="before")
        @classmethod
        def _exchange(cls, value: Any) -> str:
            return _normalize_exchange(value)

        @field_validator("timestamp", mode="before")
        @classmethod
        def _timestamp(cls, value: Any) -> datetime:
            return _parse_timestamp(value)

        @field_validator("open", "high", "low", "close", "volume", mode="before")
        @classmethod
        def _floats(cls, value: Any) -> float:
            return _coerce_float(value)

        def to_dict(self) -> dict[str, Any]:
            return self.model_dump()

else:

    class BarData(BaseModel):  # type: ignore[no-redef]
        symbol: str
        timestamp: datetime
        open: float
        high: float
        low: float
        close: float
        volume: float
        exchange: str = ""

        class Config:
            extra = "ignore"

        @validator("symbol", pre=True, always=True)
        def _symbol(cls, value: Any) -> str:  # type: ignore[override]
            return _normalize_symbol(value)

        @validator("exchange", pre=True, always=True)
        def _exchange(cls, value: Any) -> str:  # type: ignore[override]
            return _normalize_exchange(value)

        @validator("timestamp", pre=True)
        def _timestamp(cls, value: Any) -> datetime:  # type: ignore[override]
            return _parse_timestamp(value)

        @validator("open", "high", "low", "close", "volume", pre=True)
        def _floats(cls, value: Any) -> float:  # type: ignore[override]
            return _coerce_float(value)

        def to_dict(self) -> dict[str, Any]:
            return self.dict()


__all__ = ["BarData", "classify_exchange", "KNOWN_EQUITY"]
