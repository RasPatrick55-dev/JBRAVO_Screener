"""Environment helpers for Alpaca configuration."""
from __future__ import annotations

import os


def trading_base_url() -> str:
    """Return the Alpaca trading API base URL.

    Prefers explicit ``APCA_API_BASE_URL``/``ALPACA_API_BASE_URL`` but falls back to
    ``APCA_API_ENV`` to determine paper vs live trading. Defaults to the paper
    trading host.
    """

    env_base = os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_API_BASE_URL")
    if env_base:
        return env_base.rstrip("/")
    env = (os.getenv("APCA_API_ENV") or "paper").strip().lower()
    if env == "live":
        return "https://api.alpaca.markets"
    return "https://paper-api.alpaca.markets"


def market_data_base_url() -> str:
    """Return the Alpaca market data API base URL."""

    base = os.getenv("APCA_DATA_API_BASE_URL") or "https://data.alpaca.markets"
    return base.rstrip("/")


__all__ = ["trading_base_url", "market_data_base_url"]
