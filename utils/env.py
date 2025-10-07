"""Environment loading helpers for CLI and service entry points."""

from __future__ import annotations

import os
from typing import Optional, Tuple

from dotenv import load_dotenv, find_dotenv


def load_env() -> None:
    """Load project and user-level environment variables if available."""
    load_dotenv(find_dotenv(usecwd=True))
    alt = os.path.expanduser("~/.config/jbravo/.env")
    if os.path.exists(alt):
        load_dotenv(alt, override=False)


def get_alpaca_creds() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return Alpaca credentials from the environment with sensible fallbacks."""
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_API_BASE_URL")
    feed = os.getenv("ALPACA_DATA_FEED", "iex")
    return key, secret, base, feed
