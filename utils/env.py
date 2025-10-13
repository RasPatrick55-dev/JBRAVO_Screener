"""Environment loading helpers for CLI and service entry points."""

from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Tuple

from dotenv import dotenv_values, find_dotenv, load_dotenv


_REQUIRED_KEYS: tuple[tuple[str, ...], ...] = (
    ("APCA_API_KEY_ID", "ALPACA_API_KEY_ID"),
    ("APCA_API_SECRET_KEY", "ALPACA_API_SECRET_KEY"),
)


def _normalize_apca_base_url(value: str) -> str:
    """Return ``value`` without trailing ``/v2`` or slashes."""

    trimmed = value.strip()
    if not trimmed:
        return ""
    trimmed = trimmed.rstrip("/")
    if trimmed.endswith("/v2"):
        trimmed = trimmed[: -len("/v2")]
    return trimmed.rstrip("/")


def load_env() -> Dict[str, Dict[str, int]]:
    """Load environment files and normalize common Alpaca variables.

    Returns a mapping describing which variables were populated during the
    loading process. Each entry tracks whether the variable is present and the
    length of the normalized value.
    """

    summary: Dict[str, Dict[str, int]] = {}

    loaded_paths: list[str] = []
    primary = find_dotenv(usecwd=True)
    if primary:
        load_dotenv(primary)
        loaded_paths.append(primary)

    alt = os.path.expanduser("~/.config/jbravo/.env")
    if os.path.exists(alt):
        load_dotenv(alt, override=False)
        loaded_paths.append(alt)

    tracked: set[str] = set()
    for path in loaded_paths:
        try:
            values: Mapping[str, str | None] = dotenv_values(path)
        except Exception:
            continue
        tracked.update(key for key, value in values.items() if value is not None)

    tracked.update({"APCA_API_BASE_URL", "ALPACA_API_BASE_URL"})
    for keys in _REQUIRED_KEYS:
        tracked.update(keys)

    for key in tracked:
        raw = os.environ.get(key)
        if raw is None:
            summary[key] = {"present": False, "len": 0}
            continue
        normalized = raw.strip()
        if key == "APCA_API_BASE_URL":
            normalized = _normalize_apca_base_url(normalized)
        if normalized != raw:
            os.environ[key] = normalized
        summary[key] = {"present": bool(normalized), "len": len(normalized)}

    for primary_key, *aliases in _REQUIRED_KEYS:
        canon_value = os.environ.get(primary_key, "").strip()
        if canon_value:
            summary.setdefault(primary_key, {"present": True, "len": len(canon_value)})
            continue
        fallback_value = ""
        fallback_key = None
        for alias in aliases:
            alias_val = os.environ.get(alias, "").strip()
            if alias_val:
                fallback_value = alias_val
                fallback_key = alias
                break
        if fallback_value:
            os.environ[primary_key] = fallback_value
            summary[primary_key] = {"present": True, "len": len(fallback_value)}
            if fallback_key:
                summary.setdefault(
                    fallback_key, {"present": True, "len": len(fallback_value)}
                )
        else:
            summary.setdefault(primary_key, {"present": False, "len": 0})
            for alias in aliases:
                summary.setdefault(alias, {"present": False, "len": 0})

    return summary


def get_alpaca_creds() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return Alpaca credentials from the environment with sensible fallbacks."""
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_API_BASE_URL")
    feed = os.getenv("ALPACA_DATA_FEED", "iex")
    return key, secret, base, feed
