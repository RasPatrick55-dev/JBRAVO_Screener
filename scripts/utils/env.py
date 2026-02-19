"""Environment helpers for Alpaca configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv as _load_dotenv
except Exception:  # pragma: no cover - allow operation without python-dotenv
    _load_dotenv = None  # type: ignore

from utils.env import _normalize_env_aliases

_REQUIRED_DEFAULTS: tuple[str, ...] = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL",
    "APCA_DATA_API_BASE_URL",
    "ALPACA_DATA_FEED",
)


def _manual_parse_env(path: Path, *, override: bool) -> bool:
    try:
        contents = path.read_text(encoding="utf-8")
    except Exception:
        return False

    loaded = False
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        loaded = True
    return loaded


def _load_env_file(path: Path, *, override: bool = False) -> bool:
    if not path.exists():
        return False
    if _load_dotenv is not None:
        try:  # pragma: no cover - exercised via integration
            result = _load_dotenv(path, override=override)
        except Exception:
            result = False
        if result:
            return True
    return _manual_parse_env(path, override=override)


def load_env(required_keys: Sequence[str] | None = None) -> Tuple[list[str], list[str]]:
    """Load JBRAVO environment defaults from disk and report missing keys."""

    required = list(required_keys) if required_keys is not None else list(_REQUIRED_DEFAULTS)

    repo_root = Path(__file__).resolve().parents[2]
    user_env = Path(os.path.expanduser("~/.config/jbravo/.env"))
    repo_env = repo_root / ".env"

    loaded_files: list[str] = []
    for candidate in (user_env, repo_env):
        if _load_env_file(candidate):
            loaded_files.append(str(candidate))

    _normalize_env_aliases()

    if not os.environ.get("APCA_API_BASE_URL"):
        alias = os.environ.get("ALPACA_API_BASE_URL")
        if alias:
            os.environ["APCA_API_BASE_URL"] = alias.rstrip("/")

    if not os.environ.get("APCA_DATA_API_BASE_URL"):
        data_alias = os.environ.get("APCA_API_DATA_URL") or os.environ.get("ALPACA_API_DATA_URL")
        if data_alias:
            os.environ["APCA_DATA_API_BASE_URL"] = data_alias.rstrip("/")

    missing_required = [key for key in required if not os.environ.get(key)]

    return loaded_files, missing_required


def trading_base_url() -> str:
    """Return the Alpaca trading API base URL."""

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


__all__ = ["trading_base_url", "market_data_base_url", "load_env"]
