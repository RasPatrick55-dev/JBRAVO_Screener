from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Optional, Protocol

import requests

from utils.io_utils import atomic_write_bytes

LOGGER = logging.getLogger(__name__)


class SentimentProvider(Protocol):
    """Interface for loading per-symbol sentiment scores."""

    def get_symbol_sentiment(self, symbol: str, asof_utc: datetime) -> Optional[float]:
        """Return the sentiment score for ``symbol`` at ``asof_utc`` or ``None`` on error."""
        raise NotImplementedError


def _coerce_sentiment_value(value: object) -> Optional[float]:
    try:
        score = float(value)
    except Exception:
        return None
    if not (-1.0 <= score <= 1.0):
        score = max(min(score, 1.0), -1.0)
    if score != score:  # NaN check
        return None
    return score


class JsonHttpSentimentProvider:
    """HTTP provider that fetches per-symbol sentiment from a JSON endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
        session: requests.sessions.Session | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout if timeout and timeout > 0 else 8.0
        self.session = session or requests.Session()
        self._logged_error = False

    def _log_once(self, message: str, *args: object) -> None:
        if self._logged_error:
            return
        try:
            LOGGER.warning(message, *args)
        except Exception:
            LOGGER.warning(message)
        self._logged_error = True

    def get_symbol_sentiment(self, symbol: str, asof_utc: datetime) -> Optional[float]:
        params = {
            "symbol": (symbol or "").strip().upper(),
            "date": asof_utc.date().isoformat(),
        }
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        try:
            response = self.session.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            self._log_once("Sentiment fetch failed: %s", exc)
            return None

        try:
            payload = response.json()
        except ValueError as exc:
            self._log_once("Invalid sentiment payload: %s", exc)
            return None

        if not isinstance(payload, Mapping):
            self._log_once("Unexpected sentiment response type: %s", type(payload))
            return None

        raw = payload.get("sentiment", payload.get("score"))
        score = _coerce_sentiment_value(raw)
        if score is None:
            self._log_once("Sentiment response missing/invalid for %s", params["symbol"])
            return None
        return score


def load_sentiment_cache(cache_dir: Path, run_date: date) -> dict[str, float]:
    """Load cached sentiments from ``cache_dir/<date>.json``."""

    cache_path = Path(cache_dir) / f"{run_date.isoformat()}.json"
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.warning("Failed to read sentiment cache %s", cache_path)
        return {}

    cache: dict[str, float] = {}
    if isinstance(data, Mapping):
        for sym, value in data.items():
            score = _coerce_sentiment_value(value)
            if score is None:
                continue
            cache[str(sym).upper()] = score
    return cache


def persist_sentiment_cache(cache_dir: Path, run_date: date, cache: Mapping[str, float]) -> Path:
    """Persist sentiment cache to ``cache_dir/<date>.json``."""

    path = Path(cache_dir) / f"{run_date.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned: dict[str, float] = {}
    for sym, value in cache.items():
        score = _coerce_sentiment_value(value)
        if score is None:
            continue
        cleaned[str(sym).upper()] = score
    serialized = json.dumps(cleaned, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, serialized)
    return path


__all__ = [
    "JsonHttpSentimentProvider",
    "SentimentProvider",
    "load_sentiment_cache",
    "persist_sentiment_cache",
]
