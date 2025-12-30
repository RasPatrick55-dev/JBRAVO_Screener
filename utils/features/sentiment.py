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
        self.errors = 0
        self._log_count = 0
        self._suppressed_logged = False

    def _record_error(self, message: str, *args: object) -> None:
        self.errors += 1
        if self._log_count < 3:
            try:
                LOGGER.warning(message, *args)
            except Exception:
                LOGGER.warning(message)
            self._log_count += 1
        elif not self._suppressed_logged:
            LOGGER.warning("Further sentiment fetch errors suppressed")
            self._suppressed_logged = True

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
        except requests.Timeout as exc:
            self._record_error("Sentiment fetch timed out for %s: %s", params["symbol"], exc)
            return None
        except Exception as exc:
            self._record_error("Sentiment fetch failed: %s", exc)
            return None

        status = getattr(response, "status_code", 200)
        if status != 200:
            self._record_error("Sentiment HTTP %s for %s", status, params["symbol"])
            return None

        try:
            response.raise_for_status()
        except Exception as exc:
            self._record_error("Sentiment fetch failed: %s", exc)
            return None

        try:
            payload = response.json()
        except ValueError as exc:
            self._record_error("Invalid sentiment payload: %s", exc)
            return None

        if not isinstance(payload, Mapping):
            self._record_error("Unexpected sentiment response type: %s", type(payload))
            return None

        raw = payload.get("sentiment", payload.get("score"))
        score = _coerce_sentiment_value(raw)
        if score is None:
            self._record_error("Sentiment response missing/invalid for %s", params["symbol"])
            return None
        return score

    def get(self, symbol: str, asof_utc: datetime) -> Optional[float]:
        """Alias for get_symbol_sentiment to match provider expectations."""

        return self.get_symbol_sentiment(symbol, asof_utc)


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


def persist_sentiment_cache(
    cache_dir: Path, run_date: date, cache: Mapping[str, float] | None = None
) -> Path:
    """Persist sentiment cache to ``cache_dir/<date>.json``."""

    path = Path(cache_dir) / f"{run_date.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned: dict[str, float] = {}
    for sym, value in (cache or {}).items():
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
