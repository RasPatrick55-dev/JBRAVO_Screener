from __future__ import annotations

import json
import logging
import math
import os
from datetime import date
from pathlib import Path
from typing import Mapping, Optional

import requests

from utils.io_utils import atomic_write_bytes

LOGGER = logging.getLogger(__name__)
DEFAULT_CACHE_DIR = Path("data") / "cache" / "sentiment"


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _clamp_score(value: object) -> Optional[float]:
    try:
        score = float(value)
    except Exception:
        return None
    if math.isnan(score):
        return None
    if score < -1.0:
        score = -1.0
    elif score > 1.0:
        score = 1.0
    return score


class JsonHttpSentimentClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        env: Mapping[str, object] | None = None,
        session: requests.sessions.Session | None = None,
    ) -> None:
        self._env = env or os.environ
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()
        self.errors = 0
        self._log_count = 0

    def _config(self) -> dict[str, object]:
        env = self._env
        url_env = (env.get("SENTIMENT_API_URL") or "") if isinstance(env, Mapping) else ""
        key_env = env.get("SENTIMENT_API_KEY") if isinstance(env, Mapping) else None
        timeout_env = env.get("SENTIMENT_TIMEOUT_SECS") if isinstance(env, Mapping) else None
        use_flag = env.get("USE_SENTIMENT") if isinstance(env, Mapping) else False
        timeout_val = self.timeout if self.timeout not in (None, "", 0) else timeout_env
        try:
            timeout = float(timeout_val) if timeout_val not in (None, "") else 8.0
        except Exception:
            timeout = 8.0
        if timeout <= 0:
            timeout = 8.0
        return {
            "use_sentiment": _as_bool(use_flag, False),
            "url": (self.base_url or str(url_env or "")).strip(),
            "api_key": self.api_key if self.api_key not in (None, "") else key_env,
            "timeout": float(timeout),
        }

    def enabled(self) -> bool:
        cfg = self._config()
        return bool(cfg["use_sentiment"] and cfg["url"])

    def get_score(self, symbol: str, date_utc: str) -> Optional[float]:
        cfg = self._config()
        if not (cfg["use_sentiment"] and cfg["url"]):
            return None

        params = {
            "symbol": (symbol or "").strip().upper(),
            "date": str(date_utc).split("T", 1)[0],
        }
        headers: dict[str, str] = {}
        api_key = cfg.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-Key"] = str(api_key)

        try:
            response = self.session.get(
                str(cfg["url"]),
                params=params,
                headers=headers,
                timeout=float(cfg.get("timeout", 8.0)),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            self.errors += 1
            if self._log_count < 3:
                LOGGER.warning("Sentiment fetch failed: %s", exc)
                self._log_count += 1
            elif self._log_count == 3:
                LOGGER.warning("Further sentiment fetch errors suppressed")
                self._log_count += 1
            return None

        if not isinstance(payload, Mapping):
            self.errors += 1
            return None

        raw = payload.get("sentiment", payload.get("score"))
        score = _clamp_score(raw)
        if score is None:
            self.errors += 1
        return score


def load_sentiment_cache(cache_dir: Path | str = DEFAULT_CACHE_DIR, run_date: date | str = None) -> dict[str, float]:
    if run_date is None:
        return {}
    cache_path = Path(cache_dir) / f"{date.fromisoformat(str(run_date)).isoformat()}.json"
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
            score = _clamp_score(value)
            if score is None:
                continue
            cache[str(sym).upper()] = score
    return cache


def persist_sentiment_cache(
    cache_dir: Path | str = DEFAULT_CACHE_DIR, run_date: date | str = None, cache: Mapping[str, float] | None = None
) -> Path:
    if run_date is None:
        raise ValueError("run_date is required to persist the sentiment cache")
    run_dt = date.fromisoformat(str(run_date))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cleaned: dict[str, float] = {}
    for sym, value in (cache or {}).items():
        score = _clamp_score(value)
        if score is None:
            continue
        cleaned[str(sym).upper()] = score

    serialized = json.dumps(cleaned, indent=2, sort_keys=True).encode("utf-8")
    target = cache_dir / f"{run_dt.isoformat()}.json"
    atomic_write_bytes(target, serialized)
    return target


__all__ = [
    "JsonHttpSentimentClient",
    "DEFAULT_CACHE_DIR",
    "load_sentiment_cache",
    "persist_sentiment_cache",
]
