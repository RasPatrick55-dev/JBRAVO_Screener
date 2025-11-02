"""Environment loading helpers for CLI and service entry points."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

try:  # pragma: no cover - optional dependency
    from dotenv import dotenv_values, load_dotenv
except Exception:  # pragma: no cover - allow operation without python-dotenv
    dotenv_values = None  # type: ignore
    load_dotenv = None  # type: ignore


_REQUIRED_KEYS: tuple[tuple[str, ...], ...] = (
    ("APCA_API_KEY_ID", "ALPACA_API_KEY_ID"),
    ("APCA_API_SECRET_KEY", "ALPACA_API_SECRET_KEY"),
    ("APCA_API_BASE_URL", "ALPACA_API_BASE_URL"),
    ("APCA_DATA_API_BASE_URL", "APCA_API_DATA_URL", "ALPACA_API_DATA_URL"),
)

_REQUIRED_PRIMARY: tuple[str, ...] = (
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "APCA_API_BASE_URL",
    "APCA_DATA_API_BASE_URL",
    "ALPACA_DATA_FEED",
)

_ENV_SHIMMED: bool = False

METRICS_SUMMARY_COLUMNS: tuple[str, ...] = (
    "last_run_utc",
    "symbols_in",
    "with_bars",
    "bars_rows",
    "candidates",
    "status",
    "auth_reason",
    "auth_missing",
    "auth_hint",
)


class AlpacaCredentialsError(RuntimeError):
    """Raised when required Alpaca credentials are missing or malformed."""

    def __init__(
        self,
        reason: str,
        *,
        missing: Sequence[str] | None = None,
        whitespace: Sequence[str] | None = None,
        sanitized: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.missing = tuple(missing or ())
        self.whitespace = tuple(whitespace or ())
        self.sanitized = dict(sanitized or {})


class AlpacaUnauthorizedError(RuntimeError):
    """Raised when Alpaca responds with HTTP 401/403."""

    def __init__(self, endpoint: str, *, feed: str | None = None) -> None:
        super().__init__("Alpaca returned 401/403")
        self.endpoint = endpoint
        self.feed = feed or ""


def _normalize_apca_base_url(value: str) -> str:
    """Return ``value`` without trailing ``/v2`` or slashes."""

    trimmed = value.strip()
    if not trimmed:
        return ""
    trimmed = trimmed.rstrip("/")
    if trimmed.endswith("/v2"):
        trimmed = trimmed[: -len("/v2")]
    return trimmed.rstrip("/")


def _manual_parse_env(path: Path, *, override: bool) -> bool:
    loaded = False
    try:
        contents = path.read_text(encoding="utf-8")
    except Exception:
        return False
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith("\"") and value.endswith("\"")) or (
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
    if load_dotenv is not None:
        try:
            result = load_dotenv(path, override=override)
        except Exception:
            result = False
        if result:
            return True
    return _manual_parse_env(path, override=override)


def _normalize_env_aliases() -> None:
    for primary_key, *aliases in _REQUIRED_KEYS:
        value, source_key, _ = _resolve_env_value(primary_key, *aliases)
        if value:
            if primary_key == "APCA_API_BASE_URL":
                value = _normalize_apca_base_url(value)
            if primary_key == "APCA_DATA_API_BASE_URL":
                value = value.rstrip("/")
            os.environ[primary_key] = value
            if source_key and source_key != primary_key:
                os.environ[source_key] = value


def load_env(
    required_keys: Sequence[str] | None = None,
    *,
    override: bool = False,
) -> tuple[list[str], list[str]]:
    """Load environment files from well-known locations.

    Returns a tuple of ``(loaded_files, missing_required)`` so callers can emit
    diagnostics before proceeding. When ``python-dotenv`` is unavailable a
    minimal parser is used instead.
    """

    global _ENV_SHIMMED

    repo_root = Path(__file__).resolve().parents[1]
    user_env = Path(os.path.expanduser("~/.config/jbravo/.env"))
    repo_env = repo_root / ".env"

    loaded_files: list[str] = []
    for path in (user_env, repo_env):
        if _load_env_file(path, override=override):
            loaded_files.append(str(path))

    _normalize_env_aliases()

    if not os.environ.get("APCA_API_BASE_URL"):
        base = os.environ.get("ALPACA_API_BASE_URL")
        if base:
            os.environ["APCA_API_BASE_URL"] = _normalize_apca_base_url(base)

    if not os.environ.get("APCA_DATA_API_BASE_URL"):
        data_alias = os.environ.get("APCA_API_DATA_URL") or os.environ.get(
            "ALPACA_API_DATA_URL"
        )
        if data_alias:
            os.environ["APCA_DATA_API_BASE_URL"] = data_alias.rstrip("/")

    required = list(required_keys) if required_keys is not None else list(_REQUIRED_PRIMARY)
    missing_required = [key for key in required if not os.environ.get(key)]

    _ENV_SHIMMED = True
    return loaded_files, missing_required


def _resolve_env_value(primary: str, *aliases: str) -> tuple[str, str | None, bool]:
    """Return ``(value, source_key, had_whitespace)`` for env ``primary``/aliases."""

    had_whitespace = False
    keys = (primary, *aliases)
    for key in keys:
        if key not in os.environ:
            continue
        raw = os.environ.get(key) or ""
        trimmed = raw.strip()
        if raw != trimmed:
            os.environ[key] = trimmed
            had_whitespace = True
        if trimmed:
            if key != primary:
                os.environ[primary] = trimmed
            return trimmed, key, had_whitespace
        had_whitespace = True
    return "", None, had_whitespace


def _serialize_auth_hint(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except Exception:  # pragma: no cover - defensive fallback
        return str(value)


def write_metrics_summary_row(
    row: Mapping[str, object],
    *,
    path: str | os.PathLike[str] | None = None,
) -> None:
    """Write ``row`` to ``metrics_summary.csv`` with consistent headers."""

    summary_path = Path(path or Path("data") / "metrics_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    serializable: dict[str, object] = {}
    for column in METRICS_SUMMARY_COLUMNS:
        value = row.get(column, "") if isinstance(row, Mapping) else ""
        if column in {"symbols_in", "with_bars", "bars_rows", "candidates"}:
            try:
                value = int(value) if value not in ("", None) else 0
            except Exception:
                value = 0
        elif column == "auth_hint":
            value = _serialize_auth_hint(value)
        else:
            value = "" if value is None else value
        serializable[column] = value

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(serializable)


def write_auth_error_artifacts(
    *,
    reason: str,
    sanitized: Mapping[str, object],
    missing: Iterable[str] = (),
    metrics_path: str | os.PathLike[str],
    summary_path: str | os.PathLike[str],
) -> None:
    """Persist artifacts marking an authentication failure for dashboards."""

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, object] = {}
    if metrics_file.exists():
        try:
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing.update(payload)
        except Exception:  # pragma: no cover - defensive parsing
            existing = {}

    missing_list = sorted({str(item) for item in missing if str(item).strip()})

    existing.update(
        {
            "status": "auth_error",
            "auth_reason": reason,
            "auth_missing": missing_list,
            "auth_hint": dict(sanitized),
        }
    )
    existing["error"] = {
        "message": "auth_error",
        "reason": reason,
        "missing": missing_list,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    metrics_file.write_text(
        json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary_row = {
        "last_run_utc": existing.get("last_run_utc", ""),
        "symbols_in": existing.get("symbols_in", 0),
        "with_bars": existing.get("symbols_with_bars", 0),
        "bars_rows": existing.get("bars_rows_total", 0),
        "candidates": existing.get("rows", 0),
        "status": "auth_error",
        "auth_reason": reason,
        "auth_missing": ",".join(missing_list),
        "auth_hint": dict(sanitized),
    }
    write_metrics_summary_row(summary_row, path=summary_path)


def assert_alpaca_creds() -> dict[str, object]:
    """Validate Alpaca credentials and return a sanitized snapshot."""

    key, key_source, key_ws = _resolve_env_value("APCA_API_KEY_ID", "ALPACA_API_KEY_ID")
    secret, secret_source, secret_ws = _resolve_env_value(
        "APCA_API_SECRET_KEY", "ALPACA_API_SECRET_KEY"
    )
    trading_base, trading_source, trading_ws = _resolve_env_value(
        "APCA_API_BASE_URL", "ALPACA_API_BASE_URL"
    )
    data_base, data_source, data_ws = _resolve_env_value(
        "APCA_DATA_API_BASE_URL",
        "APCA_API_DATA_URL",
        "ALPACA_API_DATA_URL",
    )

    missing: list[str] = []
    whitespace: list[str] = []
    if key_ws:
        whitespace.append(key_source or "APCA_API_KEY_ID")
    if secret_ws:
        whitespace.append(secret_source or "APCA_API_SECRET_KEY")
    if trading_ws:
        whitespace.append(trading_source or "APCA_API_BASE_URL")
    if data_ws and data_base:
        whitespace.append(data_source or "APCA_DATA_API_BASE_URL")

    if not key:
        missing.append("APCA_API_KEY_ID")
    if not secret:
        missing.append("APCA_API_SECRET_KEY")
    if not trading_base:
        missing.append("APCA_API_BASE_URL")
    if not data_base:
        data_base = "https://data.alpaca.markets"

    sanitized = {
        "key_prefix": (key[:4] + "…") if key else "",
        "secret_len": len(secret),
        "base_urls": {
            "trading": trading_base,
            "data": data_base,
        },
    }

    if missing:
        raise AlpacaCredentialsError(
            "missing",
            missing=missing,
            whitespace=whitespace,
            sanitized=sanitized,
        )

    if whitespace:
        raise AlpacaCredentialsError(
            "whitespace",
            missing=missing,
            whitespace=whitespace,
            sanitized=sanitized,
        )

    key_prefix = key.upper()[:2]
    if key_prefix not in {"PK", "AK"}:
        raise AlpacaCredentialsError(
            "invalid_prefix",
            sanitized=sanitized,
        )

    if trading_base:
        parsed = urlparse(trading_base)
        host = parsed.netloc.lower()
        is_paper_host = "paper" in host
        if key_prefix == "PK" and not is_paper_host:
            raise AlpacaCredentialsError(
                "base-url mismatch",
                sanitized=sanitized,
            )
        if key_prefix == "AK" and is_paper_host:
            raise AlpacaCredentialsError(
                "base-url mismatch",
                sanitized=sanitized,
            )

    return sanitized


def get_alpaca_creds() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return Alpaca credentials from the environment with sensible fallbacks."""
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL") or os.getenv("ALPACA_API_BASE_URL")
    feed = os.getenv("ALPACA_DATA_FEED", "iex")
    return key, secret, base, feed


__all__ = [
    "AlpacaCredentialsError",
    "AlpacaUnauthorizedError",
    "assert_alpaca_creds",
    "load_env",
    "get_alpaca_creds",
    "write_auth_error_artifacts",
    "write_metrics_summary_row",
    "METRICS_SUMMARY_COLUMNS",
]
