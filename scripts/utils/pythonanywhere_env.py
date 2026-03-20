from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

SUPPORTED_PA_HOSTS: tuple[str, ...] = ("www.pythonanywhere.com", "eu.pythonanywhere.com")
PA_REQUIRED_KEYS: tuple[str, ...] = ("PA_USERNAME", "PA_TOKEN", "PA_HOST")
REPO_LOCAL_ENV_FILENAMES: tuple[str, ...] = (".env.pythonanywhere.local", ".env.local")


@dataclass(frozen=True)
class PythonAnywhereResolution:
    values: dict[str, str]
    value_sources: dict[str, str]
    checked_locations: list[str]
    missing_keys: list[str]
    validation_errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_keys and not self.validation_errors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def user_env_path() -> Path:
    return Path.home() / ".config" / "jbravo" / ".env"


def repo_local_env_paths(repo_root: Path | None = None) -> list[Path]:
    root = (repo_root or _repo_root()).resolve()
    return [root / name for name in REPO_LOCAL_ENV_FILENAMES]


def supported_lookup_locations(repo_root: Path | None = None) -> list[str]:
    root = (repo_root or _repo_root()).resolve()
    locations = [
        "process environment (PA_USERNAME, PA_TOKEN, PA_HOST)",
        str(user_env_path()),
    ]
    locations.extend(str(root / name) for name in REPO_LOCAL_ENV_FILENAMES)
    return locations


def mask_secret(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "<missing>"
    if len(raw) <= 4:
        return "*" * len(raw)
    if len(raw) <= 8:
        return f"{raw[:1]}{'*' * (len(raw) - 2)}{raw[-1:]}"
    return f"{raw[:4]}{'*' * max(len(raw) - 8, 3)}{raw[-4:]}"


def _parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        contents = path.read_text(encoding="utf-8")
    except Exception:
        return data
    for raw_line in contents.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        data[key] = value
    return data


def _merge_values(
    target: dict[str, str],
    source_map: dict[str, str],
    source_name: str,
    value_sources: dict[str, str],
) -> None:
    for key in PA_REQUIRED_KEYS:
        value = source_map.get(key)
        if value is None:
            continue
        stripped = str(value).strip()
        if not stripped:
            continue
        if key not in target:
            target[key] = stripped
            value_sources[key] = source_name


def _validate(values: Mapping[str, str]) -> tuple[list[str], list[str]]:
    missing = [key for key in PA_REQUIRED_KEYS if not str(values.get(key, "")).strip()]
    errors: list[str] = []
    host = str(values.get("PA_HOST", "")).strip()
    if host and host not in SUPPORTED_PA_HOSTS:
        errors.append(
            "PA_HOST must be one of "
            f"{', '.join(SUPPORTED_PA_HOSTS)} (found {host!r})"
        )
    return missing, errors


def resolution_error_message(
    resolution: PythonAnywhereResolution, repo_root: Path | None = None
) -> str:
    details: list[str] = []
    if resolution.missing_keys:
        details.append(f"missing={','.join(resolution.missing_keys)}")
    if resolution.validation_errors:
        details.append(f"errors={'; '.join(resolution.validation_errors)}")
    supported = "; ".join(supported_lookup_locations(repo_root))
    suffix = " ".join(details).strip()
    return (
        "PYTHONANYWHERE_ENV_INVALID "
        f"{suffix} Supported lookup locations: {supported}"
    ).strip()


def resolve_pythonanywhere_credentials(
    repo_root: Path | None = None,
) -> PythonAnywhereResolution:
    root = (repo_root or _repo_root()).resolve()
    resolved: dict[str, str] = {}
    value_sources: dict[str, str] = {}
    checked_locations: list[str] = []

    env_values = {key: str(os.environ.get(key, "")).strip() for key in PA_REQUIRED_KEYS}
    checked_locations.append("process environment (PA_USERNAME, PA_TOKEN, PA_HOST)")
    _merge_values(resolved, env_values, "process environment", value_sources)

    user_file = user_env_path()
    checked_locations.append(str(user_file))
    if user_file.exists():
        _merge_values(
            resolved,
            _parse_env_file(user_file),
            f"env file:{user_file}",
            value_sources,
        )

    for candidate in repo_local_env_paths(root):
        checked_locations.append(str(candidate))
        if candidate.exists():
            _merge_values(
                resolved,
                _parse_env_file(candidate),
                f"env file:{candidate}",
                value_sources,
            )

    missing, errors = _validate(resolved)
    return PythonAnywhereResolution(
        values=resolved,
        value_sources=value_sources,
        checked_locations=checked_locations,
        missing_keys=missing,
        validation_errors=errors,
    )


__all__ = [
    "PA_REQUIRED_KEYS",
    "REPO_LOCAL_ENV_FILENAMES",
    "SUPPORTED_PA_HOSTS",
    "PythonAnywhereResolution",
    "mask_secret",
    "repo_local_env_paths",
    "resolution_error_message",
    "resolve_pythonanywhere_credentials",
    "supported_lookup_locations",
    "user_env_path",
]
