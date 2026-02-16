"""Lightweight file rotation helpers for long-running script logs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def rotate_if_needed(
    path: str,
    *,
    max_bytes: int = 10_000_000,
    max_age_days: int = 14,
    keep: int = 14,
) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        return {"rotated": False, "reason": "missing"}

    stat = target.stat()
    size_bytes = stat.st_size
    age_days = int((datetime.now(timezone.utc).timestamp() - stat.st_mtime) // 86_400)

    reason: str | None = None
    if size_bytes > max_bytes:
        reason = "size"
    elif age_days >= max_age_days:
        reason = "age"

    if reason is None:
        return {
            "rotated": False,
            "reason": "ok",
            "size_bytes": size_bytes,
            "age_days": age_days,
        }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = target.with_name(f"{target.name}.{timestamp}.bak")
    target.rename(backup)
    target.write_text("", encoding="utf-8")

    backups = sorted(
        target.parent.glob(f"{target.name}.*.bak"),
        key=lambda item: item.name,
        reverse=True,
    )
    for stale in backups[keep:]:
        stale.unlink(missing_ok=True)

    print(
        "LOG_ROTATE "
        f"path={target} rotated=true reason={reason} backup={backup} "
        f"size_bytes={size_bytes} age_days={age_days} keep={keep}"
    )
    return {
        "rotated": True,
        "reason": reason,
        "path": str(target),
        "backup": str(backup),
        "size_bytes": size_bytes,
        "age_days": age_days,
        "keep": keep,
    }

