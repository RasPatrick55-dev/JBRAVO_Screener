from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from scripts.log_rotate import rotate_if_needed

pytestmark = pytest.mark.alpaca_optional


def _backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.*.bak"), key=lambda p: p.name)


def test_rotate_by_size(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("x" * 64, encoding="utf-8")

    result = rotate_if_needed(str(log_path), max_bytes=16, max_age_days=99, keep=14)

    assert result["rotated"] is True
    assert result["reason"] == "size"
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8") == ""
    assert len(_backups(log_path)) == 1


def test_rotate_by_age(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "execute_trades.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("old", encoding="utf-8")

    old_ts = time.time() - (20 * 86_400)
    os.utime(log_path, (old_ts, old_ts))

    result = rotate_if_needed(str(log_path), max_bytes=10_000_000, max_age_days=14, keep=14)

    assert result["rotated"] is True
    assert result["reason"] == "age"
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8") == ""
    assert len(_backups(log_path)) == 1


def test_rotate_retention_keep_two(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(4):
        log_path.write_text("x" * 64, encoding="utf-8")
        rotate_if_needed(str(log_path), max_bytes=16, max_age_days=99, keep=2)
        time.sleep(1.05)

    backups = _backups(log_path)
    assert len(backups) == 2
    assert backups[0].name < backups[1].name
