import pytest

from pathlib import Path

from scripts import pythonanywhere_usage


@pytest.mark.alpaca_optional
def test_build_payload_pythonanywhere_file_storage_percent_from_du(monkeypatch):
    used_bytes_from_du = 4329684992

    monkeypatch.setattr(pythonanywhere_usage, "pythonanywhere_cpu_usage", lambda: None)
    monkeypatch.setattr(pythonanywhere_usage, "postgres_size_bytes", lambda mode: None)
    monkeypatch.setattr(
        pythonanywhere_usage.subprocess,
        "check_output",
        lambda *args, **kwargs: str(used_bytes_from_du),
    )
    monkeypatch.setenv("PYTHONANYWHERE_DISK_QUOTA_GB", "10")

    payload = pythonanywhere_usage.build_payload(
        storage_path=Path("."),
        storage_limit=None,
        postgres_limit=None,
        storage_mode="pythonanywhere",
        postgres_mode="database",
    )

    file_storage = payload["file_storage"]
    assert file_storage["file_used_bytes"] == used_bytes_from_du
    assert file_storage["file_quota_bytes"] == 10 * 1024**3
    assert file_storage["file_storage_percent"] == 40
    assert file_storage["percent"] == 40


@pytest.mark.alpaca_optional
def test_pythonanywhere_file_storage_percent_rounding_and_clamp():
    ten_gib = 10 * 1024**3

    assert pythonanywhere_usage.pythonanywhere_file_storage_percent(4 * 1024**3, ten_gib) == 40
    assert pythonanywhere_usage.pythonanywhere_file_storage_percent(15 * 1024**3, ten_gib) == 100
    assert pythonanywhere_usage.pythonanywhere_file_storage_percent(-1, ten_gib) == 0
