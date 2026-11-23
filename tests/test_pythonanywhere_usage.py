import pytest

from pathlib import Path

from scripts import pythonanywhere_usage


@pytest.mark.alpaca_optional
def test_build_payload_pythonanywhere_file_storage_percent_from_du(monkeypatch):
    used_bytes_from_du = 4329684992

    monkeypatch.setattr(pythonanywhere_usage, "pythonanywhere_cpu_usage", lambda: None)
    monkeypatch.setattr(pythonanywhere_usage, "get_pg_used_bytes_and_pretty", lambda _: (None, None))
    monkeypatch.setattr(
        pythonanywhere_usage.subprocess,
        "check_output",
        lambda *args, **kwargs: f"100\t/tmp\n{used_bytes_from_du - 100}\t/home/test\n{used_bytes_from_du}\ttotal\n",
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
    assert file_storage["captured_at"] is not None


@pytest.mark.alpaca_optional
def test_failed_disk_measurement_does_not_publish_zero_usage(monkeypatch):
    def unavailable(*args, **kwargs):
        raise pythonanywhere_usage.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(pythonanywhere_usage, "pythonanywhere_cpu_usage", lambda: None)
    monkeypatch.setattr(pythonanywhere_usage, "get_pg_used_bytes_and_pretty", lambda _: (None, None))
    monkeypatch.setattr(pythonanywhere_usage.subprocess, "check_output", unavailable)
    monkeypatch.setenv("PYTHONANYWHERE_DISK_QUOTA_GB", "10")

    storage = pythonanywhere_usage.build_payload(Path("."), "25g", None, "pythonanywhere", "database")["file_storage"]

    assert storage["limit_bytes"] == 25 * 1024**3
    assert storage["used_bytes"] is storage["percent"] is storage["captured_at"] is None


@pytest.mark.alpaca_optional
def test_pythonanywhere_file_storage_percent_rounding_and_clamp():
    ten_gib = 10 * 1024**3

    assert pythonanywhere_usage.pythonanywhere_file_storage_percent(4 * 1024**3, ten_gib) == 40
    assert pythonanywhere_usage.pythonanywhere_file_storage_percent(15 * 1024**3, ten_gib) == 100
    assert pythonanywhere_usage.pythonanywhere_file_storage_percent(-1, ten_gib) == 0
