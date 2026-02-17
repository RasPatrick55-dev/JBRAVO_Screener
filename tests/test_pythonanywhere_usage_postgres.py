from pathlib import Path

import pytest

from scripts import pythonanywhere_usage


@pytest.mark.alpaca_optional
def test_postgres_quota_and_percent_floor(monkeypatch):
    used_bytes = 428_000_000

    monkeypatch.setattr(pythonanywhere_usage, "pythonanywhere_cpu_usage", lambda: None)
    monkeypatch.setattr(
        pythonanywhere_usage,
        "get_pg_used_bytes_and_pretty",
        lambda database_url: (used_bytes, "408 MB"),
    )
    monkeypatch.setenv("PYTHONANYWHERE_POSTGRES_QUOTA_GB", "10")

    payload = pythonanywhere_usage.build_payload(
        storage_path=Path("."),
        storage_limit=None,
        postgres_limit=None,
        storage_mode="pythonanywhere",
        postgres_mode="database",
    )

    postgres = payload["postgres_storage"]
    expected_quota_bytes = 10 * 1024**3
    expected_percent = (100 * used_bytes) // expected_quota_bytes

    assert postgres["pg_quota_bytes"] == expected_quota_bytes
    assert postgres["pg_storage_percent"] == expected_percent
    assert postgres["percent"] == expected_percent


@pytest.mark.alpaca_optional
def test_postgres_fields_exist_when_query_succeeds(monkeypatch):
    monkeypatch.setattr(pythonanywhere_usage, "pythonanywhere_cpu_usage", lambda: None)
    monkeypatch.setattr(
        pythonanywhere_usage,
        "get_pg_used_bytes_and_pretty",
        lambda database_url: (512_000_000, "488 MB"),
    )

    payload = pythonanywhere_usage.build_payload(
        storage_path=Path("."),
        storage_limit=None,
        postgres_limit=None,
        storage_mode="pythonanywhere",
        postgres_mode="database",
    )

    postgres = payload["postgres_storage"]
    assert postgres["pg_used_bytes"] == 512_000_000
    assert postgres["pg_used_pretty"] == "488 MB"
    assert postgres["pg_used_gib"] is not None
    assert postgres["pg_quota_gb"] is not None
    assert postgres["pg_error"] is None


@pytest.mark.alpaca_optional
def test_postgres_failure_sets_pg_error_and_null_fields(monkeypatch):
    monkeypatch.setattr(pythonanywhere_usage, "pythonanywhere_cpu_usage", lambda: None)

    def _raise(_database_url):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(pythonanywhere_usage, "get_pg_used_bytes_and_pretty", _raise)

    payload = pythonanywhere_usage.build_payload(
        storage_path=Path("."),
        storage_limit=None,
        postgres_limit=None,
        storage_mode="pythonanywhere",
        postgres_mode="database",
    )

    postgres = payload["postgres_storage"]
    assert postgres["pg_used_bytes"] is None
    assert postgres["pg_storage_percent"] is None
    assert postgres["pg_used_pretty"] is None
    assert postgres["pg_error"] == "db unreachable"
