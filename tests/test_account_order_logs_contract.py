"""Offline account log contract tests; no application import or external calls."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_dashboard_reporting_truthfulness import helpers

pytestmark = pytest.mark.alpaca_optional


def endpoint(db_rows=None, db_error=False, fallback=None, fallback_error=False):
    db_read = Mock(side_effect=RuntimeError("db unavailable")) if db_error else Mock(return_value=db_rows)
    file_read = Mock(side_effect=OSError("unreadable")) if fallback_error else Mock(return_value=fallback or [])
    ns = helpers("_account_order_logs", "api_account_order_logs",
        _account_order_logs_from_db=db_read, _account_order_logs_from_file=file_read,
        _parse_positive_int=lambda *a, **kw: 100,
        request=SimpleNamespace(args={}))
    return ns["api_account_order_logs"](), db_read, file_read


def test_successful_empty_is_not_failure_and_does_not_fall_back():
    result, _, fallback = endpoint(db_rows=[])
    assert result == {"ok": True, "status": "empty", "rows": [], "source_detail": "postgres:order_events"}
    fallback.assert_not_called()


def test_successful_rows_are_ready():
    result, _, fallback = endpoint(db_rows=[{"ts": "2030-01-02T10:00:00Z", "message": "BUY_FILL ABC", "level": "success"}])
    assert result["ok"] is True and result["status"] == "ready"
    assert len(result["rows"]) == 1
    fallback.assert_not_called()


@pytest.mark.parametrize("file_error", [False, True])
def test_database_failure_and_no_verified_fallback_is_unavailable(file_error):
    result, _, _ = endpoint(db_error=True, fallback_error=file_error)
    assert result["ok"] is False and result["status"] == "unavailable"
    assert result["rows"] == []


def test_database_failure_with_log_rows_is_explicitly_partial():
    result, _, _ = endpoint(db_error=True, fallback=[{"ts": "2030-01-02T10:00:00Z", "message": "BUY_FILL ABC"}])
    assert result["ok"] is True and result["status"] == "partial"
    assert result["source_detail"] == "log-fallback:execute_trades.log"


def test_database_reader_uses_error_preserving_read_and_current_day_scope():
    read = Mock(return_value=[])
    ns = helpers("_account_order_logs_from_db", _db_fetch_all_strict=read)
    assert ns["_account_order_logs_from_db"](100) == []
    sql, params = read.call_args.args
    assert "event_time >=" in sql
    assert params["start_utc"].hour == 0 and params["limit"] == 100


def test_database_read_errors_propagate():
    ns = helpers("_account_order_logs_from_db", _db_fetch_all_strict=Mock(side_effect=RuntimeError("timeout")))
    with pytest.raises(RuntimeError, match="timeout"):
        ns["_account_order_logs_from_db"](100)


def test_malformed_rows_do_not_become_verified_empty():
    ns = helpers("_account_order_logs_from_db", _db_fetch_all_strict=Mock(return_value=[{"event_time": "invalid"}]), _coerce_datetime_utc=lambda _: None)
    with pytest.raises(ValueError, match="invalid_order_event_timestamp"):
        ns["_account_order_logs_from_db"](100)


def test_malformed_row_after_valid_row_does_not_publish_complete_subset():
    valid = datetime(2030, 1, 2, 10, tzinfo=timezone.utc)
    ns = helpers("_account_order_logs_from_db",
        _db_fetch_all_strict=Mock(return_value=[{"event_time": valid}, {"event_time": "invalid"}]),
        _coerce_datetime_utc=lambda value: value if isinstance(value, datetime) else None,
        _coerce_json_mapping=lambda _: {}, _order_event_message=lambda *args: "BUY_FILL ABC",
        _order_log_level=lambda *args: "success")
    with pytest.raises(ValueError, match="invalid_order_event_timestamp"):
        ns["_account_order_logs_from_db"](100)
