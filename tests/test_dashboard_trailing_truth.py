"""Offline supplied-observation regressions; no broker, database or app imports."""
from datetime import timedelta
from unittest.mock import Mock

import pytest
from test_dashboard_reporting_truthfulness import helpers, STATUS_HELPERS, NOW


def payload(db, broker, detail="alpaca:/v2/orders", scope="all"):
    reader = Mock(return_value=(broker, detail))
    ns = helpers(
        "_execute_trailing_stops_payload", "_execute_filter_trailing_rows",
        *STATUS_HELPERS, "_execute_iso_utc", "_execute_lsx_matches", "_execute_query_matches",
        _execute_trailing_from_db=Mock(return_value=(db, "postgres:order_events")),
        _execute_trailing_from_alpaca=reader,
    )
    result = ns["_execute_trailing_stops_payload"](status_scope=scope, limit=200, query="", lsx="all")
    reader.assert_called_once_with(fetch_limit=500, status_scope="all", include_lifecycle=True)
    return result


def row(order_id="CRBG-order", status="ACTIVE", days=1, source="postgres_history", **extra):
    return dict(order_id=order_id, symbol="CRBG", status=status, raw_status=status,
                _ts=NOW-timedelta(days=days), observation_source=source,
                _is_trailing=True, trail="3%", qty=1, **extra)


def test_23_later_terminal_observations_override_only_matching_ids():
    history = [row(str(n), raw="unused") for n in range(23)]
    broker = [row(str(n), "CANCELED" if n < 22 else "FILLED", 0, "broker_response") for n in range(23)]
    result = payload(history, broker)
    assert len(result["rows"]) == 23
    assert sum(r["status"] == "CANCELED" for r in result["rows"]) == 22
    assert sum(r["status"] == "FILLED" for r in result["rows"]) == 1
    assert all(r["current_protection"] == "unverified" for r in result["rows"])
    assert all(r["observation_source"] == "broker_response" for r in result["rows"])
    assert all(r["raw_status"] == r["status"] for r in result["rows"])


@pytest.mark.parametrize("broker,detail", [([], "alpaca:/v2/orders"), ([], "alpaca:timeout"),
    ([row("different-same-symbol", "CANCELED", 0, "broker_response")], "alpaca:/v2/orders")])
def test_missing_bounded_page_never_implies_cancellation_or_protection(broker, detail):
    result = payload([row()], broker, detail)
    historical = next(r for r in result["rows"] if r["order_id"] == "CRBG-order")
    assert historical["status"] == "ACTIVE"
    assert historical["observation_source"] == "postgres_history"
    assert historical["current_protection"] == "unverified"
    assert result["coverage"] == "bounded_page_not_complete_inventory"
    assert result["broker_page_limit"] == 500


def test_latest_duplicate_broker_timestamp_wins_and_older_broker_cannot_regress_db():
    result = payload([row(status="CANCELED", days=0)], [row(days=2, source="broker_response")])
    assert result["rows"][0]["status"] == "CANCELED"
    newer = row(status="FILLED", days=0, source="broker_response")
    older = row(days=2, source="broker_response")
    for observations in ([newer, older], [older, newer]):
        result = payload([], observations)
        assert len(result["rows"]) == 1 and result["rows"][0]["status"] == "FILLED"


def test_replacement_chain_preserved_without_claiming_protection():
    old = row("ARDT-old", "PENDING_REPLACE", 0, "broker_response", replaced_by="ARDT-new")
    new = row("ARDT-new", "ACTIVE", 0, "broker_response", replaces="ARDT-old")
    result = payload([row("ARDT-old")], [old, new], scope="open")
    by_id = {r["order_id"]: r for r in result["rows"]}
    assert by_id["ARDT-old"]["replaced_by"] == "ARDT-new"
    assert by_id["ARDT-new"]["replaces"] == "ARDT-old"
    assert all(r["current_protection"] == "unverified" for r in result["rows"])
    assert payload([row()], [row(status="CANCELED", days=0)], scope="open")["rows"] == []


def test_db_latest_lifecycle_status_and_terms_are_separate_and_not_raw_new():
    record = dict(order_id="CRBG-order", symbol="CRBG", status="OrderStatus.CANCELED",
        event_type="SELL_CANCELLED", event_time=NOW,
        raw={"status":"OrderStatus.NEW", "trail_percent":3},
        lifecycle_raw={"status":"OrderStatus.CANCELED", "replaced_by":None},
        terms_observed_at=NOW-timedelta(days=1))
    fetch = Mock(return_value=[record])
    ns = helpers("_execute_trailing_from_db", *STATUS_HELPERS, "_execute_iso_utc", "_coerce_json_mapping",
        "_execute_side_label", _db_fetch_all_strict=fetch, _execute_trail_text=lambda r:"3%",
        _execute_stop_price_from_raw=lambda r:None, _execute_parent_leg=lambda r,i:i)
    result, _ = ns["_execute_trailing_from_db"](500)
    assert result[0]["status"] == "CANCELED" and result[0]["raw_status"] == "OrderStatus.CANCELED"
    assert result[0]["terms_observed_at"] != NOW.isoformat()
    record.update(status=None, event_type="SELL_CANCELLED", lifecycle_raw={})
    assert ns["_execute_trailing_from_db"](500)[0][0]["status"] == "CANCELED"
    sql = fetch.call_args.args[0]
    assert "LEFT JOIN LATERAL" in sql and "e.order_id = t.order_id" in sql
    assert "event_id DESC" in sql and "THEN latest.status ELSE t.status" in sql
    assert "? 'stop_price'" not in sql  # ordinary fixed stops are not trailing stops


def test_successful_empty_is_not_failure_but_unavailable_is():
    assert payload([], [])["ok"] is True
    assert payload([], [], "alpaca:timeout")["ok"] is False

@pytest.mark.parametrize("raw_status,expected", [(None,"UNKNOWN"), ("OrderStatus.NEW","PENDING"), ("pending_new","PENDING"), ("pending_replace","PENDING_REPLACE")])
def test_broker_missing_or_pending_status_never_becomes_active(raw_status, expected):
    observation = dict(id="ARDT-order", symbol="ARDT", type="trailing_stop", status=raw_status,
        updated_at=NOW.isoformat(), trail_percent=3)
    ns = helpers("_execute_trailing_from_alpaca", *STATUS_HELPERS, "_execute_iso_utc",
        "_execute_type_label", "_execute_side_label",
        _alpaca_rest_get_json=Mock(return_value=([observation], "ok")),
        _execute_trail_text=lambda r:"3%", _execute_stop_price_from_raw=lambda r:None,
        _execute_parent_leg=lambda r,i:i)
    result, _ = ns["_execute_trailing_from_alpaca"](fetch_limit=500, status_scope="all", include_lifecycle=True)
    assert result[0]["status"] == expected


def test_equal_timestamp_conflicts_are_unknown_in_both_arrival_orders():
    active = row(status="PENDING", days=0, source="broker_response")
    canceled = row(status="CANCELED", days=0, source="broker_response")
    for observations in ([active, canceled], [canceled, active]):
        result = payload([], observations)
        assert result["rows"][0]["status"] == "UNKNOWN"
        assert result["rows"][0]["raw_status"] == "conflicting_equal_time_observations"


def test_supplied_crbg_live_contradiction_exact_identity_and_times():
    from datetime import datetime
    order_id = "47dbfe65-22c6-46dc-8cdc-63a0c8510ed1"
    historical = row(order_id)
    historical.update(raw_status="OrderStatus.NEW",
        _ts=datetime.fromisoformat("2026-10-14T13:33:58+00:00"))
    broker = row(order_id, status="CANCELED", source="broker_response")
    broker.update(raw_status="OrderStatus.CANCELED",
        _ts=datetime.fromisoformat("2026-10-14T13:35:03+00:00"))
    result = payload([historical], [broker])["rows"][0]
    assert result["order_id"] == order_id and result["status"] == "CANCELED"
    assert result["ts_utc"] == "2026-10-14T13:35:03+00:00"
    assert result["raw_status"] == "OrderStatus.CANCELED"
    assert result["current_protection"] == "unverified"
    # A truncated page that omits CRBG cannot prove its cancellation.
    unmatched = payload([historical], [])["rows"][0]
    assert unmatched["observation_source"] == "postgres_history"
    assert unmatched["raw_status"] == "OrderStatus.NEW"


def test_database_timeout_is_unavailable_not_successful_empty():
    ns = helpers("_execute_trailing_from_db",
        _db_fetch_all_strict=Mock(side_effect=RuntimeError("statement timeout")))
    assert ns["_execute_trailing_from_db"](500) == ([], "postgres:order_events_unavailable")
