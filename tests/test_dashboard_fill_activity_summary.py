"""Synthetic, offline FILL-activity window/pagination regression tests.

The dashboard is never imported, no credentials are loaded, and no broker or
database is queried. Facts are fictional supplied data, not private captures.
"""

from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_dashboard_reporting_truthfulness import NOW, helpers


pytestmark = pytest.mark.alpaca_optional
SINCE = NOW - timedelta(hours=24)
NAMES = (
    "_execute_fill_activity_fact",
    "_execute_buy_fill_activity_summary",
    "api_execute_orders_summary",
)


def namespace(**overrides):
    return helpers(*NAMES, Decimal=Decimal, time=SimpleNamespace(monotonic=lambda: 0), **overrides)


def fill(
    identifier="fictional-fill-1",
    *,
    instant=NOW - timedelta(hours=1),
    side="buy",
    qty="2",
    price="3.25",
    **extra,
):
    return dict(
        id=identifier,
        activity_type="FILL",
        type="fill",
        transaction_time=instant.isoformat(),
        side=side,
        qty=qty,
        price=price,
        order_id="fictional-old-gtc",
        symbol="FICT",
        **extra,
    )


def run(pages, **options):
    read = Mock(
        side_effect=[(page, "ok") if not isinstance(page, Exception) else page for page in pages]
    )
    ns = namespace()
    result = ns["_execute_buy_fill_activity_summary"](
        since=SINCE, until=NOW, read_page=read, **options
    )
    return result, read


def assert_unknown(result, reason=None):
    assert result["ok"] is False and result["coverage"] == "incomplete"
    assert result["fill_count"] is result["total_value"] is result["orders_filled"] is None
    assert result["status"] in {"partial", "unavailable"}
    if reason:
        assert result["reason"] == reason


def test_old_gtc_filled_today_and_open_partial_fills_are_executions_not_orders():
    first = fill(
        submitted_at=(NOW - timedelta(days=20)).isoformat(),
        cum_qty="200",
        order_status="partially_filled",
    )
    first["type"] = "partial_fill"
    second = fill("fictional-fill-2", qty="0.5", price="4", cum_qty="200.5", order_status="filled")
    # Same order has two executions; a sell is not a buy-fill event.
    result, read = run([[first, second, fill("fictional-sell", side="sell", qty="99")], []])
    assert result["ok"] and result["status"] == "ready"
    assert result["fill_count"] == result["observed_fill_count"] == 2
    assert result["total_value"] == 8.5  # per-execution qty, never cumulative qty
    assert result["count_unit"] == "buy_fill_activity_events" and result["orders_filled"] is None
    assert result["coverage"] == "requested_24h_fill_activity_window"
    assert (
        result["coverage_scope"] == "observed_broker_fill_activities"
        and result["accounting_final"] is False
    )
    assert read.call_args_list[0].args == ("/v2/account/activities/FILL",)
    assert read.call_args_list[1].kwargs["params"]["page_token"] == "fictional-sell"


@pytest.mark.parametrize("pages", [[[]], [[fill(side="sell")], []]])
def test_genuinely_empty_buy_window_is_zero_only_after_explicit_empty_page(pages):
    result, _ = run(pages)
    assert result["ok"] and result["status"] == "empty"
    assert result["fill_count"] == result["total_value"] == 0


def test_exact_utc_execution_window_filters_padding_future_and_old_events():
    rows = [
        fill("old", instant=SINCE - timedelta(microseconds=1)),
        fill("since", instant=SINCE),
        fill("near-end", instant=NOW - timedelta(microseconds=1)),
        fill("until", instant=NOW),
        fill("future", instant=NOW + timedelta(hours=1)),
    ]
    result, read = run([rows, []])
    assert result["fill_count"] == 2 and result["total_value"] == 13
    assert result["excluded_outside_window"] == 3
    assert result["window_boundary"] == "since_inclusive_until_exclusive"
    query = read.call_args_list[0].kwargs["params"]
    assert query == dict(
        after=(SINCE - timedelta(seconds=1)).isoformat(),
        until=(NOW + timedelta(seconds=1)).isoformat(),
        direction="asc",
        page_size=100,
    )


def test_offset_timestamp_is_normalized_without_falling_back_to_submission_time():
    activity = fill()
    activity["transaction_time"] = "2030-01-02T06:00:00-05:00"
    result, _ = run([[activity], []])
    assert result["ok"] and result["fill_count"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("transaction_time", None),
        ("transaction_time", "not-a-time"),
        ("transaction_time", "2030-01-02T11:00:00"),
        ("transaction_time", "2030-01-02"),
        ("side", "unknown"),
        ("side", None),
        ("qty", 0),
        ("qty", "-1"),
        ("qty", True),
        ("qty", "NaN"),
        ("price", "Infinity"),
        ("price", "garbage"),
        ("price", None),
        ("id", None),
        ("id", ""),
        ("activity_type", "DIV"),
        ("type", "unverified"),
    ],
)
def test_invalid_activity_never_becomes_zero_or_complete_totals(field, value):
    row = fill()
    row[field] = value
    result, _ = run([[row], []])
    assert_unknown(result)


@pytest.mark.parametrize("page", [None, {}, {"activities": []}, "[]", ["bad-row"]])
def test_invalid_page_shape_is_not_empty(page):
    result, _ = run([page])
    assert_unknown(result)


def test_pagination_uses_last_id_even_after_short_pages_and_stops_only_on_empty():
    first = [fill("a"), fill("b")]
    second = [fill("c")]
    third = [fill("d")]
    result, read = run([first, second, third, []], page_size=2)
    assert result["ok"] and result["fill_count"] == 4 and result["pages_read"] == 4
    assert [call.kwargs["params"].get("page_token") for call in read.call_args_list] == [
        None,
        "b",
        "c",
        "d",
    ]
    queries = [call.kwargs["params"] for call in read.call_args_list]
    for query in queries:
        assert {key: value for key, value in query.items() if key != "page_token"} == queries[0]
    assert all(0 < call.kwargs["timeout"] <= 3 for call in read.call_args_list)


def test_identical_duplicate_ids_are_counted_once_across_pages():
    a, b = fill("a"), fill("b")
    result, _ = run([[a], [deepcopy(a), b], []])
    assert result["ok"] and result["fill_count"] == 2 and result["total_value"] == 13
    assert result["duplicate_activities"] == 1


def test_duplicate_id_with_changed_facts_fails_closed():
    result, _ = run([[fill("a")], [fill("a", price="4")], []])
    assert_unknown(result, "conflicting_fill_activity_id")


def test_repeated_page_cursor_fails_instead_of_looping_or_claiming_complete():
    page = [fill("a"), fill("b")]
    result, read = run([page, deepcopy(page), []])
    assert_unknown(result, "fill_activity_pagination_not_advancing")
    assert read.call_count == 2 and result["observed_fill_count"] == 2


@pytest.mark.parametrize("page", [[fill("a")], [fill("a"), fill("b")]])
def test_short_and_full_page_saturation_both_remain_unknown(page):
    result, read = run([page, []], max_pages=1, page_size=2)
    assert_unknown(result, "fill_activity_page_limit")
    assert read.call_count == 1


def test_failed_pagination_preserves_only_observed_lower_bound_not_headline_totals():
    result, _ = run([[fill()], RuntimeError("synthetic failure")])
    assert_unknown(result, "fill_activity_read_failed")
    assert result["observed_fill_count"] == 1


def test_http_failure_is_unavailable_not_empty():
    ns = namespace()
    result = ns["_execute_buy_fill_activity_summary"](
        since=SINCE, until=NOW, read_page=lambda *a, **kw: ({}, "http_429")
    )
    assert_unknown(result, "fill_activity_read_failed")


def test_deadline_and_invalid_nonfinite_notional_fail_closed():
    times = iter([0, 0, 9])
    result, _ = run([[fill()]], max_seconds=8, clock=lambda: next(times))
    assert_unknown(result, "fill_activity_deadline")
    result, _ = run([[fill(qty="1e308", price="1e308")], []])
    assert_unknown(result, "invalid_fill_activity_notional")
    result, _ = run([[fill(qty="1e-308", price="1e-308")], []])
    assert_unknown(result, "invalid_fill_activity_notional")


def test_route_never_requests_orders_and_uses_existing_paper_guard():
    rest = Mock(side_effect=[([fill()], "ok"), ([], "ok")])
    orders = Mock(side_effect=AssertionError("Submission-window order query is prohibited"))
    ns = namespace(
        _dashboard_broker_context=lambda: (
            SimpleNamespace(get_orders=orders),
            "configured_paper_broker",
        ),
        _alpaca_rest_get_json=rest,
        source_contracts=SimpleNamespace(bounded_read=lambda fn, seconds: fn()),
    )
    result = ns["api_execute_orders_summary"]()
    assert result["ok"] and result["fill_count"] == 1 and result["source"] == "alpaca"
    assert result["since_utc"] == SINCE.isoformat() and result["until_utc"] == NOW.isoformat()
    orders.assert_not_called()
    rest.assert_called()
    rest.reset_mock()
    ns["_dashboard_broker_context"] = lambda: (None, "broker_environment_mismatch")
    assert_unknown(ns["api_execute_orders_summary"](), "broker_environment_mismatch")
    rest.assert_not_called()


def test_route_bounded_read_timeout_cannot_publish_sample_as_complete():
    bound = Mock(side_effect=RuntimeError("source_timeout"))
    ns = namespace(
        _dashboard_broker_context=lambda: (object(), "configured_paper_broker"),
        source_contracts=SimpleNamespace(bounded_read=bound),
    )
    assert_unknown(ns["api_execute_orders_summary"](), "fill_activity_read_failed")
    assert bound.call_args.kwargs["seconds"] == 8.5
