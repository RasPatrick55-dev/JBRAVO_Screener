"""Offline timestamp/receipt regressions for PythonAnywhere monitor log output."""

from datetime import datetime, timedelta, timezone
import json

import pytest

from dashboards.log_reporting import annotate_log_text, event_time, log_receipt

pytestmark = pytest.mark.alpaca_optional

MONITOR_SOURCE = "python /home/RasPatrick/strategy_monitor_boot.py"
MONITOR_LINE = (
    'Nov8 19:43:42 {"status":"blocked",'
    '"finished_at":"2026-11-08T19:43:42.729306+00:00"}'
)
FINISHED = datetime(2026, 11, 8, 19, 43, 42, 729306, tzinfo=timezone.utc)


def test_actual_monitor_output_uses_json_completion_time():
    result = annotate_log_text(MONITOR_LINE, MONITOR_SOURCE)
    row = result["entries"][0]
    assert row["raw"] == MONITOR_LINE
    assert row["ts_utc"] == FINISHED.isoformat()
    assert row["provenance"] == "unclassified"
    assert "not broker evidence" in row["provenance_label"]
    assert result["unknown_timestamp_count"] == 0
    assert result["latest_event_at"] == FINISHED.isoformat()
    assert "JSON finished_at requires an explicit offset" in result["timestamp_basis"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-11-08T19:43:42.729306+00:00", FINISHED),
        ("2026-11-08T19:43:42.729306Z", FINISHED),
        ("2026-11-09T01:13:42.729306+05:30", FINISHED),
        ("2026-11-08T13:58:42.729306-05:45", FINISHED),
        ("2026-11-08T19:43:42Z", FINISHED.replace(microsecond=0)),
        ("2026-11-08T19:43:42.7Z", FINISHED.replace(microsecond=700000)),
        ("2026-11-08T19:43:42,729306+00:00", FINISHED),
    ],
)
@pytest.mark.parametrize("prefix", ["", "Nov8 19:43:42 ", "Nov  8 19:43:42 "])
def test_json_offsets_and_fractions_are_parsed_exactly(value, expected, prefix):
    line = prefix + json.dumps({"status": "blocked", "finished_at": value})
    assert event_time(line) == expected


@pytest.mark.parametrize(
    "value",
    [
        "2026-02-30T19:43:42Z",
        "2026-13-08T19:43:42Z",
        "2026-11-32T19:43:42Z",
        "2026-11-08T24:00:00Z",
        "2026-11-08T19:60:42Z",
        "2026-11-08T19:43:60Z",
        "2026-11-08T19:43:420Z",
        "2026-11-08T19:43:42.Z",
        "2026-11-08T19:43:42+24:00",
        "2026-11-08T19:43:42-24:00",
        "2026-11-08T19:43:42+00:60",
        "2026-11-08T19:43:42-00:60",
        "2026-11-08T19:43:42+05:300",
        "2026-11-08T19:43:42+0530",
        "2026-11-08T19:43:42+5:30",
        "2026-11-08T19:43:42+00:00:01",
        "2026-11-08T19:43:42Zbad",
        "2026-11-08T19:43:42+bad",
        "2026-11-08T19:43:42+00:00suffix",
        "2026-11-08T19:43:42Z+00:00",
        "0001-01-01T00:00:00+00:01",  # UTC conversion underflows.
        "9999-12-31T23:59:59-00:01",  # UTC conversion overflows.
    ],
)
def test_malformed_timestamp_cannot_be_accepted_by_prefix(value):
    assert event_time(value) is None
    assert event_time("[" + value + "] INFO blocked") is None
    assert event_time("Nov8 19:43:42 " + json.dumps({"finished_at": value})) is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        1794167022,
        True,
        {},
        ["2026-11-08T19:43:42Z"],
        "",
        "Nov8 19:43:42",
        "11-08T19:43:42Z",
        "2026-11-08",
        "2026-11-08T19:43:42",
        "2026-11-08 19:43:42.729306",
        "2026-11-08T19:43:42 UTC",
        "2026-11-08T19:43:42 America/New_York",
        " 2026-11-08T19:43:42Z",
        "2026-11-08T19:43:42Z ",
        "2026-11-08T19:43:42Z trailing text",
    ],
)
def test_json_requires_complete_explicit_year_and_offset(value):
    assert event_time("Nov8 19:43:42 " + json.dumps({"finished_at": value})) is None


@pytest.mark.parametrize(
    "line",
    [
        "Nov8 19:43:42 blocked",
        "Nov8 19:43:42 2026-11-08T19:43:42Z",
        "Read data dated 2026-11-08T19:43:42Z",
        "/logs/2026-11-08T19:43:42Z",
        '{"started_at":"2026-11-08T19:43:42Z"}',
        '{"market_data_at":"2026-11-08T19:43:42Z"}',
        '{"message":"received data at 2026-11-08T19:43:42Z"}',
        '{"receipt":{"finished_at":"2026-11-08T19:43:42Z"}}',
        '{"finished_at":null,"started_at":"2026-11-08T19:43:42Z"}',
        '{"finished_at":"invalid","market_data_at":"2026-11-08T19:43:42Z"}',
        '{"finished_at":"2026-11-08T19:43:42Z"',
        '{"finished_at":"2026-11-08T19:43:42Z",}',
        '{"finished_at":"2026-11-08T19:43:42Z"} unrelated trailer',
        '{"finished_at":"2026-11-08T19:43:42Z","finished_at":null}',
    ],
)
def test_unrelated_fields_and_ambiguous_or_missing_times_remain_unknown(line):
    assert event_time(line) is None
    rows = annotate_log_text(line, MONITOR_SOURCE)["entries"]
    receipt = log_receipt(rows, FINISHED)
    assert receipt["unknown_timestamp_count"] == 1
    assert receipt["freshness"] == "unknown"
    assert receipt["latest_event_at"] is None


def test_legacy_leading_log_timestamp_convention_is_preserved():
    assert event_time("[2026-11-08 19:43:42,729306] [INFO]: blocked") == FINISHED
    assert event_time("2026-11-08 19:43:42 -- Started task") == FINISHED.replace(microsecond=0)
    assert event_time("2026-11-08T13:58:42.729306-05:45 [INFO]: blocked") == FINISHED
    # The body is not a substitute for an explicit event timestamp at the start.
    assert event_time('2026-10-25 00:00:00 {"finished_at":"2026-11-08T19:43:42Z"}') == (
        datetime(2026, 10, 25, tzinfo=timezone.utc)
    )


def test_monitor_rows_preserve_order_and_receipt_uses_latest_event_not_fetch_time():
    lines = [
        MONITOR_LINE,
        'Nov8 19:43:42 {"status":"blocked","finished_at":"2026-10-25T10:00:00Z"}',
        "Nov8 19:43:42 missing completion timestamp",
        'Nov8 19:43:42 {"finished_at":"2026-11-08T13:58:41.729306-05:45"}',
    ]
    result = annotate_log_text("\n".join(lines), MONITOR_SOURCE)
    assert [row["raw"] for row in result["entries"]] == lines
    assert [row["ts_utc"] for row in result["entries"]] == [
        FINISHED.isoformat(),
        "2026-10-25T10:00:00+00:00",
        None,
        (FINISHED - timedelta(seconds=1)).isoformat(),
    ]
    for rows in (result["entries"], list(reversed(result["entries"]))):
        receipt = log_receipt(rows, FINISHED + timedelta(seconds=60))
        assert receipt["latest_event_at"] == FINISHED.isoformat()
        assert receipt["observed_at"] == (FINISHED + timedelta(seconds=60)).isoformat()
        assert receipt["event_age_seconds"] == 60
        assert receipt["freshness"] == "recent"
        assert receipt["unknown_timestamp_count"] == 1
        assert receipt["stale_after_seconds"] == 48 * 3600
        assert log_receipt(rows, FINISHED + timedelta(hours=48))["freshness"] == "recent"
        assert log_receipt(rows, FINISHED + timedelta(hours=48, seconds=1))["freshness"] == "stale"
        assert log_receipt(rows, FINISHED - timedelta(seconds=1))["freshness"] == "unknown"


def test_completion_field_is_selected_without_scanning_other_dates_or_guessing_clock(monkeypatch):
    from dashboards import log_reporting

    class NoClock(datetime):
        @classmethod
        def now(cls, tz=None):
            raise AssertionError("The current clock is not an event timestamp")

    monkeypatch.setattr(log_reporting, "datetime", NoClock)
    line = 'Dec31 23:59:59 ' + json.dumps({
        "data_date": "2030-01-01T00:00:00Z",
        "started_at": "2025-01-01T00:00:00Z",
        "finished_at": "2026-11-08T13:58:42.729306-05:45",
        "next_run_at": "2031-01-01T00:00:00Z",
    })
    assert event_time(line) == FINISHED
    assert event_time('Dec31 23:59:59 {"finished_at":null}') is None
    assert event_time('Dec31 23:59:59 {"finished_at":"2026-11-08T19:43:42"}') is None


def test_pythonanywhere_endpoint_preserves_source_metadata_and_separate_receipts(monkeypatch):
    from dashboards import log_reporting
    from test_dashboard_reporting_truthfulness import helpers

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return FINISHED + timedelta(seconds=60)

    monkeypatch.setattr(log_reporting, "datetime", Clock)
    sources = [
        {"source": MONITOR_SOURCE, "kind": "always_on", "id": 42, "text": MONITOR_LINE},
        {
            "source": "pipeline",
            "kind": "scheduled",
            "id": 17,
            "text": "2026-10-25 03:47:47 Opened PTCT @ 69.68 (63 shares)",
        },
    ]
    ns = helpers("api_pythonanywhere_logs", _pythonanywhere_log_sources=lambda: sources,
                 jsonify=lambda payload: payload)
    data = ns["api_pythonanywhere_logs"]()
    assert data["ok"] is True
    assert data["source"] == "pythonanywhere"
    assert [item["source"] for item in data["sources"]] == [MONITOR_SOURCE, "pipeline"]
    for actual, original in zip(data["sources"], sources):
        assert {key: actual[key] for key in original} == original
    monitor, pipeline = data["sources"]
    assert monitor["latest_event_at"] == FINISHED.isoformat()
    assert monitor["event_age_seconds"] == 60
    assert monitor["freshness"] == "recent"
    assert monitor["unknown_timestamp_count"] == 0
    assert monitor["entries"][0]["provenance"] == "unclassified"
    assert pipeline["freshness"] == "stale"
    assert pipeline["entries"][0]["provenance"] == "simulation_format"
    assert "runtime unverified" in pipeline["entries"][0]["provenance_label"]
