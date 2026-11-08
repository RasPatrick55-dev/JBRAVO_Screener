from datetime import datetime, timezone
import pytest

pytestmark = pytest.mark.alpaca_optional

from dashboards.log_reporting import annotate_log_text, event_time, log_provenance, log_receipt

NOW = datetime(2026, 11, 8, 14, tzinfo=timezone.utc)


def test_mixed_task_output_is_not_a_broker_ledger():
    text = "\n".join(
        [
            "[2026-11-08 03:47:47,034] [INFO]: Opened PTCT @ 69.68 (63 shares)",
            "[2026-11-08 03:47:47,051] [INFO]: Closed PTCT @ 67.59 reason=TRAIL_STOP",
            "2026-11-08 03:47:48 -- Completed task, took 164.76 seconds, return code was 0.",
            "unrecognized output without a timestamp",
        ]
    )
    result = annotate_log_text(text, "bash scripts/ops/run_primary_pipeline.sh --screener-args x")
    assert [row["provenance"] for row in result["entries"]] == [
        "simulation_format",
        "simulation_format",
        "scheduler",
        "unclassified",
    ]
    assert "runtime unverified" in result["entries"][0]["provenance_label"]
    assert "backtest_core.py" in result["entries"][0]["provenance_evidence"]
    assert result["unknown_timestamp_count"] == 1
    assert len(result["entries"]) == 4


def test_unknown_sources_and_generic_trade_words_stay_unclassified():
    message = "Opened PTCT @ 69.68 (63 shares)"
    for source in ("unknown", "unrelated-pipeline-news", "broker", "/tmp/not_pipeline.log"):
        assert log_provenance(message, source)["provenance"] == "unclassified"
    assert log_provenance("Opened a connection", "pipeline")["provenance"] == "unclassified"
    assert log_provenance(message, "pipeline")["provenance"] == "simulation_format"
    assert (
        log_provenance(message, "pythonanywhere:/home/logs/backtest.log")["provenance"]
        == "simulation_format"
    )


def test_fresh_fetch_cannot_refresh_old_event_and_filters_do_not_define_source_age():
    old = [{"ts_utc": "2026-10-25T12:00:00Z"}]
    receipt = log_receipt(old, NOW)
    assert receipt["observed_at"] == NOW.isoformat()
    assert receipt["freshness"] == "stale"
    assert receipt["event_age_seconds"] == 14 * 86400 + 7200
    assert log_receipt(old + [{"ts_utc": "2026-11-08T13:00:00Z"}], NOW)["freshness"] == "recent"
    # Each source must retain its own receipt even alongside recent sources.
    assert log_receipt(old, NOW)["freshness"] == "stale"


def test_unknown_missing_invalid_future_and_offset_timestamps():
    for timestamp in (None, "", "garbage", "2026-99-99T10:00:00Z", "2026-11-09T10:00:00Z"):
        assert log_receipt([{"ts_utc": timestamp}], NOW)["freshness"] == "unknown"
    for text in (
        "2026-11-08T08:00:00+bad",
        "2026-11-08T08:00:00+0600",
        "2026-11-08T08:00:000",
        "2026-11-08T08:00:00Zbad",
    ):
        assert event_time(text) is None
    assert log_receipt([], NOW)["latest_event_at"] is None
    assert event_time("2026-11-08T08:00:00-06:00") == NOW
    assert event_time("2026-11-08 14:00:00") == NOW
    assert "producer convention" in log_receipt([], NOW)["timestamp_basis"]


def test_log_endpoints_preserve_per_source_evidence_before_filtering(monkeypatch, tmp_path):
    from test_dashboard_api import _prepare_dashboard_data, _reload_dashboard_app

    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "_pythonanywhere_log_sources",
        lambda: [
            {
                "source": "pipeline",
                "text": "2026-11-08 03:47:47 Opened PTCT @ 69.68 (63 shares)\nunknown",
            },
        ],
    )
    with module.server.test_client() as client:
        data = client.get("/api/pythonanywhere/logs").get_json()
        assert data["sources"][0]["entries"][0]["provenance"] == "simulation_format"
        assert data["sources"][0]["unknown_timestamp_count"] == 1
        staged = [
            {
                "ts_utc": "2026-10-25T12:00:00Z",
                "_ts": datetime(2026, 10, 25, 12, tzinfo=timezone.utc),
                "message": "old failure",
                "level": "ERROR",
                "source": "screener.log",
            },
            {
                "ts_utc": "2026-11-08T13:00:00Z",
                "_ts": datetime(2026, 11, 8, 13, tzinfo=timezone.utc),
                "message": "recent info",
                "level": "INFO",
                "source": "pipeline.log",
            },
            {
                "ts_utc": None,
                "_ts": None,
                "message": "untimed",
                "level": "INFO",
                "source": "screener.log",
            },
        ]
        monkeypatch.setattr(
            module, "_screener_logs_from_files", lambda *_: (staged, "screener.log;pipeline.log")
        )
        data = client.get("/api/screener/logs?stage=screener&level=errors").get_json()
        assert len(data["rows"]) == 1
        assert data["rows"][0]["source"] == "screener.log"
        assert data["rows"][0]["provenance"] == "unclassified"
        receipts = {row["source"]: row for row in data["source_receipts"]}
        assert receipts["screener.log"]["freshness"] == "stale"
        assert receipts["pipeline.log"]["latest_event_at"].startswith("2026-11-08T13:00:00")
        assert receipts["screener.log"]["unknown_timestamp_count"] == 1
        assert data["latest_event_at"].startswith("2026-11-08T13:00:00")
