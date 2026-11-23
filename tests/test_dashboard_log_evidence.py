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


@pytest.mark.parametrize(("enabled", "logfile_prefix", "uses_current_task"), [
    (True, "", True),
    (True, "/user/tester/files", True),
    (True, "/user/another-account/files", False),
    (False, "", False),
])
def test_execute_pipeline_log_prefers_the_enabled_primary_task(
    monkeypatch, tmp_path, enabled, logfile_prefix, uses_current_task,
):
    from test_dashboard_api import _prepare_dashboard_data, _reload_dashboard_app

    _prepare_dashboard_data(tmp_path)
    module = _reload_dashboard_app(monkeypatch, tmp_path)
    task_id = 1568726
    task_path = f"/var/log/schedule-log-{task_id}.log"
    old_source = "https://www.pythonanywhere.com/api/v0/user/tester/files/path/var/log/schedule-log-1322138.log"
    command = """bash -lc 'set -e; cd /home/tester/jbravo_screener; ENV_FILE=/tmp/example bash scripts/ops/run_pythonanywhere_primary_pipeline.sh --example'"""
    monkeypatch.setenv("PYTHONANYWHERE_PIPELINE_LOG_URL", old_source)
    monkeypatch.setattr(module, "_pythonanywhere_username", lambda: "tester")
    monkeypatch.setattr(module, "_screener_logs_from_db", lambda *_: ([], "postgres:none"))
    monkeypatch.setattr(module, "_pythonanywhere_schedule_tasks", lambda: [
        {"id": task_id, "enabled": enabled, "command": command, "logfile": logfile_prefix + task_path},
        {"id": 1322138, "enabled": True, "command": "echo scripts/ops/run_pythonanywhere_primary_pipeline.sh"},
    ])
    file_reads = []

    def provider_file(url, path):
        file_reads.append((url, path))
        if path == task_path:
            return "2026-11-23 03:46:05 -- Completed task, return code was 0.", task_path
        return "2026-10-27 03:46:02 INFO scheduled output", old_source

    monkeypatch.setattr(module, "_fetch_pythonanywhere_file", provider_file)
    assert module._runs_primary_pipeline(command)
    assert not module._runs_primary_pipeline("echo scripts/ops/run_pythonanywhere_primary_pipeline.sh")
    assert not module._runs_primary_pipeline("bash -lc 'echo scripts/ops/run_pythonanywhere_primary_pipeline.sh'")
    if uses_current_task:
        expected, date, reads = f"pythonanywhere:schedule/{task_id}/", "2026-11-23", [(None, task_path)]
    else:
        expected, date, reads = f"pythonanywhere:{old_source}", "2026-10-27", [(old_source, None)]
    with module.server.test_client() as client:
        data = client.get("/api/execute/logs?stage=pipeline&limit=5").get_json()
        assert data["ok"] and data["source"] == "pythonanywhere"
        assert len(data["rows"]) == 1
        assert data["rows"][0]["ts_utc"].startswith(date)
        assert data["source_detail"] == f"postgres:none;{expected}"
        assert file_reads == reads
        if uses_current_task:
            monkeypatch.setattr(module, "_execute_summary_payload", lambda: {"ok": True})
            monkeypatch.setattr(module, "_execute_orders_payload", lambda **_: {})
            monkeypatch.setattr(module, "_execute_trailing_stops_payload", lambda **_: {})
            monkeypatch.setattr(module, "_execute_logs_from_order_events", lambda *_: ([], "postgres:none"))
            monkeypatch.setattr(module, "_execute_log_file_payload",
                lambda stage, original=module._execute_log_file_payload:
                    original(stage) if stage == "pipeline" else (None, "files:none"))
            state = client.get("/api/execute/state").get_json()
            assert state["logs"]["pipeline"] == data
