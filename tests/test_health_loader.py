import importlib
import json
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest


def _reload_data_io(monkeypatch: pytest.MonkeyPatch, base_dir: Path):
    monkeypatch.setenv("JBRAVO_HOME", str(base_dir))
    import dashboards.data_io as data_io  # local import for reload

    return importlib.reload(data_io)


def test_screener_health_prefers_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    metrics = {
        "symbols_in": 25,
        "symbols_with_bars_fetch": 20,
        "bars_rows_total_fetch": 400,
        "rows": 2,
        "rows_premetrics": 2,
        "latest_source": "screener",
        "pipeline_rc": 0,
        "last_run_utc": "2024-01-01T10:00:00+00:00",
    }
    (data_dir / "screener_metrics.json").write_text(json.dumps(metrics))

    top_df = pd.DataFrame([{"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "CCC"}])
    top_df.to_csv(data_dir / "top_candidates.csv", index=False)

    conn_payload = {
        "trading_ok": True,
        "data_ok": True,
        "trading_status": 200,
        "data_status": 200,
        "feed": "iex",
        "timestamp": "2024-01-01T10:00:00+00:00",
    }
    (data_dir / "connection_health.json").write_text(json.dumps(conn_payload))

    (logs_dir / "pipeline.log").write_text("2024-01-01 pipeline - [INFO] PIPELINE_END rc=0\n")

    data_io = _reload_data_io(monkeypatch, tmp_path)
    snapshot = data_io.screener_health()

    assert snapshot["symbols_in"] == 25
    assert snapshot["symbols_with_bars"] == 20
    assert snapshot["bars_rows_total"] == 400
    assert snapshot["rows_final"] == 3
    assert snapshot["rows_premetrics"] == 3
    assert snapshot["source"] == "screener"
    assert snapshot["pipeline_rc"] == 0
    assert snapshot["trading_ok"] is True
    assert snapshot["data_ok"] is True
    assert snapshot["run_type"] in {"nightly", "pre-market"}


def test_screener_health_fallbacks_when_files_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    metrics = {
        "symbols_in": 10,
        "symbols_with_bars": 9,
        "bars_rows_total": 99,
        "rows": 7,
    }
    (data_dir / "screener_metrics.json").write_text(json.dumps(metrics))

    # ensure top_candidates drives final row count
    top_df = pd.DataFrame([{"symbol": "X"}, {"symbol": "Y"}])
    top_df.to_csv(data_dir / "top_candidates.csv", index=False)

    log_lines = [
        "2024-01-01 pipeline - [INFO] PIPELINE_SUMMARY symbols_in=10 with_bars=9 rows=5 source=fallback",
        "2024-01-01 pipeline - [INFO] HEALTH trading_ok=False data_ok=False trading_status=503 data_status=204",
        "2024-01-01 pipeline - [INFO] PIPELINE_END rc=2",
    ]
    (logs_dir / "pipeline.log").write_text("\n".join(log_lines))

    data_io = _reload_data_io(monkeypatch, tmp_path)
    snapshot = data_io.screener_health()

    assert snapshot["rows_final"] == 2
    assert snapshot["source"] == "fallback"
    assert snapshot["pipeline_rc"] == 2
    assert snapshot["trading_ok"] is False
    assert snapshot["data_ok"] is False
    assert snapshot["symbols_with_bars"] == 9
