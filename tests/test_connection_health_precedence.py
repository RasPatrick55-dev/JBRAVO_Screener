import json
from pathlib import Path

import pandas as pd
import pytest

from tests._data_io_helpers import reload_data_io


def test_connection_health_json_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    pd.DataFrame([{"symbol": "AAA"}]).to_csv(data_dir / "top_candidates.csv", index=False)
    (data_dir / "screener_metrics.json").write_text(json.dumps({"symbols_in": 1}), encoding="utf-8")

    conn_payload = {
        "trading_ok": True,
        "data_ok": True,
        "trading_status": 200,
        "data_status": 200,
        "feed": "iex",
        "timestamp": "2024-01-01T00:00:00+00:00",
    }
    (data_dir / "connection_health.json").write_text(json.dumps(conn_payload), encoding="utf-8")

    log_lines = [
        "2024-01-01 [INFO] HEALTH trading_ok=False data_ok=False trading_status=503 data_status=204",
        "2024-01-01 [INFO] PIPELINE_END rc=1",
    ]
    (logs_dir / "pipeline.log").write_text("\n".join(log_lines), encoding="utf-8")

    data_io = reload_data_io(monkeypatch, tmp_path)
    snapshot = data_io.screener_health()

    assert snapshot["trading_ok"] is True
    assert snapshot["data_ok"] is True
    assert snapshot["trading_status"] == 200
    assert snapshot["data_status"] == 200
