import importlib
import sys

import pytest


@pytest.mark.alpaca_optional
def test_load_kpis_infers_from_pipeline(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    data_dir = repo_root / "data"
    logs_dir = repo_root / "logs"
    data_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    pipeline_log = logs_dir / "pipeline.log"
    pipeline_log.write_text(
        "2024-05-20T12:00:00Z PIPELINE_SUMMARY symbols_in=321 with_bars=300 rows=111\n"
    )

    monkeypatch.setenv("JBRAVO_HOME", str(repo_root))
    sys.modules.pop("dashboards.screener_health", None)
    screener_health = importlib.import_module("dashboards.screener_health")
    try:
        kpis = screener_health.load_kpis()

        assert kpis["symbols_in"] == 321
        assert kpis["symbols_with_bars"] == 300
        assert kpis["rows"] == 111
        assert kpis["source"] == "pipeline summary (recovered)"
        assert kpis["_kpi_inferred_from_log"] is True
    finally:
        sys.modules.pop("dashboards.screener_health", None)
