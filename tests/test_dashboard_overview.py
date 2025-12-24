import json

import pytest

import dashboards.overview as overview


@pytest.mark.alpaca_optional
def test_ops_summary_loader_returns_expected_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(overview, "DATA_DIR", tmp_path)
    monkeypatch.setattr(overview, "LOG_DIR", tmp_path)

    screener_payload = {
        "timestamp": "2024-01-01T00:00:00",
        "symbols_in": 5,
        "with_bars": 4,
        "rows_out": 3,
    }
    exec_payload = {
        "configured_max_positions": 10,
        "risk_limited_max_positions": 8,
        "open_positions": 2,
        "open_orders": 1,
        "allowed_new_positions": 3,
        "in_window": True,
        "exit_reason": "exit",
        "orders_submitted": 4,
        "fills": 2,
    }

    (tmp_path / "screener_metrics.json").write_text(json.dumps(screener_payload))
    (tmp_path / "execute_metrics.json").write_text(json.dumps(exec_payload))

    metrics = overview._read_execute_metrics()

    expected_keys = {
        "configured_max_positions",
        "risk_limited_max_positions",
        "open_positions",
        "open_orders",
        "allowed_new_positions",
        "in_window",
        "exit_reason",
    }

    assert expected_keys.issubset(metrics.keys())
    for key in expected_keys:
        assert metrics[key] == exec_payload[key]


@pytest.mark.alpaca_optional
def test_ops_summary_loader_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(overview, "DATA_DIR", tmp_path)
    monkeypatch.setattr(overview, "LOG_DIR", tmp_path)

    metrics = overview._read_execute_metrics()

    expected_defaults = {
        "configured_max_positions": 0,
        "risk_limited_max_positions": 0,
        "open_positions": 0,
        "open_orders": 0,
        "allowed_new_positions": 0,
        "exit_reason": "n/a",
        "in_window": False,
    }

    assert expected_defaults.items() <= metrics.items()
