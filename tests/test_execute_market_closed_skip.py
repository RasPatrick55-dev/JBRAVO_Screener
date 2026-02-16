import pytest

pytestmark = pytest.mark.alpaca_optional

import scripts.execute_trades as execute_mod
from scripts.execute_trades import ExecutorConfig, run_executor


class ClosedClockClient:
    def get_clock(self):
        return type(
            "Clock",
            (),
            {
                "is_open": False,
                "timestamp": "2026-02-15T14:00:00Z",
                "next_open": "2026-02-17T14:30:00Z",
                "next_close": "2026-02-17T21:00:00Z",
                "session": "closed",
            },
        )()


def test_market_closed_skips_before_candidate_load(monkeypatch, tmp_path):
    execute_mod._EXECUTE_METRICS_PAYLOAD = None

    def fail_load(*args, **kwargs):
        raise AssertionError("load_candidates should not run on market closed")

    monkeypatch.setattr(execute_mod.TradeExecutor, "load_candidates", fail_load)
    config = ExecutorConfig(source_path=tmp_path / "c.csv", source_type="db", dry_run=True)
    rc = run_executor(config, client=ClosedClockClient())
    assert rc == 0
    payload = execute_mod._EXECUTE_METRICS_PAYLOAD or {}
    assert payload.get("status") == "skipped"
    assert payload.get("skips", {}).get("MARKET_CLOSED", 0) == 1
