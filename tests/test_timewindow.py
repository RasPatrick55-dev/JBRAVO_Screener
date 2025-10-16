import pytest

from datetime import datetime, timezone

from scripts import execute_trades


pytestmark = pytest.mark.alpaca_optional


class _FrozenDateTime:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self, tz=None):  # pragma: no cover - exercised via executor
        if tz is None:
            return self._moment
        return self._moment.astimezone(tz)


def test_premarket_window_handles_7am(monkeypatch):
    config = execute_trades.ExecutorConfig(
        time_window="premarket",
        extended_hours=True,
        market_timezone="America/New_York",
    )
    metrics = execute_trades.ExecutionMetrics()
    executor = execute_trades.TradeExecutor(config, None, metrics)

    frozen = _FrozenDateTime(datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(execute_trades, "datetime", frozen)

    allowed, message, resolved = executor.evaluate_time_window()

    assert allowed is True
    assert "premarket window open" in message
    assert resolved == "premarket"
