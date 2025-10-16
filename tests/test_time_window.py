from datetime import datetime, timezone

from datetime import datetime, timezone

import pytest

from scripts import execute_trades


pytestmark = pytest.mark.alpaca_optional


class _FrozenDateTime:
    """Simple shim to freeze ``datetime.now`` used in executor tests."""

    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self, tz=None):  # pragma: no cover - exercised via executor
        if tz is None:
            return self._moment
        return self._moment.astimezone(tz)


def test_premarket_window_allows_with_timezone_fallback(monkeypatch, caplog):
    config = execute_trades.ExecutorConfig(
        time_window="premarket", extended_hours=True, market_timezone="Invalid/TZ"
    )
    metrics = execute_trades.ExecutionMetrics()
    executor = execute_trades.TradeExecutor(config, None, metrics)

    frozen = _FrozenDateTime(datetime(2024, 1, 2, 12, 30, tzinfo=timezone.utc))
    monkeypatch.setattr(execute_trades, "datetime", frozen)
    caplog.set_level("INFO", logger="execute_trades")

    allowed, message, resolved = executor.evaluate_time_window()

    assert allowed is True
    assert "premarket window open" in message
    assert resolved == "premarket"
    assert any("invalid market timezone" in msg for msg in caplog.messages)
