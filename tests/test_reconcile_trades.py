from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from scripts import db
from scripts.execute_trades import ExecutionMetrics, ExecutorConfig, TradeExecutor


@pytest.mark.alpaca_optional
def test_reconcile_closes_open_trades(monkeypatch):
    fake_engine = object()
    closed_calls: list[dict] = []
    event_calls: list[dict] = []

    monkeypatch.setattr(db, "db_enabled", lambda: True)
    monkeypatch.setattr(db, "get_engine", lambda: fake_engine)

    open_trade = {
        "trade_id": 1,
        "symbol": "XYZ",
        "qty": 5,
        "entry_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "entry_price": 10.0,
        "entry_order_id": "open-1",
    }
    monkeypatch.setattr(db, "get_open_trades", lambda engine, limit=200: [open_trade])

    def fake_close_trade(engine, trade_id, exit_order_id, exit_time, exit_price, exit_reason):
        closed_calls.append(
            {
                "trade_id": trade_id,
                "exit_order_id": exit_order_id,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
            }
        )
        return True

    def fake_insert_order_event(**kwargs):
        event_calls.append(kwargs)
        return True

    monkeypatch.setattr(db, "close_trade", fake_close_trade)
    monkeypatch.setattr(db, "insert_order_event", fake_insert_order_event)

    filled_order = SimpleNamespace(
        id="sell-1",
        symbol="XYZ",
        side="sell",
        status="filled",
        filled_qty=5,
        filled_avg_price="11.25",
        filled_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        type="trailing_stop",
        order_class="trailing_stop",
    )

    class FakeClient:
        def __init__(self, orders):
            self.orders = orders

        def get_orders(self, request):
            self.request = request
            return self.orders

    client = FakeClient([filled_order])
    config = ExecutorConfig(reconcile_lookback_days=3)
    executor = TradeExecutor(config, client, ExecutionMetrics())

    executor.reconcile_closed_trades()

    assert len(closed_calls) == 1
    assert closed_calls[0]["exit_order_id"] == "sell-1"
    assert closed_calls[0]["exit_reason"] == "TRAIL_STOP"
    assert pytest.approx(closed_calls[0]["exit_price"]) == 11.25
    assert any(call.get("event_type") == "SELL_FILL" for call in event_calls)
