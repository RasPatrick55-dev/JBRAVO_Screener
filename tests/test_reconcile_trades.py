from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts import db
from scripts.execute_trades import ExecutionMetrics, ExecutorConfig, TradeExecutor


@pytest.mark.alpaca_optional
def test_reconcile_closes_open_trades(monkeypatch):
    fake_engine = object()
    closed_calls: list[dict] = []
    decorate_calls: list[dict] = []
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

    def fake_get_closed_trades_missing_exit(engine, updated_after, limit=200):
        return [
            {
                "trade_id": open_trade["trade_id"],
                "symbol": open_trade["symbol"],
                "qty": open_trade["qty"],
                "entry_price": open_trade["entry_price"],
                "exit_price": None,
                "exit_order_id": None,
                "realized_pnl": None,
            }
        ]

    def fake_decorate_trade_exit(engine, trade_id, **kwargs):
        decorate_calls.append({"trade_id": trade_id, **kwargs})
        return True

    monkeypatch.setattr(db, "close_trade", fake_close_trade)
    monkeypatch.setattr(db, "insert_order_event", fake_insert_order_event)
    monkeypatch.setattr(db, "get_closed_trades_missing_exit", fake_get_closed_trades_missing_exit)
    monkeypatch.setattr(db, "decorate_trade_exit", fake_decorate_trade_exit)

    filled_order = SimpleNamespace(
        id="sell-1",
        symbol="XYZ",
        side="sell",
        status="filled",
        filled_qty=5,
        filled_avg_price="11.25",
        filled_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        type="trailing_stop",
        order_class="trailing_stop",
    )

    class FakeClient:
        def __init__(self, orders):
            self.orders = orders

        def get_orders(self, request):
            self.request = request
            return self.orders

        def get_all_positions(self):
            return []

    client = FakeClient([filled_order])
    config = ExecutorConfig(reconcile_lookback_days=3)
    executor = TradeExecutor(config, client, ExecutionMetrics())

    executor.reconcile_closed_trades()

    assert len(closed_calls) == 1
    assert closed_calls[0]["exit_order_id"] is None
    assert closed_calls[0]["exit_reason"] == "POSITION_CLOSED"
    assert closed_calls[0]["exit_price"] is None
    assert any(call.get("exit_order_id") == "sell-1" and call.get("exit_reason") == "TRAIL_STOP" for call in decorate_calls)
    assert any(call.get("event_type") == "SELL_FILL" for call in event_calls)


@pytest.mark.alpaca_optional
def test_reconcile_decorates_without_open_trades(monkeypatch):
    fake_engine = object()
    event_calls: list[dict] = []
    decorate_calls: list[dict] = []

    monkeypatch.setattr(db, "db_enabled", lambda: True)
    monkeypatch.setattr(db, "get_engine", lambda: fake_engine)
    monkeypatch.setattr(db, "get_open_trades", lambda engine, limit=200: [])
    monkeypatch.setattr(db, "close_trade", lambda *_, **__: pytest.fail("close_trade should not be called"))

    missing_trade = {
        "trade_id": 2,
        "symbol": "TSLA",
        "qty": 1,
        "entry_price": 100.0,
        "exit_price": None,
        "exit_order_id": None,
        "realized_pnl": None,
    }

    monkeypatch.setattr(
        db,
        "get_closed_trades_missing_exit",
        lambda engine, updated_after, limit=200: [missing_trade],
    )

    def fake_insert_order_event(**kwargs):
        event_calls.append(kwargs)
        return True

    def fake_decorate_trade_exit(engine, trade_id, **kwargs):
        decorate_calls.append({"trade_id": trade_id, **kwargs})
        return True

    monkeypatch.setattr(db, "insert_order_event", fake_insert_order_event)
    monkeypatch.setattr(db, "decorate_trade_exit", fake_decorate_trade_exit)

    filled_order = SimpleNamespace(
        id="sell-2",
        symbol="TSLA",
        side="sell",
        status="filled",
        filled_qty=1,
        filled_avg_price="105.50",
        filled_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        type="market",
        order_class="simple",
    )

    class FakeClient:
        def __init__(self, orders):
            self.orders = orders

        def get_orders(self, request):
            self.request = request
            return self.orders

    client = FakeClient([filled_order])
    config = ExecutorConfig(reconcile_lookback_days=2)
    executor = TradeExecutor(config, client, ExecutionMetrics())

    executor.reconcile_closed_trades()

    assert len(decorate_calls) == 1
    assert decorate_calls[0]["exit_order_id"] == "sell-2"
    assert decorate_calls[0]["exit_price"] == 105.50
    assert any(call.get("event_type") == "SELL_FILL" for call in event_calls)
