import json

import pandas as pd
import pytest

import scripts.execute_trades as execute_mod
from scripts.execute_trades import (
    ExecutionMetrics,
    ExecutorConfig,
    TradeExecutor,
    compute_limit_price,
    compute_quantity,
    run_executor,
)


class StubOrder:
    def __init__(self, order_id: str, symbol: str, qty: int, limit_price: float) -> None:
        self.id = order_id
        self.symbol = symbol
        self.qty = qty
        self.limit_price = limit_price
        self.status = "new"
        self.filled_qty = 0
        self.filled_avg_price = limit_price
        self._poll_count = 0


class StubTradingClient:
    def __init__(self) -> None:
        self.submitted_orders: list[StubOrder] = []
        self.trailing_orders: list[StubOrder] = []
        self.positions: list = []
        self.account = type("Account", (), {"buying_power": "50000"})()
        self.open_orders: list[StubOrder] = []

    def get_all_positions(self):
        return list(self.positions)

    def get_orders(self, request):  # pragma: no cover - request unused in stub
        return list(self.open_orders)

    def get_account(self):
        return self.account

    def submit_order(self, request):
        if getattr(request, "trail_percent", None) is not None:
            order_id = f"trail-{len(self.trailing_orders)+1}"
            order = StubOrder(order_id, request.symbol, request.qty, request.trail_percent)
            order.status = "accepted"
            self.trailing_orders.append(order)
            return order
        order_id = f"order-{len(self.submitted_orders)+1}"
        limit_price = getattr(request, "limit_price", 0.0)
        order = StubOrder(order_id, request.symbol, request.qty, limit_price)
        self.submitted_orders.append(order)
        self.open_orders.append(order)
        return order

    def get_order_by_id(self, order_id: str):
        for order in self.submitted_orders:
            if order.id == order_id:
                order._poll_count += 1
                if order._poll_count == 1:
                    order.status = "accepted"
                elif order._poll_count == 2:
                    order.status = "partially_filled"
                    order.filled_qty = order.qty - 1
                else:
                    order.status = "filled"
                    order.filled_qty = order.qty
                return order
        raise KeyError(order_id)

    def cancel_order_by_id(self, order_id: str):  # pragma: no cover - not used
        for order in self.open_orders:
            if order.id == order_id:
                order.status = "canceled"
                break


@pytest.fixture(autouse=True)
def isolate_metrics_path(monkeypatch, tmp_path):
    metrics_path = tmp_path / "execute_metrics.json"
    monkeypatch.setattr(execute_mod, "METRICS_PATH", metrics_path)
    log_path = tmp_path / "execute_trades.log"
    monkeypatch.setattr(execute_mod, "LOG_PATH", log_path)
    return metrics_path


pytestmark = pytest.mark.alpaca_optional


def test_compute_limit_price_rounds_to_tick():
    row = {"entry_price": 10.003, "close": 10.0}
    price = compute_limit_price(row, buffer_bps=50)
    assert price == pytest.approx(10.05, rel=1e-3)


def test_compute_quantity_caps_by_buying_power():
    qty = compute_quantity(5000, 0.1, 120.25)
    assert qty == 4
    assert compute_quantity(100, 0.1, 200) == 0
    assert compute_quantity(1000, 0.5, 10) == 50


def test_header_only_candidates_exit_cleanly(tmp_path, monkeypatch):
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text("symbol,close,score,universe_count,score_breakdown\n", encoding="utf-8")
    config = ExecutorConfig(source=csv_path, dry_run=True)
    rc = run_executor(config, client=StubTradingClient())
    assert rc == 0
    metrics_payload = json.loads(execute_mod.METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics_payload["symbols_in"] == 0
    assert metrics_payload["orders_submitted"] == 0


def test_execute_flow_attaches_trailing_stop(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "close": 100.0,
                "score": 2.5,
                "universe_count": 100,
                "score_breakdown": "{}",
            }
        ]
    )
    frame.to_csv(csv_path, index=False)
    config = ExecutorConfig(source=csv_path, cancel_after_min=1, allocation_pct=0.5)
    metrics = ExecutionMetrics()
    client = StubTradingClient()
    executor = TradeExecutor(config, client, metrics, sleep_fn=lambda *_: None)
    df = executor.load_candidates()
    rc = executor.execute(df)
    assert rc == 0
    assert metrics.orders_submitted == 1
    assert metrics.trailing_attached == 1
    assert metrics.orders_filled == 1
    assert client.trailing_orders, "Trailing stop should have been submitted"
