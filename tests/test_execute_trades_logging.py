import json
import logging

import pandas as pd
import pytest

import scripts.execute_trades as execute_mod
from scripts.execute_trades import (
    ExecutionMetrics,
    ExecutorConfig,
    TradeExecutor,
    compute_limit_price,
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


def test_header_only_candidates_exit_cleanly(tmp_path, monkeypatch, caplog):
    csv_path = tmp_path / "candidates.csv"
    csv_path.write_text("symbol,close,score,universe_count,score_breakdown\n", encoding="utf-8")
    config = ExecutorConfig(source=csv_path, dry_run=True)
    caplog.set_level(logging.INFO, logger="execute_trades")
    caplog.clear()
    rc = run_executor(config, client=StubTradingClient())
    assert rc == 0
    metrics_payload = json.loads(execute_mod.METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics_payload["symbols_in"] == 0
    assert metrics_payload["orders_submitted"] == 0
    logs = "\n".join(caplog.messages)
    assert "EXEC_START dry_run=True time_window=auto" in logs
    assert "ny_now=" in logs
    assert "in_window=False" in logs or "in_window=True" in logs
    skips = metrics_payload.get("skips", {})
    assert int(skips.get("NO_CANDIDATES", 0)) >= 1


def test_execute_flow_attaches_trailing_stop(tmp_path, caplog):
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
    config = ExecutorConfig(
        source=csv_path,
        cancel_after_min=1,
        allocation_pct=0.5,
        time_window="any",
        extended_hours=True,
    )
    metrics = ExecutionMetrics()
    client = StubTradingClient()
    executor = TradeExecutor(config, client, metrics, sleep_fn=lambda *_: None)
    df = executor.load_candidates()
    caplog.set_level(logging.INFO, logger="execute_trades")
    caplog.clear()
    rc = executor.execute(df)
    assert rc == 0
    assert metrics.orders_submitted == 1
    assert metrics.trailing_attached == 1
    assert metrics.orders_filled == 1
    assert client.trailing_orders, "Trailing stop should have been submitted"
    logs = "\n".join(caplog.messages)
    assert "TRAIL_SUBMIT" in logs
    assert "TRAIL_CONFIRMED" in logs


def test_time_window_skip_logs_summary(tmp_path, monkeypatch, caplog):
    csv_path = tmp_path / "candidates.csv"
    frame = pd.DataFrame(
        [
            {
                "symbol": "TSLA",
                "close": 200.0,
                "score": 3.2,
                "universe_count": 50,
                "score_breakdown": "{}",
            }
        ]
    )
    frame.to_csv(csv_path, index=False)
    config = ExecutorConfig(source=csv_path, dry_run=True)
    metrics = ExecutionMetrics()
    executor = TradeExecutor(config, None, metrics, sleep_fn=lambda *_: None)
    df = executor.load_candidates()
    caplog.set_level(logging.INFO, logger="execute_trades")
    caplog.clear()

    def fake_window(self):
        return False, "outside premarket (NY)", "premarket"

    monkeypatch.setattr(execute_mod.TradeExecutor, "evaluate_time_window", fake_window)
    rc = executor.execute(df)
    assert rc == 0
    log_text = "\n".join(caplog.messages)
    tokens = ("EXEC_START", "[INFO] EXECUTE_SUMMARY", "[INFO] TIME_WINDOW")
    assert any(token in log_text for token in tokens)
    assert "[INFO] TIME_WINDOW outside premarket (NY)" in log_text
    summary_lines = [msg for msg in caplog.messages if "[INFO] EXECUTE_SUMMARY" in msg]
    assert summary_lines, "Expected EXECUTE_SUMMARY log entry"
    summary = summary_lines[-1]
    assert "orders_submitted=0" in summary
    assert "trailing_attached=0" in summary
    for key in ("TIME_WINDOW", "ZERO_QTY", "CASH", "OPEN_ORDER", "EXISTING_POSITION", "MAX_POSITIONS", "NO_CANDIDATES"):
        assert f"skips.{key}=" in summary
    assert "skips.TIME_WINDOW=1" in summary


def test_dry_run_creates_metrics_with_zero_orders(tmp_path, caplog):
    csv_path = tmp_path / "candidates.csv"
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "close": 50.0,
                "score": 1.0,
                "universe_count": 10,
                "score_breakdown": "{}",
            },
            {
                "symbol": "BBB",
                "close": 60.0,
                "score": 1.1,
                "universe_count": 10,
                "score_breakdown": "{}",
            },
        ]
    )
    frame.to_csv(csv_path, index=False)

    config = ExecutorConfig(source=csv_path, dry_run=True, time_window="any")
    caplog.set_level(logging.INFO, logger="execute_trades")
    caplog.clear()

    rc = run_executor(config, client=StubTradingClient())
    assert rc == 0

    log_text = "\n".join(caplog.messages)
    assert "EXEC_START dry_run=True" in log_text
    assert "[INFO] EXECUTE_SUMMARY" in log_text

    metrics_payload = json.loads(execute_mod.METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics_payload.get("symbols_in") == 2
    assert metrics_payload.get("orders_submitted") == 0
    assert metrics_payload.get("trailing_attached") == 0


def test_limit_buffer_pct_relaxes_price_bounds(tmp_path):
    record = {
        "symbol": "ABC",
        "entry_price": 9.6,
        "close": 9.6,
        "score": 2.0,
        "universe_count": 10,
        "score_breakdown": "{}",
        "adv20": 5_000_000,
    }

    buffered_config = ExecutorConfig(
        limit_buffer_pct=5.0,
        min_price=10.0,
        max_price=50.0,
        extended_hours=True,
    )
    buffered_executor = TradeExecutor(buffered_config, None, ExecutionMetrics(), sleep_fn=lambda *_: None)
    allowed = buffered_executor.guard_candidates([record])
    assert allowed, "buffer should allow slightly-below-min price"

    strict_config = ExecutorConfig(
        limit_buffer_pct=0.0,
        min_price=10.0,
        max_price=50.0,
        extended_hours=True,
    )
    strict_executor = TradeExecutor(strict_config, None, ExecutionMetrics(), sleep_fn=lambda *_: None)
    denied = strict_executor.guard_candidates([record])
    assert not denied, "without buffer the same candidate should be skipped"
