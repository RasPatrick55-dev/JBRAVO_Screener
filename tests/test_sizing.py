import pandas as pd
import pytest

from scripts import execute_trades


pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def _stub_alpaca_http(monkeypatch):
    prices = {"AAA": 250.0, "BBB": 50.0, "CCC": 12.5}

    def _price(symbol: str) -> float:
        return prices.get(symbol.upper(), 50.0)

    monkeypatch.setattr(
        "scripts.execute_trades._fetch_prevclose_snapshot", lambda symbol: _price(symbol)
    )
    monkeypatch.setattr(
        "scripts.execute_trades._fetch_prev_close_from_alpaca",
        lambda symbol: _price(symbol),
    )
    monkeypatch.setattr(
        "scripts.execute_trades._fetch_latest_trade_from_alpaca",
        lambda symbol, feed=None: {"price": _price(symbol), "feed": feed},
    )
    monkeypatch.setattr(
        "scripts.execute_trades._fetch_latest_quote_from_alpaca",
        lambda symbol, feed=None: {"ask": _price(symbol), "feed": feed},
    )


def test_sizing_produces_positive_quantities(monkeypatch):
    config = execute_trades.ExecutorConfig(
        dry_run=True,
        allocation_pct=0.05,
        min_order_usd=200,
        max_positions=3,
    )
    metrics = execute_trades.ExecutionMetrics()
    executor = execute_trades.TradeExecutor(config, None, metrics)

    monkeypatch.setattr(executor, "fetch_existing_positions", lambda: set())
    monkeypatch.setattr(executor, "fetch_open_order_symbols", lambda: set())
    monkeypatch.setattr(executor, "fetch_buying_power", lambda: 10000.0)
    monkeypatch.setattr(executor, "evaluate_time_window", lambda log=True: (True, "ok", "any"))

    captured: list[dict] = []

    def capture_event(event: str, **payload):  # pragma: no cover - exercised via executor
        if event == "DRY_RUN_ORDER":
            captured.append(payload)

    monkeypatch.setattr(executor, "log_info", capture_event)

    records = [
        {
            "symbol": "AAA",
            "close": 250.0,
            "entry_price": None,
            "score": 1.0,
            "universe_count": 3,
            "score_breakdown": "{}",
        },
        {
            "symbol": "BBB",
            "close": 50.0,
            "entry_price": None,
            "score": 1.2,
            "universe_count": 3,
            "score_breakdown": "{}",
        },
        {
            "symbol": "CCC",
            "close": 12.5,
            "entry_price": None,
            "score": 0.9,
            "universe_count": 3,
            "score_breakdown": "{}",
        },
    ]

    frame = pd.DataFrame(records)
    executor.execute(frame, prefiltered=records)

    assert captured, "Expected DRY_RUN_ORDER events"
    for payload in captured:
        assert int(payload.get("qty", 0)) >= 1

    assert "ZERO_QTY" not in executor.metrics.skipped_reasons
