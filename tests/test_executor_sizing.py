import logging
from pathlib import Path

import pandas as pd
import pytest

from scripts.execute_trades import ExecutionMetrics, ExecutorConfig, TradeExecutor


@pytest.fixture(autouse=True)
def _stub_alpaca_http(monkeypatch):
    """Avoid real Alpaca HTTP calls to keep sizing tests fast/offline."""

    prices = {"BUMP": 300.0, "FLOOR": 95.0, "ATRSMALL": 50.0, "ATRMIN": 50.0}

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


@pytest.mark.alpaca_optional
def test_sizing_logs_zero_qty_after_bump(monkeypatch, caplog, tmp_path: Path):
    config = ExecutorConfig(
        source=tmp_path / "candidates.csv",
        allocation_pct=0.01,
        min_order_usd=150.0,
        allow_bump_to_one=True,
        dry_run=True,
        time_window="any",
    )
    df = pd.DataFrame(
        [
            {
                "symbol": "BUMP",
                "close": 300.0,
                "entry_price": 300.0,
                "score": 2.0,
                "universe_count": 10,
                "score_breakdown": "{}",
            }
        ]
    )
    metrics = ExecutionMetrics()
    executor = TradeExecutor(config, None, metrics)
    monkeypatch.setattr(executor, "fetch_buying_power", lambda: 1000.0)
    monkeypatch.setattr(executor, "evaluate_time_window", lambda log=True: (True, "ok", "any"))

    caplog.set_level(logging.DEBUG, logger="execute_trades")
    caplog.clear()

    rc = executor.execute(df, prefiltered=df.to_dict("records"))
    assert rc == 0
    messages = [record.getMessage() for record in caplog.records]
    assert any("ZERO_QTY_AFTER_BUMP" in msg for msg in messages)
    assert metrics.skipped_reasons.get("ZERO_QTY", 0) == 1


@pytest.mark.alpaca_optional
def test_min_order_floor_sets_quantity(monkeypatch, caplog, tmp_path: Path):
    config = ExecutorConfig(
        source=tmp_path / "candidates.csv",
        allocation_pct=0.01,
        min_order_usd=500.0,
        allow_bump_to_one=False,
        dry_run=True,
        time_window="any",
    )
    df = pd.DataFrame(
        [
            {
                "symbol": "FLOOR",
                "close": 95.0,
                "entry_price": 95.0,
                "score": 2.0,
                "universe_count": 10,
                "score_breakdown": "{}",
            }
        ]
    )
    metrics = ExecutionMetrics()
    executor = TradeExecutor(config, None, metrics)
    monkeypatch.setattr(executor, "fetch_buying_power", lambda: 2000.0)
    monkeypatch.setattr(executor, "evaluate_time_window", lambda log=True: (True, "ok", "any"))

    caplog.set_level(logging.DEBUG, logger="execute_trades")
    caplog.clear()

    rc = executor.execute(df, prefiltered=df.to_dict("records"))
    assert rc == 0
    messages = [record.getMessage() for record in caplog.records]
    assert any("CALC symbol=FLOOR" in msg and "qty=5" in msg for msg in messages)
    assert any("notional=500.00" in msg for msg in messages)
    assert metrics.skipped_reasons.get("ZERO_QTY", 0) == 0


@pytest.mark.alpaca_optional
def test_atr_position_sizer_scales_quantity(monkeypatch, caplog, tmp_path: Path):
    config = ExecutorConfig(
        source=tmp_path / "candidates.csv",
        allocation_pct=0.10,
        position_sizer="atr",
        atr_target_pct=0.02,
        limit_buffer_pct=0.0,
        allow_bump_to_one=False,
        dry_run=True,
        time_window="any",
    )
    df = pd.DataFrame(
        [
            {
                "symbol": "ATRSMALL",
                "close": 50.0,
                "entry_price": 50.0,
                "atrp": 0.10,
                "score": 2.0,
                "universe_count": 10,
                "score_breakdown": "{}",
            }
        ]
    )
    metrics = ExecutionMetrics()
    executor = TradeExecutor(config, None, metrics)
    monkeypatch.setattr(executor, "fetch_buying_power", lambda: 10_000.0)
    monkeypatch.setattr(executor, "evaluate_time_window", lambda log=True: (True, "ok", "any"))

    caplog.set_level(logging.DEBUG, logger="execute_trades")
    caplog.clear()

    rc = executor.execute(df, prefiltered=df.to_dict("records"))
    assert rc == 0
    messages = [record.getMessage() for record in caplog.records]
    atr_msgs = [msg for msg in messages if "CALC symbol=ATRSMALL" in msg]
    assert atr_msgs, "expected sizing log for ATRSMALL"
    assert any("sizer=atr" in msg and "atr=0.1000" in msg for msg in atr_msgs)
    assert any("scale=0.200" in msg for msg in atr_msgs)
    assert any("qty=3" in msg or "qty=4" in msg for msg in atr_msgs)


@pytest.mark.alpaca_optional
def test_atr_position_sizer_respects_min_order(monkeypatch, caplog, tmp_path: Path):
    config = ExecutorConfig(
        source=tmp_path / "candidates.csv",
        allocation_pct=0.10,
        position_sizer="atr",
        atr_target_pct=0.02,
        limit_buffer_pct=0.0,
        min_order_usd=300.0,
        allow_bump_to_one=False,
        dry_run=True,
        time_window="any",
    )
    df = pd.DataFrame(
        [
            {
                "symbol": "ATRMIN",
                "close": 50.0,
                "entry_price": 50.0,
                "atrp": 0.50,
                "score": 2.0,
                "universe_count": 10,
                "score_breakdown": "{}",
            }
        ]
    )
    metrics = ExecutionMetrics()
    executor = TradeExecutor(config, None, metrics)
    monkeypatch.setattr(executor, "fetch_buying_power", lambda: 10_000.0)
    monkeypatch.setattr(executor, "evaluate_time_window", lambda log=True: (True, "ok", "any"))

    caplog.set_level(logging.DEBUG, logger="execute_trades")
    caplog.clear()

    rc = executor.execute(df, prefiltered=df.to_dict("records"))
    assert rc == 0
    messages = [record.getMessage() for record in caplog.records]
    atr_msgs = [msg for msg in messages if "CALC symbol=ATRMIN" in msg]
    assert atr_msgs, "expected sizing log for ATRMIN"
    assert any("notional=300.00" in msg for msg in atr_msgs)
    assert any("sizer=atr" in msg and "atr=0.5000" in msg for msg in atr_msgs)
    assert any("qty=5" in msg or "qty=6" in msg for msg in atr_msgs)
