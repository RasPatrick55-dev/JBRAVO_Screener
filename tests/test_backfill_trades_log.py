import pandas as pd
import pytest


from scripts.backfill_trades_log import (
    AccountActivityUnavailable,
    backfill,
    build_trades_from_events,
    gather_fill_events,
    merge_trades,
)


@pytest.fixture
def dummy_events():
    return [
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 1,
            "price": 100,
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 1,
            "price": 110,
            "timestamp": "2024-01-02T00:00:00Z",
        },
    ]


def test_build_trades_from_events_pairs_buys_and_sells(dummy_events):
    trades = build_trades_from_events(dummy_events)

    assert len(trades) == 1
    trade = trades[0]
    assert trade["symbol"] == "AAPL"
    assert trade["qty"] == 1
    assert trade["entry_price"] == 100
    assert trade["exit_price"] == 110
    assert trade["net_pnl"] == 10
    assert trade["side"] == "buy"
    assert trade["order_status"] == "filled"


def test_merge_trades_deduplicates_on_key(dummy_events, tmp_path):
    trades = build_trades_from_events(dummy_events)
    existing = pd.DataFrame(trades)

    merged = merge_trades(existing, trades, merge_existing=True)

    assert len(merged) == 1
    assert merged.iloc[0]["entry_price"] == 100


def test_gather_fill_events_falls_back_to_orders(monkeypatch, dummy_events, null_logger):
    calls = {"activities": 0, "orders": 0}

    def fake_fetch_account(*args, **kwargs):
        calls["activities"] += 1
        raise AccountActivityUnavailable("activities disabled")

    def fake_fetch_orders(*args, **kwargs):
        calls["orders"] += 1
        return dummy_events

    monkeypatch.setattr("scripts.backfill_trades_log.fetch_account_fill_events", fake_fetch_account)
    monkeypatch.setattr("scripts.backfill_trades_log.fetch_order_fill_events", fake_fetch_orders)

    events = gather_fill_events(
        pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"), logger=null_logger
    )

    assert events == dummy_events
    assert calls == {"activities": 1, "orders": 1}


def test_build_trades_supports_multi_fill_entry_and_exit():
    events = [
        {"symbol": "AAPL", "side": "buy", "qty": 5, "price": 10, "timestamp": "2024-01-01T00:00:00Z"},
        {"symbol": "AAPL", "side": "buy", "qty": 5, "price": 11, "timestamp": "2024-01-01T00:01:00Z"},
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 6,
            "price": 12,
            "timestamp": "2024-01-01T01:00:00Z",
            "order_type": "trailing_stop",
            "order_status": "partial_fill",
        },
        {
            "symbol": "AAPL",
            "side": "sell",
            "qty": 4,
            "price": 13,
            "timestamp": "2024-01-01T01:30:00Z",
            "order_type": "limit",
            "order_status": "filled",
        },
    ]

    trades = build_trades_from_events(events)

    assert len(trades) == 2

    first, second = trades
    assert first["qty"] == 5
    assert first["entry_price"] == 10
    assert first["exit_price"] == 12
    assert first["exit_time"] == "2024-01-01T01:00:00+00:00"
    assert first["order_type"] == "trailing_stop"
    assert first["exit_reason"] == "TrailingStop"

    assert second["qty"] == 5
    assert second["entry_price"] == 11
    assert pytest.approx(second["exit_price"], rel=1e-6) == 12.8
    assert second["exit_time"] == "2024-01-01T01:30:00+00:00"
    assert all(trade["order_status"] == "filled" for trade in trades)
    assert all(trade["entry_time"] for trade in trades)


def test_partial_exit_accumulates_and_sets_exit_price():
    events = [
        {"symbol": "TSLA", "side": "buy", "qty": 10, "price": 20, "timestamp": "2024-02-01T00:00:00Z"},
        {"symbol": "TSLA", "side": "sell", "qty": 4, "price": 21, "timestamp": "2024-02-01T01:00:00Z"},
        {"symbol": "TSLA", "side": "sell", "qty": 6, "price": 22, "timestamp": "2024-02-01T02:00:00Z"},
    ]

    trades = build_trades_from_events(events)

    assert len(trades) == 1
    trade = trades[0]
    assert trade["qty"] == 10
    assert trade["entry_price"] == 20
    assert pytest.approx(trade["exit_price"], rel=1e-6) == 21.6
    assert trade["exit_time"] == "2024-02-01T02:00:00+00:00"
    assert trade["order_status"] == "filled"
    assert trade["exit_price"] != 0
    assert trade["entry_time"] != ""


def test_trailing_stop_exit_sets_type_and_reason():
    events = [
        {"symbol": "MSFT", "side": "buy", "qty": 2, "price": 100, "timestamp": "2024-03-01T00:00:00Z", "order_type": "market"},
        {
            "symbol": "MSFT",
            "side": "sell",
            "qty": 2,
            "price": 101,
            "timestamp": "2024-03-02T00:00:00Z",
            "order_type": "trailing_stop",
            "order_status": "filled",
        },
    ]

    trades = build_trades_from_events(events)
    assert len(trades) == 1
    trade = trades[0]
    assert trade["order_type"] == "trailing_stop"
    assert trade["exit_reason"] == "TrailingStop"
    assert trade["order_status"] == "filled"


def test_order_type_uses_alpaca_type_not_status():
    events = [
        {"symbol": "AMD", "side": "buy", "qty": 1, "price": 50, "timestamp": "2024-04-01T00:00:00Z"},
        {
            "symbol": "AMD",
            "side": "sell",
            "qty": 1,
            "price": 55,
            "timestamp": "2024-04-01T01:00:00Z",
            "order_type": "limit",
            "order_status": "partial_fill",
        },
    ]

    trades = build_trades_from_events(events)
    assert len(trades) == 1
    trade = trades[0]
    assert trade["order_type"] == "limit"
    assert trade["order_status"] == "partial_fill"
    assert trade["order_type"] not in {"filled", "partial_fill", "order_type"}
    assert trade["order_type"] in {"limit", "market", "trailing_stop", "stop", "stop_limit", "stop_loss", "take_profit"}


def test_backfill_uses_exit_order_metadata_and_sanitizes(tmp_path, monkeypatch):
    dest = tmp_path / "trades_log.csv"

    events = [
        {"symbol": "AMD", "side": "buy", "qty": 1, "price": 50, "timestamp": "2024-04-01T00:00:00Z", "order_id": "entry-1"},
        {
            "symbol": "AMD",
            "side": "sell",
            "qty": 1,
            "price": 55,
            "timestamp": "2024-04-01T01:00:00Z",
            "order_id": "exit-1",
            "order_status": "partially_filled",
            "order_type": "filled",
        },
    ]

    monkeypatch.setattr("scripts.backfill_trades_log.gather_fill_events", lambda *a, **k: events)

    def fake_fetch_order_metadata(order_ids, **kwargs):
        return {"exit-1": {"type": "stop_limit", "status": "partially_filled"}}

    monkeypatch.setattr("scripts.backfill_trades_log._fetch_order_metadata", fake_fetch_order_metadata)

    backfill(1, dest, merge=True)

    df = pd.read_csv(dest)
    assert df.shape[0] == 1
    row = df.iloc[0].to_dict()
    assert row["exit_order_id"] == "exit-1"
    assert row["entry_order_id"] == "entry-1"
    assert row["order_type"] == "stop_limit"
    assert row["order_type"] not in {"filled", "partially_filled", "order_type"}
    assert row["order_status"] == "filled"


def test_backfill_writes_atomic(tmp_path, monkeypatch, dummy_events):
    dest = tmp_path / "trades_log.csv"

    monkeypatch.setattr("scripts.backfill_trades_log.gather_fill_events", lambda *a, **k: dummy_events)

    backfill(1, dest, merge=True)

    assert dest.exists()
    df = pd.read_csv(dest)
    assert list(df.columns)[:5] == ["symbol", "qty", "entry_price", "exit_price", "entry_time"]
    assert df.shape[0] == 1
