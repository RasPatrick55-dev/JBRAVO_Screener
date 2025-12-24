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


def test_backfill_writes_atomic(tmp_path, monkeypatch, dummy_events):
    dest = tmp_path / "trades_log.csv"

    monkeypatch.setattr("scripts.backfill_trades_log.gather_fill_events", lambda *a, **k: dummy_events)

    backfill(1, dest, merge=True)

    assert dest.exists()
    df = pd.read_csv(dest)
    assert list(df.columns)[:5] == ["symbol", "qty", "entry_price", "exit_price", "entry_time"]
    assert df.shape[0] == 1
