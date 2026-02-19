import os
import types
import importlib.util
from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts import screener

pytestmark = pytest.mark.alpaca_optional


def test_merge_meta():
    bars = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "open": [100.0, 200.0],
            "high": [105.0, 205.0],
            "low": [99.0, 198.0],
            "close": [104.0, 203.0],
            "volume": [1_000_000, 2_000_000],
        }
    )
    meta = {
        "AAPL": {"exchange": "NASDAQ", "asset_class": "US_EQUITY"},
        "MSFT": {"exchange": "NYSE", "asset_class": "US_EQUITY"},
    }

    merged = screener.merge_asset_metadata(bars, meta)

    assert set(merged["exchange"]) == {"NASDAQ", "NYSE"}
    assert set(merged["kind"]) == {"EQUITY"}
    assert merged["tradable"].all()


def test_paginate(monkeypatch):
    original_request = screener.StockBarsRequest
    original_timeframe = screener.TimeFrame

    class DummyRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(screener, "StockBarsRequest", DummyRequest)
    monkeypatch.setattr(screener, "TimeFrame", types.SimpleNamespace(Day="day"))
    monkeypatch.setattr(screener, "to_bars_df", lambda response: response.df.copy())
    monkeypatch.setattr(screener, "_normalize_bars_frame", lambda df: df)

    class FakeResponse:
        def __init__(self, rows, next_token=None):
            frame = pd.DataFrame(rows)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            self.df = frame
            self.data = {str(sym): [] for sym in frame["symbol"].unique()}
            self.next_page_token = next_token

    class FakeClient:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        def get_stock_bars(self, request):
            self.calls.append(request.kwargs)
            if not self.responses:
                raise AssertionError("No more responses queued")
            return self.responses.pop(0)

    page_one = [
        {
            "symbol": "AAPL",
            "timestamp": "2024-01-08T00:00:00Z",
            "open": 101,
            "high": 103,
            "low": 99,
            "close": 102,
            "volume": 1_000,
        },
        {
            "symbol": "AAPL",
            "timestamp": "2024-01-09T00:00:00Z",
            "open": 102,
            "high": 104,
            "low": 100,
            "close": 103,
            "volume": 1_100,
        },
    ]
    page_two = [
        {
            "symbol": "MSFT",
            "timestamp": "2024-01-08T00:00:00Z",
            "open": 201,
            "high": 204,
            "low": 199,
            "close": 203,
            "volume": 2_000,
        },
        {
            "symbol": "MSFT",
            "timestamp": "2024-01-09T00:00:00Z",
            "open": 203,
            "high": 205,
            "low": 200,
            "close": 204,
            "volume": 2_100,
        },
    ]

    client = FakeClient([FakeResponse(page_one, next_token="page-2"), FakeResponse(page_two)])

    request_kwargs = {
        "symbol_or_symbols": ["AAPL", "MSFT"],
        "timeframe": "day",
        "start": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "end": datetime(2024, 1, 11, tzinfo=timezone.utc),
        "feed": "iex",
        "adjustment": "raw",
    }

    batch_frames, page_count, paged, columns_desc, success = screener._collect_batch_pages(
        client, request_kwargs
    )

    monkeypatch.setattr(screener, "StockBarsRequest", original_request)
    monkeypatch.setattr(screener, "TimeFrame", original_timeframe)

    assert success is True
    assert page_count == 2
    assert paged is True
    assert "symbol" in columns_desc
    total_rows = sum(frame.shape[0] for frame in batch_frames)
    assert total_rows == 4


def test_gate_presets_parsing():
    parsed = screener.parse_args(
        ["--mode", "screener", "--gate-preset", "strict", "--relax-gates", "cross_or_rsi"]
    )
    assert parsed.gate_preset == "strict"
    assert parsed.relax_gates == "cross_or_rsi"

    defaults = screener.parse_args(["--mode", "screener"])
    assert defaults.gate_preset == "standard"
    assert defaults.relax_gates == "none"


@pytest.mark.skipif(
    not os.getenv("APCA_API_KEY_ID") or importlib.util.find_spec("alpaca_trade_api") is None,
    reason="Alpaca credentials or alpaca_trade_api not available",
)
def test_fetch_symbols_not_empty():
    df = screener.fetch_symbols()
    assert isinstance(df, pd.DataFrame)
    assert "symbol" in df.columns
    assert not df.empty
