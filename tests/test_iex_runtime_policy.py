"""Offline guard against reintroducing implicit paid-feed fallback."""
from datetime import date
import inspect
from types import SimpleNamespace
import pandas as pd
import pytest
from scripts import backtest, screener

pytestmark=pytest.mark.alpaca_optional


def test_backtest_missing_iex_never_requests_sip(monkeypatch):
    feeds=[]
    class Client:
        def get_stock_bars(self,request):
            feeds.append(request.feed)
            return SimpleNamespace(df=pd.DataFrame())
    monkeypatch.setattr(backtest,'data_client',Client())
    monkeypatch.setenv('BACKTEST_SIP_FALLBACK','true')
    assert backtest._fetch_bars_from_api('SPY',date(2026,11,13),3).empty
    assert len(feeds)==1 and str(feeds[0]).lower().endswith('iex')


def test_screener_has_no_implicit_sip_override():
    source=inspect.getsource(screener._fetch_daily_bars)
    assert 'feed_override="sip"' not in source
