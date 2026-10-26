"""Actual Alpaca RFC3339 precision, including PythonAnywhere's Python 3.10."""
from datetime import datetime,timezone
import socket
import pytest
from scripts.strategy_execution import utc,initial_position
from scripts.strategy_execution_broker import PaperBroker
from tests.test_strategy_execution_broker import ENV,Session,Response

pytestmark=pytest.mark.alpaca_optional

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**kw):raise AssertionError('Network forbidden')
    monkeypatch.setattr(socket.socket,'connect',deny)

@pytest.mark.parametrize('fraction',['','.1','.12','.123','.1234','.12345','.123456','.1234567','.12345678','.123456789'])
@pytest.mark.parametrize('offset',['Z','+00:00','-04:00'])
def test_broker_timestamp_precision_normalizes_without_changing_second_or_offset(fraction,offset):
    result=utc('2026-10-26T11:20:12'+fraction+offset)
    expected=int((fraction[1:]+'000000')[:6]) if fraction else 0
    assert result==datetime(2026,10,26,15 if offset=='-04:00' else 11,20,12,expected,tzinfo=timezone.utc)

@pytest.mark.parametrize('value',['2026-10-26T11:20:12','2026-10-26','2026-10-26T11:20:12.1234567890Z',
    '2026-10-26T11:20:12Z\n','2026-10-26T11:20:12not-a-zone','2026-99-26T11:20:12Z'])
def test_naive_malformed_or_unsupported_precision_cannot_become_a_broker_timestamp(value):
    with pytest.raises(ValueError):utc(value)

def test_native_clock_response_and_fill_use_the_same_parser():
    value=dict(is_open=True,next_open='2026-10-27T09:30:00-04:00',next_close='2026-10-26T16:00:00-04:00',
        timestamp='2026-10-26T11:20:12.685343564-04:00')
    broker=PaperBroker(ENV,session=Session(Response(value)))
    assert broker.clock()==value
    position=initial_position(dict(symbol='SRRK',next_session='2026-10-26',signal_atr14_raw=1.5),
        quantity=10,fill_price=100,filled_at='2026-10-26T04:00:01.123456789-04:00')
    assert position['entry_price']==100 and position['atr_stop']==98.5
