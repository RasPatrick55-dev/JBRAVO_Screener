from copy import deepcopy
import json
import socket

import pytest

from scripts.strategy_execution_broker import PaperBroker,BrokerError

pytestmark=pytest.mark.alpaca_optional
ENV=dict(APCA_API_KEY_ID='inert',APCA_API_SECRET_KEY='inert',
    APCA_API_BASE_URL='https://paper-api.alpaca.markets',JBR_EXEC_PAPER='true')
ACCOUNT=dict(id='00000000-0000-0000-0000-000000000001',status='ACTIVE',currency='USD',
    trading_blocked=False,account_blocked=False,trade_suspended_by_user=False,cash='100000',buying_power='400000')


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*a,**kw):raise AssertionError('External connection forbidden in broker unit tests')
    monkeypatch.setattr(socket.socket,'connect',deny)


class Response:
    def __init__(self,value,status=200):
        self.value=value;self.status_code=status;self.content=json.dumps(value).encode();self.closed=False
    def json(self):return deepcopy(self.value)
    def close(self):self.closed=True


class Session:
    def __init__(self,*responses):self.responses=list(responses);self.calls=[]
    def request(self,*args,**kw):
        self.calls.append((args,kw));value=self.responses.pop(0)
        if isinstance(value,Exception):raise value
        return value
    def close(self):pass


@pytest.mark.parametrize('changes',[dict(APCA_API_BASE_URL='https://api.alpaca.markets'),
    dict(APCA_API_BASE_URL='https://paper-api.alpaca.markets.evil.test'),dict(JBR_EXEC_PAPER='false'),
    dict(JBR_EXEC_PAPER=''),dict(APCA_API_SECRET_KEY='')])
def test_exact_native_paper_context_required_before_any_request(changes):
    session=Session()
    with pytest.raises(ValueError):PaperBroker(ENV|changes,session=session)
    assert session.calls==[]


def test_authentication_binds_actual_account_and_checks_suspension():
    broker=PaperBroker(ENV,session=Session(Response(ACCOUNT),Response(ACCOUNT|dict(id='00000000-0000-0000-0000-000000000002'))))
    assert broker.account()['cash']=='100000' and broker.account_id==ACCOUNT['id']
    with pytest.raises(ValueError,match='account_changed'):broker.account()
    other=PaperBroker(ENV,session=Session(Response(ACCOUNT|dict(trading_blocked=True))))
    with pytest.raises(ValueError,match='not_eligible'):other.account()


def test_transport_timeout_is_one_request_and_error_does_not_expose_provider_text():
    session=Session(TimeoutError('sensitive provider exception text'))
    broker=PaperBroker(ENV,session=session)
    with pytest.raises(BrokerError,match='outcome_unknown') as result:broker.submit_order(dict(symbol='PLUG'))
    assert 'sensitive' not in str(result.value) and len(session.calls)==1


def test_reference_capture_keeps_actual_after_hours_timestamp_and_explicit_missing_symbol():
    bar=dict(t='2026-10-23T23:53:00Z',o=43.79,h=43.79,l=43.6984,c=43.6984,v=476)
    response=Response(dict(bars={'ALK':[bar]},next_page_token=None));session=Session(response)
    broker=PaperBroker(ENV,session=session)
    result=broker.references(['ALK','SRRK'],'2026-10-23',now='2026-10-24T00:15:00Z')
    assert result['ALK']['last_bar_at']==bar['t'] and result['ALK']['price']==43.6984
    assert result['SRRK']['status']=='NO_AFTER_HOURS_REFERENCE'
    args,kw=session.calls[0]
    assert args==('GET','https://data.alpaca.markets/v2/stocks/bars')
    assert (kw['params']['feed'],kw['params']['adjustment'],kw['params']['currency'])==('sip','raw','USD')
    assert kw['params']['start']=='2026-10-23T16:00:00-04:00'
    assert response.closed


def test_truncated_reference_population_does_not_fall_back_to_iex_or_daily_close():
    session=Session(Response(dict(bars={},next_page_token='more')))
    broker=PaperBroker(ENV,session=session)
    with pytest.raises(ValueError,match='incomplete'):
        broker.references(['ALK'],'2026-10-23',now='2026-10-24T00:15:00Z')
    assert len(session.calls)==1


def test_unknown_position_snapshot_does_not_mean_flat_or_trigger_other_provider_calls():
    broker=PaperBroker(ENV,session=Session(Response(None)))
    with pytest.raises(ValueError):broker.positions()
    assert broker.requests==1


def test_stale_quote_or_noncanonical_request_is_rejected():
    broker=PaperBroker(ENV,session=Session(Response(dict(quotes={'ALK':dict(t='2026-10-26T08:00:00Z',bp=44,ap=44.1)}))))
    with pytest.raises(ValueError,match='stale_or_future'):
        broker.quotes(['ALK'],now='2026-10-26T08:02:00Z')
    with pytest.raises(ValueError,match='scope_or_budget'):
        broker.request('POST','/v2/account/configurations',payload={})
    assert broker.requests==1


def test_readonly_order_absence_is_distinct_from_failed_broker_lookup():
    broker=PaperBroker(ENV,session=Session(Response({},404),Response({},503)))
    assert broker.find_by_client_order_id('jbr2-'+'a'*40) is None
    with pytest.raises(BrokerError,match='http_503'):broker.find_by_client_order_id('jbr2-'+'a'*40)
