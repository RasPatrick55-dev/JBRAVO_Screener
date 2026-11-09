"""Synthetic one-GET diagnostics. No credentials, external I/O or broker state."""
from datetime import datetime, timedelta, timezone
import json
import socket

import pytest
from requests import Response

from scripts import strategy_quote_diagnostics as diagnostics
from scripts import strategy_quote_probe as probe
from tests.fixtures.ownership_native_broker import PaperBroker
from tests.test_strategy_execution_broker import ENV

pytestmark = pytest.mark.alpaca_optional
NOW = datetime(2026, 11, 9, 10, 17, tzinfo=timezone.utc)
SECRET = 'sentinel-never-export-value'


class Session:
    def __init__(self, value=None, status=200, error=None, raw=None):
        self.calls = []; self.closed = False; self.error = error
        self.response = Response(); self.response.status_code = status
        self.response._content = raw if raw is not None else json.dumps(value).encode()
        self.response._content_consumed = True

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a, **kw): raise AssertionError('External network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', deny)


def quote(age=0):
    return {'quotes': {'ARDT': {'t': (NOW - timedelta(seconds=age)).isoformat(), 'bp': 15, 'ap': 16}}}


@pytest.mark.parametrize('age,expected', [(0,'QUOTE_AVAILABLE'), (60,'QUOTE_AVAILABLE'),
    (60.001,'QUOTE_STALE'), (-1,'QUOTE_AVAILABLE'), (-1.001,'QUOTE_FUTURE'), (216000,'QUOTE_STALE')])
def test_exact_native_config_feed_and_timestamps(age, expected):
    session = Session(quote(age))
    value = probe.probe_quote(PaperBroker, ENV | {'ALPACA_DATA_FEED':'iex'}, diagnostics,
        session=session, now=lambda:NOW)
    assert value['quote_status'] == expected and value['quote_feed'] == 'iex'
    assert value['quote_age_seconds'] == age
    assert value['data_get_attempts'] == 1 and session.closed
    args, kwargs = session.calls[0]
    assert args == ('GET',probe.QUOTE_URL)
    assert kwargs['params'] == {'symbols':'ARDT','feed':'iex'}
    assert kwargs['stream'] is True and kwargs['allow_redirects'] is False
    assert not any(k in value for k in ('bid','ask','account_id','headers','message'))


@pytest.mark.parametrize('message,reason', [
    ('subscription does not permit querying recent SIP data','SIP_SUBSCRIPTION_NOT_PERMITTED'),
    ('subscription does not permit recent SIP data','SIP_SUBSCRIPTION_NOT_PERMITTED'),
    ('subscription does not permit querying recent SIP data '+SECRET,None),
    (SECRET,None), ({'secret':SECRET},None),
])
@pytest.mark.parametrize('feed', ['sip','iex'])
def test_forbidden_classification_is_exact_and_no_message_is_exported(message, reason, feed):
    session = Session({'message':message,'token':SECRET},status=403)
    value = probe.probe_quote(PaperBroker, ENV | {'LIMIT_PRICE_FEED':feed}, diagnostics,
        session=session, now=lambda:NOW)
    assert value['quote_status']=='QUOTE_FORBIDDEN' and value['http_status']==403
    assert value.get('provider_reason') == (reason if feed=='sip' else None)
    assert SECRET not in json.dumps(value)
    assert len(session.calls)==1


@pytest.mark.parametrize('payload', [{}, {'quotes':{}}, {'quotes':None},
    {'quotes':{'ARDT':None}}, {'quotes':{'ARDT':{'t':SECRET,'bp':1,'ap':2}}}, []])
def test_malformed_response_never_exports_raw_content(payload):
    session = Session(payload)
    value = probe.probe_quote(PaperBroker, ENV, diagnostics, session=session, now=lambda:NOW)
    assert value['quote_status']!='QUOTE_AVAILABLE'
    assert SECRET not in json.dumps(value) and len(session.calls)==1


def test_transport_error_is_one_attempt_and_suppresses_message():
    session = Session(error=RuntimeError('https://private.invalid/'+SECRET))
    value = probe.probe_quote(PaperBroker, ENV, diagnostics, session=session, now=lambda:NOW)
    assert value['quote_status']=='QUOTE_TRANSPORT_UNAVAILABLE'
    assert SECRET not in json.dumps(value) and len(session.calls)==1


def test_live_context_denied_before_network():
    session=Session(quote())
    value=probe.probe_quote(PaperBroker,ENV | {'APCA_API_BASE_URL':'https://api.alpaca.markets'},
        diagnostics,session=session,now=lambda:NOW)
    assert value['quote_status']=='QUOTE_PROBE_UNAVAILABLE' and session.calls==[]


@pytest.mark.parametrize('method,url,params', [
    ('GET','https://paper-api.alpaca.markets/v2/account',{'symbols':'ARDT','feed':'sip'}),
    ('GET','https://paper-api.alpaca.markets/v2/orders',{'symbols':'ARDT','feed':'sip'}),
    ('POST',probe.QUOTE_URL,{'symbols':'ARDT','feed':'sip'}),
    ('GET',probe.QUOTE_URL,{'symbols':'OTHER','feed':'sip'}),
    ('GET',probe.QUOTE_URL,{'symbols':'ARDT','feed':'other'}),
    ('GET',probe.QUOTE_URL,{'symbols':'ARDT','feed':'sip','extra':SECRET}),
])
def test_capability_denies_other_endpoints_symbols_feeds_and_methods(method,url,params):
    session=Session(quote())
    transport=probe.QuoteOnlySession(session,utc=PaperBroker.quotes.__globals__['utc'],now=lambda:NOW)
    with pytest.raises(ValueError,match='scope_denied'):
        transport.request(method,url,params=params,json=None,allow_redirects=False)
    assert transport.calls==0 and session.calls==[]


def test_second_request_denied_and_response_size_bounded():
    session=Session(quote())
    transport=probe.QuoteOnlySession(session,utc=PaperBroker.quotes.__globals__['utc'],now=lambda:NOW)
    kwargs=dict(params={'symbols':'ARDT','feed':'sip'},json=None,allow_redirects=False)
    transport.request('GET',probe.QUOTE_URL,**kwargs)
    with pytest.raises(ValueError,match='scope_denied'):transport.request('GET',probe.QUOTE_URL,**kwargs)
    large=Session(raw=b'x'*(probe.MAX_RESPONSE_BYTES+1))
    value=probe.probe_quote(PaperBroker,ENV,diagnostics,session=large,now=lambda:NOW)
    assert value['quote_status']!='QUOTE_AVAILABLE' and len(large.calls)==1


def test_zero_account_order_or_database_operations_are_explicit():
    value=probe.probe_quote(PaperBroker,ENV,diagnostics,session=Session(quote()),now=lambda:NOW)
    assert value['account_identity_verified'] is False
    assert all(value[k]==0 for k in ('account_requests','order_requests','database_operations','broker_mutations','runtime_mutations'))


def test_post_get_validation_clock_matches_reported_observation():
    values=iter([NOW-timedelta(seconds=20),NOW,NOW+timedelta(seconds=20)])
    # The call's now argument is old, but pinned native quotes validates self.now()
    # after the GET. Probe supplies exactly the captured response observation.
    result=probe.probe_quote(PaperBroker,ENV,diagnostics,session=Session(quote(-.5)),now=lambda:next(values))
    assert result['quote_status']=='QUOTE_AVAILABLE'
    assert result['observed_at']==NOW.isoformat() and result['quote_age_seconds']==-.5


def test_cleanup_failure_preserves_performed_get_count():
    class BrokenClose(Session):
        def close(self):raise RuntimeError(SECRET)
    session=BrokenClose(quote())
    result=probe.probe_quote(PaperBroker,ENV,diagnostics,session=session,now=lambda:NOW)
    assert result['data_get_attempts']==1 and len(session.calls)==1
    assert result['quote_status']=='QUOTE_PROBE_UNAVAILABLE'
    assert SECRET not in json.dumps(result)
