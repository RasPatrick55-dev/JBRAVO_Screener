"""Offline Webull adapter/unchanged native paper monitor contract tests."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import socket
import subprocess
import sys

import pytest

from scripts import strategy_webull_quotes as webull
from tests.test_strategy_execution_broker import ENV, Response, Session
from tests.test_strategy_quote_diagnostics import modules, make_position

pytestmark = pytest.mark.alpaca_optional
NOW = datetime(2026, 11, 9, 14, 55, 24, 553000, tzinfo=timezone.utc)
SECRET = 'private-sentinel-never-export-this'


def test_adapter_imports_in_isolation_without_native_source_installation():
    code = ('import runpy; runpy.run_path(' + repr(webull.__file__) + ')')
    result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and result.stdout == '' and result.stderr == ''


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a, **kw):
        raise AssertionError('No external connection in Webull adapter tests')
    monkeypatch.setattr(socket.socket, 'connect', deny)


def payload(symbol='SPY', stamp=NOW, **changes):
    return dict(symbol=symbol, quote_time=str(int(stamp.timestamp() * 1000)),
                bids=[dict(price='747.23', size='21')],
                asks=[dict(price='747.26', size='95')]) | changes


def provider(fetch=None, **kwargs):
    return webull.WebullQuotes(fetch or (lambda s: payload(s)),
                               non_display_authorized=True, now=lambda: NOW, **kwargs)


def test_actual_mcp_observation_values_preserve_time_and_single_exchange_qualification():
    # Price/size/time from Nov9 14:55UTC observation; dict schema from SDK1.2.6
    # formatter source. This is reconstructed evidence, not a captured raw response.
    result = webull.normalize(payload(quote_time='1794236124260'), symbol='SPY', observed=NOW)
    assert result['timestamp'] == '2026-11-09T14:55:24.260000+00:00'
    assert result['bid'] == '747.23' and result['ask'] == '747.26'
    assert result['is_nbbo'] is False and result['venue_scope'] == 'single_exchange'
    assert result['venue_basis'] == 'configured_nasdaq_basic_entitlement'
    assert result['timestamp_basis'] == 'provider_quote_time_unix_ms'


@pytest.mark.parametrize('seconds,valid', [(60, True), (60.001, False), (-1, True), (-1.001, False)])
def test_exact_existing_freshness_boundaries(seconds, valid):
    p = payload(stamp=NOW - timedelta(seconds=seconds))
    if valid:
        assert webull.normalize(p, symbol='SPY', observed=NOW)
    else:
        with pytest.raises(webull.QuoteError, match='QUOTE_STALE_OR_FUTURE'):
            webull.normalize(p, symbol='SPY', observed=NOW)


@pytest.mark.parametrize('stamp', [None, True, 1794236124, 1794236124260000,
    '2026-11-09T14:55:24Z', '1794236124260.0', 1794236124260.0, SECRET])
def test_no_timestamp_guessing_or_reception_timestamp_substitution(stamp):
    with pytest.raises(webull.QuoteError):
        webull.normalize(payload(quote_time=stamp), symbol='SPY', observed=NOW)


@pytest.mark.parametrize('bad', [None, [], {}, 'MCP formatted text',
    payload(symbol='OTHER'), payload(bids=[]), payload(asks=[]),
    payload(bids=[dict(price='747', size='1'), dict(price='746', size='1')]),
    payload(bids=[dict(price='748', size='1')]),
    payload(bids=[dict(price='NaN', size='1')]),
    payload(asks=[dict(price='Infinity', size='1')]),
    payload(bids=[dict(price=True, size='1')]),
    payload(bids=[dict(price='0', size='1')]),
    payload(bids=[dict(price='0.01', size='1')]),
    payload(bids=[dict(price='1e100', size='1')], asks=[dict(price='2e100', size='1')]),
    payload(asks=[dict(price='1e999', size='1')]),
    payload(bids=[dict(price='747', size='0')]),
    payload(bids=[dict(price='747', size='-1')]),
    payload(bids=[dict(price='747', size=SECRET)]),
])
def test_invalid_depth_or_identity_fail_closed(bad):
    with pytest.raises(webull.QuoteError):
        webull.normalize(bad, symbol='SPY', observed=NOW)


def test_per_symbol_isolation_sorted_calls_no_retry_or_fallback():
    calls = []
    def fetch(symbol):
        calls.append(symbol)
        if symbol == 'BAD':
            raise RuntimeError(SECRET)
        return payload(symbol, stamp=NOW - timedelta(seconds=61) if symbol == 'OLD' else NOW)
    q = provider(fetch)
    assert set(q.quotes({'SPY', 'BAD', 'OLD'}, now=NOW)) == {'SPY'}
    assert calls == ['BAD', 'OLD', 'SPY']
    assert q.failures == {'BAD': 'QUOTE_TRANSPORT_UNAVAILABLE', 'OLD': 'QUOTE_STALE_OR_FUTURE'}
    assert SECRET not in json.dumps(q.failures)
    assert q.requests == 3
    assert q.quotes(set(), now=NOW) == {} and q.failures == {} and q.requests == 3


def test_earlier_quote_rechecked_after_slow_later_request():
    times = iter([NOW, NOW, NOW + timedelta(seconds=61), NOW + timedelta(seconds=61)])
    q = webull.WebullQuotes(lambda s: payload(s, stamp=NOW if s == 'A' else NOW + timedelta(seconds=61)),
                           non_display_authorized=True, now=lambda: next(times))
    assert set(q.quotes({'A', 'B'}, now=NOW)) == {'B'}
    assert q.failures == {'A': 'QUOTE_STALE_OR_FUTURE'}


def test_clock_rollback_never_exposes_received_quote():
    times = iter([NOW, NOW, NOW - timedelta(seconds=1)])
    q = webull.WebullQuotes(lambda s: payload(s), non_display_authorized=True, now=lambda: next(times))
    assert q.quotes({'SPY'}, now=NOW) == {}
    assert q.failures == {'SPY': 'QUOTE_CLOCK_INVALID'}


def test_call_budget_never_resets_and_failures_clear_each_cycle():
    q = provider(max_requests=1)
    assert set(q.quotes({'A', 'B'}, now=NOW)) == {'A'}
    assert q.failures == {'B': 'QUOTE_BUDGET_EXHAUSTED'}
    assert q.quotes({'A'}, now=NOW) == {} and q.requests == 1
    assert q.failures == {'A': 'QUOTE_BUDGET_EXHAUSTED'}


def test_clock_rollback_after_later_receipt_above_start_fails_closed():
    times = iter([NOW, NOW + timedelta(seconds=30), NOW + timedelta(seconds=29)])
    q = webull.WebullQuotes(lambda s: payload(s), non_display_authorized=True, now=lambda: next(times))
    assert q.quotes({'SPY'}, now=NOW) == {}
    assert q.failures == {'SPY': 'QUOTE_CLOCK_INVALID'}


def test_intermediate_clock_rollback_is_rejected_even_when_final_clock_recovers():
    times = iter([NOW, NOW + timedelta(seconds=30), NOW + timedelta(seconds=29), NOW + timedelta(seconds=31)])
    q = webull.WebullQuotes(lambda s: payload(s), non_display_authorized=True, now=lambda: next(times))
    assert set(q.quotes({'A', 'B'}, now=NOW)) == {'A'}
    assert q.failures == {'B': 'QUOTE_CLOCK_INVALID'}


@pytest.mark.parametrize('symbols', [['SPY', 'SPY'], {'bad'}, {SECRET}, {'A','B','C','D','E'}, 'SPY', {None}])
def test_invalid_symbol_scope_rejected_before_transport(symbols):
    calls = []
    q = provider(lambda s: calls.append(s))
    with pytest.raises(webull.QuoteError):
        q.quotes(symbols, now=NOW)
    assert calls == []


def test_permission_gate_and_call_time_cannot_bypass_age():
    with pytest.raises(webull.QuoteError):
        webull.WebullQuotes(lambda s: payload(s), non_display_authorized=False)
    q = provider()
    with pytest.raises(webull.QuoteError, match='QUOTE_CLOCK_INVALID'):
        q.quotes({'SPY'}, now=NOW - timedelta(minutes=2))
    assert q.requests == 0


class MarketData:
    def __init__(self, response):
        self.response = response
        self.calls = []
    def get_quotes(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.parametrize('status,code', [(403, 'QUOTE_FORBIDDEN'), (401, 'QUOTE_HTTP_ERROR'), (500, 'QUOTE_HTTP_ERROR')])
def test_transport_fixed_request_no_provider_error_export(status, code):
    response = Response({'message': SECRET}, status)
    market = MarketData(response)
    with pytest.raises(webull.QuoteError, match=code):
        webull.DepthTransport(market)('SPY')
    assert response.closed
    assert market.calls == [dict(symbol='SPY', category='US_STOCK', depth=1, overnight_required=False)]


def test_transport_reads_bounded_json_and_closes():
    response = Response(payload())
    assert webull.DepthTransport(MarketData(response))('SPY') == payload()
    assert response.closed
    response.content = b'x' * (webull.MAX_RESPONSE_BYTES + 1)
    with pytest.raises(webull.QuoteError):
        webull.DepthTransport(MarketData(response))('SPY')
    with pytest.raises(webull.QuoteError, match='QUOTE_TRANSPORT_UNAVAILABLE') as caught:
        webull.DepthTransport(MarketData(RuntimeError(SECRET)))('SPY')
    assert SECRET not in str(caught.value)


def test_factory_inherits_every_order_risk_and_ownership_method(modules):
    E, B = webull.execution_components(modules.baseline_broker.PaperBroker, modules.baseline_engine.Engine,
                                        quote_factory=provider)
    for name in ('tick', 'sync', 'manage', 'completed', '_perform', '_check_order'):
        assert getattr(E, name) is getattr(modules.baseline_engine.Engine, name)
    for name in ('request', 'submit_order', 'cancel_order', 'replace_order', 'account', 'positions', 'references', 'close'):
        assert getattr(B, name) is getattr(modules.baseline_broker.PaperBroker, name)
    with pytest.raises(ValueError):
        B(ENV | {'APCA_API_BASE_URL': 'https://api.alpaca.markets'}, session=Session())


def test_provider_setup_failure_does_not_block_broker_initialization_or_reconcile(modules):
    calls = []
    def fail():
        calls.append(1)
        raise RuntimeError(SECRET)
    E, B = webull.execution_components(modules.baseline_broker.PaperBroker, modules.baseline_engine.Engine,
                                        quote_factory=fail)
    broker = B(ENV, session=Session())
    assert calls == []
    assert broker.quotes({'SPY'}, now=NOW) == {}
    assert calls == [1] and broker.quote_failures == {'SPY': 'QUOTE_TRANSPORT_UNAVAILABLE'}


def test_factory_rejects_changed_native_exit_price_or_order_sequence(modules):
    class Changed(modules.baseline_engine.Engine):
        def manage(self, *args, **kwargs):
            return None
    with pytest.raises(ValueError, match='native_interface_changed'):
        webull.execution_components(modules.baseline_broker.PaperBroker, Changed, quote_factory=provider)


@pytest.mark.parametrize('queued', [False, True])
@pytest.mark.parametrize('mode,expected', [('stale', 'QUOTE_STALE_OR_FUTURE'),
                                         ('huge', 'QUOTE_INVALID'), ('tiny', 'QUOTE_INVALID')])
def test_invalid_webull_cannot_trigger_exit_or_cancel_existing_stop(modules, queued, mode, expected):
    from types import SimpleNamespace
    def fetch(symbol):
        if mode == 'stale':
            return payload(symbol, stamp=broker.now - timedelta(minutes=3))
        price = '1e100' if mode == 'huge' else '0.01'
        return payload(symbol, stamp=broker.now, bids=[dict(price=price, size='1')],
                       asks=[dict(price=price, size='1')])
    E, B = webull.execution_components(modules.baseline_broker.PaperBroker, modules.baseline_engine.Engine,
        quote_factory=lambda: webull.WebullQuotes(
            fetch,
            non_display_authorized=True, now=lambda: broker.now))
    engine, store, broker = make_position(SimpleNamespace(Engine=E), modules.monitor, modules.lifecycle)
    if queued:
        signal = store.signals()[0]
        signal['payload'].update(pending_exit_reasons=['RSI_EXIT'], pending_exit_session='2026-10-26')
        store.save(signal)
    adapter = B(ENV, session=Session())
    def observed(symbols, *, now):
        value = adapter.quotes(symbols, now=now)
        broker.quote_failures = adapter.quote_failures
        return value
    broker.quotes = observed
    before = deepcopy((broker.posts, broker.patches, broker.cancels))
    result = engine.tick(now=broker.now)
    assert (broker.posts, broker.patches, broker.cancels) == before
    errors = [event for event in result['events'] if event['code'] == 'STRATEGY_QUOTES_UNAVAILABLE']
    assert errors and errors[0]['quote_feed'] == webull.FEED
    assert errors[0]['quote_is_nbbo'] is False
    assert set(errors[0]['quote_failures'].values()) == {expected}
    assert SECRET not in json.dumps(result)


def test_safe_failures_never_exports_unapproved_keys_or_codes():
    assert webull.safe_failures({'SPY': SECRET, SECRET: 'QUOTE_MISSING'}) == {}
    assert webull.safe_failures({'SPY': 'QUOTE_MISSING', 'BAD': []}) == {'SPY': 'QUOTE_MISSING'}
