"""Offline tests against exact protected-native baselines and minimal candidates."""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest
pytestmark = pytest.mark.alpaca_optional

from tests.test_strategy_execution_broker import ENV, Response, Session
from tests.test_strategy_execution_engine import setup, enter
from scripts import strategy_order_lifecycle as lifecycle
from scripts import strategy_order_ownership_install as install
from scripts import strategy_pending_stop_replacement as pending

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'tests/fixtures'
NOW = datetime(2026, 11, 9, 9, 49, tzinfo=timezone.utc)
SECRET = 'private-sentinel-never-export-this'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a, **kw):
        raise AssertionError('No network permitted in quote diagnostics tests')
    monkeypatch.setattr(socket.socket, 'connect', deny)


@pytest.fixture
def modules(monkeypatch):
    import scripts
    from scripts import strategy_quote_diagnostics as diagnostics
    from tests.fixtures.ownership_quote_safe_monitor import quote_safe_engine
    rules = load('diagnostic_native_rules', ROOT / 'ownership_native_rules.py')
    monkeypatch.setattr(scripts, 'strategy_execution', rules)
    monkeypatch.setitem(sys.modules, 'scripts.strategy_execution', rules)
    baseline_broker = load('diagnostic_baseline_broker', ROOT / 'ownership_native_broker.py')
    baseline_engine = load('diagnostic_baseline_engine', ROOT / 'ownership_native_engine.py')
    Engine, PaperBroker = diagnostics.execution_components(baseline_broker.PaperBroker, baseline_engine.Engine)
    broker = SimpleNamespace(PaperBroker=PaperBroker, quote_failure_details=diagnostics.quote_failure_details)
    engine = SimpleNamespace(Engine=Engine)
    monitor = SimpleNamespace(quote_safe_engine=quote_safe_engine)
    baseline_monitor = SimpleNamespace(quote_safe_engine=quote_safe_engine)
    return SimpleNamespace(broker=broker, engine=engine, monitor=monitor, rules=rules,
        lifecycle=lifecycle, diagnostics=diagnostics,
        baseline_engine=baseline_engine, baseline_broker=baseline_broker,
        baseline_monitor=baseline_monitor)


def paper(module, payload, *, status=200, now=NOW, feed='sip'):
    response = payload if isinstance(payload, Exception) else Response(payload, status)
    session = Session(response)
    broker = module.PaperBroker(ENV | {'LIMIT_PRICE_FEED': feed}, session=session, now=lambda: now)
    return broker, session


def quote(stamp=NOW, **fields):
    return dict(quotes={'TEST': dict(t=stamp.isoformat(), bp=100, ap=101) | fields})


@pytest.mark.parametrize('payload,status,expected', [
    ({'message': SECRET, 'url': 'https://secret.invalid/' + SECRET}, 403, 'QUOTE_FORBIDDEN'),
    ({'message': SECRET}, 401, 'QUOTE_HTTP_ERROR'),
    ({'message': SECRET}, 500, 'QUOTE_HTTP_ERROR'),
    (quote(NOW - timedelta(seconds=60, microseconds=1)), 200, 'QUOTE_STALE_OR_FUTURE'),
    (quote(NOW + timedelta(seconds=1, microseconds=1)), 200, 'QUOTE_STALE_OR_FUTURE'),
    ({}, 200, 'QUOTE_MISSING'),
    ({'quotes': None}, 200, 'QUOTE_MISSING'),
    ({'quotes': {'TEST': None}}, 200, 'QUOTE_INVALID'),
    ({'quotes': {'TEST': {}}}, 200, 'QUOTE_INVALID'),
    (quote(bp=SECRET), 200, 'QUOTE_INVALID'),
    (quote(bp=102), 200, 'QUOTE_INVALID'),
    (quote(t=SECRET), 200, 'QUOTE_INVALID'),
    ({'quotes': {'OTHER': dict(t=NOW.isoformat(), bp=1, ap=2)}}, 200, 'QUOTE_INVALID'),
    (TimeoutError('https://secret.invalid/' + SECRET), 200, 'QUOTE_TRANSPORT_UNAVAILABLE'),
])
def test_fixed_quote_failures_do_not_retry_or_leak(modules, payload, status, expected):
    broker, session = paper(modules.broker, payload, status=status, feed='iex')
    with pytest.raises((ValueError, KeyError, TypeError)) as caught:
        broker.quotes({'TEST'}, now=NOW)
    details = broker.quote_failure
    assert details['quote_failure_code'] == expected
    assert details['quote_feed'] == 'iex'
    assert set(details) <= {'quote_failure_code', 'quote_feed', 'http_status'}
    assert SECRET not in json.dumps(details) and 'https://' not in json.dumps(details)
    if status != 200:
        assert details['http_status'] == status
    assert len(session.calls) == 1 and broker.requests == 1
    assert session.calls[0][1]['params']['feed'] == 'iex'


@pytest.mark.parametrize('age', [-1, 0, 60])
def test_original_freshness_boundaries_and_prices_unchanged(modules, age):
    payload = quote(NOW - timedelta(seconds=age))
    candidate, _ = paper(modules.broker, payload)
    baseline, _ = paper(modules.baseline_broker, payload)
    assert candidate.quotes({'TEST'}, now=NOW) == baseline.quotes({'TEST'}, now=NOW)


def test_empty_symbol_population_makes_no_request(modules):
    broker, session = paper(modules.broker, {})
    assert broker.quotes(set(), now=NOW) == {}
    assert broker.requests == 0 and session.calls == []


@pytest.mark.parametrize('stamp', [NOW-timedelta(seconds=61), NOW+timedelta(seconds=2)])
def test_stale_and_future_keep_exact_exception_type_and_message(modules, stamp):
    caught = []
    for module in [modules.baseline_broker, modules.broker]:
        broker, _ = paper(module, quote(stamp))
        with pytest.raises(ValueError) as error:
            broker.quotes({'TEST'}, now=NOW)
        caught.append((type(error.value), error.value.args))
    assert caught[0] == caught[1] == (ValueError, ('strategy_quote_stale_or_future',))


@pytest.mark.parametrize('changes,feed', [({},'sip'), ({'ALPACA_DATA_FEED':'iex'},'iex'),
    ({'LIMIT_PRICE_FEED':'sip','ALPACA_DATA_FEED':'iex'},'sip')])
def test_feed_selection_precedence_unchanged(modules, changes, feed):
    session = Session(Response(quote()))
    broker = modules.broker.PaperBroker(ENV | changes, session=session, now=lambda:NOW)
    assert broker.quotes({'TEST'}, now=NOW)['TEST']['feed'] == feed
    assert session.calls[0][1]['params']['feed'] == feed


def test_malformed_list_exception_boundary_is_not_broadened(modules):
    for module in [modules.baseline_broker, modules.broker]:
        broker, session = paper(module, [])
        with pytest.raises(AttributeError):
            broker.quotes({'TEST'}, now=NOW)
        assert len(session.calls) == 1


@pytest.mark.parametrize('error', [ValueError(SECRET), ValueError('https://secret.invalid/' + SECRET),
    ValueError({'headers': SECRET}), RuntimeError(SECRET)])
def test_unrecognized_errors_are_fixed_and_suppress_all_arbitrary_text(modules, error):
    error.quote_feed = SECRET
    error.http_status = SECRET
    error.broker_message = SECRET
    assert modules.broker.quote_failure_details(error) == {
        'quote_failure_code': 'QUOTE_FAILURE_UNKNOWN', 'quote_feed': 'unknown'}


@pytest.mark.parametrize('status', [True, False, 99, 600, '403', None, {'secret': SECRET}])
def test_only_safe_integer_http_status_is_exported(modules, status):
    error = ValueError(SECRET)
    error.http_status = status
    assert 'http_status' not in modules.broker.quote_failure_details(error)


def make_position(module, monitor, lifetime):
    _, store, broker, handoff = setup()
    install.install_monitor(monitor, module.Engine, lifetime, pending_replacement=pending)
    engine = monitor.quote_safe_engine(module.Engine)(store, broker, guard=lambda *a: True,
        now=lambda: broker.now)
    enter(engine, broker, handoff)
    broker.fill('entry', broker.posts[0]['qty'])
    engine.tick(now=broker.now)
    engine.pipeline_orders = True
    engine.events.clear()
    return engine, store, broker


def wire_quotes(broker, module, mode):
    if mode == '403':
        payload, status = {'message': SECRET}, 403
    elif mode == 'empty':
        payload, status = {'quotes': {}}, 200
    else:
        stamp = broker.now - timedelta(seconds=61) if mode == 'stale' else broker.now + timedelta(seconds=2)
        symbol = next(iter(broker.held))
        payload, status = {'quotes': {symbol: {'t': stamp.isoformat(), 'bp': 1, 'ap': 2}}}, 200
    adapter, session = paper(module, payload, status=status, now=broker.now)
    broker.quotes = adapter.quotes
    broker.quote_feed = adapter.quote_feed
    broker.quote_failure = None
    def observed(*args, **kwargs):
        try: return adapter.quotes(*args, **kwargs)
        finally: broker.quote_failure = getattr(adapter, 'quote_failure', None)
    broker.quotes = observed
    return session


@pytest.mark.parametrize('mode,expected', [('403','QUOTE_FORBIDDEN'), ('stale','QUOTE_STALE_OR_FUTURE'),
    ('future','QUOTE_STALE_OR_FUTURE'), ('empty','QUOTE_MISSING')])
@pytest.mark.parametrize('queued', [False, True])
def test_native_monitor_stop_behavior_unchanged_and_no_stale_price_exit(modules, mode, expected, queued):
    observed = []
    for engine_module, broker_module, monitor, lifetime in [
        (modules.baseline_engine, modules.baseline_broker, modules.baseline_monitor, lifecycle),
        (modules.engine, modules.broker, modules.monitor, modules.lifecycle),
    ]:
        engine, store, broker = make_position(engine_module, monitor, lifetime)
        if queued:
            signal = store.signals()[0]
            signal['payload'].update(pending_exit_reasons=['RSI_EXIT'], pending_exit_session='2026-10-26')
            store.save(signal)
        session = wire_quotes(broker, broker_module, mode)
        before = deepcopy((broker.posts, broker.patches, broker.cancels))
        result = engine.tick(now=broker.now)
        assert (broker.posts, broker.patches, broker.cancels) == before
        assert len(session.calls) == 1
        if queued:
            assert any(e['code'] == 'EXIT_QUOTE_UNAVAILABLE' for e in result['events'])
        observed.append((deepcopy(store.rows), deepcopy(store.outbox), result))
    assert observed[0][:2] == observed[1][:2]
    assert [e['code'] for e in observed[0][2]['events']] == [e['code'] for e in observed[1][2]['events']]
    failures = [e for e in observed[1][2]['events'] if e['code'] == 'STRATEGY_QUOTES_UNAVAILABLE']
    assert failures and all(e['quote_failure_code'] == expected and e['quote_feed'] == 'sip' for e in failures)
    assert SECRET not in json.dumps(observed[1][2])


def test_unknown_caught_error_is_sanitized_and_uncaught_error_still_propagates(modules):
    engine, store, broker = make_position(modules.engine, modules.monitor, modules.lifecycle)
    def caught(*a, **kw):
        broker.quote_failure = modules.diagnostics.quote_failure_details(ValueError(SECRET))
        raise ValueError(SECRET)
    broker.quotes = caught
    engine.tick(now=broker.now)
    assert any(e.get('quote_failure_code') == 'QUOTE_FAILURE_UNKNOWN' for e in engine.events)
    assert SECRET not in json.dumps(engine.events)
    def uncaught(*a, **kw):
        raise RuntimeError(SECRET)
    broker.quotes = uncaught
    with pytest.raises(RuntimeError):
        engine.tick(now=broker.now)


def test_missing_protective_stop_is_reconciled_with_unavailable_quote(modules):
    observations = []
    for engine_module, broker_module, monitor, lifetime in [
        (modules.baseline_engine, modules.baseline_broker, modules.baseline_monitor, lifecycle),
        (modules.engine, modules.broker, modules.monitor, modules.lifecycle),
    ]:
        engine, store, broker = make_position(engine_module, monitor, lifetime)
        signal = store.signals()[0]
        signal['payload'].update(pending_exit_reasons=['RSI_EXIT'], pending_exit_session='2026-10-26')
        store.save(signal)
        old = next(o for o in broker.orders.values() if o['type']=='stop')
        old['status'] = 'canceled'
        wire_quotes(broker, broker_module, '403')
        engine.tick(now=broker.now)
        assert len(broker.posts) == 3 and broker.posts[-1]['type'] == 'stop'
        assert broker.posts[-1]['qty'] == broker.posts[0]['qty']
        assert not broker.posts[-1]['extended_hours']
        assert broker.cancels == []
        observations.append(deepcopy((store.rows, store.outbox, broker.posts)))
    assert observations[0] == observations[1]


def test_exact_inherited_native_methods_and_globals_stay_unchanged(modules):
    for name in ('tick', 'sync', 'manage', 'completed', '_perform', '_check_order'):
        assert getattr(modules.engine.Engine, name) is getattr(modules.baseline_engine.Engine, name)
    for name in ('request','submit_order','cancel_order','replace_order','account','positions'):
        assert getattr(modules.broker.PaperBroker,name) is getattr(modules.baseline_broker.PaperBroker,name)
    assert issubclass(lifecycle.monitor_engine(modules.engine.Engine), modules.engine.Engine)


def test_metadata_clears_on_next_quote_and_empty_input(modules):
    session = Session(Response({'message':SECRET},403), Response(quote()), Response({'quotes':{}}))
    broker = modules.broker.PaperBroker(ENV, session=session, now=lambda:NOW)
    with pytest.raises(ValueError): broker.quotes({'TEST'},now=NOW)
    assert broker.quote_failure['quote_failure_code']=='QUOTE_FORBIDDEN'
    broker.quotes({'TEST'},now=NOW)
    assert broker.quote_failure is None
    broker.quotes({'TEST'},now=NOW)
    assert broker.quote_failure['quote_failure_code']=='QUOTE_MISSING'
    broker.quotes(set(),now=NOW)
    assert broker.quote_failure is None and len(session.calls)==3


@pytest.mark.parametrize('value',[None, SECRET, {'quote_failure_code':SECRET,'quote_feed':SECRET,'http_status':SECRET,'headers':SECRET},
    {'quote_failure_code':[], 'quote_feed':{},'http_status':True,'broker_message':SECRET}])
def test_event_boundary_suppresses_all_unapproved_fields(modules,value):
    details=modules.diagnostics.safe_details(value)
    assert set(details)<={'quote_failure_code','quote_feed','http_status'}
    assert SECRET not in json.dumps(details)


def test_factory_rejects_unpinned_interfaces(modules):
    class Changed(modules.baseline_engine.Engine):
        def tick(self, **kwargs): return {}
    with pytest.raises(ValueError,match='native_interface_changed'):
        modules.diagnostics.execution_components(modules.baseline_broker.PaperBroker,Changed)


def test_native_fixture_binding():
    import hashlib
    assert hashlib.sha256((ROOT/'ownership_native_broker.py').read_bytes()).hexdigest()==(
        'bf6835a7721b0485b8a2ca6528cb8295193c4431c38542659ae8375abe45a931')
