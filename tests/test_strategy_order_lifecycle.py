"""Offline calendar contracts and races against the reviewed deployed Engine.

Bundled nonsecret engine/rules are exact reviewed deployment snapshots. Override
with JBR_LIFECYCLE_NATIVE_ENGINE / JBR_LIFECYCLE_NATIVE_RULES (same hashes only).
The runtime adapter deliberately rejects the different, older repository engine.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

import pytest

from scripts import strategy_order_lifecycle as lifecycle

pytestmark = pytest.mark.alpaca_optional


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def handoff_for(day='2026-11-09', close='16:00'):
    date = datetime.fromisoformat(day).date()
    pre = datetime.combine(date, datetime.min.time().replace(hour=4), lifecycle.NY)
    regular = pre.replace(hour=9, minute=30)
    handoff = dict(session=dict(next_session=day, next_open_at=regular.isoformat()),
                   execution_policy=dict(entry_open_at=pre.isoformat(),
                                         cancel_at=(pre + timedelta(minutes=35)).isoformat()))
    handoff['handoff_id'] = digest(handoff)
    return handoff, dict(date=day, open='09:30', close=close)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('No network in lifecycle tests')
    monkeypatch.setattr(socket.socket, 'connect', deny)


@pytest.mark.parametrize('day,close,expected', [
    ('2026-11-09', '16:00', '2026-11-09T21:00:00+00:00'),
    ('2026-11-27', '13:00', '2026-11-27T18:00:00+00:00'),
    ('2026-03-06', '16:00', '2026-03-06T21:00:00+00:00'),
    ('2026-03-09', '16:00', '2026-03-09T20:00:00+00:00'),
    ('2026-10-30', '16:00', '2026-10-30T20:00:00+00:00'),
])
def test_calendar_close_and_dst(day, close, expected):
    handoff, calendar = handoff_for(day, close)
    before = deepcopy(handoff)
    marker = lifecycle.marker_for(handoff, calendar)
    payload = dict(handoff=handoff, entry_lifecycle=marker)
    assert marker['cancel_at'] == expected
    assert lifecycle.cancellation_deadline(payload).isoformat() == expected
    assert handoff == before
    assert marker['owner'] == 'MONITOR_POSITIONS'


@pytest.mark.parametrize('change', [
    {'date': '2026-11-10'}, {'open': '04:00'}, {'close': '20:00'},
    {'close': '09:00'}, {'open': '09:30+00:00'}, {'date': 'garbage'},
])
def test_calendar_invalid(change):
    handoff, calendar = handoff_for()
    calendar.update(change)
    with pytest.raises(ValueError):
        lifecycle.marker_for(handoff, calendar)


def test_legacy_fallback_and_strict_missing():
    handoff, calendar = handoff_for()
    payload = dict(handoff=handoff)
    assert lifecycle.cancellation_deadline(payload) == datetime.fromisoformat(
        handoff['execution_policy']['cancel_at'])
    with pytest.raises(ValueError, match='marker_missing'):
        lifecycle.cancellation_deadline(payload, require_marker=True)
    for marker in (None, {}, 'bad'):
        with pytest.raises(ValueError):
            lifecycle.cancellation_deadline(dict(payload, entry_lifecycle=marker))


@pytest.mark.parametrize('field', ['owner', 'model', 'cancel_at', 'handoff_sha256', 'marker_sha256'])
def test_marker_tampering_fails_closed(field):
    handoff, calendar = handoff_for()
    marker = lifecycle.marker_for(handoff, calendar)
    marker[field] = 'changed'
    with pytest.raises(ValueError):
        lifecycle.cancellation_deadline(dict(handoff=handoff, entry_lifecycle=marker))


def test_handoff_tampering_and_session_conflict():
    handoff, calendar = handoff_for()
    marker = lifecycle.marker_for(handoff, calendar)
    handoff['execution_policy']['cancel_at'] = marker['cancel_at']
    with pytest.raises(ValueError):
        lifecycle.cancellation_deadline(dict(handoff=handoff, entry_lifecycle=marker))
    handoff['handoff_id'] = digest({k: v for k, v in handoff.items() if k != 'handoff_id'})
    with pytest.raises(ValueError, match='window_mismatch'):
        lifecycle.marker_for(handoff, calendar)


@pytest.fixture
def native_rules():
    path = Path(os.environ.get('JBR_LIFECYCLE_NATIVE_RULES',
        str(Path(__file__).parent / 'fixtures' / 'ownership_native_rules.py')))
    assert path.is_file(), 'Pinned native rules fixture missing'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        '515401c0b8d6652383cd87abf2967004905ca907858fc205213f8ab9d6e29cca')
    spec = importlib.util.spec_from_file_location('pinned_lifecycle_rules', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native(native_rules, monkeypatch):
    path = Path(os.environ.get('JBR_LIFECYCLE_NATIVE_ENGINE',
        str(Path(__file__).parent / 'fixtures' / 'ownership_native_engine.py')))
    assert path.is_file(), 'Pinned native Engine fixture missing'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        '8de13a699757c7cbd67f3b5e85245001ff302be98d5c9e9ba565a55d645039f0')
    spec = importlib.util.spec_from_file_location('pinned_lifecycle_engine', path)
    module = importlib.util.module_from_spec(spec)
    import scripts
    with monkeypatch.context() as context:
        context.setattr(scripts, 'strategy_execution', native_rules, raising=False)
        context.setitem(sys.modules, 'scripts.strategy_execution', native_rules)
        spec.loader.exec_module(module)
    assert module.rules is native_rules
    return module


def enrolled(native, *, marker=True):
    from tests.test_strategy_execution_engine import setup, enter
    _, store, broker, handoff = setup()
    base = lifecycle.monitor_engine(native.Engine)
    engine = base(store, broker, guard=lambda *args: True, now=lambda: broker.now)
    enter(engine, broker, handoff)
    signal = next(iter(store.rows.values()))
    if marker:
        signal['payload']['entry_lifecycle'] = lifecycle.marker_for(
            handoff, dict(date='2026-10-26', open='09:30', close='16:00'))
    signal['payload']['history_sentinel'] = [{'old': 'preserve'}]
    return engine, store, broker, handoff


def test_0435_and_regular_close_cancel_readback(native):
    engine, store, broker, handoff = enrolled(native)
    original = deepcopy(next(iter(store.rows.values()))['payload'])
    broker.now = datetime(2026, 10, 26, 8, 35, tzinfo=timezone.utc)
    engine.tick(now=broker.now)
    assert not broker.cancels
    broker.now = datetime(2026, 10, 26, 20, tzinfo=timezone.utc)
    engine.tick(now=broker.now)
    assert len(broker.cancels) == 1
    assert next(iter(store.rows.values()))['state'] == 'ENTRY_PENDING'
    engine.tick(now=broker.now)
    signal = next(iter(store.rows.values()))
    assert signal['state'] == 'EXPIRED'
    assert signal['payload']['handoff'] == handoff == original['handoff']
    assert signal['payload']['history_sentinel'] == original['history_sentinel']


def test_partial_fills_stop_actual_quantity_and_native_risk_cancel(native):
    engine, store, broker, _ = enrolled(native)
    broker.now = datetime(2026, 10, 26, 8, 35, tzinfo=timezone.utc)
    broker.fill('entry', 2, status='partially_filled')
    engine.tick(now=broker.now)
    assert broker.posts[-1]['side'] == 'sell'
    assert broker.posts[-1]['type'] == 'stop'
    assert broker.posts[-1]['qty'] == 2
    assert not broker.cancels
    broker.fill('entry', 3, status='partially_filled')
    engine.tick(now=broker.now)
    assert broker.patches[-1][1]['qty'] == 3
    broker.bid = 1
    engine.tick(now=broker.now)
    assert broker.cancels
    assert any(e['code'] == 'ENTRY_CANCELLATION_BEFORE_EXIT' for e in engine.events)


def test_full_fill_and_invalid_marker_cannot_manage(native):
    engine, store, broker, _ = enrolled(native)
    qty = broker.posts[0]['qty']
    broker.fill('entry', qty)
    engine.tick(now=broker.now)
    assert broker.posts[-1]['qty'] == qty
    signal = next(iter(store.rows.values()))
    signal['payload']['entry_lifecycle']['owner'] = 'wrong'
    posts = deepcopy(broker.posts)
    with pytest.raises(ValueError, match='marker_changed'):
        engine.tick(now=broker.now)
    assert broker.posts == posts


def test_prepared_expiry_and_native_unchanged(native):
    from tests.test_strategy_execution_engine import setup, reference
    _, store, broker, handoff = setup()
    signal = store.enroll(handoff, handoff['proposals'][0], reference())
    signal['payload']['entry_lifecycle'] = lifecycle.marker_for(
        handoff, dict(date='2026-10-26', open='09:30', close='16:00'))
    store.save(signal)
    sync, tick = native.Engine.sync, native.Engine.tick
    globals_before = dict(sync.__globals__)
    derived = lifecycle.monitor_engine(native.Engine)
    engine = derived(store, broker, guard=lambda *args: True)
    broker.now = datetime(2026, 10, 26, 8, 35, tzinfo=timezone.utc)
    engine.tick(now=broker.now)
    assert next(iter(store.rows.values()))['state'] == 'PREPARED'
    broker.now = datetime(2026, 10, 26, 20, tzinfo=timezone.utc)
    engine.tick(now=broker.now)
    assert next(iter(store.rows.values()))['state'] == 'EXPIRED'
    assert native.Engine.sync is sync and native.Engine.tick is tick
    assert sync.__globals__ == globals_before


def test_legacy_native_still_cancels_at_0435(native):
    engine, store, broker, _ = enrolled(native, marker=False)
    broker.now = datetime(2026, 10, 26, 8, 35, tzinfo=timezone.utc)
    engine.tick(now=broker.now)
    assert len(broker.cancels) == 1


def test_early_close_pending_cancel_then_actual_fill_readback(native):
    engine, store, broker, handoff = enrolled(native)
    signal = next(iter(store.rows.values()))
    signal['payload']['entry_lifecycle'] = lifecycle.marker_for(
        handoff, dict(date='2026-10-26', open='09:30', close='13:00'))
    broker.now = datetime(2026, 10, 26, 16, 59, 59, tzinfo=timezone.utc)
    engine.tick(now=broker.now)
    assert not broker.cancels
    broker.now = datetime(2026, 10, 26, 17, tzinfo=timezone.utc)
    broker.cancel_pending = True
    engine.tick(now=broker.now)
    assert len(broker.cancels) == 1
    assert any(e['code'] == 'ENTRY_CANCELLATION_UNCONFIRMED' for e in engine.events)
    # Fill during cancellation: only subsequent broker readback can protect it.
    broker.fill('entry', 2, status='canceled')
    engine.tick(now=broker.now)
    assert broker.posts[-1]['type'] == 'stop'
    assert broker.posts[-1]['qty'] == 2
    assert next(iter(store.rows.values()))['payload']['entry_filled_qty'] == 2


def test_required_marker_missing_fails_before_orders(native):
    _, store, broker, _ = enrolled(native, marker=False)
    engine = lifecycle.monitor_engine(native.Engine, require_marker=True)(
        store, broker, guard=lambda *args: True)
    before = deepcopy(broker.posts)
    with pytest.raises(ValueError, match='marker_missing'):
        engine.tick(now=broker.now)
    assert broker.posts == before and not broker.cancels


def test_unknown_native_source_rejected():
    class ChangedEngine:
        def sync(self, signal, *, now):
            pass
    with pytest.raises(ValueError, match='source_changed'):
        lifecycle.monitor_engine(ChangedEngine)


def pipeline_fixture(native):
    """Synthetic valid publication, using real deployed ticket validation."""
    from tests.test_strategy_execution_engine import setup
    _, store, broker, handoff = setup()
    rules = native.rules
    handoff['execution_policy']['model'] = rules.PIPELINE_LIMIT_MODEL
    handoff['pipeline_limit_placement_v2'] = True
    handoff['unified_selection'] = dict(run_id='synthetic-selection-run')
    published = '2026-10-26T07:59:00+00:00'
    proposal = handoff['proposals'][0]
    ticket = dict(schema='jbravo.pipeline-paper-limit-order.v2',
        primary_run_id=handoff['primary_run_id'],
        selection_run_id=handoff['unified_selection']['run_id'],
        **{key: proposal[key] for key in ('signal_id', 'symbol', 'signal_session', 'next_session')},
        limit_price=101.23, currency='USD', price_basis='SYNTHETIC_VERIFIED_PIPELINE_LIMIT',
        price_as_of=proposal['signal_session'] + 'T19:59:00-04:00',
        valid_from=published, valid_until=handoff['execution_policy']['cancel_at'])
    handoff.update(pipeline_limit_order=ticket, pipeline_limit_published_at=published)
    handoff['handoff_id'] = rules.digest({k: v for k, v in handoff.items() if k != 'handoff_id'})
    assert rules.validate_pipeline_limit_order(handoff, proposal, ticket, published_at=published) == ticket
    return store, broker, handoff


@pytest.mark.parametrize('fill_mode', ['unfilled', 'partial', 'full'])
def test_native_pipeline_entry_only_to_monitor_ownership(native, fill_mode, source_variant="retained"):
    from scripts.strategy_entry_only import EntryOnlyStore, EntryOnlyBroker, entry_engine
    store, broker, handoff = pipeline_fixture(native)
    proposal = handoff['proposals'][0]
    before = deepcopy(handoff)
    calendar = dict(date=proposal['next_session'], open='09:30', close='16:00')
    scoped = EntryOnlyStore(store, calendar, lifecycle.marker_for)
    real_submit = broker.submit_order
    posted_with_marker = []

    def submit(request):
        # Prove durable ownership and claim both precede the broker write.
        signal = store.rows[proposal['signal_id']]
        assert signal['state'] == 'ENTRY_PENDING'
        marker = signal['payload']['entry_lifecycle']
        assert marker == lifecycle.marker_for(before, calendar)
        assert store.orders(proposal['signal_id'])[0]['state'] == 'CLAIMED'
        assert request['side'] == 'buy' and request['type'] == 'limit'
        posted_with_marker.append(deepcopy(marker))
        return real_submit(request)

    broker.submit_order = submit

    def guard(kind, signal):
        assert kind == 'entry'
        assert signal['payload']['handoff'] == before
        return True

    entry = entry_engine(native.Engine)(scoped, EntryOnlyBroker(broker), guard=guard,
        pipeline_orders=True, now=lambda: broker.now)
    entry.enter(handoff, {proposal['symbol']: handoff['pipeline_limit_order']},
        now=broker.now, clock=broker.clock(), account=broker.account(),
        positions=broker.positions(), open_orders=broker.open_orders(),
        selected_signal_ids=[proposal['signal_id']])
    assert len(posted_with_marker) == len(broker.posts) == 1
    assert broker.posts[0]['limit_price'] == 101.23
    assert not broker.cancels and not broker.patches
    assert next(iter(store.rows.values()))['payload']['position'] is None
    broker.submit_order = real_submit
    monitor = lifecycle.monitor_engine(native.Engine, require_marker=True, source_variant=source_variant)(
        store, broker, guard=lambda *args: True, pipeline_orders=True, now=lambda: broker.now)
    broker.now = datetime(2026, 10, 26, 8, 35, tzinfo=timezone.utc)
    qty = 0 if fill_mode == 'unfilled' else 2 if fill_mode == 'partial' else broker.posts[0]['qty']
    if qty:
        broker.fill('entry', qty, status='partially_filled' if fill_mode == 'partial' else 'filled')
    monitor.tick(now=broker.now, observe_prices=False)
    assert not broker.cancels
    if qty:
        stop = broker.posts[-1]
        assert stop['side'] == 'sell' and stop['type'] == 'stop' and stop['qty'] == qty
        assert stop['time_in_force'] == 'gtc' and stop['extended_hours'] is False
        assert next(iter(store.rows.values()))['payload']['entry_filled_qty'] == qty
    else:
        assert len(broker.posts) == 1
    broker.now = datetime(2026, 10, 26, 20, tzinfo=timezone.utc)
    monitor.tick(now=broker.now, observe_prices=False)
    assert len(broker.cancels) == (0 if fill_mode == 'full' else 1)
    monitor.tick(now=broker.now, observe_prices=False)
    payload = next(iter(store.rows.values()))['payload']
    assert payload['handoff'] == handoff == before
    assert payload['handoff']['pipeline_limit_order']['valid_until'] == '2026-10-26T08:35:00+00:00'
    assert payload['entry_lifecycle'] == posted_with_marker[0]
    assert payload['entry_lifecycle']['cancel_at'] == '2026-10-26T20:00:00+00:00'


def test_native_pipeline_rejects_extended_submission_ticket_before_post(native):
    from scripts.strategy_entry_only import EntryOnlyStore, EntryOnlyBroker, entry_engine
    store, broker, handoff = pipeline_fixture(native)
    proposal = handoff['proposals'][0]
    handoff['pipeline_limit_order']['valid_until'] = '2026-10-26T20:00:00+00:00'
    handoff['handoff_id'] = native.rules.digest({k: v for k, v in handoff.items() if k != 'handoff_id'})
    scoped = EntryOnlyStore(store, dict(date=proposal['next_session'], open='09:30', close='16:00'),
                            lifecycle.marker_for)
    entry = entry_engine(native.Engine)(scoped, EntryOnlyBroker(broker), guard=lambda *args: True,
        pipeline_orders=True, now=lambda: broker.now)
    with pytest.raises(ValueError, match='pipeline_limit_session_changed'):
        entry.enter(handoff, {proposal['symbol']: handoff['pipeline_limit_order']},
            now=broker.now, clock=broker.clock(), account=broker.account(),
            positions=broker.positions(), open_orders=broker.open_orders(),
            selected_signal_ids=[proposal['signal_id']])
    assert not broker.posts and not store.rows
