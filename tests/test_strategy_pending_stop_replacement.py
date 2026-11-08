"""Offline native-source reproduction and narrow pending-replacement regression."""
from copy import deepcopy
import hashlib
import inspect
import socket
import sys
import textwrap
from types import SimpleNamespace

import pytest

from scripts import strategy_order_lifecycle as lifecycle
from scripts import strategy_order_ownership_install as install
from scripts import strategy_pending_stop_replacement as pending
from tests.fixtures.ownership_quote_safe_monitor import quote_safe_engine
from tests.fixtures import ownership_native_protection as protection
from tests.test_strategy_execution_engine import setup, enter, bars, reference
from tests.test_strategy_order_lifecycle import native, native_rules

pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('No network in pending replacement tests')
    monkeypatch.setattr(socket.socket, 'connect', deny)


def composed(native, *, adapted=True):
    monitor = SimpleNamespace(quote_safe_engine=quote_safe_engine)
    install.install_monitor(monitor, native.Engine, lifecycle,
                            pending_replacement=pending if adapted else None)
    return monitor.quote_safe_engine(native.Engine)


def pair(native, *, adapted=True, example_pair=False):
    _, store, broker, handoff = setup()
    Engine = composed(native, adapted=adapted)
    engine = Engine(store, broker, guard=lambda *args: True, now=lambda: broker.now)
    if example_pair:
        # Synthetic round-number example; no live account/order data.
        handoff['proposals'][0]['symbol'] = 'TEST'
        handoff['handoff_id'] = native.rules.digest({k: v for k, v in handoff.items() if k != 'handoff_id'})
        broker.account = lambda: dict(cash='603000', buying_power='603000')
        ref = reference()
        ref['symbol'] = 'TEST'
        ref['reference_id'] = native.rules.digest({k: v for k, v in ref.items() if k != 'reference_id'})
        engine.enter(handoff, {'TEST': ref}, now=broker.now, clock=broker.clock(),
                     account=broker.account(), positions={}, open_orders=[])
    else:
        enter(engine, broker, handoff)
    qty = broker.posts[0]['qty']
    broker.fill('entry', qty, price=21.0 if example_pair else 100.0)
    signal = store.signals()[0]
    records = engine.sync(signal, now=broker.now)
    if example_pair:
        signal['payload']['position'].update(trailing_stop=20.0, atr_stop=20.0)
        store.save(signal)
    engine.manage(signal, records, now=broker.now, quote=None, regular_open=False)
    signal['payload']['position']['trailing_stop'] = 25.0 if example_pair else 105.0
    store.save(signal)
    engine.manage(signal, store.orders(signal['signal_id']), now=broker.now,
                  quote=None, regular_open=False)
    stops = [r for r in store.orders(signal['signal_id']) if r['role'] == 'stop']
    old, new = [broker.orders[r['client_order_id']] for r in stops]
    old.update(status='pending_replace', replaced_by=new['id'])
    new.update(replaces=old['id'], replaced_by=None)
    engine.sync(signal, now=broker.now)
    engine.pipeline_orders = True
    engine.events.clear()
    return engine, store, broker, signal


def snapshot(store, broker):
    return deepcopy((store.rows, store.outbox, broker.posts, broker.patches, broker.cancels))


def manage(engine, store, broker, signal, *, quote=None):
    return engine.manage(signal, store.orders(signal['signal_id']), now=broker.now,
                         quote=quote, regular_open=False)


def completed(engine, signal):
    return engine.completed(signal, bars(), session='2026-10-26', next_session='2026-10-27',
                            now='2026-10-27T00:15:00+00:00')


def test_quote_factory_fixture_and_native_globals_preserved(native):
    source = textwrap.dedent(inspect.getsource(quote_safe_engine)).strip()
    assert hashlib.sha256(source.encode()).hexdigest() == (
        '5c523dd598350cac67c392be5b6d65f6f462a3bfbc4cd5c4ac29ba564ebd81bb')
    methods = {k: getattr(native.Engine, k) for k in ('manage', 'completed', 'sync', 'tick')}
    before = dict(native.Engine.manage.__globals__)
    Engine = composed(native)
    assert Engine.__mro__[1].__name__ == 'QuoteSafeEngine'
    assert Engine.__mro__[2].__name__ == 'MonitorOwnedEntryEngine'
    assert {k: getattr(native.Engine, k) for k in methods} == methods
    assert native.Engine.manage.__globals__ == before
    assert Engine.sync is Engine.__mro__[2].sync


def test_original_native_and_quote_safe_abort_exact_pair(native):
    engine, store, broker, signal = pair(native, adapted=False)
    before = snapshot(store, broker)
    for callback in (native.Engine.manage, type(engine).manage):
        with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
            callback(engine, signal, store.orders(signal['signal_id']), now=broker.now,
                     quote=None, regular_open=False)
    assert snapshot(store, broker) == before


@pytest.mark.parametrize('quote', [None, {'bid': 110}, {'bid': 1}])
@pytest.mark.parametrize('reason', [[], ['PARTIAL_5PCT'], ['RSI_EXIT']])
def test_pending_pair_no_mutation_or_intent_rewrite(native, quote, reason):
    engine, store, broker, signal = pair(native)
    signal['payload'].update(pending_exit_reasons=reason, pending_exit_session='2026-10-26')
    store.save(signal)
    before = snapshot(store, broker)
    manage(engine, store, broker, signal, quote=quote)
    completed(engine, signal)
    assert snapshot(store, broker) == before
    assert len(engine.events) == 2
    assert all(e['code'] == 'STOP_REPLACEMENT_PENDING' and
               e['protection_confirmed'] is False for e in engine.events)
    assert signal['payload']['last_completed_session'] is None
    assert engine.events[-1]['completed_session_deferred'] == '2026-10-26'


def test_example_pair_shape_pending_and_entry_hook_unchanged(native):
    engine, store, broker, signal = pair(native, example_pair=True)
    stops = [r['observation'] for r in store.orders(signal['signal_id']) if r['role'] == 'stop']
    assert [o['qty'] for o in stops] == [300, 300]
    assert [o['stop_price'] for o in stops] == [20.0, 25.0]
    before = snapshot(store, broker)
    manage(engine, store, broker, signal)
    assert snapshot(store, broker) == before
    assert engine.events[-1]['code'] == 'STOP_REPLACEMENT_PENDING'
    assert type(engine).enter is native.Engine.enter


def change_record(record, path, value):
    current = record
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


@pytest.mark.parametrize('which,path,value', [
    ('old', ('observation', 'replaced_by'), 'wrong'),
    ('new', ('observation', 'replaces'), 'wrong'),
    ('new', ('observation', 'replaced_by'), 'third'),
    ('new', ('request', 'order_id'), 'wrong'),
    ('new', ('operation',), 'POST'),
    ('old', ('operation',), 'DELETE'),
    ('old', ('state',), 'UNKNOWN'),
    ('new', ('state',), 'CLAIMED'),
    ('entry', ('state',), 'UNKNOWN'),
    ('new', ('client_order_id',), 'wrong'),
    ('new', ('request', 'order', 'client_order_id'), 'wrong'),
    ('new', ('observation', 'client_order_id'), 'wrong'),
    ('new', ('sequence',), 7),
    ('new', ('signal_id',), 'wrong'),
    ('new', ('account_id',), 'wrong'),
    ('new', ('observation', 'symbol'), 'OTHER'),
    ('new', ('observation', 'side'), 'buy'),
    ('new', ('observation', 'type'), 'limit'),
    ('new', ('observation', 'time_in_force'), 'day'),
    ('new', ('observation', 'extended_hours'), True),
    ('old', ('observation', 'status'), 'new'),
    ('new', ('observation', 'status'), 'pending_new'),
    ('new', ('observation', 'qty'), '48'),
    ('new', ('observation', 'qty'), '49.5'),
    ('new', ('observation', 'qty'), True),
    ('old', ('observation', 'filled_qty'), '1'),
    ('new', ('observation', 'filled_qty'), '-1'),
    ('new', ('observation', 'stop_price'), 'NaN'),
    ('entry', ('observation', 'status'), 'partially_filled'),
    ('entry', ('observation', 'filled_qty'), '48'),
])
def test_malformed_or_unknown_pair_uses_original_fail_closed(native, which, path, value):
    engine, store, broker, signal = pair(native)
    records = list(store.outbox.values())
    record = records[0] if which == 'entry' else records[1 if which == 'old' else 2]
    change_record(record, path, value)
    if path == ('signal_id',):
        # Simulate a port returning cross-scope records, rather than allowing
        # MemoryStore's normal filter to omit the malformed record entirely.
        store.orders = lambda sid: deepcopy(records)
    before = snapshot(store, broker)
    with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
        manage(engine, store, broker, signal)
    with pytest.raises(ValueError, match='multiple_active_stops|order_unresolved'):
        completed(engine, signal)
    assert snapshot(store, broker) == before
    assert not any(e['code'] == 'STOP_REPLACEMENT_PENDING' for e in engine.events)


@pytest.mark.parametrize('kind', ['loosen', 'qty', 'fills', 'duplicate_id', 'third_stop',
                                   'manual', 'recognition', 'position', 'entry_total'])
def test_consistent_but_unsafe_shapes_are_not_recognized(native, kind):
    engine, store, broker, signal = pair(native)
    old, new = [r for r in store.outbox.values() if r['role'] == 'stop']
    if kind == 'loosen':
        new['request']['order']['stop_price'] = new['observation']['stop_price'] = 98
    elif kind == 'qty':
        new['request']['order']['qty'] = new['observation']['qty'] = 48
    elif kind == 'fills':
        new['observation'].update(filled_qty='1', filled_avg_price='105')
    elif kind == 'duplicate_id':
        new['observation']['id'] = old['observation']['id']
        old['observation']['replaced_by'] = old['observation']['id']
    elif kind == 'third_stop':
        store.outbox['extra'] = deepcopy(new)
    elif kind == 'manual':
        signal['payload']['handoff']['manual_paper_test'] = True
    elif kind == 'recognition':
        signal['payload']['execution_owner'] = 'recognition-only'
    elif kind == 'position':
        broker.held[signal['payload']['position']['symbol']]['qty'] = '48'
    elif kind == 'entry_total':
        signal['payload']['entry_filled_qty'] = 48
    before = snapshot(store, broker)
    with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
        manage(engine, store, broker, signal)
    with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
        completed(engine, signal)
    assert snapshot(store, broker) == before


def test_competing_exit_preserves_original_pending_semantics_but_no_completed_save(native):
    engine, store, broker, signal = pair(native)
    record = deepcopy(next(r for r in store.outbox.values() if r['role'] == 'stop'))
    record.update(role='partial', client_order_id='extra')
    store.outbox['extra'] = record
    before = snapshot(store, broker)
    manage(engine, store, broker, signal)
    assert engine.events[-1]['code'] == 'EXIT_ORDER_PENDING_CONFIRMATION'
    with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
        completed(engine, signal)
    assert snapshot(store, broker) == before


def settle(store, broker):
    old = next(r for r in store.outbox.values() if r['role'] == 'stop')
    old['observation']['status'] = 'replaced'
    broker.orders[old['client_order_id']]['status'] = 'replaced'


def test_completed_fresh_read_before_manage_then_terminal_catchup(native):
    engine, store, broker, signal = pair(native)
    before = snapshot(store, broker)
    completed(engine, signal)
    assert snapshot(store, broker) == before
    settle(store, broker)
    completed(engine, signal)
    assert signal['payload']['last_completed_session'] == '2026-10-26'
    assert signal['payload']['position']['atr_stop'] == 104.0
    revision = signal['revision']
    completed(engine, signal)
    assert signal['revision'] == revision


def test_terminal_restores_original_manage_and_completed_exactly(native):
    adapted = pair(native)
    original = pair(native, adapted=False)
    for engine, store, broker, signal in (adapted, original):
        settle(store, broker)
        manage(engine, store, broker, signal)
        completed(engine, signal)
    assert adapted[0].events == original[0].events
    assert snapshot(adapted[1], adapted[2]) == snapshot(original[1], original[2])
    assert adapted[3] == original[3]


def test_completed_rechecks_after_manage_if_orders_become_unknown(native):
    engine, store, broker, signal = pair(native)
    manage(engine, store, broker, signal)
    list(store.outbox.values())[-1]['state'] = 'UNKNOWN'
    before = snapshot(store, broker)
    with pytest.raises(ValueError, match='order_unresolved'):
        completed(engine, signal)
    assert snapshot(store, broker) == before


def test_zero_or_one_stop_completed_keeps_native_behavior_and_unknown_blocks(native):
    for count in (0, 1):
        engine, store, broker, signal = pair(native)
        stop_keys = [k for k, r in store.outbox.items() if r['role'] == 'stop']
        for key in stop_keys[count:]:
            del store.outbox[key]
        completed(engine, signal)
        assert signal['payload']['last_completed_session'] == '2026-10-26'
    list(store.outbox.values())[0]['state'] = 'UNKNOWN'
    with pytest.raises(ValueError, match='order_unresolved'):
        completed(engine, signal)


def test_native_sync_stays_factual_and_other_symbol_reaches_protection(native):
    engine, store, broker, signal = pair(native)
    other = deepcopy(signal)
    other.update(signal_id='f' * 64, revision=0)
    other['payload']['proposal'].update(symbol='OTHER', signal_id=other['signal_id'])
    other['payload']['position']['symbol'] = 'OTHER'
    store.rows[other['signal_id']] = deepcopy(other)
    entry = deepcopy(next(r for r in store.outbox.values() if r['role'] == 'entry'))
    key = native.rules.client_order_id(store.account_id, other['signal_id'])
    entry.update(signal_id=other['signal_id'], client_order_id=key)
    entry['request']['order'].update(symbol='OTHER', client_order_id=key)
    entry['observation'].update(symbol='OTHER', client_order_id=key, id='other-entry')
    store.outbox[key] = deepcopy(entry)
    broker.orders[key] = deepcopy(entry['observation'])
    broker.held['OTHER'] = dict(symbol='OTHER', qty=str(other['payload']['position']['qty']), side='long')
    old_payload = deepcopy(signal['payload'])
    initial_posts = len(broker.posts)
    engine.tick(now=broker.now, observe_prices=False)
    assert any(e['code'] == 'STOP_REPLACEMENT_PENDING' for e in engine.events)
    assert len(broker.posts) == initial_posts + 1 and broker.posts[-1]['symbol'] == 'OTHER'
    assert store.signals()[0]['payload'] == old_payload
    assert store.rows[signal['signal_id']]['revision'] > signal['revision']
    signal = store.signals()[0]
    completed(engine, signal)  # Actual release ordering: tick, completed, tick.
    completed(engine, store.signals()[1])
    assert store.signals()[1]['payload']['last_completed_session'] == '2026-10-26'
    engine.tick(now=broker.now, observe_prices=False)
    assert store.signals()[0]['payload']['last_completed_session'] is None
    assert len(broker.posts) == initial_posts + 1


def test_source_drift_rejected_before_any_engine_is_constructed(native):
    class Changed(native.Engine):
        def manage(self, *args, **kwargs):
            return None
    with pytest.raises(ValueError, match='manage_source_changed'):
        pending.monitor_engine(Changed, NativeEngine=Changed)


def test_optional_install_checks_source_eagerly_and_is_atomic(native, monkeypatch):
    monitor = SimpleNamespace(quote_safe_engine=quote_safe_engine)
    monkeypatch.setitem(pending._METHOD_HASHES, 'manage', '0' * 64)
    with pytest.raises(ValueError, match='manage_source_changed'):
        install.install_monitor(monitor, native.Engine, lifecycle, pending_replacement=pending)
    assert monitor.quote_safe_engine is quote_safe_engine
    assert not getattr(monitor, '_entry_monitor_ownership_installed', False)


def test_optional_install_caches_validated_class_and_retains_guards(native):
    monitor = SimpleNamespace(quote_safe_engine=quote_safe_engine)
    install.install_monitor(monitor, native.Engine, lifecycle, pending_replacement=pending)
    assert monitor.quote_safe_engine(native.Engine) is monitor.quote_safe_engine(native.Engine)
    with pytest.raises(ValueError, match='duplicate_install'):
        install.install_monitor(monitor, native.Engine, lifecycle, pending_replacement=pending)
    with pytest.raises(ValueError, match='identity_changed'):
        monitor.quote_safe_engine(object)


@pytest.mark.parametrize('quote', [None, {'bid': 110}, {'bid': 1}])
def test_pattern_target_never_creates_intent_during_pending_pair(native, quote):
    engine, store, broker, signal = pair(native)
    signal['payload']['proposal'].update(strategy_id='bull_flag_breakout_v1',
        pattern_exit={'target': 108, 'invalidation': 95})
    store.save(signal)
    before = snapshot(store, broker)
    manage(engine, store, broker, signal, quote=quote)
    completed(engine, signal)
    assert snapshot(store, broker) == before
    assert all(e['code'] == 'STOP_REPLACEMENT_PENDING' for e in engine.events)


def add_completed_partial(native, store, broker, signal):
    symbol = signal['payload']['position']['symbol']
    key = native.rules.client_order_id(store.account_id, signal['signal_id'], role='partial')
    request = dict(symbol=symbol, qty=5, side='sell', type='limit', limit_price=110,
                   time_in_force='day', extended_hours=True, client_order_id=key)
    order = dict(request, id='partial-id', status='filled', filled_qty='5', filled_avg_price='110')
    store.outbox[key] = dict(client_order_id=key, signal_id=signal['signal_id'], role='partial',
        sequence=0, operation='POST', request=dict(order_id=None, order=request),
        state='OBSERVED', observation=order)
    broker.orders[key] = deepcopy(order)
    broker.held[symbol]['qty'] = '44'
    for record in store.outbox.values():
        if record['role'] == 'stop':
            record['request']['order']['qty'] = record['observation']['qty'] = 44
            broker.orders[record['client_order_id']]['qty'] = 44
    signal['payload']['position']['qty'] = 44
    signal['payload'].update(partial_target_qty=5, pending_exit_reasons=['PARTIAL_5PCT'],
                             pending_exit_session='2026-10-26')
    store.save(signal)


def test_unreconciled_partial_completion_rejected_then_native_sync_clears_intent(native):
    engine, store, broker, signal = pair(native)
    add_completed_partial(native, store, broker, signal)
    before = snapshot(store, broker)
    with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
        manage(engine, store, broker, signal)
    with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
        completed(engine, signal)
    assert snapshot(store, broker) == before
    engine.sync(signal, now=broker.now)
    assert signal['payload']['pending_exit_reasons'] == []
    assert signal['payload']['position']['partial_taken'] is True
    before = snapshot(store, broker)
    manage(engine, store, broker, signal)
    completed(engine, signal)
    assert snapshot(store, broker) == before
    assert engine.events[-1]['code'] == 'STOP_REPLACEMENT_PENDING'


def test_incomplete_partial_target_can_defer_without_erasing_intent(native):
    engine, store, broker, signal = pair(native)
    add_completed_partial(native, store, broker, signal)
    signal['payload']['partial_target_qty'] = 10
    store.save(signal)
    before = snapshot(store, broker)
    manage(engine, store, broker, signal)
    completed(engine, signal)
    assert snapshot(store, broker) == before
    assert engine.events[-1]['code'] == 'STOP_REPLACEMENT_PENDING'


@pytest.mark.parametrize('bad_ancestor', [False, True])
def test_previous_patch_requires_terminal_owned_ancestor(native, bad_ancestor):
    engine, store, broker, signal = pair(native)
    old, new = [r for r in store.outbox.values() if r['role'] == 'stop']
    ancestor = deepcopy(old)
    # Rotate the two current records to sequences 1/2, leaving sequence0 as
    # the terminal historical ancestor. All client IDs remain native-derived.
    for record in (old, new):
        previous = record['client_order_id']
        record['sequence'] += 1
        key = native.rules.client_order_id(store.account_id, signal['signal_id'],
                                          role='stop', sequence=record['sequence'])
        record['client_order_id'] = key
        record['request']['order']['client_order_id'] = key
        record['observation']['client_order_id'] = key
        record['operation'] = 'PATCH'
        broker.orders.pop(previous, None)
    ancestor['observation'].update(id='ancestor', status='replaced', replaced_by=old['observation']['id'])
    ancestor['observation'].pop('replaces', None)
    old['request']['order_id'] = 'wrong' if bad_ancestor else 'ancestor'
    old['observation']['replaces'] = 'ancestor'
    entries = [r for r in store.outbox.values() if r['role'] == 'entry']
    store.outbox = {r['client_order_id']: r for r in entries + [ancestor, old, new]}
    for record in (ancestor, old, new):
        broker.orders[record['client_order_id']] = deepcopy(record['observation'])
    before = snapshot(store, broker)
    if bad_ancestor:
        with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
            manage(engine, store, broker, signal)
        with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
            completed(engine, signal)
    else:
        manage(engine, store, broker, signal)
        completed(engine, signal)
        assert engine.events[-1]['code'] == 'STOP_REPLACEMENT_PENDING'
    assert snapshot(store, broker) == before


def test_exact_native_protection_report_keeps_pair_unconfirmed(native, monkeypatch):
    source = textwrap.dedent(inspect.getsource(protection.entry_protection_state)).strip()
    assert hashlib.sha256(source.encode()).hexdigest() == (
        'e36da7399f91b90ea9ae5cf725272eca5e237a296f3f6b60a575486cecbded36')
    monkeypatch.setattr(protection, 'rules', native.rules)
    monkeypatch.setitem(sys.modules, 'scripts.strategy_execution_engine', native)
    engine, store, broker, signal = pair(native)
    before = deepcopy(store.orders(signal['signal_id']))
    manage(engine, store, broker, signal)
    completed(engine, signal)
    result = protection.entry_protection_state(store)
    assert store.orders(signal['signal_id']) == before
    assert result['confirmed'] is False
    assert result['positions'][0]['regular_hours_stop_confirmed'] is False
    assert result['positions'][0]['stop_order_id'] is None
    assert result['unresolved'][0]['reason'] == 'FULL_QUANTITY_STOP_NOT_CONFIRMED'
    settle(store, broker)
    assert protection.entry_protection_state(store)['confirmed'] is True


def test_missing_request_and_links_are_not_benign_pending(native):
    for field in ('request', 'replaces'):
        engine, store, broker, signal = pair(native)
        new = list(store.outbox.values())[-1]
        del (new if field == 'request' else new['observation'])[field]
        before = snapshot(store, broker)
        with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
            manage(engine, store, broker, signal)
        with pytest.raises(ValueError, match='strategy_multiple_active_stops'):
            completed(engine, signal)
        assert snapshot(store, broker) == before


def test_pending_wrappers_cannot_call_mutation_ports(native, monkeypatch):
    engine, store, broker, signal = pair(native)
    def deny(*args, **kwargs):
        raise AssertionError('Pending wrapper called a mutation port')
    for name in ('save', 'claim', 'observed', 'rejected', 'uncertain', 'transaction'):
        monkeypatch.setattr(store, name, deny)
    for name in ('submit_order', 'replace_order', 'cancel_order'):
        monkeypatch.setattr(broker, name, deny)
    manage(engine, store, broker, signal, quote={'bid': 1})
    completed(engine, signal)
    assert len(engine.events) == 2


def test_terminal_resumption_retains_existing_no_quote_stop_tightening(native):
    adapted = pair(native)
    original = pair(native, adapted=False)
    for engine, store, broker, signal in (adapted, original):
        settle(store, broker)
        signal['payload']['position']['trailing_stop'] = 106
        signal['payload'].update(pending_exit_reasons=['RSI_EXIT'], pending_exit_session='2026-10-26')
        store.save(signal)
        prior = len(broker.patches)
        manage(engine, store, broker, signal)
        assert len(broker.patches) == prior + 1
        assert broker.patches[-1][1]['stop_price'] == 106
        assert not broker.cancels
        assert signal['payload']['pending_exit_reasons'] == ['RSI_EXIT']
    assert snapshot(adapted[1], adapted[2]) == snapshot(original[1], original[2])
    assert adapted[0].events == original[0].events
