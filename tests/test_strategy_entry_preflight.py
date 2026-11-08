"""Offline preflight proof: actual native identity checks, no mutation ports."""
from copy import deepcopy
from types import SimpleNamespace
import socket

import pytest

from scripts.strategy_entry_preflight import broker_entry_preflight
from scripts.strategy_execution_engine import Engine

pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', deny)


class Store:
    account_id = 'paper'

    def __init__(self):
        self.rows = []; self.records = {}

    def signals(self, *, active=True):
        assert active is False
        return deepcopy(self.rows)

    def orders(self, sid):
        return deepcopy(self.records[sid])


class Broker:
    paper = True; account_id = 'paper'

    def __init__(self):
        self.orders = {}; self.held = {}

    def account(self):
        return {'id': 'paper'}

    def positions(self):
        return deepcopy(self.held)

    def open_orders(self):
        return deepcopy([o for o in self.orders.values() if o['status'] not in
                         {'filled', 'canceled', 'expired', 'rejected', 'replaced'}])

    def find_by_client_order_id(self, key):
        return deepcopy(self.orders.get(key))

    def get_order(self, oid):
        return deepcopy(next(o for o in self.orders.values() if o['id'] == oid))


def fixture():
    store = Store(); broker = Broker()
    runtime = SimpleNamespace(source_guard=lambda *a, **k: True,
        lifecycle_handoff=lambda s: s['payload']['handoff'],
        validate_entry_recognition=lambda *a, **k: {'signal_id': a[0]['signal_id']})
    handoff = {'source_bindings': {}, 'session': {'next_session': '2026-11-09'},
               'execution_policy': {'model': 'pipeline'}}
    return store, broker, runtime, handoff


def add(store, broker, handoff, sid='one', symbol='ARDT', filled=10, stop=True):
    position = {'symbol': symbol, 'qty': filled, 'trailing_stop': 17.49, 'atr_stop': 16}
    signal = {'signal_id': sid, 'state': 'OPEN' if filled else 'ENTRY_PENDING',
              'payload': {'account_id': 'paper', 'proposal': {'symbol': symbol},
                          'position': position if filled else None, 'handoff': deepcopy(handoff)}}
    store.rows.append(signal); store.records[sid] = []
    def order(role, qty, status, fill=0):
        key = sid + role
        request = dict(symbol=symbol, qty=qty, side='buy' if role == 'entry' else 'sell',
                       type='limit' if role == 'entry' else 'stop',
                       time_in_force='day' if role == 'entry' else 'gtc',
                       extended_hours=role == 'entry', client_order_id=key)
        request['limit_price' if role == 'entry' else 'stop_price'] = 20 if role == 'entry' else 17.49
        observation = dict(request, id=key, status=status, filled_qty=fill,
                           filled_avg_price=20 if fill else None)
        record = dict(role=role, operation='POST', sequence=0, client_order_id=key,
                      request={'order': request, 'order_id': None}, state='OBSERVED',
                      observation=deepcopy(observation))
        store.records[sid].append(record); broker.orders[key] = observation
    order('entry', 10, 'filled' if filled == 10 else 'partially_filled' if filled else 'new', filled)
    if filled:
        broker.held[symbol] = {'symbol': symbol, 'qty': filled, 'side': 'long'}
        if stop:
            order('stop', filled, 'new')
    return signal


def run(args, **kwargs):
    store, broker, runtime, handoff = args
    before = deepcopy((store.rows, store.records, broker.orders, broker.held))
    try:
        return broker_entry_preflight(runtime, Engine, store, broker,
            base_dir='unused', handoff=handoff, proposed_symbols={'NEW'}, **kwargs)
    finally:
        assert before == (store.rows, store.records, broker.orders, broker.held)


@pytest.mark.parametrize('filled', [0, 4, 10])
def test_pending_partial_and_protected_pass(filled):
    args = fixture(); add(args[0], args[1], args[3], filled=filled)
    assert run(args)['status'] == 'pass'


def test_flat_pass():
    assert run(fixture())['status'] == 'pass'


@pytest.mark.parametrize('status', ['pending_new', 'pending_cancel', 'pending_replace', 'held',
                                   'canceled', 'rejected', 'partially_filled'])
def test_unconfirmed_stop_blocks(status):
    args = fixture(); add(args[0], args[1], args[3])
    args[1].orders['onestop']['status'] = status
    with pytest.raises(ValueError, match='ENTRY_PREFLIGHT_STOP_COVERAGE_UNCONFIRMED'):
        run(args)


@pytest.mark.parametrize('field,value', [('qty', 9), ('stop_price', 13.67),
                                        ('filled_qty', 1), ('extended_hours', True)])
def test_altered_protection_blocks(field, value):
    args = fixture(); add(args[0], args[1], args[3])
    args[1].orders['onestop'][field] = value
    with pytest.raises(ValueError, match='ENTRY_PREFLIGHT_'):
        run(args)


def test_pending_replacement_pair_blocks_then_terminal_old_passes():
    args = fixture(); add(args[0], args[1], args[3])
    old = args[1].orders['onestop']; old['status'] = 'pending_replace'; old['replaced_by'] = 'newstop'
    record = deepcopy(args[0].records['one'][1])
    record.update(operation='PATCH', client_order_id='newstop', sequence=1)
    record['request']['order_id'] = old['id']
    record['request']['order']['client_order_id'] = 'newstop'
    new = dict(old, id='newstop', client_order_id='newstop', status='new', replaces=old['id'])
    record['observation'] = deepcopy(new)
    args[0].records['one'].append(record); args[1].orders['newstop'] = new
    with pytest.raises(ValueError, match='STOP_COVERAGE_UNCONFIRMED'):
        run(args)
    old['status'] = 'replaced'
    assert run(args)['status'] == 'pass'


@pytest.mark.parametrize('state', ['CLAIMED', 'UNKNOWN'])
def test_unknown_historical_order_always_blocks(state):
    args = fixture(); signal = add(args[0], args[1], args[3])
    signal['state'] = 'CLOSED'; args[0].records['one'][0]['state'] = state
    with pytest.raises(ValueError, match='ORDER_OUTCOME_UNRESOLVED'):
        run(args, exempt_signal_ids={'one'})


def test_batch_exemption_does_not_waive_entry_identity():
    args = fixture(); add(args[0], args[1], args[3], stop=False)
    with pytest.raises(ValueError, match='STOP_COVERAGE_UNCONFIRMED'):
        run(args)
    assert run(args, exempt_signal_ids={'one'})['exempt_signal_ids'] == ['one']
    args[1].orders['oneentry']['qty'] = 11
    with pytest.raises(ValueError, match='ENTRY_PREFLIGHT_'):
        run(args, exempt_signal_ids={'one'})


def test_exemption_cannot_cover_stop_or_unreconciled_position():
    args = fixture(); add(args[0], args[1], args[3])
    with pytest.raises(ValueError, match='EXEMPT_SCOPE_INVALID'):
        run(args, exempt_signal_ids={'one'})
    args = fixture(); add(args[0], args[1], args[3], stop=False)
    args[1].held['ARDT']['qty'] = 9
    with pytest.raises(ValueError, match='POSITION_NOT_RECONCILED'):
        run(args, exempt_signal_ids={'one'})


def test_recognition_only_delegates_native_validation_without_stop():
    args = fixture(); signal = add(args[0], args[1], args[3], symbol='PTCT', stop=False)
    signal['payload']['execution_owner'] = {'scope': 'recognition'}
    assert run(args)['recognized_existing_positions'] == [{'signal_id': 'one'}]
    def fail(*a, **k):
        raise ValueError('private details')
    args[2].validate_entry_recognition = fail
    with pytest.raises(ValueError, match='^ENTRY_PREFLIGHT_READ_OR_VALIDATION_FAILED$'):
        run(args)


def test_untracked_sell_blocks():
    args = fixture(); add(args[0], args[1], args[3])
    args[1].orders['rogue'] = dict(args[1].orders['onestop'], id='rogue', client_order_id='rogue')
    with pytest.raises(ValueError, match='UNTRACKED_ORDER'):
        run(args)


def test_read_error_sanitized():
    args = fixture()
    def fail():
        raise RuntimeError('private details')
    args[1].positions = fail
    with pytest.raises(ValueError, match='^ENTRY_PREFLIGHT_READ_OR_VALIDATION_FAILED$'):
        run(args)


def test_missing_observation_and_terminal_regression_block():
    args = fixture(); add(args[0], args[1], args[3])
    args[1].orders.pop('onestop')
    with pytest.raises(ValueError, match='ORDER_MISSING'):
        run(args)
    args = fixture(); add(args[0], args[1], args[3])
    args[1].orders['oneentry']['status'] = 'new'
    with pytest.raises(ValueError, match='TERMINAL_OBSERVATION_CHANGED'):
        run(args)


def test_position_snapshot_race_blocks():
    args = fixture(); add(args[0], args[1], args[3])
    calls = []
    def positions():
        calls.append(1)
        return deepcopy(args[1].held) if len(calls) == 1 else {}
    args[1].positions = positions
    with pytest.raises(ValueError, match='SNAPSHOT_CHANGED'):
        run(args)


def test_batch_exemption_session_bound_and_preexisting_still_protected():
    args = fixture(); add(args[0], args[1], args[3], stop=False)
    add(args[0], args[1], args[3], sid='two', symbol='OTHER', stop=False)
    with pytest.raises(ValueError, match='STOP_COVERAGE_UNCONFIRMED'):
        run(args, exempt_signal_ids={'two'})
    args[0].rows[0]['payload']['handoff']['session']['next_session'] = '2026-11-10'
    with pytest.raises(ValueError, match='EXEMPT_SCOPE_INVALID'):
        run(args, exempt_signal_ids={'one', 'two'})


def test_account_mismatch_and_unknown_exemption_block():
    args = fixture(); args[1].account = lambda: {'id': 'other'}
    with pytest.raises(ValueError, match='ACCOUNT_MISMATCH'):
        run(args)
    with pytest.raises(ValueError, match='SIGNAL_SCOPE_INVALID'):
        run(fixture(), exempt_signal_ids={'not-enrolled'})


def test_duplicate_ownership_and_stale_durable_quantity_block():
    args = fixture(); add(args[0], args[1], args[3])
    add(args[0], args[1], args[3], sid='two')
    with pytest.raises(ValueError, match='OWNERSHIP_CONFLICT'):
        run(args)
    args = fixture(); signal = add(args[0], args[1], args[3])
    signal['payload']['position']['qty'] = 9
    with pytest.raises(ValueError, match='POSITION_NOT_RECONCILED'):
        run(args)
