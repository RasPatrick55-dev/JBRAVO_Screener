"""GET/SELECT-only entry admission, independent of lifecycle-monitor health.

Caller holds the source and account leases. Exempt IDs may only be entries
created by that caller in its current short batch, never persisted exemptions.
No observation is written and no lifecycle method is invoked here.
"""
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
from types import SimpleNamespace


class _Unsafe(ValueError):
    pass


def _require(value, code):
    if not value:
        raise _Unsafe('ENTRY_PREFLIGHT_' + code)


def _number(value):
    _require(not isinstance(value, bool), 'NUMBER_INVALID')
    result = Decimal(str(value))
    _require(result.is_finite() and result >= 0, 'NUMBER_INVALID')
    return result


def _qty(value):
    result = _number(value)
    _require(result == result.to_integral_value(), 'QUANTITY_INVALID')
    return int(result)


def broker_entry_preflight(runtime, NativeEngine, store, broker, *, base_dir,
                           handoff, proposed_symbols, exempt_signal_ids=()):
    """Return fresh proof or a fixed ValueError code; never change broker/store.

Read failures and malformed external values are deliberately sanitized. A
proof is valid only for this admission, not a receipt that may be cached.
"""
    try:
        return _preflight(runtime, NativeEngine, store, broker, base_dir=base_dir,
                          handoff=handoff, proposed_symbols=proposed_symbols,
                          exempt_signal_ids=exempt_signal_ids)
    except _Unsafe:
        raise
    except Exception:
        raise ValueError('ENTRY_PREFLIGHT_READ_OR_VALIDATION_FAILED') from None


def _preflight(runtime, NativeEngine, store, broker, *, base_dir, handoff,
               proposed_symbols, exempt_signal_ids):
    terminal = {'filled', 'canceled', 'expired', 'rejected', 'replaced'}
    live_statuses = {'new', 'accepted', 'pending_new', 'partially_filled',
                     'accepted_for_bidding', 'held', 'pending_replace', 'pending_cancel',
                     'stopped', 'suspended', 'calculated', 'done_for_day'}
    _require(broker.paper is True and broker.account_id == store.account_id,
             'ACCOUNT_MISMATCH')
    _require(broker.account()['id'] == store.account_id, 'ACCOUNT_MISMATCH')
    runtime.source_guard(handoff, base_dir=base_dir)
    signals = store.signals(active=False)
    _require(len(signals) <= 2048, 'POPULATION_LIMIT')
    ids = {s['signal_id'] for s in signals}
    exempt = set(exempt_signal_ids)
    _require(len(ids) == len(signals) and exempt <= ids and len(exempt) <= 3,
             'SIGNAL_SCOPE_INVALID')
    records = {s['signal_id']: store.orders(s['signal_id']) for s in signals}
    all_records = [r for rows in records.values() for r in rows]
    _require(len(all_records) <= 8192, 'POPULATION_LIMIT')
    _require(all(r['state'] in {'OBSERVED', 'REJECTED'} for r in all_records),
             'ORDER_OUTCOME_UNRESOLVED')
    keys = [r['client_order_id'] for r in all_records]
    _require(len(keys) == len(set(keys)), 'ORDER_IDENTITY_CONFLICT')
    positions = broker.positions()
    opens = broker.open_orders()
    _require(isinstance(positions, dict) and isinstance(opens, list) and len(opens) < 500,
             'SNAPSHOT_INVALID')
    open_ids = [o['id'] for o in opens]
    _require(len(open_ids) == len(set(open_ids)), 'ORDER_IDENTITY_CONFLICT')
    checker = SimpleNamespace()
    observed = {}; observed_owner = {}; per_signal = {}
    for signal in signals:
        sid = signal['signal_id']; payload = signal['payload']
        _require(payload['account_id'] == broker.account_id, 'ACCOUNT_MISMATCH')
        _require(signal['state'] in {'PREPARED', 'ENTRY_PENDING', 'OPEN',
                                    'EXIT_PENDING', 'CLOSED', 'EXPIRED'}, 'SIGNAL_UNSETTLED')
        rows = []
        for record in records[sid]:
            _require(record['role'] in {'entry', 'stop', 'exit', 'partial', 'cancel'} and
                     record['operation'] in {'POST', 'PATCH', 'DELETE'}, 'ORDER_IDENTITY_CONFLICT')
            order = (broker.get_order(record['request']['order_id'])
                     if record['operation'] == 'DELETE' else
                     broker.find_by_client_order_id(record['client_order_id']))
            if record['state'] == 'REJECTED':
                _require(order is None or order['status'] == 'rejected', 'REJECTION_CHANGED')
                if order is None:
                    continue
            _require(isinstance(order, dict), 'ORDER_MISSING')
            _require(order['status'] in terminal | live_statuses, 'ORDER_STATUS_INVALID')
            NativeEngine._check_order(checker, record, order)
            _require(order['symbol'] == payload['proposal']['symbol'], 'ORDER_IDENTITY_CONFLICT')
            if record['operation'] != 'DELETE':
                _require(order['side'] == ('buy' if record['role'] == 'entry' else 'sell') and
                         record['role'] != 'cancel', 'ORDER_IDENTITY_CONFLICT')
            _require(_qty(order.get('filled_qty', 0)) <= _qty(order['qty']), 'FILL_INVALID')
            old = record.get('observation')
            if old:
                _require(order['id'] == old['id'] and
                         _qty(order.get('filled_qty', 0)) >= _qty(old.get('filled_qty', 0)),
                         'ORDER_IDENTITY_CONFLICT')
                if old['status'] in terminal:
                    _require(order['status'] == old['status'] and
                             _qty(order.get('filled_qty', 0)) == _qty(old.get('filled_qty', 0)),
                             'TERMINAL_OBSERVATION_CHANGED')
            oid = order['id']
            if oid in observed:
                _require(observed[oid] == order and observed_owner[oid] == sid and
                         record['operation'] == 'DELETE', 'ORDER_IDENTITY_CONFLICT')
            else:
                observed[oid] = order; observed_owner[oid] = sid
            if record['operation'] != 'DELETE':
                rows.append((record, order))
        per_signal[sid] = rows
    # Require consistent list/get snapshots and refuse unknown orders on owned
    # or candidate symbols. Unrelated manual exposure remains globally visible.
    relevant = set(proposed_symbols) | {s['payload']['proposal']['symbol'] for s in signals
                                      if s['state'] not in {'CLOSED', 'EXPIRED'}}
    fields = ('id', 'client_order_id', 'symbol', 'side', 'type', 'status', 'qty',
              'filled_qty', 'stop_price', 'limit_price', 'time_in_force', 'extended_hours')
    for order in opens:
        if order['symbol'] in relevant or order['id'] in observed:
            _require(order['id'] in observed, 'UNTRACKED_ORDER')
            current = observed[order['id']]
            _require(all(order.get(k) == current.get(k) for k in fields), 'SNAPSHOT_CHANGED')
    _require(all(o['status'] in terminal or oid in open_ids for oid, o in observed.items()),
             'SNAPSHOT_CHANGED')
    recognized = []; protected = []; occupied = set()
    for signal in signals:
        sid = signal['signal_id']; payload = signal['payload']
        symbol = payload['proposal']['symbol']; rows = per_signal[sid]
        active = signal['state'] not in {'CLOSED', 'EXPIRED'}
        if active:
            _require(symbol not in occupied, 'OWNERSHIP_CONFLICT'); occupied.add(symbol)
        if 'execution_owner' in payload or 'ownership_history' in payload:
            _require(sid not in exempt, 'EXEMPT_SCOPE_INVALID')
            recognized.append(runtime.validate_entry_recognition(signal, records[sid],
                positions=positions, open_orders=opens, account_id=broker.account_id,
                current_source_bindings=handoff['source_bindings']))
            continue
        _require('manual_paper_test' not in payload['handoff'], 'OWNERSHIP_UNRECOGNIZED')
        if active:
            runtime.source_guard(runtime.lifecycle_handoff(signal), base_dir=base_dir, lifecycle=True)
        entries = [o for r, o in rows if r['role'] == 'entry']
        _require(len(entries) <= 1, 'OWNERSHIP_CONFLICT')
        sells = [o for r, o in rows if r['role'] in {'stop', 'exit', 'partial'}]
        remaining = sum(_qty(o.get('filled_qty', 0)) for o in entries) - sum(
            _qty(o.get('filled_qty', 0)) for o in sells)
        _require(remaining >= 0, 'FILL_INVALID')
        live = [(r, o) for r, o in rows if o['status'] not in terminal]
        if not active:
            _require(remaining == 0 and not live, 'CLOSED_SIGNAL_EXPOSURE')
            continue
        actual = positions.get(symbol)
        _require((remaining == 0 and actual is None) or
                 (remaining > 0 and actual is not None and actual.get('side') == 'long' and
                  _qty(actual['qty']) == remaining), 'POSITION_NOT_RECONCILED')
        if sid in exempt:
            _require(len(entries) == 1 and all(r['role'] == 'entry' and r['operation'] == 'POST'
                     for r in records[sid]) and
                     payload['handoff']['session'] == handoff['session'] and
                     payload['handoff']['execution_policy'] == handoff['execution_policy'],
                     'EXEMPT_SCOPE_INVALID')
            continue
        if remaining == 0:
            _require(not any(r['role'] != 'entry' for r, o in live) and
                     not payload.get('position'), 'FLAT_STATE_UNSETTLED')
            continue
        position = payload.get('position')
        _require(isinstance(position, dict) and position['symbol'] == symbol and
                 _qty(position['qty']) == remaining, 'POSITION_NOT_RECONCILED')
        sells_live = [(r, o) for r, o in live if r['role'] in {'stop', 'exit', 'partial'}]
        _require(len(sells_live) == 1 and sells_live[0][0]['role'] == 'stop',
                 'STOP_COVERAGE_UNCONFIRMED')
        stop = sells_live[0][1]
        minimum = max(_number(position.get('trailing_stop') or 0),
                      _number(position.get('atr_stop') or 0)).quantize(
                          Decimal('.01'), rounding=ROUND_FLOOR)
        _require(minimum > 0 and stop['status'] in {'new', 'accepted'} and
                 stop['symbol'] == symbol and stop['side'] == 'sell' and stop['type'] == 'stop' and
                 stop['time_in_force'] == 'gtc' and stop['extended_hours'] is False and
                 _qty(stop.get('filled_qty', 0)) == 0 and _qty(stop['qty']) == remaining and
                 _number(stop['stop_price']) >= minimum, 'STOP_COVERAGE_UNCONFIRMED')
        protected.append(sid)
    _require(broker.positions() == positions, 'SNAPSHOT_CHANGED')
    return dict(status='pass', protected_signal_ids=protected,
                recognized_existing_positions=recognized, exempt_signal_ids=sorted(exempt),
                observed_at=datetime.now(timezone.utc).isoformat(),
                scope='FRESH_BROKER_REGULAR_HOURS_STOPS_ONLY', mutations=0)
