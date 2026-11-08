"""Narrow, inert adapter for an unresolved native stop-tightening replacement.

Install OUTSIDE the release's QuoteSafeEngine. No order/position/state is
rewritten, no broker mutation is added, and entry admission is unchanged.
The caller still performs native sync before manage/completed. A pending pair
is uncertainty, never evidence of confirmed protection or successful replace.
"""
import hashlib
import inspect
import textwrap


_METHOD_HASHES = {
    '_check_order': 'a42c977f8cc39cac131608cde86ab013027a9227bf5beef3084276c574ffe6e3',
    'completed': 'ea80ca4c4c716f9719a04323861c9481722ea79c960ef8fa3d548bc47dbe5e73',
    'manage': '7c26244b97053637569daab0eccfab599c2f1b5c27408c39181813c71ddf6043',
}


def _require(condition):
    if not condition:
        raise ValueError('stop_replacement_pending_proof_invalid')


def monitor_engine(Engine, *, NativeEngine):
    """Wrap the already composed lifecycle/quote-safe engine, never its base.

Source pins and the unmodified native checker establish the supported
interface. The release bootstrap must also retain its full native inventory
and account/source/recognition guards. This factory has no external effects.
"""
    for name, digest in _METHOD_HASHES.items():
        source = textwrap.dedent(inspect.getsource(getattr(NativeEngine, name))).strip()
        if hashlib.sha256(source.encode()).hexdigest() != digest:
            raise ValueError('stop_replacement_native_' + name + '_source_changed')
    _require(issubclass(Engine, NativeEngine))
    core = NativeEngine.manage.__globals__
    rules, quantity, terminal = core['rules'], core['quantity'], core['TERMINAL']

    def active_stops(records):
        return [r for r in records if r['role'] == 'stop' and r.get('observation')
                and r['observation']['status'] not in terminal]

    def pending_pair(self, signal, records):
        """Return IDs only on complete current proof; never cache a skip flag.

        Store.orders is scoped by account/signal and does not expose its lineage
        column. Recompute native account/signal/role/sequence client identities,
        check native request/observation identities, and bind the PATCH target.
        """
        try:
            stops = active_stops(records)
            if len(stops) != 2:
                return None
            payload = signal['payload']
            position = payload['position']
            symbol = payload['proposal']['symbol']
            own_qty = quantity(position['qty'])
            _require(self.broker.paper is True and self.broker.account_id ==
                     self.store.account_id == payload['account_id'])
            _require(signal['state'] == 'OPEN' and own_qty > 0 and
                     position['symbol'] == symbol and
                     not payload['handoff'].get('manual_paper_test') and
                     'execution_owner' not in payload and 'ownership_history' not in payload)
            _require(all(r['state'] in {'OBSERVED', 'REJECTED'} for r in records))
            keys, observed_ids = set(), set()
            entries, sold, partial_sold = [], 0, 0
            for record in records:
                role = record['role']
                _require(record['client_order_id'] == rules.client_order_id(
                    self.store.account_id, signal['signal_id'], role=role,
                    sequence=record['sequence']))
                _require(record['client_order_id'] not in keys)
                keys.add(record['client_order_id'])
                # These fields are present in some in-memory ports, but not in
                # the pinned store's scoped SELECT. Never ignore them if given.
                _require(record.get('signal_id', signal['signal_id']) == signal['signal_id'])
                _require(record.get('account_id', self.store.account_id) == self.store.account_id)
                observation = record['observation']
                if record['state'] == 'REJECTED':
                    _require(observation is None)
                    continue
                _require(isinstance(observation, dict))
                NativeEngine._check_order(self, record, observation)
                _require(observation['symbol'] == symbol)
                if record['operation'] == 'DELETE':
                    _require(role == 'cancel' and observation['status'] in terminal)
                    continue
                _require(record['operation'] in {'POST', 'PATCH'} and
                         role in {'entry', 'stop', 'exit', 'partial'})
                _require(isinstance(observation['id'], str) and observation['id'] and
                         observation['id'] not in observed_ids)
                observed_ids.add(observation['id'])
                _require(record['request']['order']['client_order_id'] == record['client_order_id'])
                filled = quantity(observation['filled_qty'])
                _require(quantity(observation['qty']) > 0 and
                         observation['side'] == ('buy' if role == 'entry' else 'sell'))
                if role == 'entry':
                    _require(record['operation'] == 'POST' and observation['status'] in terminal)
                    entries.append(observation)
                else:
                    sold += filled
                    if role == 'partial':
                        partial_sold += filled
                    _require(role == 'stop' or observation['status'] in terminal)
            _require(len(entries) == 1)
            bought = quantity(entries[0]['filled_qty'])
            _require(bought > 0 and bought - sold == own_qty and
                     quantity(payload['entry_filled_qty']) == bought)
            actual = self.broker.positions().get(symbol)
            _require(actual is not None and actual.get('side') == 'long' and
                     quantity(actual['qty']) == own_qty)
            if payload['pending_exit_reasons'] == ['PARTIAL_5PCT'] and payload.get('partial_target_qty') is not None:
                # Native sync clears this intent when observed partial fills
                # complete it. Do not hide a stale, unreconciled strategy state.
                _require(partial_sold < quantity(payload['partial_target_qty']))
            old = next(r for r in stops if r['observation']['status'] == 'pending_replace')
            new = next(r for r in stops if r is not old)
            before, after = old['observation'], new['observation']
            _require(old['state'] == new['state'] == 'OBSERVED' and
                     old['operation'] in {'POST', 'PATCH'} and new['operation'] == 'PATCH')
            _require(new['sequence'] == old['sequence'] + 1 and
                     new['sequence'] == max(r['sequence'] for r in records if r['role'] == 'stop'))
            _require(after['status'] in {'new', 'accepted'} and
                     before['replaced_by'] == after['id'] and after['replaces'] == before['id'] and
                     new['request']['order_id'] == before['id'] and
                     after.get('replaced_by') is None)
            for record in (old, new):
                order = record['observation']
                _require(order['side'] == 'sell' and order['type'] == 'stop' and
                         order['time_in_force'] == 'gtc' and order['extended_hours'] is False and
                         quantity(order['qty']) == own_qty and quantity(order['filled_qty']) == 0)
            _require(rules.number(after['stop_price'], positive=True) >=
                     rules.number(before['stop_price'], positive=True))
            # If the predecessor is itself a PATCH, its own durable target
            # must still be a distinct terminal historical stop in this scope.
            if old['operation'] == 'PATCH':
                ancestor = [r for r in records if r['role'] == 'stop' and r.get('observation') and
                            r['observation']['id'] == old['request']['order_id']]
                _require(len(ancestor) == 1 and ancestor[0]['observation']['status'] in terminal and
                         before['replaces'] == ancestor[0]['observation']['id'])
            else:
                _require(old['request'].get('order_id') is None and before.get('replaces') is None)
            return dict(old_order_id=before['id'], replacement_order_id=after['id'],
                        protection_confirmed=False)
        except (ValueError, KeyError, TypeError, StopIteration, AttributeError, OverflowError):
            return None

    class PendingStopReplacementEngine(Engine):
        def manage(self, signal, records, *, now, quote, regular_open):
            pending = pending_pair(self, signal, records)
            if pending is not None:
                self.note(signal, 'STOP_REPLACEMENT_PENDING', **pending)
                return
            return super().manage(signal, records, now=now, quote=quote, regular_open=regular_open)

        def completed(self, signal, frame, *, session, next_session, now):
            # The real release calls completed between ticks. Always read its
            # current scoped durable orders, even if manage has not been called.
            records = self.store.orders(signal['signal_id'])
            pending = pending_pair(self, signal, records)
            if pending is not None:
                self.note(signal, 'STOP_REPLACEMENT_PENDING', completed_session_deferred=session, **pending)
                return
            if any(r['state'] in {'CLAIMED', 'UNKNOWN'} for r in records):
                raise ValueError('stop_replacement_completed_order_unresolved')
            if len(active_stops(records)) > 1:
                raise ValueError('strategy_multiple_active_stops')
            return super().completed(signal, frame, session=session, next_session=next_session, now=now)

    return PendingStopReplacementEngine
