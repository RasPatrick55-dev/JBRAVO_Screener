"""Offline normal-module entry tests; existing transaction/SDK boundaries are doubles.

No configuration, network, PostgreSQL or provider is accessed. The real recorder,
compatibility entry point and atomic reconciliation writer execute together.
"""
import contextlib
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import TestCase, mock
import pytest

import test_account_sync as hourly
import test_reconciliation_integrity as integrity
import test_open_exit_lineage as lineage

pytestmark = pytest.mark.alpaca_optional


class SellFillClosureTests(TestCase):
    setUp = hourly.ExecutorGateIntegrationTests.setUp
    prepare = integrity.ActivityTests.prepare
    read_module = integrity.DatabaseReadTests.read_module

    def setup_writer(self, **trade_changes):
        self.read_module()  # Full scripts/db.py module, not selected functions.
        self.connection = lineage.Connection()
        self.connection.trades[0].update(trade_changes)
        self.stack.enter_context(mock.patch.object(self.module, '_maybe_conn',
            side_effect=lambda unused: contextlib.nullcontext(self.connection)))
        self.policy = mock.Mock()
        self.policy.revision = 7
        self.stack.enter_context(mock.patch.object(self.module, 'trade_policy', return_value=self.policy))
        self.writable = self.stack.enter_context(mock.patch.object(self.module.disposition, 'require_writable'))
        self.real_require_order_owner = self.module.disposition.require_order_owner
        self.owner = self.stack.enter_context(mock.patch.object(self.module.disposition, 'require_order_owner'))
        self.stack.enter_context(mock.patch.object(self.executor, 'db', self.module))
        self.stack.enter_context(mock.patch.object(self.module, '_log_write_result'))
        self.stack.enter_context(mock.patch.object(self.module, '_conn_or_none', side_effect=self.forbidden))
        self.insert_execution = self.stack.enter_context(mock.patch.object(self.module, 'insert_executed_trade', side_effect=self.forbidden))
        self.insert_event = self.stack.enter_context(mock.patch.object(self.module, 'insert_order_event', side_effect=self.forbidden))
        self.args = dict(symbol='FIXTURE', side='sell', qty=Decimal('2.0000'),
            price=Decimal('11.00'), status='filled', order_id=lineage.CHILD,
            order_type='trailing_stop', timestamp=lineage.START + timedelta(seconds=2),
            event_type='SELL_FILL', raw=lineage.provider(), trade_id=1,
            expected_policy_revision=7)

    def record(self, **changes):
        return self.executor.record_executed_trade(**(self.args | changes))

    def assert_no_changes(self, before):
        self.assertEqual(self.connection.trades, before)
        self.assertEqual(self.connection.fills, [])
        self.assertNotIn('financial_update', self.connection.actions)

    def test_full_exit_chooses_explicit_owned_lot_not_newest_symbol(self):
        self.setup_writer()
        other = lineage.trade(trade_id=2, entry_order_id=lineage.OTHER,
            entry_time=lineage.START + timedelta(seconds=1), qty=Decimal('9'))
        self.connection.trades.append(other)
        self.record()
        self.assertEqual(self.connection.trades[0]['status'], 'CLOSED')
        self.assertEqual(self.connection.trades[0]['qty'], Decimal('2'))
        self.assertEqual(self.connection.trades[1], other)
        self.assertEqual(len(self.connection.fills), 1)
        self.assertEqual(self.connection.actions.count('financial_update'), 1)
        self.insert_execution.assert_not_called()
        self.insert_event.assert_not_called()  # Canonical event written inside atomic transaction.

    def test_identical_replay_cannot_close_next_lot_or_duplicate_event(self):
        self.setup_writer()
        self.connection.trades.append(lineage.trade(trade_id=2,
            entry_order_id=lineage.OTHER, qty=Decimal('2')))
        self.record()
        after = deepcopy(self.connection.trades)
        self.record()
        self.assertEqual(self.connection.trades, after)
        self.assertEqual(len(self.connection.fills), 1)
        self.assertEqual(self.connection.actions.count('financial_update'), 1)

    def test_missing_explicit_owner_never_reads_or_writes_by_symbol(self):
        self.setup_writer()
        self.assertIs(self.record(trade_id=None, raw=lineage.provider(trade_id=1)), False)
        self.assertEqual(self.connection.queries, [])
        self.assertEqual(self.connection.fills, [])
        for value in (None, True, 0, -1, '1'):
            self.assertFalse(self.module.close_trade_on_sell_fill('FIXTURE', lineage.CHILD,
                self.args['timestamp'], '11', trade_id=value, qty='2', status='filled'))
        self.assertEqual(self.connection.queries, [])

    def test_partial_and_excess_fills_do_not_change_trade_quantity_or_close(self):
        for qty in ('1', '3'):
            with self.subTest(qty=qty):
                self.setup_writer()
                before = deepcopy(self.connection.trades)
                self.record(qty=qty, raw=lineage.provider(filled_qty=qty))
                self.assert_no_changes(before)
                self.assertIn('rollback', self.connection.actions)

    def test_nonfinal_partial_status_is_not_promoted_to_final(self):
        self.setup_writer()
        self.record(status='partially_filled', event_type='SELL_PARTIAL_FILL')
        self.assertEqual(self.connection.queries, [])

    def test_exact_decimal_quantity_reaches_writer_without_float_coercion(self):
        self.setup_writer()
        with mock.patch.object(self.module, 'reconcile_sell_fill', return_value=True) as writer:
            self.record(qty=Decimal('0.12345678901234567890123456789'))
        self.assertEqual(writer.call_args.kwargs['qty'], Decimal('0.12345678901234567890123456789'))
        self.assertEqual(writer.call_args.kwargs['expected_policy_revision'], 7)

    def test_ambiguous_owners_and_wrong_explicit_lot_are_refused(self):
        for change in ('duplicate_entry', 'duplicate_exit', 'wrong_lot'):
            with self.subTest(change=change):
                self.setup_writer()
                other = lineage.trade(trade_id=2, entry_order_id=lineage.OTHER)
                if change == 'duplicate_entry': other['entry_order_id'] = lineage.PARENT
                if change == 'duplicate_exit': other['exit_order_id'] = lineage.CHILD
                self.connection.trades.append(other)
                before = deepcopy(self.connection.trades)
                self.record(trade_id=2 if change == 'wrong_lot' else 1)
                self.assert_no_changes(before)

    def test_already_bound_exit_with_duplicate_entry_owner_never_mutates(self):
        for selected_status in ('OPEN', 'CLOSED'):
            for other_status in ('OPEN', 'CLOSED'):
                with self.subTest(selected_status=selected_status, other_status=other_status):
                    self.setup_writer(exit_order_id=lineage.CHILD, status=selected_status)
                    if selected_status == 'CLOSED':
                        self.connection.trades[0].update(exit_time=self.args['timestamp'],
                            exit_price=Decimal('11'), exit_reason='SELL_FILL', realized_pnl=Decimal('2'))
                    self.connection.trades.append(lineage.trade(trade_id=2,
                        entry_order_id=lineage.PARENT, status=other_status))
                    before = deepcopy(self.connection.trades)
                    self.assertIs(self.record(), False)
                    self.assert_no_changes(before)
                    self.assertIn('rollback', self.connection.actions)
                    self.assertIn('all_physical_binding_owners', self.connection.actions)

    def test_unbound_legacy_fill_event_does_not_gain_owner_from_entry_union(self):
        self.setup_writer()
        self.connection.fills = [dict(order_id=lineage.CHILD, symbol='FIXTURE',
            qty='2', status='filled', event_time=self.args['timestamp'])]
        before = deepcopy(self.connection.trades)
        self.assertIs(self.record(), False)
        self.assertEqual(self.connection.trades, before)
        self.assertEqual(len(self.connection.fills), 1)
        self.assertNotIn('financial_update', self.connection.actions)

    def test_entry_reservation_is_checked_even_for_already_bound_exit(self):
        self.setup_writer(exit_order_id=lineage.CHILD)
        def reservation(policy, trade_id, *orders):
            if lineage.PARENT in orders: raise ValueError('synthetic_entry_reserved_elsewhere')
        self.owner.side_effect = reservation
        before = deepcopy(self.connection.trades)
        self.assertIs(self.record(), False)
        self.assert_no_changes(before)
        self.owner.assert_any_call(self.policy, 1, lineage.PARENT, lineage.CHILD)

    def test_retained_crc110_entry_claim_blocks_bound_exit_with_real_policy_validator(self):
        self.setup_writer(exit_order_id=lineage.CHILD)
        self.policy.allocations = []
        self.policy.allocation_metadata_json = {'reserved_order_owners': {lineage.PARENT:110}}
        self.owner.side_effect = self.real_require_order_owner
        before = deepcopy(self.connection.trades)
        self.assertIs(self.record(), False)
        self.assert_no_changes(before)
        self.owner.assert_any_call(self.policy, 1, lineage.PARENT, lineage.CHILD)

    def test_replacement_is_not_inferred_from_payload_or_equal_quantity(self):
        self.setup_writer(exit_order_id=lineage.OTHER)
        before = deepcopy(self.connection.trades)
        self.record(raw=lineage.provider(replaces=lineage.OTHER))
        self.assert_no_changes(before)

    def test_already_correlated_replacement_can_close_and_old_order_replay_cannot(self):
        self.setup_writer(exit_order_id=lineage.CHILD)
        self.record(raw=lineage.provider(replaces=lineage.OTHER))
        after = deepcopy(self.connection.trades)
        self.assertEqual(after[0]['status'], 'CLOSED')
        self.record(order_id=lineage.OTHER, raw=lineage.provider(id=lineage.OTHER))
        self.assertEqual(self.connection.trades, after)
        self.assertEqual(len(self.connection.fills), 1)

    def test_reserved_owner_is_checked_even_when_physical_exit_is_already_bound(self):
        self.setup_writer(exit_order_id=lineage.CHILD)
        self.owner.side_effect = ValueError('synthetic_reserved_owner_conflict')
        before = deepcopy(self.connection.trades)
        self.record()
        self.assert_no_changes(before)
        self.owner.assert_called_once_with(self.policy, 1, lineage.CHILD)

    def test_policy_exclusion_and_changed_revision_cannot_reach_mutation(self):
        self.setup_writer()
        self.writable.side_effect = ValueError('synthetic_excluded')
        self.record()
        self.assertEqual(self.connection.fills, [])
        with mock.patch.object(self.module, 'trade_policy', side_effect=ValueError('synthetic_revision_drift')) as read:
            self.record()
        self.assertEqual(read.call_args.kwargs['expected_revision'], 7)
        self.assertNotIn('financial_update', self.connection.actions)

    def test_reversed_future_and_missing_fill_time_are_refused(self):
        for stamp in (lineage.START - timedelta(microseconds=1),
                datetime.now(timezone.utc) + timedelta(days=1), None):
            with self.subTest(stamp=stamp):
                self.setup_writer(exit_order_id=lineage.CHILD)
                before = deepcopy(self.connection.trades)
                self.record(timestamp=stamp)
                self.assert_no_changes(before)

    def test_conflicting_duplicate_fill_events_refuse_without_merging(self):
        self.setup_writer(exit_order_id=lineage.CHILD)
        fill = dict(order_id=lineage.CHILD, symbol='FIXTURE', qty='2', status='filled', event_time=self.args['timestamp'])
        self.connection.fills = [fill, deepcopy(fill)]
        before = deepcopy(self.connection.trades)
        self.record()
        self.assertEqual(self.connection.trades, before)
        self.assertEqual(len(self.connection.fills), 2)
        self.assertNotIn('financial_update', self.connection.actions)

    def test_update_failure_rolls_back_canonical_event_as_well(self):
        self.setup_writer()
        original = lineage.Cursor.execute
        def execute(cursor, sql, args=None):
            if sql.startswith('UPDATE public.trades SET exit_order_id'):
                raise RuntimeError('synthetic update refusal')
            return original(cursor, sql, args)
        before = deepcopy(self.connection.trades)
        with mock.patch.object(lineage.Cursor, 'execute', execute): self.record()
        self.assert_no_changes(before)
        self.assertIn('rollback', self.connection.actions)

    def test_lost_commit_acknowledgement_does_not_make_replay_close_another_lot(self):
        self.setup_writer()
        self.connection.trades.append(lineage.trade(trade_id=2, entry_order_id=lineage.OTHER))
        original = lineage.Connection.__exit__
        def exit_once(connection, kind, value, trace):
            answer = original(connection, kind, value, trace)
            if kind is None: raise RuntimeError('synthetic lost commit acknowledgement')
            return answer
        with mock.patch.object(lineage.Connection, '__exit__', exit_once): self.record()
        committed = deepcopy(self.connection.trades)
        self.assertEqual(committed[0]['status'], 'CLOSED')
        self.record()
        self.assertEqual(self.connection.trades, committed)
        self.assertEqual(len(self.connection.fills), 1)
        self.assertEqual(self.connection.actions.count('financial_update'), 1)

    def test_lock_busy_and_malformed_numbers_cannot_reach_trade_mutation(self):
        self.setup_writer()
        self.connection.order_busy = True
        before = deepcopy(self.connection.trades)
        self.record()
        self.assert_no_changes(before)
        for field, values in [('qty', [True, '0', '-1', 'NaN', 'Infinity']),
                              ('price', [True, '0', '-1', 'NaN', 'Infinity'])]:
            for value in values:
                self.setup_writer()
                self.record(**{field:value})
                self.assertEqual(self.connection.queries, [])

    def test_complete_owned_exit_preserves_existing_gross_gain_loss_and_zero(self):
        for price, pnl in [('9', -2), ('10', 0), ('11', 2)]:
            self.setup_writer(exit_order_id=lineage.CHILD)
            self.record(price=price, raw=lineage.provider(filled_avg_price=price))
            self.assertEqual(self.connection.trades[0]['realized_pnl'], pnl)
            self.assertEqual(self.connection.trades[0]['qty'], Decimal('2'))

    def test_buy_and_trailing_submission_keep_existing_recording_route(self):
        self.setup_writer()
        with mock.patch.object(self.module, 'insert_executed_trade', return_value=True) as execution, \
             mock.patch.object(self.module, 'insert_order_event', return_value=True) as event, \
             mock.patch.object(self.module, 'upsert_trade_on_buy_fill', return_value=True), \
             mock.patch.object(self.module, 'count_open_trades', return_value=1), \
             mock.patch.object(self.module, 'close_trade_on_sell_fill', side_effect=self.forbidden):
            self.record(side='buy', event_type='BUY_FILL')
            self.record(event_type='TRAIL_SUBMIT', status='submitted')
        self.assertEqual(execution.call_count, 2)
        self.assertEqual(event.call_count, 2)
