"""Normal-import offline reconciliation tests; all external effects are doubles.

Run with unittest (repository pytest conftest imports the external Alpaca SDK).
The executor uses the same pre-import isolation as the hourly gate tests.
"""
import contextlib
import copy
from datetime import datetime, timedelta, timezone
import importlib
import io
import json
import logging
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest import mock

import test_account_sync as hourly_tests
from scripts import account_sync as sync


NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


class FrozenRunClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


def order(**changes):
    return dict(id='sell-1', symbol='XYZ', side='sell', status='filled',
                filled_at=NOW.isoformat(), updated_at=NOW.isoformat(),
                filled_qty='2', filled_avg_price='11', type='market') | changes


class ReconciliationTests(unittest.TestCase):
    setUp = hourly_tests.ExecutorGateIntegrationTests.setUp

    def prepare(self, *, open_trades=None, missing=None, orders=None, positions=None):
        module = self.executor
        self.stack.enter_context(mock.patch.object(module, 'datetime', FrozenRunClock))
        self.messages = io.StringIO()
        logger = logging.Logger('synthetic-reconciliation', logging.INFO)
        logger.addHandler(logging.StreamHandler(self.messages))
        self.stack.enter_context(mock.patch.object(module, 'LOGGER', logger))
        self.stack.enter_context(mock.patch.object(module, 'log_info'))
        db = module.db
        self.db = db
        self.trade = dict(trade_id=1, symbol='XYZ', qty=2, entry_price=10,
                          entry_time=NOW, entry_order_id='entry-1')
        self.writes = {}
        for name, result in dict(db_enabled=True, get_db_conn=object(),
                get_reconcile_state={'last_after': NOW},
                get_open_trades=[self.trade] if open_trades is None else open_trades,
                get_closed_trades_missing_exit=[] if missing is None else missing,
                set_reconcile_state=True, close_trade=True, insert_order_event=True,
                decorate_trade_exit=True, reconcile_sell_fill=True).items():
            patched = self.stack.enter_context(mock.patch.object(db, name,
                        return_value=result, create=True))
            self.writes[name] = patched

        def timestamp(value, **unused):
            try:
                if value is None:
                    return None
                return value if isinstance(value, datetime) else datetime.fromisoformat(value)
            except (TypeError, ValueError):
                return None
        self.stack.enter_context(mock.patch.object(db, 'normalize_ts',
                                                  side_effect=timestamp, create=True))
        self.http_orders = module.alpaca_list_orders_http
        self.orders = self.stack.enter_context(mock.patch.object(module, 'alpaca_list_orders_http',
                               return_value=[] if orders is None else orders))
        self.client = mock.Mock()
        self.client.get_all_positions.return_value = [] if positions is None else positions
        self.client.submit_order.side_effect = self.forbidden
        self.client.cancel_order_by_id.side_effect = self.forbidden
        self.run = module.TradeExecutor(module.ExecutorConfig(), self.client,
                                       module.ExecutionMetrics())

    def no_mutations(self):
        for name in ('close_trade', 'insert_order_event', 'decorate_trade_exit',
                     'reconcile_sell_fill', 'set_reconcile_state'):
            self.writes[name].assert_not_called()
        self.client.submit_order.assert_not_called()
        self.client.cancel_order_by_id.assert_not_called()

    def test_each_future_order_time_blocks_before_any_trade_write(self):
        for field in ('updated_at', 'filled_at', 'submitted_at', 'created_at'):
            with self.subTest(field=field):
                self.prepare(orders=[order(**{field: (NOW + timedelta(microseconds=1)).isoformat()})])
                self.assertFalse(self.run.reconcile_closed_trades())
                self.no_mutations()
                self.assertIn('reason=future_timestamp', self.messages.getvalue())

    def test_order_time_cutoff_and_equivalent_offset_are_accepted(self):
        for value in (NOW.isoformat(), (NOW - timedelta(microseconds=1)).isoformat(),
                      '2026-10-09T08:00:00-04:00'):
            with self.subTest(value=value):
                self.prepare(open_trades=[], orders=[order(updated_at=value, filled_at=value)])
                self.assertTrue(self.run.reconcile_closed_trades())
                saved = self.writes['set_reconcile_state'].call_args.args[1]
                self.assertLessEqual(saved, NOW)

    def test_future_saved_order_watermark_stops_before_broker_read(self):
        self.prepare(open_trades=[])
        self.writes['get_reconcile_state'].return_value = {'last_after': NOW + timedelta(microseconds=1)}
        self.assertFalse(self.run.reconcile_closed_trades())
        self.orders.assert_not_called()
        self.no_mutations()

    def test_position_failure_is_not_an_empty_account(self):
        self.prepare()
        self.client.get_all_positions.side_effect = RuntimeError('secret-provider-text')
        self.assertFalse(self.run.reconcile_closed_trades())
        self.no_mutations()
        self.assertIn('RECONCILE_POSITIONS_UNAVAILABLE', self.messages.getvalue())
        self.assertNotIn('secret-provider-text', self.messages.getvalue())

    def test_malformed_position_evidence_blocks_writes(self):
        for positions in ({}, [SimpleNamespace(symbol='XYZ')],
                          [SimpleNamespace(symbol='XYZ', qty='NaN')],
                          [SimpleNamespace(symbol='', qty='2')],
                          [SimpleNamespace(symbol='XYZ', qty=True)]):
            with self.subTest(positions=positions):
                self.prepare(positions=positions)
                self.assertFalse(self.run.reconcile_closed_trades())
                self.no_mutations()

    def test_orders_failure_precedes_position_absence_closure(self):
        self.prepare()
        self.orders.side_effect = RuntimeError('secret-response')
        self.assertFalse(self.run.reconcile_closed_trades())
        self.no_mutations()
        self.client.get_all_positions.assert_not_called()
        self.assertNotIn('secret-response', self.messages.getvalue())

    def test_full_order_window_is_incomplete_and_does_not_advance(self):
        self.prepare(orders=[order()])
        self.assertFalse(self.run.reconcile_closed_trades(limit=1))
        self.no_mutations()
        self.assertIn('RECONCILE_ORDERS_INCOMPLETE', self.messages.getvalue())

    def test_malformed_duplicate_or_untimed_orders_are_not_empty_results(self):
        for orders in ({}, [None], [{}], [order(), order()],
                       [{**order(), 'updated_at': None, 'filled_at': None}],
                       [{**order(), 'filled_avg_price': 'NaN'}]):
            with self.subTest(orders=orders):
                self.prepare(orders=orders)
                self.assertFalse(self.run.reconcile_closed_trades())
                self.no_mutations()

    def test_valid_empty_orders_and_positions_preserve_closure_contract(self):
        self.prepare()
        self.assertTrue(self.run.reconcile_closed_trades())
        self.assertEqual(self.writes['close_trade'].call_args.kwargs['exit_reason'], 'POSITION_CLOSED')
        self.writes['set_reconcile_state'].assert_called_once()
        self.assertEqual(self.writes['set_reconcile_state'].call_args.args[1], NOW)

    def test_empty_database_and_empty_broker_results_complete(self):
        self.prepare(open_trades=[])
        self.assertTrue(self.run.reconcile_closed_trades())
        self.writes['close_trade'].assert_not_called()
        self.writes['set_reconcile_state'].assert_called_once()

    def test_usable_filled_exit_retains_existing_terms_and_overlap(self):
        self.prepare(orders=[order()], positions=[SimpleNamespace(symbol='XYZ', qty='2')])
        self.assertTrue(self.run.reconcile_closed_trades())
        args = self.writes['reconcile_sell_fill'].call_args.kwargs
        self.assertEqual((args['trade_id'], args['order_id']), (1, 'sell-1'))
        self.assertEqual((args['exit_price'], args['exit_reason']), (11.0, 'SELL_FILL'))
        self.writes['close_trade'].assert_not_called()
        self.writes['insert_order_event'].assert_not_called()
        self.assertEqual(self.writes['set_reconcile_state'].call_args.args[1].timestamp(),
                         NOW.timestamp() - 300)

    def test_failed_database_read_does_not_become_empty_account(self):
        self.prepare()
        self.writes['get_open_trades'].side_effect = RuntimeError('read-failed')
        self.assertFalse(self.run.reconcile_closed_trades())
        self.no_mutations()

    def test_invalid_or_unavailable_watermark_blocks_writes(self):
        for state in (None, {'last_after': 'bad'}):
            with self.subTest(state=state):
                self.prepare()
                self.writes['get_reconcile_state'].return_value = state
                self.assertFalse(self.run.reconcile_closed_trades())
                self.no_mutations()

    def test_failed_close_holds_watermark(self):
        self.prepare()
        self.writes['close_trade'].return_value = False
        self.assertFalse(self.run.reconcile_closed_trades())
        self.writes['set_reconcile_state'].assert_not_called()

    def test_missing_decoration_holds_watermark(self):
        self.prepare(open_trades=[])
        self.writes['get_closed_trades_missing_exit'].return_value = [self.trade]
        self.assertFalse(self.run.reconcile_closed_trades())
        self.writes['set_reconcile_state'].assert_not_called()
        self.assertIn('RECONCILE_DECORATE_MISS', self.messages.getvalue())

    def test_failed_atomic_fill_or_decoration_holds_watermark(self):
        for decorate in (False, True):
            with self.subTest(decorate=decorate):
                self.prepare(open_trades=[], orders=[order()])
                self.writes['get_closed_trades_missing_exit' if decorate else 'get_open_trades'].return_value = [self.trade]
                self.client.get_all_positions.return_value = [SimpleNamespace(symbol='XYZ', qty='2')]
                self.writes['reconcile_sell_fill'].return_value = False
                self.assertFalse(self.run.reconcile_closed_trades())
                self.writes['set_reconcile_state'].assert_not_called()
                self.writes['insert_order_event'].assert_not_called()
                self.writes['decorate_trade_exit'].assert_not_called()

    def test_atomic_decoration_uses_the_same_fill_contract(self):
        self.prepare(open_trades=[], orders=[order()])
        self.writes['get_closed_trades_missing_exit'].return_value = [self.trade]
        self.assertTrue(self.run.reconcile_closed_trades())
        self.assertTrue(self.writes['reconcile_sell_fill'].call_args.kwargs['decorate'])
        self.writes['insert_order_event'].assert_not_called()
        self.writes['decorate_trade_exit'].assert_not_called()

    def test_atomic_fill_exception_holds_watermark_without_independent_writes(self):
        self.prepare(orders=[order()], positions=[SimpleNamespace(symbol='XYZ', qty='2')])
        self.writes['reconcile_sell_fill'].side_effect = RuntimeError('private-db-text')
        self.assertFalse(self.run.reconcile_closed_trades())
        self.writes['close_trade'].assert_not_called()
        self.writes['insert_order_event'].assert_not_called()
        self.writes['set_reconcile_state'].assert_not_called()
        self.assertNotIn('private-db-text', self.messages.getvalue())

    def test_failed_closed_trade_read_holds_watermark(self):
        self.prepare(open_trades=[])
        self.writes['get_closed_trades_missing_exit'].side_effect = RuntimeError('read-failed')
        self.assertFalse(self.run.reconcile_closed_trades())
        self.writes['set_reconcile_state'].assert_not_called()

    def test_failed_watermark_write_is_unsuccessful(self):
        self.prepare(open_trades=[])
        self.writes['set_reconcile_state'].return_value = False
        self.assertFalse(self.run.reconcile_closed_trades())

    def test_malformed_http_envelope_is_not_an_empty_order_list(self):
        self.prepare()
        response = mock.Mock(status_code=200)
        response.json.return_value = {'error': 'not-orders'}
        with mock.patch.object(self.executor, 'load_env'), mock.patch.object(
                self.executor, 'alpaca_http_get', return_value=response), mock.patch.object(
                self.executor, '_alpaca_url', return_value='https://synthetic.invalid'):
            with self.assertRaisesRegex(ValueError, 'RECONCILE_ORDERS_INVALID'):
                self.http_orders()


class ActivityTests(unittest.TestCase):
    setUp = hourly_tests.ExecutorGateIntegrationTests.setUp

    def prepare(self):
        def module(name, **fields):
            result = ModuleType(name)
            result.__dict__.update(fields)
            return result
        pg = module('psycopg2', __path__=[])
        extras = module('psycopg2.extras', execute_batch=self.forbidden)
        extras.RealDictCursor = object
        extensions = module('psycopg2.extensions', connection=object)
        pg.extras = extras
        boundaries = {'psycopg2': pg, 'psycopg2.extras': extras,
                      'psycopg2.extensions': extensions,
                      'dotenv': module('dotenv', load_dotenv=self.forbidden)}
        self.stack.enter_context(mock.patch.dict(sys.modules, boundaries))
        self.stack.enter_context(mock.patch.object(self.executor.requests, 'Response', object, create=True))
        sys.modules.pop('scripts.fetch_account_activities', None)
        self.addCleanup(lambda: sys.modules.pop('scripts.fetch_account_activities', None))
        self.activities = importlib.import_module('scripts.fetch_account_activities')
        self.stack.enter_context(mock.patch.object(self.activities, 'datetime', FrozenRunClock))
        self.insert = self.stack.enter_context(mock.patch.object(self.activities, 'insert_activities',
                                                                return_value=(0, None, None)))
        self.watermark = self.stack.enter_context(mock.patch.object(self.activities, 'save_watermark',
                                                                   return_value=True))
        self.args = SimpleNamespace(since_ts=None, lookback_days=30, max_pages=2, page_size=2)

    def run_pages(self, pages):
        with mock.patch.object(self.activities, '_request_activity_page', side_effect=pages) as fetch:
            self.activities._run_incremental(self.args, 'https://synthetic.invalid', object(), NOW.isoformat())
        return fetch

    def test_valid_empty_activity_response_is_successful(self):
        self.prepare()
        self.run_pages([([], None)])
        self.insert.assert_called_once()
        self.watermark.assert_not_called()

    def test_future_activity_time_blocks_entire_batch_before_insertion(self):
        for field in ('transaction_time', 'processed_at', 'date', 'timestamp'):
            with self.subTest(field=field):
                self.prepare()
                record = self.activity()
                record.pop('transaction_time')
                record['id'] = 'act-future'
                record[field] = (NOW + timedelta(microseconds=1)).isoformat()
                with self.assertRaisesRegex(RuntimeError, 'ACT_PAYLOAD_INVALID'):
                    self.run_pages([([self.activity(), record], 'next'), ([], None)])
                self.insert.assert_not_called()
                self.watermark.assert_not_called()

    def test_activity_time_cutoff_offsets_and_fixed_run_clock(self):
        self.prepare()
        values = (NOW.isoformat(), (NOW - timedelta(microseconds=1)).isoformat(),
                  '2026-10-09T08:00:00-04:00')
        records = [self.activity() | {'id': str(i), 'transaction_time': value}
                   for i, value in enumerate(values)]
        self.insert.return_value = (3, values[-1], values[-1])
        with mock.patch.object(self.activities, 'fetch_activities', return_value=records), \
                mock.patch.object(self.activities, 'datetime', wraps=FrozenRunClock) as clock:
            self.activities._run_incremental(self.args, 'https://synthetic.invalid', object(), NOW.isoformat())
        clock.now.assert_called_once_with(timezone.utc)
        self.watermark.assert_called_once()

    def test_future_since_or_stored_activity_watermark_stops_before_fetch(self):
        future = (NOW + timedelta(microseconds=1)).isoformat()
        for since, saved in ((future, NOW.isoformat()), (None, future), (NOW.isoformat(), future)):
            with self.subTest(since=since, saved=saved):
                self.prepare()
                self.args.since_ts = since
                with mock.patch.object(self.activities, 'fetch_activities') as fetch, \
                        self.assertRaisesRegex(RuntimeError, 'ACT_WATERMARK_INVALID'):
                    self.activities._run_incremental(self.args, 'https://synthetic.invalid', object(), saved)
                fetch.assert_not_called()
                self.insert.assert_not_called()
                self.watermark.assert_not_called()

    def test_partial_page_failure_never_inserts_or_advances(self):
        self.prepare()
        with self.assertRaises(RuntimeError):
            self.run_pages([([self.activity()], 'next'), RuntimeError('page-failed')])
        self.insert.assert_not_called()
        self.watermark.assert_not_called()

    @staticmethod
    def activity():
        return dict(id='act-1', activity_type='FILL', transaction_time=NOW.isoformat())

    def test_page_limit_repeated_token_and_full_unpaged_response_are_incomplete(self):
        for pages in ([([self.activity()], 'next'), ([self.activity()], 'again')],
                      [([self.activity()], 'next'), ([], 'next')],
                      [([self.activity(), self.activity()], None)]):
            with self.subTest(pages=pages):
                self.prepare()
                with self.assertRaisesRegex(RuntimeError, 'ACT_PAGINATION_INCOMPLETE'):
                    self.run_pages(pages)
                self.insert.assert_not_called()
                self.watermark.assert_not_called()

    def test_complete_multi_page_batch_inserts_then_updates(self):
        self.prepare()
        self.insert.return_value = (1, NOW.isoformat(), NOW.isoformat())
        fetch = self.run_pages([([self.activity()], 'next'), ([], None)])
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(fetch.call_args.args[1]['page_token'], 'next')
        self.watermark.assert_called_once()

    def test_invalid_activity_identity_time_and_watermark_block_writes(self):
        for record in ({}, {**self.activity(), 'transaction_time': 'bad'}):
            with self.subTest(record=record):
                self.prepare()
                with self.assertRaisesRegex(RuntimeError, 'ACT_PAYLOAD_INVALID'):
                    self.run_pages([([record], None)])
                self.insert.assert_not_called()
                self.watermark.assert_not_called()
        self.args.since_ts = 'invalid'
        with self.assertRaisesRegex(RuntimeError, 'ACT_WATERMARK_INVALID'):
            self.run_pages([])

    def test_malformed_envelope_and_items_are_not_empty_data(self):
        self.prepare()
        for payload in ({}, {'activities': None}, [None], {'activities': [], 'next_page_token': True}):
            with self.subTest(payload=payload):
                response = mock.Mock(ok=True, headers={})
                response.json.return_value = payload
                with mock.patch.object(self.activities, '_build_headers', return_value={}), \
                        mock.patch.object(self.activities.requests, 'get', return_value=response), \
                        self.assertRaisesRegex(RuntimeError, 'ACT_PAYLOAD_INVALID'):
                    self.activities._request_activity_page('https://synthetic.invalid', {})

    def test_failed_watermark_write_is_unsuccessful(self):
        self.prepare()
        self.insert.return_value = (1, NOW.isoformat(), NOW.isoformat())
        self.watermark.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'ACT_WATERMARK_FAIL'):
            self.run_pages([([self.activity()], None)])

    def test_unavailable_activity_watermark_is_not_missing_initial_state(self):
        self.prepare()
        conn = mock.MagicMock()
        conn.cursor.side_effect = RuntimeError('read-failed')
        with self.assertRaisesRegex(RuntimeError, 'ACT_DB_READ_FAILED'):
            self.activities.load_watermark(conn, strict=True)

    def test_all_supplied_tokens_are_validated_before_any_write(self):
        self.prepare()
        for key in ('next_page_token', 'page_token', 'next_token'):
            for token in (0, False, [], {}, 0.0, True, '   '):
                with self.subTest(key=key, token=token):
                    payload = {'activities': [self.activity()], key: token}
                    response = mock.Mock(ok=True, headers={})
                    response.json.return_value = payload
                    with mock.patch.object(self.activities, '_build_headers', return_value={}), \
                            mock.patch.object(self.activities.requests, 'get', return_value=response), \
                            self.assertRaisesRegex(RuntimeError, 'ACT_PAYLOAD_INVALID'):
                        self.activities._run_incremental(self.args, 'https://synthetic.invalid',
                                                         object(), NOW.isoformat())
                    self.insert.assert_not_called()
                    self.watermark.assert_not_called()

    def test_token_absence_null_empty_and_literal_zero_string(self):
        self.prepare()
        for payload in ({}, {'next_page_token': None}, {'next_page_token': ''}):
            with self.subTest(payload=payload):
                self.assertIsNone(self.activities._pagination_token(SimpleNamespace(headers={}), payload))
        self.assertEqual(self.activities._pagination_token(SimpleNamespace(headers={}),
                         {'next_page_token': '0'}), '0')

    def test_valid_aliases_cannot_mask_invalid_or_conflicting_tokens(self):
        self.prepare()
        for headers, payload in (
            ({'Next-Page-Token': 'next'}, {'next_page_token': False}),
            ({}, {'next_page_token': 'next', 'page_token': []}),
            ({'Next-Page-Token': 0}, {}),
            ({'Next-Page-Token': 'next'}, {'next_page_token': 'other'}),
        ):
            with self.subTest(headers=headers, payload=payload), \
                    self.assertRaisesRegex(RuntimeError, 'ACT_PAYLOAD_INVALID'):
                self.activities._pagination_token(SimpleNamespace(headers=headers), payload)
        self.assertEqual(self.activities._pagination_token(
            SimpleNamespace(headers={'Next-Page-Token': 'next'}),
            {'next_page_token': 'next', 'page_token': 'next'}), 'next')


class HealthTests(unittest.TestCase):
    invoke = hourly_tests.StageTests.invoke

    def test_integrity_reasons_are_specific_and_sanitized(self):
        for stage in sync.STAGES[:2]:
            for marker in sync.INTEGRITY_REASONS:
                if marker not in stage.failures:
                    continue
                with self.subTest(marker=marker):
                    result = self.invoke(stage, '\n'.join((*stage.required, marker, 'private-text')))
                    self.assertEqual(result['status'], 'failed')
                    self.assertEqual(result['reason'], marker)
                    self.assertIn(marker, result['health_reasons'])
                    self.assertNotIn('private-text', json.dumps(result))

    def test_missing_coverage_marker_is_not_success(self):
        for stage in sync.STAGES[:2]:
            text = '\n'.join(x for x in stage.required if 'COVERAGE_COMPLETE' not in x)
            self.assertEqual(self.invoke(stage, text)['status'], 'failed')

    def test_specific_reason_survives_saved_workflow_readback(self):
        import tempfile
        from pathlib import Path
        stage = sync.STAGES[0]
        result = self.invoke(stage, 'ACT_PAGINATION_INCOMPLETE')
        runner = mock.Mock(return_value=result)
        with tempfile.TemporaryDirectory() as storage:
            report = sync.run_workflow(Path(storage), runner=runner)
            saved = json.loads((Path(storage) / 'latest.json').read_bytes())
            self.assertEqual(saved, report)
            self.assertEqual(saved['stages'][0]['reason'], 'ACT_PAGINATION_INCOMPLETE')
            self.assertEqual(saved['stages'][0]['health_reasons'], ['ACT_PAGINATION_INCOMPLETE'])
            self.assertEqual(saved['stages'][1]['status'], 'not_run')
            runner.assert_called_once()


class DatabaseReadTests(unittest.TestCase):
    setUp = hourly_tests.ExecutorGateIntegrationTests.setUp
    prepare = ActivityTests.prepare

    def read_module(self):
        self.prepare()
        # Normal file-module execution, with psycopg2 replaced before import.
        # No historical helper or extracted function is executed.
        spec = importlib.util.spec_from_file_location('offline_db', sync.ROOT / 'scripts/db.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.conn = mock.MagicMock()
        self.conn.autocommit = False
        self.conn.get_transaction_status.return_value = 0
        self.cursor = self.conn.cursor.return_value.__enter__.return_value
        self.original_maybe_conn = self.module._maybe_conn
        self.stack.enter_context(mock.patch.object(self.module, '_maybe_conn',
                                    side_effect=lambda unused: contextlib.nullcontext(self.conn)))

    def test_strict_read_errors_remain_errors(self):
        self.read_module()
        self.conn.cursor.side_effect = RuntimeError('synthetic-read-failed')
        for name, args in (('get_reconcile_state', (self.conn,)),
                           ('get_open_trades', (self.conn,)),
                           ('get_closed_trades_missing_exit', (self.conn, NOW))):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                getattr(self.module, name)(*args, strict=True)

    def test_missing_initial_state_and_empty_trade_lists_are_valid(self):
        self.read_module()
        self.cursor.fetchone.return_value = None
        self.cursor.fetchall.return_value = []
        self.assertEqual(self.module.get_reconcile_state(self.conn, strict=True), {})
        self.assertEqual(self.module.get_open_trades(self.conn, strict=True), [])
        self.assertEqual(self.module.get_closed_trades_missing_exit(self.conn, NOW, strict=True), [])

    def test_strict_read_rejects_bad_identity_even_after_a_valid_record(self):
        for field, values in (('trade_id', (None, True, 0, -1, '1', 1.5)),
                              ('symbol', (None, '', ' ', 3, ' XYZ', 'XYZ '))):
            for value in values:
                for name in ('get_open_trades', 'get_closed_trades_missing_exit'):
                    with self.subTest(field=field, value=value, reader=name):
                        self.read_module()
                        self.cursor.fetchall.return_value = [dict(trade_id=1, symbol='XYZ'),
                            dict(trade_id=2, symbol='ABC') | {field: value}]
                        args = (self.conn,) if name == 'get_open_trades' else (self.conn, NOW)
                        with self.assertRaisesRegex(RuntimeError, 'RECONCILE_DB_INVALID_TRADE'):
                            getattr(self.module, name)(*args, strict=True)

    def test_legacy_reads_keep_records_while_strict_valid_identity_passes(self):
        self.read_module()
        for name, args in (('get_open_trades', (self.conn,)),
                           ('get_closed_trades_missing_exit', (self.conn, NOW))):
            with self.subTest(reader=name):
                self.cursor.fetchall.return_value = [dict(trade_id=1, symbol='BRK.B')]
                self.assertEqual(getattr(self.module, name)(*args, strict=True), self.cursor.fetchall.return_value)
                self.cursor.fetchall.return_value = [dict(trade_id=None, symbol='')]
                self.assertEqual(getattr(self.module, name)(*args), self.cursor.fetchall.return_value)

    def test_saturated_database_windows_are_not_complete(self):
        self.read_module()
        self.cursor.fetchall.return_value = [{'trade_id': 1}]
        for name, args in (('get_open_trades', (self.conn,)),
                           ('get_closed_trades_missing_exit', (self.conn, NOW))):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                getattr(self.module, name)(*args, strict=True, limit=1)

    def test_invalid_saved_watermark_is_not_missing_state(self):
        self.read_module()
        self.cursor.fetchone.return_value = {'last_after': 'invalid'}
        with self.assertRaisesRegex(RuntimeError, 'RECONCILE_DB_INVALID_WATERMARK'):
            self.module.get_reconcile_state(self.conn, strict=True)


class TransactionFixture:
    """In-memory transaction model; never connects to a database.

    SQL route assertions require the real helper to acquire both locks, read
    existing events and perform its writes within one transaction. Commit
    acknowledgement loss can be injected after durable state is recorded.
    """
    autocommit = False

    def __init__(self, *, closed=False):
        self.trades = {}
        self.trade = ['CLOSED' if closed else 'OPEN', 2, 10, 'XYZ', None, None, None, None, None]
        self.events = []
        self.fail = None
        self.statements = []
        self.commits = self.rollbacks = 0
        self.rowcount = 1
        self.transaction_status = 0

    @property
    def trade(self):
        return self.trades[1]

    @trade.setter
    def trade(self, value):
        self.trades[1] = value

    def get_transaction_status(self):
        return self.transaction_status

    def __enter__(self):
        self.pending_trades = copy.deepcopy(self.trades)
        self.pending_trade = self.pending_trades[1]
        self.pending_events = copy.deepcopy(self.events)
        self.order_locked = self.trade_locked = False
        self.ownership_checked = False
        self.read_committed = False
        return self

    def __exit__(self, kind, value, trace):
        self.transaction_status = 0
        if kind is not None:
            self.rollbacks += 1
        else:
            self.trades = self.pending_trades
            self.events = self.pending_events
            self.commits += 1
            if self.fail == 'commit_ack':
                self.fail = None
                raise RuntimeError('synthetic lost commit acknowledgement')
        return False

    @contextlib.contextmanager
    def cursor(self, **unused):
        yield self

    def execute(self, sql, params=None):
        self.transaction_status = 2  # SELECT also begins a psycopg2 transaction.
        self.statements.append(sql)
        if sql == 'SET TRANSACTION ISOLATION LEVEL READ COMMITTED':
            self.read_committed = True
        elif 'pg_advisory_xact_lock' in sql:
            if not self.read_committed:
                raise AssertionError('statement snapshot contract required before lock')
            if params['key'] != 'jbravo:SELL_FILL:sell-1':
                raise AssertionError('unexpected advisory lock identity')
            self.order_locked = True
        elif 'SELECT status' in sql:
            if not self.order_locked or 'FOR UPDATE' not in sql:
                raise AssertionError('trade must be locked after broker order')
            self.trade_locked = True
            self.pending_trade = self.pending_trades[params['trade_id']]
        elif 'SELECT trade_id FROM trades' in sql:
            if not self.trade_locked or 'LIMIT 2' not in sql:
                raise AssertionError('bounded ownership lookup must follow locks')
            self.result_kind = 'owners'
            self.ownership_checked = True
            self.owners = [(key,) for key, row in self.pending_trades.items()
                           if row[4] == params['order_id']][:2]
        elif 'SELECT symbol' in sql:
            if not self.trade_locked or not self.ownership_checked or "event_type='SELL_FILL'" not in sql:
                raise AssertionError('event identity lookup must follow locks and ownership check')
            self.result_kind = 'events'
        elif 'INSERT INTO order_events' in sql:
            if self.fail == 'insert':
                raise RuntimeError('synthetic event insert failure')
            self.pending_events.append((params['symbol'], params['qty'], params['status'],
                                        params['event_time']))
        elif 'UPDATE trades' in sql:
            if self.fail == 'update':
                raise RuntimeError('synthetic trade update failure')
            self.pending_trade[0] = 'CLOSED'
            self.pending_trade[4:8] = [params['order_id'], params['event_time'],
                                      params['exit_price'], params['exit_reason']]
            self.realized_pnl = params['realized_pnl']
            self.pending_trade[8] = params['realized_pnl']
        else:
            raise AssertionError('unexpected SQL route')

    def fetchone(self):
        return tuple(self.pending_trade)

    def fetchall(self):
        if self.result_kind == 'owners':
            return self.owners
        return copy.deepcopy(self.pending_events)


class AtomicFillTests(unittest.TestCase):
    setUp = hourly_tests.ExecutorGateIntegrationTests.setUp
    prepare = ActivityTests.prepare
    read_module = DatabaseReadTests.read_module

    def setup_transaction(self, **options):
        self.read_module()
        self.transaction = TransactionFixture(**options)
        self.stack.enter_context(mock.patch.object(self.module, '_maybe_conn',
            side_effect=lambda unused: contextlib.nullcontext(self.transaction)))
        self.parameters = dict(symbol='XYZ', qty='2', order_id='sell-1', status='filled',
                               event_time=NOW, raw=order(), exit_price=11.0, exit_reason='SELL_FILL')

    def fill(self, **changes):
        return self.module.reconcile_sell_fill(self.transaction, 1, **(self.parameters | changes))

    def test_insert_failure_rolls_back_and_retry_closes_once(self):
        self.setup_transaction()
        self.transaction.fail = 'insert'
        self.assertFalse(self.fill())
        self.assertEqual((self.transaction.trade[0], self.transaction.events), ('OPEN', []))
        self.transaction.fail = None
        self.assertTrue(self.fill())
        self.assertEqual((self.transaction.trade[0], len(self.transaction.events)), ('CLOSED', 1))
        self.assertEqual(self.transaction.realized_pnl, 2.0)

    def test_invalid_trade_basis_blocks_open_partial_and_complete_exits(self):
        invalid = (None, '', 'bad', True, 0, -1, 'NaN', 'Infinity', '1e400', '1e-400')
        for state in ('open', 'partial', 'complete'):
            for field in (1, 2):
                for value in invalid:
                    with self.subTest(state=state, field=field, value=value):
                        self.setup_transaction(closed=state != 'open')
                        if state != 'open':
                            self.transaction.trade[4:8] = ['sell-1', NOW, 11, 'SELL_FILL']
                            self.transaction.events = [('XYZ', 2, 'filled', NOW)]
                        if state == 'complete':
                            self.transaction.trade[8] = 2.0
                        self.transaction.trade[field] = value
                        self.assert_fill_conflict_preserves_state(decorate=state != 'open')

    def test_realized_pnl_preserves_gain_loss_zero_and_fractional_quantity(self):
        for qty, price, expected in ((2, 11, 2), (2, 9, -2), (2, 10, 0), ('0.5', 11, 0.5)):
            with self.subTest(qty=qty, price=price):
                self.setup_transaction()
                self.transaction.trade[1] = qty
                self.assertTrue(self.fill(qty=qty, exit_price=price))
                self.assertEqual(self.transaction.trade[8], expected)
                self.assertTrue(self.fill(qty=qty, exit_price=price))
                self.assertEqual(len(self.transaction.events), 1)

    def test_nonfinite_computed_pnl_rolls_back_before_writes(self):
        self.setup_transaction()
        self.transaction.trade[1:3] = [1e308, 1]
        original = copy.deepcopy(self.transaction.trade)
        self.assertFalse(self.fill(qty=1e308, exit_price=1e308))
        self.assertEqual(self.transaction.trade, original)
        self.assertEqual(self.transaction.events, [])
        self.assertEqual(self.transaction.rollbacks, 1)
        self.assertFalse(any('INSERT INTO order_events' in sql or 'UPDATE trades' in sql
                             for sql in self.transaction.statements))

    def test_invalid_or_conflicting_saved_pnl_is_not_overwritten_or_reused(self):
        for value in (True, 'bad', 'NaN', 'Infinity', 3.0):
            for complete in (False, True):
                with self.subTest(value=value, complete=complete):
                    self.setup_transaction(closed=True)
                    self.transaction.trade[4:9] = ['sell-1', NOW, 11, 'SELL_FILL', value]
                    if not complete:
                        self.transaction.trade[5] = None
                    self.transaction.events = [('XYZ', 2, 'filled', NOW)]
                    self.assert_fill_conflict_preserves_state()

    def test_close_and_decoration_failure_roll_back_event_and_retry(self):
        for decorate in (False, True):
            with self.subTest(decorate=decorate):
                self.setup_transaction(closed=decorate)
                original = copy.deepcopy(self.transaction.trade)
                self.transaction.fail = 'update'
                self.assertFalse(self.fill(decorate=decorate))
                self.assertEqual(self.transaction.events, [])
                self.assertEqual(self.transaction.trade, original)
                self.assertEqual(self.transaction.rollbacks, 1)
                self.transaction.fail = None
                self.assertTrue(self.fill(decorate=decorate))
                self.assertTrue(self.fill(decorate=decorate))
                self.assertEqual(len(self.transaction.events), 1)
                self.assertEqual(sum('UPDATE trades' in sql for sql in self.transaction.statements), 2)

    def test_lost_commit_acknowledgement_retries_without_duplicate(self):
        self.setup_transaction()
        self.transaction.fail = 'commit_ack'
        self.assertFalse(self.fill())
        self.assertEqual((self.transaction.trade[0], len(self.transaction.events)), ('CLOSED', 1))
        self.assertTrue(self.fill())
        self.assertEqual(len(self.transaction.events), 1)
        self.assertEqual(sum('UPDATE trades' in sql for sql in self.transaction.statements), 1)

    def test_matching_prior_event_is_reused_for_incomplete_trade(self):
        self.setup_transaction()
        self.transaction.trade[4] = 'sell-1'
        self.transaction.events = [('XYZ', 2, 'filled', NOW)]
        self.assertTrue(self.fill())
        self.assertEqual(len(self.transaction.events), 1)
        self.assertFalse(any('INSERT INTO order_events' in sql for sql in self.transaction.statements))

    def test_conflicting_duplicate_events_and_trade_exit_block(self):
        self.setup_transaction()
        self.transaction.trade[4] = 'sell-1'
        for events in ([('XYZ', 3, 'filled', NOW)],
                       [('XYZ', 2, 'filled', NOW)] * 2,
                       [('OTHER', 2, 'filled', NOW)]):
            with self.subTest(events=events):
                self.transaction.events = events
                self.assertFalse(self.fill())
                self.assertEqual(self.transaction.trade[0], 'OPEN')
                self.assertEqual(self.transaction.events, events)
        self.transaction.events = []
        self.transaction.trade[4] = 'different-order'
        self.assertFalse(self.fill())
        self.assertEqual(self.transaction.events, [])

    def test_autocommit_cannot_claim_atomic_success(self):
        self.setup_transaction()
        self.transaction.autocommit = True
        self.assertFalse(self.fill())
        self.assertEqual(self.transaction.statements, [])

    def assert_fill_conflict_preserves_state(self, *, decorate=True):
        original_trade = copy.deepcopy(self.transaction.trade)
        original_events = copy.deepcopy(self.transaction.events)
        self.assertFalse(self.fill(decorate=decorate))
        self.assertEqual(self.transaction.trade, original_trade)
        self.assertEqual(self.transaction.events, original_events)
        self.assertEqual(self.transaction.commits, 0)
        self.assertEqual(self.transaction.rollbacks, 1)
        self.assertFalse(any('INSERT INTO order_events' in sql or 'UPDATE trades' in sql
                             for sql in self.transaction.statements))

    def test_partial_closed_exit_conflicts_block_each_present_field(self):
        conflicts = {4: 'other-order', 5: NOW.replace(hour=11),
                     6: 12, 7: 'TRAIL_STOP'}
        matching = ['sell-1', NOW, 11, 'SELL_FILL']
        for field, value in conflicts.items():
            for missing in (5, 6, 7):
                if missing == field:
                    continue
                with self.subTest(field=field, missing=missing):
                    self.setup_transaction(closed=True)
                    self.transaction.trade[4:8] = matching
                    self.transaction.trade[field] = value
                    self.transaction.trade[missing] = None
                    self.transaction.events = [('XYZ', 2, 'filled', NOW)]
                    self.assert_fill_conflict_preserves_state()

    def test_partial_closed_exit_matches_fill_each_missing_field_once(self):
        for missing in (4, 5, 6, 7):
            with self.subTest(missing=missing):
                self.setup_transaction(closed=True)
                self.transaction.trade[4:8] = ['sell-1', NOW.isoformat(), '11.00', 'SELL_FILL']
                self.transaction.trade[missing] = None
                self.transaction.events = [] if missing == 4 else [('XYZ', 2, 'filled', NOW)]
                self.assertTrue(self.fill(decorate=True))
                self.assertEqual(self.transaction.trade[4:9], ['sell-1', NOW, 11.0, 'SELL_FILL', 2.0])
                self.assertTrue(self.fill(decorate=True))
                self.assertEqual(len(self.transaction.events), 1)
                self.assertEqual(sum('UPDATE trades' in sql for sql in self.transaction.statements), 1)
                self.assertEqual(sum('INSERT INTO order_events' in sql
                                     for sql in self.transaction.statements), int(missing == 4))

    def test_existing_unbound_event_cannot_be_assigned_by_matching_fields(self):
        self.setup_transaction(closed=True)
        self.transaction.events = [('XYZ', 2, 'filled', NOW)]
        self.assert_fill_conflict_preserves_state()

    def test_order_cannot_be_reused_by_a_second_trade(self):
        for closed in (False, True):
            with self.subTest(closed=closed):
                self.setup_transaction(closed=closed)
                self.transaction.trades[2] = copy.deepcopy(self.transaction.trade)
                self.assertTrue(self.fill(decorate=closed))
                original = copy.deepcopy(self.transaction.trades)
                events = copy.deepcopy(self.transaction.events)
                self.assertFalse(self.module.reconcile_sell_fill(
                    self.transaction, 2, **(self.parameters | {'decorate': closed})))
                self.assertEqual(self.transaction.trades, original)
                self.assertEqual(self.transaction.events, events)
                self.assertTrue(self.fill(decorate=closed))
                self.assertEqual(len(self.transaction.events), 1)

    def test_duplicate_trade_owners_and_other_owner_without_event_block(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                self.setup_transaction(closed=True)
                other = copy.deepcopy(self.transaction.trade)
                other[4] = 'sell-1'
                self.transaction.trades[2] = other
                if duplicate:
                    self.transaction.trade[4] = 'sell-1'
                original = copy.deepcopy(self.transaction.trades)
                self.assert_fill_conflict_preserves_state()
                self.assertEqual(self.transaction.trades, original)

    def test_failed_write_does_not_leave_fill_owned(self):
        self.setup_transaction(closed=True)
        self.transaction.trades[2] = copy.deepcopy(self.transaction.trade)
        self.transaction.fail = 'update'
        self.assertFalse(self.fill(decorate=True))
        self.assertIsNone(self.transaction.trade[4])
        self.assertEqual(self.transaction.events, [])
        self.transaction.fail = None
        self.assertTrue(self.module.reconcile_sell_fill(
            self.transaction, 2, **(self.parameters | {'decorate': True})))
        self.assertEqual(self.transaction.trades[2][4], 'sell-1')
        self.assertIsNone(self.transaction.trade[4])

    def test_only_one_matching_exit_field_can_be_completed(self):
        for field, value in {4: 'sell-1', 5: NOW, 6: '11.00', 7: 'SELL_FILL'}.items():
            with self.subTest(field=field):
                self.setup_transaction(closed=True)
                self.transaction.trade[field] = value
                self.assertTrue(self.fill(decorate=True))
                self.assertEqual(self.transaction.trade[4:9], ['sell-1', NOW, 11.0, 'SELL_FILL', 2.0])
                self.assertEqual(len(self.transaction.events), 1)

    def test_invalid_present_partial_exit_fields_are_not_missing(self):
        for field, value in ((5, 'invalid'), (6, 0), (6, float('nan')),
                             (6, 'invalid'), (6, True), (7, '')):
            with self.subTest(field=field, value=value):
                self.setup_transaction(closed=True)
                self.transaction.trade[field] = value
                self.assert_fill_conflict_preserves_state()

    def test_complete_closed_exit_conflicts_still_block(self):
        for field, value in {4: 'other-order', 5: NOW.replace(hour=11),
                             6: 12, 7: 'TRAIL_STOP'}.items():
            with self.subTest(field=field):
                self.setup_transaction(closed=True)
                self.transaction.trade[4:9] = ['sell-1', NOW, 11, 'SELL_FILL', 2]
                self.transaction.trade[field] = value
                self.assert_fill_conflict_preserves_state()

    def test_existing_transaction_cannot_be_committed_by_this_helper(self):
        self.setup_transaction()
        self.transaction.transaction_status = 2
        self.assertFalse(self.fill())
        self.assertEqual(self.transaction.statements, [])
        self.assertEqual(self.transaction.commits, 0)


class SharedConnectionFixture(TransactionFixture):
    """Model read-to-write ownership on one connection, not separate mocks."""
    def __init__(self, **options):
        super().__init__(**options)
        self.last_after = NOW
        self.read_failure = None
        self.read_commit_failure = False
        self.transaction_modes = []

    def __enter__(self):
        super().__enter__()
        self.read_only = False
        return self

    def __exit__(self, kind, value, trace):
        self.transaction_modes.append('read' if self.read_only else 'write')
        if kind is None and self.read_only and self.read_commit_failure:
            self.transaction_status = 0
            self.rollbacks += 1
            raise RuntimeError('synthetic read completion failure')
        return super().__exit__(kind, value, trace)

    def execute(self, sql, params=None):
        self.transaction_status = 2
        if sql == 'SET TRANSACTION READ ONLY':
            self.statements.append(sql)
            self.read_only = True
        elif 'SELECT last_after' in sql:
            self.statements.append(sql)
            self.selected = 'watermark'
        elif "WHERE status='OPEN'" in sql:
            self.statements.append(sql)
            if self.read_failure == 'open':
                raise RuntimeError('synthetic open read failure')
            self.selected = 'open'
        elif "WHERE status='CLOSED'" in sql:
            if 'exit_time IS NULL' not in sql or 'exit_reason IS NULL' not in sql:
                raise AssertionError('missing time and reason must be queried')
            self.statements.append(sql)
            self.selected = 'missing'
        elif 'INSERT INTO reconcile_state' in sql:
            if self.read_only:
                raise AssertionError('watermark write in read transaction')
            self.statements.append(sql)
            self.last_after = params['last_after']
        else:
            if self.read_only:
                raise AssertionError('atomic write must start a separate transaction')
            self.selected = 'fill'
            super().execute(sql, params)

    def fetchone(self):
        if self.selected == 'watermark':
            return {'last_after': self.last_after, 'last_ran_at': NOW}
        return super().fetchone()

    def fetchall(self):
        if self.selected in ('open', 'missing'):
            rows = []
            for trade_id, row in self.trades.items():
                eligible = row[0] == 'OPEN' if self.selected == 'open' else (
                    row[0] == 'CLOSED' and any(row[i] is None for i in (4, 5, 6, 7, 8)))
                if eligible:
                    rows.append(dict(trade_id=trade_id, symbol=row[3], qty=row[1], entry_price=row[2],
                         entry_time=NOW, entry_order_id='entry-1', exit_order_id=row[4],
                         exit_time=row[5], exit_price=row[6], exit_reason=row[7], realized_pnl=row[8]))
            return rows
        return super().fetchall()


class SharedConnectionCallerTests(unittest.TestCase):
    setUp = hourly_tests.ExecutorGateIntegrationTests.setUp
    prepare = ActivityTests.prepare
    read_module = DatabaseReadTests.read_module

    def setup_caller(self, *, closed=False):
        self.read_module()
        self.stack.enter_context(mock.patch.object(self.executor, 'datetime', FrozenRunClock))
        self.connection = SharedConnectionFixture(closed=closed)
        # Restore the actual module's connection ownership path; get_db_conn is
        # the sole fake database boundary. No reader or writer is mocked.
        self.stack.enter_context(mock.patch.object(self.module, '_maybe_conn',
                                                  self.original_maybe_conn))
        self.stack.enter_context(mock.patch.object(self.module, 'db_enabled', return_value=True))
        self.stack.enter_context(mock.patch.object(self.module, 'get_db_conn',
                                                  return_value=self.connection))
        self.stack.enter_context(mock.patch.object(self.executor, 'db', self.module))
        self.stack.enter_context(mock.patch.object(self.executor, 'log_info'))
        self.stack.enter_context(mock.patch.object(self.executor, 'alpaca_list_orders_http',
                                                  return_value=[order()]))
        client = mock.Mock()
        client.get_all_positions.return_value = [SimpleNamespace(symbol='XYZ', qty='2')]
        client.submit_order.side_effect = self.forbidden
        client.cancel_order_by_id.side_effect = self.forbidden
        self.caller = self.executor.TradeExecutor(self.executor.ExecutorConfig(), client,
                                                 self.executor.ExecutionMetrics())

    def test_shared_connection_reads_end_before_close_and_decoration(self):
        for closed in (False, True):
            with self.subTest(closed=closed):
                self.setup_caller(closed=closed)
                self.assertTrue(self.caller.reconcile_closed_trades())
                self.assertEqual(self.connection.get_transaction_status(), 0)
                self.assertEqual(self.connection.trade[0], 'CLOSED')
                self.assertEqual(len(self.connection.events), 1)
                self.assertEqual(self.connection.transaction_modes,
                    ['read', 'read', 'read', 'write', 'write'] if closed else
                    ['read', 'read', 'write', 'read', 'write'])
                # Run again through the same caller and connection. Completed
                # trades disappear from the readers and no fill is reinserted.
                self.assertTrue(self.caller.reconcile_closed_trades())
                self.assertEqual(len(self.connection.events), 1)
                self.assertEqual(self.connection.get_transaction_status(), 0)

    def test_invalid_trade_basis_holds_caller_watermark_without_fill_writes(self):
        for closed in (False, True):
            for field, value in ((1, None), (1, 0), (2, None), (2, 'NaN')):
                with self.subTest(closed=closed, field=field, value=value):
                    self.setup_caller(closed=closed)
                    if closed:
                        self.connection.trade[4:8] = ['sell-1', NOW, 11, 'SELL_FILL']
                        self.connection.events = [('XYZ', 2, 'filled', NOW)]
                    self.connection.trade[field] = value
                    original = copy.deepcopy(self.connection.trade)
                    events = copy.deepcopy(self.connection.events)
                    watermark = self.connection.last_after
                    self.assertFalse(self.caller.reconcile_closed_trades())
                    self.assertEqual(self.connection.trade, original)
                    self.assertEqual(self.connection.events, events)
                    self.assertEqual(self.connection.last_after, watermark)
                    self.assertFalse(any('UPDATE trades' in sql or 'INSERT INTO order_events' in sql
                                         or 'INSERT INTO reconcile_state' in sql
                                         for sql in self.connection.statements))

    def test_invalid_symbol_strict_reader_blocks_shared_caller_watermark(self):
        for closed in (False, True):
            with self.subTest(closed=closed):
                self.setup_caller(closed=closed)
                self.connection.trade[3] = ''
                original = copy.deepcopy(self.connection.trade)
                watermark = self.connection.last_after
                self.assertFalse(self.caller.reconcile_closed_trades())
                self.assertEqual(self.connection.trade, original)
                self.assertEqual(self.connection.last_after, watermark)
                self.assertEqual(self.connection.events, [])
                self.assertFalse(any('UPDATE trades' in sql or 'INSERT INTO order_events' in sql
                                     or 'INSERT INTO reconcile_state' in sql
                                     for sql in self.connection.statements))

    def test_read_failure_rolls_back_without_fill_or_watermark_write(self):
        self.setup_caller()
        self.connection.read_failure = 'open'
        self.assertFalse(self.caller.reconcile_closed_trades())
        self.assertEqual(self.connection.get_transaction_status(), 0)
        self.assertEqual(self.connection.events, [])
        self.assertEqual(self.connection.trade[0], 'OPEN')
        self.assertEqual(self.connection.rollbacks, 1)
        self.assertNotIn('write', self.connection.transaction_modes)

    def test_decoration_of_missing_realized_pnl_reuses_the_existing_event(self):
        self.setup_caller(closed=True)
        self.connection.trade[4:8] = ['sell-1', NOW, 11, 'SELL_FILL']
        self.connection.events = [('XYZ', 2, 'filled', NOW)]
        self.assertTrue(self.caller.reconcile_closed_trades())
        self.assertEqual(self.connection.trade[8], 2.0)
        self.assertEqual(len(self.connection.events), 1)
        self.assertTrue(self.caller.reconcile_closed_trades())
        self.assertEqual(len(self.connection.events), 1)

    def test_read_completion_failure_stops_caller_before_writes(self):
        self.setup_caller()
        self.connection.read_commit_failure = True
        self.assertFalse(self.caller.reconcile_closed_trades())
        self.assertEqual(self.connection.events, [])
        self.assertNotIn('write', self.connection.transaction_modes)

    def test_partial_exit_conflict_holds_caller_watermark_and_state(self):
        for field, value in {5: NOW.replace(hour=11), 6: 12, 7: 'TRAIL_STOP'}.items():
            with self.subTest(field=field):
                self.setup_caller(closed=True)
                self.connection.trade[4] = 'sell-1'
                self.connection.trade[field] = value
                original = copy.deepcopy(self.connection.trade)
                previous_watermark = self.connection.last_after
                self.assertFalse(self.caller.reconcile_closed_trades())
                self.assertEqual(self.connection.trade, original)
                self.assertEqual(self.connection.events, [])
                self.assertEqual(self.connection.last_after, previous_watermark)
                self.assertEqual(self.connection.get_transaction_status(), 0)
                self.assertEqual(self.connection.rollbacks, 1)
                self.assertFalse(any('UPDATE trades' in sql or 'INSERT INTO order_events' in sql
                                     or 'INSERT INTO reconcile_state' in sql
                                     for sql in self.connection.statements))

    def test_matching_partial_exit_is_completed_through_shared_caller(self):
        self.setup_caller(closed=True)
        self.connection.trade[4:8] = ['sell-1', NOW.isoformat(), None, 'SELL_FILL']
        self.connection.events = [('XYZ', 2, 'filled', NOW)]
        self.assertTrue(self.caller.reconcile_closed_trades())
        self.assertEqual(self.connection.trade[4:9], ['sell-1', NOW, 11.0, 'SELL_FILL', 2.0])
        self.assertEqual(self.connection.get_transaction_status(), 0)
        self.assertTrue(any('INSERT INTO reconcile_state' in sql for sql in self.connection.statements))
        self.assertTrue(self.caller.reconcile_closed_trades())
        self.assertEqual(len(self.connection.events), 1)
        self.assertEqual(sum('UPDATE trades' in sql for sql in self.connection.statements), 1)

    def test_missing_only_exit_time_or_reason_is_decorated(self):
        for missing in (5, 7):
            with self.subTest(missing=missing):
                self.setup_caller(closed=True)
                self.connection.trade[4:9] = ['sell-1', NOW, 11, 'SELL_FILL', 2.0]
                self.connection.trade[missing] = None
                self.connection.events = [('XYZ', 2, 'filled', NOW)]
                self.assertTrue(self.caller.reconcile_closed_trades())
                self.assertEqual(self.connection.trade[4:9], ['sell-1', NOW, 11.0, 'SELL_FILL', 2.0])
                self.assertEqual(len(self.connection.events), 1)
                self.assertTrue(self.caller.reconcile_closed_trades())
                self.assertEqual(sum('UPDATE trades' in sql for sql in self.connection.statements), 1)

    def test_conflict_when_only_time_or_reason_is_missing_holds_watermark(self):
        for missing, conflicting, value in ((5, 6, 12), (7, 5, NOW.replace(hour=11))):
            with self.subTest(missing=missing):
                self.setup_caller(closed=True)
                self.connection.trade[4:9] = ['sell-1', NOW, 11, 'SELL_FILL', 2.0]
                self.connection.trade[missing] = None
                self.connection.trade[conflicting] = value
                original = copy.deepcopy(self.connection.trade)
                watermark = self.connection.last_after
                self.assertFalse(self.caller.reconcile_closed_trades())
                self.assertEqual(self.connection.trade, original)
                self.assertEqual(self.connection.last_after, watermark)
                self.assertEqual(self.connection.events, [])

    def test_two_missing_trades_cannot_share_latest_symbol_fill(self):
        self.setup_caller(closed=True)
        self.connection.trades[2] = copy.deepcopy(self.connection.trade)
        original_second = copy.deepcopy(self.connection.trades[2])
        watermark = self.connection.last_after
        self.assertFalse(self.caller.reconcile_closed_trades())
        self.assertEqual(self.connection.trade[4], 'sell-1')
        self.assertEqual(self.connection.trades[2], original_second)
        self.assertEqual(len(self.connection.events), 1)
        self.assertEqual(self.connection.last_after, watermark)
        self.assertFalse(self.caller.reconcile_closed_trades())
        self.assertEqual(self.connection.trades[2], original_second)
        self.assertEqual(len(self.connection.events), 1)

    def test_orphan_event_holds_caller_watermark_without_decorating(self):
        self.setup_caller(closed=True)
        self.connection.events = [('XYZ', 2, 'filled', NOW)]
        original = copy.deepcopy(self.connection.trade)
        watermark = self.connection.last_after
        self.assertFalse(self.caller.reconcile_closed_trades())
        self.assertEqual(self.connection.trade, original)
        self.assertEqual(self.connection.last_after, watermark)

    def test_strict_read_rejects_caller_owned_transaction_without_committing(self):
        self.setup_caller()
        self.connection.transaction_status = 2
        with self.assertRaisesRegex(RuntimeError, 'RECONCILE_DB_READ_FAILED'):
            self.module.get_open_trades(self.connection, strict=True)
        self.assertEqual(self.connection.get_transaction_status(), 2)
        self.assertEqual(self.connection.commits, 0)
        self.assertEqual(self.connection.rollbacks, 0)
        self.assertEqual(self.connection.statements, [])

    def test_legacy_non_strict_read_does_not_end_caller_transaction(self):
        self.setup_caller()
        self.connection.transaction_status = 2
        self.assertEqual(len(self.module.get_open_trades(self.connection)), 1)
        self.assertEqual(self.connection.get_transaction_status(), 2)
        self.assertEqual(self.connection.commits, 0)
        self.assertEqual(self.connection.transaction_modes, [])


class MigrationIndexTests(unittest.TestCase):
    setUp = hourly_tests.ExecutorGateIntegrationTests.setUp

    def prepare_migration(self):
        spec = importlib.util.spec_from_file_location('offline_migration', sync.ROOT / 'scripts/db_migrate.py')
        self.migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.migration)
        self.stack.enter_context(mock.patch.object(self.migration, '_repair_screener_candidates_schema'))
        self.stack.enter_context(mock.patch.object(self.migration, 'ensure_schema', return_value=True))

    def test_normal_upgrade_routes_nonunique_idempotent_ownership_index(self):
        self.prepare_migration()
        ddl = 'CREATE INDEX IF NOT EXISTS idx_trades_exit_order_id ON trades(exit_order_id);'
        conn = mock.MagicMock()
        conn.__enter__.return_value = conn
        cursor = conn.cursor.return_value.__enter__.return_value
        # Exercise the normal upgrade and statement execution paths, not a live DB.
        self.assertTrue(self.migration.run_upgrade(conn))
        self.assertTrue(self.migration.run_upgrade(conn))
        self.assertEqual(cursor.execute.call_args_list.count(mock.call(ddl)), 2)

    def test_ownership_index_failure_makes_upgrade_unsuccessful(self):
        self.prepare_migration()
        conn = mock.MagicMock()
        conn.__enter__.return_value = conn
        cursor = conn.cursor.return_value.__enter__.return_value
        def execute(statement):
            if 'idx_trades_exit_order_id' in statement:
                raise RuntimeError('synthetic index failure')
        cursor.execute.side_effect = execute
        self.assertFalse(self.migration.run_upgrade(conn))
        self.migration.ensure_schema.assert_not_called()


if __name__ == '__main__':
    unittest.main()
