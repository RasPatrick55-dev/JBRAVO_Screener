"""Normal-import offline reconciliation tests; all external effects are doubles.

Run with unittest (repository pytest conftest imports the external Alpaca SDK).
The executor uses the same pre-import isolation as the hourly gate tests.
"""
import contextlib
from datetime import datetime, timezone
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


def order(**changes):
    return dict(id='sell-1', symbol='XYZ', side='sell', status='filled',
                filled_at=NOW.isoformat(), updated_at=NOW.isoformat(),
                filled_qty='2', filled_avg_price='11', type='market') | changes


class ReconciliationTests(unittest.TestCase):
    setUp = hourly_tests.ExecutorGateIntegrationTests.setUp

    def prepare(self, *, open_trades=None, missing=None, orders=None, positions=None):
        module = self.executor
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
                decorate_trade_exit=True).items():
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
        for name in ('close_trade', 'insert_order_event', 'decorate_trade_exit', 'set_reconcile_state'):
            self.writes[name].assert_not_called()
        self.client.submit_order.assert_not_called()
        self.client.cancel_order_by_id.assert_not_called()

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
        args = self.writes['close_trade'].call_args.args
        self.assertEqual(args[1:3], (1, 'sell-1'))
        self.assertEqual(args[4:], (11.0, 'SELL_FILL'))
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

    def test_failed_event_or_decoration_holds_watermark(self):
        for failing in ('insert_order_event', 'decorate_trade_exit'):
            with self.subTest(failing=failing):
                self.prepare(open_trades=[], orders=[order()])
                self.writes['get_closed_trades_missing_exit'].return_value = [self.trade]
                self.writes[failing].return_value = False
                self.assertFalse(self.run.reconcile_closed_trades())
                self.writes['set_reconcile_state'].assert_not_called()

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
        self.cursor = self.conn.cursor.return_value.__enter__.return_value
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


if __name__ == '__main__':
    unittest.main()
