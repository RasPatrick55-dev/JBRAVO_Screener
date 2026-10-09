"""Synthetic normal executor import/caller and inert health/log checks.

External modules/environment are replaced before import. No production DB, SDK,
credentials, provider, child processes or host paths are used. Run via unittest.
"""
import builtins
import contextlib
from datetime import datetime, timedelta, timezone
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import pandas as pd
from scripts import executor_health as health
from scripts import pipeline_postflight as pipeline

SOURCE = Path(__file__).resolve().parents[1]


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for path in set(health.SOURCES + pipeline.SOURCE_FILES):
            dest = self.root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / path, dest)
        self.now = datetime(2026, 10, 9, 11, tzinfo=timezone.utc)
        self.start = self.now.replace(hour=3, minute=45)
        self.finish = self.now.replace(hour=4)
        self.batch = self.start + timedelta(minutes=5)
        self.rows = [{'symbol': 'AAA', 'run_ts_utc': self.batch.isoformat(), 'run_date': '2026-10-08'},
                     {'symbol': 'BBB', 'run_ts_utc': self.batch.isoformat(), 'run_date': '2026-10-08'}]
        self.facts = dict(started_at=self.start, finished_at=self.finish, pipeline_rc=0,
            steps=pipeline.REQUIRED_STEPS, step_rcs={s: 0 for s in pipeline.REQUIRED_STEPS},
            stage_times={s: 1.0 for s in pipeline.REQUIRED_STEPS}, degraded=False, labels_rows=5,
            freshness={'stale': False, 'pred_compatible': True, 'predict_rc': None,
                       'snapshot_date': '2026-10-08'},
            coverage={'total': 2, 'non_null': 2, 'pct': 100.0, 'run_ts_utc': self.batch.isoformat()},
            ml_health={'decision': 'allow', 'reasons': []},
            controls={k: True for k in ('strict_predictions_meta', 'strict_auto_refresh_predictions',
                'auto_refresh_features', 'auto_refresh_predictions', 'use_champion', 'ml_health_guard',
                'refresh_predictions_for_candidates')})
        self.publish()
        self.metrics = {k: 0 for k in ('symbols_in', 'orders_submitted', 'orders_filled',
                                      'orders_canceled', 'trailing_attached', 'api_failures')}
        self.metrics.update(status='ok', skips={}, in_window=True)
        for name in ('socket.socket.connect', 'socket.create_connection', 'subprocess.Popen',
                     'subprocess.run', 'os.system'):
            p = mock.patch(name, side_effect=AssertionError('external_effect_forbidden'))
            p.start(); self.addCleanup(p.stop)

    def publish(self, **changes):
        return pipeline.publish_health(self.root, **(self.facts | changes))

    def gate(self, **changes):
        return health.preflight(self.root, **({'now': self.now, 'rows': self.rows} | changes))

    def receipt(self, **changes):
        return health.publish_receipt(self.root, **(dict(role='premarket', started_at=self.now,
             finished_at=self.now, rc=0, metrics=self.metrics) | changes))


class GateTests(Fixture):
    def test_overnight_primary_and_previous_session_labels_are_bound(self):
        result = self.gate()
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['candidate_count'], 2)
        self.assertFalse(result['research_qualified'])

    def test_old_day_and_expired_age_fail(self):
        for now in (self.now + timedelta(days=1), self.start + timedelta(hours=13)):
            with self.subTest(now=now):
                self.assertEqual(self.gate(now=now)['status'], 'blocked')

    def test_missing_report(self):
        (self.root / 'reports/pipeline_postflight/latest.json').unlink()
        self.assertEqual(self.gate()['status'], 'blocked')

    def test_failed_and_degraded_primary_fail(self):
        for change in ({'pipeline_rc': 1}, {'degraded': True}, {'labels_rows': 0},
                       {'ml_health': {'decision': 'warn', 'reasons': ['stale_monitor']}},
                       {'step_rcs': self.facts['step_rcs'] | {'screener': 1}}):
            with self.subTest(change=change):
                self.publish(**change)
                self.assertEqual(self.gate()['status'], 'blocked')

    def test_malformed_json_and_duplicate_keys(self):
        path = self.root / 'reports/pipeline_postflight/latest.json'
        for raw in (b'{', b'{"schema_version":1,"schema_version":1}', b'{"v":NaN}'):
            path.write_bytes(raw)
            self.assertEqual(self.gate()['status'], 'blocked')

    def test_source_and_history_binding_failures(self):
        (self.root / pipeline.SOURCE_FILES[0]).write_text('changed')
        self.assertEqual(self.gate()['status'], 'blocked')
        shutil.copyfile(SOURCE / pipeline.SOURCE_FILES[0], self.root / pipeline.SOURCE_FILES[0])
        record = self.publish()
        (self.root / 'reports/pipeline_postflight' / ('run-' + record['run_id'] + '.json')).write_text('{}')
        self.assertEqual(self.gate()['status'], 'blocked')

    def test_fallback_batch_cannot_pass_by_matching_count(self):
        stale = [row | {'run_ts_utc': (self.batch - timedelta(days=1)).isoformat()} for row in self.rows]
        self.assertEqual(self.gate(rows=stale)['reasons'], ['candidate_batch_mismatch'])

    def test_missing_null_naive_and_mixed_batch(self):
        for value in (None, '', '2026-10-09T03:50:00', 'NaT', (self.batch + timedelta(seconds=1)).isoformat()):
            with self.subTest(value=value):
                self.assertEqual(self.gate(rows=[self.rows[0], self.rows[1] | {'run_ts_utc': value}])['status'], 'blocked')

    def test_population_count_and_duplicates(self):
        for rows in ([], self.rows[:1], self.rows + self.rows, [self.rows[0], self.rows[0]]):
            self.assertEqual(self.gate(rows=rows)['status'], 'blocked')

    def test_changed_primary_after_wait_blocks(self):
        old = self.gate()['pipeline_binding']
        self.publish()
        self.assertEqual(self.gate(prior_binding=old)['reasons'], ['pipeline_changed_after_candidate_load'])

    def test_timestamp_and_row_order_representation(self):
        rows = [r | {'run_ts_utc': self.batch} for r in reversed(self.rows)]
        self.assertEqual(self.gate(rows=rows)['candidate_binding'], self.gate()['candidate_binding'])

    def test_no_write_during_gate(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.gate()
        self.assertEqual(before, {p: p.read_bytes() for p in before})


class ReceiptTests(Fixture):
    def test_reconcile_receipt_cannot_replace_premarket(self):
        entry = self.receipt()
        reconcile = self.receipt(role='reconciliation')
        self.assertNotEqual(entry['run_id'], reconcile['run_id'])
        result = health.inspect_receipt(self.root, now=self.now)
        self.assertEqual(result['record']['run_id'], entry['run_id'])
        self.assertEqual(health.inspect_receipt(self.root, role='reconciliation', now=self.now)['record']['run_id'], reconcile['run_id'])

    def test_immutable_history_not_shared_metrics(self):
        r = self.receipt()
        (self.root / 'data').mkdir()
        (self.root / 'data/execute_metrics.json').write_text('{"exit_reason":"RECONCILE_ONLY"}')
        self.assertEqual(health.inspect_receipt(self.root, now=self.now)['record']['run_id'], r['run_id'])

    def test_requested_is_pending_until_actual_terminal_observation(self):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='cancel_requested')
        self.assertEqual(self.receipt(observations=metric._order_observations)['health'], 'pending_confirmation')
        health.observe(metric, order_id='o1', symbol='AAA', state='canceled')
        r = self.receipt(observations=metric._order_observations)
        self.assertEqual(r['pending_terminal_confirmations'], 0)
        self.assertTrue(r['order_observations'][-1]['terminal_confirmed'])

    def test_resting_order_and_legacy_counter_cannot_claim_confirmed_cancel(self):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='resting', deadline=self.now + timedelta(minutes=35))
        r = self.receipt(metrics=self.metrics | {'orders_canceled': 1}, observations=metric._order_observations)
        self.assertEqual(r['health'], 'pending_confirmation')
        self.assertEqual(r['cancellation_policy'], 'minutes_from_submission')

    def test_failed_cancel_and_process_error(self):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='cancel_failed')
        self.assertEqual(self.receipt(observations=metric._order_observations)['health'], 'failed')
        self.assertEqual(self.receipt(rc=2)['health'], 'failed')

    def test_missing_counts_and_error_status_are_not_green(self):
        for m in ({}, self.metrics | {'status': 'error'}, self.metrics | {'orders_submitted': True}):
            self.assertEqual(self.receipt(metrics=m)['health'], 'failed')

    def test_unknown_truncated_observations_fail_green(self):
        self.assertEqual(self.receipt(observations_truncated=True)['health'], 'pending_confirmation')

    def test_submitted_without_observations_is_not_green(self):
        config = SimpleNamespace(_entry_preflight=self.gate())
        r = self.receipt(config=config, metrics=self.metrics | {'orders_submitted': 1})
        self.assertEqual((r['health'], r['unobserved_submissions']), ('pending_confirmation', 1))
        self.assertEqual(self.receipt(metrics=self.metrics | {'orders_submitted': 1})['health'], 'failed')

    def test_inspector_recomputes_outcome_and_rejects_inconsistent_records(self):
        r = self.receipt(rc=1)
        for mutation in ({'health': 'ok'}, {'schema_version': True}, {'observations_truncated': 'false'},
                         {'research_qualified': True}, {'pending_terminal_confirmations': 1}):
            data = json.dumps(r | mutation).encode()
            for path in ('latest.json', 'run-' + r['run_id'] + '.json'):
                (self.root / 'reports/executor/premarket' / path).write_bytes(data)
            self.assertEqual(health.inspect_receipt(self.root, now=self.now)['status'], 'failed')

    def test_bad_history_sources_and_age(self):
        r = self.receipt()
        self.assertEqual(health.inspect_receipt(self.root, now=self.now + timedelta(days=1))['status'], 'failed')
        hist = self.root / 'reports/executor/premarket' / ('run-' + r['run_id'] + '.json')
        hist.write_text('{}')
        self.assertEqual(health.inspect_receipt(self.root, now=self.now)['status'], 'failed')
        self.receipt()
        (self.root / health.SOURCES[0]).write_text('changed')
        self.assertEqual(health.inspect_receipt(self.root, now=self.now)['status'], 'failed')


class LogTests(Fixture):
    def test_large_log_seeks_bounded_tail_and_sanitizes(self):
        path = self.root / 'logs/execute_task.log'
        path.parent.mkdir()
        lines = ['[2026-10-09 11:00:00] EXEC_START dry_run=False candidates=2'] * 20000
        lines += ['[2026-10-09 11:01:00] EXEC_START secret=DO_NOT_EMIT candidates=999',
                  '[2026-10-09 11:02:00] ORDER_RESTING until_utc=2026-10-09T11:35:00+00:00',
                  '[2026-10-09 11:03:00] EXECUTE_SUMMARY orders_submitted=2 orders_filled=1',
                  '[2026-10-09 11:04:00] EXEC_START candidates=unrestricted_private_text']
        path.write_text('\n'.join(lines) + '\n')
        before = path.read_bytes()
        result = health.log_tail(self.root)
        self.assertLessEqual(result['tail_binding']['bytes'], 65536)
        self.assertFalse(result['complete_log'])
        self.assertTrue(result['sample_truncated'])
        rendered = json.dumps(result)
        for forbidden in ('DO_NOT_EMIT', 'unrestricted_private_text', '999'):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(result['task_success_established'])

    def test_invalid_limits_and_missing_logs(self):
        for limit in (0, 65537, True):
            with self.assertRaises(ValueError): health.log_tail(self.root, max_bytes=limit)
        with self.assertRaises(OSError): health.log_tail(self.root)

    def test_logs_cli_is_read_only_and_bounded(self):
        path = self.root / 'logs/execute_task.log';path.parent.mkdir()
        path.write_text('EXECUTE_SUMMARY orders_submitted=0 orders_filled=0\n')
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(health.main(['logs', '--base-dir', str(self.root)]), 0)
        self.assertFalse(json.loads(output.getvalue())['raw_lines_returned'])


class CallerTests(Fixture):
    def setUp(self):
        super().setUp()
        self.stack = contextlib.ExitStack(); self.addCleanup(self.stack.close)
        self.old_cwd = Path.cwd(); os.chdir(self.root); self.addCleanup(os.chdir, self.old_cwd)
        def forbidden(*a, **k): raise AssertionError('external_boundary_forbidden')
        db = ModuleType('scripts.db'); db.db_enabled = lambda: False
        queries = ModuleType('scripts.db_queries'); queries.get_latest_screener_candidates = forbidden
        utils = ModuleType('utils'); utils.__path__ = [str(SOURCE / 'utils')]; utils.write_csv_atomic = forbidden
        sutils = ModuleType('scripts.utils'); sutils.__path__ = [str(SOURCE / 'scripts/utils')]
        env = ModuleType('scripts.utils.env'); env.load_env = forbidden
        alerts = ModuleType('utils.alerts'); alerts.send_alert = lambda *a, **k: None
        telemetry = ModuleType('utils.telemetry'); telemetry.log_event = lambda *a, **k: None
        requests = ModuleType('requests'); requests.HTTPError = type('HTTPError', (Exception,), {})
        requests.get = requests.post = requests.request = forbidden
        self.stack.enter_context(mock.patch.dict(sys.modules, {'scripts.db': db, 'scripts.db_queries': queries,
            'scripts.utils': sutils, 'scripts.utils.env': env, 'utils': utils, 'utils.alerts': alerts,
            'utils.telemetry': telemetry, 'requests': requests, 'alpaca': None}))
        self.stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
        sys.modules.pop('scripts.execute_trades', None)
        self.mod = importlib.import_module('scripts.execute_trades')
        self.addCleanup(sys.modules.pop, 'scripts.execute_trades', None)
        self.mod.METRICS_PATH = self.root / 'data/execute_metrics.json'
        self.mod._EXECUTE_START_UTC = None
        self.stack.enter_context(mock.patch.object(self.mod, '_bootstrap_env', return_value=[]))
        self.stack.enter_context(mock.patch.object(self.mod, 'assert_alpaca_creds', return_value={}))
        self.stack.enter_context(mock.patch.object(self.mod, 'configure_logging', return_value=None))
        self.stack.enter_context(mock.patch.object(self.mod, 'send_alert', return_value=None))
        self.stack.enter_context(mock.patch.object(self.mod, '_count_db_candidates', return_value=0))
        actual = health.preflight
        self.stack.enter_context(mock.patch.object(health, 'preflight', side_effect=lambda *a, **k: actual(*a, now=self.now, **k)))

    def test_real_cli_role_separation_and_recording(self):
        def run(config, **kwargs):
            self.mod.write_execute_metrics(self.metrics | {'exit_reason': 'RECONCILE_ONLY' if config.reconcile_only else None})
            return 0
        with mock.patch.object(self.mod, 'run_executor', side_effect=run):
            self.assertEqual(self.mod.main(['--source', 'db']), 0)
            entry = json.loads((self.root / 'reports/executor/premarket/latest.json').read_bytes())
            self.assertEqual(self.mod.main(['--reconcile-only', 'true']), 0)
        self.assertEqual(json.loads((self.root / 'reports/executor/premarket/latest.json').read_bytes())['run_id'], entry['run_id'])
        self.assertEqual(json.loads((self.root / 'reports/executor/reconciliation/latest.json').read_bytes())['role'], 'reconciliation')

    def test_receipt_failure_changes_only_success_outcome(self):
        for rc in (0, 1, 2):
            with self.subTest(rc=rc), mock.patch.object(self.mod, 'run_executor', return_value=rc), \
                 mock.patch.object(health, 'publish_receipt', side_effect=OSError('synthetic_recording_failure')):
                self.assertEqual(self.mod.main([]), 1 if rc == 0 else rc)

    def test_entry_gate_after_wait_and_fallback_rows(self):
        config = self.mod.ExecutorConfig(source='db', reconcile_auto=False)
        frame = pd.DataFrame([r | {'score': 1, 'close': 10} for r in self.rows])
        clock = SimpleNamespace(timestamp=self.now, is_open=False, next_open=self.now.replace(hour=13, minute=30),
                                next_close=self.now.replace(hour=20), session='premarket')
        with mock.patch.object(self.mod.TradeExecutor, '_get_trading_clock', return_value=clock), \
             mock.patch.object(self.mod.TradeExecutor, 'load_candidates', return_value=frame), \
             mock.patch.object(self.mod.TradeExecutor, 'persist_metrics', return_value=None), \
             mock.patch.object(self.mod.TradeExecutor, '_rank_candidates', side_effect=lambda x: x), \
             mock.patch.object(self.mod.TradeExecutor, '_apply_alloc_weight_key', side_effect=lambda x: x), \
             mock.patch.object(self.mod.TradeExecutor, '_log_top_candidates'), \
             mock.patch.object(self.mod, '_wait_until_submit_at', side_effect=lambda x: self.publish()), \
             mock.patch.object(self.mod, '_create_trading_client', side_effect=AssertionError('no_trading_client')):
            self.assertEqual(self.mod.run_executor(config), 1)
        self.assertEqual(config._entry_preflight['reasons'], ['pipeline_changed_after_candidate_load'])

    def test_direct_buy_cannot_bypass_preflight_and_sell_is_unchanged(self):
        config = self.mod.ExecutorConfig()
        executor = self.mod.TradeExecutor(config, SimpleNamespace(), self.mod.ExecutionMetrics())
        with mock.patch.object(executor, '_submit_order', return_value=SimpleNamespace(id='synthetic')) as submit:
            self.assertIsNone(executor.submit_with_retries(SimpleNamespace(side='buy')))
            submit.assert_not_called()
            self.assertIsNotNone(executor.submit_with_retries(SimpleNamespace(side='sell')))
            self.assertEqual(submit.call_count, 1)

    def test_bound_buy_and_nonzero_parameters_unchanged(self):
        config = self.mod.build_config(self.mod.parse_args(['--allocation-pct', '0.06', '--min-order-usd', '300',
             '--max-new-positions', '4', '--trailing-percent', '3', '--cancel-after-min', '35']))
        self.assertEqual((config.allocation_pct, config.min_order_usd, config.max_new_positions,
                          config.trailing_percent, config.cancel_after_min, config.poll_detach_secs), (.06, 300, 4, 3, 35, 60))
        config._entry_preflight = health.preflight(self.root, rows=self.rows)
        config._entry_candidate_rows = self.rows
        executor = self.mod.TradeExecutor(config, SimpleNamespace(), self.mod.ExecutionMetrics())
        with mock.patch.object(executor, '_submit_order', return_value=SimpleNamespace(id='synthetic')) as submit:
            request = SimpleNamespace(side='buy', symbol='AAA', limit_price=10)
            self.assertIsNotNone(executor.submit_with_retries(request))
            submit.assert_called_once_with(request)

    def test_cancel_acknowledgment_and_failure_are_not_terminal_confirmation(self):
        metric = self.mod.ExecutionMetrics()
        client = SimpleNamespace(cancel_order_by_id=mock.Mock())
        executor = self.mod.TradeExecutor(self.mod.ExecutorConfig(), client, metric)
        executor.cancel_order('o1', 'AAA')
        self.assertEqual(metric._order_observations[-1]['state'], 'cancel_requested')
        self.assertFalse(metric._order_observations[-1]['terminal_confirmed'])
        client.cancel_order_by_id.side_effect = RuntimeError('synthetic')
        executor.cancel_order('o1', 'AAA')
        self.assertEqual(metric._order_observations[-1]['state'], 'cancel_failed')

    def test_actual_poll_detachment_is_pending_without_cancellation(self):
        metric = self.mod.ExecutionMetrics()
        client = SimpleNamespace(get_order_by_id=mock.Mock(), cancel_order_by_id=mock.Mock())
        config = self.mod.ExecutorConfig(cancel_after_min=35, poll_detach_secs=60)
        executor = self.mod.TradeExecutor(config, client, metric)
        state = self.mod.SubmittedOrderState('o1', 'AAA', 10, 10, self.now, 0, 'synthetic')
        with mock.patch.object(self.mod.time, 'time', return_value=61) as clock:
            clock.side_effect = [0, 61, 61]
            executor.poll_submitted_orders([state])
        client.get_order_by_id.assert_not_called()
        client.cancel_order_by_id.assert_not_called()
        item = metric._order_observations[-1]
        self.assertEqual(item['state'], 'resting')
        self.assertEqual(health.utc(item['deadline']), self.now + timedelta(minutes=35))
        self.assertFalse(item['terminal_confirmed'])

    def test_actual_poll_terminal_broker_state_confirms_without_new_requests(self):
        for status in ('canceled', 'expired', 'rejected', 'filled'):
            with self.subTest(status=status):
                metric = self.mod.ExecutionMetrics()
                qty = 10 if status == 'filled' else 0
                client = SimpleNamespace(get_order_by_id=mock.Mock(return_value=SimpleNamespace(
                    status=status, filled_qty=qty, filled_avg_price=10)), cancel_order_by_id=mock.Mock())
                executor = self.mod.TradeExecutor(self.mod.ExecutorConfig(), client, metric)
                state = self.mod.SubmittedOrderState('o1', 'AAA', 10, 10, self.now, 0, 'synthetic')
                with mock.patch.object(self.mod.time, 'time', return_value=0), \
                     mock.patch.object(executor, '_record_buy_execution'), mock.patch.object(executor, 'attach_trailing_stop'):
                    executor.poll_submitted_orders([state])
                client.get_order_by_id.assert_called_once_with('o1')
                client.cancel_order_by_id.assert_not_called()
                self.assertEqual(metric._order_observations[-1]['state'], status)
                self.assertTrue(metric._order_observations[-1]['terminal_confirmed'])

    def test_actual_loader_preserves_batch_binding_across_model_filter(self):
        config = self.mod.ExecutorConfig(source='db', min_model_score=.5)
        executor = self.mod.TradeExecutor(config, None, self.mod.ExecutionMetrics())
        frame = pd.DataFrame([r | {'score': 1, 'close': 10, 'model_score': score}
                              for r, score in zip(self.rows, (.8, .2))])
        with mock.patch.object(self.mod, 'load_candidates_from_db', return_value=frame):
            loaded = executor.load_candidates(rank=False)
        self.assertEqual(list(loaded['symbol']), ['AAA'])
        result = self.mod._check_entry_preflight(config, loaded)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['candidate_count'], 2)
        self.assertEqual(config._entry_allowed_symbols, {'AAA'})
        with mock.patch.object(executor, '_submit_order') as submit:
            self.assertIsNone(executor.submit_with_retries(SimpleNamespace(side='buy', symbol='BBB')))
            submit.assert_not_called()

    def test_actual_loader_rejects_old_fallback_before_ranking(self):
        config = self.mod.ExecutorConfig(source='db')
        executor = self.mod.TradeExecutor(config, None, self.mod.ExecutionMetrics())
        stale = pd.DataFrame([r | {'run_ts_utc': (self.batch - timedelta(days=1)).isoformat()} for r in self.rows])
        with mock.patch.object(self.mod, 'load_candidates_from_db', return_value=stale), \
             mock.patch.object(executor, '_rank_candidates') as rank:
            with self.assertRaises(self.mod.CandidateLoadError): executor.load_candidates()
            rank.assert_not_called()
        self.assertEqual(config._entry_preflight['reasons'], ['candidate_batch_mismatch'])

    def test_missing_primary_blocks_before_candidate_load(self):
        (self.root / 'reports/pipeline_postflight/latest.json').unlink()
        clock = SimpleNamespace(timestamp=self.now, is_open=False,
                                next_open=self.now.replace(hour=13, minute=30), next_close=self.now.replace(hour=20))
        with mock.patch.object(self.mod.TradeExecutor, '_get_trading_clock', return_value=clock), \
             mock.patch.object(self.mod.TradeExecutor, 'load_candidates') as load, \
             mock.patch.object(self.mod.TradeExecutor, 'persist_metrics'):
            self.assertEqual(self.mod.run_executor(self.mod.ExecutorConfig(reconcile_auto=False)), 1)
        load.assert_not_called()

    def test_enum_buy_cannot_skip_gate_and_unlisted_symbol_is_blocked(self):
        from enum import Enum
        class Side(str, Enum): BUY = 'buy'
        config = self.mod.ExecutorConfig()
        config._entry_preflight = health.preflight(self.root, rows=self.rows)
        config._entry_candidate_rows = self.rows
        executor = self.mod.TradeExecutor(config, SimpleNamespace(), self.mod.ExecutionMetrics())
        with mock.patch.object(executor, '_submit_order') as submit:
            self.assertIsNone(executor.submit_with_retries(SimpleNamespace(side=Side.BUY, symbol='ZZZ')))
            submit.assert_not_called()

    def test_error_receipt_does_not_inherit_old_shared_metrics(self):
        self.mod.METRICS_PATH.parent.mkdir(exist_ok=True)
        self.mod.METRICS_PATH.write_text(json.dumps(self.metrics | {'orders_submitted': 99, 'orders_filled': 99}))
        with mock.patch.object(self.mod, 'run_executor', side_effect=RuntimeError('synthetic')):
            self.assertEqual(self.mod.main([]), 1)
        r = json.loads((self.root / 'reports/executor/premarket/latest.json').read_bytes())
        self.assertNotEqual(r['metrics']['orders_submitted'], 99)
        self.assertEqual(r['health'], 'failed')

    def test_persist_metrics_does_not_inherit_previous_run_skips(self):
        self.mod._EXECUTE_METRICS_PAYLOAD = None
        self.mod.METRICS_PATH.parent.mkdir(exist_ok=True)
        self.mod.METRICS_PATH.write_text(json.dumps(self.metrics | {'skips': {'OLD_UNKNOWN_SKIP': 8}}))
        executor = self.mod.TradeExecutor(self.mod.ExecutorConfig(), None, self.mod.ExecutionMetrics())
        executor.persist_metrics()
        self.assertNotIn('OLD_UNKNOWN_SKIP', self.mod._EXECUTE_METRICS_PAYLOAD['skips'])

    def test_cli_reference_matches_actual_normal_import_help(self):
        with mock.patch.object(sys, 'argv', ['execute_trades.py']), contextlib.redirect_stdout(io.StringIO()) as out:
            self.mod.main(['--help'])
        docs = (SOURCE / 'docs/reference/cli_reference.md').read_text(encoding='utf-8')
        section = docs.split('## `python -m scripts.execute_trades --help`', 1)[1]
        text = section.split('```text\n', 1)[1].split('```', 1)[0]
        self.assertEqual(text.rstrip(), out.getvalue().rstrip())

    def test_buy_gate_failure_produces_unsuccessful_actual_caller_outcome(self):
        config = self.mod.ExecutorConfig(source='db', reconcile_auto=False, submit_at_ny='')
        frame = pd.DataFrame([r | {'score': 1, 'close': 10} for r in self.rows])
        clock = SimpleNamespace(timestamp=self.now, is_open=False, next_open=self.now.replace(hour=13, minute=30),
                                next_close=self.now.replace(hour=20))
        def execute(executor, *args, **kwargs):
            # Simulate an inconsistent selection immediately before its BUY. This
            # runs the real submission gate and must not inherit execute's legacy 0.
            executor.submit_with_retries(SimpleNamespace(side='buy', symbol='ZZZ'))
            return 0
        with mock.patch.object(self.mod.TradeExecutor, '_get_trading_clock', return_value=clock), \
             mock.patch.object(self.mod.TradeExecutor, 'load_candidates', return_value=frame), \
             mock.patch.object(self.mod.TradeExecutor, '_rank_candidates', side_effect=lambda x: x), \
             mock.patch.object(self.mod.TradeExecutor, '_apply_alloc_weight_key', side_effect=lambda x: x), \
             mock.patch.object(self.mod.TradeExecutor, '_log_top_candidates'), \
             mock.patch.object(self.mod.TradeExecutor, 'hydrate_candidates', return_value=[]), \
             mock.patch.object(self.mod.TradeExecutor, 'guard_candidates', return_value=[]), \
             mock.patch.object(self.mod.TradeExecutor, 'execute', autospec=True, side_effect=execute), \
             mock.patch.object(self.mod.TradeExecutor, '_submit_order', side_effect=AssertionError('no_order')), \
             mock.patch.object(self.mod, '_resolve_time_window', return_value=('premarket', True, self.now)), \
             mock.patch.object(self.mod, '_paper_only_guard'):
            self.assertEqual(self.mod.run_executor(config, client=SimpleNamespace()), 1)
        self.assertEqual(config._entry_preflight['reasons'], ['submission_symbol_not_in_bound_batch'])

    def test_reconciliation_and_path_dry_run_keep_no_entry_gate(self):
        for config in (self.mod.ExecutorConfig(reconcile_only=True), self.mod.ExecutorConfig(diagnostic=True),
                       self.mod.ExecutorConfig(source='path', dry_run=True)):
            self.assertEqual(self.mod._check_entry_preflight(config)['status'], 'not_applicable_no_entry_orders')

    def test_main_module_normal_import_and_help_documentation(self):
        self.assertEqual(Path(self.mod.__file__).resolve(), SOURCE / 'scripts/execute_trades.py')
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(self.mod.main(['--help']), 0)
        self.assertIn('Minutes from order submission', out.getvalue())


if __name__ == '__main__':
    unittest.main()
