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
try:
    import pytest
except ModuleNotFoundError:  # Keep the documented unittest runner available.
    pytest = None
from scripts import executor_health as health
from scripts import pipeline_postflight as pipeline
from scripts import signal_handoff

SOURCE = Path(__file__).resolve().parents[1]
if pytest is not None:
    pytestmark = pytest.mark.alpaca_optional


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


class AssuranceTests(Fixture):
    def test_candidate_binding_detects_price_feature_and_optional_presence_changes(self):
        original = [r | {'close': 10.0, 'features': {'atr': 2.0}, 'optional': None} for r in self.rows]
        first = self.gate(rows=original)
        for field, value in (('close', 10.01), ('features', {'atr': 2.01})):
            changed = [original[0] | {field: value}, original[1]]
            self.assertEqual(self.gate(rows=changed, prior_value_binding=first['candidate_value_binding'])['reasons'], ['candidate_values_changed'])
        missing = [dict(row) for row in original]; del missing[0]['optional']
        self.assertNotEqual(health.candidate_values(missing), health.candidate_values(original))
        self.assertEqual(health.candidate_values(original), health.candidate_values(list(reversed(original))))
        self.assertNotEqual(health.candidate_values(original, ordered=True), health.candidate_values(list(reversed(original)), ordered=True))

    def test_candidate_scalar_precision_and_unsupported_values(self):
        from decimal import Decimal
        row = self.rows[0] | {'price': Decimal('1.0000000000000001')}
        self.assertNotEqual(health.candidate_values([row]), health.candidate_values([row | {'price': Decimal('1.0000000000000002')}]))
        for value in (float('inf'), Decimal('NaN'), object()):
            with self.assertRaises(ValueError): health.candidate_values([row | {'price': value}])
        health.candidate_values([row | {'price': float('nan'), 'optional': pd.NA}])

    def test_explicit_task_attribution_never_becomes_scheduler_proof(self):
        config = SimpleNamespace(scheduler_task_id='1326478')
        record = self.receipt(config=config)
        self.assertFalse(record['task_attribution']['independently_verified'])
        self.assertEqual(health.inspect_receipt(self.root, now=self.now, expected_task_id='1326478')['status'], 'no_orders')
        self.assertEqual(health.inspect_receipt(self.root, now=self.now, expected_task_id='1322138')['reason'], 'task_attribution_missing_or_mismatched')
        self.receipt()
        self.assertEqual(health.inspect_receipt(self.root, now=self.now, expected_task_id='1326478')['status'], 'failed')

    def pending(self, *, deadline=None, observations=None, **changes):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='resting', deadline=deadline or self.now + timedelta(minutes=35))
        config = SimpleNamespace(_entry_preflight=self.gate())
        self.receipt(config=config, metrics=self.metrics | {'orders_submitted': 1},
                     observations=observations or metric._order_observations, **changes)
        return health.inspect_receipt(self.root, now=self.now)

    def test_deadline_due_cancel_plan_and_early_wait_are_read_only(self):
        inspection = self.pending(deadline=self.now + timedelta(seconds=1))
        early = health.deadline_plan(inspection, now=self.now)
        due = health.deadline_plan(inspection, now=self.now + timedelta(seconds=1))
        self.assertEqual(early['orders'][0]['proposed_action'], 'wait_until_deadline')
        self.assertEqual(due['orders'][0]['proposed_action'], 'cancel_and_confirm')
        self.assertEqual(due['broker_calls'], 0)
        self.assertFalse(due['cancellation_enforced'])
        self.assertTrue(due['orders'][0]['requires_broker_identity_side_account_and_state_check'])

    def test_acknowledged_cancel_only_plans_confirmation_and_terminal_plans_nothing(self):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='resting', deadline=self.now)
        health.observe(metric, order_id='o1', symbol='AAA', state='cancel_requested')
        inspection = self.pending(observations=metric._order_observations)
        self.assertEqual(health.deadline_plan(inspection, now=self.now)['orders'][0]['proposed_action'], 'confirm_existing_request')
        health.observe(metric, order_id='o1', symbol='AAA', state='canceled')
        self.assertEqual(health.deadline_plan(self.pending(observations=metric._order_observations), now=self.now)['orders'], [])

    def test_acknowledgement_without_deadline_plans_confirmation_only(self):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='cancel_requested')
        for omit_deadline in (False, True):
            with self.subTest(omit_deadline=omit_deadline):
                observation = dict(metric._order_observations[0])
                if omit_deadline:
                    del observation['deadline']
                inspection = self.pending(observations=[observation])
                self.assertEqual(inspection['status'], 'pending_confirmation')
                before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
                result = health.deadline_plan(inspection, now=self.now)
                self.assertEqual(result['status'], 'action_required')
                self.assertEqual(result['reasons'], [])
                self.assertEqual(result['orders'], [{
                    'order_id': 'o1', 'symbol': 'AAA', 'deadline': None,
                    'proposed_action': 'confirm_existing_request',
                    'requires_broker_identity_side_account_and_state_check': True}])
                self.assertEqual(result['broker_calls'], 0)
                self.assertFalse(result['cancellation_enforced'])
                self.assertFalse(result['operating_acceptance'])
                self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_acknowledgement_retains_known_deadline_before_due(self):
        metric = SimpleNamespace()
        deadline = self.now + timedelta(minutes=35)
        health.observe(metric, order_id='o1', symbol='AAA', state='cancel_requested', deadline=deadline)
        result = health.deadline_plan(self.pending(observations=metric._order_observations), now=self.now)
        self.assertEqual(result['status'], 'action_required')
        self.assertEqual(result['orders'][0]['proposed_action'], 'confirm_existing_request')
        self.assertEqual(result['orders'][0]['deadline'], deadline.isoformat())

    def test_acknowledgement_terminal_followup_without_deadline_plans_nothing(self):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='cancel_requested')
        health.observe(metric, order_id='o1', symbol='AAA', state='canceled')
        result = health.deadline_plan(self.pending(observations=metric._order_observations), now=self.now)
        self.assertEqual(result['status'], 'no_action_due')
        self.assertEqual(result['orders'], [])

    def test_acknowledgement_does_not_mask_another_missing_deadline(self):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='cancel_requested')
        for state in ('submitted', 'resting'):
            with self.subTest(state=state):
                other = dict(metric._order_observations[0], order_id='o2', symbol='BBB', state=state)
                self.receipt(config=SimpleNamespace(_entry_preflight=self.gate()),
                             observations=[metric._order_observations[0], other],
                             metrics=self.metrics | {'orders_submitted': 2})
                inspection = health.inspect_receipt(self.root, now=self.now)
                result = health.deadline_plan(inspection, now=self.now)
                self.assertEqual(result['status'], 'blocked')
                self.assertEqual(result['reasons'], ['missing_cancellation_deadline'])
                self.assertEqual(result['orders'], [])

    def test_acknowledgement_malformed_and_conflicting_deadlines_remain_blocked(self):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='cancel_requested', deadline=self.now)
        first = metric._order_observations[0]
        with self.assertRaises(ValueError):
            self.pending(observations=[first | {'deadline': 'not-a-timestamp'}])
        inspection = self.pending(observations=[first, first | {
            'deadline': (self.now + timedelta(seconds=1)).isoformat()}])
        result = health.deadline_plan(inspection, now=self.now)
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['reasons'], ['contradictory_order_deadline'])
        self.assertEqual(result['orders'], [])

    def test_missing_deadline_unknown_or_truncated_population_cannot_plan(self):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='submitted')
        self.assertEqual(health.deadline_plan(self.pending(observations=metric._order_observations), now=self.now)['reasons'], ['missing_cancellation_deadline'])
        self.assertEqual(health.deadline_plan(self.pending(observations_truncated=True), now=self.now)['reasons'], ['incomplete_order_population'])
        self.assertEqual(health.deadline_plan({'status': 'failed'}, now=self.now)['orders'], [])

    def test_contradictory_order_evidence_cannot_plan(self):
        metric = SimpleNamespace()
        health.observe(metric, order_id='o1', symbol='AAA', state='resting', deadline=self.now)
        original = self.pending(observations=metric._order_observations)
        first = original['record']['order_observations'][0]
        for changes, reason in (({'symbol': 'BBB'}, 'contradictory_order_identity'),
                                ({'deadline': (self.now + timedelta(seconds=1)).isoformat()}, 'contradictory_order_deadline')):
            altered = dict(original); altered['record'] = original['record'] | {'order_observations': [first, first | changes]}
            self.assertEqual(health.deadline_plan(altered, now=self.now)['reasons'], [reason])
        for state, reason in (('resting', 'terminal_state_regressed'), ('expired', 'contradictory_terminal_state')):
            altered = dict(original); altered['record'] = original['record'] | {'order_observations': [first | {'state': 'canceled'}, first | {'state': state}]}
            self.assertEqual(health.deadline_plan(altered, now=self.now)['reasons'], [reason])

    def test_forged_scheduler_attestation_is_rejected(self):
        record = self.receipt(config=SimpleNamespace(scheduler_task_id='1326478'))
        record['task_attribution']['independently_verified'] = True
        data = json.dumps(record).encode()
        for name in ('latest.json', 'run-' + record['run_id'] + '.json'):
            (self.root / 'reports/executor/premarket' / name).write_bytes(data)
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
    def publish(self, **changes):
        if not hasattr(self, 'session'):
            return super().publish(**changes)
        snapshot = signal_handoff.freeze_candidates(self.rows, self.session, run_ts=self.batch)
        record = super().publish(session=self.session, frozen_candidates=snapshot,
                                 invocation_id=self.invocation['id'], **changes)
        result = pipeline.check_health(self.root, now=self.now, require_session=True)
        if result['status'] == 'ok':
            pipeline.complete_primary(self.root, self.invocation, result)
        return record

    def setUp(self):
        super().setUp()
        calendar = [{'date': day, 'open': '09:30', 'close': '16:00'}
                    for day in ('2026-10-08', '2026-10-09')]
        self.session = signal_handoff.session_contract(calendar, now=self.start)
        self.rows = [row | {'timestamp': '2026-10-08T04:00:00Z', 'close': 10.0, 'score': 1}
                     for row in self.rows]
        self.invocation = pipeline.begin_primary(self.root, self.start)
        self.publish()
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
        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                value = self.now
                return value.astimezone(tz) if tz else value.replace(tzinfo=None)
        self.stack.enter_context(mock.patch.object(self.mod, 'datetime', FixedDateTime))
        self.stack.enter_context(mock.patch.object(self.mod, '_bootstrap_env', return_value=[]))
        self.stack.enter_context(mock.patch.object(self.mod, 'assert_alpaca_creds', return_value={}))
        self.stack.enter_context(mock.patch.object(self.mod, 'configure_logging', return_value=None))
        self.stack.enter_context(mock.patch.object(self.mod, 'send_alert', return_value=None))
        self.stack.enter_context(mock.patch.object(self.mod, '_count_db_candidates', return_value=0))
        actual = health.preflight
        self.stack.enter_context(mock.patch.object(health, 'preflight', side_effect=lambda *a, **k: actual(*a, **({'now': self.now} | k))))

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

    def test_task_id_is_delivered_by_actual_config_builder_and_invalid_value_rejected(self):
        with mock.patch.dict(os.environ, {'JBRAVO_SCHEDULER_TASK_ID': '1326478'}):
            self.assertEqual(self.mod.build_config(self.mod.parse_args([])).scheduler_task_id, '1326478')
        with mock.patch.dict(os.environ, {'JBRAVO_SCHEDULER_TASK_ID': 'not-a-task'}):
            with self.assertRaises(ValueError): self.mod.build_config(self.mod.parse_args([]))

    def test_changed_raw_or_filtered_values_block_actual_gate(self):
        config = self.mod.ExecutorConfig()
        raw = pd.DataFrame([r | {'score': 1, 'close': 10.0} for r in self.rows])
        self.assertEqual(self.mod._check_entry_preflight(config, raw, raw_batch=True)['status'], 'pass')
        self.assertEqual(self.mod._check_entry_preflight(config, raw)['status'], 'pass')
        raw.loc[0, 'close'] = 10.5
        self.assertEqual(self.mod._check_entry_preflight(config, raw)['reasons'], ['filtered_candidate_values_changed'])
        config._entry_candidate_rows[0]['close'] = 999.0
        self.assertEqual(self.mod._check_entry_preflight(config, raw)['reasons'], ['producer_candidate_values_changed'])

    def test_changed_execution_frame_blocks_buy_without_altering_sell(self):
        config = self.mod.ExecutorConfig()
        config._entry_preflight = self.gate()
        config._entry_candidate_rows = self.rows
        config._entry_execution_frame = pd.DataFrame([r | {'close': 10.0} for r in self.rows])
        config._entry_execution_binding = health.candidate_values(config._entry_execution_frame.to_dict('records'), ordered=True)
        config._entry_execution_frame.loc[0, 'close'] = 11.0
        executor = self.mod.TradeExecutor(config, SimpleNamespace(), self.mod.ExecutionMetrics())
        with mock.patch.object(executor, '_submit_order', return_value=SimpleNamespace(id='synthetic')) as submit:
            self.assertIsNone(executor.submit_with_retries(SimpleNamespace(side='buy', symbol='AAA')))
            submit.assert_not_called()
            self.assertIsNotNone(executor.submit_with_retries(SimpleNamespace(side='sell', symbol='AAA')))
            self.assertEqual(submit.call_count, 1)

    def test_real_loader_uses_one_readonly_snapshot_and_rolls_back_before_close(self):
        events = []
        connection = SimpleNamespace(
            set_session=lambda **kw: events.append(('session', kw)),
            rollback=lambda: events.append(('rollback',)), close=lambda: events.append(('close',)))
        frame = pd.DataFrame(self.rows)
        def query(day, *, connection):
            events.append(('query', connection)); return frame, self.batch
        with mock.patch.object(self.mod.db, 'db_config_preview', return_value={'enabled': True}, create=True), \
             mock.patch.object(self.mod.db, 'get_db_conn', return_value=connection, create=True) as connect, \
             mock.patch.object(self.mod, 'get_latest_screener_candidates', side_effect=query):
            loaded = self.mod.load_candidates_from_db()
        connect.assert_called_once()
        self.assertEqual(events[0], ('session', {'isolation_level': 'REPEATABLE READ', 'readonly': True, 'autocommit': False}))
        self.assertIs(events[1][1], connection)
        self.assertEqual(events[-2:], [('rollback',), ('close',)])
        self.assertEqual(loaded.attrs['executor_snapshot']['scope'], 'candidate_load_only')

    def test_query_join_fallback_keeps_caller_snapshot_open(self):
        prior = sys.modules.pop('scripts.db_queries')
        try:
            queries = importlib.import_module('scripts.db_queries')
            commands = []
            cursor = mock.MagicMock()
            cursor.__enter__.return_value = cursor
            cursor.execute.side_effect = lambda sql, *args: commands.append(sql.strip())
            cursor.fetchone.return_value = (self.batch,)
            connection = SimpleNamespace(cursor=lambda: cursor, close=mock.Mock(), rollback=mock.Mock())
            with mock.patch.object(queries, '_fetch_latest_candidate_rows', side_effect=[RuntimeError('join unavailable'), ([('AAA', 1)], ['symbol', 'score'])]):
                frame, timestamp = queries.get_latest_screener_candidates(self.now.date(), connection=connection)
            self.assertEqual(timestamp, self.batch)
            self.assertEqual(frame['symbol'].tolist(), ['AAA'])
            self.assertIn('SAVEPOINT executor_ranker_join', commands)
            self.assertIn('ROLLBACK TO SAVEPOINT executor_ranker_join', commands)
            self.assertIn('RELEASE SAVEPOINT executor_ranker_join', commands)
            connection.rollback.assert_not_called(); connection.close.assert_not_called()
        finally:
            sys.modules['scripts.db_queries'] = prior

    def test_loader_date_fallback_and_query_error_preserve_connection_ownership(self):
        cursor = mock.MagicMock(); cursor.__enter__.return_value = cursor
        cursor.fetchone.return_value = (self.now.date() - timedelta(days=1),)
        connection = SimpleNamespace(cursor=lambda: cursor, set_session=mock.Mock(), rollback=mock.Mock(), close=mock.Mock())
        with mock.patch.object(self.mod.db, 'db_config_preview', return_value={'enabled': True}, create=True), \
             mock.patch.object(self.mod.db, 'get_db_conn', return_value=connection, create=True), \
             mock.patch.object(self.mod, 'get_latest_screener_candidates', side_effect=[(pd.DataFrame(), None), (pd.DataFrame(self.rows), self.batch)]) as query:
            self.mod.load_candidates_from_db()
        self.assertEqual(query.call_count, 2)
        self.assertTrue(all(call.kwargs['connection'] is connection for call in query.call_args_list))
        cursor.execute.assert_called_once_with('SELECT MAX(run_date) FROM screener_candidates')
        connection.rollback.assert_called_once(); connection.close.assert_called_once()
        connection.rollback.reset_mock(); connection.close.reset_mock()
        with mock.patch.object(self.mod.db, 'db_config_preview', return_value={'enabled': True}, create=True), \
             mock.patch.object(self.mod.db, 'get_db_conn', return_value=connection, create=True), \
             mock.patch.object(self.mod, 'get_latest_screener_candidates', side_effect=RuntimeError('synthetic')):
            with self.assertRaises(self.mod.CandidateLoadError): self.mod.load_candidates_from_db()
        connection.rollback.assert_called_once(); connection.close.assert_called_once()

    def test_standalone_query_still_closes_its_owned_connection(self):
        prior = sys.modules.pop('scripts.db_queries')
        try:
            queries = importlib.import_module('scripts.db_queries')
            cursor = mock.MagicMock(); cursor.__enter__.return_value = cursor
            cursor.fetchone.return_value = (None,)
            connection = SimpleNamespace(cursor=lambda: cursor, close=mock.Mock())
            with mock.patch.object(queries.db, 'get_db_conn', return_value=connection, create=True):
                frame, timestamp = queries.get_latest_screener_candidates(self.now.date())
            self.assertTrue(frame.empty); self.assertIsNone(timestamp)
            connection.close.assert_called_once()
        finally:
            sys.modules['scripts.db_queries'] = prior

    def test_actual_sizing_dry_run_uses_normal_executor_with_mocked_transport(self):
        config = self.mod.ExecutorConfig(source='path', source_path=self.root / 'unused.csv',
            dry_run=True, time_window='any', allocation_pct=.01, min_order_usd=500,
            allow_bump_to_one=False, price_source='blended')
        frame = pd.DataFrame([{'symbol': 'AAA', 'close': 95.0, 'entry_price': 95.0,
            'score': 2.0, 'universe_count': 10, 'score_breakdown': '{}'}])
        metrics = self.mod.ExecutionMetrics()
        executor = self.mod.TradeExecutor(config, None, metrics)
        with mock.patch.object(executor, 'fetch_buying_power', return_value=2000), \
             mock.patch.object(executor, 'fetch_open_order_symbols', return_value=(set(), 0, 0)), \
             mock.patch.object(executor, 'evaluate_time_window', return_value=(True, 'ok', 'any')), \
             mock.patch.object(self.mod, '_fetch_prevclose_snapshot', return_value=95.0), \
             mock.patch.object(self.mod, '_fetch_prev_close_from_alpaca', return_value=95.0), \
             mock.patch.object(self.mod, '_fetch_latest_trade_from_alpaca', return_value={'price': 95.0, 'feed': 'synthetic'}), \
             mock.patch.object(self.mod, '_fetch_latest_quote_from_alpaca', return_value={'ask': 95.0, 'feed': 'synthetic'}), \
             mock.patch.object(executor, '_submit_order', side_effect=AssertionError('no_order')) as submit, \
             mock.patch.object(executor, 'log_info') as logged:
            self.assertEqual(executor.execute(frame, prefiltered=frame.to_dict('records')), 0)
        submit.assert_not_called()
        calculations = [c for c in logged.call_args_list if c.args and c.args[0] == 'DRY_RUN_ORDER']
        self.assertTrue(calculations)
        self.assertEqual(int(calculations[0].kwargs['qty']), 5)

    def test_receipt_failure_changes_only_success_outcome(self):
        for rc in (0, 1, 2):
            with self.subTest(rc=rc), mock.patch.object(self.mod, 'run_executor', return_value=rc), \
                 mock.patch.object(health, 'publish_receipt', side_effect=OSError('synthetic_recording_failure')):
                self.assertEqual(self.mod.main([]), 1 if rc == 0 else rc)

    def test_entry_gate_after_wait_and_fallback_rows(self):
        config = self.mod.ExecutorConfig(source='db', reconcile_auto=False)
        frame = pd.DataFrame(self.rows)
        clock = SimpleNamespace(timestamp=self.now, is_open=False, next_open=self.now.replace(hour=13, minute=30),
                                next_close=self.now.replace(hour=20), session='premarket')
        with mock.patch.object(self.mod.TradeExecutor, '_get_trading_clock', return_value=clock), \
             mock.patch.object(self.mod.TradeExecutor, 'load_candidates', return_value=frame), \
             mock.patch.object(self.mod.TradeExecutor, 'persist_metrics', return_value=None), \
             mock.patch.object(self.mod.TradeExecutor, '_rank_candidates', side_effect=lambda x: x), \
             mock.patch.object(self.mod.TradeExecutor, '_apply_alloc_weight_key', side_effect=lambda x: x), \
             mock.patch.object(self.mod.TradeExecutor, '_log_top_candidates'), \
             mock.patch.object(self.mod, '_wait_until_submit_at', side_effect=lambda x, **kw: self.publish()), \
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

    def test_bound_loader_queries_exact_signal_date_and_batch_without_latest_fallback(self):
        cursor = mock.MagicMock(); cursor.__enter__.return_value = cursor
        conn = SimpleNamespace(cursor=lambda: cursor, set_session=mock.Mock(), rollback=mock.Mock(), close=mock.Mock())
        gate = self.gate()
        with mock.patch.object(self.mod.db, 'db_config_preview', return_value={'enabled': True}, create=True), \
             mock.patch.object(self.mod.db, 'get_db_conn', return_value=conn, create=True), \
             mock.patch.object(self.mod, 'get_latest_screener_candidates', return_value=(pd.DataFrame(), self.batch)) as query:
            result = self.mod.load_candidates_from_db(handoff=gate)
        self.assertTrue(result.empty)
        query.assert_called_once_with(datetime(2026, 10, 8).date(), connection=conn, exact_run_ts=self.batch)
        cursor.execute.assert_not_called()
        conn.rollback.assert_called_once(); conn.close.assert_called_once()

    def test_session_policy_rejects_old_opening_and_price_overrides(self):
        for changes in ({'submit_at_ny': '07:00'}, {'price_source': 'blended'},
                        {'time_window': 'regular'}, {'extended_hours': False}):
            with self.subTest(changes=changes):
                result = self.mod._check_entry_preflight(self.mod.ExecutorConfig(**changes))
                self.assertEqual(result['reasons'], ['session_entry_policy_incompatible'])
        for changes in ({'limit_buffer_pct': float('nan')}, {'max_gap_pct': -1}, {'max_gap_pct': float('inf')}):
            with self.subTest(changes=changes):
                result = self.mod._check_entry_preflight(self.mod.ExecutorConfig(**changes))
                self.assertEqual(result['reasons'], ['invalid_signal_price_policy'])

    def test_actual_opening_wait_and_premarket_bounds_are_four_eastern(self):
        self.now = self.now.replace(hour=7, minute=59, second=58)
        calls = []
        def advance(seconds):
            calls.append(seconds)
            self.now += timedelta(seconds=seconds)
        with mock.patch.object(self.mod.time, 'sleep', side_effect=advance):
            self.mod._wait_until_submit_at('04:00', session_open=self.session['entry_open_utc'])
        self.assertEqual(sum(calls), 2)
        self.assertEqual(self.now.hour, 8)
        self.assertEqual(self.mod._premarket_bounds_strings(), ('04:00', '09:30'))
        win, allowed, _ = self.mod._resolve_time_window('auto')
        self.assertEqual((win, allowed), ('premarket', True))

    def test_direct_buy_and_chase_replacement_cannot_exceed_frozen_cap_or_expiry(self):
        for limit, expected in ((10.05, True), (10.3, True), (10.31, False), (float('nan'), False)):
            with self.subTest(limit=limit):
                config = self.mod.ExecutorConfig()
                config._entry_preflight = self.gate(); config._entry_candidate_rows = self.rows
                executor = self.mod.TradeExecutor(config, SimpleNamespace(), self.mod.ExecutionMetrics())
                with mock.patch.object(executor, '_submit_order', return_value=SimpleNamespace(id='synthetic')) as submit:
                    result = executor.submit_with_retries(SimpleNamespace(side='buy', symbol='AAA', limit_price=limit))
                self.assertEqual(result is not None, expected)
                self.assertEqual(submit.call_count, int(expected))
        self.now = self.now.replace(hour=13, minute=30)
        with mock.patch.object(executor, '_submit_order') as submit:
            self.assertIsNone(executor.submit_with_retries(SimpleNamespace(side='buy', symbol='AAA', limit_price=10)))
            submit.assert_not_called()

    def test_real_planner_uses_signal_close_without_live_price_replacement(self):
        config = self.mod.ExecutorConfig(source='db', dry_run=True, min_order_usd=500, allocation_pct=.01)
        config._entry_preflight = self.gate(); config._entry_candidate_rows = self.rows
        frame = pd.DataFrame(self.rows)
        executor = self.mod.TradeExecutor(config, None, self.mod.ExecutionMetrics())
        with mock.patch.object(executor, 'fetch_buying_power', return_value=2000), \
             mock.patch.object(executor, 'fetch_open_order_symbols', return_value=(set(), 0, 0)), \
             mock.patch.object(executor, 'evaluate_time_window', return_value=(True, 'ok', 'premarket')), \
             mock.patch.object(executor, 'resolve_limit_price', side_effect=AssertionError('no_prevclose_replacement')), \
             mock.patch.object(self.mod, '_fetch_latest_trade_from_alpaca', side_effect=AssertionError('no_live_replacement')), \
             mock.patch.object(self.mod, '_fetch_latest_quote_from_alpaca', side_effect=AssertionError('no_live_replacement')), \
             mock.patch.object(executor, '_submit_order', side_effect=AssertionError('no_order')), \
             mock.patch.object(executor, 'log_info') as logged:
            self.assertEqual(executor.execute(frame, prefiltered=frame.to_dict('records')), 0)
        plans = [call for call in logged.call_args_list if call.args and call.args[0] == 'DRY_RUN_ORDER']
        self.assertEqual(len(plans), 2)
        for call in plans:
            self.assertEqual(float(call.kwargs['limit_price']), 10.05)
            self.assertEqual(int(call.kwargs['qty']), 49)

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

    def test_actual_acknowledgement_only_receipt_reaches_confirmation_cli(self):
        metric = self.mod.ExecutionMetrics()
        client = SimpleNamespace(cancel_order_by_id=mock.Mock())
        config = self.mod.ExecutorConfig(scheduler_task_id='1326478')
        config._entry_preflight = self.gate()
        executor = self.mod.TradeExecutor(config, client, metric)
        executor.cancel_order('o1', 'AAA')
        client.cancel_order_by_id.assert_called_once_with('o1')
        self.assertIsNone(metric._order_observations[0]['deadline'])
        record = self.receipt(config=config, metrics=self.metrics | {
            'orders_submitted': 1, 'orders_canceled': 1}, observations=metric._order_observations)
        self.assertEqual(record['health'], 'pending_confirmation')
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        actual_inspect = health.inspect_receipt
        with mock.patch.object(health, 'inspect_receipt', side_effect=lambda base, **kw:
                               actual_inspect(base, now=self.now, **kw)), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(health.main(['deadlines', '--base-dir', str(self.root),
                                        '--expected-task-id', '1326478']), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'action_required')
        self.assertEqual(result['orders'][0]['proposed_action'], 'confirm_existing_request')
        self.assertIsNone(result['orders'][0]['deadline'])
        self.assertEqual(result['broker_calls'], 0)
        self.assertFalse(result['cancellation_enforced'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        client.cancel_order_by_id.assert_called_once_with('o1')

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
        self.rows = frame.to_dict('records')
        self.publish()
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
        # Python 3.13 prints a shared metavar once for these two aliases. Preserve
        # all other wording/values; this is formatting, not a relaxed CLI contract.
        def alias_format(value):
            return value.replace('--market-tz MARKET_TIMEZONE, --market-timezone MARKET_TIMEZONE',
                                 '--market-tz, --market-timezone MARKET_TIMEZONE').rstrip()
        self.assertEqual(alias_format(text), alias_format(out.getvalue()))

    def test_buy_gate_failure_produces_unsuccessful_actual_caller_outcome(self):
        config = self.mod.ExecutorConfig(source='db', reconcile_auto=False)
        frame = pd.DataFrame(self.rows)
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
