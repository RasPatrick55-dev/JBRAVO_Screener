"""Offline dashboard provenance regressions. No application/config/DB imports.

Run with python -m unittest discover -s tests -p test_dashboard_status_provenance.py.
Functions are compiled from the source AST; dependency ports are mocked.
"""
import ast
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import types
import unittest
from typing import Any, Mapping
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)


def source_module(relative):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
    module = types.ModuleType(relative)
    module.__dict__.update(ast=ast, date=date, datetime=datetime, time=time, timedelta=timedelta,
        timezone=timezone, hashlib=hashlib, json=json, Path=Path, re=re, shlex=shlex,
        Any=Any, Mapping=Mapping, ZoneInfo=ZoneInfo, __file__=str(ROOT / relative))
    exec(compile(ast.Module(body=body, type_ignores=[]), relative, 'exec'), module.__dict__)
    return module


def artifact(stamp='2026-11-06T23:00:00+00:00', session='2026-11-06'):
    return dict(present=True, source='db', created_at=NOW.isoformat(), run_date=session,
        payload=dict(generated_at_utc=stamp, snapshot_date=session, feature_compat=dict(compatible=True)))


class OfflineTests(unittest.TestCase):
    def setUp(self):
        for target in ('socket.socket.connect', 'socket.create_connection', 'subprocess.Popen', 'subprocess.run'):
            p = patch(target, side_effect=AssertionError('external_execution_forbidden'))
            p.start()
            self.addCleanup(p.stop)
        self.ml = source_module('dashboards/ml_overview_api.py')
        self.ops = source_module('dashboards/operational_status.py')
        self.strategy = source_module('scripts/strategy_dashboard.py')

    def test_old_compatible_predictions_never_become_fresh(self):
        value = self.ml.artifact_freshness(artifact('2026-10-25T04:04:59+00:00'), now=NOW,
            expected_session='2026-11-06', historical_stale=False)
        self.assertEqual(value['status'], 'stale')
        self.assertTrue(value['stale'])
        self.assertGreater(value['age_seconds'], 14 * 86400)
        self.assertEqual(value['timestamp_source'], 'payload.generated_at_utc')

    def test_weekend_uses_bound_session_and_elapsed_age(self):
        value = self.ml.artifact_freshness(artifact(), now=NOW, expected_session='2026-11-06')
        self.assertEqual(value['status'], 'fresh')
        self.assertEqual(value['session_status'], 'current')
        mismatch = self.ml.artifact_freshness(artifact(session='2026-11-05'), now=NOW, expected_session='2026-11-06')
        self.assertEqual(mismatch['reason'], 'artifact_signal_session_mismatch')
        unknown = self.ml.artifact_freshness(artifact(), now=NOW)
        self.assertEqual(unknown['status'], 'unknown')
        self.assertIsNone(unknown['stale'])

    def test_future_invalid_naive_timestamp_does_not_fall_back_to_storage_time(self):
        for stamp in ('bad', '2026-11-08T10:00:00', '2026-11-09T00:00:00Z'):
            with self.subTest(stamp=stamp):
                result = self.ml.artifact_freshness(artifact(stamp), now=NOW, expected_session='2026-11-06')
                self.assertEqual(result['status'], 'unknown')
                self.assertIsNone(result['stale'])

    def test_missing_and_producer_stale_are_not_fresh(self):
        result = self.ml.artifact_freshness(dict(artifact(), present=False), now=NOW, expected_session='2026-11-06')
        self.assertEqual(result['status'], 'missing')
        result = self.ml.artifact_freshness(artifact(), now=NOW, expected_session='2026-11-06', historical_stale=True)
        self.assertEqual(result['status'], 'stale')

    def test_native_guard_priority_is_preserved(self):
        args = dict(freshness_stale=False, recommended_action=None, coverage_pct=100, has_any_ml_data=True)
        self.assertEqual(self.ml._status_label(**args, guard_decision='block'), ('Guard held', 'error'))
        self.assertEqual(self.ml._status_label(**args, guard_decision=' BLOCK '), ('Guard held', 'error'))
        self.assertEqual(self.ml._status_label(**args, refresh_due=True)[1], 'warning')
        self.assertEqual(self.ml._status_label(**dict(args, freshness_stale=None))[0], 'Unknown')

    def test_refresh_due_boolean_precedence_and_alias(self):
        cases = [
            ({'refresh_due': True}, True),
            ({'monitor_refresh_due': True}, True),
            ({'refresh_due': False, 'monitor_refresh_due': True}, False),
            ({'refresh_due': True, 'monitor_refresh_due': False}, True),
            ({'refresh_due': None, 'monitor_refresh_due': True}, True),
            ({'refresh_due': 'false', 'monitor_refresh_due': True}, True),
            ({'refresh_due': 'true'}, False),
            ({'refresh_due': 1, 'monitor_refresh_due': 'true'}, False),
            ({'monitor_refresh_due': 1}, False),
            ({}, False),
        ]
        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertIs(self.ml._guard_refresh_due(payload), expected)

    def test_overview_preserves_native_guard_fields_and_normalization(self):
        self.ml._artifact_record = lambda *a, **kw: artifact()
        self.ml._latest_model_snapshot = lambda *a: artifact()
        self.ml._expected_prediction_session = lambda *a: '2026-11-06'
        self.ml.load_latest_champion = lambda *a: {}
        self.ml.champion_execution_overrides = lambda *a: {}
        self.ml.safe_tail_lines = lambda *a, **kw: []
        for health, expected in [
            ({'refresh_due': True}, 'Guard warning / refresh due'),
            ({'monitor_refresh_due': True}, 'Guard warning / refresh due'),
            ({'refresh_due': False, 'monitor_refresh_due': True}, 'Predictions fresh'),
            ({'refresh_due': 'false', 'monitor_refresh_due': True}, 'Guard warning / refresh due'),
            ({'decision': 'BLOCK', 'refresh_due': True}, 'Guard held'),
            ({'decision': ' WARNING '}, 'Guard warning / refresh due'),
        ]:
            with self.subTest(health=health):
                self.ml._latest_status_payload = lambda *a: dict(
                    model_score_coverage=dict(pct=100), ml_health=health)
                result = self.ml.build_ml_overview(ROOT, now=NOW)
                self.assertEqual(result['status']['label'], expected)
                self.assertEqual(result['monitor']['guard_decision'],
                    str(health.get('decision') or '').strip().lower() or None)

    def test_full_overview_distinguishes_artifacts_and_matching_execution(self):
        records = dict(predictions=artifact(), labels=artifact(), features=artifact(), ranker_eval=artifact(),
            ranker_monitor=artifact(), ranker_recalibrate=artifact(), ranker_autoremediate=artifact(),
            ranker_trade_attribution=artifact(), ranker_oos_predictions=artifact())
        self.ml._artifact_record = lambda kind, **kw: records[kind]
        self.ml._latest_status_payload = lambda *a: dict(model_score_coverage=dict(pct=100))
        self.ml._latest_model_snapshot = lambda *a: artifact()
        self.ml._expected_prediction_session = lambda *a: '2026-11-06'
        self.ml.load_latest_champion = lambda *a: {}
        self.ml.champion_execution_overrides = lambda *a: {}
        self.ml.safe_tail_lines = lambda *a, **kw: []
        result = self.ml.build_ml_overview(ROOT, now=NOW)
        self.assertEqual(result['freshness']['label'], 'Fresh')
        self.assertEqual(result['fresh_success_count'], 0)
        self.assertIsNone(result['last_successful_execution'])
        self.assertFalse(any(s['status'] == 'success' for s in result['pipeline_stages']))
        self.ml.safe_tail_lines = lambda *a, **kw: ['2026-11-06 23:00:01 [INFO] RANKER_PREDICT rc=0']
        result = self.ml.build_ml_overview(ROOT, now=NOW)
        self.assertEqual(result['fresh_success_count'], 1)
        records['predictions'] = artifact('2026-10-25T04:04:59Z')
        self.ml.safe_tail_lines = lambda *a, **kw: ['2026-11-08 11:00:00 [INFO] PREDICTIONS_FRESHNESS stale=false']
        result = self.ml.build_ml_overview(ROOT, now=NOW)
        self.assertEqual(result['freshness']['label'], 'Stale')
        self.assertEqual(result['timestamps']['last_predict'], '2026-10-25T04:04:59+00:00')
        self.assertEqual(result['fresh_success_count'], 0)

    def test_prediction_failure_is_not_hidden_by_stale_artifact(self):
        self.ml._artifact_record = lambda *a, **kw: artifact('2026-10-25T04:04:59Z')
        self.ml._latest_status_payload = lambda *a: dict(model_score_coverage=dict(pct=100))
        self.ml._latest_model_snapshot = lambda *a: artifact()
        self.ml._expected_prediction_session = lambda *a: '2026-11-06'
        self.ml.load_latest_champion = lambda *a: {}
        self.ml.champion_execution_overrides = lambda *a: {}
        self.ml.safe_tail_lines = lambda *a, **kw: ['2026-11-08 11:00:00 [INFO] RANKER_PREDICT rc=1']
        result = self.ml.build_ml_overview(ROOT, now=NOW)
        prediction = next(s for s in result['pipeline_stages'] if s['label'] == 'Predict')
        self.assertEqual(prediction['freshness_status'], 'stale')
        self.assertEqual(prediction['status'], 'blocked')
        self.assertEqual(prediction['tone'], 'error')
        self.assertFalse(prediction['fresh_success'])
        self.assertIsNone(result['last_successful_execution'])

    def test_compute_completion_is_not_research_or_entry_authority(self):
        verified = dict(primary=dict(run_id='a'*32, finished_at=NOW.isoformat()), record=dict(run_id='b'*32))
        result = self.strategy.computation_completion(verified)
        self.assertEqual(result['completion_status'], 'verified')
        self.assertTrue(result['computation_complete'])
        self.assertFalse(result['research_qualified'])
        self.assertFalse(result['order_submission_authorized'])
        self.strategy.snapshot = lambda **kw: dict(result, ok=True, status='EMPTY', rows=[], rows_final=0,
            run_ts_utc=NOW.isoformat())
        different = self.strategy.api_payload('candidates', dict(run_ts_utc='2026-11-07T12:00:00Z'))
        self.assertEqual(different['completion_status'], 'unverified')
        self.assertFalse(different['computation_complete'])
        missing = self.strategy.api_payload('backtest', {})
        self.assertEqual(missing['completion_status'], 'unverified')

    def test_newer_guarded_invocation_preserves_actual_execution_receipt(self):
        prior = dict(ok=True, status='STALE', source='executor_receipt', rc=0,
            finished_utc='2026-11-06T09:35:11+00:00')
        self.ops._execution_receipt = lambda *a, **kw: prior
        text = '\n'.join([
            '2026-11-06 07:55:04 Started task',
            '2026-11-06 09:35:11 Completed task, took 6007 seconds, return code was 0',
            '2026-11-07 07:55:09 Started task',
            json.dumps(dict(status='SPLIT_EXECUTOR_LAUNCHER_REFUSED',
                reason='executor_scheduled_start_outside_bound', orders_submitted=0)),
            '2026-11-07 07:55:14 Completed task, took 5 seconds, return code was 1'])
        result = self.ops.execute_task(ROOT, now=NOW, scheduled_log=(text, '/var/log/schedule-log-1234.log'))
        self.assertEqual(result['rc'], 0)
        self.assertEqual(result['last_execution_receipt'], prior)
        attempt = result['last_scheduled_attempt']
        self.assertEqual(attempt['status'], 'GUARDED_NO_ENTRY')
        self.assertEqual(attempt['rc'], 1)
        self.assertEqual(attempt['orders_submitted'], 0)
        self.assertEqual(attempt['task_id'], 1234)
        self.assertEqual(attempt['release_attribution'], 'unverified_historical_attempt')
        newer = text + '\n2026-11-08 11:55:00 Started task'
        attempt = self.ops.scheduled_attempt((newer, '/var/log/schedule-log-1234.log'), now=NOW)
        self.assertEqual(attempt['status'], 'RUNNING')
        self.assertIsNone(attempt['rc'])
        self.assertIsNone(attempt['reason_code'])

    def test_attempt_missing_or_untrusted_reason_is_sanitized(self):
        self.assertFalse(self.ops.scheduled_attempt(None, now=NOW)['ok'])
        text = '2026-11-07 07:55:09 Started task\n' + json.dumps(dict(
            status='SPLIT_EXECUTOR_LAUNCHER_REFUSED', reason='password=secret', orders_submitted=0))
        result = self.ops.scheduled_attempt((text, 'schedule-log-1234.log'), now=NOW)
        self.assertEqual(result['reason_code'], 'input_or_source_gate')
        self.assertNotIn('secret', json.dumps(result))
        self.assertFalse(self.ops.scheduled_attempt((text, 'pipeline.log'), now=NOW)['ok'])

    def release(self, seconds=60):
        base = '/home/test/jbravo-research-tools/TEST_RELEASE'
        entry = f'''from datetime import timedelta
from zoneinfo import ZoneInfo
def run_entry_only(config):
    deadline = opening + timedelta(seconds={seconds})
    config._entry_lifecycle_owner = 'MONITOR_POSITIONS'
    config._entry_cancellation_policy = 'monitor_exchange_regular_close'
'''.encode()
        lifecycle = b"OWNER = 'MONITOR_POSITIONS'\nNY = ZoneInfo('America/New_York')\n"
        sources = {'strategy_entry_only.py': entry, 'strategy_order_lifecycle.py': lifecycle,
                   'strategy_order_ownership_install.py': b'def install_entry(): pass\n'}
        pins = {n:dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n,b in sources.items()}
        support = ('PINS = ' + repr(pins) + '''\ndef install_entry(runtime):
    load('strategy_entry_only.py')
    load('strategy_order_lifecycle.py')
    load('strategy_order_ownership_install.py')
''').encode()
        support_pin = dict(bytes=len(support), sha256=hashlib.sha256(support).hexdigest())
        launcher = ("ARGV = ['--submit-at-ny','04:00','--cancel-after-min','35']\n" +
            'def ownership_support():\n    pin = ' + repr(support_pin) +
            '\ndef main():\n    ownership_support().install_entry(runtime)\n').encode()
        pin = hashlib.sha256(launcher).hexdigest()
        path = base + '/executor_boot.py'
        command = f"printf '%s\\\\n' '{pin}  {path}' | /usr/bin/sha256sum --check --status && /usr/bin/timeout --signal=TERM --kill-after=10s 40200s /home/test/.virtualenvs/test/bin/python -I -B {path} --scheduled"
        sources.update({'executor_boot.py':launcher, 'ownership_boot_support.py':support})
        return dict(id=1234, enabled=True, command=command), {base+'/'+n:b for n,b in sources.items()}

    def test_schedule_derives_pinned_active_source_not_legacy_deadline(self):
        task, sources = self.release()
        result = self.ops.execution_schedule(ROOT, scheduled_tasks=[task], now=NOW, source_reader=sources.__getitem__)
        self.assertEqual(result['status'], 'verified')
        self.assertEqual((result['entry_start_local'], result['entry_end_local']), ('04:00','04:01'))
        self.assertEqual(result['lifecycle_owner'], 'MONITOR_POSITIONS')
        self.assertEqual(result['legacy_cancel_after_min'], 35)
        self.assertFalse(result['legacy_cancel_after_min_is_deadline'])
        self.assertIsNone(result['cancellation_deadline'])
        self.assertIsNone(result['in_window'])
        self.assertNotIn('command', result)
        task, sources = self.release(seconds=90)
        result = self.ops.execution_schedule(ROOT, scheduled_tasks=[task], now=NOW, source_reader=sources.__getitem__)
        self.assertEqual(result['submission_window_seconds'], 90)

    def test_schedule_current_session_and_dst_are_bound_not_guessed(self):
        task, sources = self.release()
        module = types.ModuleType('scripts.strategy_dashboard')
        module.primary_receipt = lambda *a, **kw: (dict(session=dict(next_session='2026-11-09')), b'')
        now = datetime(2026, 11, 9, 9, 0, 30, tzinfo=timezone.utc)
        with patch.dict(sys.modules, {'scripts.strategy_dashboard': module}):
            result = self.ops.execution_schedule(ROOT, scheduled_tasks=[task], now=now, source_reader=sources.__getitem__)
        self.assertEqual(result['entry_open_at'], '2026-11-09T09:00:00+00:00')
        self.assertTrue(result['in_window'])
        self.assertIsNone(result['cancellation_deadline'])
        module.primary_receipt = lambda *a, **kw: (dict(session=dict(next_session='2026-10-30')), b'')
        now = datetime(2026, 10, 30, 8, 1, tzinfo=timezone.utc)
        with patch.dict(sys.modules, {'scripts.strategy_dashboard': module}):
            result = self.ops.execution_schedule(ROOT, scheduled_tasks=[task], now=now, source_reader=sources.__getitem__)
        self.assertEqual(result['entry_open_at'], '2026-10-30T08:00:00+00:00')
        self.assertFalse(result['in_window'])

    def test_explicit_data_root_reaches_snapshot(self):
        seen = []
        self.strategy.snapshot = lambda **kw: seen.append(kw) or dict(ok=False, rows=[], rows_final=0)
        self.strategy.api_payload('backtest', {}, base_dir=Path('isolated-data-root'), now=NOW)
        self.assertEqual(seen, [dict(base_dir=Path('isolated-data-root'), now=NOW)])

    def test_schedule_unknown_for_source_drift_missing_multiple_or_disabled_tasks(self):
        task, sources = self.release()
        for tasks in (None, [], [dict(task, enabled=False)], [task, task], [dict(task, command='execute_trades')]):
            result = self.ops.execution_schedule(ROOT, scheduled_tasks=tasks, now=NOW, source_reader=sources.__getitem__)
            self.assertEqual(result['status'], 'unknown')
            self.assertIsNone(result['in_window'])
        key = next(k for k in sources if k.endswith('/strategy_entry_only.py'))
        sources[key] += b'\n# changed\n'
        result = self.ops.execution_schedule(ROOT, scheduled_tasks=[task], now=NOW, source_reader=sources.__getitem__)
        self.assertEqual(result['status'], 'unknown')
        self.assertIsNone(result['entry_start_local'])


if __name__ == '__main__':
    unittest.main()
