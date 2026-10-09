"""Synthetic offline checks; existing pipeline execution and all host I/O are mocked."""
import ast
import argparse
import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
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

from scripts import pipeline_postflight as postflight
from scripts import primary_pipeline as primary

SOURCE_ROOT = Path(__file__).resolve().parents[1]


class PostflightTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.root = Path(self.storage.name)
        for relative in postflight.SOURCE_FILES:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE_ROOT / relative, target)
        self.now = datetime.now(timezone.utc)
        self.started = self.now - timedelta(minutes=1)
        self.finished = self.now - timedelta(seconds=1)
        self.facts = dict(
            started_at=self.started, finished_at=self.finished, pipeline_rc=0,
            steps=postflight.REQUIRED_STEPS,
            step_rcs={step: 0 for step in postflight.REQUIRED_STEPS},
            stage_times={step: 1.0 for step in postflight.REQUIRED_STEPS},
            degraded=False, labels_rows=120,
            freshness={'stale': False, 'pred_compatible': True, 'predict_rc': None,
                       'snapshot_date': self.started.date().isoformat()},
            coverage={'total': 6, 'non_null': 6, 'pct': 100.0,
                      'run_ts_utc': (self.started + timedelta(seconds=2)).isoformat()},
            ml_health={'decision': 'allow', 'reasons': []},
            controls={key: True for key in ('strict_predictions_meta', 'strict_auto_refresh_predictions',
                       'auto_refresh_features', 'auto_refresh_predictions', 'use_champion',
                       'ml_health_guard', 'refresh_predictions_for_candidates')},
        )
        # Guard even accidental external dispatch during these tests.
        for target in ('socket.socket.connect', 'subprocess.Popen', 'subprocess.run'):
            patcher = mock.patch(target, side_effect=AssertionError('external_execution_forbidden'))
            patcher.start()
            self.addCleanup(patcher.stop)

    def publish(self, **changes):
        return postflight.publish_health(self.root, **(self.facts | changes))

    def check(self, **changes):
        return postflight.check_health(self.root, now=self.now, **changes)

    def test_clean_run_history_and_source_bindings(self):
        report = self.publish()
        result = self.check()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['run_id'], report['run_id'])
        self.assertFalse(result['research_qualified'])
        self.assertFalse(result['operating_acceptance'])
        latest = self.root / 'reports/pipeline_postflight/latest.json'
        self.assertEqual(latest.read_bytes(), (latest.parent / ('run-' + report['run_id'] + '.json')).read_bytes())
        for relative, identity in report['source_bindings'].items():
            data = (self.root / relative).read_bytes()
            self.assertEqual(identity, {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})

    def test_checker_does_not_write_or_import_application_modules(self):
        self.publish()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        real_import = __import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith(('alpaca', 'requests', 'psycopg', 'scripts.db', 'scripts.run_pipeline')):
                raise AssertionError('application_import_forbidden')
            return real_import(name, *args, **kwargs)
        with mock.patch('builtins.__import__', side_effect=guarded_import), \
             mock.patch.object(Path, 'mkdir', side_effect=AssertionError('write_forbidden')), \
             mock.patch.object(os, 'replace', side_effect=AssertionError('write_forbidden')):
            self.assertEqual(self.check()['status'], 'ok')
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()})

    def test_missing_report_fails_without_creating_output(self):
        self.assertEqual(self.check()['status'], 'failed')
        self.assertFalse((self.root / 'reports').exists())

    def test_failed_required_and_auxiliary_stages_cannot_hide_behind_zero_pipeline_rc(self):
        for step in (*postflight.REQUIRED_STEPS, 'features', 'ranker_predict'):
            with self.subTest(step=step):
                record = self.publish(step_rcs=self.facts['step_rcs'] | {step: 1})
                self.assertEqual(self.check()['status'], 'failed')
                self.assertEqual(record['pipeline_rc'], 0)

    def test_missing_and_skipped_stages_fail(self):
        for step in postflight.REQUIRED_STEPS:
            with self.subTest(step=step):
                self.publish(stage_times=self.facts['stage_times'] | {step: 0.0})
                self.assertIn('skipped_or_unmeasured_stage_' + step, self.check()['failures'])
                self.publish(step_rcs={s: 0 for s in postflight.REQUIRED_STEPS if s != step})
                self.assertIn('missing_stage_' + step, self.check()['failures'])

    def test_empty_labels_fail(self):
        self.publish(labels_rows=0)
        self.assertIn('labels_empty_or_unknown', self.check()['failures'])

    def test_monitor_warning_is_degraded_not_success(self):
        self.publish(ml_health={'decision': 'warn', 'reasons': ['stale_monitor']})
        result = self.check()
        self.assertEqual(result['status'], 'degraded')
        self.assertIn('ml_health_reason_stale_monitor', result['qualifications'])

    def test_blocked_missing_or_contradictory_health_never_passes(self):
        for health in (None, {'decision': 'block', 'reasons': ['action_retrain']},
                       {'decision': 'allow', 'reasons': ['stale_monitor']},
                       {'decision': 'allow', 'reasons': None}):
            with self.subTest(health=health):
                self.publish(ml_health=health)
                self.assertNotEqual(self.check()['status'], 'ok')

    def test_stale_incompatible_missing_and_failed_predictions_fail(self):
        for changes in ({'stale': True}, {'pred_compatible': False}, {'pred_compatible': None},
                        {'predict_rc': 1}, {'predict_rc': True}, {'snapshot_date': None},
                        {'snapshot_date': (self.now.date() + timedelta(days=1)).isoformat()}):
            with self.subTest(changes=changes):
                self.publish(freshness=self.facts['freshness'] | changes)
                self.assertEqual(self.check()['status'], 'failed')

    def test_explicit_compatible_string_and_successful_refresh_are_supported(self):
        self.publish(freshness=self.facts['freshness'] | {'pred_compatible': 'True', 'predict_rc': 0})
        self.assertEqual(self.check()['status'], 'ok')

    def test_disabled_required_controls_fail(self):
        for key in self.facts['controls']:
            with self.subTest(key=key):
                self.publish(controls=self.facts['controls'] | {key: False})
                self.assertIn('required_ml_controls_missing', self.check()['failures'])

    def test_coverage_loss_and_zero_candidates_are_degraded(self):
        for coverage in ({'total': 6, 'non_null': 3, 'pct': 50.0, 'run_ts_utc': self.started.isoformat()},
                         {'total': 0, 'non_null': 0, 'pct': 0.0, 'run_ts_utc': None}):
            with self.subTest(coverage=coverage):
                self.publish(coverage=coverage)
                self.assertEqual(self.check()['status'], 'degraded')

    def test_invalid_and_previous_run_coverage_fail(self):
        for changes in ({'non_null': 7}, {'pct': 50.0}, {'total': True},
                        {'run_ts_utc': None}, {'run_ts_utc': (self.started - timedelta(seconds=1)).isoformat()}):
            with self.subTest(changes=changes):
                self.publish(coverage=self.facts['coverage'] | changes)
                self.assertEqual(self.check()['status'], 'failed')

    def test_age_date_future_and_invocation_binding(self):
        self.publish()
        self.assertEqual(self.check(max_age_seconds=1)['status'], 'failed')
        self.assertEqual(self.check(expected_day=self.now.date() - timedelta(days=1))['status'], 'failed')
        self.assertEqual(self.check(not_before=self.started + timedelta(seconds=1))['status'], 'failed')
        self.assertEqual(postflight.check_health(self.root, now=self.started)['status'], 'failed')

    def test_malformed_empty_duplicate_and_nonfinite_json_fail(self):
        path = self.root / 'reports/pipeline_postflight/latest.json'
        path.parent.mkdir(parents=True)
        for data in (b'', b'{', b'[]', b'null', b'{"schema_version":1,"schema_version":1}', b'{"value":NaN}'):
            with self.subTest(data=data):
                path.write_bytes(data)
                self.assertEqual(self.check()['status'], 'failed')

    def test_missing_history_and_source_mismatch_fail(self):
        record = self.publish()
        history = self.root / 'reports/pipeline_postflight' / ('run-' + record['run_id'] + '.json')
        history.unlink()
        self.assertEqual(self.check()['status'], 'failed')
        self.publish()
        with (self.root / 'scripts/run_pipeline.py').open('ab') as target:
            target.write(b'\n# modified\n')
        self.assertEqual(self.check()['status'], 'failed')

    def test_oversize_and_bad_settings_fail(self):
        path = self.root / 'reports/pipeline_postflight/latest.json'
        path.parent.mkdir(parents=True)
        path.write_bytes(b' ' * (postflight.MAX_REPORT_BYTES + 1))
        self.assertEqual(self.check()['status'], 'failed')
        for value in (0, -1, float('nan'), float('inf'), 86401, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(max_age_seconds=value)

    def test_ambiguous_timestamp_cannot_be_recorded(self):
        with self.assertRaises(ValueError):
            self.publish(started_at=self.started.replace(tzinfo=None))

    def test_recording_failure_does_not_publish_false_success(self):
        self.publish()
        path = self.root / 'reports/pipeline_postflight/latest.json'
        before = path.read_bytes()
        with mock.patch.object(os, 'replace', side_effect=OSError('synthetic_recording_failure')):
            with self.assertRaises(OSError):
                self.publish()
        self.assertEqual(path.read_bytes(), before)

    def test_cli_success_and_degraded_exit_status(self):
        actual_check = postflight.check_health
        for health, expected in (({'decision': 'allow', 'reasons': []}, 0),
                                  ({'decision': 'warn', 'reasons': ['stale_monitor']}, 1)):
            with self.subTest(expected=expected):
                self.publish(ml_health=health)
                with mock.patch.object(postflight, 'ROOT', self.root), \
                     mock.patch.object(postflight, 'check_health', wraps=lambda **kw: actual_check(self.root, now=self.now, **kw)), \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(postflight.main([]), expected)
                self.assertIn('status', json.loads(output.getvalue()))

    def test_primary_runs_pipeline_once_and_binds_new_report(self):
        calls = []
        def pipeline(arguments):
            calls.append(arguments)
            started = datetime.now(timezone.utc)
            self.publish(started_at=started, finished_at=datetime.now(timezone.utc),
                         coverage=self.facts['coverage'] | {'run_ts_utc': started.isoformat()})
            return 0
        with mock.patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(primary.main(['--backtest-quick', 'true', '--reload-web', 'false'],
                             pipeline_main=pipeline, base_dir=self.root), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].count('--steps'), 1)
        self.assertEqual(calls[0][calls[0].index('--steps') + 1], ','.join(postflight.REQUIRED_STEPS))
        self.assertNotIn('--limit 30', calls[0])
        self.assertIn('--backtest-quick', calls[0])

    def test_primary_rejects_fixed_overrides_before_import_or_execution(self):
        for arguments in (['--steps', 'screener'], ['--steps=screener'], ['--auto-refresh-features=false'],
                          ['--allow-no-screener'], ['--ml-health-guard-mode', 'block'],
                          ['--step', 'screener'], ['--auto-refresh-f=false']):
            with self.subTest(arguments=arguments), contextlib.redirect_stdout(io.StringIO()):
                runner = mock.Mock()
                self.assertEqual(primary.main(arguments, pipeline_main=runner), 1)
                runner.assert_not_called()

    def test_primary_preserves_failure_and_rejects_old_success_report(self):
        self.publish()
        for outcome in (1, None, True, SystemExit(2), RuntimeError('synthetic')):
            with self.subTest(outcome=outcome), contextlib.redirect_stdout(io.StringIO()):
                runner = mock.Mock(side_effect=outcome) if isinstance(outcome, BaseException) else mock.Mock(return_value=outcome)
                self.assertEqual(primary.main([], pipeline_main=runner, base_dir=self.root), 1)
                runner.assert_called_once()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(primary.main([], pipeline_main=lambda args: 0, base_dir=self.root), 1)

    def test_primary_normal_entry_deferred_import_and_explicit_zero_completion(self):
        module = ModuleType('scripts.run_pipeline')
        def pipeline(arguments):
            started = datetime.now(timezone.utc)
            self.publish(started_at=started, finished_at=datetime.now(timezone.utc),
                         coverage=self.facts['coverage'] | {'run_ts_utc': started.isoformat()})
            raise SystemExit(0)
        module.main = pipeline
        with mock.patch.dict(sys.modules, {'scripts.run_pipeline': module}), \
             mock.patch.dict(os.environ, {'JBR_STRICT_PREDICTIONS_META': 'false'}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(primary.main([], base_dir=self.root), 0)
            self.assertEqual(os.environ['JBR_STRICT_PREDICTIONS_META'], 'false')

    def test_real_pipeline_argument_parser_accepts_primary_contract(self):
        # The whole legacy pipeline has application/provider imports. Extract
        # only its exact parser functions; this test does not claim full import.
        tree = ast.parse((SOURCE_ROOT / 'scripts/run_pipeline.py').read_text(encoding='utf-8'))
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                       ('parse_args', '_extract_split_tokens', 'determine_steps')]
        mapping = next(n.value for n in tree.body if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == '_SPLIT_FLAG_MAP' for t in n.targets))
        namespace = {'argparse': argparse, 'os': os, 'sys': sys, 'Optional': object,
                     'Iterable': object, 'Sequence': object, '_SPLIT_FLAG_MAP': ast.literal_eval(mapping),
                     'DEFAULT_LABELS_BARS_PATH': Path('data') / 'daily_bars.csv'}
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                                 *definitions], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), 'retained_pipeline_parser', 'exec'), namespace)
        args = namespace['parse_args'](primary.build_arguments(['--reload-web', 'false', '--backtest-quick', 'true']))
        self.assertTrue(args.postflight_report)
        self.assertTrue(args.use_champion)
        self.assertEqual(args.auto_refresh_features, 'true')
        self.assertEqual(args.auto_refresh_predictions, 'true')
        self.assertEqual(namespace['determine_steps'](args.steps), list(postflight.REQUIRED_STEPS))

    def test_actual_pipeline_recording_hook_success_and_failure(self):
        tree = ast.parse((SOURCE_ROOT / 'scripts/run_pipeline.py').read_text(encoding='utf-8'))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        hooks = [n for n in ast.walk(main) if isinstance(n, ast.If) and isinstance(n.test, ast.Call)
                 and isinstance(n.test.func, ast.Name) and n.test.func.id == 'getattr'
                 and any(isinstance(a, ast.Constant) and a.value == 'postflight_report' for a in n.test.args)]
        self.assertEqual(len(hooks), 1)
        executable = compile(ast.fix_missing_locations(ast.Module(body=hooks, type_ignores=[])), 'pipeline_recording_hook', 'exec')
        namespace = dict(base_dir=self.root, args=SimpleNamespace(postflight_report=True), started_dt=self.started,
            datetime=datetime, timezone=timezone, rc=0, steps=self.facts['steps'], step_rcs=self.facts['step_rcs'],
            stage_times=self.facts['stage_times'], degraded=False, labels_rows=120,
            enrichment_freshness=self.facts['freshness'], model_score_coverage_summary=self.facts['coverage'],
            ml_health_summary=self.facts['ml_health'], LOG=mock.Mock(), **{key: True for key in
            ('strict_predictions_meta', 'strict_auto_refresh_predictions', 'auto_refresh_features',
             'auto_refresh_predictions', 'use_champion', 'ml_health_guard_enabled', 'refresh_predictions_for_candidates')})
        exec(executable, namespace)
        self.assertEqual(postflight.check_health(self.root)['status'], 'ok')
        for original_rc in (0, 7):
            with self.subTest(original_rc=original_rc), mock.patch.object(postflight, 'publish_health', side_effect=OSError('synthetic')):
                namespace['rc'] = original_rc
                exec(executable, namespace)
                self.assertEqual(namespace['rc'], 1 if original_rc == 0 else original_rc)

    def test_shell_checker_and_primary_launcher_contracts(self):
        checker = (SOURCE_ROOT / 'scripts/ops/run_canary_smoke.sh').read_text()
        for forbidden in ('scripts.run_pipeline', 'ENV_FILE', 'mkdir', 'tee', 'grep', '--limit', 'CANARY_LOG'):
            self.assertNotIn(forbidden, checker)
        self.assertIn('exec python -B -m scripts.pipeline_postflight "$@"', checker)
        launcher = (SOURCE_ROOT / 'scripts/ops/run_primary_pipeline.sh').read_text()
        self.assertIn('flock -n 9', launcher)
        self.assertIn('exec python -B -m scripts.primary_pipeline "$@"', launcher)

    def test_symlink_path_rejected_without_writing(self):
        self.publish()
        real_check = Path.is_symlink
        def redirected(path):
            return path == self.root / 'reports' or real_check(path)
        with mock.patch.object(Path, 'is_symlink', redirected):
            self.assertEqual(self.check()['status'], 'failed')
            with self.assertRaises(ValueError):
                self.publish()

    def test_unrecognized_health_reason_is_bounded_and_not_green(self):
        record = self.publish(ml_health={'decision': 'allow', 'reasons': ['private text ' * 500]})
        self.assertEqual(record['ml_health']['reasons'], ['unknown_ml_health_reason'])
        self.assertEqual(self.check()['status'], 'degraded')


if __name__ == '__main__':
    unittest.main()
