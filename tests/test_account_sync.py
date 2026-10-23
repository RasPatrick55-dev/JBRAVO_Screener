"""Offline supervisor tests and normal executor gate integration with external doubles."""
import contextlib
from datetime import datetime, timedelta
import hashlib
import importlib
import io
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

from scripts import account_sync as sync


def successful_stage():
    return dict(status='ok', exit_code=0, reason=None, elapsed_seconds=0.01,
                output_bytes=10, failure_markers=[], missing_completion_markers=[],
                qualifications=[])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.state = Path(self.storage.name) / 'account_sync'
        self.addCleanup(self.storage.cleanup)

    def test_successful_stage_order_and_saved_health(self):
        calls = []
        def run(stage, timeout, descriptor):
            calls.append((stage.name, timeout, descriptor))
            return successful_stage()
        report = sync.run_workflow(self.state, runner=run)
        self.assertEqual([x[0] for x in calls], ['activities', 'reconciliation', 'snapshot'])
        self.assertTrue(all(x[1] == 300 and isinstance(x[2], int) for x in calls))
        self.assertEqual(report['status'], 'ok')
        self.assertTrue(report['no_order_workflow'])
        latest = (self.state / 'latest.json').read_bytes()
        self.assertEqual(json.loads(latest), report)
        self.assertEqual((self.state / ('run-' + report['run_id'] + '.json')).read_bytes(), latest)
        for relative, binding in report['source_bindings'].items():
            data = (sync.ROOT / relative).read_bytes()
            self.assertEqual(binding, {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        self.assertIn('scripts/db.py', report['source_bindings'])

    def test_failure_stops_all_later_stages(self):
        for failed_index in range(3):
            with self.subTest(index=failed_index):
                calls = []
                def run(stage, timeout, descriptor):
                    calls.append(stage.name)
                    result = successful_stage()
                    if len(calls) == failed_index + 1:
                        result.update(status='failed', exit_code=1)
                    return result
                report = sync.run_workflow(self.state, runner=run)
                self.assertEqual(len(calls), failed_index + 1)
                self.assertEqual(report['status'], 'failed')
                self.assertTrue(all(x['status'] == 'not_run' for x in report['stages'][failed_index + 1:]))

    def test_reconciliation_degradation_is_not_overall_success(self):
        def run(stage, timeout, descriptor):
            result = successful_stage()
            if stage.name == 'reconciliation':
                result.update(status='degraded', qualifications=['RECONCILE_DECORATE_MISS'])
            return result
        report = sync.run_workflow(self.state, runner=run)
        self.assertEqual(report['status'], 'degraded')
        self.assertEqual(report['stages'][2]['status'], 'not_run')

    def test_exception_preserves_failure_type_without_secret_text(self):
        runner = mock.Mock(side_effect=RuntimeError('APCA_API_SECRET_KEY=private-value'))
        report = sync.run_workflow(self.state, runner=runner)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['stages'][0]['error_type'], 'RuntimeError')
        self.assertNotIn('private-value', json.dumps(report))
        runner.assert_called_once()

    def test_incomplete_or_contradictory_runner_results_fail(self):
        bad_results = [{'status': 'ok'}, {**successful_stage(), 'exit_code': 1},
                       {**successful_stage(), 'exit_code': True},
                       {**successful_stage(), 'failure_markers': ['ACT_FAIL']},
                       {**successful_stage(), 'qualifications': ['unknown']},
                       {**successful_stage(), 'status': 'unknown'}]
        for result in bad_results:
            with self.subTest(result=result):
                report = sync.run_workflow(self.state, runner=mock.Mock(return_value=result))
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['stages'][1]['status'], 'not_run')

    def test_overlap_runs_no_stage_and_does_not_replace_latest(self):
        sync.run_workflow(self.state, runner=mock.Mock(side_effect=lambda *a: successful_stage()))
        original = (self.state / 'latest.json').read_bytes()
        with sync.workflow_lock(self.state / 'account-sync.lock'):
            runner = mock.Mock()
            report = sync.run_workflow(self.state, runner=runner)
            runner.assert_not_called()
            self.assertEqual(report['status'], 'overlap')
        self.assertEqual((self.state / 'latest.json').read_bytes(), original)
        self.assertTrue((self.state / ('run-' + report['run_id'] + '.json')).exists())

    def test_lock_releases_after_failed_run(self):
        sync.run_workflow(self.state, runner=mock.Mock(side_effect=ValueError()))
        report = sync.run_workflow(self.state, runner=mock.Mock(side_effect=lambda *a: successful_stage()))
        self.assertEqual(report['status'], 'ok')

    def test_runs_keep_distinct_history(self):
        for _ in range(2):
            sync.run_workflow(self.state, runner=mock.Mock(side_effect=lambda *a: successful_stage()))
        self.assertEqual(len(list(self.state.glob('run-*.json'))), 2)

    def test_invalid_timeout_cannot_launch_or_create_output(self):
        for timeout in (0, -1, float('nan'), float('inf'), 901):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                sync.run_workflow(self.state, timeout, runner=mock.Mock())
        self.assertFalse(self.state.exists())

    def test_cli_returns_nonzero_for_failed_degraded_and_overlap(self):
        for status, code in [('ok', 0), ('failed', 1), ('degraded', 1), ('overlap', 1)]:
            with self.subTest(status=status), mock.patch.object(sync, 'run_workflow',
                    return_value={'run_id': 'synthetic', 'status': status, 'stages': []}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sync.main(['--state-dir', str(self.state)]), code)

    def test_recording_failure_is_unsuccessful_and_does_not_echo_exception(self):
        captured = io.StringIO()
        with mock.patch.object(sync, 'save_report', side_effect=OSError('secret-value')), \
                mock.patch.object(sync, 'run_stage'), contextlib.redirect_stdout(captured):
            # main's default runner is bound at definition; replace the workflow
            # only here so this CLI boundary test cannot start application code.
            with mock.patch.object(sync, 'run_workflow', side_effect=OSError('secret-value')):
                self.assertEqual(sync.main(['--state-dir', str(self.state)]), 1)
        self.assertNotIn('secret-value', captured.getvalue())
        self.assertEqual(json.loads(captured.getvalue())['error_type'], 'OSError')

    def test_cli_has_no_arbitrary_stage_or_command_arguments(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            sync.main(['--no-reconcile-only'])

    def test_stage_arguments_preserve_old_settings_and_fix_no_order_modes(self):
        self.assertEqual(sync.STAGES[0].args, ('--lookback-days', '30'))
        args = dict(zip(sync.STAGES[1].args[::2], sync.STAGES[1].args[1::2]))
        self.assertEqual(args, {'--reconcile-only': 'true', '--dry-run': 'true',
            '--submit-at-ny': '', '--ignore-market-gate': 'true',
            '--reconcile-use-watermark': 'true', '--reconcile-lookback-days': '14',
            '--reconcile-limit': '500', '--reconcile-overlap-secs': '300', '--max-poll-secs': '1'})
        self.assertEqual(sync.STAGES[2].args, ())


class StageTests(unittest.TestCase):
    def invoke(self, stage, text, code=0):
        process = mock.Mock(returncode=code)
        process.poll.return_value = code
        def create(command, **options):
            options['stdout'].write(text.encode('utf-8'))
            return process
        with mock.patch.dict(os.environ, {'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets'}, clear=True), \
                mock.patch.object(sync.subprocess, 'Popen', side_effect=create) as popen:
            result = sync.run_stage(stage, 300, 123)
            self.assertEqual(popen.call_args.args[0], [sync.sys.executable, '-m', stage.module, *stage.args])
            self.assertFalse(popen.call_args.kwargs['shell'])
            if sync.os.name == 'posix':
                self.assertEqual(popen.call_args.kwargs['pass_fds'], (123,))
                self.assertTrue(popen.call_args.kwargs['start_new_session'])
            return result

    def test_all_stages_require_successful_completion(self):
        for stage in sync.STAGES:
            with self.subTest(stage=stage.name):
                self.assertEqual(self.invoke(stage, '\n'.join(stage.required))['status'],
                                 'failed' if stage.name == 'reconciliation' else 'ok')

    def test_zero_exit_without_completion_is_failed(self):
        for stage in sync.STAGES:
            with self.subTest(stage=stage.name):
                result = self.invoke(stage, '')
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['missing_completion_markers'], list(stage.required))

    def test_nonzero_exit_cannot_be_hidden_by_success_markers(self):
        stage = sync.STAGES[0]
        result = self.invoke(stage, '\n'.join(stage.required), code=2)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['exit_code'], 2)

    def test_every_suppressed_failure_marker_is_detected(self):
        for stage in sync.STAGES:
            for marker in (*stage.failures, *sync.GENERIC_FAILURES):
                with self.subTest(stage=stage.name, marker=marker):
                    result = self.invoke(stage, '\n'.join((*stage.required, marker)))
                    self.assertEqual(result['status'], 'failed')
                    self.assertIn(marker, result['failure_markers'])

    def test_missing_watermark_update_is_failed(self):
        result = self.invoke(sync.STAGES[1], 'RECONCILE_START\nRECONCILE_END')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('RECONCILE_WATERMARK_UPDATE', result['missing_completion_markers'])

    def test_unresolved_decoration_without_scoped_proof_is_failed(self):
        stage = sync.STAGES[1]
        result = self.invoke(stage, '\n'.join((*stage.required, 'RECONCILE_DECORATE_MISS')))
        self.assertEqual(result['status'], 'failed')

    def test_arbitrary_child_output_is_not_saved(self):
        stage = sync.STAGES[0]
        result = self.invoke(stage, '\n'.join(stage.required) + '\nsecret-value')
        self.assertNotIn('secret-value', json.dumps(result))
        self.assertEqual(result['status'], 'ok')

    def test_output_limit_remains_failure_after_exit_zero(self):
        stage = sync.STAGES[0]
        result = self.invoke(stage, '\n'.join(stage.required) + 'x' * sync.OUTPUT_LIMIT)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'output_limit')

    def test_timeout_stops_owned_child_without_retry(self):
        process = mock.Mock(pid=4321, returncode=-9)
        process.poll.return_value = None
        with mock.patch.object(sync.subprocess, 'Popen', return_value=process) as popen, \
                mock.patch.object(sync.time, 'monotonic', side_effect=[0, 301, 302]), \
                mock.patch.object(sync.os, 'killpg', create=True) as killpg:
            result = sync.run_stage(sync.STAGES[0], 300, 123)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'timeout')
        popen.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)
        if sync.os.name == 'posix':
            killpg.assert_called_once_with(4321, sync.signal.SIGKILL)
        else:
            process.kill.assert_called_once()

    def test_spawn_failure_propagates(self):
        with mock.patch.object(sync.subprocess, 'Popen', side_effect=OSError()), self.assertRaises(OSError):
            sync.run_stage(sync.STAGES[0], 300, 123)


class ExecutorGateIntegrationTests(unittest.TestCase):
    """Normal executor import/parser/dispatcher, with all external effects blocked.

    The real wait helper and market gates execute. Candidate retrieval/ranking,
    broker authentication, reconciliation persistence and metrics are doubles;
    this verifies gate reachability, not live reconciliation semantics.
    """

    def setUp(self):
        # Import pandas before blocking sockets; no application imports yet.
        import pandas as pd
        self.pd = pd
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.storage = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(mock.patch.dict(os.environ, {
            'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets'}, clear=True))

        def forbidden(*args, **kwargs):
            raise AssertionError('external effect forbidden')

        self.forbidden = forbidden
        for target in ('socket.socket', 'socket.create_connection', 'subprocess.Popen', 'os.system'):
            self.stack.enter_context(mock.patch(target, side_effect=forbidden))
        self.stack.enter_context(mock.patch('time.sleep', side_effect=forbidden))

        def module(name, **fields):
            value = ModuleType(name)
            value.__dict__.update(fields)
            return value

        boundaries = {
            'requests': module('requests', get=forbidden, post=forbidden, HTTPError=RuntimeError),
            'alpaca': module('alpaca', __path__=[]),
            'scripts.db': module('scripts.db', db_enabled=forbidden, get_engine=forbidden),
            'scripts.db_queries': module('scripts.db_queries', get_latest_screener_candidates=forbidden),
            'scripts.utils': module('scripts.utils', __path__=[]),
            'scripts.utils.champion_config': module('scripts.utils.champion_config',
                champion_execution_overrides=forbidden, load_latest_champion=forbidden),
            'scripts.utils.env': module('scripts.utils.env', load_env=forbidden),
            'utils': module('utils', __path__=[], write_csv_atomic=forbidden),
            'utils.alerts': module('utils.alerts', send_alert=forbidden),
            'utils.env': module('utils.env', AlpacaCredentialsError=RuntimeError,
                AlpacaUnauthorizedError=RuntimeError, assert_alpaca_creds=forbidden,
                get_alpaca_creds=forbidden),
            'utils.telemetry': module('utils.telemetry', log_event=forbidden, get_version=forbidden),
        }
        self.stack.enter_context(mock.patch.dict(sys.modules, boundaries))
        self.stack.enter_context(mock.patch.object(sys.modules['scripts'], 'db', boundaries['scripts.db'], create=True))
        sys.modules.pop('scripts.execute_trades', None)
        self.executor = importlib.import_module('scripts.execute_trades')
        self.addCleanup(lambda: sys.modules.pop('scripts.execute_trades', None))
        for name in ('configure_logging', 'write_execute_metrics', '_paper_only_guard'):
            self.stack.enter_context(mock.patch.object(self.executor, name))
        self.stack.enter_context(mock.patch.object(self.executor, '_load_execute_metrics', return_value={}))
        self.stack.enter_context(mock.patch.object(self.executor, '_create_trading_client', side_effect=forbidden))

    def exercise_clock(self, when, session, next_open_days=0, reconciliation_complete=True):
        now = datetime.fromisoformat(when).replace(tzinfo=ZoneInfo('America/New_York'))
        clock = SimpleNamespace(timestamp=now.isoformat(), is_open=False, session=session,
            next_open=(now + timedelta(days=next_open_days)).replace(hour=9, minute=30).isoformat(),
            next_close=(now + timedelta(days=next_open_days)).replace(hour=16, minute=0).isoformat())
        frame = self.pd.DataFrame(columns=['symbol', 'run_date'])
        client = mock.Mock()
        client.get_clock.return_value = clock
        client.submit_order.side_effect = self.forbidden
        loader = mock.Mock()
        loader._get_trading_clock.return_value = clock
        loader.load_candidates.return_value = frame
        loader._rank_candidates.return_value = frame
        loader._apply_alloc_weight_key.return_value = frame
        execution = mock.Mock()
        execution.metrics.reconciliation = {
            'scope': 'ACTIVE_RECONCILIATION',
            'operational_complete': reconciliation_complete,
            'recovery_complete': False, 'accounting_complete': False,
            'activity_ingestion_complete': False, 'watermark_advanced': False,
            'trade_policy': {'scope': 'ACTIVE_RECONCILIATION', 'policy_revision': 0},
        }
        execution.execute.side_effect = self.forbidden
        execution.execute_order.side_effect = self.forbidden
        execution.hydrate_candidates.side_effect = self.forbidden
        messages = io.StringIO()
        logger = logging.Logger('offline-executor-gates', logging.INFO)
        logger.addHandler(logging.StreamHandler(messages))
        def reconcile():
            logger.info(operational_output() if reconciliation_complete else 'RECONCILE_START')
            if not reconciliation_complete:
                logger.warning('RECONCILE_WATERMARK_HELD')
            return reconciliation_complete
        execution.reconcile_closed_trades.side_effect = reconcile
        stage = sync.STAGES[1]
        args = self.executor.parse_args(list(stage.args))
        config = self.executor.build_config(args)
        self.assertTrue(config.reconcile_only)
        self.assertTrue(config.dry_run)
        self.assertTrue(config.ignore_market_gate)
        self.assertEqual(config.submit_at_ny, '')
        self.assertTrue(config.reconcile_use_watermark)
        calls = []

        def runner(current, timeout, descriptor):
            calls.append(current.name)
            if current.name != 'reconciliation':
                return successful_stage()
            with mock.patch.object(self.executor, 'LOGGER', logger), \
                    mock.patch.object(self.executor, '_ny_now', return_value=now), \
                    mock.patch.object(self.executor, 'TradeExecutor', side_effect=[loader, execution]), \
                    mock.patch.object(self.executor, '_wait_until_submit_at', wraps=self.executor._wait_until_submit_at) as wait:
                code = self.executor.run_executor(config, client=client)
                self.assertEqual(code, 0 if reconciliation_complete else 1)
                wait.assert_called_once_with('', session_open=None)
            # Feed the actual dispatcher log into the unchanged supervisor check.
            process = mock.Mock(returncode=code)
            process.poll.return_value = code
            def spawn(command, **options):
                self.assertEqual(command, [sys.executable, '-m', current.module, *stage.args])
                options['stdout'].write(messages.getvalue().encode())
                return process
            with mock.patch.object(sync.subprocess, 'Popen', side_effect=spawn):
                return sync.run_stage(current, timeout, descriptor)

        result = sync.run_workflow(Path(self.storage) / 'health', runner=runner)
        self.assertEqual(result['status'], 'degraded' if reconciliation_complete else 'failed')
        self.assertEqual(calls, ['activities', 'reconciliation'])
        execution.reconcile_closed_trades.assert_called_once_with()
        self.assertFalse(execution.metrics.reconciliation['accounting_complete'])
        self.assertFalse(execution.metrics.reconciliation['recovery_complete'])
        execution.execute.assert_not_called()
        execution.hydrate_candidates.assert_not_called()
        client.submit_order.assert_not_called()
        self.assertEqual(result['stages'][2]['status'], 'not_run')
        if not reconciliation_complete:
            self.assertIn('RECONCILE_WATERMARK_HELD', result['stages'][1]['health_reasons'])
        return messages.getvalue()

    def test_before_0700_et_reconciles_without_waiting(self):
        self.exercise_clock('2026-10-09T06:15:00', 'premarket')

    def test_incomplete_reconciliation_returns_failure_and_stops_snapshot(self):
        self.exercise_clock('2026-10-09T06:15:00', 'premarket', reconciliation_complete=False)

    def test_weekend_scoped_completion_holds_snapshot(self):
        text = self.exercise_clock('2026-10-10T06:15:00', 'closed', 2)
        self.assertIn('MARKET_GATE_BYPASSED', text)

    def test_weekday_holiday_closed_session_reconciles(self):
        text = self.exercise_clock('2026-12-25T06:15:00', 'holiday', 3)
        self.assertIn('MARKET_GATE_BYPASSED', text)


class PaperContextTests(unittest.TestCase):
    """Validate the actual child environment, with no real child or credentials."""

    def invoke(self, environment, stage=None, text=None):
        stage = stage or sync.STAGES[1]
        text = (operational_output() if stage.name == 'reconciliation' else '\n'.join(stage.required)) if text is None else text
        process = mock.Mock(returncode=0)
        process.poll.return_value = 0
        calls = []

        def spawn(command, **options):
            calls.append((command, options))
            options['stdout'].write(text.encode())
            return process

        with mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch.object(sync.subprocess, 'Popen', side_effect=spawn):
            parent_before = dict(os.environ)
            result = sync.run_stage(stage, 300, 123)
            self.assertEqual(dict(os.environ), parent_before)
        return result, calls

    def test_absent_flag_is_declared_only_for_validated_paper_child(self):
        environment = {'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets',
                       'SYNTHETIC_SENTINEL': 'preserved'}
        result, calls = self.invoke(environment)
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]['env'], {**environment, 'JBR_EXEC_PAPER': 'true'})
        self.assertNotIn('JBR_EXEC_PAPER', environment)

    def test_explicit_true_and_unrelated_child_values_are_preserved(self):
        environment = {'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets',
                       'JBR_EXEC_PAPER': 'true', 'SYNTHETIC_SENTINEL': 'preserved'}
        result, calls = self.invoke(environment)
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(calls[0][1]['env'], environment)
        self.assertIsNot(calls[0][1]['env'], os.environ)

    def test_false_or_malformed_flags_refuse_before_capture_or_child(self):
        for flag in ('false', 'False', 'TRUE', '1', '0', '', 'true ', ' true'):
            with self.subTest(flag=flag), mock.patch.dict(os.environ, {
                    'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets',
                    'JBR_EXEC_PAPER': flag}, clear=True), \
                    mock.patch.object(sync.subprocess, 'Popen') as popen, \
                    mock.patch.object(sync.tempfile, 'TemporaryFile') as capture:
                before = dict(os.environ)
                with self.assertRaisesRegex(ValueError, '^primary_paper_context_conflict$'):
                    sync.run_stage(sync.STAGES[1], 300, 123)
                popen.assert_not_called()
                capture.assert_not_called()
                self.assertEqual(dict(os.environ), before)

    def test_missing_live_and_noncanonical_endpoints_refuse(self):
        for endpoint in (None, 'https://api.alpaca.markets', 'https://example.invalid',
                         'https://paper-api.alpaca.markets/', 'HTTP://paper-api.alpaca.markets'):
            for flag in (None, 'true'):
                with self.subTest(endpoint=endpoint, flag=flag):
                    environment = {}
                    if endpoint is not None:
                        environment['APCA_API_BASE_URL'] = endpoint
                    if flag is not None:
                        environment['JBR_EXEC_PAPER'] = flag
                    with mock.patch.dict(os.environ, environment, clear=True), \
                            mock.patch.object(sync.subprocess, 'Popen') as popen, \
                            mock.patch.object(sync.tempfile, 'TemporaryFile') as capture:
                        with self.assertRaisesRegex(ValueError, '^primary_paper_endpoint_required$'):
                            sync.run_stage(sync.STAGES[1], 300, 123)
                        popen.assert_not_called()
                        capture.assert_not_called()
                        self.assertEqual(dict(os.environ), environment)

    def test_uses_existing_context_validator_once(self):
        from scripts import primary_paper_pipeline
        actual = primary_paper_pipeline.configure_paper_context
        with mock.patch.object(primary_paper_pipeline, 'configure_paper_context', wraps=actual) as validate:
            _, calls = self.invoke({'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets'})
        validate.assert_called_once()
        self.assertIs(validate.call_args.args[0], calls[0][1]['env'])

    def test_only_reconciliation_has_explicit_environment_copy(self):
        environment = {'APCA_API_BASE_URL': 'https://api.alpaca.markets', 'JBR_EXEC_PAPER': 'false'}
        for stage in (sync.STAGES[0], sync.STAGES[2]):
            with self.subTest(stage=stage.name):
                result, calls = self.invoke(environment, stage)
                self.assertEqual(result['status'], 'ok')
                self.assertNotIn('env', calls[0][1])

    def test_no_order_arguments_and_shell_behavior_unchanged(self):
        _, calls = self.invoke({'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets'})
        command, options = calls[0]
        stage = sync.STAGES[1]
        self.assertEqual(command, [sys.executable, '-m', stage.module, *stage.args])
        args = dict(zip(stage.args[::2], stage.args[1::2]))
        self.assertEqual(args['--reconcile-only'], 'true')
        self.assertEqual(args['--dry-run'], 'true')
        self.assertEqual(args['--submit-at-ny'], '')
        self.assertEqual(args['--reconcile-use-watermark'], 'true')
        self.assertFalse(options['shell'])
        self.assertEqual(options['cwd'], sync.ROOT)

    def test_configured_values_inherit_without_being_exported(self):
        environment = {'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets',
                       'APCA_API_KEY_ID': 'SYNTHETIC_KEY_ONLY',
                       'APCA_API_SECRET_KEY': 'SYNTHETIC_SECRET_ONLY'}
        result, calls = self.invoke(environment)
        child = calls[0][1]['env']
        self.assertEqual(child['APCA_API_KEY_ID'], environment['APCA_API_KEY_ID'])
        self.assertEqual(child['APCA_API_SECRET_KEY'], environment['APCA_API_SECRET_KEY'])
        self.assertNotIn('SYNTHETIC_KEY_ONLY', json.dumps(result))
        self.assertNotIn('SYNTHETIC_SECRET_ONLY', json.dumps(result))

    def test_rejected_context_stops_workflow_before_snapshot(self):
        with tempfile.TemporaryDirectory() as storage:
            calls = []
            def run(stage, timeout, descriptor):
                calls.append(stage.name)
                if stage.name == 'activities':
                    return successful_stage()
                return sync.run_stage(stage, timeout, descriptor)
            with mock.patch.dict(os.environ, {
                    'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets',
                    'JBR_EXEC_PAPER': 'false'}, clear=True), \
                    mock.patch.object(sync.subprocess, 'Popen') as popen:
                report = sync.run_workflow(Path(storage) / 'health', runner=run)
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(calls, ['activities', 'reconciliation'])
            self.assertEqual(report['stages'][1]['reason'], 'stage_exception')
            self.assertEqual(report['stages'][1]['error_type'], 'ValueError')
            self.assertEqual(report['stages'][2]['status'], 'not_run')
            popen.assert_not_called()

    def test_admission_does_not_hide_reconciliation_failure(self):
        stage = sync.STAGES[1]
        result, _ = self.invoke({'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets'},
                                text='\n'.join((*stage.required, 'RECONCILE_WATERMARK_HELD')))
        self.assertEqual(result['status'], 'failed')
        self.assertIn('RECONCILE_WATERMARK_HELD', result['failure_markers'])
        self.assertIn('RECONCILE_WATERMARK_HELD', result['health_reasons'])

    def test_spawn_failure_cannot_mutate_parent_context(self):
        environment = {'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets'}
        with mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch.object(sync.subprocess, 'Popen', side_effect=OSError()):
            with self.assertRaises(OSError):
                sync.run_stage(sync.STAGES[1], 300, 123)
            self.assertEqual(dict(os.environ), environment)

    def test_actual_policy_context_guard_accepts_only_the_declared_child(self):
        import ast
        tree = ast.parse((sync.ROOT / 'scripts/db.py').read_text(encoding='utf-8'))
        guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == '_paper_disposition_identity')
        binding = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == '_PAPER_DISPOSITION_BINDING'
                               for t in n.targets))
        namespace = {'os': os, 'disposition': SimpleNamespace(TradePolicyError=RuntimeError),
                     '_PAPER_DISPOSITION_BINDING': ast.literal_eval(binding),
                     'configure_disposition_identity': lambda *args, **kwargs: kwargs}
        exec(compile(ast.Module(body=[guard], type_ignores=[]), 'pure-policy-guard', 'exec'), namespace)
        environment = {'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets'}
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, '^TRADE_POLICY_PAPER_CONTEXT_REQUIRED$'):
                namespace['_paper_disposition_identity']()
        _, calls = self.invoke(environment)
        with mock.patch.dict(os.environ, calls[0][1]['env'], clear=True):
            accepted = namespace['_paper_disposition_identity']()
        self.assertEqual(accepted['environment'], 'paper')
        self.assertTrue(accepted['verified'])


def operational_output(drift=False, formatter=None):
    """Render the unchanged real executor publication tail with synthetic state."""
    import ast
    import copy
    source = ast.parse((sync.ROOT / 'scripts/execute_trades.py').read_text(encoding='utf-8'))
    executor = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'TradeExecutor')
    method = next(n for n in executor.body if isinstance(n, ast.FunctionDef) and n.name == 'reconcile_closed_trades')
    tail = copy.deepcopy(method.body[-1])
    assert isinstance(tail, ast.Try)
    wrapper = ast.parse('def publish(self, db, engine, policy_revision, policy_status, LOGGER, closed_count=0, decorated_count=0):\n    pass\n').body[0]
    wrapper.body = [tail]
    recorded_failures = []
    namespace = {'record_failure': lambda code, trade_id=None: recorded_failures.append((code, trade_id))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                 'source-exact-publication', 'exec'), namespace)
    status = dict(scope='ACTIVE_RECONCILIATION', policy_revision=0, excluded_count=1)
    @contextlib.contextmanager
    def publication(engine, *, expected_revision):
        assert expected_revision == 0
        yield {**status, 'policy_revision': 1} if drift else status
    metrics = SimpleNamespace(reconciliation=dict(recovery_pending_case_ids=[
        '00000000-0000-4000-8000-000000000001']))
    stream = io.StringIO()
    logger = logging.Logger('synthetic-exact-publication', logging.INFO)
    handler = logging.StreamHandler(stream)
    if formatter is not None:
        handler.setFormatter(logging.Formatter(formatter))
    logger.addHandler(handler)
    logger.info('[INFO] RECONCILE_START open_trades=0')
    result = namespace['publish'](SimpleNamespace(metrics=metrics),
        SimpleNamespace(reconciliation_publication=publication), object(), 0, status, logger)
    assert result is (not drift)
    assert recorded_failures == ([('POLICY_REVISION_CHANGED', None),
                                  ('POLICY_PUBLICATION_UNAVAILABLE', None)] if drift else [])
    return stream.getvalue()


class CompletionContractTests(unittest.TestCase):
    def invoke(self, text, code=0):
        child = mock.Mock(returncode=code)
        child.poll.return_value = code
        def spawn(command, **options):
            options['stdout'].write(text.encode())
            return child
        with mock.patch.dict(os.environ, {'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets'}, clear=True), \
                mock.patch.object(sync.subprocess, 'Popen', side_effect=spawn):
            return sync.run_stage(sync.STAGES[1], 300, 123)

    def test_real_scoped_publication_is_degraded_with_all_holds_explicit(self):
        text = operational_output()
        self.assertNotIn('RECONCILE_COVERAGE_COMPLETE', text)
        self.assertNotIn('RECONCILE_WATERMARK_UPDATE', text)
        result = self.invoke(text)
        self.assertEqual(result['status'], 'degraded')
        self.assertEqual(result['reason'], 'RECONCILE_RECOVERY_HELD')
        self.assertEqual(result['completion_contract'], 'ACTIVE_RECONCILIATION_OPERATIONAL_V1')
        self.assertEqual(result['failure_markers'], [])
        self.assertEqual(result['missing_completion_markers'], [])
        self.assertIn('RECONCILE_WATERMARK_HELD', result['health_reasons'])
        self.assertIn('RECONCILE_RECOVERY_HELD', result['qualifications'])
        scoped = result['reconciliation']
        self.assertIs(scoped['operational_complete'], True)
        self.assertEqual(scoped['scope'], 'ACTIVE_RECONCILIATION')
        self.assertEqual(scoped['policy_revision'], 0)
        self.assertEqual(scoped['excluded_count'], 1)
        self.assertEqual(scoped['recovery_pending_case_ids'], ['00000000-0000-4000-8000-000000000001'])
        for flag in ('recovery_complete', 'accounting_complete', 'activity_ingestion_complete',
                     'watermark_advanced', 'snapshot_eligible'):
            with self.subTest(flag=flag): self.assertIs(scoped[flag], False)

    def test_fatal_markers_never_become_qualified_success(self):
        text = operational_output()
        for marker in (*sync.STAGES[1].failures, *sync.GENERIC_FAILURES, 'RECONCILE_DECORATE_MISS',
                       'RECONCILE_FILL_HELD', 'RECONCILE_DECORATE_HELD',
                       'RECONCILE_INCOMPLETE', 'RECONCILE_AUTO_FAIL'):
            with self.subTest(marker=marker):
                result = self.invoke(text + '\n' + marker)
                self.assertEqual(result['status'], 'failed')
                self.assertNotIn('reconciliation', result)

    def test_nonzero_and_output_limit_preserve_failure(self):
        for code in (1, 2, -9):
            with self.subTest(code=code):
                self.assertEqual(self.invoke(operational_output(), code)['status'], 'failed')
        result = self.invoke(operational_output() + 'x' * sync.OUTPUT_LIMIT)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'output_limit')

    def test_timeout_after_valid_output_cannot_become_degraded_completion(self):
        child = mock.Mock(pid=4321, returncode=-9)
        child.poll.return_value = None
        def spawn(command, **options):
            options['stdout'].write(operational_output().encode())
            return child
        with mock.patch.dict(os.environ, {'APCA_API_BASE_URL': 'https://paper-api.alpaca.markets'}, clear=True), \
                mock.patch.object(sync.subprocess, 'Popen', side_effect=spawn), \
                mock.patch.object(sync.time, 'monotonic', side_effect=[0, 301, 302]), \
                mock.patch.object(sync.os, 'killpg', create=True):
            result = sync.run_stage(sync.STAGES[1], 300, 123)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'timeout')
        self.assertNotIn('reconciliation', result)
        child.wait.assert_called_once_with(timeout=5)

    def test_missing_duplicate_and_out_of_order_publications_are_refused(self):
        lines = operational_output().splitlines()
        for index in range(4):
            for mode in ('missing', 'duplicate'):
                with self.subTest(index=index, mode=mode):
                    changed = lines[:index] + lines[index + 1:] if mode == 'missing' else lines + [lines[index]]
                    self.assertEqual(self.invoke('\n'.join(changed))['status'], 'failed')
        with self.subTest(order='reversed'):
            self.assertEqual(self.invoke('\n'.join(reversed(lines)))['status'], 'failed')

    def test_overclaims_unknown_scope_reason_and_corrupt_fields_are_refused(self):
        text = operational_output()
        changes = [('scope=ACTIVE_RECONCILIATION', 'scope=GLOBAL'),
                   ('policy_revision=0', 'policy_revision=-1'),
                   ('policy_revision=0', 'policy_revision=True'),
                   ('policy_revision=0', 'policy_revision=' + '9' * 20),
                   ('excluded_count=1', 'excluded_count=-1'),
                   ('reason=independent_durable_activity_ingestion_unproved', 'reason=unknown'),
                   ('operational_complete=true', 'operational_complete=false'),
                   ('00000000-0000-4000-8000-000000000001', 'UNTRUSTED_SECRET_TEXT'),
                   ('open_trades=0', 'open_trades=0\x08'),
                   ('closed=0', 'closed=0 extra=unexpected')]
        for flag in ('recovery_complete', 'accounting_complete', 'activity_ingestion_complete', 'watermark_advanced'):
            changes.append((flag + '=false', flag + '=true'))
        for before, after in changes:
            with self.subTest(field=before):
                result = self.invoke(text.replace(before, after))
                self.assertEqual(result['status'], 'failed')
                self.assertNotIn('reconciliation', result)
                self.assertNotIn('UNTRUSTED_SECRET_TEXT', json.dumps(result))

    def test_duplicate_cases_and_contradictory_watermark_are_refused(self):
        text = operational_output()
        case = "'00000000-0000-4000-8000-000000000001'"
        variants = [text.replace('[' + case + ']', '[' + case + ', ' + case + ']'),
                    text + '\nRECONCILE_WATERMARK_UPDATE',
                    text.replace('[WARN] RECONCILE_WATERMARK_HELD', '[INFO] RECONCILE_WATERMARK_HELD'),
                    text.replace('RECONCILE_OPERATIONAL_COMPLETE', 'RECONCILE_OPERATIONAL_COMPLETE_EXTRA')]
        for index, changed in enumerate(variants):
            with self.subTest(index=index):
                self.assertEqual(self.invoke(changed)['status'], 'failed')

    def test_legacy_markers_and_zero_exit_do_not_claim_current_readiness(self):
        result = self.invoke('\n'.join(sync.STAGES[1].required))
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'RECONCILE_SCOPED_RESULT_UNPROVED')

    def test_real_policy_publication_drift_remains_failed(self):
        result = self.invoke(operational_output(drift=True), code=1)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('RECONCILE_DB_FAIL', result['failure_markers'])
        self.assertNotIn('reconciliation', result)

    def test_degraded_report_keeps_nonzero_cli_and_never_runs_snapshot(self):
        stage_result = self.invoke(operational_output())
        visited = []
        def runner(stage, timeout, lock):
            visited.append(stage.name)
            return stage_result if stage.name == 'reconciliation' else successful_stage()
        with tempfile.TemporaryDirectory() as storage:
            state = Path(storage) / 'reports'
            report = sync.run_workflow(state, runner=runner)
            self.assertEqual(json.loads((state / 'latest.json').read_text()), report)
            self.assertEqual(json.loads((state / ('run-' + report['run_id'] + '.json')).read_text()), report)
        self.assertEqual(visited, ['activities', 'reconciliation'])
        self.assertEqual(report['status'], 'degraded')
        self.assertEqual(report['stages'][2]['status'], 'not_run')
        with mock.patch.object(sync, 'run_workflow', return_value=report), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sync.main([]), 1)



class PublicationUniquenessTests(unittest.TestCase):
    invoke = CompletionContractTests.invoke

    def assert_failed_capture(self, text):
        result = self.invoke(text)
        self.assertEqual(result['status'], 'failed')
        self.assertNotIn('reconciliation', result)
        self.assertNotIn('completion_contract', result)
        return result

    def test_identical_same_line_duplicates_of_every_marker_are_refused(self):
        lines = operational_output().splitlines()
        for index in range(4):
            for separator in (' ', '\t', '', ' | '):
                with self.subTest(marker=index, separator=repr(separator)):
                    changed = list(lines)
                    changed[index] += separator + lines[index]
                    self.assert_failed_capture('\n'.join(changed))

    def test_different_publications_cannot_share_one_physical_line(self):
        lines = operational_output().splitlines()
        for first in range(4):
            for second in range(first + 1, 4):
                for reverse in (False, True):
                    with self.subTest(first=first, second=second, reverse=reverse):
                        changed = list(lines)
                        pair = [lines[first], lines[second]]
                        changed[first] = ' '.join(reversed(pair) if reverse else pair)
                        del changed[second]
                        self.assert_failed_capture('\n'.join(changed))

    def test_conflicting_publication_before_or_after_valid_one_is_refused(self):
        lines = operational_output().splitlines()
        invalid = [lines[0].replace('open_trades=0', 'open_trades=-1'),
                   lines[1].replace('scope=ACTIVE_RECONCILIATION', 'scope=GLOBAL'),
                   lines[2].replace('independent_durable_activity_ingestion_unproved', 'unknown'),
                   lines[3].replace('accounting_complete=false', 'accounting_complete=true')]
        for index in range(4):
            for invalid_first in (False, True):
                with self.subTest(marker=index, invalid_first=invalid_first):
                    changed = list(lines)
                    changed[index] = ' '.join([invalid[index], lines[index]] if invalid_first
                                              else [lines[index], invalid[index]])
                    self.assert_failed_capture('\n'.join(changed))

    def test_only_actual_source_logging_formats_have_accepted_prefixes(self):
        import ast
        source = ast.parse((sync.ROOT / 'scripts/execute_trades.py').read_text(encoding='utf-8'))
        configure = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_logging')
        formats = {n.args[0].value for n in ast.walk(configure) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == 'Formatter'
                   and n.args and isinstance(n.args[0], ast.Constant)}
        self.assertEqual(formats, {'%(message)s', '[%(asctime)s] [%(levelname)s] %(message)s'})
        for formatter in sorted(formats):
            with self.subTest(formatter=formatter):
                result = self.invoke(operational_output(formatter=formatter))
                self.assertEqual(result['status'], 'degraded')
                self.assertIs(result['reconciliation']['snapshot_eligible'], False)
                self.assertIs(result['reconciliation']['accounting_complete'], False)
        lines = operational_output().splitlines()
        prefixes = ('untrusted arbitrary text ', '[INFO] ',
                    '[2026-10-22 20:00:00,123] [ERROR] ',
                    '[2026-10-22 20:00:00,123] [INFO] untrusted ',
                    '[2026-99-99 99:99:99,999] [INFO] ')
        for index in range(4):
            for prefix in prefixes:
                with self.subTest(marker=index, prefix=prefix):
                    changed = list(lines)
                    changed[index] = prefix + lines[index]
                    self.assert_failed_capture('\n'.join(changed))

    def test_exact_independent_counterexample_fails_without_external_snapshot(self):
        lines = operational_output().splitlines()
        prefix = ('[INFO] RECONCILE_OPERATIONAL_COMPLETE scope=GLOBAL policy_revision=0 '
                  'excluded_count=0 recovery_pending_case_ids=[] ')
        lines[1] = prefix + lines[1]
        text = '\n'.join(lines)
        self.assertIsNone(sync._scoped_reconciliation_result(text))
        result = self.assert_failed_capture(text)
        visited = []
        def runner(stage, timeout, lock):
            visited.append(stage.name)
            return result if stage.name == 'reconciliation' else successful_stage()
        with tempfile.TemporaryDirectory() as storage:
            report = sync.run_workflow(Path(storage) / 'reports', runner=runner)
        self.assertEqual(visited, ['activities', 'reconciliation'])
        self.assertEqual(report['stages'][2]['status'], 'not_run')
        self.assertEqual(report['status'], 'failed')
        with mock.patch.object(sync, 'run_workflow', return_value=report), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sync.main([]), 1)



if __name__ == '__main__':
    unittest.main()
