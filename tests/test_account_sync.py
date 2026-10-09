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
        with mock.patch.object(sync.subprocess, 'Popen', side_effect=create) as popen:
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
                self.assertEqual(self.invoke(stage, '\n'.join(stage.required))['status'], 'ok')

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

    def test_unresolved_decoration_is_degraded(self):
        stage = sync.STAGES[1]
        result = self.invoke(stage, '\n'.join((*stage.required, 'RECONCILE_DECORATE_MISS')))
        self.assertEqual(result['status'], 'degraded')

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
        self.stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))

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

    def exercise_clock(self, when, session, next_open_days=0):
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
        execution.execute.side_effect = self.forbidden
        execution.execute_order.side_effect = self.forbidden
        execution.hydrate_candidates.side_effect = self.forbidden
        messages = io.StringIO()
        logger = logging.Logger('offline-executor-gates', logging.INFO)
        logger.addHandler(logging.StreamHandler(messages))
        execution.reconcile_closed_trades.side_effect = lambda: logger.info(
            'RECONCILE_START\nRECONCILE_END\nRECONCILE_WATERMARK_UPDATE')
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
                self.assertEqual(code, 0)
                wait.assert_called_once_with('')
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
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(calls, ['activities', 'reconciliation', 'snapshot'])
        execution.reconcile_closed_trades.assert_called_once_with()
        execution.execute.assert_not_called()
        execution.hydrate_candidates.assert_not_called()
        client.submit_order.assert_not_called()
        self.assertEqual(result['stages'][2]['status'], 'ok')
        return messages.getvalue()

    def test_before_0700_et_reconciles_without_waiting(self):
        self.exercise_clock('2026-10-09T06:15:00', 'premarket')

    def test_weekend_closed_session_reconciles_and_reaches_snapshot(self):
        text = self.exercise_clock('2026-10-10T06:15:00', 'closed', 2)
        self.assertIn('MARKET_GATE_BYPASSED', text)

    def test_weekday_holiday_closed_session_reconciles(self):
        text = self.exercise_clock('2026-12-25T06:15:00', 'holiday', 3)
        self.assertIn('MARKET_GATE_BYPASSED', text)


if __name__ == '__main__':
    unittest.main()
