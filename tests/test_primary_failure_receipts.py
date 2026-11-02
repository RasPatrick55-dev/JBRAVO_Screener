"""Terminal primary failures survive an older success; no provider/DB/broker I/O."""
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import pipeline_postflight as health, primary_pipeline as primary

try:
    import pytest
except ImportError:
    pass
else:
    pytestmark = pytest.mark.alpaca_optional


class PrimaryFailureReceipts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root/'reports/pipeline_postflight'
        self.directory.mkdir(parents=True)
        self.old = b'{"old": "stage-report-must-stay-byte-identical"}\n'
        (self.directory/'latest.json').write_bytes(self.old)
        for target in ('socket.socket.connect', 'socket.create_connection', 'subprocess.Popen'):
            guard = patch(target, side_effect=AssertionError('External effect forbidden'))
            guard.start()
            self.addCleanup(guard.stop)
        guard = patch.dict(os.environ, {}, clear=True)
        guard.start()
        self.addCleanup(guard.stop)

    def run_primary(self, pipeline, **patches):
        with contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(patch.object(health, name, **value))
            output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            rc = primary.main([], pipeline_main=pipeline, base_dir=self.root)
        return rc, json.loads(output.getvalue())

    def failure(self):
        active = json.loads((self.directory/'active-primary.json').read_bytes())
        return json.loads((self.directory/('failure-'+active['id']+'.json')).read_bytes())

    def test_missing_archive_has_fresh_bound_failure_without_overwriting_stage_report(self):
        pipeline = Mock(side_effect=ValueError('strategy_scan_not_archived_for_session'))
        rc, output = self.run_primary(pipeline)
        self.assertEqual(rc, 1)
        pipeline.assert_called_once()
        receipt = self.failure()
        self.assertEqual(receipt['code'], 'strategy_scan_not_archived_for_session')
        self.assertEqual(receipt['phase'], 'PIPELINE')
        self.assertEqual(receipt['reason'], 'pipeline_exception')
        self.assertEqual(health.current_primary_failure(self.root), receipt)
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        checked = health.check_health(self.root)
        self.assertEqual(checked['status'], 'failed')
        self.assertEqual(checked['primary_failure'], receipt)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})
        self.assertEqual((self.directory/'latest.json').read_bytes(), self.old)
        self.assertFalse((self.directory/'completed-primary.json').exists())

    def test_exception_messages_and_unknown_classes_are_withheld(self):
        class PrivateException(RuntimeError):
            pass
        for error in (ValueError('synthetic private error detail'),
                      PrivateException('private account or SQL query')):
            rc, output = self.run_primary(Mock(side_effect=error))
            text = json.dumps(output)+json.dumps(self.failure())
            self.assertEqual(rc, 1)
            for secret in ('synthetic private error detail', 'private account', 'PrivateException'):
                self.assertNotIn(secret, text)
            self.assertIsNone(self.failure()['code'])

    def test_nonzero_unknown_and_system_exit_results_stay_failed(self):
        for outcome in (1, 2, -9, None, True, 'secret exit text', SystemExit('secret exit text')):
            pipeline = Mock(side_effect=outcome) if isinstance(outcome, BaseException) else Mock(return_value=outcome)
            rc, output = self.run_primary(pipeline)
            self.assertEqual(rc, 1)
            self.assertEqual(self.failure()['reason'], 'pipeline_unsuccessful')
            self.assertNotIn('secret exit text', json.dumps(output))
            self.assertEqual(self.failure()['pipeline_rc'], outcome if type(outcome) is int else None)

    def test_postflight_failures_and_unexpected_exception_get_terminal_receipts(self):
        for behavior in ({'return_value': {'status': 'failed', 'failures': ['test']}},
                         {'side_effect': RuntimeError('private postflight detail')}):
            rc, _ = self.run_primary(Mock(return_value=0), check_health=behavior)
            self.assertEqual(rc, 1)
            self.assertEqual(self.failure()['phase'], 'POSTFLIGHT')
            self.assertFalse((self.directory/'completed-primary.json').exists())

    def test_completion_write_failure_stays_failed(self):
        rc, _ = self.run_primary(Mock(return_value=0),
            check_health={'return_value': {'status': 'ok'}},
            complete_primary={'side_effect': OSError('private filesystem message')})
        self.assertEqual(rc, 1)
        self.assertEqual(self.failure()['phase'], 'COMPLETION')
        self.assertEqual(self.failure()['reason'], 'postflight_completion_record_failed')

    def test_recording_failure_does_not_turn_pipeline_successful(self):
        rc, output = self.run_primary(Mock(side_effect=RuntimeError('private')),
            fail_primary={'side_effect': OSError('private recording detail')})
        self.assertEqual(rc, 1)
        self.assertEqual(output['failure_receipt_status'], 'FAILED_TO_RECORD')
        self.assertNotIn('private', json.dumps(output))

    def test_previous_failure_does_not_poison_a_new_invocation(self):
        self.run_primary(Mock(return_value=1))
        old = self.failure()
        health.begin_primary(self.root, datetime.now(timezone.utc))
        self.assertIsNone(health.current_primary_failure(self.root))
        self.assertTrue((self.directory/('failure-'+old['run_id']+'.json')).exists())
        rc, _ = self.run_primary(Mock(return_value=0),
            check_health={'return_value': {'status': 'ok'}}, complete_primary={'return_value': None})
        self.assertEqual(rc, 0)

    def test_changed_failure_history_fails_closed(self):
        self.run_primary(Mock(return_value=1))
        record = self.failure()
        path = self.directory/('failure-'+record['run_id']+'.json')
        path.write_bytes(path.read_bytes()+b' ')
        self.assertEqual(health.check_health(self.root)['failures'], ['primary_failure_binding_mismatch'])

    def test_exclusive_history_and_active_identity_refuse_replay(self):
        invocation = health.begin_primary(self.root, datetime.now(timezone.utc))
        health.fail_primary(self.root, invocation, phase='PIPELINE', reason='pipeline_unsuccessful')
        with self.assertRaises(FileExistsError):
            health.fail_primary(self.root, invocation, phase='PIPELINE', reason='pipeline_unsuccessful')
        health.begin_primary(self.root, datetime.now(timezone.utc))
        with self.assertRaisesRegex(ValueError, 'primary_failure_invocation_mismatch'):
            health.fail_primary(self.root, invocation, phase='PIPELINE', reason='pipeline_unsuccessful')


if __name__ == '__main__':
    unittest.main()
