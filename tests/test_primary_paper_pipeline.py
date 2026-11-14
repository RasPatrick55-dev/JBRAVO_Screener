"""Synthetic context/normal-caller checks; no credentials, DB, provider or jobs."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import unittest
from unittest import mock

from scripts import primary_paper_pipeline as paper
from scripts import primary_pipeline as primary
from scripts import pipeline_postflight as postflight

try:
    import pytest
except ImportError:
    pass  # Keep standalone unittest execution available without pytest.
else:
    pytestmark = pytest.mark.alpaca_optional


ROOT = Path(__file__).resolve().parents[1]


class PolicyRefusal(ValueError):
    pass


def actual_context_gate():
    """Execute the pinned pure identity gate, without importing db.py."""
    tree = ast.parse((ROOT / 'scripts/db.py').read_text(encoding='utf-8'))
    selected = [node for node in tree.body if
        (isinstance(node, ast.FunctionDef) and node.name == '_paper_disposition_identity') or
        (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id == '_PAPER_DISPOSITION_BINDING' for t in node.targets))]
    assert len(selected) == 2
    namespace = {'os': os, 'disposition': mock.Mock(TradePolicyError=PolicyRefusal),
                 'configure_disposition_identity': mock.Mock(return_value='validated-context')}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'pure-context-gate', 'exec'), namespace)
    return namespace


class PrimaryPaperTests(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        # A mistaken provider/process call must fail rather than reach a target.
        for target in ('socket.socket', 'socket.create_connection', 'subprocess.Popen'):
            guard = mock.patch(target, side_effect=AssertionError('external effect forbidden'))
            guard.start()
            self.addCleanup(guard.stop)

    def call(self, argv=None):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = paper.main(argv)
        return result, output.getvalue()

    def test_missing_declaration_reproduces_observed_consumer_refusal(self):
        os.environ['APCA_API_BASE_URL'] = paper.PAPER_ENDPOINT
        gate = actual_context_gate()
        with self.assertRaisesRegex(PolicyRefusal, 'PAPER_CONTEXT_REQUIRED'):
            gate['_paper_disposition_identity']()
        with mock.patch.object(primary, 'main', side_effect=lambda argv:
                0 if gate['_paper_disposition_identity']() == 'validated-context' else 1) as run:
            self.assertEqual(self.call([])[0], 0)
        run.assert_called_once_with([])
        gate['configure_disposition_identity'].assert_called_once_with(
            *gate['_PAPER_DISPOSITION_BINDING'][:2], environment='paper',
            evidence_ref=gate['_PAPER_DISPOSITION_BINDING'][2], verified=True)
        self.assertEqual(os.environ['JBR_EXEC_PAPER'], 'true')

    def test_endpoint_refusals_do_not_delegate_or_change_environment(self):
        for endpoint in (None, '', 'https://api.alpaca.markets',
                paper.PAPER_ENDPOINT + '/', 'http://paper-api.alpaca.markets',
                'https://paper-api.alpaca.markets.evil.invalid',
                'https://user:secret@paper-api.alpaca.markets'):
            with self.subTest(endpoint=endpoint):
                os.environ.clear()
                if endpoint is not None:
                    os.environ['APCA_API_BASE_URL'] = endpoint
                before = dict(os.environ)
                with mock.patch.object(primary, 'main') as run:
                    rc, output = self.call(['--run-date', '2026-10-20'])
                self.assertEqual(rc, 1)
                self.assertEqual(json.loads(output)['reason'], 'primary_paper_endpoint_required')
                self.assertNotIn('secret', output)
                self.assertEqual(dict(os.environ), before)
                run.assert_not_called()

    def test_conflicting_declarations_refuse_without_overwriting_them(self):
        for declaration in ('false', '', 'TRUE', '1', ' true '):
            with self.subTest(declaration=declaration):
                os.environ.clear()
                os.environ.update(APCA_API_BASE_URL=paper.PAPER_ENDPOINT,
                                  JBR_EXEC_PAPER=declaration)
                before = dict(os.environ)
                with mock.patch.object(primary, 'main') as run:
                    rc, output = self.call()
                self.assertEqual(rc, 1)
                self.assertEqual(json.loads(output)['reason'], 'primary_paper_context_conflict')
                self.assertEqual(dict(os.environ), before)
                run.assert_not_called()

    def test_existing_paper_context_arguments_and_exit_status_are_preserved(self):
        for rc in (0, 1, 2):
            with self.subTest(rc=rc):
                os.environ.clear()
                os.environ.update(APCA_API_BASE_URL=paper.PAPER_ENDPOINT, JBR_EXEC_PAPER='true',
                                  JBR_SENTINEL='unchanged')
                before = dict(os.environ)
                argv = ['--run-date', '2026-10-20', '--reload-web', 'false']
                with mock.patch.object(primary, 'main', return_value=rc) as run:
                    self.assertEqual(self.call(argv)[0], rc)
                run.assert_called_once_with(argv)
                self.assertEqual(dict(os.environ), before)
                self.assertEqual(argv, ['--run-date', '2026-10-20', '--reload-web', 'false'])

    def test_none_arguments_keep_normal_sys_argv_contract(self):
        os.environ['APCA_API_BASE_URL'] = paper.PAPER_ENDPOINT
        with mock.patch.object(primary, 'main', return_value=0) as run:
            self.assertEqual(self.call()[0], 0)
        run.assert_called_once_with(None)

    def test_real_primary_caller_observes_context_before_invocation_and_pipeline(self):
        os.environ['APCA_API_BASE_URL'] = paper.PAPER_ENDPOINT
        seen = []
        original_main = primary.main
        def begin(*args):
            self.assertEqual(os.environ['JBR_EXEC_PAPER'], 'true')
            seen.append('begin')
            return {'id': 'synthetic-invocation'}
        def pipeline(argv):
            gate = actual_context_gate()
            self.assertEqual(gate['_paper_disposition_identity'](), 'validated-context')
            self.assertIn('--pipeline-readiness-only', argv)
            self.assertIn('--session-handoff', argv)
            self.assertEqual(argv[-2:], ['--handoff-invocation', 'synthetic-invocation'])
            self.assertIn('2026-10-20', argv)
            seen.append('pipeline')
            return 0
        with mock.patch.object(postflight, 'begin_primary', side_effect=begin), \
             mock.patch.object(postflight, 'check_health', return_value={'status': 'ok'}), \
             mock.patch.object(postflight, 'complete_primary') as complete, \
             mock.patch.object(primary, 'main', side_effect=lambda argv:
                 original_main(argv, pipeline_main=pipeline, base_dir=ROOT)):
            self.assertEqual(self.call(['--run-date', '2026-10-20'])[0], 0)
        self.assertEqual(seen, ['begin', 'pipeline'])
        complete.assert_called_once()

    def test_pipeline_policy_or_parity_failure_stays_failed(self):
        for outcome in (2, PolicyRefusal('synthetic-policy-refusal')):
            with self.subTest(outcome=type(outcome).__name__):
                os.environ.clear()
                os.environ['APCA_API_BASE_URL'] = paper.PAPER_ENDPOINT
                original_main = primary.main
                pipeline = mock.Mock(side_effect=outcome) if isinstance(outcome, Exception) \
                    else mock.Mock(return_value=outcome)
                with mock.patch.object(postflight, 'begin_primary', return_value={'id': 'fixture'}), \
                     mock.patch.object(postflight, 'check_health') as health, \
                     mock.patch.object(postflight, 'complete_primary') as complete, \
                     mock.patch.object(primary, 'main', side_effect=lambda argv:
                         original_main(argv, pipeline_main=pipeline, base_dir=ROOT)):
                    self.assertEqual(self.call([])[0], 1)
                health.assert_not_called()
                complete.assert_not_called()

    def test_fixed_primary_override_still_refuses_before_begin(self):
        os.environ['APCA_API_BASE_URL'] = paper.PAPER_ENDPOINT
        original_main = primary.main
        with mock.patch.object(postflight, 'begin_primary') as begin, \
             mock.patch.object(primary, 'main', side_effect=lambda argv:
                 original_main(argv, pipeline_main=mock.Mock(), base_dir=ROOT)):
            self.assertEqual(self.call(['--step', 'metrics'])[0], 1)
        begin.assert_not_called()

    def test_launcher_and_postflight_bind_the_new_entrypoint(self):
        launcher = (ROOT / 'scripts/ops/run_primary_pipeline.sh').read_text(encoding='utf-8')
        self.assertIn('flock -n 9', launcher)
        self.assertIn('. "${ENV_FILE}"', launcher)
        self.assertIn('exec python -B -m scripts.primary_paper_pipeline "$@"', launcher)
        self.assertLess(launcher.index('. "${ENV_FILE}"'), launcher.index('exec python -B'))
        self.assertIn('scripts/primary_paper_pipeline.py', postflight.SOURCE_FILES)


if __name__ == '__main__':
    unittest.main()
