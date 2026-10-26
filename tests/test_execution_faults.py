"""Actual executor/monitor import paths with inert broker clients and state."""
from types import SimpleNamespace
import unittest
from unittest import mock

import pytest
from tests import test_executor_preflight_health as executor_fixtures
from tests import test_monitor_task_log_regressions as monitor_fixtures

pytestmark = pytest.mark.alpaca_optional


class SubmissionFaults(unittest.TestCase):
    def setUp(self):
        self.fixture = executor_fixtures.CallerTests('test_direct_buy_cannot_bypass_preflight_and_sell_is_unchanged')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.executor = self.fixture.entry_executor(SimpleNamespace(status='active', tradable=True))

    def test_timeout_cannot_repeat_a_post_or_continue_to_another_buy(self):
        with mock.patch.object(self.executor, '_submit_order', side_effect=TimeoutError('lost accepted response')) as submit, \
             mock.patch.object(self.executor, 'sleep') as sleep:
            request = SimpleNamespace(side='buy',symbol='AAA',limit_price=10.)
            self.assertIsNone(self.executor.submit_with_retries(request))
            self.assertIsNone(self.executor.submit_with_retries(request))
        submit.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(self.executor.metrics.exit_reason,'ORDER_SUBMISSION_OUTCOME_UNKNOWN')
        self.assertEqual(self.executor.metrics.api_failures,1)
        self.assertEqual(self.executor.metrics.api_retries,0)

    def test_server_error_is_ambiguous_and_does_not_retry(self):
        error=self.fixture.mod.requests.HTTPError('upstream failure')
        error.response=SimpleNamespace(status_code=503,text='upstream failure')
        with mock.patch.object(self.executor,'_submit_order',side_effect=error) as submit:
            self.assertIsNone(self.executor.submit_with_retries(SimpleNamespace(side='buy',symbol='AAA',limit_price=10.)))
        submit.assert_called_once()
        self.assertTrue(self.executor.config._order_submission_unknown)


class PositionSnapshotFaults(unittest.TestCase):
    def setUp(self):
        self.fixture=monitor_fixtures.MonitorRegressions('test_verified_exit_order_is_not_appended_twice')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_failed_broker_read_cannot_record_flat_or_closed_positions(self):
        monitor=self.fixture.monitor
        self.fixture.client.get_all_positions.side_effect=TimeoutError('snapshot unavailable')
        with mock.patch.object(monitor,'save_positions_csv') as save, \
             mock.patch.object(monitor,'log_closed_positions') as closed, \
             mock.patch.object(monitor,'refresh_ml_risk_state',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'broker_position_snapshot_unavailable'):
                monitor.process_positions_cycle()
        save.assert_not_called();closed.assert_not_called()

    def test_confirmed_empty_snapshot_is_allowed_but_malformed_snapshot_is_not(self):
        self.fixture.client.get_all_positions.return_value=[]
        self.assertEqual(self.fixture.monitor.get_open_positions(),[])
        self.fixture.client.get_all_positions.return_value=None
        with self.assertRaisesRegex(RuntimeError,'broker_position_snapshot_unavailable'):
            self.fixture.monitor.get_open_positions()
