"""Existing caller regressions updated for strict reads and the atomic fill API.

Normal executor imports use the shared pre-import offline boundary fixture.
These tests run under unittest and pytest; pytest's credential skip remains
disabled by alpaca_optional because no authenticated operation is performed.
"""
import unittest

import pytest

import test_account_sync as hourly_tests
import test_reconciliation_integrity as integrity_tests


@pytest.mark.alpaca_optional
class ReconcileTradesTests(unittest.TestCase):
    setUp = hourly_tests.ExecutorGateIntegrationTests.setUp
    prepare = integrity_tests.ReconciliationTests.prepare

    def test_reconcile_closes_open_trades(self):
        trade = dict(trade_id=1, symbol='XYZ', qty=5, entry_time=integrity_tests.NOW,
                     entry_price=10.0, entry_order_id='open-1')
        self.prepare(open_trades=[trade], missing=[trade], positions=[],
                     orders=[integrity_tests.order(filled_qty='5', filled_avg_price='11.25',
                                   type='trailing_stop')])
        self.assertTrue(self.run.reconcile_closed_trades())
        self.writes['get_open_trades'].assert_called_once_with(
            self.writes['get_db_conn'].return_value, strict=True)
        self.assertTrue(self.writes['get_closed_trades_missing_exit'].call_args.kwargs['strict'])
        closed = self.writes['close_trade'].call_args.kwargs
        self.assertEqual(closed['trade_id'], 1)
        self.assertIsNone(closed['exit_order_id'])
        self.assertIsNone(closed['exit_price'])
        self.assertEqual(closed['exit_reason'], 'POSITION_CLOSED')
        self.writes['reconcile_sell_fill'].assert_called_once_with(
            engine=self.writes['get_db_conn'].return_value, trade_id=1, symbol='XYZ',
            qty='5', order_id='sell-1', status='filled', event_time=integrity_tests.NOW,
            raw=dict(id='sell-1', symbol='XYZ', side='sell', status='filled',
                     type='trailing_stop', filled_qty='5', filled_avg_price='11.25'),
            exit_price=11.25, exit_reason='TRAIL_STOP', decorate=True)
        self.writes['insert_order_event'].assert_not_called()
        self.writes['decorate_trade_exit'].assert_not_called()
        self.writes['set_reconcile_state'].assert_called_once()

    def test_reconcile_decorates_without_open_trades(self):
        trade = dict(trade_id=2, symbol='TSLA', qty=1, entry_price=100.0,
                     exit_price=None, exit_order_id=None, realized_pnl=None)
        fill = integrity_tests.order(id='sell-2', symbol='TSLA', filled_qty='1', filled_avg_price='105.50')
        self.prepare(open_trades=[], missing=[trade], orders=[fill])
        self.assertTrue(self.run.reconcile_closed_trades())
        self.writes['close_trade'].assert_not_called()
        self.assertTrue(self.writes['get_closed_trades_missing_exit'].call_args.kwargs['strict'])
        self.writes['reconcile_sell_fill'].assert_called_once_with(
            engine=self.writes['get_db_conn'].return_value, trade_id=2, symbol='TSLA',
            qty='1', order_id='sell-2', status='filled', event_time=integrity_tests.NOW,
            raw=dict(id='sell-2', symbol='TSLA', side='sell', status='filled',
                     type='market', filled_qty='1', filled_avg_price='105.50'),
            exit_price=105.50, exit_reason='SELL_FILL', decorate=True)
        self.writes['insert_order_event'].assert_not_called()
        self.writes['decorate_trade_exit'].assert_not_called()
        self.writes['set_reconcile_state'].assert_called_once()


if __name__ == '__main__':
    unittest.main()
