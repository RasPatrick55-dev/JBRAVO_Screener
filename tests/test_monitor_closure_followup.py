"""Synthetic closure accounting against the normal inert monitor import."""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd
from tests import test_monitor_task_log_regressions as fixtures


class ClosureAccountingTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.MonitorRegressions('test_verified_exit_order_is_not_appended_twice')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.monitor = self.f.monitor
        self.path = Path(self.monitor.BASE_DIR) / 'data/open_positions.csv'
        self.export = Path(self.monitor.trades_log_real_path)
        columns = ['symbol', 'qty', 'entry_price', 'exit_price', 'entry_time', 'exit_time',
                   'net_pnl', 'order_status', 'order_type', 'exit_reason', 'side', 'order_id']
        pd.DataFrame(columns=columns).to_csv(self.export, index=False)
        self.f.frame.to_csv(self.path, index=False)
        self.f.logged.side_effect = self.f.real_trade_exit
        self.monitor.PARTIAL_EXIT_TAKEN.clear()

    def rows(self):
        return pd.read_csv(self.export)

    def order(self, **changes):
        return self.f.order(**({'symbol': 'AAA'} | changes))

    def cycle(self, *, partial=False, verified=True, quantity=10, basis=100,
              created_at=None, orders=(), prior=None, refresh_error=False):
        if prior is not None:
            prior.to_csv(self.path, index=False)
        self.f.position.qty = quantity
        self.f.position.avg_entry_price = basis
        self.f.position.current_price = max(120, basis * 1.1)
        self.f.position.created_at = created_at
        self.f.client.get_orders.return_value = list(orders)
        real_save = self.monitor.save_positions_csv
        half = max(1, int(float(quantity) / 2))

        def submit(position, **kwargs):
            if not partial:
                return False
            self.assertEqual(kwargs['qty_override'], half)
            self.monitor.log_trade_exit('AAA', half, basis, '110' if verified else '',
                self.f.entry.isoformat(), (self.f.entry + timedelta(days=1)).isoformat() if verified else '',
                'filled' if verified else 'filled_unverified', 'limit', 'synthetic partial', 'sell',
                order_id='partial')
            if 'fill_receipt' in kwargs:
                kwargs['fill_receipt'].update(order_id='partial', verified=verified,
                    qty=float(half) if verified else None)
            return True

        with self.f.cycle_patches(filled=partial):
            self.monitor.save_positions_csv.side_effect = real_save
            self.monitor.submit_sell_market_order.side_effect = submit
            if refresh_error:
                self.f.client.get_all_positions.side_effect=RuntimeError('synthetic_refresh_failure')
                with self.assertRaisesRegex(RuntimeError,'synthetic_refresh_failure'):
                    self.monitor.process_positions_cycle()
            else:
                self.f.client.get_all_positions.return_value = []
                self.assertEqual(self.monitor.process_positions_cycle(), [])
        return self.rows()

    def test_changed_basis_does_not_reuse_prior_price_or_timestamp(self):
        rows = self.cycle(basis=105, orders=[self.order(id='recent', filled_avg_price='120',
                         filled_at=self.f.entry + timedelta(days=10))])
        self.assertEqual(rows.iloc[0]['entry_price'], 105)
        self.assertTrue(pd.isna(rows.iloc[0]['entry_time']))
        self.assertEqual(rows.iloc[0]['order_status'], 'closed_unverified')
        self.assertTrue(pd.isna(rows.iloc[0]['net_pnl']))

    def test_unchanged_basis_retains_entry_and_matches_verified_close(self):
        rows = self.cycle(orders=[self.order(id='recent', filled_avg_price='120')])
        self.assertEqual(rows.iloc[0]['entry_price'], 100)
        self.assertEqual(rows.iloc[0]['entry_time'], self.f.entry.isoformat())
        self.assertEqual(rows.iloc[0]['net_pnl'], 200)
        self.assertEqual(rows.iloc[0]['order_id'], 'recent')

    def test_changed_basis_uses_current_known_time_and_excludes_previous_holding(self):
        current_entry = self.f.entry + timedelta(days=5)
        rows = self.cycle(basis=105, created_at=current_entry,
                          orders=[self.order(id='old-holding')])
        self.assertEqual(rows.iloc[0]['entry_price'], 105)
        self.assertEqual(rows.iloc[0]['entry_time'], current_entry.isoformat())
        self.assertEqual(rows.iloc[0]['order_status'], 'closed_unverified')
        self.assertTrue(pd.isna(rows.iloc[0]['order_id']))

    def test_changed_basis_with_current_fill_uses_current_cost(self):
        current_entry = self.f.entry + timedelta(days=5)
        rows = self.cycle(basis=105, created_at=current_entry,
            orders=[self.order(id='old-holding'), self.order(id='current', filled_avg_price='120',
                    filled_at=current_entry + timedelta(days=1))])
        self.assertEqual(rows.iloc[0]['net_pnl'], 150)
        self.assertEqual(rows.iloc[0]['order_id'], 'current')

    def test_partial_then_residual_exports_only_remaining_shares(self):
        rows = self.cycle(partial=True, orders=[self.order(id='partial', filled_qty='5'),
            self.order(id='residual', filled_qty='5', filled_avg_price='112',
                       filled_at=self.f.entry + timedelta(days=2))])
        self.assertEqual(rows['qty'].tolist(), [5, 5])
        self.assertEqual(rows['order_id'].tolist(), ['partial', 'residual'])
        self.assertEqual(rows['net_pnl'].tolist(), [50, 60])
        self.assertEqual(rows['qty'].sum(), 10)
        self.assertNotIn('AAA', self.monitor.PARTIAL_EXIT_TAKEN)

    def test_partial_fill_is_never_reused_as_residual_when_history_is_reordered(self):
        rows = self.cycle(partial=True, orders=[
            self.order(id='residual', filled_qty='5', filled_avg_price='112'),
            self.order(id='partial', filled_qty='5', filled_at=self.f.entry + timedelta(days=2))])
        self.assertEqual(rows['order_id'].tolist(), ['partial', 'residual'])
        self.assertEqual(rows['qty'].tolist(), [5, 5])

    def test_missing_residual_fill_preserves_only_unverified_remaining_quantity(self):
        rows = self.cycle(partial=True, orders=[self.order(id='partial', filled_qty='5')])
        self.assertEqual(rows['qty'].tolist(), [5, 5])
        self.assertEqual(rows.iloc[1]['order_status'], 'closed_unverified')
        self.assertTrue(pd.isna(rows.iloc[1]['order_id']))
        self.assertTrue(pd.isna(rows.iloc[1]['net_pnl']))

    def test_unverified_partial_does_not_invent_residual_quantity_or_fill(self):
        rows = self.cycle(partial=True, verified=False,
                          orders=[self.order(id='partial', filled_qty='5')])
        self.assertTrue(pd.isna(rows.iloc[1]['qty']))
        self.assertEqual(rows.iloc[1]['order_status'], 'closed_unverified')
        self.assertTrue(pd.isna(rows.iloc[1]['exit_price']))

    def test_odd_and_fractional_residuals_are_not_rounded_to_original_quantity(self):
        for total, half, remaining in ((11, 5, 6), (1.5, 1, .5)):
            with self.subTest(total=total):
                self.rows().iloc[:0].to_csv(self.export, index=False)
                self.f.frame.to_csv(self.path, index=False)
                rows = self.cycle(partial=True, quantity=total, orders=[
                    self.order(id='partial', filled_qty=str(half)),
                    self.order(id='residual', filled_qty=str(remaining),
                               filled_at=self.f.entry + timedelta(days=2))])
                self.assertEqual(rows['qty'].tolist(), [half, remaining])
                self.assertEqual(rows['qty'].sum(), total)

    def test_repeated_closure_does_not_append_a_second_verified_fill(self):
        rows = self.cycle(partial=True, orders=[self.order(id='partial', filled_qty='5'),
            self.order(id='residual', filled_qty='5', filled_at=self.f.entry + timedelta(days=2))])
        residual = self.f.frame.copy(); residual['qty'] = 5
        self.monitor.log_closed_positions(self.f.client, {'AAA'}, residual,
                                          excluded_order_ids={'partial'})
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(self.rows()['order_id'].tolist(), ['partial', 'residual'])

    def test_wrong_symbol_side_quantity_and_missing_identity_never_match(self):
        for changes in ({'symbol':'BBB'}, {'side':'buy'}, {'filled_qty':'5'}, {'id':''},
                        {'filled_avg_price':'nan'}, {'filled_at':None}):
            with self.subTest(changes=changes):
                self.rows().iloc[:0].to_csv(self.export,index=False)
                self.f.frame.to_csv(self.path,index=False)
                rows=self.cycle(orders=[self.order(**changes)])
                self.assertEqual(rows.iloc[0]['order_status'],'closed_unverified')
                self.assertTrue(pd.isna(rows.iloc[0]['net_pnl']))

    def test_saved_and_closure_metadata_share_continuity_rules(self):
        self.f.position.avg_entry_price=105
        self.monitor.save_positions_csv([self.f.position],str(self.path))
        saved=pd.read_csv(self.path)
        self.assertEqual(saved.iloc[0]['entry_price'],105)
        self.assertTrue(pd.isna(saved.iloc[0]['entry_time']))

    def test_verified_fill_receipt_reports_actual_identity_and_quantity(self):
        self.f.client.get_all_positions.return_value=[SimpleNamespace(symbol='AAA',qty_available=10)]
        self.f.client.get_order_by_id.return_value=self.order(id='partial',filled_qty='5')
        receipt={}
        with mock.patch.object(self.monitor,'broker_submit_order',return_value=SimpleNamespace(id='partial',dryrun=False)), \
             mock.patch.object(self.monitor,'wait_for_order_terminal',return_value='filled'):
            self.assertTrue(self.monitor.submit_sell_market_order(self.f.position,'partial','test',
                              qty_override=5,fill_receipt=receipt))
        self.assertTrue(receipt['verified'])
        self.assertEqual((receipt['order_id'],receipt['qty']),('partial',5))

    def test_actual_partial_submit_and_closure_paths_agree_on_residual(self):
        real_submit=self.monitor.submit_sell_market_order
        self.f.client.get_order_by_id.return_value=self.order(id='partial',filled_qty='5')
        self.f.client.get_orders.return_value=[self.order(id='partial',filled_qty='5'),
            self.order(id='residual',filled_qty='5',filled_at=self.f.entry+timedelta(days=2))]
        with self.f.cycle_patches(filled=True), \
             mock.patch.object(self.monitor,'broker_submit_order',return_value=SimpleNamespace(id='partial',dryrun=False)), \
             mock.patch.object(self.monitor,'wait_for_order_terminal',return_value='filled'):
            self.monitor.submit_sell_market_order.side_effect=real_submit
            self.f.client.get_all_positions.side_effect=[[SimpleNamespace(symbol='AAA',qty_available=10)],[]]
            self.monitor.process_positions_cycle()
        self.assertEqual(self.rows()['qty'].tolist(),[5,5])
        self.assertEqual(self.rows()['order_id'].tolist(),['partial','residual'])

    def test_unverified_fill_receipt_never_claims_requested_quantity_as_verified(self):
        self.f.client.get_all_positions.return_value=[SimpleNamespace(symbol='AAA',qty_available=10)]
        self.f.client.get_order_by_id.return_value=self.order(id='partial',filled_qty='7')
        receipt={'verified':True,'qty':99}
        with mock.patch.object(self.monitor,'broker_submit_order',return_value=SimpleNamespace(id='partial',dryrun=False)), \
             mock.patch.object(self.monitor,'wait_for_order_terminal',return_value='filled'):
            # Preserve legacy terminal-status behavior without certifying quantity.
            self.assertTrue(self.monitor.submit_sell_market_order(self.f.position,'partial','test',
                              qty_override=5,fill_receipt=receipt))
        self.assertFalse(receipt['verified'])
        self.assertIsNone(receipt['qty'])
        self.assertEqual(self.rows().iloc[0]['order_status'],'filled_unverified')

    def test_partial_checkpoint_survives_refresh_failure_and_next_cycle(self):
        partial=self.order(id='partial',filled_qty='5',filled_at=self.f.entry+timedelta(days=2))
        residual=self.order(id='residual',filled_qty='5')
        self.cycle(partial=True,orders=[partial,residual],refresh_error=True)
        checkpoint=pd.read_csv(self.path)
        self.assertEqual(checkpoint.iloc[0]['qty'],5)
        self.assertEqual(checkpoint.iloc[0]['accounted_exit_order_ids'],'["partial"]')
        with mock.patch.object(self.monitor,'get_open_positions',return_value=[]), \
             mock.patch.object(self.monitor,'refresh_ml_risk_state',return_value={}):
            self.monitor.process_positions_cycle()
        self.assertEqual(self.rows()['qty'].tolist(),[5,5])
        self.assertEqual(self.rows()['order_id'].tolist(),['partial','residual'])

    def test_current_position_refresh_keeps_prior_partial_ids_for_later_disappearance(self):
        self.cycle(partial=True,orders=[],refresh_error=True)
        self.f.position.qty=5
        self.monitor.save_positions_csv([self.f.position],str(self.path))
        checkpoint=pd.read_csv(self.path)
        self.assertEqual(checkpoint.iloc[0]['accounted_exit_order_ids'],'["partial"]')
        self.f.client.get_orders.return_value=[self.order(id='partial',filled_qty='5')]
        self.monitor.log_closed_positions(self.f.client,{'AAA'},checkpoint)
        self.assertEqual(self.rows().iloc[1]['order_status'],'closed_unverified')
        self.assertTrue(pd.isna(self.rows().iloc[1]['order_id']))

    def test_changed_side_does_not_borrow_prior_timestamp(self):
        prior=self.f.frame.copy();prior['side']='short'
        rows=self.cycle(prior=prior,orders=[self.order()])
        self.assertTrue(pd.isna(rows.iloc[0]['entry_time']))
        self.assertEqual(rows.iloc[0]['order_status'],'closed_unverified')

    def test_short_closure_matches_absolute_remaining_quantity(self):
        prior=self.f.frame.copy();prior['qty']=-10;prior['side']='short'
        self.f.client.get_orders.return_value=[self.order(side='buy',filled_avg_price='90')]
        self.monitor.log_closed_positions(self.f.client,{'AAA'},prior)
        self.assertEqual(self.rows().iloc[0]['qty'],10)
        self.assertEqual(self.rows().iloc[0]['net_pnl'],100)

    def test_resolved_entry_time_does_not_lose_accounted_partial_identity(self):
        prior=self.f.frame.copy();prior['entry_time']=None;prior['qty']=5
        prior['accounted_exit_order_ids']='["partial"]'
        prior.to_csv(self.path,index=False)
        self.f.position.qty=5;self.f.position.created_at=self.f.entry
        self.monitor.save_positions_csv([self.f.position],str(self.path))
        self.assertEqual(pd.read_csv(self.path).iloc[0]['accounted_exit_order_ids'],'["partial"]')

    def test_checkpoint_write_failure_preserves_previous_file(self):
        before=self.path.read_bytes()
        with mock.patch.object(self.monitor.os,'replace',side_effect=OSError('synthetic_write_failure')), \
             self.assertRaises(OSError):
            self.monitor._save_position_frame(self.f.frame.assign(qty=5),str(self.path))
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(list(self.path.parent.glob('.positions-*.tmp')),[])

    def test_partial_strategy_state_survives_checkpoint_failure(self):
        def filled(position,**kwargs):
            kwargs['fill_receipt'].update(order_id='partial',verified=True,qty=5)
            return True
        with self.f.cycle_patches(filled=True), \
             mock.patch.object(self.monitor,'_save_position_frame',side_effect=OSError('synthetic_write_failure')):
            self.monitor.submit_sell_market_order.side_effect=filled
            with self.assertRaises(OSError):self.monitor.process_positions_cycle()
            self.monitor.submit_sell_market_order.assert_called_once()
        self.assertTrue(self.monitor.PARTIAL_EXIT_TAKEN['AAA'])


if __name__=='__main__':unittest.main()
