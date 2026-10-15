"""Interleave the real monitor and dashboard CSV writers with inert clients."""
import importlib.util
from pathlib import Path
import shutil
import unittest
from unittest import mock

import pandas as pd
from tests import test_monitor_closure_followup as closure


class PositionWriters(unittest.TestCase):
    def setUp(self):
        self.case=closure.ClosureAccountingTests('test_partial_then_residual_exports_only_remaining_shares')
        self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.monitor=self.case.monitor
        self.case.f.position.unrealized_pl='100'
        source=Path(self.monitor.BASE_DIR)/'scripts/update_dashboard_data.py'
        shutil.copyfile(closure.fixtures.ROOT/'scripts/update_dashboard_data.py',source)
        spec=importlib.util.spec_from_file_location('scripts._dashboard_writer_regression',source)
        self.dashboard=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.dashboard)
        self.addCleanup(self.close_handlers)

    def close_handlers(self):
        for handler in list(self.dashboard.logger.handlers):
            if getattr(handler,'baseFilename',None)==str(Path(self.monitor.BASE_DIR)/'logs/update_dashboard_data.log'):
                handler.close();self.dashboard.logger.removeHandler(handler)

    def refresh_dashboard(self,positions):
        self.case.f.client.get_all_positions.side_effect=None
        self.case.f.client.get_all_positions.return_value=positions
        self.dashboard.update_open_positions()
        return pd.read_csv(self.case.path)

    def test_dashboard_between_partial_and_disappearance_preserves_exclusion(self):
        self.case.cycle(partial=True,refresh_error=True)
        position=self.case.f.position;position.qty=5
        frame=self.refresh_dashboard([position])
        self.assertIn('accounted_exit_order_ids',frame)
        self.assertEqual(frame.iloc[0]['accounted_exit_order_ids'],'["partial"]')
        self.case.f.client.get_orders.return_value=[self.case.order(id='partial',filled_qty='5')]
        with mock.patch.object(self.monitor,'get_open_positions',return_value=[]), \
             mock.patch.object(self.monitor,'refresh_ml_risk_state',return_value={}):
            self.monitor.process_positions_cycle()
        rows=self.case.rows()
        self.assertEqual(rows['qty'].tolist(),[5,5])
        self.assertEqual(rows.iloc[1]['order_status'],'closed_unverified')
        self.assertTrue(pd.isna(rows.iloc[1]['order_id']))

    def test_partial_checkpoint_after_failed_refresh_has_dashboard_schema(self):
        self.case.cycle(partial=True,refresh_error=True)
        frame=pd.read_csv(self.case.path)
        required={'symbol','qty','net_pnl','entry_price','entry_time','side','days_in_trade'}
        self.assertTrue(required.issubset(frame.columns),sorted(required-set(frame.columns)))
        self.assertEqual(frame.iloc[0]['qty'],5)
        self.assertTrue(pd.isna(frame.iloc[0]['net_pnl']))

    def test_dashboard_changed_basis_does_not_retain_prior_time_or_exit_ids(self):
        previous=self.case.f.frame.copy();previous['accounted_exit_order_ids']='["partial"]'
        previous.to_csv(self.case.path,index=False)
        self.case.f.position.avg_entry_price=105
        frame=self.refresh_dashboard([self.case.f.position])
        self.assertEqual(frame.iloc[0]['entry_price'],105)
        self.assertTrue(pd.isna(frame.iloc[0]['entry_time']))
        self.assertEqual(frame.iloc[0]['accounted_exit_order_ids'],'[]')

    def test_both_writers_preserve_known_continuing_entry_time_and_exit_ids(self):
        previous=self.case.f.frame.copy();previous['accounted_exit_order_ids']='["partial"]'
        previous.to_csv(self.case.path,index=False)
        first=self.refresh_dashboard([self.case.f.position])
        self.monitor.save_positions_csv([self.case.f.position],str(self.case.path))
        second=pd.read_csv(self.case.path)
        for frame in (first,second):
            self.assertEqual(frame.iloc[0]['accounted_exit_order_ids'],'["partial"]')
            self.assertEqual(pd.Timestamp(frame.iloc[0]['entry_time']),pd.Timestamp(self.case.f.entry))

    def test_empty_dashboard_and_monitor_frames_have_complete_schema(self):
        dashboard=self.refresh_dashboard([])
        self.monitor.save_positions_csv([],str(self.case.path))
        monitor=pd.read_csv(self.case.path)
        required={'symbol','qty','net_pnl','entry_price','entry_time','side','days_in_trade','accounted_exit_order_ids'}
        self.assertTrue(required.issubset(dashboard.columns))
        self.assertTrue(required.issubset(monitor.columns))

    def test_dashboard_sqlite_export_accepts_broker_objects_with_nested_metadata(self):
        import sqlite3
        self.case.f.position.extra_metadata={'broker':'internal'}
        frame=self.refresh_dashboard([self.case.f.position])
        self.assertNotIn('extra_metadata',frame)
        with sqlite3.connect(self.dashboard.DB_PATH) as connection:
            stored=pd.read_sql_query('SELECT * FROM open_positions',connection)
        self.assertEqual(stored.iloc[0]['qty'],10)
        self.assertEqual(stored.iloc[0]['net_pnl'],100)

    def test_changed_side_and_changed_known_time_clear_accounted_identity(self):
        from datetime import timedelta
        for change in ('side','time'):
            with self.subTest(change=change):
                previous=self.case.f.frame.copy();previous['accounted_exit_order_ids']='["partial"]'
                previous.to_csv(self.case.path,index=False)
                self.case.f.position.side='short' if change=='side' else 'long'
                self.case.f.position.created_at=self.case.f.entry+timedelta(days=1) if change=='time' else None
                frame=self.refresh_dashboard([self.case.f.position])
                self.assertEqual(frame.iloc[0]['accounted_exit_order_ids'],'[]')

    def test_dashboard_inflight_write_cannot_erase_a_partial_checkpoint(self):
        import threading
        entered=threading.Event();release=threading.Event();finished=threading.Event()
        errors=[]
        original=self.dashboard.write_csv_atomic
        def paused_write(*args):
            entered.set()
            if not release.wait(3):raise RuntimeError('synthetic_test_timeout')
            original(*args)
        def dashboard():
            try:self.dashboard.update_open_positions()
            except Exception as exc:errors.append(exc)
        def monitor():
            try:self.case.cycle(partial=True,refresh_error=True)
            except Exception as exc:errors.append(exc)
            finally:finished.set()
        self.case.f.client.get_all_positions.return_value=[self.case.f.position]
        with mock.patch.object(self.dashboard,'write_csv_atomic',side_effect=paused_write):
            first=threading.Thread(target=dashboard);first.start()
            self.assertTrue(entered.wait(2))
            second=threading.Thread(target=monitor);second.start()
            try:self.assertFalse(finished.wait(.15))
            finally:
                release.set();first.join(3);second.join(3)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors,[])
        frame=pd.read_csv(self.case.path)
        self.assertEqual(frame.iloc[0]['qty'],5)
        self.assertEqual(frame.iloc[0]['accounted_exit_order_ids'],'["partial"]')


if __name__=='__main__':unittest.main()
