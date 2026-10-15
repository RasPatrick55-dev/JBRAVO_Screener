"""Normal monitor import with inert clients and temporary state; no credentials."""
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import importlib.util
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import pandas as pd
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus
import scripts

ROOT = Path(__file__).resolve().parents[1]


class MonitorRegressions(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        source = Path(temporary) / "scripts/monitor_positions.py"
        source.parent.mkdir()
        shutil.copyfile(ROOT / "scripts/monitor_positions.py", source)
        self.client = mock.Mock()
        self.stack.enter_context(mock.patch("alpaca.trading.client.TradingClient", return_value=self.client))
        self.stack.enter_context(mock.patch("alpaca.data.historical.StockHistoricalDataClient", return_value=mock.Mock()))
        helper = ModuleType("scripts.utils")
        helper.fetch_bars_with_cutoff = mock.Mock()
        alerts = ModuleType("utils.alerts")
        alerts.send_alert = mock.Mock()
        self.stack.enter_context(mock.patch.dict(sys.modules, {"scripts.utils": helper, "utils.alerts": alerts}))
        self.stack.enter_context(mock.patch.object(scripts, "utils", helper, create=True))
        self.stack.enter_context(mock.patch.dict(os.environ, {
            "APCA_API_KEY_ID": "inert", "APCA_API_SECRET_KEY": "inert",
            "APCA_API_BASE_URL": "https://paper-api.alpaca.markets",
            "APCA_DATA_API_BASE_URL": "https://data.alpaca.markets", "ALPACA_DATA_FEED": "iex",
        }, clear=True))
        for target in ("socket.socket.connect", "socket.create_connection", "subprocess.Popen", "subprocess.run"):
            self.stack.enter_context(mock.patch(target, side_effect=AssertionError("external_forbidden")))
        spec = importlib.util.spec_from_file_location("scripts._monitor_regressions", source)
        self.monitor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.monitor)
        self.addCleanup(self.close_handlers)
        self.stack.enter_context(mock.patch.object(self.monitor, "db_logging_enabled", return_value=False))
        self.stack.enter_context(mock.patch.object(self.monitor, "_persist_metrics"))
        self.real_trade_exit = self.monitor.log_trade_exit
        self.logged = self.stack.enter_context(mock.patch.object(self.monitor, "log_trade_exit"))
        self.entry = datetime(2026, 10, 1, tzinfo=timezone.utc)
        self.frame = pd.DataFrame([dict(symbol="AAA", qty=10, entry_price=100,
                                      entry_time=self.entry.isoformat(), side="long")])
        self.position = SimpleNamespace(symbol="AAA", qty=10, qty_available=0,
            avg_entry_price=100, current_price=110, created_at=None)

    def close_handlers(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "baseFilename", None) == self.monitor.logfile:
                handler.close()
                root.removeHandler(handler)

    def order(self, **changes):
        return SimpleNamespace(**(dict(id="fill", symbol="AAA", side=OrderSide.SELL, status=OrderStatus.FILLED,
            filled_at=self.entry + timedelta(days=1), filled_avg_price="110", filled_qty="10",
            order_type="limit") | changes))

    def test_canceled_first_order_does_not_crash_or_mask_fill(self):
        self.client.get_orders.return_value = [self.order(status=OrderStatus.CANCELED, filled_at=None), self.order()]
        self.monitor.log_closed_positions(self.client, {"AAA"}, self.frame)
        request = self.client.get_orders.call_args.args[0]
        self.assertEqual(request.status, QueryOrderStatus.CLOSED)
        record = self.logged.call_args
        self.assertEqual(record.args[3], "110")
        self.assertEqual(record.args[5], (self.entry + timedelta(days=1)).isoformat())
        self.assertEqual(record.kwargs["order_id"], "fill")

    def test_missing_wrong_side_old_and_partial_fills_remain_unverified(self):
        for orders in ([], [self.order(filled_at=None)], [self.order(side=OrderSide.BUY)],
                       [self.order(filled_at=self.entry - timedelta(days=1))],
                       [self.order(filled_qty="5")], [self.order(filled_avg_price="nan")]):
            with self.subTest(orders=orders):
                self.client.get_orders.return_value = orders
                self.monitor.log_closed_positions(self.client, {"AAA"}, self.frame)
                record = self.logged.call_args.args
                self.assertEqual((record[3], record[5], record[6]), ("", "", "closed_unverified"))

    def test_lookup_failure_keeps_cycle_alive_without_invented_fill(self):
        self.client.get_orders.side_effect = RuntimeError("inert_lookup_failure")
        self.monitor.log_closed_positions(self.client, {"AAA"}, self.frame)
        self.assertEqual(self.logged.call_args.args[6], "closed_unverified")

    def test_short_close_uses_buy_fill(self):
        self.frame.loc[0, "side"] = "short"
        self.client.get_orders.return_value = [self.order(side=OrderSide.BUY)]
        self.monitor.log_closed_positions(self.client, {"AAA"}, self.frame)
        self.assertEqual(self.logged.call_args.args[9], "buy")

    def test_short_exit_pnl_has_correct_direction(self):
        pd.DataFrame(columns=["symbol"]).to_csv(self.monitor.trades_log_real_path, index=False)
        self.real_trade_exit("AAA", 10, 100, "90", self.entry.isoformat(),
                             (self.entry + timedelta(days=1)).isoformat(), "closed", "limit", "exit", "buy")
        saved = pd.read_csv(self.monitor.trades_log_real_path)
        self.assertEqual(saved.iloc[0]["net_pnl"], 100)

    def test_latest_matching_fill_wins_independent_of_response_order(self):
        self.client.get_orders.return_value = [self.order(filled_at=self.entry + timedelta(days=2), id="new"), self.order(id="old")]
        self.monitor.log_closed_positions(self.client, {"AAA"}, self.frame)
        self.assertEqual(self.logged.call_args.kwargs["order_id"], "new")

    def test_partial_exit_refreshes_reserved_snapshot_and_sells_exact_requested_qty(self):
        self.client.get_all_positions.return_value = [SimpleNamespace(symbol="AAA", qty_available="10.0")]
        with mock.patch.object(self.monitor, "broker_submit_order", return_value=SimpleNamespace(id="exit", dryrun=False)) as submit, \
             mock.patch.object(self.monitor, "wait_for_order_terminal", return_value="filled"):
            self.assertTrue(self.monitor.submit_sell_market_order(self.position, "partial", "partial_gain", qty_override=5))
        self.client.get_all_positions.assert_called_once()
        self.assertEqual(submit.call_args.args[0].qty, 5)
        self.assertEqual(self.logged.call_args.args[4], "")  # Unknown created_at, not fabricated.

    def test_unknown_missing_and_insufficient_quantity_never_submit_or_close(self):
        for positions in ([], [SimpleNamespace(symbol="AAA", qty_available=0)],
                          [SimpleNamespace(symbol="AAA", qty_available=4)]):
            with self.subTest(positions=positions), mock.patch.object(self.monitor, "broker_submit_order") as submit, \
                 mock.patch.object(self.monitor, "broker_close_position") as close:
                self.client.get_all_positions.return_value = positions
                self.assertFalse(self.monitor.submit_sell_market_order(self.position, "partial", "partial_gain", qty_override=5))
                submit.assert_not_called()
                close.assert_not_called()
        self.client.get_all_positions.side_effect = RuntimeError("inert_lookup_failure")
        with mock.patch.object(self.monitor, "broker_submit_order") as submit:
            self.assertFalse(self.monitor.submit_sell_market_order(self.position, "exit", "monitor"))
            submit.assert_not_called()

    def test_rejected_or_resting_exit_is_not_a_filled_partial_exit(self):
        self.client.get_all_positions.return_value = [SimpleNamespace(symbol="AAA", qty_available=10)]
        for status in ("rejected", "canceled", "new"):
            with self.subTest(status=status), mock.patch.object(self.monitor, "broker_submit_order", return_value=SimpleNamespace(id="exit", dryrun=False)), \
                 mock.patch.object(self.monitor, "wait_for_order_terminal", return_value=status):
                self.assertFalse(self.monitor.submit_sell_market_order(self.position, "partial", "partial_gain", qty_override=5))

    def test_missing_submission_response_cannot_be_logged_as_success_or_close_position(self):
        self.client.get_all_positions.return_value = [SimpleNamespace(symbol="AAA", qty_available=10)]
        with mock.patch.object(self.monitor, "broker_submit_order", return_value=None), \
             mock.patch.object(self.monitor, "broker_close_position") as close, \
             mock.patch.object(self.monitor, "wait_for_order_terminal") as wait:
            self.assertFalse(self.monitor.submit_sell_market_order(self.position, "exit", "monitor"))
            close.assert_not_called()
            wait.assert_not_called()
            self.logged.assert_not_called()

    def test_cancel_failure_never_escalates_to_position_close(self):
        with mock.patch.object(self.monitor, "broker_cancel_order", return_value=False), \
             mock.patch.object(self.monitor, "broker_close_position") as close:
            self.assertFalse(self.monitor.cancel_order_safe("stop", "AAA"))
            close.assert_not_called()

    def cycle_patches(self, cancellation=True, filled=False):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(self.client, "get_all_positions", return_value=[self.position]))
        for name, result in (("refresh_ml_risk_state", {}), ("get_open_positions", [self.position]),
                             ("save_positions_csv", None), ("calculate_days_held", 0),
                             ("fetch_indicators", {"close": 110, "EMA20": 101, "RSI": 60}),
                             ("has_pending_sell_order", False),
                             ("get_trailing_stop_order", SimpleNamespace(id="stop", trail_percent=3)),
                             ("cancel_order_safe", cancellation), ("submit_sell_market_order", filled)):
            stack.enter_context(mock.patch.object(self.monitor, name, return_value=result))
        return stack

    def test_partial_state_changes_only_after_confirmed_fill(self):
        for filled in (False, True):
            self.monitor.PARTIAL_EXIT_TAKEN.clear()
            with self.cycle_patches(filled=filled):
                self.monitor.process_positions_cycle()
            self.assertEqual(bool(self.monitor.PARTIAL_EXIT_TAKEN.get("AAA")), filled)

    def test_unconfirmed_stop_cancellation_defers_partial_exit(self):
        with self.cycle_patches(cancellation=False, filled=True):
            self.monitor.process_positions_cycle()
            self.monitor.submit_sell_market_order.assert_not_called()
            self.assertNotIn("AAA", self.monitor.PARTIAL_EXIT_TAKEN)

    def test_sell_export_uses_broker_fill_not_limit_or_poll_time(self):
        self.client.get_all_positions.return_value = [SimpleNamespace(symbol="AAA", qty_available=10)]
        fill = self.order(id="exit", symbol="AAA", filled_avg_price="112.25")
        self.client.get_order_by_id.return_value = fill
        with mock.patch.object(self.monitor, "broker_submit_order", return_value=SimpleNamespace(id="exit", dryrun=False)), \
                mock.patch.object(self.monitor, "wait_for_order_terminal", return_value="filled"):
            self.assertTrue(self.monitor.submit_sell_market_order(self.position, "exit", "monitor"))
        args = self.logged.call_args.args
        self.assertEqual(args[3], "112.25")
        self.assertEqual(args[5], fill.filled_at.isoformat())
        self.assertEqual(args[6], "filled")

    def test_unverified_fill_never_exports_limit_as_execution(self):
        self.client.get_all_positions.return_value = [SimpleNamespace(symbol="AAA", qty_available=10)]
        for changes in (dict(id="other"), dict(symbol="BBB"), dict(side="buy"), dict(filled_qty=5),
                        dict(filled_avg_price="nan"), dict(filled_at=None),
                        dict(filled_at=datetime(2026, 10, 2))):
            with self.subTest(changes=changes):
                self.client.get_order_by_id.return_value = self.order(**(dict(id="exit", symbol="AAA") | changes))
                with mock.patch.object(self.monitor, "broker_submit_order", return_value=SimpleNamespace(id="exit", dryrun=False)), \
                        mock.patch.object(self.monitor, "wait_for_order_terminal", return_value="filled"):
                    self.assertTrue(self.monitor.submit_sell_market_order(self.position, "exit", "monitor"))
                self.assertEqual(self.logged.call_args.args[3], "")
                self.assertEqual(self.logged.call_args.args[5], "")
                self.assertEqual(self.logged.call_args.args[6], "filled_unverified")

    def test_one_share_target_returns_flat_snapshot_without_partial_flag(self):
        self.position.qty = 1
        self.monitor.PARTIAL_EXIT_TAKEN.clear()
        with self.cycle_patches(filled=True):
            self.client.get_all_positions.return_value = []
            self.assertEqual(self.monitor.process_positions_cycle(), [])
            self.monitor.save_positions_csv.assert_called_with([], mock.ANY)
        self.assertNotIn("AAA", self.monitor.PARTIAL_EXIT_TAKEN)

    def test_partial_target_returns_remaining_broker_quantity(self):
        remaining = SimpleNamespace(symbol="AAA", qty=5, qty_available=5)
        with self.cycle_patches(filled=True):
            self.client.get_all_positions.return_value = [remaining]
            self.assertEqual(self.monitor.process_positions_cycle(), [remaining])
            self.monitor.save_positions_csv.assert_called_with([remaining], mock.ANY)

    def test_post_exit_position_failure_cannot_be_interpreted_as_flat(self):
        with self.cycle_patches(filled=True):
            self.client.get_all_positions.side_effect = RuntimeError("unavailable")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                self.monitor.process_positions_cycle()

    def test_verified_exit_order_is_not_appended_twice(self):
        pd.DataFrame(columns=["symbol", "order_id", "order_status"]).to_csv(self.monitor.trades_log_real_path, index=False)
        args = ("AAA", 10, 100, "110", self.entry.isoformat(),
                (self.entry + timedelta(days=1)).isoformat(), "filled", "limit", "exit", "sell")
        self.real_trade_exit(*args, order_id="same-order")
        self.real_trade_exit(*args, order_id="same-order")
        saved = pd.read_csv(self.monitor.trades_log_real_path)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.iloc[0]["order_id"], "same-order")

    def test_late_fill_disappearance_is_logged_before_snapshot_replacement(self):
        csv_path=Path(self.monitor.BASE_DIR)/'data/open_positions.csv'
        self.frame.to_csv(csv_path,index=False)
        self.monitor.PARTIAL_EXIT_TAKEN['AAA']=True
        order=[]
        with self.cycle_patches(filled=False), \
             mock.patch.object(self.monitor,'log_closed_positions',side_effect=lambda client,symbols,frame,**kwargs:order.append(('closed',symbols,frame.copy()))), \
             mock.patch.object(self.monitor,'save_positions_csv',side_effect=lambda positions,path:order.append(('save',list(positions)))):
            self.client.get_all_positions.return_value=[]
            # Permit the normal target branch to attempt an exit this cycle.
            self.monitor.PARTIAL_EXIT_TAKEN.clear()
            self.monitor.process_positions_cycle()
        self.assertEqual(order[-2][0:2],('closed',{'AAA'}))
        self.assertEqual(order[-2][2].iloc[0]['entry_time'],self.entry.isoformat())
        self.assertEqual(order[-1],('save',[]))

    def test_unrelated_position_closing_during_exit_keeps_metadata_and_clears_partial(self):
        other=SimpleNamespace(**(vars(self.position)|{'symbol':'BBB'}))
        frame=pd.concat([self.frame,self.frame.assign(symbol='BBB')],ignore_index=True)
        frame.to_csv(Path(self.monitor.BASE_DIR)/'data/open_positions.csv',index=False)
        self.monitor.PARTIAL_EXIT_TAKEN['BBB']=True
        self.client.get_orders.return_value=[self.order(symbol='BBB',id='bbb-fill')]
        with self.cycle_patches(filled=False):
            self.monitor.get_open_positions.return_value=[self.position,other]
            self.client.get_all_positions.return_value=[self.position]
            self.monitor.process_positions_cycle()
        self.assertNotIn('BBB',self.monitor.PARTIAL_EXIT_TAKEN)
        self.assertEqual(self.logged.call_args.args[0],'BBB')
        self.assertEqual(self.logged.call_args.args[4],self.entry.isoformat())

    def test_refresh_preserves_entry_metadata_but_does_not_invent_new_time(self):
        path=Path(self.monitor.BASE_DIR)/'data/open_positions.csv'
        self.frame.to_csv(path,index=False)
        self.monitor.save_positions_csv([self.position],str(path))
        saved=pd.read_csv(path)
        self.assertEqual(pd.to_datetime(saved.iloc[0]['entry_time'],utc=True),self.entry)
        self.position.avg_entry_price=105
        self.monitor.save_positions_csv([self.position],str(path))
        self.assertTrue(pd.isna(pd.read_csv(path).iloc[0]['entry_time']))


if __name__ == "__main__":
    unittest.main()
