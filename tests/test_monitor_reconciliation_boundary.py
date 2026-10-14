"""Normal monitor imports and real reconciliation writes with offline boundaries."""
import copy
import importlib
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import test_account_sync as hourly_tests
import test_reconciliation_integrity as integrity


def load_monitor(test):
    """Replace provider, environment and persistence boundaries before import."""
    import pytz
    pytz.timezone('US/Eastern')

    def module(name, **fields):
        value = ModuleType(name)
        value.__dict__.update(fields)
        return value

    client = mock.Mock()
    for name in ('submit_order', 'close_position', 'cancel_order_by_id',
                 'get_orders', 'get_order_by_id', 'get_all_positions'):
        getattr(client, name).side_effect = test.forbidden
    request = lambda **fields: SimpleNamespace(**fields)
    boundaries = {
        'alpaca.trading': module('alpaca.trading', __path__=[]),
        'alpaca.trading.client': module('alpaca.trading.client', TradingClient=lambda *a, **k: client),
        'alpaca.trading.enums': module('alpaca.trading.enums', OrderSide=SimpleNamespace(SELL='sell'),
            TimeInForce=SimpleNamespace(DAY='day', GTC='gtc'), QueryOrderStatus=SimpleNamespace(OPEN='open')),
        'alpaca.trading.requests': module('alpaca.trading.requests', GetOrdersRequest=request,
            TrailingStopOrderRequest=request, LimitOrderRequest=request, StopOrderRequest=request),
        'alpaca.data': module('alpaca.data', __path__=[]),
        'alpaca.data.historical': module('alpaca.data.historical',
            StockHistoricalDataClient=lambda *a, **k: SimpleNamespace()),
        'alpaca.data.requests': module('alpaca.data.requests', StockLatestTradeRequest=request),
        'scripts.db': test.module,
    }
    test.stack.enter_context(mock.patch.dict(sys.modules, boundaries))
    test.stack.enter_context(mock.patch.object(sys.modules['scripts'], 'db', test.module))
    test.stack.enter_context(mock.patch.object(sys.modules['scripts.utils'],
        'fetch_bars_with_cutoff', test.forbidden, create=True))
    # Prevent local runtime-file writes/reads and dotenv discovery during import.
    runtime = Path(test.storage)
    (runtime / 'logs').mkdir(exist_ok=True)
    (runtime / 'data').mkdir(exist_ok=True)
    original_abspath = os.path.abspath
    source = integrity.sync.ROOT / 'scripts' / 'monitor_positions.py'
    def runtime_abspath(path):
        if Path(path) == source:
            return str(runtime / 'scripts' / 'monitor_positions.py')
        return original_abspath(path)
    test.stack.enter_context(mock.patch('os.path.abspath', side_effect=runtime_abspath))
    test.stack.enter_context(mock.patch('os.path.exists', return_value=False))
    test.stack.enter_context(mock.patch.object(Path, 'exists', return_value=False))
    class OfflineFileHandler(logging.NullHandler):
        def __init__(self, *args, **kwargs):
            super().__init__()
    test.stack.enter_context(mock.patch('logging.FileHandler', OfflineFileHandler))
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    levels = {handler: handler.level for handler in handlers}
    def restore_logging():
        root.handlers[:] = handlers
        root.setLevel(level)
        for handler, saved_level in levels.items():
            handler.setLevel(saved_level)
    test.addCleanup(restore_logging)
    test.stack.enter_context(mock.patch.dict(__import__('os').environ, {
        'APCA_API_KEY_ID': 'synthetic', 'APCA_API_SECRET_KEY': 'synthetic',
        'APCA_API_BASE_URL': 'https://paper.invalid', 'APCA_DATA_API_BASE_URL': 'https://data.invalid',
        'ALPACA_DATA_FEED': 'iex', 'MONITOR_DISABLE_SELLS': 'true',
    }))
    sys.modules.pop('scripts.monitor_positions', None)
    test.monitor = importlib.import_module('scripts.monitor_positions')
    test.addCleanup(lambda: sys.modules.pop('scripts.monitor_positions', None))
    test.stack.enter_context(mock.patch.object(test.monitor, '_persist_metrics'))
    test.stack.enter_context(mock.patch.object(test.monitor, '_save_monitor_state'))
    test.stack.enter_context(mock.patch.object(test.monitor, 'broker_close_position', side_effect=test.forbidden))
    return test.monitor


class MonitorReconciliationBoundaryTests(unittest.TestCase):
    setUp = hourly_tests.ExecutorGateIntegrationTests.setUp
    prepare = integrity.ActivityTests.prepare
    read_module = integrity.DatabaseReadTests.read_module
    setup_caller = integrity.SharedConnectionCallerTests.setup_caller

    def test_monitor_observations_allow_later_owned_reconciliation(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                self.setup_caller(closed=True)
                self.connection.trade[5:8] = [integrity.NOW, None, 'POSITION_CLOSED']
                monitor = load_monitor(self)
                with mock.patch.object(monitor, 'db_logging_enabled', return_value=True):
                    self.assertTrue(monitor.db_log_event(event_type=monitor.MONITOR_DB_EVENT_TYPES['sell_fill'],
                        symbol='XYZ', qty=1 if partial else 2, order_id='sell-1', status='filled',
                        event_time=integrity.NOW, raw={'partial': partial}))
                self.assertEqual(self.connection.observations[0]['event_type'], 'SELL_FILL_OBSERVED')
                self.assertEqual(self.connection.events, [])
                # The actual SQL event reader must filter canonical SELL_FILL,
                # so observation events cannot be mistaken for orphan evidence.
                self.assertTrue(self.caller.reconcile_closed_trades())
                self.assertEqual(self.connection.trade[4:9],
                    ['sell-1', integrity.NOW, 11.0, 'SELL_FILL', 2.0])
                self.assertEqual(len(self.connection.events), 1)
                self.assertTrue(self.caller.reconcile_closed_trades())
                self.assertEqual(len(self.connection.events), 1)
                self.assertEqual(len(self.connection.observations), 1)

    def test_monitor_cannot_request_canonical_sell_fill_logging(self):
        self.setup_caller()
        monitor = load_monitor(self)
        original = copy.deepcopy(self.connection.trade)
        with mock.patch.object(monitor, 'db_logging_enabled', return_value=True), \
                mock.patch.object(self.module, 'insert_order_event') as insert:
            self.assertFalse(monitor.db_log_event(event_type='SELL_FILL', symbol='XYZ', qty=2,
                order_id='sell-1', status='filled', event_time=integrity.NOW))
        insert.assert_not_called()
        self.assertEqual(self.connection.trade, original)
