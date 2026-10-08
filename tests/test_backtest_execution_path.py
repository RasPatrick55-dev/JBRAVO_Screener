"""Normal backtest import/caller/CLI with pre-import external boundary doubles.

No AST extraction. Real indicators, scoring, simulator and scripts.utils CSV
writer execute. Environment/credential loading, logger placement, SDK and DB
connections are replaced before import; SQL is interpreted only by an inert
fixture, never sent to a database. Run with --noconftest and plugin autoload off.
"""

import builtins
import importlib
import io
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.alpaca_optional


def synthetic_bars(count=240):
    index = pd.bdate_range("2024-01-01", periods=count, name="date")
    close = 100 + np.arange(count) * 0.08 + np.sin(np.arange(count)) * 0.4
    return pd.DataFrame({"open": close - 0.1, "high": close + 0.5,
                         "low": close - 0.6, "close": close,
                         "volume": 5000 + np.arange(count)}, index=index)


class FixtureStore:
    def __init__(self):
        self.bars = {"AAA": synthetic_bars()}
        self.symbols = ["AAA"]
        self.run_date = self.bars["AAA"].index[-1].date()
        self.queries = []
        self.connections = []
        self.outputs = []
        self.insert_result = True
        self.enabled = True

    def connect(self):
        connection = SimpleNamespace(closed=False)
        connection.cursor = lambda: FixtureCursor(self)
        connection.close = lambda: setattr(connection, "closed", True)
        self.connections.append(connection)
        return connection

    def insert(self, run_date, frame):
        self.outputs.append((run_date, frame.copy(deep=True)))
        return self.insert_result


class FixtureCursor:
    def __init__(self, store):
        self.store = store
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        self.store.queries.append((sql, params))
        if sql == "SELECT 1":
            self.rows = [(1,)]
        elif "information_schema.columns" in sql:
            self.rows = [(name,) for name in
                         ("run_date", "symbol", "score", "exchange", "entry_price")]
        elif "SELECT MAX(run_date)" in sql:
            self.rows = [(self.store.run_date,)]
        elif "SELECT symbol, run_date" in sql:
            self.rows = [(symbol, params.get("run_date") or self.store.run_date)
                         for symbol in self.store.symbols[:params["limit"]]]
        elif "FROM daily_bars" in sql:
            frame = self.store.bars.get(params["symbol"], pd.DataFrame())
            self.rows = [(timestamp.date(), *row)
                         for timestamp, row in frame.iterrows()
                         if timestamp.date() <= params["end_date"]]
        else:
            raise AssertionError(f"Unexpected fixture SQL: {sql}")

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


@pytest.fixture
def normal_module(tmp_path, monkeypatch):
    """Establish all safety boundaries before importing the actual module."""
    storage = tmp_path.resolve()
    (storage / "data").mkdir()
    forbidden_effects = []

    def forbidden(*args, **kwargs):
        forbidden_effects.append("external execution attempted")
        raise AssertionError("Provider/network/process access forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(os, "environ", {})

    def allowed_path(path):
        resolved = Path(path).resolve()
        assert resolved.is_relative_to(storage), f"Write outside temporary storage: {resolved}"

    original_open, original_io_open = builtins.open, io.open
    original_os_open, original_mkdir = os.open, os.mkdir
    original_makedirs = os.makedirs

    def guarded_open(delegate):
        def open_file(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                allowed_path(file)
            return delegate(file, mode, *args, **kwargs)
        return open_file

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            allowed_path(path)
        return original_os_open(path, flags, *args, **kwargs)

    def guarded_mkdir(path, *args, **kwargs):
        allowed_path(path)
        return original_mkdir(path, *args, **kwargs)

    def redirected_makedirs(path, *args, **kwargs):
        # Backtest initialization's repository data mkdir is redirected, not run.
        if Path(path).resolve() == REPO / "data":
            path = storage / "data"
        allowed_path(path)
        return original_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open(original_open))
    monkeypatch.setattr(io, "open", guarded_open(original_io_open))
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(os, "mkdir", guarded_mkdir)
    monkeypatch.setattr(os, "makedirs", redirected_makedirs)
    for operation in ("rename", "replace"):
        delegate = getattr(os, operation)
        def guarded_move(source, destination, *args, _delegate=delegate, **kwargs):
            allowed_path(source)
            allowed_path(destination)
            return _delegate(source, destination, *args, **kwargs)
        monkeypatch.setattr(os, operation, guarded_move)
    for operation in ("unlink", "remove", "rmdir"):
        delegate = getattr(os, operation)
        def guarded_remove(path, *args, _delegate=delegate, **kwargs):
            allowed_path(path)
            return _delegate(path, *args, **kwargs)
        monkeypatch.setattr(os, operation, guarded_remove)

    def install(name, **fields):
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    # Only boundary packages are substituted; scripts.backtest, indicators and
    # scripts.utils are imported normally from the checkout.
    install("utils", __path__=[str(REPO / "utils")])
    env = install("utils.env", load_env=Mock(),
                  get_alpaca_creds=Mock(return_value=(None, None, None, None)))
    handler = logging.FileHandler(storage / "backtest.log", encoding="utf-8")
    logger = logging.Logger("offline-backtest", logging.INFO)
    logger.addHandler(handler)
    logging_boundary = install("utils.logger_utils", init_logging=Mock(return_value=logger))
    install("psycopg2", __path__=[])
    install("psycopg2.extensions", connection=object)
    for name in ("alpaca", "alpaca.data", "alpaca.common"):
        install(name, __path__=[])
    install("alpaca.data.historical", StockHistoricalDataClient=forbidden)
    install("alpaca.data.requests", StockBarsRequest=SimpleNamespace,
            StockLatestTradeRequest=SimpleNamespace)
    install("alpaca.data.timeframe", TimeFrame=SimpleNamespace(Day="day", Minute="minute"))
    install("alpaca.common.exceptions", APIError=RuntimeError)
    store = FixtureStore()
    install("scripts.db", db_enabled=lambda: store.enabled, get_db_conn=store.connect,
            insert_backtest_results=store.insert, upsert_daily_bars_frame=forbidden)
    scripts = importlib.import_module("scripts")
    monkeypatch.delattr(scripts, "db", raising=False)
    for name in ("scripts.backtest", "scripts.indicators", "scripts.utils"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("scripts.backtest")
    module.BASE_DIR = str(storage)
    actual_backtester = module.PortfolioBacktester
    instances = []

    def capture(*args, **kwargs):
        instance = actual_backtester(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(module, "PortfolioBacktester", capture)
    try:
        yield SimpleNamespace(module=module, store=store, instances=instances,
                              storage=storage, env=env, logging=logging_boundary,
                              bt_type=actual_backtester)
    finally:
        handler.close()
        for name in ("scripts.backtest", "scripts.indicators", "scripts.utils"):
            sys.modules.pop(name, None)
        for name in ("backtest", "indicators", "utils", "db"):
            monkeypatch.delattr(scripts, name, raising=False)
        assert not forbidden_effects, "A blocked external operation was attempted"


def run_caller(context, **overrides):
    arguments = dict(run_date=context.store.run_date, lookback_days=240,
                     min_history_bars=1, enable_db=False, export_csv=True)
    arguments.update(overrides)
    return context.module.run_backtest(["AAA"], **arguments)


def cli_arguments(context):
    return ["--run-date", str(context.store.run_date), "--lookback-days", "240",
            "--min-history-bars", "1", "--trade-cost", "1.25", "--slippage", "0.001"]


def test_normal_import_and_actual_cli_parameter_delivery(normal_module):
    context = normal_module
    context.env.load_env.assert_called_once_with()
    context.env.get_alpaca_creds.assert_called_once_with()
    context.logging.init_logging.assert_called_once_with("scripts.backtest", "backtest.log")
    assert context.module.CONFIG == {"trail_pct": 0.03, "max_hold_days": 7}
    assert context.module.main(cli_arguments(context)) == 0
    bt = context.instances[0]
    assert (bt.trade_cost, bt.slippage) == (1.25, 0.001)
    assert set(bt.data) == {"AAA"}
    assert bt.dates[-1].date() == context.store.run_date
    assert "ma200" in bt.data["AAA"] and bt.data["AAA"]["ma200"].notna().any()
    assert context.module.compute_indicators.__module__ == "scripts.indicators"
    assert context.store.outputs[0][0] == context.store.run_date
    assert all(connection.closed for connection in context.store.connections)
    assert any(params == {"symbol": "AAA", "end_date": context.store.run_date}
               for _, params in context.store.queries)
    assert not (context.storage / "data" / "trades_log.csv").exists()
    assert bt.equity().iloc[-1]["equity"] == pytest.approx(
        bt.cash + sum(position.qty * bt.last_prices[symbol]
                      for symbol, position in bt.positions.items()))
    assert bt.cash == pytest.approx(bt.initial_cash + sum(t.net_pnl for t in bt.trades)
                                  - sum(p.cost_basis for p in bt.positions.values()))
    assert isinstance(bt.pending_entries, dict) and isinstance(bt.pending_exits, dict)


def test_actual_nonempty_csv_exports_and_costs(normal_module):
    context = normal_module
    assert run_caller(context, trade_cost=2.5, slippage=0.002) == {"tested": 1, "skipped": 0}
    bt = context.instances[0]
    trades = pd.read_csv(context.storage / "data" / "trades_log.csv")
    assert len(trades) and list(trades) == list(context.module.Trade.__dataclass_fields__)
    assert bt.trade_cost == 2.5 and bt.slippage == 0.002
    np.testing.assert_allclose(trades.pnl, trades.net_pnl)
    np.testing.assert_allclose(trades.net_pnl, trades.gross_pnl - trades.entry_fee
                               - trades.entry_slippage - trades.exit_fee - trades.exit_slippage)
    equity = pd.read_csv(context.storage / "data" / "equity_curve.csv")
    summary = pd.read_csv(context.storage / "data" / "backtest_results.csv")
    assert list(equity) == ["date", "equity"]
    assert summary.net_pnl.sum() == pytest.approx(sum(t.net_pnl for t in bt.trades))
    assert (context.storage / "data" / "exit_reason_metrics.csv").is_file()


def fail_main_connection_cleanup(context, monkeypatch):
    """Fail only the CLI-owned connection, leaving bar-loader cleanup ordinary."""
    connect = context.module.db.get_db_conn

    def connect_with_failing_close():
        is_main_connection = not context.store.connections
        connection = connect()
        if is_main_connection:
            connection.close = Mock(side_effect=OSError("synthetic close failure"))
        return connection

    monkeypatch.setattr(context.module.db, "get_db_conn", connect_with_failing_close)


def test_main_success_survives_connection_close_failure(normal_module, monkeypatch):
    context = normal_module
    fail_main_connection_cleanup(context, monkeypatch)

    assert context.module.main(cli_arguments(context)) == 0
    assert context.instances and context.store.outputs
    context.store.connections[0].close.assert_called_once_with()
    assert all(connection.closed for connection in context.store.connections[1:])
    log = (context.storage / "backtest.log").read_text(encoding="utf-8")
    assert "BACKTEST_CONNECTION_CLEANUP_FAILED: synthetic close failure" in log
    assert "Backtest failed:" not in log


@pytest.mark.parametrize("failure", ["simulation", "output"])
def test_main_primary_failure_survives_connection_close_failure(
    normal_module, monkeypatch, failure
):
    context = normal_module
    fail_main_connection_cleanup(context, monkeypatch)
    if failure == "simulation":
        def fail_run(self):
            raise RuntimeError("synthetic simulation failure")
        monkeypatch.setattr(context.bt_type, "run", fail_run)
        expected_error = "synthetic simulation failure"
    else:
        def fail_output(*args):
            raise OSError("synthetic output failure")
        monkeypatch.setattr(context.module.db, "insert_backtest_results", fail_output)
        expected_error = "synthetic output failure"

    assert context.module.main(cli_arguments(context)) == 1
    context.store.connections[0].close.assert_called_once_with()
    assert context.instances and not context.store.outputs
    assert all(connection.closed for connection in context.store.connections[1:])
    log = (context.storage / "backtest.log").read_text(encoding="utf-8")
    assert f"Backtest failed: {expected_error}" in log
    assert "BACKTEST_CONNECTION_CLEANUP_FAILED: synthetic close failure" in log


def test_main_success_with_ordinary_connection_cleanup(normal_module):
    context = normal_module

    assert context.module.main(cli_arguments(context)) == 0
    assert context.instances and context.store.outputs
    assert all(connection.closed for connection in context.store.connections)
    log = (context.storage / "backtest.log").read_text(encoding="utf-8")
    assert "BACKTEST_CONNECTION_CLEANUP_FAILED" not in log
    assert "Backtest failed:" not in log


@pytest.mark.parametrize("pending", ["entry", "exit"])
def test_zero_trade_run_succeeds_and_exports_schema(normal_module, pending):
    context = normal_module
    frame = synthetic_bars(1 if pending == "entry" else 2)
    if pending == "exit":
        frame.iloc[1] = [100, 101, 89, 90, 5000]
    context.store.bars = {"AAA": frame}
    context.store.run_date = frame.index[-1].date()
    context.module.CONFIG = dict(use_trailing_stop=False, atr_multiple=0,
                                 enable_macd_exit=False, enable_candlestick_exit=False)
    assert run_caller(context, lookback_days=len(frame)) == {"tested": 1, "skipped": 0}
    bt = context.instances[0]
    assert not bt.trades
    trades = pd.read_csv(context.storage / "data" / "trades_log.csv")
    assert trades.empty and list(trades) == list(context.module.Trade.__dataclass_fields__)
    summary = pd.read_csv(context.storage / "data" / "backtest_results.csv")
    assert len(summary) == 1 and summary.trades.iloc[0] == 0
    assert not pd.read_csv(context.storage / "data" / "equity_curve.csv").empty
    assert bt.pending_entries if pending == "entry" else bt.pending_exits
    assert context.module.main(cli_arguments(context)) == 0


def test_future_append_preserves_earlier_features_and_entry_signals(normal_module):
    context = normal_module
    prefix = synthetic_bars()
    extended = synthetic_bars(260)
    prefix_features = context.module.compute_indicators(prefix)
    extended_features = context.module.compute_indicators(extended)
    prefix_features["score"] = context.module.composite_score(prefix_features)
    extended_features["score"] = context.module.composite_score(extended_features)
    prefix_features = context.module.prepare_series(prefix_features)
    extended_features = context.module.prepare_series(extended_features)
    pd.testing.assert_frame_equal(prefix_features, extended_features.loc[prefix_features.index])
    # Record actual queued entry signals from each run, without replacing run().
    def signal_trace(frame):
        bt = context.module.PortfolioBacktester({"AAA": frame})
        signals = []
        class EntryQueue(dict):
            def __setitem__(self, symbol, row):
                signals.append((symbol, row.name, row["score"]))
                super().__setitem__(symbol, row)
        bt.pending_entries = EntryQueue()
        bt.run()
        return signals
    prefix_signals = signal_trace(prefix_features)
    extended_signals = signal_trace(extended_features)
    assert prefix_signals
    assert prefix_signals == [event for event in extended_signals if event[1] <= prefix.index[-1]]
    # The actual loader/caller also clips future rows at the requested date.
    context.store.bars["AAA"] = extended
    run_caller(context)
    pd.testing.assert_frame_equal(context.instances[-1].data["AAA"], prefix_features,
                                  check_freq=False, check_dtype=False)


@pytest.mark.parametrize("values", [(-1, 0), (float("nan"), 0), (float("inf"), 0),
                                    (0, -0.001), (0, 1), (0, float("nan")),
                                    (0, float("inf")), (True, 0), (0, None)])
def test_invalid_costs_fail_before_input_access(normal_module, values):
    context = normal_module
    with pytest.raises(ValueError):
        run_caller(context, trade_cost=values[0], slippage=values[1])
    assert not context.store.queries and not context.instances


@pytest.mark.parametrize("args", [["--trade-cost", "-1"], ["--trade-cost", "nan"],
                                 ["--slippage", "1"], ["--slippage", "inf"]])
def test_cli_rejects_invalid_costs(normal_module, args):
    with pytest.raises(SystemExit) as captured:
        normal_module.module.main(args)
    assert captured.value.code == 2 and not normal_module.store.queries


def test_legacy_cost_defaults(normal_module):
    context = normal_module
    assert context.module._parse_args([]).trade_cost == context.module._parse_args([]).slippage == 0
    run_caller(context)
    assert context.instances[0].trade_cost == context.instances[0].slippage == 0


@pytest.mark.parametrize("failure", ["empty", "invalid_symbol", "missing_bars", "short_history"])
def test_no_evaluated_symbols_is_failure(normal_module, failure):
    context = normal_module
    symbols = [] if failure == "empty" else ["INVALID!" if failure == "invalid_symbol" else "AAA"]
    if failure == "missing_bars":
        context.store.bars = {}
    if failure == "short_history":
        context.store.bars = {"AAA": synthetic_bars(1)}
    with pytest.raises(ValueError, match="No symbols were successfully evaluated"):
        context.module.run_backtest(symbols, run_date=context.store.run_date,
                                    lookback_days=240, min_history_bars=200, enable_db=False)
    assert not context.instances


@pytest.mark.parametrize("failure", ["no_candidates", "missing_bars", "simulation", "db_output",
                                    "db_output_exception",
                                    "db_disabled", "db_connection", "run_date"])
def test_actual_main_failure_outcomes(normal_module, monkeypatch, failure):
    context = normal_module
    args = cli_arguments(context)
    if failure == "no_candidates":
        context.store.symbols = []
    elif failure == "missing_bars":
        context.store.bars = {}
    elif failure == "simulation":
        def fail_run(self):
            raise RuntimeError("synthetic simulation failure")
        monkeypatch.setattr(context.bt_type, "run", fail_run)
    elif failure == "db_output":
        context.store.insert_result = False
    elif failure == "db_output_exception":
        def failed_insert(*args):
            raise OSError("synthetic database output failure")
        monkeypatch.setattr(context.module.db, "insert_backtest_results", failed_insert)
    elif failure == "db_disabled":
        context.store.enabled = False
    elif failure == "db_connection":
        monkeypatch.setattr(context.module.db, "get_db_conn", lambda: None)
    else:
        args = ["--run-date", "not-a-date"]
    assert context.module.main(args) != 0
    assert all(connection.closed for connection in context.store.connections)


def test_export_error_propagates_and_write_guard_is_active(normal_module):
    context = normal_module
    with pytest.raises(AssertionError, match="Write outside temporary storage"):
        builtins.open(REPO / "unauthorized-output.txt", "w")
    # A real filesystem error in the unchanged writer, entirely inside storage.
    (context.storage / "data").rmdir()
    (context.storage / "data").write_text("not a directory", encoding="utf-8")
    with pytest.raises(OSError):
        run_caller(context)
