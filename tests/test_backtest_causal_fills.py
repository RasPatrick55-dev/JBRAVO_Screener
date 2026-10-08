"""Synthetic tests of the actual simulator and caller, without runtime imports.

backtest's module initialization loads credentials/config and initializes logging.
Compile its simulator and run_backtest definitions; mock external caller boundaries.
Run with --noconftest to avoid the repository's Alpaca-dependent conftest import.
"""

import ast
from dataclasses import dataclass
from datetime import date, datetime, timezone
import logging
import math
import os
from pathlib import Path
import re
import socket
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(scope="module")
def simulator():
    source = Path(__file__).parents[1] / "scripts" / "backtest.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = {"Position", "Trade", "evaluate_exit_signals", "PortfolioBacktester",
             "_validate_execution_costs", "run_backtest"}
    definitions = [node for node in parsed.body if getattr(node, "name", None) in names]
    assert {node.name for node in definitions} == names
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    isolated = ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[]))
    module = ModuleType("jbravo_causal_simulator_tests")
    module.__dict__.update(
        dataclass=dataclass, math=math, np=np, pd=pd, Dict=Dict, List=List,
        Optional=Optional, logger=logging.getLogger("jbravo_causal_simulator_tests"),
        date=date, datetime=datetime, timezone=timezone, os=os, re=re,
    )
    sys.modules[module.__name__] = module
    try:
        exec(compile(isolated, str(source), "exec"), module.__dict__)
        yield module
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in causal-fill tests")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def bars(*overrides, dates=None):
    rows = []
    for override in overrides:
        row = dict(open=100.0, high=100.0, low=100.0, close=100.0,
                   score=np.nan, ATR14=np.nan, ema20=10.0, rsi=50.0)
        row.update(override)
        rows.append(row)
    index = pd.to_datetime(dates) if dates else pd.date_range("2024-01-02", periods=len(rows))
    return pd.DataFrame(rows, index=index)


def backtester(simulator, data, **overrides):
    options = dict(initial_cash=10_000.0, alloc_pct=0.1, top_n=1, max_positions=1,
                   trail_pct=None, max_hold_days=100, atr_multiple=0,
                   enable_macd_exit=False, enable_partial_exit=False,
                   enable_candlestick_exit=False, enable_ema_exit=False)
    options.update(overrides)
    return simulator.PortfolioBacktester(data, **options)


def seed(simulator, bt, *, atr_stop=None, trailing_stop=None, qty=10):
    bt.positions["AAA"] = simulator.Position(
        symbol="AAA", qty=qty, entry_price=100.0, entry_time=pd.Timestamp("2024-01-01"),
        highest_close=100.0, max_price=100.0, atr_stop=atr_stop, trailing_stop=trailing_stop,
    )
    bt.cash -= qty * 100.0


def test_entry_fills_next_open_using_signal_atr(simulator):
    frame = bars(dict(open=80, high=101, low=79, close=100, score=5, ATR14=5),
                 dict(open=120, high=125, low=119, close=124, ATR14=1000))
    bt = backtester(simulator, {"AAA": frame}, atr_multiple=1)
    bt.run()
    pos = bt.positions["AAA"]
    assert (pos.entry_time, pos.entry_price) == (frame.index[1], 120)
    assert pos.atr_stop == 115  # Signal ATR, never the fill bar's completed ATR.
    assert bt.equity_curve[0][1] == bt.initial_cash


def test_terminal_signal_remains_unfilled(simulator):
    bt = backtester(simulator, {"AAA": bars(dict(score=5))})
    bt.run()
    assert not bt.positions and not bt.trades
    assert list(bt.pending_entries) == ["AAA"]
    assert bt.cash == bt.initial_cash


def test_terminal_exit_remains_pending_without_forced_liquidation(simulator):
    bt = backtester(simulator, {"AAA": bars(dict(ema20=110))}, enable_ema_exit=True)
    seed(simulator, bt)
    bt.run()
    assert bt.pending_exits == {"AAA": ["EMA20_BREAK"]}
    assert bt.positions["AAA"].qty == 10 and not bt.trades
    assert bt.equity_curve[-1][1] == 10_000
    assert bt.results().empty
    assert list(bt.results().columns) == TRADE_COLUMNS


def test_ranking_order_is_preserved(simulator):
    bt = backtester(simulator, {
        "AAA": bars(dict(score=1), {}), "BBB": bars(dict(score=2), {}),
    })
    bt.run()
    assert list(bt.positions) == ["BBB"]


def test_trailing_tightening_cannot_trigger_on_its_source_bar(simulator):
    frame = bars(dict(high=110, low=98, close=109),
                 dict(open=109, high=109, low=108, close=108.5))
    bt = backtester(simulator, {"AAA": frame}, trail_pct=0.03)
    seed(simulator, bt, trailing_stop=97)
    bt.run()
    assert len(bt.trades) == 1
    trade = bt.trades[0]
    assert trade.exit_time == frame.index[1]
    assert trade.exit_price == pytest.approx(108.9)
    assert trade.exit_reason == "TRAIL_STOP"


def test_atr_tightening_cannot_trigger_on_its_source_bar(simulator):
    frame = bars(dict(high=106, low=96, close=105, ATR14=2),
                 dict(open=104, high=105, low=102, close=104))
    bt = backtester(simulator, {"AAA": frame}, atr_multiple=1)
    seed(simulator, bt, atr_stop=95)
    bt.run()
    assert bt.trades[0].exit_time == frame.index[1]
    assert bt.trades[0].exit_price == 103


@pytest.mark.parametrize("atr,trail,opening,low,price,reason", [
    (95, 97, 90, 89, 90, "TRAIL_STOP"),
    (98, 97, 99, 94, 98, "ATR_STOP"),
    (95, 95, 100, 94, 95, "ATR_STOP"),
])
def test_gap_and_multiple_stop_precedence(simulator, atr, trail, opening, low, price, reason):
    frame = bars(dict(open=opening, high=max(opening, 100), low=low, close=opening))
    bt = backtester(simulator, {"AAA": frame}, trail_pct=0.03)
    seed(simulator, bt, atr_stop=atr, trailing_stop=trail)
    bt.run()
    assert len(bt.trades) == 1
    assert (bt.trades[0].exit_price, bt.trades[0].exit_reason) == (price, reason)
    assert bt.trades[0].mfe_pct == 0  # No assumed pre-stop intrabar high.


def test_full_exit_suppresses_competing_partial_and_fills_next_open(simulator):
    frame = bars(dict(high=106, close=106, ema20=107),
                 dict(open=108, high=109, low=107, close=108))
    bt = backtester(simulator, {"AAA": frame}, enable_ema_exit=True, enable_partial_exit=True)
    seed(simulator, bt)
    bt.run()
    assert len(bt.trades) == 1
    trade = bt.trades[0]
    assert (trade.qty, trade.exit_price, trade.exit_reason) == (10, 108, "EMA20_BREAK")
    assert trade.exit_time == frame.index[1]


def test_existing_stop_wins_over_same_bar_close_signals(simulator):
    bt = backtester(simulator, {"AAA": bars(dict(high=108, low=94, close=106, ema20=107))},
                    enable_ema_exit=True, enable_partial_exit=True)
    seed(simulator, bt, atr_stop=95)
    bt.run()
    assert [(t.qty, t.exit_price, t.exit_reason) for t in bt.trades] == [(10, 95, "ATR_STOP")]
    assert bt.trades[0].mfe_pct == 0
    assert not bt.pending_exits


@pytest.mark.parametrize("queued", [["EMA20_BREAK"], ["PARTIAL_5PCT"]])
def test_opening_gap_stop_precedes_queued_exits(simulator, queued):
    bt = backtester(simulator, {"AAA": bars(dict(open=90, high=93, low=89, close=92))})
    seed(simulator, bt, atr_stop=95)
    bt.pending_exits["AAA"] = queued
    bt.run()
    assert [(t.qty, t.exit_price, t.exit_reason) for t in bt.trades] == [(10, 90, "ATR_STOP")]


def test_queued_full_exit_at_open_precedes_later_intrabar_stop(simulator):
    bt = backtester(simulator, {"AAA": bars(dict(open=102, high=103, low=94, close=96))})
    seed(simulator, bt, atr_stop=95)
    bt.pending_exits["AAA"] = ["EMA20_BREAK"]
    bt.run()
    assert (bt.trades[0].exit_price, bt.trades[0].exit_reason) == (102, "EMA20_BREAK")


def test_queued_partial_at_open_then_existing_stop_closes_remainder(simulator):
    bt = backtester(simulator, {"AAA": bars(dict(open=106, high=108, low=96, close=100))},
                    trail_pct=0.03, enable_partial_exit=True)
    seed(simulator, bt, trailing_stop=97)
    bt.pending_exits["AAA"] = ["PARTIAL_5PCT"]
    bt.run()
    assert [(t.qty, t.exit_price, t.exit_reason) for t in bt.trades] == [
        (5, 106, "PARTIAL_5PCT"), (5, 97, "TRAIL_STOP"),
    ]
    assert not bt.positions
    assert bt.cash == 10_015


def test_partial_signal_fills_once_on_next_bar(simulator):
    frame = bars(dict(high=106, close=106),
                 dict(open=107, high=108, low=106, close=107),
                 dict(open=108, high=109, low=107, close=108))
    bt = backtester(simulator, {"AAA": frame}, enable_partial_exit=True)
    seed(simulator, bt)
    bt.run()
    assert [(t.qty, t.exit_price, t.exit_time) for t in bt.trades] == [(5, 107, frame.index[1])]
    assert bt.positions["AAA"].qty == 5
    assert not bt.pending_exits


def test_entry_stop_can_trigger_on_fill_bar_without_using_its_atr(simulator):
    frame = bars(dict(score=5, ATR14=5), dict(open=110, high=112, low=104, close=108))
    bt = backtester(simulator, {"AAA": frame}, atr_multiple=1)
    bt.run()
    assert (bt.trades[0].entry_price, bt.trades[0].exit_price) == (110, 105)
    assert bt.trades[0].entry_time == bt.trades[0].exit_time == frame.index[1]


def test_missing_bar_defers_entry_and_reserves_capacity(simulator):
    aaa = bars(dict(score=5), dict(open=110, high=111, low=109, close=110),
               dates=["2024-01-02", "2024-01-04"])
    bbb = bars({}, dict(score=100), {})
    bt = backtester(simulator, {"AAA": aaa, "BBB": bbb})
    bt.run()
    assert list(bt.positions) == ["AAA"]
    assert bt.positions["AAA"].entry_time == pd.Timestamp("2024-01-04")
    assert bt.positions["AAA"].entry_price == 110
    assert bt.equity_curve[1][1] == bt.initial_cash


def test_missing_bar_carries_last_close_without_inventing_stop_fill(simulator):
    aaa = bars(dict(high=103, close=102), dict(open=103, high=105, low=102, close=104),
               dates=["2024-01-02", "2024-01-04"])
    bt = backtester(simulator, {"AAA": aaa, "BBB": bars({}, {}, {})})
    seed(simulator, bt, atr_stop=95)
    bt.run()
    assert [equity for _, equity in bt.equity_curve] == [10_020, 10_020, 10_040]
    assert not bt.trades


def test_missing_bar_defers_discretionary_exit(simulator):
    aaa = bars(dict(high=106, close=106, ema20=107),
               dict(open=110, high=111, low=109, close=110),
               dates=["2024-01-02", "2024-01-04"])
    bt = backtester(simulator, {"AAA": aaa, "BBB": bars({}, {}, {})}, enable_ema_exit=True)
    seed(simulator, bt)
    bt.run()
    assert (bt.trades[0].exit_time, bt.trades[0].exit_price) == (aaa.index[1], 110)


@pytest.mark.parametrize("invalid", [dict(open=np.nan), dict(open=0), dict(high=99),
                                    dict(low=101), dict(close=np.inf)])
def test_invalid_ohlc_fails_instead_of_inventing_prices(simulator, invalid):
    bt = backtester(simulator, {"AAA": bars(invalid)})
    with pytest.raises(ValueError, match="consistent OHLC"):
        bt.run()
    assert bt.cash == bt.initial_cash and not bt.positions


def test_cash_and_position_invariants_with_existing_cost_model(simulator):
    frame = bars(dict(score=5), dict(ema20=110),
                 dict(open=110, high=110, low=110, close=110))
    bt = backtester(simulator, {"AAA": frame}, initial_cash=1000, alloc_pct=0.5,
                    trade_cost=2, slippage=0.01, enable_ema_exit=True)
    bt.run()
    assert len(bt.trades) == 1 and bt.trades[0].qty == 5
    assert not bt.positions and not bt.pending_exits
    assert bt.cash == pytest.approx(1000 - 5 * 100 * 1.01 - 2 + 5 * 110 * 0.99 - 2)
    assert all(equity >= 0 for _, equity in bt.equity_curve)
    assert bt.trades[0].gross_pnl == 50
    assert bt.trades[0].pnl == bt.trades[0].net_pnl == pytest.approx(35.5)


def test_unaffordable_entry_attempt_does_not_create_position(simulator):
    bt = backtester(simulator, {"AAA": bars(dict(score=5), {})}, initial_cash=100,
                    alloc_pct=1, trade_cost=2)
    bt.run()
    assert bt.cash == 100 and not bt.positions and not bt.pending_entries


def test_source_signals_and_bars_are_not_mutated(simulator):
    frame = bars(dict(score=5), {})
    before = frame.copy(deep=True)
    bt = backtester(simulator, {"AAA": frame})
    bt.run()
    pd.testing.assert_frame_equal(frame, before)


def test_missing_open_column_is_not_replaced_with_close(simulator):
    frame = bars(dict(score=5)).drop(columns=["open"])
    bt = backtester(simulator, {"AAA": frame})
    with pytest.raises(ValueError, match="consistent OHLC"):
        bt.run()
    assert not bt.positions and bt.cash == bt.initial_cash


def test_single_share_partial_does_not_sell_or_charge_a_fee(simulator):
    bt = backtester(simulator, {"AAA": bars({})}, enable_partial_exit=True, trade_cost=2)
    seed(simulator, bt, qty=1)
    bt.pending_exits["AAA"] = ["PARTIAL_5PCT"]
    bt.run()
    assert bt.positions["AAA"].qty == 1
    assert bt.cash == 9900 and not bt.trades


TRADE_COLUMNS = [
    "symbol", "entry_time", "exit_time", "entry_price", "exit_price", "qty",
    "pnl", "exit_reason", "mfe_pct", "exit_pct", "exit_efficiency",
    "gross_pnl", "entry_fee", "entry_slippage", "exit_fee", "exit_slippage", "net_pnl",
]


def test_empty_results_preserve_trade_schema(simulator):
    bt = backtester(simulator, {"AAA": bars({})})
    bt.run()
    result = bt.results()
    assert result.empty
    assert list(result.columns) == TRADE_COLUMNS
    assert not bt.trades and not bt.positions


def test_nonempty_results_preserve_existing_schema_and_values(simulator):
    bt = backtester(simulator, {"AAA": bars(dict(low=94))})
    seed(simulator, bt, atr_stop=95)
    bt.run()
    result = bt.results()
    assert len(result) == 1
    assert list(result.columns) == TRADE_COLUMNS
    pd.testing.assert_frame_equal(result, pd.DataFrame(bt.trades))


@pytest.mark.parametrize("pending", ["entry", "exit"])
def test_actual_caller_exports_zero_trade_summary_with_terminal_signal(
    simulator, monkeypatch, pending,
):
    # Execute the unchanged caller body and real simulator, not a fake result object.
    # Indicator/provider/DB/export boundaries stay mocked; no file is written.
    frame = (bars({}, dict(score=5)) if pending == "entry"
             else bars(dict(score=5), dict(ema20=110)))
    load = Mock(return_value=frame.copy(deep=True))
    backfill = Mock(side_effect=AssertionError("Unexpected provider backfill"))
    insert = Mock(return_value=True)
    export = Mock()
    monkeypatch.setattr(simulator, "BASE_DIR", "synthetic-only", raising=False)
    monkeypatch.setattr(simulator, "CONFIG", {
        "use_trailing_stop": False, "atr_multiple": 0, "max_hold_days": 100,
        "enable_macd_exit": False, "enable_partial_exit": False,
        "enable_candlestick_exit": False, "enable_ema_exit": True,
    }, raising=False)
    monkeypatch.setattr(simulator, "load_bars_from_db", load, raising=False)
    monkeypatch.setattr(simulator, "_maybe_backfill_bars", backfill, raising=False)
    monkeypatch.setattr(simulator, "compute_indicators", lambda df: df, raising=False)
    monkeypatch.setattr(simulator, "composite_score", lambda df: df["score"], raising=False)
    monkeypatch.setattr(simulator, "prepare_series", lambda df: df, raising=False)
    monkeypatch.setattr(simulator, "db", SimpleNamespace(insert_backtest_results=insert),
                        raising=False)
    monkeypatch.setattr(simulator, "write_csv_atomic", export, raising=False)
    actual_backtester = simulator.PortfolioBacktester
    instances = []

    def capture_backtester(*args, **kwargs):
        bt = actual_backtester(*args, **kwargs)
        instances.append(bt)
        return bt

    monkeypatch.setattr(simulator, "PortfolioBacktester", capture_backtester)
    run_date = date(2024, 1, 3)
    outcome = simulator.run_backtest(
        ["AAA"], run_date=run_date, lookback_days=2, min_history_bars=2,
        export_csv=True, enable_db=True,
    )
    assert outcome == {"tested": 1, "skipped": 0}
    load.assert_called_once_with("AAA", end_date=run_date)
    backfill.assert_not_called()
    assert len(instances) == 1
    bt = instances[0]
    assert not bt.trades
    if pending == "entry":
        assert list(bt.pending_entries) == ["AAA"] and not bt.positions
        assert bt.cash == bt.initial_cash
    else:
        assert bt.pending_exits == {"AAA": ["EMA20_BREAK"]}
        assert bt.positions["AAA"].qty > 0

    assert export.call_count == 3
    exported = {os.path.basename(call.args[0]): call.args[1]
                for call in export.call_args_list}
    assert set(exported) == {"trades_log.csv", "equity_curve.csv", "backtest_results.csv"}
    trades = exported["trades_log.csv"]
    assert trades.empty and list(trades.columns) == TRADE_COLUMNS
    pd.testing.assert_frame_equal(exported["equity_curve.csv"], bt.equity().reset_index())
    summary = exported["backtest_results.csv"]
    assert len(summary) == 1 and summary.iloc[0]["symbol"] == "AAA"
    for column in ["trades", "wins", "losses", "net_pnl", "win_rate", "expectancy",
                   "profit_factor", "max_drawdown", "sharpe", "sortino"]:
        assert summary.iloc[0][column] == 0
    assert summary.iloc[0]["run_date"] == run_date
    assert summary.iloc[0]["symbols_tested"] == 1
    assert summary.iloc[0]["timestamp"]
    insert.assert_called_once()
    assert insert.call_args.args[0] == run_date
    pd.testing.assert_frame_equal(insert.call_args.args[1], summary)
