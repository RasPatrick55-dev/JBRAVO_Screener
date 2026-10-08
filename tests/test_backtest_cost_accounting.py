"""Offline cost ledger checks using actual definitions and mocked caller boundaries."""

import ast
from datetime import date
from decimal import Decimal
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from test_backtest_causal_fills import backtester, bars, forbid_network, simulator


pytestmark = pytest.mark.alpaca_optional
ABS_TOL = 1e-9  # Currency units in these small synthetic portfolios; no rounding in the ledger.


def assert_reconciles(bt):
    realized = sum(trade.net_pnl for trade in bt.trades)
    basis = sum(pos.qty * pos.entry_price + pos.entry_fee_remaining
                + pos.entry_slippage_remaining for pos in bt.positions.values())
    marked = sum(pos.qty * bt.last_prices.get(symbol, pos.entry_price)
                 for symbol, pos in bt.positions.items())
    assert bt.cash == pytest.approx(bt.initial_cash + realized - basis, rel=0, abs=ABS_TOL)
    if bt.equity_curve:
        assert bt.equity_curve[-1][1] == pytest.approx(bt.cash + marked, rel=0, abs=ABS_TOL)
        assert bt.equity_curve[-1][1] - bt.initial_cash == pytest.approx(
            realized + marked - basis, rel=0, abs=ABS_TOL,
        )
    for trade in bt.trades:
        assert trade.pnl == trade.net_pnl
        assert trade.net_pnl == pytest.approx(
            trade.gross_pnl - trade.entry_fee - trade.entry_slippage
            - trade.exit_fee - trade.exit_slippage, rel=0, abs=ABS_TOL,
        )


def partial_bars():
    return bars(dict(score=5), dict(high=106, close=106),
                dict(open=110, high=110, low=109, close=109, ema20=110),
                dict(open=90, high=90, low=90, close=90))


@pytest.mark.parametrize("exit_price", [90, 100, 110])
def test_zero_cost_compatibility(simulator, exit_price):
    frame = bars(dict(score=5), dict(ema20=110),
                 dict(open=exit_price, high=exit_price, low=exit_price, close=exit_price))
    bt = backtester(simulator, {"AAA": frame}, enable_ema_exit=True)
    bt.run()
    trade = bt.trades[0]
    assert trade.pnl == trade.net_pnl == trade.gross_pnl == (exit_price - 100) * 10
    assert trade.entry_fee == trade.entry_slippage == trade.exit_fee == trade.exit_slippage == 0
    assert not bt.positions
    assert_reconciles(bt)


@pytest.mark.parametrize("alloc,quantity", [(0.10, 10), (0.09, 9)])
def test_even_and_odd_partial_then_final_cost_allocation(simulator, alloc, quantity):
    bt = backtester(simulator, {"AAA": partial_bars()}, alloc_pct=alloc,
                    trade_cost=3, slippage=0.01, enable_partial_exit=True, enable_ema_exit=True)
    bt.run()
    partial, final = bt.trades
    assert (partial.qty, final.qty) == (quantity // 2, quantity - quantity // 2)
    assert (partial.exit_reason, final.exit_reason) == ("PARTIAL_5PCT", "EMA20_BREAK")
    assert partial.entry_fee == pytest.approx(3 * partial.qty / quantity)
    assert final.entry_fee == 3 - partial.entry_fee
    assert final.entry_slippage == quantity - partial.entry_slippage
    assert sum(t.entry_fee for t in bt.trades) == 3
    assert sum(t.entry_slippage for t in bt.trades) == quantity
    assert [t.exit_fee for t in bt.trades] == [3, 3]
    assert partial.exit_slippage == pytest.approx(partial.qty * 110 * 0.01)
    assert final.exit_slippage == pytest.approx(final.qty * 90 * 0.01)
    assert not bt.positions and not bt.pending_exits
    assert_reconciles(bt)


def test_final_exit_takes_residual_after_repeated_fractional_allocations(simulator):
    bt = backtester(simulator, {"AAA": bars({})}, alloc_pct=0.07, trade_cost=0.1, slippage=0.001)
    row = bars({}).iloc[0]
    when = pd.Timestamp("2024-01-02")
    bt._open_position("AAA", row, when, row)
    pos = bt.positions["AAA"]
    fee, slippage = pos.entry_fee_remaining, pos.entry_slippage_remaining
    for _ in range(3):
        bt._scale_out_position("AAA", 103, when, "PARTIAL_5PCT")
        assert_reconciles(bt)
    remaining_fee, remaining_slippage = pos.entry_fee_remaining, pos.entry_slippage_remaining
    bt._close_position("AAA", 99, when, "FINAL")
    assert bt.trades[-1].entry_fee == remaining_fee
    assert bt.trades[-1].entry_slippage == remaining_slippage
    assert sum(t.entry_fee for t in bt.trades) == pytest.approx(fee, rel=0, abs=ABS_TOL)
    assert sum(t.entry_slippage for t in bt.trades) == pytest.approx(slippage, rel=0, abs=ABS_TOL)
    assert_reconciles(bt)


@pytest.mark.parametrize("alloc,quantity", [(0.10, 10), (0.09, 9)])
def test_partial_exit_keeps_remaining_cost_basis_and_unrealized_mark(simulator, alloc, quantity):
    bt = backtester(simulator, {"AAA": partial_bars().iloc[:3]}, alloc_pct=alloc,
                    trade_cost=3, slippage=0.01, enable_partial_exit=True, enable_ema_exit=True)
    bt.run()
    assert len(bt.trades) == 1 and bt.pending_exits == {"AAA": ["EMA20_BREAK"]}
    pos, partial = bt.positions["AAA"], bt.trades[0]
    assert pos.qty == quantity - quantity // 2
    assert pos.entry_fee_remaining == 3 - partial.entry_fee
    assert pos.entry_slippage_remaining == quantity - partial.entry_slippage
    assert pos.cost_basis == pytest.approx(
        (quantity * 101 + 3) * pos.qty / quantity, rel=0, abs=ABS_TOL,
    )
    assert bt.equity_curve[-1][1] == pytest.approx(bt.cash + pos.qty * 109)
    assert_reconciles(bt)


def test_single_share_skipped_partial_charges_nothing(simulator):
    frame = bars(dict(score=5), dict(high=106, close=106),
                 dict(open=107, high=107, low=107, close=107))
    bt = backtester(simulator, {"AAA": frame}, alloc_pct=0.01, trade_cost=3,
                    slippage=0.01, enable_partial_exit=True)
    bt.run()
    assert bt.positions["AAA"].qty == 1
    assert not bt.trades
    assert bt.cash == 9896
    assert bt.positions["AAA"].cost_basis == 104
    assert_reconciles(bt)


@pytest.mark.parametrize("opening,initial", [(100, 100), (1000, 100)])
def test_skipped_entry_has_no_cost(simulator, opening, initial):
    frame = bars(dict(score=5), dict(open=opening, high=opening, low=opening, close=opening))
    bt = backtester(simulator, {"AAA": frame}, initial_cash=initial, alloc_pct=1,
                    trade_cost=3, slippage=0.01)
    bt.run()
    assert not bt.positions and not bt.trades
    assert bt.cash == initial
    assert_reconciles(bt)


def test_multiple_symbols_with_realized_and_open_cost_basis(simulator):
    other = bars(dict(score=4), {}, {}, dict(high=105, close=105))
    bt = backtester(simulator, {"AAA": partial_bars(), "BBB": other}, top_n=2,
                    max_positions=2, trade_cost=3, slippage=0.01,
                    enable_partial_exit=True, enable_ema_exit=True)
    bt.run()
    assert {t.symbol for t in bt.trades} == {"AAA"}
    assert set(bt.positions) == {"BBB"}
    assert bt.positions["BBB"].entry_fee_remaining == 3
    assert bt.positions["BBB"].entry_slippage_remaining > 0
    assert_reconciles(bt)
    metrics = bt.metrics()
    assert metrics["Realized Net P&L"] == round(sum(t.net_pnl for t in bt.trades), 2)
    assert metrics["Remaining Cost Basis"] == round(bt.positions["BBB"].cost_basis, 2)
    assert metrics["Unrealized P&L"] == round(
        bt.positions["BBB"].qty * 105 - bt.positions["BBB"].cost_basis, 2,
    )


def test_empty_results_retain_cost_columns_and_single_mark_metrics(simulator):
    bt = backtester(simulator, {"AAA": bars({})}, trade_cost=3, slippage=0.01)
    bt.run()
    result = bt.results()
    assert result.empty
    assert {"pnl", "gross_pnl", "net_pnl", "entry_fee", "entry_slippage",
            "exit_fee", "exit_slippage"}.issubset(result.columns)
    assert bt.metrics()["Realized Net P&L"] == 0
    assert bt.metrics()["Remaining Cost Basis"] == 0
    assert all(np.isfinite(v) for v in bt.metrics().values() if v != float("inf"))
    assert_reconciles(bt)


def test_independent_decimal_example_partial_and_final(simulator):
    # Independently calculate expected cash/basis from specified quantities/prices.
    # Decimal arithmetic does not call the simulator's cost-allocation implementation.
    D = Decimal
    bt = backtester(simulator, {"AAA": bars({})}, alloc_pct=0.09,
                    trade_cost=3, slippage=0.01)
    row = bars({}).iloc[0]
    when = pd.Timestamp("2024-01-02")
    bt._open_position("AAA", row, when, row)
    assert bt.positions["AAA"].qty == 9
    entry = D(9) * D(100) * D("1.01") + D(3)
    partial_proceeds = D(4) * D(110) * D("0.99") - D(3)
    allocated_entry = entry * D(4) / D(9)
    partial_net = partial_proceeds - allocated_entry
    remaining_basis = entry - allocated_entry
    bt._scale_out_position("AAA", 110, when, "PARTIAL_5PCT")
    assert bt.cash == pytest.approx(float(D(10000) - entry + partial_proceeds), abs=ABS_TOL, rel=0)
    assert bt.trades[0].net_pnl == pytest.approx(float(partial_net), abs=ABS_TOL, rel=0)
    assert bt.positions["AAA"].cost_basis == pytest.approx(float(remaining_basis), abs=ABS_TOL, rel=0)
    assert_reconciles(bt)
    final_proceeds = D(5) * D(90) * D("0.99") - D(3)
    bt._close_position("AAA", 90, when, "FINAL")
    assert bt.trades[-1].net_pnl == pytest.approx(float(final_proceeds - remaining_basis),
                                               abs=ABS_TOL, rel=0)
    assert bt.cash == pytest.approx(float(D(10000) - entry + partial_proceeds + final_proceeds),
                                   abs=ABS_TOL, rel=0)
    assert bt.cash == pytest.approx(9963.1, abs=ABS_TOL, rel=0)
    assert_reconciles(bt)


@pytest.mark.parametrize("exit_prices", [[100.1], [100.1, 99, 98]])
def test_actual_caller_and_pnl_first_consumer_use_net_results(simulator, monkeypatch, exit_prices):
    rows = []
    for price in exit_prices:
        rows.extend([dict(score=5), dict(ema20=110),
                     dict(open=price, high=price, low=price, close=price)])
    frame = bars(*rows)
    monkeypatch.setattr(simulator, "BASE_DIR", "synthetic-only", raising=False)
    monkeypatch.setattr(simulator, "CONFIG", {
        "use_trailing_stop": False, "atr_multiple": 0, "max_hold_days": 100,
        "enable_macd_exit": False, "enable_partial_exit": False,
        "enable_candlestick_exit": False, "enable_ema_exit": True,
    }, raising=False)
    load = Mock(return_value=frame.copy(deep=True))
    backfill = Mock(side_effect=AssertionError("Unexpected provider backfill"))
    insert, export = Mock(return_value=True), Mock()
    monkeypatch.setattr(simulator, "load_bars_from_db", load, raising=False)
    monkeypatch.setattr(simulator, "_maybe_backfill_bars", backfill, raising=False)
    monkeypatch.setattr(simulator, "compute_indicators", lambda df: df, raising=False)
    monkeypatch.setattr(simulator, "composite_score", lambda df: df["score"], raising=False)
    monkeypatch.setattr(simulator, "prepare_series", lambda df: df, raising=False)
    monkeypatch.setattr(simulator, "db", SimpleNamespace(insert_backtest_results=insert), raising=False)
    monkeypatch.setattr(simulator, "write_csv_atomic", export, raising=False)
    actual_backtester = simulator.PortfolioBacktester
    instances = []

    def with_costs(data, **kwargs):
        # Existing caller defaults remain unchanged. Inject existing constructor
        # parameters at this mocked boundary to exercise nonzero-cost caller data.
        bt = actual_backtester(data, trade_cost=3, slippage=0.01, **kwargs)
        instances.append(bt)
        return bt

    monkeypatch.setattr(simulator, "PortfolioBacktester", with_costs)
    run_date = frame.index[-1].date()
    assert simulator.run_backtest(["AAA"], run_date=run_date, lookback_days=len(frame),
                                 min_history_bars=len(frame)) == {"tested": 1, "skipped": 0}
    backfill.assert_not_called()
    load.assert_called_once_with("AAA", end_date=run_date)
    bt = instances[0]
    assert_reconciles(bt)
    assert bt.trades[0].gross_pnl > 0 and bt.trades[0].net_pnl < 0
    assert bt.metrics()["Win Rate"] == 0 and bt.metrics()["Profit Factor"] == 0
    exported = {os.path.basename(call.args[0]): call.args[1] for call in export.call_args_list}
    assert set(exported) == {"trades_log.csv", "equity_curve.csv", "backtest_results.csv",
                             "exit_reason_metrics.csv"}
    trades = exported["trades_log.csv"]
    expected_exits = [float(Decimal(t.qty) * (Decimal(str(price)) - Decimal(100))
                           - Decimal(t.qty) * Decimal(100) * Decimal("0.01")
                           - Decimal(t.qty) * Decimal(str(price)) * Decimal("0.01") - Decimal(6))
                      for t, price in zip(bt.trades, exit_prices)]
    assert len(bt.trades) == len(exit_prices)
    assert trades["net_pnl"].tolist() == pytest.approx(expected_exits, abs=ABS_TOL, rel=0)
    assert trades["pnl"].tolist() == trades["net_pnl"].tolist()
    expected_net = sum(expected_exits)
    summary = exported["backtest_results.csv"].iloc[0]
    assert summary["net_pnl"] == pytest.approx(expected_net)
    assert summary["expectancy"] == pytest.approx(expected_net / len(exit_prices))
    assert (summary["trades"], summary["wins"], summary["losses"]) == (
        len(exit_prices), 0, len(exit_prices),
    )
    assert summary["win_rate"] == summary["profit_factor"] == 0
    # Every exit is a net loss: drawdown starts at zero, so the first loss
    # and the complete all-loss sequence must both count, in currency units.
    assert summary["max_drawdown"] == pytest.approx(expected_net, abs=ABS_TOL, rel=0)
    reasons = exported["exit_reason_metrics.csv"].iloc[0]
    assert reasons["exit_reason"] == "EMA20_BREAK"
    assert reasons["total_pnl"] == pytest.approx(expected_net)
    assert reasons["avg_pnl"] == pytest.approx(expected_net / len(exit_prices))
    assert reasons["win_rate"] == 0
    insert.assert_called_once()
    pd.testing.assert_frame_equal(insert.call_args.args[1], exported["backtest_results.csv"])

    # Run the real metrics consumer definition without its module initialization.
    source = Path(__file__).parents[1] / "scripts" / "metrics.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    node = next(node for node in parsed.body if getattr(node, "name", None) == "calculate_metrics")
    isolated = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"pd": pd, "np": np, "logger": logging.getLogger("cost_consumer_test")}
    exec(compile(isolated, str(source), "exec"), namespace)
    consumer = namespace["calculate_metrics"](trades.copy(deep=True))
    assert consumer["net_pnl"] == pytest.approx(expected_net)
    assert consumer["win_rate"] == 0


@pytest.mark.parametrize("exit_prices,expected_nets,expected_factor", [
    ([], [], 0),
    ([102], [0], 0),
    ([104], [20], float("inf")),
    ([102, 104], [0, 20], float("inf")),
    ([102, 100], [0, -20], 0),
    ([102, 104, 100], [0, 20, -20], 1),
    ([100, 100], [-20, -20], 0),
])
def test_simulator_profit_factor_zero_loss_contract(
    simulator, exit_prices, expected_nets, expected_factor,
):
    bt = backtester(simulator, {"AAA": bars({})}, trade_cost=10)
    row = bars({}).iloc[0]
    starting = pd.Timestamp("2024-01-01")
    bt.equity_curve.append((starting, bt.initial_cash))
    for index, exit_price in enumerate(exit_prices, 1):
        when = starting + pd.Timedelta(days=index)
        bt._open_position("AAA", row, when, row)
        bt._close_position("AAA", exit_price, when, "TEST")
        bt.equity_curve.append((when, bt.cash))
        assert_reconciles(bt)
    assert [t.net_pnl for t in bt.trades] == expected_nets
    for trade in bt.trades:
        if trade.net_pnl == 0:
            # Gross-winning exit exactly offset by the two actual fees.
            assert trade.gross_pnl == 20
            assert trade.entry_fee == trade.exit_fee == 10
    assert bt.metrics()["Profit Factor"] == expected_factor
    assert_reconciles(bt)
