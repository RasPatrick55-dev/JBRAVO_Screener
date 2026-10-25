"""Research adapter over the shared JBravo portfolio kernel; no network or DB access."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, time as daytime
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.backtest_core import PortfolioBacktester, _validate_execution_costs
from scripts.research_strategies import BASELINE, BENCHMARK, CHALLENGER

NY = ZoneInfo("America/New_York")


def availability_clock(day, opening=False):
    value = datetime.combine(pd.Timestamp(day).date(), daytime(9, 30) if opening else daytime(20, 15), NY)
    return value.astimezone(ZoneInfo("UTC")).isoformat()


class ResearchPortfolio(PortfolioBacktester):
    def __init__(self, data, *, method, dividends=(), deadline=None, stop_file=None, **options):
        _validate_execution_costs(options.get("trade_cost", 0), options.get("slippage", 0))
        if method not in (BASELINE, CHALLENGER, BENCHMARK):
            raise ValueError("Unknown research method")
        super().__init__(data, **options)
        self.method = method
        self.dividends = dividends
        self.deadline = deadline
        self.stop_file = Path(stop_file) if stop_file else None
        self.entitlements = {}
        self.dividend_income = 0.0
        self.dividend_cash = 0.0
        self.last_reclaims = {}
        self.signals = []
        self.fills = []
        self.action_events = []
        self.reconciliations = []
        self.terminal_entries = []
        self._run_claimed = False

    def run(self):
        if self._run_claimed:
            raise ValueError("A research simulation instance cannot be replayed")
        self._run_claimed = True
        return super().run()

    def _before_session(self, day):
        if (self.deadline is not None and time.monotonic() >= self.deadline) or (self.stop_file and self.stop_file.exists()):
            raise RuntimeError("Research stopped by deadline or stop file")
        key = day.strftime("%Y-%m-%d")
        for event in self.dividends:
            if event["ex_date"] != key or event["symbol"] not in self.data:
                continue
            if event["id"] in self.entitlements:
                raise ValueError("Duplicate dividend entitlement")
            quantity = self.positions[event["symbol"]].qty if event["symbol"] in self.positions else 0
            amount = quantity * float(event["rate"])
            self.entitlements[event["id"]] = dict(amount=amount, paid=False, payable_date=event["payable_date"])
            self.dividend_income += amount
            self.action_events.append(dict(event_id=event["id"], session=key, symbol=event["symbol"],
                                           type="DIVIDEND_ENTITLEMENT", prior_close_qty=quantity, amount=amount))
        for event_id, entitlement in self.entitlements.items():
            if not entitlement["paid"] and entitlement["payable_date"] <= key:
                entitlement["paid"] = True
                self.cash += entitlement["amount"]
                self.dividend_cash += entitlement["amount"]
                self.action_events.append(dict(event_id=event_id, session=key, type="DIVIDEND_CASH_PAYMENT", amount=entitlement["amount"]))
        for symbol, intent in self.pending_entries.items():
            if intent.name >= day:
                raise ValueError("Same-session entry forbidden")
            previous = self.dates[self.dates.index(day) - 1] if self.dates.index(day) else None
            if intent.name != previous:
                raise ValueError("Pending entry exceeded its next-session expiry")

    def _open_position(self, symbol, row, day, signal_row):
        if availability_clock(signal_row.name) >= availability_clock(day, opening=True):
            raise ValueError("Signal unavailable before modeled regular open")
        before = self.cash
        allocation = self.alloc_pct
        if self.method == BENCHMARK:
            # All-in whole-share benchmark reserves entry fee and slippage.
            self.alloc_pct = max(0.0, min(allocation, (self.cash - self.trade_cost) / (self.cash * (1 + self.slippage))))
        try:
            super()._open_position(symbol, row, day, signal_row)
        finally:
            self.alloc_pct = allocation
        if symbol in self.positions:
            position = self.positions[symbol]
            self.fills.append(dict(symbol=symbol, side="BUY", session=day.strftime("%Y-%m-%d"),
                                   signal_session=signal_row.name.strftime("%Y-%m-%d"),
                                   signal_available_at_assumed=availability_clock(signal_row.name),
                                   execution_at_assumed=availability_clock(day, opening=True),
                                   quantity=position.qty, raw_price=float(row["open"]), fee=self.trade_cost,
                                   slippage=before - self.cash - position.qty * float(row["open"]) - self.trade_cost,
                                   cash_delta=self.cash - before))

    def _record_exit(self, position, quantity, price, day, reason):
        before = self.cash
        super()._record_exit(position, quantity, price, day, reason)
        self.fills.append(dict(symbol=position.symbol, side="SELL", session=day.strftime("%Y-%m-%d"),
                               quantity=quantity, raw_price=float(price), fee=self.trade_cost,
                               slippage=self.trades[-1].exit_slippage, cash_delta=self.cash - before,
                               net_pnl=self.trades[-1].net_pnl, reason=reason,
                               execution_model="DAILY_OHLC_PRIOR_STOP_OR_NEXT_OPEN"))

    def _entry_candidates(self, day, rows):
        pool = {item[0]: item for item in super()._entry_candidates(day, rows)}
        chosen = []
        for symbol, row in rows.items():
            consumed = False
            if symbol in self.positions:
                reason = "HELD"
            elif symbol in self.pending_entries:
                reason = "PENDING"
            elif not bool(row.get("entry_ready", False)):
                reason = "WARMUP_OR_FEATURE_READINESS"
            elif symbol not in pool:
                reason = "NO_QUALIFYING_SETUP"
            else:
                details = row.get("entry_details", {})
                last = self.last_reclaims.get(symbol)
                if self.method == CHALLENGER and last is not None and details["pullback_start"] <= last:
                    reason = "REARM_REQUIRES_PULLBACK_AFTER_CONSUMED_CLOSE"
                else:
                    reason = "QUALIFYING_CLOSE_BEFORE_RANKING"
                    consumed = True
                    if self.method == CHALLENGER:
                        self.last_reclaims[symbol] = day.strftime("%Y-%m-%d")
                    chosen.append(pool[symbol])
            value = row.get("score")
            self.signals.append(dict(symbol=symbol, session=day.strftime("%Y-%m-%d"), reason=reason,
                                     consumed=consumed, score=float(value) if pd.notna(value) else None,
                                     available_at_assumed=availability_clock(day),
                                     pullback_start=row.get("entry_details", {}).get("pullback_start")))
        return chosen

    def _after_session(self, day):
        receivable = sum(x["amount"] for x in self.entitlements.values() if not x["paid"])
        last_day, base_equity = self.equity_curve[-1]
        self.equity_curve[-1] = last_day, base_equity + receivable
        basis = sum(p.cost_basis for p in self.positions.values())
        marked = sum(p.qty * self.last_prices[s] for s, p in self.positions.items())
        realized = sum(t.net_pnl for t in self.trades)
        expected_cash = self.initial_cash + realized - basis + self.dividend_cash
        expected_equity = self.initial_cash + realized + marked - basis + self.dividend_income
        if self.cash < -1e-6 or not math.isclose(self.cash, expected_cash, abs_tol=1e-6, rel_tol=1e-10):
            raise ValueError("Research cash ledger does not reconcile")
        if not math.isclose(self.equity_curve[-1][1], expected_equity, abs_tol=1e-6, rel_tol=1e-10):
            raise ValueError("Research NAV ledger does not reconcile")
        self.reconciliations.append(dict(session=day.strftime("%Y-%m-%d"), cash=self.cash,
                                         realized_trade_net=realized, open_cost_basis=basis,
                                         marked_positions=marked, dividend_income=self.dividend_income,
                                         dividend_cash=self.dividend_cash, dividend_receivable=receivable,
                                         equity=self.equity_curve[-1][1]))
        if day == self.dates[-1]:
            self.terminal_entries = [dict(symbol=s, signal_session=r.name.strftime("%Y-%m-%d"),
                                          reason="NO_NEXT_SESSION_INSIDE_EVALUATION_WINDOW")
                                     for s, r in self.pending_entries.items()]
            self.pending_entries.clear()

    def summary(self):
        values = np.array([self.initial_cash] + [v for _, v in self.equity_curve], dtype=float)
        returns = values[1:] / values[:-1] - 1
        if len(returns) == 0 or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Missing/nonpositive/nonfinite research equity")
        peak = np.maximum.accumulate(values)
        trades = self.results()
        wins = sum(t.net_pnl > 0 for t in self.trades)
        losses = -sum(min(t.net_pnl, 0) for t in self.trades)
        profit_factor = sum(max(t.net_pnl, 0) for t in self.trades) / losses if losses > 0 else None
        volatility = returns.std(ddof=1) if len(returns) > 1 else 0
        return dict(total_return_pct=(values[-1] / values[0] - 1) * 100,
                    annualized_return_pct=((values[-1] / values[0]) ** (252 / len(returns)) - 1) * 100,
                    max_drawdown_pct=float(np.min(values / peak - 1) * 100),
                    sharpe_zero_cash_rate=float(returns.mean() / volatility * np.sqrt(252)) if volatility else 0.0,
                    exit_fills=len(trades), winning_exit_fills=wins,
                    exit_fill_win_rate_pct=100 * wins / len(trades) if len(trades) else None,
                    profit_factor=profit_factor, initial_cash=self.initial_cash, terminal_equity=float(values[-1]),
                    realized_trade_net=sum(t.net_pnl for t in self.trades), dividend_income=self.dividend_income,
                    dividend_cash=self.dividend_cash,
                    dividend_receivable=sum(e["amount"] for e in self.entitlements.values() if not e["paid"]),
                    terminal_open_positions=[asdict(p) for p in self.positions.values()],
                    terminal_pending_exits=self.pending_exits,
                    terminal_canceled_entries=self.terminal_entries,
                    signal_observations=len(self.signals), qualifying_consumed_signals=sum(s["consumed"] for s in self.signals),
                    reconciled_sessions=len(self.reconciliations), fill_count=len(self.fills))
