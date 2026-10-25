"""Shared in-memory simulator. Importing performs no environment, DB or provider I/O.

Relocated from scripts.backtest; its legacy CLI and research runner share these
same indicator preparation, scoring, risk, fill and cost-accounting definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("scripts.backtest")

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute a rolling Average True Range (ATR) series."""

    if df.empty:
        return pd.Series(dtype="float64")

    high = pd.to_numeric(df.get("high"), errors="coerce")
    low = pd.to_numeric(df.get("low"), errors="coerce")
    close = pd.to_numeric(df.get("close"), errors="coerce")
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(window=int(period), min_periods=int(period)).mean()


def prepare_series(df: pd.DataFrame) -> pd.DataFrame:
    """Align indicator columns to price index and drop warm-up NaNs."""

    if df.empty:
        return df

    frame = df.copy().sort_index()
    prices = pd.to_numeric(frame.get("close"), errors="coerce")
    base_index = prices.index

    if "ATR14" in frame.columns:
        atr_source = pd.to_numeric(frame["ATR14"], errors="coerce")
    elif "atr" in frame.columns:
        atr_source = pd.to_numeric(frame["atr"], errors="coerce")
    else:
        atr_source = compute_atr(frame)

    atr_series = atr_source.reindex(base_index)
    ema_series = (
        pd.to_numeric(frame["ema20"], errors="coerce").reindex(base_index)
        if "ema20" in frame.columns
        else prices.ewm(span=20, adjust=False).reindex(base_index)
    )

    frame = frame.reindex(base_index)
    frame["close"] = prices
    frame["ATR14"] = atr_series
    if "atr" in frame.columns:
        frame["atr"] = atr_series
    frame["ema20"] = ema_series

    required = ["close", "ATR14", "ema20"]
    if "score" in frame.columns:
        required.append("score")

    aligned = frame.dropna(subset=required)
    aligned = aligned.loc[~aligned.index.duplicated(keep="last")]
    return aligned


def composite_score(df: pd.DataFrame) -> pd.Series:
    """Calculate the screener's composite score for each row."""

    df = df.copy()
    prev = df.shift(1)
    score = pd.Series(0.0, index=df.index)

    score += np.where(df["close"] > df["ma50"], 1, -1)
    score += np.where(df["close"] > df["ma200"], 1, -1)
    score += np.where((df["ma50"] > df["ma200"]) & (prev["ma50"] <= prev["ma200"]), 1.5, 0)

    score += np.where((df["rsi"] > 50) & (prev["rsi"] <= 50), 1, 0)
    score += np.where((df["rsi"] > 30) & (prev["rsi"] <= 30), 1, 0)
    score += np.where(df["rsi"] > 70, -1, 0)

    score += np.where(df["macd"] > df["macd_signal"], 1, -1)
    score += np.where(df["macd_hist"] > prev["macd_hist"], 1, 0)

    score += np.where(df["adx"] > 20, 1, 0)
    score += np.where(df["adx"] > 40, 0.5, 0)

    score += np.where(
        (df["aroon_up"] > df["aroon_down"]) & (prev["aroon_up"] <= prev["aroon_down"]),
        1,
        0,
    )
    score += np.where(df["aroon_up"] > 70, 1, 0)

    score += np.where(df["obv"] > prev["obv"], 1, -1)
    score += np.where(df["volume"] > 2 * df["vol_avg30"], 1, 0)
    score += np.where(df["close"] > df["month_high"], 1, 0)

    body = (df["close"] - df["open"]).abs()
    lower = df["low"] - np.minimum(df["close"], df["open"])
    upper = df["high"] - np.maximum(df["close"], df["open"])
    score += np.where((lower > 2 * body) & (upper <= body), 1, 0)

    prev_body = (prev["close"] - prev["open"]).abs()
    score += np.where(
        (
            (prev["close"] < prev["open"])
            & (df["close"] > df["open"])
            & (df["close"] > prev["open"])
            & (df["open"] < prev["close"])
            & (prev_body > 0)
        ),
        1,
        0,
    )

    return score.round(2)


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    entry_time: pd.Timestamp
    highest_close: float
    max_price: float
    trailing_stop: Optional[float] = None
    atr_stop: Optional[float] = None
    partial_taken: bool = False
    entry_fee_remaining: float = 0.0
    entry_slippage_remaining: float = 0.0

    @property
    def cost_basis(self) -> float:
        """Remaining cash-funded basis, including unallocated entry costs."""
        return (
            self.qty * self.entry_price + self.entry_fee_remaining
            + self.entry_slippage_remaining
        )


@dataclass
class Trade:
    """One executed exit: pnl is a net_pnl compatibility alias; gross is explicit."""

    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    exit_reason: str
    mfe_pct: float = 0.0
    exit_pct: float = 0.0
    exit_efficiency: float = 0.0
    gross_pnl: float = 0.0
    entry_fee: float = 0.0
    entry_slippage: float = 0.0
    exit_fee: float = 0.0
    exit_slippage: float = 0.0
    net_pnl: float = 0.0


def evaluate_exit_signals(position_state, indicators, trail_state, debug_flags=None) -> list[str]:
    """
    Returns a list of exit reasons similar to the monitor logic.

    Parameters
    ----------
    position_state:
        Mapping with keys like ``entry_price``, ``partial_taken``, ``hold_days``.
    indicators:
        Mapping containing at least ``current`` (pd.Series). ``previous`` is
        optional for cross-based exits.
    trail_state:
        Mapping with trailing/ATR stops and feature flags.
    debug_flags:
        Unused placeholder to mirror monitor signature.
    """
    _ = debug_flags

    reasons: list[str] = []
    current = indicators.get("current")
    previous = indicators.get("previous")

    if current is None or current.empty:
        return reasons

    close_price = float(current.get("close", np.nan))
    low_price = float(current.get("low", close_price))
    entry_price = position_state.get("entry_price")

    if trail_state.get("enable_ema_exit", True):
        ema_value = pd.to_numeric(current.get("ema20"), errors="coerce")
        if pd.notna(ema_value) and close_price < float(ema_value):
            reasons.append("EMA20_BREAK")

    atr_stop = trail_state.get("atr_stop")
    if atr_stop is not None and low_price <= atr_stop:
        reasons.append("ATR_STOP")

    trailing_stop = trail_state.get("trailing_stop")
    if (
        trail_state.get("enable_trailing_exit", True)
        and trailing_stop is not None
        and low_price <= trailing_stop
    ):
        reasons.append("TRAIL_STOP")

    max_hold_days = trail_state.get("max_hold_days")
    hold_days = position_state.get("hold_days")
    if max_hold_days is not None and hold_days is not None and hold_days >= max_hold_days:
        reasons.append("MAX_HOLD")

    if trail_state.get("enable_macd_exit", True) and previous is not None:
        macd_val = pd.to_numeric(current.get("macd"), errors="coerce")
        macd_signal_val = pd.to_numeric(current.get("macd_signal"), errors="coerce")
        prev_macd = pd.to_numeric(previous.get("macd"), errors="coerce")
        prev_signal = pd.to_numeric(previous.get("macd_signal"), errors="coerce")
        if all(pd.notna(v) for v in (macd_val, macd_signal_val, prev_macd, prev_signal)):
            if float(macd_val) < float(macd_signal_val) and float(prev_macd) >= float(prev_signal):
                reasons.append("MACD_CROSS")

    if trail_state.get("enable_rsi_divergence", False) and indicators.get("rsi_divergence", False):
        reasons.append("RSI_DIVERGENCE")

    if trail_state.get("enable_candlestick_exit", True) and indicators.get(
        "is_shooting_star", False
    ):
        reasons.append("PATTERN_SHOOTING_STAR")

    if (
        trail_state.get("enable_partial_exit", True)
        and not position_state.get("partial_taken")
        and entry_price
    ):
        gain_pct = (close_price - entry_price) / entry_price * 100
        if gain_pct >= 5:
            reasons.append("PARTIAL_5PCT")

    return reasons


class PortfolioBacktester:
    """Daily-bar simulation with next-observed-open fills.

    Completed-bar signals cannot fill on their signal bar. Protective stops
    use levels known before the bar; closing information adjusts future levels.
    See docs/explanation/backtest_data_flow.md for the OHLC ambiguity contract.
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_cash: float = 100_000.0,
        alloc_pct: float = 0.03,
        top_n: int = 3,
        max_positions: int = 4,
        trail_pct: Optional[float] = 0.03,
        max_hold_days: int = 7,
        trade_cost: float = 0.0,
        slippage: float = 0.0,
        atr_multiple: float = 1.0,
        enable_macd_exit: bool = True,
        enable_partial_exit: bool = True,
        enable_rsi_divergence: bool = False,
        enable_candlestick_exit: bool = True,
        enable_ema_exit: bool = True,
        enable_trailing_exit: bool = True,
    ) -> None:
        self.data = data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.alloc_pct = alloc_pct
        self.top_n = top_n
        self.max_positions = max_positions
        self.trail_pct = trail_pct if trail_pct and trail_pct > 0 else None
        self.max_hold_days = max_hold_days
        self.trade_cost = trade_cost
        self.slippage = slippage
        self.atr_multiple = max(float(atr_multiple), 0.0)
        self.use_trailing = self.trail_pct is not None
        self.enable_macd_exit = enable_macd_exit
        self.enable_partial_exit = enable_partial_exit
        self.enable_rsi_divergence = enable_rsi_divergence
        self.enable_candlestick_exit = enable_candlestick_exit
        self.enable_ema_exit = enable_ema_exit
        self.enable_trailing_exit = enable_trailing_exit

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[tuple[pd.Timestamp, float]] = []
        self.pending_entries: Dict[str, pd.Series] = {}
        self.pending_exits: Dict[str, List[str]] = {}
        self.last_prices: Dict[str, float] = {}

        # Build a unified date index
        indices = [df.index for df in self.data.values()]
        self.dates = sorted(set().union(*indices))
        self.rsi_high_memory: Dict[str, dict] = {}

    def _open_position(
        self, symbol: str, row: pd.Series, date: pd.Timestamp, signal_row: pd.Series
    ) -> None:
        price = float(row["open"])
        alloc = self.cash * self.alloc_pct
        qty = math.floor(alloc / price)
        if qty <= 0:
            return
        cost = qty * price * (1 + self.slippage) + self.trade_cost
        if cost > self.cash:
            return
        self.cash -= cost
        trailing = None
        if self.use_trailing:
            trailing = price * (1 - float(self.trail_pct))

        atr_stop = None
        # Only the preceding signal bar's ATR is available at this open.
        atr_value = pd.to_numeric(signal_row.get("ATR14"), errors="coerce")
        if pd.notna(atr_value) and atr_value > 0 and self.atr_multiple > 0:
            atr_stop = max(0.0, price - self.atr_multiple * float(atr_value))
        self.positions[symbol] = Position(
            symbol=symbol,
            qty=qty,
            entry_price=price,
            entry_time=date,
            highest_close=price,
            max_price=price,
            trailing_stop=trailing,
            atr_stop=atr_stop,
            entry_fee_remaining=self.trade_cost,
            entry_slippage_remaining=cost - qty * price - self.trade_cost,
        )
        logger.info("Opened %s @ %.2f (%d shares)", symbol, price, qty)

    @staticmethod
    def _desired_trail_pct(gain: float) -> float:
        if gain >= 10:
            return 0.01
        if gain >= 5:
            return 0.02
        return 0.03

    @staticmethod
    def _is_shooting_star(row: pd.Series) -> bool:
        open_price = float(row.get("open", np.nan))
        close_price = float(row.get("close", np.nan))
        high_price = float(row.get("high", np.nan))
        low_price = float(row.get("low", np.nan))

        if any(math.isnan(v) for v in (open_price, close_price, high_price, low_price)):
            return False

        real_body = abs(close_price - open_price)
        if real_body == 0:
            return False
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        return (
            close_price < open_price
            and upper_shadow > 2 * real_body
            and lower_shadow <= 0.2 * real_body
        )

    def _record_rsi_high(self, symbol: str, price: float, rsi_value: float) -> None:
        existing = self.rsi_high_memory.get(symbol)
        if existing is None or (
            price > existing.get("price", 0) and rsi_value >= existing.get("rsi", 0)
        ):
            self.rsi_high_memory[symbol] = {"price": price, "rsi": rsi_value}

    def _check_rsi_divergence(self, symbol: str, price: float, rsi_value: float) -> bool:
        state = self.rsi_high_memory.get(symbol, {"price": price, "rsi": rsi_value})
        triggered = (
            rsi_value > 70
            and price > state.get("price", price)
            and rsi_value < state.get("rsi", rsi_value)
        )
        if price > state.get("price", price) and rsi_value >= state.get("rsi", rsi_value):
            self._record_rsi_high(symbol, price, rsi_value)
        elif symbol not in self.rsi_high_memory:
            self._record_rsi_high(symbol, price, rsi_value)
        return triggered

    def _scale_out_position(
        self, symbol: str, price: float, date: pd.Timestamp, reason: str
    ) -> None:
        pos = self.positions.get(symbol)
        if pos is None or pos.qty <= 1:
            return
        sell_qty = max(1, pos.qty // 2)
        self._record_exit(pos, sell_qty, price, date, reason)
        pos.qty -= sell_qty
        pos.partial_taken = True
        pos.max_price = max(pos.max_price, price)
        logger.info("Scaled out %s: sold %d @ %.2f reason=%s", symbol, sell_qty, price, reason)

    def _close_position(self, symbol: str, price: float, date: pd.Timestamp, reason: str) -> None:
        pos = self.positions.pop(symbol)
        self._record_exit(pos, pos.qty, price, date, reason)
        logger.info("Closed %s @ %.2f reason=%s", symbol, price, reason)

    def _record_exit(
        self, pos: Position, qty: int, price: float, date: pd.Timestamp, reason: str
    ) -> None:
        """Allocate entry costs once; the final exit absorbs the remaining residual."""
        if qty <= 0 or qty > pos.qty:
            raise ValueError("Exit quantity must be positive and within the position")
        if qty == pos.qty:
            entry_fee = pos.entry_fee_remaining
            entry_slippage = pos.entry_slippage_remaining
        else:
            fraction = qty / pos.qty
            entry_fee = pos.entry_fee_remaining * fraction
            entry_slippage = pos.entry_slippage_remaining * fraction
        proceeds = qty * price * (1 - self.slippage) - self.trade_cost
        exit_slippage = qty * price - proceeds - self.trade_cost
        gross_pnl = (price - pos.entry_price) * qty
        net_pnl = proceeds - (qty * pos.entry_price + entry_fee + entry_slippage)
        self.cash += proceeds
        pos.entry_fee_remaining -= entry_fee
        pos.entry_slippage_remaining -= entry_slippage
        metrics = self._compute_trade_metrics(pos, price)
        self.trades.append(
            Trade(
                symbol=pos.symbol,
                entry_time=pos.entry_time,
                exit_time=date,
                entry_price=pos.entry_price,
                exit_price=price,
                qty=qty,
                pnl=net_pnl,
                exit_reason=reason,
                mfe_pct=metrics["mfe_pct"],
                exit_pct=metrics["exit_pct"],
                exit_efficiency=metrics["exit_efficiency"],
                gross_pnl=gross_pnl,
                entry_fee=entry_fee,
                entry_slippage=entry_slippage,
                exit_fee=self.trade_cost,
                exit_slippage=exit_slippage,
                net_pnl=net_pnl,
            )
        )

    @staticmethod
    def _compute_trade_metrics(pos: Position, exit_price: float) -> dict:
        entry_price = pos.entry_price
        max_price = pos.max_price if pos.max_price else entry_price
        mfe_pct = ((max_price - entry_price) / entry_price * 100) if entry_price else 0
        exit_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0
        exit_efficiency = (exit_price / max_price) if max_price else 0
        return {
            "mfe_pct": mfe_pct,
            "exit_pct": exit_pct,
            "exit_efficiency": exit_efficiency,
        }

    @staticmethod
    def _bar_prices(row: pd.Series) -> tuple[float, float, float, float]:
        """Reject missing/invalid OHLC instead of inventing a fill price."""
        prices = tuple(
            float(row.get(key, np.nan)) for key in ("open", "high", "low", "close")
        )
        opening, high, low, close = prices
        if (
            not all(math.isfinite(price) and price > 0 for price in prices)
            or low > min(opening, close)
            or high < max(opening, close)
            or low > high
        ):
            raise ValueError("Causal fills require finite, positive, consistent OHLC")
        return opening, high, low, close

    def _protective_stop(self, pos: Position) -> tuple[Optional[float], str]:
        """For a long, the highest active stop is encountered first; ATR wins ties."""
        stops = []
        if pos.atr_stop is not None:
            stops.append((pos.atr_stop, "ATR_STOP"))
        if self.use_trailing and self.enable_trailing_exit and pos.trailing_stop is not None:
            stops.append((pos.trailing_stop, "TRAIL_STOP"))
        return max(stops, key=lambda stop: stop[0]) if stops else (None, "")

    def _update_completed_bar(self, symbol: str, row: pd.Series, date: pd.Timestamp) -> None:
        """Adjust surviving stops and queue close-derived exits for a later open."""
        pos = self.positions[symbol]
        _, high, _, close = self._bar_prices(row)
        pos.max_price = max(pos.max_price, high, close)
        pos.highest_close = max(pos.highest_close, close)
        gain_pct = (pos.max_price - pos.entry_price) / pos.entry_price * 100
        if self.use_trailing and self.enable_trailing_exit:
            candidate = pos.max_price * (1 - self._desired_trail_pct(gain_pct))
            pos.trailing_stop = max(pos.trailing_stop or 0.0, candidate)
        atr = pd.to_numeric(row.get("ATR14"), errors="coerce")
        if pd.notna(atr) and atr > 0 and self.atr_multiple > 0:
            candidate = max(0.0, close - self.atr_multiple * float(atr))
            pos.atr_stop = max(pos.atr_stop or 0.0, candidate)

        df = self.data[symbol]
        prev_idx = df.index.get_loc(date) - 1
        previous = df.iloc[prev_idx] if prev_idx >= 0 else None
        divergence = (
            self.enable_rsi_divergence
            and "rsi" in row
            and self._check_rsi_divergence(symbol, close, float(row["rsi"]))
        )
        reasons = evaluate_exit_signals(
            position_state={
                "entry_price": pos.entry_price,
                "partial_taken": pos.partial_taken,
                "hold_days": (date - pos.entry_time).days,
            },
            indicators={
                "current": row,
                "previous": previous,
                "rsi_divergence": divergence,
                "is_shooting_star": self._is_shooting_star(row),
            },
            trail_state={
                # Protective levels have already been tested for this bar.
                "enable_ema_exit": self.enable_ema_exit,
                "enable_macd_exit": self.enable_macd_exit,
                "enable_partial_exit": self.enable_partial_exit,
                "enable_rsi_divergence": self.enable_rsi_divergence,
                "enable_candlestick_exit": self.enable_candlestick_exit,
                "max_hold_days": self.max_hold_days,
            },
        )
        full_exit = [reason for reason in reasons if reason != "PARTIAL_5PCT"]
        if reasons:
            self.pending_exits[symbol] = full_exit or ["PARTIAL_5PCT"]

    def _before_session(self, date: pd.Timestamp) -> None:
        """Optional research accounting hook; legacy execution has no action adapter."""

    def _after_session(self, date: pd.Timestamp) -> None:
        """Optional research accounting/reconciliation hook."""

    def _entry_candidates(self, date, rows):
        """Finite scores in original data order; subclasses can impose research predicates."""
        scores = []
        for symbol, df in self.data.items():
            if symbol in self.positions or symbol in self.pending_entries or symbol not in rows:
                continue
            score = df.loc[date, "score"]
            if not pd.isna(score):
                scores.append((symbol, score, df.loc[date]))
        return scores

    def run(self) -> None:
        for date in self.dates:
            self._before_session(date)
            rows = {
                symbol: df.loc[date] for symbol, df in self.data.items() if date in df.index
            }
            prices = {symbol: self._bar_prices(row) for symbol, row in rows.items()}

            # Opening gap stops precede queued full exits, which precede partials.
            for symbol in list(self.positions):
                if symbol not in rows:
                    continue
                pos = self.positions[symbol]
                opening = prices[symbol][0]
                pos.max_price = max(pos.max_price, opening)
                stop, stop_reason = self._protective_stop(pos)
                reasons = self.pending_exits.pop(symbol, [])
                if stop is not None and opening <= stop:
                    self._close_position(symbol, opening, date, stop_reason)
                elif reasons and reasons != ["PARTIAL_5PCT"]:
                    self._close_position(symbol, opening, date, ";".join(reasons))
                elif reasons:
                    self._scale_out_position(symbol, opening, date, "PARTIAL_5PCT")

            # Each signal is attempted once, at its symbol's next observed open.
            # Missing symbol bars leave the signal intact; no bar is fabricated.
            for symbol in list(self.pending_entries):
                if symbol not in rows:
                    continue
                signal_row = self.pending_entries.pop(symbol)
                if symbol not in self.positions and len(self.positions) < self.max_positions:
                    self._open_position(symbol, rows[symbol], date, signal_row)

            # Intrabar stops use only pre-existing levels (including entry stops).
            # Do not assume the current high happened before a stopped-out low.
            for symbol in list(self.positions):
                if symbol not in rows:
                    continue
                pos = self.positions[symbol]
                opening, _, low, _ = prices[symbol]
                stop, stop_reason = self._protective_stop(pos)
                if stop is not None and low <= stop:
                    self._close_position(symbol, min(opening, stop), date, stop_reason)
                    self.pending_exits.pop(symbol, None)
                else:
                    self._update_completed_bar(symbol, rows[symbol], date)

            # Ranking formula/order is unchanged; only execution is deferred.
            scores = self._entry_candidates(date, rows)

            scores.sort(key=lambda x: x[1], reverse=True)
            new_trades = 0
            for symbol, _, row in scores:
                if len(self.positions) + len(self.pending_entries) >= self.max_positions:
                    break
                if new_trades >= self.top_n:
                    break
                self.pending_entries[symbol] = row.copy()
                new_trades += 1

            # Calculate equity
            self.last_prices.update({symbol: price[3] for symbol, price in prices.items()})
            equity = self.cash
            for sym, pos in self.positions.items():
                equity += pos.qty * self.last_prices.get(sym, pos.entry_price)
            self.equity_curve.append((date, equity))
            self._after_session(date)

    def results(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades, columns=list(Trade.__dataclass_fields__))

    def equity(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve, columns=["date", "equity"]).set_index("date")

    def metrics(self) -> Dict[str, float]:
        if not self.equity_curve:
            return {}

        equity_df = self.equity()
        daily_returns = equity_df["equity"].pct_change().dropna()
        total_return = equity_df["equity"].iloc[-1] / self.initial_cash - 1
        cagr = (
            (1 + total_return) ** (252 / len(daily_returns)) - 1
            if len(daily_returns) else 0.0
        )

        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in self.trades if t.net_pnl < 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        gains, loss_total = sum(wins), abs(sum(losses))
        profit_factor = (
            gains / loss_total if loss_total > 0
            else float("inf") if gains > 0 else 0.0
        )

        cummax = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] - cummax) / cummax
        max_dd = drawdown.min()

        sharpe = (
            np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            if pd.notna(daily_returns.std()) and daily_returns.std() != 0
            else 0
        )
        downside = daily_returns[daily_returns < 0].std()
        sortino = (
            np.sqrt(252) * daily_returns.mean() / downside
            if pd.notna(downside) and downside != 0 else 0
        )
        open_basis = sum(pos.cost_basis for pos in self.positions.values())
        marked_value = sum(
            pos.qty * self.last_prices.get(symbol, pos.entry_price)
            for symbol, pos in self.positions.items()
        )

        return {
            "Total Return": round(total_return * 100, 2),
            "CAGR": round(cagr * 100, 2),
            "Win Rate": round(win_rate, 2),
            "Profit Factor": round(profit_factor, 2),
            "Max Drawdown": round(max_dd * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Sortino": round(sortino, 2),
            "Realized Net P&L": round(sum(t.net_pnl for t in self.trades), 2),
            "Remaining Cost Basis": round(open_basis, 2),
            "Unrealized P&L": round(marked_value - open_basis, 2),
        }


def _validate_execution_costs(trade_cost: float, slippage: float) -> None:
    """Fees are nonnegative currency per fill; slippage is a fraction in [0, 1)."""
    try:
        valid_fee = not isinstance(trade_cost, bool) and math.isfinite(trade_cost) and trade_cost >= 0
        valid_slippage = (
            not isinstance(slippage, bool) and math.isfinite(slippage) and 0 <= slippage < 1
        )
    except (TypeError, ValueError):
        raise ValueError("Costs must be finite numeric values") from None
    if not valid_fee:
        raise ValueError("trade_cost must be finite and nonnegative")
    if not valid_slippage:
        raise ValueError("slippage must be finite and in [0, 1)")
