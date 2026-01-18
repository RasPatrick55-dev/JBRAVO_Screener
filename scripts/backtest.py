"""Comprehensive JBravo strategy backtester."""

from __future__ import annotations

import argparse
import os
import sys

import logging
import logging.handlers
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as PGConnection

from scripts import db
from utils import logger_utils
from utils.env import load_env, get_alpaca_creds

# Import indicator helpers from screener to keep the scoring consistent
from .indicators import adx, aroon, macd, obv, rsi, compute_indicators
from .utils import write_csv_atomic, cache_bars

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_env()
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

logger = logger_utils.init_logging(__name__, "backtest.log")
start_time = datetime.utcnow()
logger.info("Script started")

CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
CONFIG = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        CONFIG = json.load(f)

API_KEY, API_SECRET, _, _ = get_alpaca_creds()

try:
    from alpaca.data.historical import StockHistoricalDataClient
    if API_KEY and API_SECRET:
        data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    else:
        data_client = None
        logger.error("Missing Alpaca credentials; data client unavailable.")
except Exception as exc:  # pragma: no cover - optional dependency
    data_client = None
    logger.error("Alpaca client unavailable: %s", exc)


def compute_recent_performance(
    bars: Optional[pd.DataFrame],
    *,
    lookback: int = 90,
) -> dict[str, float]:
    """Compute simple expectancy and win-rate statistics for recent bars.

    Parameters
    ----------
    bars:
        DataFrame containing at least ``timestamp`` and ``close`` columns.
    lookback:
        Maximum number of most-recent return observations to include when
        computing the summary statistics.

    Returns
    -------
    dict[str, float]
        Mapping with keys ``expectancy`` (mean daily return), ``win_rate``
        (fraction of positive returns), ``samples`` (number of returns used)
        and ``lookback`` (applied window).
    """

    if bars is None or bars.empty:
        return {"expectancy": 0.0, "win_rate": 0.0, "samples": 0, "lookback": int(lookback)}

    frame = bars.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.sort_values("timestamp")

    closes = pd.to_numeric(frame.get("close"), errors="coerce")
    returns = closes.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if lookback and lookback > 0:
        returns = returns.tail(int(lookback))

    if returns.empty:
        return {"expectancy": 0.0, "win_rate": 0.0, "samples": 0, "lookback": int(lookback)}

    expectancy = float(returns.mean())
    win_rate = float((returns > 0).mean())
    samples = int(returns.shape[0])

    return {
        "expectancy": expectancy,
        "win_rate": win_rate,
        "samples": samples,
        "lookback": int(lookback),
    }


def load_bars_from_db(symbol: str) -> Optional[pd.DataFrame]:
    if not db.db_enabled():
        logger.warning("BAR_CACHE_MISS symbol=%s reason=db_disabled", symbol)
        return None

    conn = db.get_db_conn()
    if conn is None:
        logger.warning("BAR_CACHE_MISS symbol=%s reason=db_connect_failed", symbol)
        return None

    symbol_key = symbol.strip().upper()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT date, open, high, low, close, volume
                FROM daily_bars
                WHERE symbol = %(symbol)s
                ORDER BY date ASC
                """,
                {"symbol": symbol_key},
            )
            rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(
            rows, columns=["date", "open", "high", "low", "close", "volume"]
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
        return frame
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("BAR_CACHE_MISS symbol=%s reason=db_query_failed err=%s", symbol_key, exc)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


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
    score += np.where(
        (df["ma50"] > df["ma200"]) & (prev["ma50"] <= prev["ma200"]), 1.5, 0
    )

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


@dataclass
class Trade:
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
    if trail_state.get("enable_trailing_exit", True) and trailing_stop is not None and low_price <= trailing_stop:
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

    if trail_state.get("enable_candlestick_exit", True) and indicators.get("is_shooting_star", False):
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
    """Simple portfolio level backtester for the JBravo strategy."""

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

        # Build a unified date index
        indices = [df.index for df in self.data.values()]
        self.dates = sorted(set().union(*indices))
        self.rsi_high_memory: Dict[str, dict] = {}

    def _open_position(
        self, symbol: str, row: pd.Series, date: pd.Timestamp
    ) -> None:
        price = float(row["close"])
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
        atr_value = pd.to_numeric(row.get("ATR14"), errors="coerce")
        if pd.notna(atr_value) and atr_value > 0 and self.atr_multiple > 0:
            atr_stop = max(0.0, price - self.atr_multiple * float(atr_value))
        high_price = float(row.get("high", price))
        self.positions[symbol] = Position(
            symbol=symbol,
            qty=qty,
            entry_price=price,
            entry_time=date,
            highest_close=price,
            max_price=max(price, high_price),
            trailing_stop=trailing,
            atr_stop=atr_stop,
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
        return close_price < open_price and upper_shadow > 2 * real_body and lower_shadow <= 0.2 * real_body

    def _record_rsi_high(self, symbol: str, price: float, rsi_value: float) -> None:
        existing = self.rsi_high_memory.get(symbol)
        if existing is None or (price > existing.get("price", 0) and rsi_value >= existing.get("rsi", 0)):
            self.rsi_high_memory[symbol] = {"price": price, "rsi": rsi_value}

    def _check_rsi_divergence(self, symbol: str, price: float, rsi_value: float) -> bool:
        state = self.rsi_high_memory.get(symbol, {"price": price, "rsi": rsi_value})
        triggered = rsi_value > 70 and price > state.get("price", price) and rsi_value < state.get("rsi", rsi_value)
        if price > state.get("price", price) and rsi_value >= state.get("rsi", rsi_value):
            self._record_rsi_high(symbol, price, rsi_value)
        elif symbol not in self.rsi_high_memory:
            self._record_rsi_high(symbol, price, rsi_value)
        return triggered

    def _scale_out_position(self, symbol: str, price: float, date: pd.Timestamp, reason: str) -> None:
        pos = self.positions.get(symbol)
        if pos is None or pos.qty <= 1:
            return
        sell_qty = max(1, pos.qty // 2)
        proceeds = sell_qty * price * (1 - self.slippage) - self.trade_cost
        self.cash += proceeds
        pnl = (price - pos.entry_price) * sell_qty
        metrics = self._compute_trade_metrics(pos, price)
        self.trades.append(
            Trade(
                symbol=symbol,
                entry_time=pos.entry_time,
                exit_time=date,
                entry_price=pos.entry_price,
                exit_price=price,
                qty=sell_qty,
                pnl=pnl,
                exit_reason=reason,
                mfe_pct=metrics["mfe_pct"],
                exit_pct=metrics["exit_pct"],
                exit_efficiency=metrics["exit_efficiency"],
            )
        )
        pos.qty -= sell_qty
        pos.partial_taken = True
        gain_pct = (price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price else 0
        desired_pct = self._desired_trail_pct(gain_pct)
        pos.highest_close = max(pos.highest_close, price)
        pos.max_price = max(pos.max_price, price)
        pos.trailing_stop = pos.highest_close * (1 - desired_pct)
        logger.info("Scaled out %s: sold %d @ %.2f reason=%s", symbol, sell_qty, price, reason)

    def _close_position(
        self, symbol: str, price: float, date: pd.Timestamp, reason: str
    ) -> None:
        pos = self.positions.pop(symbol)
        proceeds = pos.qty * price * (1 - self.slippage) - self.trade_cost
        self.cash += proceeds
        pnl = (price - pos.entry_price) * pos.qty
        metrics = self._compute_trade_metrics(pos, price)
        self.trades.append(
            Trade(
                symbol=symbol,
                entry_time=pos.entry_time,
                exit_time=date,
                entry_price=pos.entry_price,
                exit_price=price,
                qty=pos.qty,
                pnl=pnl,
                exit_reason=reason,
                mfe_pct=metrics["mfe_pct"],
                exit_pct=metrics["exit_pct"],
                exit_efficiency=metrics["exit_efficiency"],
            )
        )
        logger.info("Closed %s @ %.2f reason=%s", symbol, price, reason)

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

    def run(self) -> None:
        for date in self.dates:
            # Update trailing stops and evaluate exits
            for symbol in list(self.positions):
                df = self.data[symbol]
                if date not in df.index:
                    continue
                row = df.loc[date]
                pos = self.positions[symbol]
                close = float(row["close"])
                low = float(row.get("low", close))
                high = float(row.get("high", close))

                pos.max_price = max(pos.max_price, high, close)
                pos.highest_close = max(pos.highest_close, close)

                gain_pct = (pos.max_price - pos.entry_price) / pos.entry_price * 100 if pos.entry_price else 0
                desired_pct = self._desired_trail_pct(gain_pct)

                if self.use_trailing and self.enable_trailing_exit:
                    trailing_candidate = pos.max_price * (1 - desired_pct)
                    if pos.trailing_stop is None:
                        pos.trailing_stop = trailing_candidate
                    else:
                        pos.trailing_stop = max(pos.trailing_stop, trailing_candidate)

                atr_value = pd.to_numeric(row.get("ATR14"), errors="coerce")
                if pd.notna(atr_value) and atr_value > 0 and self.atr_multiple > 0:
                    atr_candidate = close - self.atr_multiple * float(atr_value)
                    atr_candidate = max(0.0, atr_candidate)
                    if pos.atr_stop is None:
                        pos.atr_stop = atr_candidate
                    else:
                        pos.atr_stop = max(pos.atr_stop, atr_candidate)

                hold_days = (date - pos.entry_time).days

                prev_idx = df.index.get_loc(date) - 1
                prev_row = df.iloc[prev_idx] if prev_idx >= 0 else None
                rsi_divergence = (
                    self.enable_rsi_divergence
                    and "rsi" in row
                    and self._check_rsi_divergence(symbol, close, float(row["rsi"]))
                )

                reasons = evaluate_exit_signals(
                    position_state={
                        "entry_price": pos.entry_price,
                        "partial_taken": pos.partial_taken,
                        "hold_days": hold_days,
                    },
                    indicators={
                        "current": row,
                        "previous": prev_row,
                        "rsi_divergence": rsi_divergence,
                        "is_shooting_star": self._is_shooting_star(row),
                    },
                    trail_state={
                        "atr_stop": pos.atr_stop,
                        "trailing_stop": pos.trailing_stop,
                        "enable_trailing_exit": self.enable_trailing_exit,
                        "enable_macd_exit": self.enable_macd_exit,
                        "enable_partial_exit": self.enable_partial_exit,
                        "enable_rsi_divergence": self.enable_rsi_divergence,
                        "enable_candlestick_exit": self.enable_candlestick_exit,
                        "enable_ema_exit": self.enable_ema_exit,
                        "max_hold_days": self.max_hold_days,
                    },
                )

                if "PARTIAL_5PCT" in reasons and self.enable_partial_exit and not pos.partial_taken:
                    self._scale_out_position(symbol, close, date, "PARTIAL_5PCT")
                    reasons = [r for r in reasons if r != "PARTIAL_5PCT"]
                    if not reasons:
                        continue

                exit_price: Optional[float] = None
                if "ATR_STOP" in reasons and pos.atr_stop is not None and low <= pos.atr_stop:
                    exit_price = pos.atr_stop
                elif "TRAIL_STOP" in reasons and pos.trailing_stop is not None and low <= pos.trailing_stop:
                    exit_price = pos.trailing_stop
                else:
                    exit_price = close

                if reasons:
                    reason_text = ";".join(dict.fromkeys(reasons))
                    self._close_position(symbol, float(exit_price), date, reason_text)

            # Determine today's top candidates
            scores = []
            for symbol, df in self.data.items():
                if symbol in self.positions and len(self.positions) >= self.max_positions:
                    continue
                if date not in df.index:
                    continue
                score = df.loc[date, "score"]
                if not pd.isna(score):
                    scores.append((symbol, score, df.loc[date]))

            scores.sort(key=lambda x: x[1], reverse=True)
            new_trades = 0
            for symbol, _, row in scores:
                if symbol in self.positions:
                    continue
                if len(self.positions) >= self.max_positions:
                    break
                if new_trades >= self.top_n:
                    break
                self._open_position(symbol, row, date)
                new_trades += 1

            # Calculate equity
            equity = self.cash
            for sym, pos in self.positions.items():
                df = self.data[sym]
                if date in df.index:
                    equity += pos.qty * df.loc[date, "close"]
            self.equity_curve.append((date, equity))

    def results(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def equity(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve, columns=["date", "equity"]).set_index(
            "date"
        )

    def metrics(self) -> Dict[str, float]:
        if not self.equity_curve:
            return {}

        equity_df = self.equity()
        daily_returns = equity_df["equity"].pct_change().dropna()
        total_return = equity_df["equity"].iloc[-1] / self.initial_cash - 1
        cagr = (1 + total_return) ** (252 / len(daily_returns)) - 1

        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

        cummax = equity_df["equity"].cummax()
        drawdown = (equity_df["equity"] - cummax) / cummax
        max_dd = drawdown.min()

        sharpe = (
            np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            if daily_returns.std() != 0
            else 0
        )
        downside = daily_returns[daily_returns < 0].std()
        sortino = (
            np.sqrt(252) * daily_returns.mean() / downside
            if downside != 0
            else 0
        )

        return {
            "Total Return": round(total_return * 100, 2),
            "CAGR": round(cagr * 100, 2),
            "Win Rate": round(win_rate, 2),
            "Profit Factor": round(profit_factor, 2),
            "Max Drawdown": round(max_dd * 100, 2),
            "Sharpe": round(sharpe, 2),
            "Sortino": round(sortino, 2),
        }


def run_backtest(
    symbols: List[str],
    *,
    max_symbols: Optional[int] = None,
    max_days: Optional[int] = None,
    quick: bool = False,
    run_date: Optional[date] = None,
    export_csv: bool = True,
    enable_db: bool = True,
) -> dict:
    QUICK_MAX_SYMBOLS = 20
    QUICK_MAX_DAYS = 120

    quick_mode = quick or max_symbols is not None or max_days is not None
    if quick_mode:
        max_symbols = max_symbols if max_symbols is not None else QUICK_MAX_SYMBOLS
        max_days = max_days if max_days is not None else QUICK_MAX_DAYS
        logger.info(
            "Quick backtest mode enabled: max_symbols=%s max_days=%s",
            max_symbols,
            max_days,
        )

    valid_symbols: List[str] = []
    for symbol in symbols:
        if re.match(r'^[A-Z]{1,5}$', symbol):
            valid_symbols.append(symbol)
        else:
            logger.warning("Invalid symbol skipped: %s", symbol)

    if quick_mode and max_symbols:
        valid_symbols = sorted(valid_symbols)[:max_symbols]
        logger.info("Universe limited to first %d symbols", len(valid_symbols))

    data = {}
    lookback_days = max_days if max_days else 800
    for sym in valid_symbols:
        logger.info("Fetching data for %s", sym)
        df = load_bars_from_db(sym)
        if df is None or len(df) < 750:
            logger.warning("BAR_CACHE_MISS symbol=%s reason=not_enough_bars", sym)
            continue
        logger.info("BARS_LOADED_FROM_DB symbol=%s rows=%d", sym, len(df))
        if len(df) < 250:
            logger.warning("Skipping %s: insufficient data", sym)
            continue
        df = compute_indicators(df)
        df["score"] = composite_score(df)
        if max_days:
            df = df.tail(max_days)
        df = prepare_series(df)
        if df.empty:
            logger.warning("Skipping %s: insufficient aligned data", sym)
            continue
        data[sym] = df

    if not data:
        logger.error("No valid data to run backtest.")
        return {"tested": 0, "skipped": len(valid_symbols)}

    trail_pct = CONFIG.get('trail_pct', 0.03)
    if not CONFIG.get('use_trailing_stop', True):
        trail_pct = None

    bt = PortfolioBacktester(
        data,
        trail_pct=trail_pct,
        max_hold_days=CONFIG.get('max_hold_days', 7),
        atr_multiple=CONFIG.get('atr_multiple', 1.0),
        enable_macd_exit=CONFIG.get('enable_macd_exit', True),
        enable_partial_exit=CONFIG.get('enable_partial_exit', True),
        enable_rsi_divergence=CONFIG.get('enable_rsi_divergence', False),
        enable_candlestick_exit=CONFIG.get('enable_candlestick_exit', True),
        enable_ema_exit=CONFIG.get('enable_ema_exit', True),
        enable_trailing_exit=CONFIG.get('enable_trailing_exit', True),
    )

    trades_path = os.path.join(BASE_DIR, "data", "trades_log.csv")
    equity_path = os.path.join(BASE_DIR, "data", "equity_curve.csv")
    metrics_path = os.path.join(BASE_DIR, "data", "backtest_results.csv")

    required_columns = ["symbol", "entry_time", "exit_time", "pnl", "net_pnl"]

    try:
        bt.run()
        trades_df = bt.results()

        if not trades_df.empty:
            trades_df["net_pnl"] = trades_df["pnl"]
        else:
            trades_df["net_pnl"] = []

        # Ensure timestamp columns for dashboard compatibility
        if "entry_time" not in trades_df.columns and "entry_date" in trades_df.columns:
            trades_df["entry_time"] = trades_df["entry_date"]
        if "exit_time" not in trades_df.columns and "exit_date" in trades_df.columns:
            trades_df["exit_time"] = trades_df["exit_date"]
        for col in ["entry_date", "exit_date"]:
            if col in trades_df.columns:
                trades_df.drop(columns=col, inplace=True)

        missing_columns = [col for col in required_columns if col not in trades_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        equity_df = bt.equity()

        if "exit_reason" in trades_df.columns and not trades_df.empty:
            grouped = trades_df.copy()
            grouped["win"] = grouped["pnl"] > 0
            reason_stats = (
                grouped
                .groupby("exit_reason")
                .agg(
                    trades=("pnl", "size"),
                    win_rate=("win", "mean"),
                    avg_pnl=("pnl", "mean"),
                    total_pnl=("pnl", "sum"),
                )
                .reset_index()
            )
            reason_stats["win_rate"] = reason_stats["win_rate"] * 100
            if export_csv:
                write_csv_atomic(
                    os.path.join(BASE_DIR, "data", "exit_reason_metrics.csv"),
                    reason_stats,
                )

        # Aggregate per-symbol metrics from the trades log
        if not trades_df.empty:
            symbol_groups = trades_df.groupby("symbol")

            summary_df = (
                symbol_groups
                .agg(
                    trades=("pnl", "size"),
                    wins=("pnl", lambda x: (x > 0).sum()),
                    losses=("pnl", lambda x: (x <= 0).sum()),
                    net_pnl=("pnl", "sum"),
                    expectancy=("pnl", "mean"),
                )
                .reset_index()
            )

            summary_df["win_rate"] = summary_df["wins"] / summary_df["trades"] * 100

            def _profit_factor(series: pd.Series) -> float:
                gains = series[series > 0].sum()
                losses = series[series < 0].sum()
                if losses == 0:
                    return float("inf") if gains > 0 else 0.0
                return float(gains / abs(losses))

            profit_factors = symbol_groups["pnl"].apply(_profit_factor)

            def _max_drawdown(group: pd.DataFrame) -> float:
                ordered = group.sort_values("exit_time") if "exit_time" in group else group
                cumulative = ordered["pnl"].cumsum()
                if cumulative.empty:
                    return 0.0
                drawdown = cumulative - cumulative.cummax()
                return float(drawdown.min()) if not drawdown.empty else 0.0

            max_drawdowns = symbol_groups.apply(_max_drawdown, include_groups=False)

            def _trade_returns(group: pd.DataFrame) -> pd.Series:
                if {"entry_price", "qty"}.issubset(group.columns):
                    entry_val = (
                        pd.to_numeric(group["entry_price"], errors="coerce")
                        * pd.to_numeric(group["qty"], errors="coerce").abs()
                    )
                    returns = group["pnl"] / entry_val.replace(0, np.nan)
                else:
                    returns = group["pnl"]
                return returns.replace([np.inf, -np.inf], np.nan).dropna()

            def _sharpe(group: pd.DataFrame) -> float:
                returns = _trade_returns(group)
                if returns.empty:
                    return 0.0
                std = returns.std(ddof=0)
                if std == 0 or np.isnan(std):
                    return 0.0
                return float(np.sqrt(len(returns)) * returns.mean() / std)

            def _sortino(group: pd.DataFrame) -> float:
                returns = _trade_returns(group)
                if returns.empty:
                    return 0.0
                downside = returns[returns < 0]
                downside_std = downside.std(ddof=0)
                if downside_std == 0 or np.isnan(downside_std):
                    return 0.0
                return float(np.sqrt(len(returns)) * returns.mean() / downside_std)

            sharpes = symbol_groups.apply(_sharpe, include_groups=False)
            sortinos = symbol_groups.apply(_sortino, include_groups=False)

            summary_df["profit_factor"] = summary_df["symbol"].map(profit_factors)
            summary_df["max_drawdown"] = summary_df["symbol"].map(max_drawdowns)
            summary_df["sharpe"] = summary_df["symbol"].map(sharpes)
            summary_df["sortino"] = summary_df["symbol"].map(sortinos)
        else:
            summary_df = pd.DataFrame(
                columns=[
                    "symbol",
                    "trades",
                    "wins",
                    "losses",
                    "net_pnl",
                    "win_rate",
                    "expectancy",
                    "profit_factor",
                    "max_drawdown",
                    "sharpe",
                    "sortino",
                ]
            )

        if valid_symbols:
            base_symbols = pd.DataFrame({"symbol": valid_symbols})
            summary_df = base_symbols.merge(summary_df, on="symbol", how="left", sort=False)
            summary_df["trades"] = (
                pd.to_numeric(summary_df["trades"], errors="coerce").fillna(0).astype(int)
            )
            summary_df["wins"] = (
                pd.to_numeric(summary_df["wins"], errors="coerce").fillna(0).astype(int)
            )
            summary_df["losses"] = (
                pd.to_numeric(summary_df["losses"], errors="coerce").fillna(0).astype(int)
            )
            for col in [
                "net_pnl",
                "expectancy",
                "profit_factor",
                "max_drawdown",
                "sharpe",
                "sortino",
            ]:
                summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce").fillna(0.0)
            summary_df["win_rate"] = np.where(
                summary_df["trades"] > 0,
                summary_df["wins"] / summary_df["trades"] * 100,
                0.0,
            )

        summary_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        summary_df["symbols_tested"] = len(symbols)
        backtest_run_date = run_date or datetime.now(timezone.utc).date()
        summary_df["run_date"] = backtest_run_date

        for _, row in summary_df.iterrows():
            logger.info(
                "Backtest %s win_rate=%.2f%% net_pnl=%.2f",
                row.symbol,
                row.win_rate,
                row.net_pnl,
            )

        if enable_db:
            try:
                inserted = db.insert_backtest_results(backtest_run_date, summary_df)
                if inserted:
                    logger.info(
                        "BACKTEST_DB_OK run_date=%s rows=%s",
                        backtest_run_date,
                        len(summary_df),
                    )
                    logger.info("BACKTEST_RESULTS_INSERTED rows=%d", len(summary_df))
                else:
                    logger.warning(
                        "BACKTEST_DB_FAIL run_date=%s err=%s",
                        backtest_run_date,
                        "noop_or_error",
                    )
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("BACKTEST_DB_FAIL run_date=%s err=%s", backtest_run_date, exc)

        if export_csv:
            write_csv_atomic(trades_path, trades_df)
            write_csv_atomic(equity_path, equity_df.reset_index())
            write_csv_atomic(metrics_path, summary_df)
        if export_csv:
            logger.info(f"Trades log successfully updated with net_pnl at {trades_path}.")
        else:
            logger.info("Trades log generated in-memory; CSV export disabled.")

    except Exception as e:
        logger.error(f"Error during backtest trades log generation: {e}", exc_info=True)
        raise


    processed = len(valid_symbols)
    tested = len(data)
    skipped = processed - tested
    logger.info(
        "Processed %d symbols, %d tested, %d skipped",
        processed,
        tested,
        skipped,
    )
    logger.info("Backtest complete. Results saved to data directory")

    return {"tested": tested, "skipped": skipped}


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the JBRAVO backtest")
    parser.add_argument(
        "--source",
        default=os.path.join(BASE_DIR, "data", "latest_candidates.csv"),
        help="(deprecated) CSV source ignored; DB is source of truth",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Enable quick smoke-test mode (limits symbols and lookback)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Limit number of symbols to backtest (defaults to 20 in quick mode)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Limit number of most recent trading days to use (defaults to 120 in quick mode)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of candidates fetched from the database (default: 15)",
    )
    parser.add_argument(
        "--export-csv",
        choices=["true", "false"],
        default=None,
        help="(deprecated) CSV export disabled; DB is source of truth",
    )
    return parser.parse_args(argv if argv is not None else None)


def _load_symbols(source_csv: Path) -> List[str]:
    if not isinstance(source_csv, Path):
        source_csv = Path(str(source_csv))

    if not source_csv.exists() or source_csv.stat().st_size == 0:
        return []

    try:
        frame = pd.read_csv(source_csv)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        logger.error("Backtest: failed to read %s: %s", source_csv, exc)
        return []

    if frame.empty or "symbol" not in frame.columns:
        return []

    series = frame["symbol"].astype("string").str.strip()
    return [symbol for symbol in series.tolist() if symbol]


def _db_engine_if_available() -> Optional[PGConnection]:
    if not db.db_enabled():
        return None
    conn = db.get_db_conn()
    if conn is None:
        return None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.info("BACKTEST_DB_SKIP reason=connect_test err=%s", exc)
        try:
            conn.close()
        except Exception:
            pass
        return None
    return conn


def _determine_run_date(conn: Optional[PGConnection]) -> date:
    if conn is None:
        return datetime.utcnow().date()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT MAX(run_date) FROM screener_candidates")
            row = cursor.fetchone()
            latest = row[0] if row else None
        if latest:
            return latest
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.info("BACKTEST_DB_RUN_DATE_FALLBACK err=%s", exc)
    return datetime.utcnow().date()


def _validate_candidates_schema(conn: PGConnection) -> bool:
    required_columns = {"run_date", "symbol", "score", "exchange", "entry_price"}
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'screener_candidates'
                """
            )
            available = {row[0] for row in cursor.fetchall()}
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.info("BACKTEST_DB_CANDIDATES_SCHEMA_FAIL err=%s", exc)
        return False

    missing = sorted(required_columns - available)
    if missing:
        logger.info("BACKTEST_DB_CANDIDATES_SCHEMA_MISSING cols=%s", ",".join(missing))
        return False
    return True


def _load_symbols_db(conn: Optional[PGConnection], limit: int) -> tuple[List[str], Optional[date]]:
    if conn is None:
        return [], None
    if not _validate_candidates_schema(conn):
        return [], None
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT symbol, run_date
                FROM screener_candidates
                WHERE run_date = (SELECT MAX(run_date) FROM screener_candidates)
                ORDER BY score DESC NULLS LAST
                LIMIT %(limit)s
                """,
                {"limit": limit},
            )
            rows = cursor.fetchall()
        symbols = [row[0] for row in rows if row and row[0]]
        run_date = rows[0][1] if rows and rows[0] else None
        logger.info("CANDIDATES_LOADED db_count=%d", len(symbols))
        if run_date:
            logger.info("BACKTEST_DB_CANDIDATES_OK run_date=%s rows=%s", run_date, len(symbols))
        return symbols, run_date
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.info("BACKTEST_DB_CANDIDATES_FALLBACK err=%s", exc)
        return [], None


def _parse_bool_flag(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.lower() == "true"


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or [])
    if not db.db_enabled():
        logger.error("BACKTEST_DB_REQUIRED: DATABASE_URL/DB_* not configured.")
        return 2
    conn = _db_engine_if_available()
    if conn is None:
        logger.error("BACKTEST_DB_REQUIRED: unable to connect to database.")
        return 2
    export_default = False
    export_csv = _parse_bool_flag(args.export_csv, export_default)
    if export_csv:
        logger.warning("BACKTEST_CSV_EXPORT_DISABLED: DB is source of truth.")
        export_csv = False
    candidate_limit = args.limit if args.limit is not None else 15

    symbols: List[str] = []
    run_date: Optional[date] = None
    symbols_source: Optional[str] = None

    symbols, run_date = _load_symbols_db(conn, candidate_limit)
    if symbols:
        symbols_source = "screener_candidates"

    run_date = run_date or _determine_run_date(conn)

    if len(symbols) == 0:
        logger.info("BACKTEST_CANDIDATES_EMPTY source=db")
        logger.info("Backtest: no candidates today - skipping.")
        end_time = datetime.utcnow()
        elapsed_time = end_time - start_time
        logger.info("Script finished in %s", elapsed_time)
        try:
            if conn:
                conn.close()
        except Exception:
            pass
        return 0

    try:
        if symbols_source:
            logger.info("Loaded %d symbols from %s", len(symbols), symbols_source)
        run_backtest(
            symbols,
            max_symbols=args.max_symbols,
            max_days=args.max_days,
            quick=args.quick,
            run_date=run_date,
            export_csv=export_csv,
            enable_db=bool(conn),
        )
        logger.info("Backtest script finished")
    except Exception as exc:
        logger.error("Backtest failed: %s", exc)
    finally:
        end_time = datetime.utcnow()
        elapsed_time = end_time - start_time
        logger.info("Script finished in %s", elapsed_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
