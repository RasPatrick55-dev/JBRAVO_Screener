"""Comprehensive JBravo strategy backtester."""

from __future__ import annotations

import os
import sys

# Insert the project root directory into Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import logging.handlers
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import json

import numpy as np
import pandas as pd

from dotenv import load_dotenv

# Import indicator helpers from screener to keep the scoring consistent
from scripts.indicators import adx, aroon, macd, obv, rsi, compute_indicators
from scripts.utils import write_csv_atomic, cache_bars

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

error_log_path = os.path.join(BASE_DIR, "logs", "error.log")
error_handler = logging.handlers.RotatingFileHandler(
    error_log_path, maxBytes=2_000_000, backupCount=5
)
error_handler.setLevel(logging.ERROR)

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "logs", "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
)
logging.getLogger().addHandler(error_handler)

logger = logging.getLogger(__name__)
logger.info("Backtest script started.")

dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
CONFIG = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        CONFIG = json.load(f)

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

try:
    from alpaca.data.historical import StockHistoricalDataClient

    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
except Exception as exc:  # pragma: no cover - optional dependency
    data_client = None
    logger.error("Alpaca client unavailable: %s", exc)


def get_data(symbol: str, days: int = 800) -> pd.DataFrame:
    if data_client is None:
        logger.error("Data client not initialized. Returning empty DataFrame.")
        return pd.DataFrame()

    df = cache_bars(symbol, data_client, os.path.join(BASE_DIR, "data", "history_cache"), days)
    return df[["open", "high", "low", "close", "volume"]] if not df.empty else pd.DataFrame()




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
    trailing_stop: float


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


class PortfolioBacktester:
    """Simple portfolio level backtester for the JBravo strategy."""

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_cash: float = 100_000.0,
        alloc_pct: float = 0.03,
        top_n: int = 3,
        max_positions: int = 4,
        trail_pct: float = 0.03,
        max_hold_days: int = 7,
        trade_cost: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        self.data = data
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.alloc_pct = alloc_pct
        self.top_n = top_n
        self.max_positions = max_positions
        self.trail_pct = trail_pct
        self.max_hold_days = max_hold_days
        self.trade_cost = trade_cost
        self.slippage = slippage

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[tuple[pd.Timestamp, float]] = []

        # Build a unified date index
        indices = [df.index for df in self.data.values()]
        self.dates = sorted(set().union(*indices))

    def _open_position(self, symbol: str, price: float, date: pd.Timestamp) -> None:
        alloc = self.cash * self.alloc_pct
        qty = math.floor(alloc / price)
        if qty <= 0:
            return
        cost = qty * price * (1 + self.slippage) + self.trade_cost
        if cost > self.cash:
            return
        self.cash -= cost
        trailing = price * (1 - self.trail_pct)
        self.positions[symbol] = Position(
            symbol=symbol,
            qty=qty,
            entry_price=price,
            entry_time=date,
            highest_close=price,
            trailing_stop=trailing,
        )
        logger.info("Opened %s @ %.2f (%d shares)", symbol, price, qty)

    def _close_position(
        self, symbol: str, price: float, date: pd.Timestamp, reason: str
    ) -> None:
        pos = self.positions.pop(symbol)
        proceeds = pos.qty * price * (1 - self.slippage) - self.trade_cost
        self.cash += proceeds
        pnl = (price - pos.entry_price) * pos.qty
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
            )
        )
        logger.info("Closed %s @ %.2f reason=%s", symbol, price, reason)

    def run(self) -> None:
        for date in self.dates:
            # Update trailing stops and evaluate exits
            for symbol in list(self.positions):
                df = self.data[symbol]
                if date not in df.index:
                    continue
                row = df.loc[date]
                pos = self.positions[symbol]
                close = row["close"]

                pos.highest_close = max(pos.highest_close, close)
                pos.trailing_stop = max(
                    pos.trailing_stop, pos.highest_close * (1 - self.trail_pct)
                )
                hold_days = (date - pos.entry_time).days

                exit_price = close
                reason = None
                if close <= pos.trailing_stop:
                    exit_price = pos.trailing_stop
                    reason = "Trailing Stop"
                elif close < row["ema20"]:
                    reason = "EMA20"
                elif row["rsi"] > 70:
                    reason = "RSI70"
                elif hold_days >= self.max_hold_days:
                    reason = "MaxHold"

                if reason:
                    self._close_position(symbol, exit_price, date, reason)

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
                self._open_position(symbol, row["close"], date)
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


def run_backtest(symbols: List[str]) -> None:
    valid_symbols: List[str] = []
    for symbol in symbols:
        if re.match(r'^[A-Z]{1,5}$', symbol):
            valid_symbols.append(symbol)
        else:
            logger.warning("Invalid symbol skipped: %s", symbol)

    data = {}
    for sym in valid_symbols:
        logger.info("Fetching data for %s", sym)
        df = get_data(sym)
        if df.empty or len(df) < 250:
            logger.warning("Skipping %s: insufficient data", sym)
            continue
        df = compute_indicators(df)
        df["score"] = composite_score(df)
        data[sym] = df.dropna(subset=["score"])

    if not data:
        logger.error("No valid data to run backtest.")
        return

    bt = PortfolioBacktester(
        data,
        trail_pct=CONFIG.get('trail_pct', 0.03),
        max_hold_days=CONFIG.get('max_hold_days', 7),
    )
    bt.run()

    trades_df = bt.results()
    # Ensure timestamp columns for dashboard compatibility
    if "entry_time" not in trades_df.columns and "entry_date" in trades_df.columns:
        trades_df["entry_time"] = trades_df["entry_date"]
    if "exit_time" not in trades_df.columns and "exit_date" in trades_df.columns:
        trades_df["exit_time"] = trades_df["exit_date"]
    for col in ["entry_date", "exit_date"]:
        if col in trades_df.columns:
            trades_df.drop(columns=col, inplace=True)

    equity_df = bt.equity()

    # Aggregate per-symbol metrics from the trades log
    if not trades_df.empty:
        summary_df = (
            trades_df.groupby("symbol")
            .agg(
                trades=("pnl", "size"),
                wins=("pnl", lambda x: (x > 0).sum()),
                losses=("pnl", lambda x: (x <= 0).sum()),
                net_pnl=("pnl", "sum"),
            )
            .reset_index()
        )
        summary_df["win_rate"] = (
            summary_df["wins"] / summary_df["trades"] * 100
        )
    else:
        summary_df = pd.DataFrame(
            columns=["symbol", "trades", "wins", "losses", "net_pnl", "win_rate"]
        )

    summary_df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary_df["symbols_tested"] = len(symbols)

    for _, row in summary_df.iterrows():
        logger.info(
            "Backtest %s win_rate=%.2f%% net_pnl=%.2f",
            row.symbol,
            row.win_rate,
            row.net_pnl,
        )

    trades_path = os.path.join(BASE_DIR, "data", "trades_log.csv")
    equity_path = os.path.join(BASE_DIR, "data", "equity_curve.csv")
    metrics_path = os.path.join(BASE_DIR, "data", "backtest_results.csv")

    write_csv_atomic(trades_df, trades_path)
    write_csv_atomic(equity_df.reset_index(), equity_path)
    write_csv_atomic(summary_df, metrics_path)

    logger.info("Backtest complete. Results saved to data directory")


if __name__ == "__main__":
    try:
        csv_path = os.path.join(BASE_DIR, "data", "top_candidates.csv")
        symbols_df = pd.read_csv(csv_path)
        top_candidates = pd.read_csv(csv_path)
        symbol_list = top_candidates["symbol"].tolist()
        logger.info("Loaded %d symbols from %s", len(symbol_list), csv_path)
        run_backtest(symbol_list)
        logger.info("Backtest script finished.")
    except Exception as exc:
        logger.error("Backtest failed: %s", exc)

