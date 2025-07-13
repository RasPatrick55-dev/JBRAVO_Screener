"""Comprehensive JBravo strategy backtester."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

from dotenv import load_dotenv

# Import indicator helpers from screener to keep the scoring consistent
from . import screener

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

log_path = os.path.join(BASE_DIR, "logs", "backtest.log")
error_log_path = os.path.join(BASE_DIR, "logs", "error.log")

error_handler = logging.handlers.RotatingFileHandler(
    error_log_path, maxBytes=2_000_000, backupCount=5
)
error_handler.setLevel(logging.ERROR)

logging.basicConfig(
    handlers=[
        logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=5
        ),
        error_handler,
    ],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
except Exception as exc:  # pragma: no cover - optional dependency
    data_client = None
    logging.error("Alpaca client unavailable: %s", exc)


def get_data(symbol: str, days: int = 800) -> pd.DataFrame:
    """Fetch OHLCV data from Alpaca. Returns empty DataFrame on failure."""

    if data_client is None:
        logging.error("Data client not initialized. Returning empty DataFrame.")
        return pd.DataFrame()

    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            limit=days,
        )
        bars = data_client.get_stock_bars(request_params).df
    except Exception as exc:  # pragma: no cover - network call
        logging.error("%s: data fetch failed: %s", symbol, exc)
        return pd.DataFrame()

    if bars.empty:
        logging.warning("%s: no bars returned", symbol)
        return pd.DataFrame()

    bars = bars.reset_index()
    bars["datetime"] = pd.to_datetime(bars["timestamp"])
    bars.set_index("datetime", inplace=True)
    return bars[["open", "high", "low", "close", "volume"]]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicator columns used in the screener."""

    df = df.copy()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    df["rsi"] = screener.rsi(df["close"])
    macd_line, macd_signal, macd_hist = screener.macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["adx"] = screener.adx(df)
    df["aroon_up"], df["aroon_down"] = screener.aroon(df)
    df["obv"] = screener.obv(df)
    df["vol_avg30"] = df["volume"].rolling(30).mean()
    df["month_high"] = df["high"].rolling(21).max().shift(1)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df


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
    entry_date: pd.Timestamp
    highest_close: float
    trailing_stop: float


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
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
            entry_date=date,
            highest_close=price,
            trailing_stop=trailing,
        )
        logging.info("Opened %s @ %.2f (%d shares)", symbol, price, qty)

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
                entry_date=pos.entry_date,
                exit_date=date,
                entry_price=pos.entry_price,
                exit_price=price,
                qty=pos.qty,
                pnl=pnl,
                exit_reason=reason,
            )
        )
        logging.info("Closed %s @ %.2f reason=%s", symbol, price, reason)

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
                hold_days = (date - pos.entry_date).days

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
    data = {}
    for sym in symbols:
        logging.info("Fetching data for %s", sym)
        df = get_data(sym)
        if df.empty or len(df) < 250:
            logging.warning("Skipping %s: insufficient data", sym)
            continue
        df = compute_indicators(df)
        df["score"] = composite_score(df)
        data[sym] = df.dropna(subset=["score"])

    if not data:
        logging.error("No valid data to run backtest.")
        return

    bt = PortfolioBacktester(data)
    bt.run()

    trades_df = bt.results()
    equity_df = bt.equity()
    metrics = bt.metrics()

    trades_path = os.path.join(BASE_DIR, "data", "trades_log.csv")
    equity_path = os.path.join(BASE_DIR, "data", "equity_curve.csv")
    metrics_path = os.path.join(BASE_DIR, "data", "backtest_results.csv")

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    logging.info("Backtest complete. Results saved to data directory")


if __name__ == "__main__":
    try:
        csv_path = os.path.join(BASE_DIR, "data", "top_candidates.csv")
        symbols_df = pd.read_csv(csv_path)
        symbol_list = symbols_df.iloc[:, 0].tolist()
        run_backtest(symbol_list)
    except Exception as exc:
        logging.error("Backtest failed: %s", exc)

