"""Stock screener using a composite indicator based score.

This version expands on the previous basic moving average check and assigns
points based on a variety of popular technical indicators.  The goal is to
produce a ranked list of symbols with the strongest bullish characteristics for
short term swing trading.  Only Alpaca tradable assets are evaluated.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False).mean()
    hist = line - signal_line
    return line, signal_line, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0.0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(period).sum() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.rolling(period).mean()


def aroon(df: pd.DataFrame, period: int = 25):
    """Calculate Aroon Up and Aroon Down."""
    high_idx = (
        df["high"].rolling(period + 1).apply(lambda x: period - 1 - np.argmax(x), raw=True)
    )
    low_idx = (
        df["low"].rolling(period + 1).apply(lambda x: period - 1 - np.argmin(x), raw=True)
    )
    aroon_up = 100 * (period - high_idx) / period
    aroon_down = 100 * (period - low_idx) / period
    return aroon_up, aroon_down


def obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

log_path = os.path.join(BASE_DIR, 'logs', 'screener.log')
error_log_path = os.path.join(BASE_DIR, 'logs', 'error.log')

# Configure a rotating error handler so logs don't grow unbounded
error_handler = RotatingFileHandler(error_log_path, maxBytes=2_000_000, backupCount=5)
error_handler.setLevel(logging.ERROR)

logging.basicConfig(
    handlers=[
        RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5),
        error_handler,
    ],
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load environment variables
dotenv_path = os.path.join(BASE_DIR, '.env')
logging.info("Loading environment variables from %s", dotenv_path)
load_dotenv(dotenv_path)

# Ensure historical candidates file exists
hist_init_path = os.path.join(BASE_DIR, 'data', 'historical_candidates.csv')
if not os.path.exists(hist_init_path):
    pd.DataFrame(columns=['date', 'symbol', 'score']).to_csv(hist_init_path, index=False)

# Ensure top candidates file exists
top_init_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
if not os.path.exists(top_init_path):
    pd.DataFrame(columns=["symbol", "score"]).to_csv(top_init_path, index=False)

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    logging.error("Missing API credentials. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the .env file.")
    raise SystemExit(1)

# Initialize Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Fetch all tradable symbols
assets = trading_client.get_all_assets()
symbols = [a.symbol for a in assets if a.tradable and a.status == "active" and a.exchange in ("NYSE", "NASDAQ")]

# Required number of daily bars for indicator calculations
required_bars = 250

ranked_candidates: list[dict] = []

# Screening and ranking criteria
for symbol in symbols:
    logging.info("Processing %s...", symbol)
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=(datetime.now(timezone.utc) - timedelta(days=800)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=(datetime.now(timezone.utc) - timedelta(minutes=16)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        bars = data_client.get_stock_bars(request_params).df

        logging.debug(
            "%s: bar-count=%d required=%d", symbol, len(bars), required_bars
        )

        if len(bars) < required_bars:
            logging.warning("Skipping %s: insufficient data (%d bars)", symbol, len(bars))
            continue

        df = bars.copy().sort_index()
        df["ma50"] = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()
        df["rsi"] = rsi(df["close"])
        macd_line, macd_signal, macd_hist = macd(df["close"])
        df["macd"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        df["adx"] = adx(df)
        df["aroon_up"], df["aroon_down"] = aroon(df)
        df["obv"] = obv(df)
        df["vol_avg30"] = df["volume"].rolling(30).mean()
        df["month_high"] = df["high"].rolling(21).max().shift(1)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        score = 0.0

        # Moving averages
        score += 1 if last["close"] > last["ma50"] else -1
        score += 1 if last["close"] > last["ma200"] else -1
        if last["ma50"] > last["ma200"] and prev["ma50"] <= prev["ma200"]:
            score += 1.5

        # RSI
        if last["rsi"] > 50 and prev["rsi"] <= 50:
            score += 1
        if last["rsi"] > 30 and prev["rsi"] <= 30:
            score += 1
        if last["rsi"] > 70:
            score -= 1

        # MACD
        score += 1 if last["macd"] > last["macd_signal"] else -1
        if last["macd_hist"] > prev["macd_hist"]:
            score += 1

        # ADX
        if last["adx"] > 20:
            score += 1
        if last["adx"] > 40:
            score += 0.5

        # Aroon
        if last["aroon_up"] > last["aroon_down"] and prev["aroon_up"] <= prev["aroon_down"]:
            score += 1
        if last["aroon_up"] > 70:
            score += 1

        # OBV trend
        score += 1 if last["obv"] > prev["obv"] else -1

        # Volume spike
        if last["volume"] > 2 * last["vol_avg30"]:
            score += 1

        # Breakout
        if last["close"] > last["month_high"]:
            score += 1

        # Candlestick patterns
        body = abs(last["close"] - last["open"])
        lower = last["low"] - min(last["close"], last["open"])
        upper = last["high"] - max(last["close"], last["open"])
        if lower > 2 * body and upper <= body:
            score += 1
        prev_body = abs(prev["close"] - prev["open"])
        if (
            prev["close"] < prev["open"]
            and last["close"] > last["open"]
            and last["close"] > prev["open"]
            and last["open"] < prev["close"]
            and prev_body > 0
        ):
            score += 1

        ranked_candidates.append({"symbol": symbol, "score": round(score, 2)})

    except Exception as e:
        logging.error("%s failed: %s", symbol, e)

# Convert to DataFrame and rank
ranked_df = pd.DataFrame(ranked_candidates)
ranked_df.sort_values(by="score", ascending=False, inplace=True)

csv_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
if ranked_df.empty:
    logging.warning("No candidates met the screening criteria.")
    ranked_df = pd.DataFrame(columns=["symbol", "score"])

ranked_df.head(15).to_csv(csv_path, index=False)
logging.info("Top 15 ranked candidates saved to %s", csv_path)

# Append to historical candidates log
hist_path = os.path.join(BASE_DIR, 'data', 'historical_candidates.csv')
append_df = ranked_df.head(15).copy()
append_df.insert(0, 'date', datetime.now().strftime('%Y-%m-%d'))
if not os.path.exists(hist_path):
    append_df.to_csv(hist_path, index=False)
else:
    append_df.to_csv(hist_path, mode='a', header=False, index=False)
logging.info("Historical candidates updated at %s", hist_path)
