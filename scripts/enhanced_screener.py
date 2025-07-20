"""
Enhanced stock screener based on the original JBravo composite indicator
with additional signals inspired by the Johnny‑Bravo methodology.

This version preserves all of the existing scoring logic from the
previous screener but augments it with new features aimed at detecting
the onset of strong up‑trends and spotting signs of weakness before
momentum fades.  The goal is to rank symbols in a way that favours
high‑probability swing trades while still respecting risk and volatility.

Key additions over the baseline implementation:

* **JBravo moving‑average alignment** – compute short/medium/long moving
  averages (SMA9, EMA20, SMA180) and reward symbols where today’s
  close has crossed above the 9‑day SMA and the averages are stacked
  bullishly (SMA9 > EMA20 > SMA180) with positive slopes【514888614236530†L127-L165】.
* **Momentum confirmation** – emphasise RSI above 50 and MACD above
  zero; penalise overbought conditions when RSI exceeds 70 or the
  MACD histogram is declining【514888614236530†L167-L179】.
* **Volatility filter** – compute a 14‑period Average True Range (ATR)
  and scale it by the current close; penalise very high relative
  volatility and slightly reward low volatility to avoid whipsaw trades.
* **Longer term breakout context** – compare the current close to the
  52‑week high to reward names approaching new highs and down‑rank
  those far below their year‑long range.
* **Exit warning signals** – subtract points when price closes below
  the EMA20 or when momentum indicators roll over.  These factors
  don’t automatically remove a symbol but help rank it lower so the
  executor focuses on fresher trends.

The remainder of the pipeline (data caching, ranking and output) is
unchanged from the original screener.  See the project documentation
for more detail on the underlying trading strategy and risk
management rules【514888614236530†L41-L53】.
"""

import os
import sys

# Ensure the root of the repository is on the Python path so we can
# import shared modules such as ``indicators`` and ``utils``.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import sqlite3
from logging.handlers import RotatingFileHandler
from datetime import datetime

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import requests

from indicators import adx, aroon, macd, obv, rsi
from utils import write_csv_atomic, cache_bars


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

log_path = os.path.join(BASE_DIR, 'logs', 'enhanced_screener.log')
error_log_path = os.path.join(BASE_DIR, 'logs', 'error.log')

# Configure logging with rotation to prevent unbounded growth
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

# Load environment variables (API credentials, webhook endpoints, etc.)
dotenv_path = os.path.join(BASE_DIR, '.env')
logging.info("Loading environment variables from %s", dotenv_path)
load_dotenv(dotenv_path)

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")
DATA_CACHE_DIR = os.path.join(BASE_DIR, 'data', 'history_cache')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'data', 'pipeline.db')


def send_alert(message: str) -> None:
    """Send a simple JSON message to the configured webhook.

    This function is used to surface failures in the screener via
    Slack/Discord/Teams.  It is intentionally resilient: on any
    exception it simply logs the error and returns.  See
    ``ALERT_WEBHOOK_URL`` in the environment.
    """
    if not ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(ALERT_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as exc:
        logging.error("Failed to send alert: %s", exc)


def init_db() -> None:
    """Ensure the SQLite table for historical candidates exists."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS historical_candidates (date TEXT, symbol TEXT, score REAL)"
        )
        columns = [row[1] for row in conn.execute("PRAGMA table_info(historical_candidates)")]
        if "rsi" not in columns:
            conn.execute("ALTER TABLE historical_candidates ADD COLUMN rsi REAL;")


init_db()

# Prepare initial CSVs if they do not already exist
hist_init_path = os.path.join(BASE_DIR, 'data', 'historical_candidates.csv')
if not os.path.exists(hist_init_path):
    write_csv_atomic(pd.DataFrame(columns=['date', 'symbol', 'score']), hist_init_path)

top_init_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
if not os.path.exists(top_init_path):
    write_csv_atomic(pd.DataFrame(columns=["symbol", "score"]), top_init_path)


# Gather all tradable tickers via Alpaca.  We defer the Alpaca imports
# until after loading environment variables to avoid unnecessary errors
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    logging.error("Missing API credentials. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the .env file.")
    raise SystemExit(1)

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Filter for active, tradable NYSE/NASDAQ symbols only
assets = trading_client.get_all_assets()
symbols = [a.symbol for a in assets if a.tradable and a.status == "active" and a.exchange in ("NYSE", "NASDAQ")]

# We need at least 250 bars to compute long period indicators
required_bars = 250


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute the Average True Range (ATR).

    ATR is calculated as the rolling mean of the True Range, which
    considers gaps from one period to the next.  We use a simple
    moving average here; you could switch to an EMA if preferred.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def compute_score(symbol: str, df: pd.DataFrame) -> dict | None:
    """Calculate a composite score for a single symbol.

    The scoring system sums up discrete points from multiple indicators.
    Positive values indicate a stronger bullish setup, negative values
    reflect bearish or weak conditions.  See module‑level docstring for
    details on the individual components.
    """
    try:
        # Work on a copy to avoid mutating the shared DataFrame
        df = df.copy().sort_index()

        # Basic moving averages from the original screener
        df["ma50"] = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()

        # JBravo specific averages
        df["sma9"] = df["close"].rolling(9).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["sma180"] = df["close"].rolling(180).mean()

        # Traditional momentum indicators
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
        df["year_high"] = df["high"].rolling(252).max().shift(1)
        df["atr"] = compute_atr(df)

        # Grab the last two rows for indicator comparisons
        last = df.iloc[-1]
        prev = df.iloc[-2]

        score = 0.0

        # === Original scoring logic ===
        score += 1 if last["close"] > last["ma50"] else -1
        score += 1 if last["close"] > last["ma200"] else -1
        if last["ma50"] > last["ma200"] and prev["ma50"] <= prev["ma200"]:
            score += 1.5
        if last["rsi"] > 50 and prev["rsi"] <= 50:
            score += 1
        if last["rsi"] > 30 and prev["rsi"] <= 30:
            score += 1
        if last["rsi"] > 70:
            score -= 1
        score += 1 if last["macd"] > last["macd_signal"] else -1
        if last["macd_hist"] > prev["macd_hist"]:
            score += 1
        if last["adx"] > 20:
            score += 1
        if last["adx"] > 40:
            score += 0.5
        if last["aroon_up"] > last["aroon_down"] and prev["aroon_up"] <= prev["aroon_down"]:
            score += 1
        if last["aroon_up"] > 70:
            score += 1
        score += 1 if last["obv"] > prev["obv"] else -1
        if last["volume"] > 2 * last["vol_avg30"]:
            score += 1
        if last["close"] > last["month_high"]:
            score += 1
        # Simple hammer/engulfing candle patterns
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

        # === JBravo enhancements ===
        # Price crosses above the 9‑day SMA
        if last["close"] > last["sma9"] and prev["close"] <= prev["sma9"]:
            score += 2
        # Moving‑average alignment: SMA9 > EMA20 > SMA180
        if last["sma9"] > last["ema20"] and last["ema20"] > last["sma180"]:
            score += 2
        # Reward upward slope of SMA9 and EMA20, penalise downward slope
        sma9_slope = last["sma9"] - prev["sma9"]
        ema20_slope = last["ema20"] - prev["ema20"]
        if sma9_slope > 0:
            score += 0.5
        else:
            score -= 0.5
        if ema20_slope > 0:
            score += 0.5
        else:
            score -= 0.5
        # Momentum confirmation: RSI must be above 50; stronger push if >60
        if last["rsi"] > 60:
            score += 1
        elif last["rsi"] > 50:
            score += 0.5
        # Punish overbought conditions more aggressively
        if last["rsi"] > 70:
            score -= 2
        # MACD above zero (bullish trend) earns extra weight
        if last["macd"] > 0:
            score += 0.5
        if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
            score += 0.5
        if last["macd_hist"] < prev["macd_hist"]:
            score -= 0.5
        # Volatility filter: relative ATR
        atr_ratio = last["atr"] / last["close"] if last["close"] > 0 else 0
        if pd.notna(atr_ratio):
            if atr_ratio < 0.05:
                score += 0.5  # stable
            elif atr_ratio > 0.07:
                score -= 0.5  # very volatile
        # Longer‑term breakout context
        if pd.notna(last["year_high"]):
            # Reward proximity to the 52‑week high
            if last["close"] >= 0.9 * last["year_high"]:
                score += 0.5
            if last["close"] >= last["year_high"]:
                score += 1
        # Exit warnings: price closing below EMA20 or momentum rolling over
        if last["close"] < last["ema20"] and prev["close"] >= prev["ema20"]:
            score -= 2
        if last["macd_hist"] < 0 and prev["macd_hist"] >= 0:
            score -= 1
        if last["rsi"] < prev["rsi"]:
            score -= 0.5

        return {"symbol": symbol, "score": round(score, 2)}
    except Exception as exc:
        logging.error("%s processing failed: %s", symbol, exc)
        send_alert(f"Screener failed for {symbol}: {exc}")
        return None


# Main execution: compute scores in parallel and write outputs
def main() -> None:
    ranked_candidates: list[dict] = []
    futures = []
    executor = ThreadPoolExecutor(max_workers=4)
    for symbol in symbols:
        logging.info("Processing %s...", symbol)
        df = cache_bars(symbol, data_client, DATA_CACHE_DIR)
        if len(df) < required_bars:
            logging.warning("Skipping %s: insufficient data (%d bars)", symbol, len(df))
            continue
        futures.append(executor.submit(compute_score, symbol, df))
    for fut in futures:
        res = fut.result()
        if res:
            ranked_candidates.append(res)
    # Convert to DataFrame and rank
    ranked_df = pd.DataFrame(ranked_candidates)
    if "score" not in ranked_df.columns:
        logging.error(
            "Screener output missing 'score'. DataFrame columns: %s",
            ranked_df.columns.tolist(),
        )
        sys.exit(1)
    ranked_df.sort_values(by="score", ascending=False, inplace=True)
    csv_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
    if ranked_df.empty:
        logging.warning("No candidates met the screening criteria.")
        ranked_df = pd.DataFrame(columns=["symbol", "score"])
    write_csv_atomic(ranked_df.head(15), csv_path)
    logging.info("Top 15 ranked candidates saved to %s", csv_path)
    # Append to historical candidates log
    hist_path = os.path.join(BASE_DIR, 'data', 'historical_candidates.csv')
    append_df = ranked_df.head(15).copy()
    append_df.insert(0, 'date', datetime.now().strftime('%Y-%m-%d'))
    write_csv_atomic(append_df, hist_path)
    logging.info("Historical candidates updated at %s", hist_path)
    with sqlite3.connect(DB_PATH) as conn:
        append_df.to_sql("historical_candidates", conn, if_exists="append", index=False)


if __name__ == "__main__":
    main()