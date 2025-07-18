"""Stock screener using a composite indicator based score.

This version expands on the previous basic moving average check and assigns
points based on a variety of popular technical indicators.  The goal is to
produce a ranked list of symbols with the strongest bullish characteristics for
short term swing trading.  Only Alpaca tradable assets are evaluated.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import sqlite3
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from dotenv import load_dotenv
import requests

from indicators import adx, aroon, macd, obv, rsi
from utils import write_csv_atomic, cache_bars



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

ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")
DATA_CACHE_DIR = os.path.join(BASE_DIR, 'data', 'history_cache')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, 'data', 'pipeline.db')

# Tunable strategy parameters
RSI_OVERBOUGHT = 70
RSI_BULLISH = 50
SMA_SHORT = 9
EMA_MID = 20
SMA_LONG = 180
TRAIL_PERCENT = 3.0
MAX_HOLD_DAYS = 7


def send_alert(message: str) -> None:
    if not ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(ALERT_WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as exc:
        logging.error("Failed to send alert: %s", exc)


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS historical_candidates (date TEXT, symbol TEXT, score REAL)"
        )


init_db()

# Ensure historical candidates file exists
hist_init_path = os.path.join(BASE_DIR, 'data', 'historical_candidates.csv')
if not os.path.exists(hist_init_path):
    write_csv_atomic(pd.DataFrame(columns=['date', 'symbol', 'score']), hist_init_path)

# Ensure top candidates file exists
top_init_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
if not os.path.exists(top_init_path):
    write_csv_atomic(pd.DataFrame(columns=["symbol", "score"]), top_init_path)

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

required_bars = 250


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_score(symbol: str, df: pd.DataFrame) -> dict | None:
    try:
        logging.debug("Running compute_score for %s with %d rows", symbol, len(df))
        df = df.copy().sort_index()
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
        df["sma9"] = df["close"].rolling(SMA_SHORT).mean()
        df["ema20"] = df["close"].ewm(span=EMA_MID, adjust=False).mean()
        df["sma180"] = df["close"].rolling(SMA_LONG).mean()
        df["atr"] = compute_atr(df)
        df["year_high"] = df["high"].rolling(252).max().shift(1)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        score = 0.0
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
        if last["close"] > last["sma9"] and prev["close"] <= prev["sma9"]:
            score += 2
        if last["sma9"] > last["ema20"] and last["ema20"] > last["sma180"]:
            score += 2
        sma9_slope = last["sma9"] - prev["sma9"]
        ema20_slope = last["ema20"] - prev["ema20"]
        score += 0.5 if sma9_slope > 0 else -0.5
        score += 0.5 if ema20_slope > 0 else -0.5
        if last["rsi"] > 60:
            score += 1
        elif last["rsi"] > RSI_BULLISH:
            score += 0.5
        if last["rsi"] > RSI_OVERBOUGHT:
            score -= 2
        if last["macd"] > 0:
            score += 0.5
        if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]:
            score += 0.5
        if last["macd_hist"] < prev["macd_hist"]:
            score -= 0.5
        atr_ratio = last["atr"] / last["close"] if last["close"] > 0 else 0
        if pd.notna(atr_ratio):
            if atr_ratio < 0.05:
                score += 0.5
            elif atr_ratio > 0.07:
                score -= 0.5
        if pd.notna(last["year_high"]):
            if last["close"] >= 0.9 * last["year_high"]:
                score += 0.5
            if last["close"] >= last["year_high"]:
                score += 1
        if last["close"] < last["ema20"] and prev["close"] >= prev["ema20"]:
            score -= 2
        if last["macd_hist"] < 0 and prev["macd_hist"] >= 0:
            score -= 1
        if last["rsi"] < prev["rsi"]:
            score -= 0.5
        score_dict = {"symbol": symbol, "score": round(score, 2)}
        logging.debug("compute_score for %s returned %s", symbol, score_dict)
        return score_dict
    except Exception as exc:
        logging.error("%s processing failed: %s", symbol, exc)
        send_alert(f"Screener failed for {symbol}: {exc}")
        return None


records: list[dict] = []
for symbol in symbols:
    logging.info("Processing %s...", symbol)
    df = cache_bars(symbol, data_client, DATA_CACHE_DIR)
    if len(df) < required_bars:
        logging.warning("Skipping %s: insufficient data (%d bars)", symbol, len(df))
        continue
    try:
        rec = compute_score(symbol, df)
        if rec is not None:
            records.append(rec)
        else:
            logging.info("Skipping %s: compute_score returned None", symbol)
    except Exception as e:
        logging.error("compute_score failed for %s: %s", symbol, e, exc_info=True)

# Convert to DataFrame and rank
ranked_df = pd.DataFrame(records)
logging.debug("Screener output columns: %s", ranked_df.columns.tolist())
if (
    ranked_df.empty
    or "score" not in ranked_df.columns
    or ranked_df["score"].isnull().all()
):
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
