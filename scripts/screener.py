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
from datetime import datetime, timedelta, timezone
from utils import logger_utils

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import requests

from indicators import adx, aroon, macd, obv, rsi
from utils import write_csv_atomic, cache_bars



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

logger = logger_utils.init_logging(__name__, 'screener.log')
start_time = datetime.utcnow()
logger.info('Script started')

# Load environment variables
dotenv_path = os.path.join(BASE_DIR, '.env')
logger.info("Loading environment variables from %s", dotenv_path)
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
        logger.error("Failed to send alert: %s", exc)


from scripts.ensure_db_indicators import (
    ensure_columns,
    sync_columns_from_dataframe,
    REQUIRED_COLUMNS,
)


def init_db() -> None:
    """Ensure all required columns exist in the database."""
    ensure_columns(DB_PATH, REQUIRED_COLUMNS)


init_db()

# Ensure historical candidates file exists
hist_init_path = os.path.join(BASE_DIR, 'data', 'historical_candidates.csv')
if not os.path.exists(hist_init_path):
    write_csv_atomic(hist_init_path, pd.DataFrame(columns=['date', 'symbol', 'score']))

# Ensure top candidates file exists
top_init_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
if not os.path.exists(top_init_path):
    write_csv_atomic(top_init_path, pd.DataFrame(columns=["symbol", "score"]))

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    logger.error("Missing API credentials. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the .env file.")
    raise SystemExit(1)

# Initialize Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Fetch all tradable symbols
assets = trading_client.get_all_assets()
symbols = [a.symbol for a in assets if a.tradable and a.status == "active" and a.exchange in ("NYSE", "NASDAQ")]


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
        logger.debug("Running compute_score for %s with %d rows", symbol, len(df))
        if len(df) < 2:
            logger.warning("Skipping %s: not enough bars (%d)", symbol, len(df))
            return None
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
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        score = 0.0
        reasons: list[str] = []

        def add(val: float, reason: str, condition: bool = True) -> None:
            nonlocal score
            if not condition:
                return
            score += val
            reasons.append(f"{val:+g} {reason}")

        add(1 if last["close"] > last["ma50"] else -1, "close vs MA50", pd.notna(last.get("ma50")))
        add(1 if last["close"] > last["ma200"] else -1, "close vs MA200", pd.notna(last.get("ma200")))
        add(1.5, "MA50 crossover", pd.notna(last.get("ma50")) and pd.notna(prev.get("ma50")) and pd.notna(last.get("ma200")) and pd.notna(prev.get("ma200")) and last["ma50"] > last["ma200"] and prev["ma50"] <= prev["ma200"])
        add(1, "RSI > 50", pd.notna(last.get("rsi")) and pd.notna(prev.get("rsi")) and last["rsi"] > 50 and prev["rsi"] <= 50)
        add(1, "RSI > 30", pd.notna(last.get("rsi")) and pd.notna(prev.get("rsi")) and last["rsi"] > 30 and prev["rsi"] <= 30)
        add(-1, "RSI > 70", pd.notna(last.get("rsi")) and last["rsi"] > 70)
        add(1 if last["macd"] > last["macd_signal"] else -1, "MACD cross", pd.notna(last.get("macd")) and pd.notna(last.get("macd_signal")))
        add(1, "MACD hist rising", pd.notna(last.get("macd_hist")) and pd.notna(prev.get("macd_hist")) and last["macd_hist"] > prev["macd_hist"])
        add(1, "ADX > 20", pd.notna(last.get("adx")) and last["adx"] > 20)
        add(0.5, "ADX > 40", pd.notna(last.get("adx")) and last["adx"] > 40)
        add(1, "Aroon up cross", all(pd.notna([last.get("aroon_up"), last.get("aroon_down"), prev.get("aroon_up"), prev.get("aroon_down")])) and last["aroon_up"] > last["aroon_down"] and prev["aroon_up"] <= prev["aroon_down"])
        add(1, "Aroon > 70", pd.notna(last.get("aroon_up")) and last["aroon_up"] > 70)
        add(1 if last.get("obv") > prev.get("obv") else -1, "OBV", pd.notna(last.get("obv")) and pd.notna(prev.get("obv")))
        add(1, "Volume spike", pd.notna(last.get("volume")) and pd.notna(last.get("vol_avg30")) and last["volume"] > 2 * last["vol_avg30"])
        add(1, "New month high", pd.notna(last.get("month_high")) and last["close"] > last["month_high"])
        body = abs(last["close"] - last["open"])
        lower = last["low"] - min(last["close"], last["open"])
        upper = last["high"] - max(last["close"], last["open"])
        add(1, "Hammer", lower > 2 * body and upper <= body)
        prev_body = abs(prev["close"] - prev["open"])
        engulf = (
            prev["close"] < prev["open"]
            and last["close"] > last["open"]
            and last["close"] > prev["open"]
            and last["open"] < prev["close"]
            and prev_body > 0
        )
        add(1, "Bull engulf", engulf)

        # === JBravo enhancements ===
        add(2, "SMA9 crossover", pd.notna(last.get("sma9")) and pd.notna(prev.get("sma9")) and last["close"] > last["sma9"] and prev["close"] <= prev["sma9"])
        add(2, "MA stack", all(pd.notna([last.get("sma9"), last.get("ema20"), last.get("sma180")])) and last["sma9"] > last["ema20"] and last["ema20"] > last["sma180"])
        if pd.notna(last.get("sma9")) and pd.notna(prev.get("sma9")):
            add(0.5 if last["sma9"] - prev["sma9"] > 0 else -0.5, "SMA9 slope")
        if pd.notna(last.get("ema20")) and pd.notna(prev.get("ema20")):
            add(0.5 if last["ema20"] - prev["ema20"] > 0 else -0.5, "EMA20 slope")
        if pd.notna(last.get("rsi")):
            add(1, "RSI > 60", last["rsi"] > 60)
            add(0.5, "RSI > 50", 50 < last["rsi"] <= 60)
            add(-2, "RSI overbought", last["rsi"] > RSI_OVERBOUGHT)
        add(0.5, "MACD > 0", pd.notna(last.get("macd")) and last["macd"] > 0)
        add(0.5, "MACD hist up", pd.notna(last.get("macd_hist")) and pd.notna(prev.get("macd_hist")) and last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"])
        add(-0.5, "MACD hist down", pd.notna(last.get("macd_hist")) and pd.notna(prev.get("macd_hist")) and last["macd_hist"] < prev["macd_hist"])
        atr_ratio = last["atr"] / last["close"] if last["close"] > 0 else 0
        add(0.5, "Low ATR", pd.notna(atr_ratio) and atr_ratio < 0.05)
        add(-0.5, "High ATR", pd.notna(atr_ratio) and atr_ratio > 0.07)
        add(0.5, "Near 52w high", pd.notna(last.get("year_high")) and last["close"] >= 0.9 * last["year_high"])
        add(1, "New 52w high", pd.notna(last.get("year_high")) and last["close"] >= last["year_high"])
        add(-2, "Close < EMA20", all(pd.notna([last.get("ema20"), prev.get("ema20")])) and last["close"] < last["ema20"] and prev["close"] >= prev["ema20"])
        add(-1, "MACD hist flip", pd.notna(last.get("macd_hist")) and pd.notna(prev.get("macd_hist")) and last["macd_hist"] < 0 and prev["macd_hist"] >= 0)
        add(-0.5, "RSI falling", pd.notna(last.get("rsi")) and pd.notna(prev.get("rsi")) and last["rsi"] < prev["rsi"])

        result = {
            "symbol": symbol,
            "score": round(score, 2),
            "rsi": round(last.get("rsi", float("nan")), 2) if pd.notna(last.get("rsi")) else None,
            "macd": round(last.get("macd", float("nan")), 2) if pd.notna(last.get("macd")) else None,
            "macd_hist": round(last.get("macd_hist", float("nan")), 2) if pd.notna(last.get("macd_hist")) else None,
            "adx": round(last.get("adx", float("nan")), 2) if pd.notna(last.get("adx")) else None,
            "aroon_up": round(last.get("aroon_up", float("nan")), 2) if pd.notna(last.get("aroon_up")) else None,
            "aroon_down": round(last.get("aroon_down", float("nan")), 2) if pd.notna(last.get("aroon_down")) else None,
            "sma9": round(last.get("sma9", float("nan")), 2) if pd.notna(last.get("sma9")) else None,
            "ema20": round(last.get("ema20", float("nan")), 2) if pd.notna(last.get("ema20")) else None,
            "sma180": round(last.get("sma180", float("nan")), 2) if pd.notna(last.get("sma180")) else None,
            "atr": round(last.get("atr", float("nan")), 2) if pd.notna(last.get("atr")) else None,
            "score_breakdown": "; ".join(reasons),
        }
        logger.debug("compute_score for %s returned %s", symbol, result)
        return result
    except Exception as exc:
        logger.error("%s processing failed: %s", symbol, exc)
        send_alert(f"Screener failed for {symbol}: {exc}")
        return None


def main() -> None:
    records: list[dict] = []
    skipped = 0
    for symbol in symbols:
        logger.info("Processing %s...", symbol)
        now_utc = datetime.now(timezone.utc)
        end_safe = now_utc - timedelta(minutes=16)
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=datetime.now(timezone.utc) - timedelta(days=1500),
            end=end_safe.isoformat(),
            feed="iex",
        )
        bars = data_client.get_stock_bars(request_params)
        cache_bars(symbol, bars, DATA_CACHE_DIR)
        df = bars.df.reset_index()
        logger.debug("%s has %d bars", symbol, len(df))
        try:
            rec = compute_score(symbol, df)
            if rec is not None:
                records.append(rec)
            else:
                skipped += 1
                logger.info("Skipping %s: compute_score returned None", symbol)
        except Exception as e:
            skipped += 1
            logger.error("compute_score failed for %s: %s", symbol, e, exc_info=True)

    # Convert to DataFrame and rank
    logger.info(
        "Processed %d symbols, %d scored, %d skipped",
        len(symbols),
        len(records),
        skipped,
    )
    if not records:
        logger.error("All symbols skipped; no valid output.")
        sys.exit(1)

    ranked_df = pd.DataFrame(records)
    logger.debug("Screener output columns: %s", ranked_df.columns.tolist())
    if "score" not in ranked_df.columns:
        logger.error(
            "Screener output missing 'score'. DataFrame columns: %s",
            ranked_df.columns.tolist(),
        )
        sys.exit(1)
    if ranked_df.empty or ranked_df["score"].isnull().all():
        logger.error(
            "Screener output missing valid scores. DataFrame columns: %s",
            ranked_df.columns.tolist(),
        )
        sys.exit(1)
    ranked_df.sort_values(by="score", ascending=False, inplace=True)

# Save full scored list
    scored_path = os.path.join(BASE_DIR, "data", "scored_candidates.csv")
    write_csv_atomic(scored_path, ranked_df)
    logger.info("All scored candidates saved to %s", scored_path)

# Log details for top symbols
    for _, row in ranked_df.head(15).iterrows():
        logger.info("%s score %.2f: %s", row.symbol, row.score, row.score_breakdown)

# Prepare top candidates with timestamp and universe info
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    top15 = ranked_df.head(15).copy()
    top15.insert(0, "timestamp", timestamp)
    top15["universe_count"] = len(symbols)

    cols = [
        "symbol",
        "score",
        "timestamp",
    ] + [c for c in top15.columns if c not in ["symbol", "score", "timestamp"]]
    top15 = top15[cols]

    csv_path = os.path.join(BASE_DIR, "data", "top_candidates.csv")
    write_csv_atomic(csv_path, top15)
    logger.info(
        "Top candidates updated: %d records written to %s",
        len(top15),
        csv_path,
    )

# Append to historical candidates log
    hist_path = os.path.join(BASE_DIR, "data", "historical_candidates.csv")
    append_df = top15.copy()
    append_df.insert(0, "date", datetime.now().strftime("%Y-%m-%d"))
    write_csv_atomic(hist_path, append_df)
    logger.info("Historical candidates updated at %s", hist_path)
    # Synchronize SQLite schema to match the DataFrame before insertion
    sync_columns_from_dataframe(append_df, DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        append_df.to_sql("historical_candidates", conn, if_exists="append", index=False)

# Update screener summary CSV
    summary = "; ".join([
        f"{r.symbol}({r.score}: {r.score_breakdown})" for r in top15.itertuples()
    ])
    summary_path = os.path.join(BASE_DIR, "data", "screener_summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
    else:
        summary_df = pd.DataFrame(columns=["date", "time", "summary"])
    new_row = pd.DataFrame(
        [[datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M"), summary]],
        columns=["date", "time", "summary"],
    )
    summary_df = pd.concat([summary_df, new_row], ignore_index=True)
    write_csv_atomic(summary_path, summary_df)
    logger.info("Screener script finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Unhandled exception in Screener: %s", e, exc_info=True)
        sys.exit(1)
    else:
        logger.info("Screener completed successfully")
    finally:
        end_time = datetime.utcnow()
        elapsed_time = end_time - start_time
        logger.info("Script finished in %s", elapsed_time)
        sys.exit(0)
