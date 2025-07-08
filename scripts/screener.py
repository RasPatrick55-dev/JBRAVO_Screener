# screener.py with debugging and robust scoring
import os
import logging
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load environment variables
env_path = os.environ.get("BRAVO_ENV_PATH")
if not env_path:
    cwd_env = os.path.join(os.getcwd(), ".env")
    default_env = os.path.expanduser("~/jbravo_screener/.env")
    env_path = cwd_env if os.path.exists(cwd_env) else default_env

logging.info("Loading environment variables from %s", env_path)
load_dotenv(env_path)

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    logging.error("Missing API credentials. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the .env file.")
    raise SystemExit(1)

# Initialize Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Fetch all tradable symbols
assets = trading_client.get_all_assets()
symbols = [a.symbol for a in assets if a.tradable and a.status == "active" and a.exchange in ("NYSE", "NASDAQ")]

ranked_candidates = []

# Screening and ranking criteria
for symbol in symbols:
    logging.info("Processing %s...", symbol)
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=(datetime.now(timezone.utc) - timedelta(days=750)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=(datetime.now(timezone.utc) - timedelta(minutes=16)).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        bars = data_client.get_stock_bars(request_params).df

        if len(bars) < 200:
            logging.warning("Skipping %s: insufficient data (%d bars)", symbol, len(bars))
            continue

        df = bars.copy().sort_index()
        df['sma9'] = df['close'].rolling(9, min_periods=1).mean()
        df['ema20'] = df['close'].ewm(span=20, min_periods=1).mean()
        df['sma180'] = df['close'].rolling(180, min_periods=1).mean()

        last = df.iloc[-1]
        prior = df.iloc[-2]

        if (
            last['close'] > last['sma9'] and
            prior['close'] < prior['sma9'] and
            last['sma9'] > last['ema20'] and
            last['ema20'] > last['sma180']
        ):
            # Robust ranking score calculation
            momentum_score = (last['close'] - last['sma9']) / last['sma9']
            alignment_score = (last['sma9'] - last['ema20']) / last['ema20']
            long_term_trend_score = (last['ema20'] - last['sma180']) / last['sma180']
            total_score = (0.5 * momentum_score) + (0.3 * alignment_score) + (0.2 * long_term_trend_score)

            ranked_candidates.append({
                'symbol': symbol,
                'momentum_score': momentum_score,
                'alignment_score': alignment_score,
                'long_term_trend_score': long_term_trend_score,
                'total_score': total_score
            })

    except Exception as e:
        logging.error("%s failed: %s", symbol, e)

# Convert to DataFrame and rank
ranked_df = pd.DataFrame(ranked_candidates)
ranked_df.sort_values(by="total_score", ascending=False, inplace=True)

if ranked_df.empty:
    logging.warning("No candidates met the screening criteria. CSV file not created.")
else:
    ranked_df.head(15).to_csv("top_candidates.csv", index=False)
    logging.info("Top 15 ranked candidates saved to top_candidates.csv")