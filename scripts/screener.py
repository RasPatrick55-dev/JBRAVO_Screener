# screener.py with debugging and robust scoring
import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

log_path = os.path.join(BASE_DIR, 'logs', 'screener.log')
error_log_path = os.path.join(BASE_DIR, 'logs', 'error.log')

error_handler = RotatingFileHandler(error_log_path, maxBytes=5_000_000, backupCount=5)
error_handler.setLevel(logging.ERROR)

logging.basicConfig(
    handlers=[
        RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5),
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
    pd.DataFrame(
        columns=['symbol', 'momentum_score', 'alignment_score', 'long_term_trend_score', 'total_score']
    ).to_csv(top_init_path, index=False)

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

csv_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
if ranked_df.empty:
    logging.warning("No candidates met the screening criteria.")
    ranked_df = pd.DataFrame(columns=[
        'symbol', 'momentum_score', 'alignment_score',
        'long_term_trend_score', 'total_score'
    ])

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
