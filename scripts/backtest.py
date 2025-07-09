# backtest.py - Updated with robust error handling and guaranteed CSV output
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import backtrader as bt
from dotenv import load_dotenv
import os
from logging.handlers import RotatingFileHandler
import logging
from datetime import datetime, timedelta, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, '.env')
log_path = os.path.join(BASE_DIR, 'logs', 'backtest.log')

logging.basicConfig(
    handlers=[RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)],
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

load_dotenv(dotenv_path)

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

# Initialize Alpaca clients
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Fetch historical data
def get_data(symbol, days=750):
    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date,
        limit=days
    )
    bars = data_client.get_stock_bars(request_params).df
    if bars.empty:
        return pd.DataFrame()

    bars = bars.reset_index()
    bars['datetime'] = pd.to_datetime(bars['timestamp'])
    bars.set_index('datetime', inplace=True)
    return bars[['open', 'high', 'low', 'close', 'volume']]

# JBravo Swing Trading Strategy
class JBravoStrategy(bt.Strategy):
    def __init__(self):
        self.sma9 = bt.ind.SMA(period=9)
        self.ema20 = bt.ind.EMA(period=20)
        self.sma180 = bt.ind.SMA(period=180)
        self.rsi = bt.ind.RSI(period=14)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.sma9[0] and self.sma9[0] > self.ema20[0] > self.sma180[0] and self.rsi[0] > 50:
                self.buy()
        else:
            if self.data.close[0] < self.ema20[0] or self.rsi[0] > 70:
                self.close()

# Run backtest for a single symbol
def run_backtest(symbol):
    try:
        data = get_data(symbol)
        if len(data) < 180:
            print(f"[WARN] {symbol}: insufficient bars ({len(data)}), skipping.")
            return None

        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=data))
        cerebro.addstrategy(JBravoStrategy)
        cerebro.broker.setcash(10000.0)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        result = cerebro.run()
        analysis = result[0].analyzers.trades.get_analysis()

        total_trades = analysis.total.closed if 'closed' in analysis.total else 0
        won_trades = analysis.won.total if 'won' in analysis and 'total' in analysis.won else 0
        lost_trades = analysis.lost.total if 'lost' in analysis and 'total' in analysis.lost else 0
        pnl_net = analysis.pnl.net.total if 'pnl' in analysis and 'net' in analysis.pnl else 0

        win_rate = (won_trades / total_trades) * 100 if total_trades else 0

        return {
            'symbol': symbol,
            'trades': total_trades,
            'wins': won_trades,
            'losses': lost_trades,
            'win_rate': win_rate,
            'net_pnl': pnl_net
        }

    except Exception as e:
        print(f"[ERROR] Exception during backtest for {symbol}: {e}")
        return None

# Run backtests on a list of symbols and save results
def backtest_symbols(symbols):
    results = []
    for symbol in symbols:
        print(f"[INFO] Backtesting {symbol}...")
        result = run_backtest(symbol)
        if result:
            results.append(result)

    # Ensure CSV is always written (empty if no results)
    results_df = pd.DataFrame(results, columns=['symbol', 'trades', 'wins', 'losses', 'win_rate', 'net_pnl'])
    results_df.sort_values(by='win_rate', ascending=False, inplace=True)
    csv_path = os.path.join(BASE_DIR, 'data', 'backtest_results.csv')
    results_df.to_csv(csv_path, index=False)
    logging.info("Backtesting complete. Results saved to %s", csv_path)

if __name__ == '__main__':
    try:
        csv_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
        symbols_df = pd.read_csv(csv_path)
        symbols = symbols_df.iloc[:, 0].tolist()
        backtest_symbols(symbols)
    except Exception as e:
        logging.error("Failed to execute backtesting pipeline: %s", e)

