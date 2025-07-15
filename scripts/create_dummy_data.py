# create_dummy_data.py
import os
import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generate dummy data
def generate_dummy_trades(num_trades=20):
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 'AMD', 'INTC']
    trades = []
    now = datetime.now()
    for i in range(num_trades):
        symbol = random.choice(symbols)
        entry_price = round(random.uniform(100, 500), 2)
        exit_price = entry_price + round(random.uniform(-20, 50), 2)
        quantity = random.randint(10, 200)
        entry_time = now - timedelta(days=random.randint(1, 10))
        exit_time = entry_time + timedelta(days=random.randint(1, 7))
        pnl = (exit_price - entry_price) * quantity
        exit_reason = random.choice(['Trailing Stop', 'Max Hold Time', 'Manual Close'])

        trades.append({
            'trade_id': f"{symbol}_{entry_time.strftime('%Y%m%d%H%M%S')}",
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pnl': pnl,
            'exit_reason': exit_reason
        })
    return trades

# Populate SQLite database and CSVs
def populate_dummy_data():
    trades = generate_dummy_trades()

    # Populate SQLite
    db_path = os.path.join(BASE_DIR, 'data', 'trades.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
                 trade_id TEXT PRIMARY KEY, symbol TEXT,
                 entry_price REAL, exit_price REAL,
                 quantity INTEGER, entry_time TEXT, 
                 exit_time TEXT, pnl REAL, exit_reason TEXT)''')

    for trade in trades:
        c.execute("INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?)", (
            trade['trade_id'], trade['symbol'], trade['entry_price'], trade['exit_price'],
            trade['quantity'], trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
            trade['pnl'], trade['exit_reason']
        ))
    conn.commit()
    conn.close()

    # Populate trades_log.csv using standardized columns
    normalized = []
    for t in trades:
        normalized.append(
            {
                'id': t['trade_id'],
                'symbol': t['symbol'],
                'side': 'sell',
                'filled_qty': t['quantity'],
                'entry_price': t['entry_price'],
                'exit_price': t['exit_price'],
                'entry_time': t['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': t['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'order_status': 'Filled',
                'pnl': t['pnl'],
            }
        )
    df_trades_log = pd.DataFrame(normalized)
    trades_log_path = os.path.join(BASE_DIR, 'data', 'trades_log.csv')
    df_trades_log.to_csv(trades_log_path, index=False)

    # Populate top_candidates.csv
    top_candidates = [{'symbol': sym, 'score': round(random.uniform(50, 100), 2)} for sym in random.sample(['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META'], 5)]
    df_top_candidates = pd.DataFrame(top_candidates)
    top_candidates_path = os.path.join(BASE_DIR, 'data', 'top_candidates.csv')
    df_top_candidates.to_csv(top_candidates_path, index=False)

    # Populate metrics_summary.csv
    metrics_summary = {
        'Total Trades': len(trades),
        'Total Wins': len([t for t in trades if t['pnl'] > 0]),
        'Total Losses': len([t for t in trades if t['pnl'] <= 0]),
        'Win Rate (%)': round(100 * len([t for t in trades if t['pnl'] > 0]) / len(trades), 2),
        'Total Net PnL': round(sum([t['pnl'] for t in trades]), 2),
        'Average Return per Trade': round(sum([t['pnl'] for t in trades])/len(trades), 2)
    }
    df_metrics_summary = pd.DataFrame([metrics_summary])
    metrics_summary_path = os.path.join(BASE_DIR, 'data', 'metrics_summary.csv')
    df_metrics_summary.to_csv(metrics_summary_path, index=False)

if __name__ == "__main__":
    populate_dummy_data()
    print("[INFO] Dummy data created successfully.")
