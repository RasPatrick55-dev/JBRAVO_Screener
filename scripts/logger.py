# logger.py
import sqlite3
import csv

class TradeLogger:
    def __init__(self, db_path='trades.db', csv_path='trades_log.csv'):
        self.db_path = db_path
        self.csv_path = csv_path
        conn = sqlite3.connect(self.db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS trades (
                            trade_id TEXT PRIMARY KEY,
                            symbol TEXT,
                            entry_price REAL,
                            exit_price REAL,
                            quantity INTEGER,
                            entry_time TEXT,
                            exit_time TEXT,
                            pnl REAL,
                            exit_reason TEXT
                        );''')
        conn.close()

    def log_trade(self, symbol, entry_price, exit_price, quantity, entry_time, exit_time, exit_reason):
        pnl = (exit_price - entry_price) * quantity
        trade_id = f"{symbol}_{entry_time}"

        # SQLite
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?)",
                     (trade_id, symbol, entry_price, exit_price, quantity, entry_time, exit_time, pnl, exit_reason))
        conn.commit()
        conn.close()

        # CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([trade_id, symbol, entry_price, exit_price, quantity, entry_time, exit_time, pnl, exit_reason])
