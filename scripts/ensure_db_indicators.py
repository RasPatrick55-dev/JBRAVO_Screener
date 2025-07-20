import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'pipeline.db')

REQUIRED_COLUMNS = [
    'date',
    'symbol',
    'score',
    'timestamp',
    'rsi',
    'macd',
    'ema20',
    'sma9',
    'sma180',
    'macd_hist',
    'adx',
    'aroon_up',
    'aroon_down',
    'win_rate',
    'net_pnl',
    'trades',
    'wins',
    'losses',
    'avg_return',
]

def ensure_columns(db_path: str = DB_PATH, indicators: list[str] = REQUIRED_COLUMNS) -> None:
    """Ensure required columns exist in the historical_candidates table."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS historical_candidates (date TEXT, symbol TEXT, score REAL)"
    )
    cur.execute("PRAGMA table_info(historical_candidates);")
    existing = [row[1] for row in cur.fetchall()]

    for col in indicators:
        if col not in existing:
            cur.execute(f"ALTER TABLE historical_candidates ADD COLUMN {col} REAL;")
            print(f"\u2714 Added column: {col}")
        else:
            print(f"\u23E9 Column already exists: {col}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    ensure_columns()

