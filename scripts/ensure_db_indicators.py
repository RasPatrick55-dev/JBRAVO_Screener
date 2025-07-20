import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'pipeline.db')

REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "score",
    "timestamp",
    "rsi",
    "macd",
    "macd_hist",
    "ema20",
    "sma9",
    "sma180",
    "atr",
    "adx",
    "aroon_up",
    "aroon_down",
    "win_rate",
    "net_pnl",
    "trades",
    "wins",
    "losses",
    "avg_return",
]

def ensure_columns(db_path: str = DB_PATH, indicators: list[str] = REQUIRED_COLUMNS) -> None:
    """Ensure required columns exist in the ``historical_candidates`` table."""

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create the table if it does not yet exist
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

    # Verify final schema contains all required columns
    cur.execute("PRAGMA table_info(historical_candidates);")
    final_cols = [row[1] for row in cur.fetchall()]
    missing = [c for c in indicators if c not in final_cols]
    if missing:
        raise RuntimeError(f"Schema synchronization failed, missing columns: {missing}")
    else:
        print("All required columns verified.")

    conn.close()


if __name__ == "__main__":
    ensure_columns()

