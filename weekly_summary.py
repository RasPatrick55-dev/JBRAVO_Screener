# weekly_summary.py - Weekly trade performance summary

import pandas as pd
import sqlite3
from datetime import datetime, timedelta

# Connect to SQLite database and load trades
def load_trades(db_path='trades.db'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    return df

# Calculate weekly performance summary
def calculate_weekly_summary(trades_df):
    today = datetime.utcnow().date()
    one_week_ago = today - timedelta(days=7)

    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.date
    weekly_trades = trades_df[trades_df['exit_time'] >= one_week_ago]

    total_trades = len(weekly_trades)
    wins = len(weekly_trades[weekly_trades['pnl'] > 0])
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100 if total_trades else 0
    total_pnl = weekly_trades['pnl'].sum()
    avg_trade_return = weekly_trades['pnl'].mean()
    best_trade = weekly_trades['pnl'].max()
    worst_trade = weekly_trades['pnl'].min()

    summary = {
        'Total Trades': total_trades,
        'Wins': wins,
        'Losses': losses,
        'Win Rate (%)': round(win_rate, 2),
        'Total Net PnL': round(total_pnl, 2),
        'Average Return per Trade': round(avg_trade_return, 2),
        'Best Trade': round(best_trade, 2),
        'Worst Trade': round(worst_trade, 2)
    }

    return summary

# Save weekly summary to CSV
def save_weekly_summary(summary, output_file='weekly_summary.csv'):
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(output_file, index=False)
    print(f"[INFO] Weekly summary saved to {output_file}")

if __name__ == '__main__':
    print("[INFO] Generating weekly trade performance summary...")
    trades_df = load_trades()
    summary = calculate_weekly_summary(trades_df)
    save_weekly_summary(summary)
    print("[INFO] Weekly trade performance summary generated successfully.")