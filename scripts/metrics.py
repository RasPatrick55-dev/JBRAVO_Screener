# metrics.py (enhanced with comprehensive metrics)
import os
import pandas as pd
import logging
from utils import logger_utils
from utils import write_csv_atomic
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logger_utils.init_logging(__name__, "metrics.log")
start_time = datetime.utcnow()
logger.info("Script started")


# Load backtest results
def load_results(csv_file='backtest_results.csv'):
    csv_path = os.path.join(BASE_DIR, 'data', csv_file)
    return pd.read_csv(csv_path)

# Calculate additional performance metrics
def calculate_metrics(df):
    total_trades = df['trades'].sum() if 'trades' in df.columns else 0
    if 'trades' not in df.columns:
        logger.warning("Column 'trades' missing. Using 0 for total trades")
    total_wins = df['wins'].sum() if 'wins' in df.columns else 0
    if 'wins' not in df.columns:
        logger.warning("Column 'wins' missing. Using 0 for wins")
    total_losses = df['losses'].sum() if 'losses' in df.columns else 0
    if 'losses' not in df.columns:
        logger.warning("Column 'losses' missing. Using 0 for losses")
    total_pnl = df['net_pnl'].sum() if 'net_pnl' in df.columns else 0
    if 'net_pnl' not in df.columns:
        logger.warning("Column 'net_pnl' missing. Using 0 for net_pnl")

    win_rate = (total_wins / total_trades) * 100 if total_trades else 0
    avg_return_per_trade = df['net_pnl'].sum() / total_trades if total_trades and 'net_pnl' in df.columns else 0
    avg_win = df[df['net_pnl'] > 0]['net_pnl'].mean() if 'net_pnl' in df.columns else 0
    avg_loss = df[df['net_pnl'] < 0]['net_pnl'].mean() if 'net_pnl' in df.columns else 0

    metrics_summary = {
        'Total Trades': total_trades,
        'Total Wins': total_wins,
        'Total Losses': total_losses,
        'Win Rate (%)': win_rate,
        'Total Net PnL': total_pnl,
        'Average Return per Trade': avg_return_per_trade,
        'Average Win': avg_win,
        'Average Loss': avg_loss
    }

    return metrics_summary

# Scoring and ranking candidates based on performance metrics
def rank_candidates(df):
    weights = {
        'win_rate': 0.40,
        'net_pnl': 0.30,
        'trades': 0.20,
        'avg_return': 0.10
    }

    missing = [c for c in ['win_rate', 'net_pnl', 'trades'] if c not in df.columns]
    if missing:
        logger.warning("Missing columns for ranking: %s", missing)
        for col in missing:
            df[col] = 0

    df['avg_return'] = df['net_pnl'] / df['trades'].replace(0, 1) if 'net_pnl' in df.columns and 'trades' in df.columns else 0

    df['win_rate_norm'] = df['win_rate'] / df['win_rate'].max() if 'win_rate' in df.columns else 0
    df['net_pnl_norm'] = df['net_pnl'] / df['net_pnl'].max() if 'net_pnl' in df.columns else 0
    df['trades_norm'] = df['trades'] / df['trades'].max() if 'trades' in df.columns else 0
    df['avg_return_norm'] = df['avg_return'] / df['avg_return'].max() if 'avg_return' in df.columns else 0

    # Compute weighted score
    df['score'] = (
        df['win_rate_norm'] * weights['win_rate'] +
        df['net_pnl_norm'] * weights['net_pnl'] +
        df['trades_norm'] * weights['trades'] +
        df['avg_return_norm'] * weights['avg_return']
    )

    ranked_df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    return ranked_df

# Save top-ranked candidates
def save_top_candidates(df, top_n=15, output_file='top_candidates.csv'):
    required_cols = ['symbol', 'score', 'win_rate', 'net_pnl']
    csv_path = os.path.join(BASE_DIR, 'data', output_file)
    if all(col in df.columns for col in required_cols):
        cols = required_cols + [c for c in df.columns if c not in required_cols]
        top_candidates = df[cols].head(top_n)
        try:
            write_csv_atomic(csv_path, top_candidates)
            logger.info(
                "Successfully updated %s with %d records", csv_path, len(top_candidates)
            )
        except Exception as e:
            logger.error("Failed appending to %s: %s", csv_path, e)
    else:
        missing_cols = [col for col in required_cols if col not in df.columns]
        logger.error(f"Missing columns: {missing_cols}")

# Save overall metrics summary
def save_metrics_summary(metrics_summary, symbols, output_file='metrics_summary.csv'):
    summary_df = pd.DataFrame([metrics_summary])
    summary_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    summary_df['symbols'] = ';'.join(symbols)
    csv_path = os.path.join(BASE_DIR, 'data', output_file)
    try:
        write_csv_atomic(csv_path, summary_df)
        logger.info("Successfully appended data to %s", csv_path)
    except Exception as e:
        logger.error("Failed appending to %s: %s", csv_path, e)

# Full execution of metrics calculation, ranking, and summary
def main():
    results_df = load_results()

    # Detect missing symbol-level metrics and compute from trades_log.csv
    if "net_pnl" not in results_df.columns:
        trades_df = pd.read_csv(os.path.join(BASE_DIR, "data", "trades_log.csv"))
        grouped = trades_df.groupby("symbol")["net_pnl"]

        symbol_metrics = grouped.agg(
            trades="count",
            wins=lambda s: (s > 0).sum(),
            losses=lambda s: (s <= 0).sum(),
            net_pnl="sum",
        ).reset_index()
        symbol_metrics["win_rate"] = (
            symbol_metrics["wins"] / symbol_metrics["trades"] * 100
        )

        results_df = symbol_metrics

    ranked_df = rank_candidates(results_df)
    logger.info(
        "Screener Metrics Summary: total_candidates=%s, avg_score=%.2f",
        len(ranked_df),
        ranked_df['score'].mean(),
    )
    logger.info(
        "Top 15 Screener Symbols: %s",
        ", ".join(ranked_df['symbol'].head(15).tolist()),
    )
    save_top_candidates(ranked_df)
    logger.info(
        "Top Candidates: %s",
        ranked_df[['symbol', 'score', 'win_rate', 'net_pnl']].head(15).to_string(index=False)
    )

    metrics_summary = calculate_metrics(ranked_df)
    save_metrics_summary(metrics_summary, ranked_df['symbol'].tolist())
    logger.info(
        "Metrics summary: trades=%s win_rate=%.2f%% net_pnl=%.2f",
        metrics_summary['Total Trades'],
        metrics_summary['Win Rate (%)'],
        metrics_summary['Total Net PnL'],
    )

if __name__ == "__main__":
    logger.info("Starting metrics calculation")
    main()
    logger.info("Metrics calculation complete")
    end_time = datetime.utcnow()
    elapsed_time = end_time - start_time
    logger.info("Script finished in %s", elapsed_time)

