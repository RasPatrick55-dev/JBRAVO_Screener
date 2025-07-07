# metrics.py (enhanced with comprehensive metrics)
import pandas as pd

# Load backtest results
def load_results(csv_file='backtest_results.csv'):
    return pd.read_csv(csv_file)

# Calculate additional performance metrics
def calculate_metrics(df):
    total_trades = df['trades'].sum()
    total_wins = df['wins'].sum()
    total_losses = df['losses'].sum()
    total_pnl = df['net_pnl'].sum()

    win_rate = (total_wins / total_trades) * 100 if total_trades else 0
    avg_return_per_trade = df['net_pnl'].sum() / total_trades if total_trades else 0
    avg_win = df[df['net_pnl'] > 0]['net_pnl'].mean()
    avg_loss = df[df['net_pnl'] < 0]['net_pnl'].mean()

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

    # Calculate additional metrics
    df['avg_return'] = df['net_pnl'] / df['trades'].replace(0, 1)

    # Normalize metrics
    df['win_rate_norm'] = df['win_rate'] / df['win_rate'].max()
    df['net_pnl_norm'] = df['net_pnl'] / df['net_pnl'].max()
    df['trades_norm'] = df['trades'] / df['trades'].max()
    df['avg_return_norm'] = df['avg_return'] / df['avg_return'].max()

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
    top_candidates = df.head(top_n)
    top_candidates.to_csv(output_file, index=False)
    print(f"[INFO] Top {top_n} candidates saved to {output_file}")

# Save overall metrics summary
def save_metrics_summary(metrics_summary, output_file='metrics_summary.csv'):
    summary_df = pd.DataFrame([metrics_summary])
    summary_df.to_csv(output_file, index=False)
    print(f"[INFO] Metrics summary saved to {output_file}")

# Full execution of metrics calculation, ranking, and summary
def main():
    results_df = load_results()
    ranked_df = rank_candidates(results_df)
    save_top_candidates(ranked_df)

    metrics_summary = calculate_metrics(ranked_df)
    save_metrics_summary(metrics_summary)

if __name__ == "__main__":
    main()
