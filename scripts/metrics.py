# metrics.py (enhanced with comprehensive metrics)
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from utils.logger_utils import init_logging  # Fix import by adjusting the Python path correctly

logger = init_logging(__name__, "metrics.log")
logger.info("Metrics script started.")

import pandas as pd
import logging
from utils import write_csv_atomic
from datetime import datetime

start_time = datetime.utcnow()


# Load backtest results
def load_results(csv_file='backtest_results.csv'):
    csv_path = os.path.join(BASE_DIR, 'data', csv_file)
    return pd.read_csv(csv_path)

# Calculate additional performance metrics
def calculate_metrics(trades_df: pd.DataFrame) -> dict:
    """Return overall trading metrics from ``trades_df`` using dashboard schema."""

    if trades_df.empty:
        return {
            "total_trades": 0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    if "pnl" not in trades_df.columns:
        if "net_pnl" in trades_df.columns:
            trades_df = trades_df.rename(columns={"net_pnl": "pnl"})
        else:
            trades_df["pnl"] = 0.0

    total_trades = len(trades_df)
    net_pnl = trades_df["pnl"].sum()
    win_rate = (trades_df["pnl"] > 0).mean() * 100
    expectancy = trades_df["pnl"].mean()
    profits = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    losses = trades_df[trades_df["pnl"] < 0]["pnl"].sum()
    profit_factor = profits / abs(losses) if losses != 0 else float("inf")
    cumulative = trades_df["pnl"].cumsum()
    max_drawdown = (cumulative - cumulative.cummax()).min() if not cumulative.empty else 0.0

    return {
        "total_trades": int(total_trades),
        "net_pnl": float(net_pnl),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
    }

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
def save_metrics_summary(metrics_summary, symbols, output_file="metrics_summary.csv"):
    metrics_summary_df = pd.DataFrame(
        {
            "total_trades": [metrics_summary["total_trades"]],
            "net_pnl": [metrics_summary["net_pnl"]],
            "win_rate": [metrics_summary["win_rate"]],
            "expectancy": [metrics_summary["expectancy"]],
            "profit_factor": [metrics_summary["profit_factor"]],
            "max_drawdown": [metrics_summary["max_drawdown"]],
        }
    )
    csv_path = os.path.join(BASE_DIR, "data", output_file)
    metrics_summary_df.to_csv(csv_path, index=False)
    logger.info(f"Successfully updated metrics_summary.csv: {csv_path}")

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

    trades_log_path = os.path.join(BASE_DIR, "data", "trades_log.csv")
    trades_df = pd.read_csv(trades_log_path) if os.path.exists(trades_log_path) else pd.DataFrame()
    metrics_summary = calculate_metrics(trades_df)
    save_metrics_summary(metrics_summary, ranked_df['symbol'].tolist())
    logger.info(
        "Metrics summary: trades=%s win_rate=%.2f%% net_pnl=%.2f",
        metrics_summary['total_trades'],
        metrics_summary['win_rate'],
        metrics_summary['net_pnl'],
    )

if __name__ == "__main__":
    logger.info("Starting metrics calculation")
    main()
    logger.info("Metrics calculation complete")
    end_time = datetime.utcnow()
    elapsed_time = end_time - start_time
    logger.info("Script finished in %s", elapsed_time)

