# metrics.py (enhanced with comprehensive metrics)
import sys
import os

# Ensure project root is on ``sys.path`` before third-party imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from utils import write_csv_atomic
from utils.env import load_env

load_env()

logfile = os.path.join(BASE_DIR, "logs", "metrics.log")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info("Metrics script started")

start_time = datetime.utcnow()

# Columns expected in ``metrics_summary.csv``
REQUIRED_COLUMNS = [
    "total_trades",
    "net_pnl",
    "win_rate",
    "expectancy",
    "profit_factor",
    "max_drawdown",
]

# Required columns expected in the trades log
required_columns = ["symbol", "net_pnl", "entry_time", "exit_time"]


_TRADES_CANONICAL_COLUMNS = [
    "timestamp",
    "symbol",
    "action",
    "qty",
    "price",
    "order_id",
    "status",
    "net_pnl",
    "entry_time",
    "exit_time",
]


def load_trades_log(file_path: Path) -> pd.DataFrame:
    """Load ``trades_log.csv`` while tolerating missing or empty files."""

    canonical = list(_TRADES_CANONICAL_COLUMNS)
    if not isinstance(file_path, Path):
        file_path = Path(str(file_path))

    if not file_path.exists() or file_path.stat().st_size == 0:
        logger.warning("Trades log missing/empty; using empty DataFrame at %s", file_path)
        return pd.DataFrame(columns=canonical)

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        logger.error("Failed to read trades log %s: %s", file_path, exc)
        return pd.DataFrame(columns=canonical)

    for column in canonical:
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.warning("Trades log missing required metrics columns: %s", missing_cols)

    return df[canonical]


def validate_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Ensure a column contains numeric data and drop invalid rows."""
    df[column] = pd.to_numeric(df[column], errors="coerce")
    invalid_count = df[column].isna().sum()
    if invalid_count:
        logger.warning(
            "%d rows dropped due to non-numeric %s values", invalid_count, column
        )
        df = df.dropna(subset=[column])
    return df


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
    trades_df["pnl"] = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
    for col in ("entry_time", "exit_time"):
        if col in trades_df.columns:
            trades_df[col] = pd.to_datetime(trades_df[col], errors="coerce")

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
        [[metrics_summary.get(col, 0) for col in REQUIRED_COLUMNS]],
        columns=REQUIRED_COLUMNS,
    )
    csv_path = os.path.join(BASE_DIR, "data", output_file)
    metrics_summary_df.to_csv(csv_path, index=False)
    logger.info(f"Successfully updated metrics_summary.csv: {csv_path}")

# Full execution of metrics calculation, ranking, and summary
def main():
    try:
        results_df = load_results()
    except Exception as e:
        logger.error(f"Error encountered in load_results: {e}")
        raise

    # Detect missing symbol-level metrics and compute from trades_log.csv
    if "net_pnl" not in results_df.columns:
        trades_path = Path(BASE_DIR) / "data" / "trades_log.csv"
        trades_df = load_trades_log(trades_path)
        if trades_df.empty:
            results_df = pd.DataFrame(
                columns=["symbol", "trades", "wins", "losses", "net_pnl", "win_rate"]
            )
        else:
            trades_df = validate_numeric(trades_df, "net_pnl")
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

    trade_log_path = Path(BASE_DIR) / "data" / "trades_log.csv"
    metrics_summary_file = Path(BASE_DIR) / "data" / "metrics_summary.csv"

    trades_df = load_trades_log(trade_log_path)

    if trades_df.empty:
        logger.warning("Trades log is empty. Writing default metrics.")
        metrics_summary = pd.DataFrame([
            {
                "total_trades": 0,
                "net_pnl": 0.0,
                "win_rate": 0.0,
                "expectancy": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
            }
        ])
    else:
        trades_df = validate_numeric(trades_df, "net_pnl")

        total_trades = len(trades_df)
        net_pnl = trades_df["net_pnl"].sum()

        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] < 0]

        win_rate = (len(wins) / total_trades * 100) if total_trades else 0.0
        expectancy = (net_pnl / total_trades) if total_trades else 0.0

        try:
            profit_factor = (
                wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())
                if not losses.empty
                else np.inf
            )
        except ZeroDivisionError:
            profit_factor = np.inf

        max_drawdown = trades_df["net_pnl"].cumsum().min()

        metrics_summary = pd.DataFrame([
            {
                "total_trades": total_trades,
                "net_pnl": net_pnl,
                "win_rate": win_rate,
                "expectancy": expectancy,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
            }
        ])

        logger.info(
            "Calculated Metrics: Trades=%s, Net PnL=%.2f, Win Rate=%.2f%%, "
            "Expectancy=%.2f, Profit Factor=%s, Max Drawdown=%.2f",
            total_trades,
            net_pnl,
            win_rate,
            expectancy,
            profit_factor,
            max_drawdown,
        )

    try:
        metrics_summary.to_csv(metrics_summary_file, index=False)
        logger.info(
            f"Metrics summary CSV successfully updated: {metrics_summary_file}"
        )
    except Exception as e:
        logger.error(f"Failed to write metrics_summary.csv: {e}")

if __name__ == "__main__":
    logger.info("Starting metrics calculation")
    main()
    logger.info("Metrics calculation complete")
    end_time = datetime.utcnow()
    elapsed_time = end_time - start_time
    logger.info("Script finished in %s", elapsed_time)

