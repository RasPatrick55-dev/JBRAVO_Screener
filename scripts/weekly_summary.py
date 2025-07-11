# weekly_summary.py - Generate weekly trade summary from CSVs and logs

import os
import re
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import shutil
from tempfile import NamedTemporaryFile
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("weekly_summary")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "weekly_summary.log"), maxBytes=2_000_000, backupCount=5
)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def write_csv_atomic(df: pd.DataFrame, dest: str) -> None:
    """Write DataFrame to ``dest`` atomically."""
    tmp = NamedTemporaryFile("w", delete=False, dir=os.path.dirname(dest), newline="")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    shutil.move(tmp.name, dest)


def load_csv(filename: str) -> pd.DataFrame:
    """Load CSV from the data directory with logging."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        logger.warning("CSV file missing: %s", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        logger.info("Loaded %s (%d rows)", filename, len(df))
        return df
    except Exception as exc:
        logger.error("Failed reading %s: %s", filename, exc)
        return pd.DataFrame()


def parse_execution_log(log_name: str = "execute_trades.log") -> dict:
    """Parse ``execute_trades.log`` and return weekly stats."""
    path = os.path.join(LOG_DIR, log_name)
    stats = {"orders_placed": 0, "orders_filled": 0, "orders_rejected": 0, "errors": 0}
    if not os.path.exists(path):
        logger.warning("Execution log not found: %s", path)
        return stats

    one_week_ago = datetime.utcnow() - timedelta(days=7)
    ts_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})\s(\d{2}:\d{2}:\d{2})")
    with open(path, "r") as fh:
        for line in fh:
            match = ts_pattern.match(line)
            if not match:
                continue
            try:
                dt = datetime.strptime(
                    f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H:%M:%S"
                )
            except Exception:
                continue
            if dt < one_week_ago:
                continue
            lower = line.lower()
            if "submitting" in lower and "order" in lower:
                stats["orders_placed"] += 1
            if "order filled" in lower:
                stats["orders_filled"] += 1
            if "rejected" in lower:
                stats["orders_rejected"] += 1
            if "error" in lower:
                stats["errors"] += 1
    logger.info("Execution log stats: %s", stats)
    return stats


def calculate_weekly_summary() -> dict:
    """Compute weekly trading metrics from CSV files."""
    today = datetime.utcnow().date()
    one_week_ago = today - timedelta(days=7)

    executed_trades = load_csv("executed_trades.csv")
    trades_log = load_csv("trades_log.csv")
    open_positions = load_csv("open_positions.csv")
    metrics_summary = load_csv("metrics_summary.csv")
    backtest_results = load_csv("backtest_results.csv")
    historical_candidates = load_csv("historical_candidates.csv")
    top_candidates = load_csv("top_candidates.csv")

    # Filter executed trades for the week
    if "entry_time" in executed_trades.columns:
        executed_trades["entry_time"] = pd.to_datetime(executed_trades["entry_time"])
        weekly_exec = executed_trades[executed_trades["entry_time"] >= pd.Timestamp(one_week_ago)]
    else:
        weekly_exec = pd.DataFrame()
    total_trades = len(weekly_exec)

    # Closed trades for the week
    if not trades_log.empty:
        trades_log["exit_time"] = pd.to_datetime(trades_log["exit_time"])
        closed_week = trades_log[trades_log["exit_time"] >= pd.Timestamp(one_week_ago)]
    else:
        closed_week = pd.DataFrame()

    wins = (closed_week["pnl"] > 0).sum() if not closed_week.empty else 0
    win_rate = (wins / len(closed_week) * 100) if len(closed_week) else 0
    realized_pnl = closed_week["pnl"].sum() if not closed_week.empty else 0

    unrealized_pnl = (
        open_positions["unrealized_pl"].sum() if "unrealized_pl" in open_positions.columns else 0
    )

    top_symbols = (
        top_candidates["symbol"].head(3).tolist() if "symbol" in top_candidates.columns else []
    )

    best_trade_symbol = ""
    if not closed_week.empty:
        best_idx = closed_week["pnl"].idxmax()
        if pd.notna(best_idx):
            best_trade_symbol = closed_week.loc[best_idx, "symbol"]

    best_backtest_symbol = ""
    if not backtest_results.empty:
        best_backtest_symbol = (
            backtest_results.sort_values("net_pnl", ascending=False)["symbol"].iloc[0]
        )

    best_candidate = ""
    if not historical_candidates.empty:
        historical_candidates["date"] = pd.to_datetime(historical_candidates["date"])
        recent_candidates = historical_candidates[
            historical_candidates["date"] >= pd.Timestamp(one_week_ago)
        ]
        if not recent_candidates.empty:
            best_candidate = recent_candidates.sort_values("score", ascending=False)["symbol"].iloc[0]

    metrics_row = metrics_summary.iloc[0] if not metrics_summary.empty else pd.Series()

    summary = {
        "Week Start": one_week_ago.isoformat(),
        "Week End": today.isoformat(),
        "Total Trades": int(total_trades),
        "Win Rate (%)": round(win_rate, 2),
        "Realized PnL": round(realized_pnl, 2),
        "Unrealized PnL": round(unrealized_pnl, 2),
        "Best Trade Symbol": best_trade_symbol,
        "Top Candidates": ";".join(top_symbols),
        "Best Backtest Symbol": best_backtest_symbol,
        "Best Candidate": best_candidate,
    }

    if not metrics_row.empty:
        summary.update(
            {
                "Overall Win Rate (%)": metrics_row.get("Win Rate (%)", 0),
                "Overall Net PnL": metrics_row.get("Total Net PnL", 0),
            }
        )

    return summary


def save_weekly_summary(summary: dict, filename: str = "weekly_summary.csv") -> None:
    """Write ``summary`` to a CSV in the data directory."""
    dest = os.path.join(DATA_DIR, filename)
    df = pd.DataFrame([summary])
    write_csv_atomic(df, dest)
    logger.info("Weekly summary saved to %s", dest)


if __name__ == "__main__":
    logger.info("Generating weekly trade summary")
    log_stats = parse_execution_log()
    summary = calculate_weekly_summary()
    summary.update(log_stats)
    save_weekly_summary(summary)
    logger.info("Weekly trade summary generation complete")
