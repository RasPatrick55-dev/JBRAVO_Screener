# metrics.py (enhanced with comprehensive metrics)
import sys
import os
import json
import csv
from typing import Any, Mapping, Optional

# Ensure project root is on ``sys.path`` before third-party imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import logging
from datetime import datetime, timezone, date
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as PGConnection
from scripts import db
from utils import write_csv_atomic
from utils.screener_metrics import ensure_canonical_metrics, write_screener_metrics_json
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


def derive_prefix_counts_from_scored_candidates(base_dir: Path) -> dict:
    if not db.db_enabled():
        logging.warning("Prefix count skipped: DB disabled")
        return {}
    try:
        df, _ = db.fetch_latest_screener_candidates()
    except Exception as exc:
        logging.warning("Prefix count skipped: DB query failed: %s", exc)
        return {}
    if df.empty or "symbol" not in df.columns:
        logging.warning("Prefix count skipped: no symbols in DB results")
        return {}
    counts: Counter[str] = Counter()
    for symbol in df["symbol"].astype(str).tolist():
        symbol = symbol.strip()
        if symbol and symbol[0].isalpha():
            counts[symbol[0].upper()] += 1
    return dict(sorted(counts.items()))

start_time = datetime.utcnow()

# Columns expected in ``metrics_summary.csv``
REQUIRED_COLUMNS = [
    "total_trades",
    "net_pnl",
    "win_rate",
    "expectancy",
    "profit_factor",
    "max_drawdown",
    "sharpe",
    "sortino",
]

# Required columns expected in the trades log
required_columns = ["symbol", "net_pnl", "entry_time", "exit_time"]


CANON_TRADES_COLS = [
    "timestamp",
    "symbol",
    "action",
    "qty",
    "price",
    "order_id",
    "status",
]

_TRADES_OPTIONAL_COLUMNS = ["net_pnl", "entry_time", "exit_time"]

_TRADES_CANONICAL_COLUMNS = CANON_TRADES_COLS + _TRADES_OPTIONAL_COLUMNS
_TRADES_LOG_WARNED = False


def _summary_path() -> Path:
    return Path(BASE_DIR) / "data" / "metrics_summary.csv"


def _coerce_run_date(value: object) -> Optional[date]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(ts):  # type: ignore[arg-type]
        return None
    return ts.date()


def _fetch_dataframe(
    conn: PGConnection, query: str, params: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    with conn.cursor() as cursor:
        cursor.execute(query, params or {})
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _resolve_run_date(conn: Optional[PGConnection]) -> date:
    default = datetime.now(timezone.utc).date()
    if conn is None:
        return default

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT run_date FROM pipeline_runs ORDER BY run_date DESC LIMIT 1")
            row = cursor.fetchone()
            run_date = row[0] if row else None
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("[WARN] METRICS_RUN_DATE_LOOKUP_FAILED err=%s", exc)
        return default

    return _coerce_run_date(run_date) or default


def _load_trades_from_db(conn: Optional[PGConnection]) -> Optional[pd.DataFrame]:
    if conn is None:
        return None

    try:
        df = _fetch_dataframe(
            conn,
            """
            SELECT trade_id, symbol, qty, entry_price, exit_price, realized_pnl, entry_time, exit_time
            FROM trades
            WHERE status='CLOSED' AND exit_time IS NOT NULL AND entry_time IS NOT NULL
            ORDER BY exit_time DESC
            """,
        )
    except Exception as exc:
        logger.warning("[WARN] METRICS_DB_LOAD_FAILED err=%s", exc)
        return None

    if df.empty:
        return df

    renamed = df.rename(columns={"realized_pnl": "net_pnl"})
    renamed["symbol"] = renamed["symbol"].astype(str).str.upper()
    renamed["pnl"] = pd.to_numeric(renamed["net_pnl"], errors="coerce")
    for column in ("qty", "entry_price", "exit_price"):
        if column in renamed.columns:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    for column in ("entry_time", "exit_time"):
        if column in renamed.columns:
            renamed[column] = pd.to_datetime(renamed[column], errors="coerce", utc=True)

    return renamed


def load_trades_log(file_path: Path) -> pd.DataFrame:
    """Load ``trades_log.csv`` while tolerating missing or empty files."""

    global _TRADES_LOG_WARNED

    if not isinstance(file_path, Path):
        file_path = Path(str(file_path))

    canonical = list(_TRADES_CANONICAL_COLUMNS)

    if not file_path.exists():
        if not _TRADES_LOG_WARNED:
            logger.info("[INFO] METRICS trades_log_absent path=%s", file_path)
            _TRADES_LOG_WARNED = True
        return pd.DataFrame(columns=canonical)

    if file_path.stat().st_size == 0:
        if not _TRADES_LOG_WARNED:
            logger.warning("[WARN] METRICS_TRADES_LOG_MISSING path=%s", file_path)
            _TRADES_LOG_WARNED = True
        return pd.DataFrame(columns=canonical)

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        logger.error("Failed to read trades log %s: %s", file_path, exc)
        return pd.DataFrame(columns=canonical)

    for column in CANON_TRADES_COLS:
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")

    for column in _TRADES_OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.Series(dtype="object")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.warning("Trades log missing required metrics columns: %s", missing_cols)

    ordered = [col for col in canonical if col in df.columns]
    remainder = [col for col in df.columns if col not in ordered]
    return df[ordered + remainder]


def write_zero_metrics_summary(path: Path | str) -> None:
    target = Path(path)
    if not target.is_absolute():
        target = Path(BASE_DIR) / target
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {col: 0.0 for col in REQUIRED_COLUMNS}
    payload["total_trades"] = 0
    frame = pd.DataFrame([payload], columns=REQUIRED_COLUMNS)
    write_csv_atomic(str(target), frame)


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
def load_results(csv_file: str = "backtest_results.csv") -> pd.DataFrame:
    if not db.db_enabled():
        logger.warning("Backtest results skipped: DB disabled")
        return pd.DataFrame()
    df, _ = db.fetch_latest_backtest_results()
    return df

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
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    if "pnl" not in trades_df.columns:
        if "net_pnl" in trades_df.columns:
            trades_df = trades_df.rename(columns={"net_pnl": "pnl"})
        else:
            trades_df["pnl"] = 0.0
    trades_df["pnl"] = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
    if "qty" in trades_df.columns:
        trades_df["qty"] = pd.to_numeric(trades_df["qty"], errors="coerce").abs()
    if "entry_price" in trades_df.columns:
        trades_df["entry_price"] = pd.to_numeric(
            trades_df["entry_price"], errors="coerce"
        )
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

    if {"entry_price", "qty"}.issubset(trades_df.columns):
        exposure = (trades_df["entry_price"].abs() * trades_df["qty"].replace(0, np.nan))
        returns = trades_df["pnl"] / exposure
    else:
        returns = trades_df["pnl"]
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    sharpe = 0.0
    sortino = 0.0
    if not returns.empty:
        mean_return = returns.mean()
        std_return = returns.std(ddof=0)
        if std_return and not np.isnan(std_return):
            sharpe = float(np.sqrt(len(returns)) * mean_return / std_return)
        downside = returns[returns < 0]
        downside_std = downside.std(ddof=0)
        if downside_std and not np.isnan(downside_std):
            sortino = float(np.sqrt(len(returns)) * mean_return / downside_std)

    return {
        "total_trades": int(total_trades),
        "net_pnl": float(net_pnl),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
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

    new_nan_rows = pd.Series(False, index=df.index)
    for col in ("win_rate", "net_pnl", "trades"):
        if col not in df.columns:
            continue
        before_na = df[col].isna()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_na = df[col].isna()
        introduced = after_na & ~before_na
        if introduced.any():
            new_nan_rows |= introduced
            logger.warning(
                "Ranking numeric conversion introduced %d NaNs in %s",
                int(introduced.sum()),
                col,
            )

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

    if new_nan_rows.any():
        nan_scores = df['score'].isna()
        drop_mask = nan_scores & new_nan_rows
        if drop_mask.any():
            logger.warning(
                "Dropping %d rows with NaN scores after numeric conversion",
                int(drop_mask.sum()),
            )
            df = df.loc[~drop_mask].copy()

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
        logger.warning("Missing columns for top candidates CSV: %s", missing_cols)

# Save overall metrics summary
def save_metrics_summary(metrics_summary, symbols, output_file="metrics_summary.csv"):
    metrics_summary_df = pd.DataFrame(
        [[metrics_summary.get(col, 0) for col in REQUIRED_COLUMNS]],
        columns=REQUIRED_COLUMNS,
    )
    csv_path = Path(BASE_DIR) / "data" / output_file
    write_csv_atomic(str(csv_path), metrics_summary_df)
    logger.info("Metrics summary CSV successfully updated: %s", csv_path)

def _write_exit_reason_placeholder(base_dir: Path) -> None:
    path = base_dir / "data" / "exit_reason_summary.csv"
    frame = pd.DataFrame(columns=["exit_reason", "trades", "total_pnl", "avg_pnl"])
    write_csv_atomic(str(path), frame)


def main():
    if not db.db_enabled():
        logger.error("[ERROR] METRICS_DB_REQUIRED: DATABASE_URL/DB_* not configured.")
        return 2
    conn: PGConnection | None = db.get_db_conn()
    if conn is None:
        logger.error("[ERROR] METRICS_DB_REQUIRED: unable to connect to database.")
        return 2

    results_df = load_results()

    trades_df: pd.DataFrame
    db_trades = _load_trades_from_db(conn)
    if db_trades is None:
        logger.error("[ERROR] METRICS_DB_LOAD_FAILED: trades unavailable.")
        try:
            conn.close()
        except Exception:
            pass
        return 2
    trades_df = db_trades
    try:
        conn.close()
    except Exception:
        pass

    # Detect missing symbol-level metrics and compute from trades_log.csv
    if "net_pnl" not in results_df.columns:
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
    logger.info(
        "Top Candidates: %s",
        ranked_df[['symbol', 'score', 'win_rate', 'net_pnl']].head(15).to_string(index=False)
    )
    save_top_candidates(ranked_df)

    if not trades_df.empty:
        trades_df = validate_numeric(trades_df, "net_pnl")

    summary_metrics = calculate_metrics(trades_df.copy())
    if trades_df.empty:
        logger.warning("Trades dataset empty; writing default metrics row.")
    metrics_summary = pd.DataFrame(
        [[summary_metrics.get(col, 0) for col in REQUIRED_COLUMNS]],
        columns=REQUIRED_COLUMNS,
    )

    symbol_series = ranked_df["symbol"] if "symbol" in ranked_df.columns else pd.Series(dtype="object")
    save_metrics_summary(summary_metrics, symbol_series.tolist())

    if not trades_df.empty:
        logger.info(
            "Calculated Metrics: Trades=%s, Net PnL=%.2f, Win Rate=%.2f%%, "
            "Expectancy=%.2f, Profit Factor=%s, Max Drawdown=%.2f, Sharpe=%.2f, Sortino=%.2f",
            summary_metrics.get("total_trades", 0),
            summary_metrics.get("net_pnl", 0.0),
            summary_metrics.get("win_rate", 0.0),
            summary_metrics.get("expectancy", 0.0),
            summary_metrics.get("profit_factor", 0.0),
            summary_metrics.get("max_drawdown", 0.0),
            summary_metrics.get("sharpe", 0.0),
            summary_metrics.get("sortino", 0.0),
        )

        if "exit_reason" in trades_df.columns:
            logger.info("Exit reason breakdown skipped: DB is source of truth.")

    run_date_conn = db.get_db_conn()
    try:
        run_date = _resolve_run_date(run_date_conn)
    finally:
        if run_date_conn is not None:
            try:
                run_date_conn.close()
            except Exception:
                pass

    if db.db_enabled():
        try:
            if db.upsert_top_candidates(run_date, ranked_df):
                logger.info(
                    "[INFO] METRICS_DB_TOP_CANDIDATES_UPSERT rows=%s run_date=%s",
                    len(ranked_df.index),
                    run_date,
                )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("[WARN] DB_WRITE_FAILED table=top_candidates err=%s", exc)

    try:
        db.upsert_metrics_daily(run_date, summary_metrics)
        logger.info(
            "[INFO] METRICS_DB_OK run_date=%s total_trades=%s net_pnl=%s",
            run_date,
            summary_metrics.get("total_trades"),
            summary_metrics.get("net_pnl"),
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("[WARN] DB_WRITE_FAILED table=metrics_daily err=%s", exc)

    metrics_path = Path(BASE_DIR) / "data" / "screener_metrics.json"
    metrics: dict = {}
    if metrics_path.exists():
        try:
            candidate = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                metrics.update(candidate)
            else:
                logger.warning(
                    "Existing screener_metrics.json is not a dict; resetting prefix counts context."
                )
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.warning("Failed to read %s: %s", metrics_path, exc)

    if not metrics.get("universe_prefix_counts"):
        metrics["universe_prefix_counts"] = derive_prefix_counts_from_scored_candidates(
            Path(BASE_DIR)
        )

    if "universe_prefix_counts" not in metrics:
        metrics["universe_prefix_counts"] = {}

    try:
        metrics = ensure_canonical_metrics(metrics)
        write_screener_metrics_json(metrics_path, metrics)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        logger.warning("Failed to update %s: %s", metrics_path, exc)

    return 0

if __name__ == "__main__":
    logger.info("Starting metrics calculation")
    exit_code = main()
    logger.info("Metrics calculation complete")
    end_time = datetime.utcnow()
    elapsed_time = end_time - start_time
    logger.info("Script finished in %s", elapsed_time)
    sys.exit(exit_code)
