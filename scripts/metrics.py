print(">>> METRICS FILE EXECUTED <<<")
# metrics.py (enhanced with comprehensive metrics)
import sys
import os
import json
import argparse
import logging
from typing import Any, Mapping, Optional, Iterable

# Ensure project root is on ``sys.path`` before third-party imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from datetime import datetime, timezone, date
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from psycopg2.extensions import connection as PGConnection
from scripts import db
from scripts.db_queries import get_latest_screener_candidates
from utils.screener_metrics import ensure_canonical_metrics, write_screener_metrics_json
from utils.env import load_env

load_env()

logger = logging.getLogger("metrics")
logging.basicConfig(level=logging.INFO)

logfile = os.path.join(BASE_DIR, "logs", "metrics.log")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    force=True,
)
logger.info("Metrics script started")


def derive_prefix_counts_from_scored_candidates(base_dir: Path) -> dict:
    if not db.db_enabled():
        logging.warning("Prefix count skipped: DB disabled")
        return {}
    try:
        run_date = db.fetch_latest_run_date("screener_candidates")
        df, _ = (
            get_latest_screener_candidates(run_date)
            if run_date is not None
            else (pd.DataFrame(), None)
        )
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


def _load_trades_from_db(conn: Optional[PGConnection]) -> pd.DataFrame:
    # No CSV/unfiltered fallback: exact selected population is part of publication.
    with db._policy_read_transaction(conn, scope='VALIDATED_PERFORMANCE') as policy:
        records = db.performance_input(conn, policy)
        evidence = db.performance_evidence(policy, records)
    df = pd.DataFrame(records, columns=['trade_id', 'symbol', 'qty', 'entry_price',
        'exit_price', 'realized_pnl', 'entry_time', 'exit_time'])
    renamed = df.rename(columns={'realized_pnl': 'net_pnl'})
    renamed['pnl'] = pd.to_numeric(renamed['net_pnl'], errors='raise')
    for column in ('qty', 'entry_price', 'exit_price'):
        renamed[column] = pd.to_numeric(renamed[column], errors='raise')
    for column in ('entry_time', 'exit_time'):
        renamed[column] = pd.to_datetime(renamed[column], errors='raise', utc=True)
    renamed.attrs['trade_policy'] = evidence
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


def validate_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Ensure a column contains numeric data and drop invalid rows."""
    df[column] = pd.to_numeric(df[column], errors="coerce")
    invalid_count = df[column].isna().sum()
    if invalid_count:
        logger.warning("%d rows dropped due to non-numeric %s values", invalid_count, column)
        df = df.dropna(subset=[column])
    return df


# Load backtest results
def load_results(
    csv_file: str = "backtest_results.csv",
    run_date: Optional[date] = None,
) -> pd.DataFrame:
    _ = csv_file
    if not db.db_enabled():
        logger.warning("Backtest results skipped: DB disabled")
        return pd.DataFrame()
    df, _ = db.fetch_latest_backtest_results(run_date=run_date)
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
        trades_df["entry_price"] = pd.to_numeric(trades_df["entry_price"], errors="coerce")
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
        exposure = trades_df["entry_price"].abs() * trades_df["qty"].replace(0, np.nan)
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
    weights = {"win_rate": 0.40, "net_pnl": 0.30, "trades": 0.20, "avg_return": 0.10}

    missing = [c for c in ["win_rate", "net_pnl", "trades"] if c not in df.columns]
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

    df["avg_return"] = (
        df["net_pnl"] / df["trades"].replace(0, 1)
        if "net_pnl" in df.columns and "trades" in df.columns
        else 0
    )

    df["win_rate_norm"] = df["win_rate"] / df["win_rate"].max() if "win_rate" in df.columns else 0
    df["net_pnl_norm"] = df["net_pnl"] / df["net_pnl"].max() if "net_pnl" in df.columns else 0
    df["trades_norm"] = df["trades"] / df["trades"].max() if "trades" in df.columns else 0
    df["avg_return_norm"] = (
        df["avg_return"] / df["avg_return"].max() if "avg_return" in df.columns else 0
    )

    # Compute weighted score
    df["score"] = (
        df["win_rate_norm"] * weights["win_rate"]
        + df["net_pnl_norm"] * weights["net_pnl"]
        + df["trades_norm"] * weights["trades"]
        + df["avg_return_norm"] * weights["avg_return"]
    )

    if new_nan_rows.any():
        nan_scores = df["score"].isna()
        drop_mask = nan_scores & new_nan_rows
        if drop_mask.any():
            logger.warning(
                "Dropping %d rows with NaN scores after numeric conversion",
                int(drop_mask.sum()),
            )
            df = df.loc[~drop_mask].copy()

    ranked_df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    return ranked_df


def rank_and_filter_candidates(
    screener_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    run_date: date,
) -> Optional[pd.DataFrame]:
    if backtest_df is None:
        logger.warning("[WARN] backtest_df is None — skipping ranking")
        return None

    ranked_df = rank_candidates(backtest_df.copy())
    if ranked_df is None:
        return None

    if ranked_df.empty:
        return ranked_df

    if screener_df.empty or "symbol" not in screener_df.columns:
        logger.warning(
            "[WARN] METRICS_TOP_CANDIDATES_FILTERED reason=screener_empty run_date=%s",
            run_date,
        )
        return ranked_df.iloc[0:0].copy()

    screener_df = screener_df.copy()
    screener_df["symbol"] = screener_df["symbol"].astype(str).str.upper()
    ranked_df["symbol"] = ranked_df["symbol"].astype(str).str.upper()
    before_rows = int(ranked_df.shape[0])
    merged = ranked_df.merge(
        screener_df,
        on="symbol",
        how="inner",
        suffixes=("", "_sc"),
    )
    dropped = before_rows - int(merged.shape[0])
    if dropped:
        logger.warning(
            "[WARN] METRICS_TOP_CANDIDATES_FILTERED dropped=%s remaining=%s",
            dropped,
            int(merged.shape[0]),
        )
    for col in ("entry_price", "adv20", "atrp", "exchange", "source"):
        sc_col = f"{col}_sc"
        if sc_col in merged.columns:
            merged[col] = merged[sc_col]
            merged.drop(columns=[sc_col], inplace=True)
    return merged


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metrics ranking + summary", allow_abbrev=False)
    parser.add_argument(
        "--run-date",
        default=None,
        help="Override run_date for DB writes (YYYY-MM-DD)",
    )
    parser.add_argument('--paper-gross-pipeline', action='store_true',
        help='Validated gross-paper Pipeline metrics; fees/net and entry holds retained')
    parser.add_argument('--paper-gross-report-only', action='store_true',
        help='Publish separate paper gross report; fees/net unknown and financial hold exit2 retained')
    return parser.parse_args(list(argv) if argv is not None else None)


def _run_metrics(argv=None):
    print(">>> METRICS MAIN ENTERED <<<")
    logger.info(">>> METRICS MAIN ENTERED <<<")
    logger.info("⚙️ [METRICS] main() start")
    args = parse_args(argv)
    if getattr(args, 'paper_gross_pipeline', False):
        if getattr(args, 'paper_gross_report_only', False):
            return 2
        return _run_paper_gross_metrics(args)
    if getattr(args, 'paper_gross_report_only', False):
        if getattr(args, 'run_date', None):
            logger.error('PAPER_GROSS_REPORT_RUN_DATE_OVERRIDE_REFUSED')
            return 2
        try:
            db.publish_paper_gross_report()
        except Exception as exc:
            logger.error('PAPER_GROSS_REPORT_HELD error_type=%s', type(exc).__name__)
            return 2
        logger.warning('PAPER_GROSS_REPORT_PUBLISHED fees=UNKNOWN net=UNKNOWN financial_hold=true')
        # A successful gross artifact is not a successful financial metrics
        # stage, pipeline-readiness declaration or entry authorization.
        return 2
    run_date_override = _coerce_run_date(getattr(args, "run_date", None))
    exit_code = 0
    if getattr(args, "run_date", None) and run_date_override is None:
        logger.error("[ERROR] METRICS_RUN_DATE_INVALID value=%s", args.run_date)
        exit_code = 2

    if not db.db_enabled():
        logger.warning("[WARN] METRICS_DB_DISABLED: DATABASE_URL/DB_* not configured.")
        conn = None
        exit_code = max(exit_code, 2)
    else:
        conn = db.get_db_conn()
        if conn is None:
            logger.error("[ERROR] METRICS_DB_REQUIRED: unable to connect to database.")
            exit_code = max(exit_code, 2)

    if run_date_override:
        run_date = run_date_override
        run_date_source = "cli"
    else:
        run_date = db.fetch_latest_run_date("backtest_results")
        run_date_source = "backtest_results"
        if run_date is None:
            run_date = db.fetch_latest_run_date("screener_candidates")
            run_date_source = "screener_candidates"
        if run_date is None:
            logger.error("[ERROR] METRICS_NO_UPSTREAM_RUN_DATE")
            run_date = _resolve_run_date(conn)
            run_date_source = "fallback"
            exit_code = max(exit_code, 2)
        logger.info(
            "[INFO] METRICS_RUN_DATE_RESOLVED run_date=%s source=%s",
            run_date,
            run_date_source,
        )

    logger.info(
        "[INFO] STEP_RUN_DATE step=metrics run_date=%s source=%s",
        run_date,
        "cli" if run_date_override else "default",
    )

    backtest_df = load_results(run_date=run_date)
    try:
        screener_df, _ = get_latest_screener_candidates(run_date)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("[ERROR] Failed to load screener candidates: %s", exc)
        screener_df = pd.DataFrame()

    print(">>> Loaded screener_df and backtest_df")
    print(f">>> screener_df rows: {len(screener_df)}")
    logger.info(f">>> screener_df rows: {len(screener_df)}")
    logger.warning(f">>> screener_df rows: {len(screener_df)}")
    print(f">>> backtest_df rows: {len(backtest_df)}")
    logger.info(f">>> backtest_df rows: {len(backtest_df)}")
    logger.warning(f">>> backtest_df rows: {len(backtest_df)}")
    if screener_df.empty or backtest_df.empty:
        print(">>> Input data is empty — skipping metrics <<<")
    else:
        print(">>> Input data loaded — proceeding to ranking <<<")

    results_df = backtest_df

    trades_df: pd.DataFrame
    try:
        db_trades = _load_trades_from_db(conn)
    finally:
        if conn is not None:
            conn.close()
    if db_trades is None:
        logger.error("[ERROR] METRICS_DB_LOAD_FAILED: trades unavailable.")
        trades_df = pd.DataFrame()
        exit_code = max(exit_code, 2)
    else:
        trades_df = db_trades
    try:
        if conn is not None:
            conn.close()
    except Exception:
        pass

    # Scoped performance exclusions must never become a trade-derived ranking input.
    if "net_pnl" not in results_df.columns:
        raise db.disposition.TradePolicyError('TRADE_POLICY_TRADE_RANKING_FALLBACK_UNSUPPORTED')

    try:
        print(">>> Starting rank_and_filter_candidates")
        logger.info(">>> Starting rank_and_filter_candidates()")
        logger.info("🚦 [METRICS] Starting candidate ranking")
        ranked_df = rank_and_filter_candidates(screener_df, results_df, run_date)
        if ranked_df is None:
            print(">>> ranked_df is None — insert attempted anyway")
            logger.warning(">>> ranked_df is None — insert attempted anyway")
            print(">>> ranked_df rows: 0")
            logger.warning(">>> ranked_df rows: 0")
        else:
            print(f">>> ranked_df rows: {len(ranked_df)}")
            logger.info(f">>> ranked_df rows: {len(ranked_df)}")
            logger.warning(f">>> ranked_df rows: {len(ranked_df)}")
            if ranked_df.empty:
                print(">>> ranked_df is empty — insert attempted anyway")
                logger.warning(">>> ranked_df is empty — no rows expected in DB")
                print(f">>> ranked_df columns: {list(ranked_df.columns)}")
                logger.warning(f">>> ranked_df columns: {list(ranked_df.columns)}")
            else:
                logger.info("✅ [METRICS] ranked_df rows: %s", len(ranked_df))
                logger.info(f"[DEBUG] ranked_df shape: {ranked_df.shape}")
                logger.info(f"[DEBUG] Ranked DataFrame columns: {list(ranked_df.columns)}")
                logger.info(
                    "Screener Metrics Summary: total_candidates=%s, avg_score=%.2f",
                    len(ranked_df),
                    ranked_df["score"].mean(),
                )
                logger.info(
                    "Top 15 Screener Symbols: %s",
                    ", ".join(ranked_df["symbol"].head(15).tolist()),
                )
                logger.info(
                    "Top Candidates: %s",
                    ranked_df[["symbol", "score", "win_rate", "net_pnl"]]
                    .head(15)
                    .to_string(index=False),
                )
    except Exception:
        logger.exception("❌ [METRICS] Exception during ranking")
        ranked_df = None

    if isinstance(ranked_df, pd.DataFrame):
        top_candidates_df = ranked_df
    else:
        top_candidates_df = pd.DataFrame()
        print(">>> ranked_df is None — insert attempted anyway")
        logger.warning(">>> ranked_df is None — insert attempted anyway")

    if top_candidates_df.empty:
        print(">>> ranked_df is empty — insert attempted anyway")
        logger.warning(">>> ranked_df is empty — no rows expected in DB")

    top_candidates_rows = int(top_candidates_df.shape[0])
    db_enabled = db.db_enabled()
    print(f">>> db.db_enabled(): {db_enabled}")
    logger.warning(f">>> db.db_enabled(): {db_enabled}")
    try:
        print(f">>> Attempting to insert {top_candidates_rows} rows into top_candidates")
        logger.warning(f">>> Inserting into top_candidates: rows={top_candidates_rows}")
        db.insert_top_candidates(top_candidates_df, run_date)
        print(">>> insert_top_candidates completed successfully")
    except Exception as exc:
        print(f">>> ERROR in insert_top_candidates: {exc}")
        logger.exception("Failed to insert into top_candidates")

    if not trades_df.empty:
        trades_df = validate_numeric(trades_df, "net_pnl")

    summary_metrics = calculate_metrics(trades_df.copy())
    if trades_df.empty:
        logger.warning("Trades dataset empty; writing default metrics row.")
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

    if not summary_metrics:
        print(">>> summary_metrics is empty; columns: []")
        logger.warning(">>> summary_metrics is empty; columns: []")
        print(">>> summary_metrics is empty — upsert attempted anyway")
        logger.warning(">>> summary_metrics is empty — check metric extraction")
        summary_metrics = calculate_metrics(pd.DataFrame())
        print(">>> summary_metrics missing; using fallback zeroed metrics")
        logger.warning("summary_metrics missing; using fallback zeroed metrics.")

    print(f">>> summary_metrics: {summary_metrics}")
    logger.info(f">>> summary_metrics: {summary_metrics}")
    logger.warning(f">>> summary_metrics: {summary_metrics}")
    logger.info("🧮 [METRICS] summary_metrics: %s", summary_metrics)
    db_enabled = db.db_enabled()
    print(f">>> db.db_enabled(): {db_enabled}")
    logger.warning(f">>> db.db_enabled(): {db_enabled}")
    try:
        print(f">>> Attempting to upsert metrics_daily: keys={list(summary_metrics.keys())}")
        logger.warning(f">>> Upserting metrics_daily: keys={list(summary_metrics.keys())}")
        publication = db.bind_performance_summary(trades_df.attrs['trade_policy'], summary_metrics)
        if db.upsert_metrics_daily(summary_metrics, run_date, trade_policy_metadata=publication) is not True:
            return 2
        print(">>> upsert_metrics_daily completed successfully recovery_complete=false accounting_complete=false activity_ingestion_complete=false")
    except Exception as exc:
        print(f">>> ERROR in upsert_metrics_daily: {exc}")
        logger.error("METRICS_PERFORMANCE_PUBLICATION_REFUSED error_type=%s", type(exc).__name__)
        return 2

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
        metrics["universe_prefix_counts"] = derive_prefix_counts_from_scored_candidates(Path(BASE_DIR))

    if "universe_prefix_counts" not in metrics:
        metrics["universe_prefix_counts"] = {}

    try:
        metrics = ensure_canonical_metrics(metrics)
        write_screener_metrics_json(metrics_path, metrics)
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        logger.warning("Failed to update %s: %s", metrics_path, exc)

    print(">>> METRICS COMPLETE <<<")
    print(">>> METRICS FILE COMPLETED <<<")
    logger.info(">>> METRICS MAIN COMPLETED <<<")
    logger.info("🏁 [METRICS] main() complete")
    logger.info("[INFO] metrics.py completed successfully")

    return exit_code


def _run_paper_gross_metrics(args):
    """Actual paper metrics stage: unchanged backtest ranking, explicit gross daily output."""
    try:
        if not db.db_enabled():
            raise ValueError('paper_gross_run_context_required')
        # Validate paper context before upstream reads or any output writes.
        db._paper_disposition_identity()
        run_date = db.fetch_latest_run_date('backtest_results')
        if run_date is None or run_date != db.fetch_latest_run_date('screener_candidates'):
            raise ValueError('paper_gross_upstream_run_mismatch')
        supplied_date = getattr(args, 'run_date', None)
        if supplied_date is not None and supplied_date != run_date.isoformat():
            raise ValueError('paper_gross_upstream_run_mismatch')
        logger.info('[INFO] STEP_RUN_DATE step=metrics run_date=%s source=backtest_results', run_date)
        results = load_results(run_date=run_date)
        candidates, _ = get_latest_screener_candidates(run_date)
        if results is None or candidates is None or results.empty or candidates.empty:
            raise ValueError('paper_gross_upstream_empty')
        if 'net_pnl' not in results.columns:
            raise ValueError('paper_gross_backtest_contract_missing')
        # No trade P&L may become a ranking fallback. Retain backtest semantics.
        ranked = rank_and_filter_candidates(candidates, results, run_date)
        if ranked is None or ranked.empty:
            raise ValueError('paper_gross_ranked_candidates_empty')
        snapshot = db.gross_paper_performance_snapshot()
        publication = db.publish_paper_gross_pipeline(run_date, ranked, snapshot)
        # Publish the already reviewed dashboard artifact only after authoritative
        # DB commit, then revalidate that exact daily publication for this run.
        db.publish_paper_gross_report(expected_revision=snapshot['evidence']['policy']['policy_revision'])
        with db.paper_gross_pipeline_evidence(run_date,
                expected_publication=publication['publication_fingerprint']):
            pass
        logger.info('METRICS_PAPER_GROSS_COMPLETE basis=%s fees=UNKNOWN net=UNKNOWN '
            'accounting_complete=false entry_ready=false', publication['pnl_basis'])
        print('>>> METRICS COMPLETE <<<')
        print('>>> METRICS FILE COMPLETED <<<')
        logger.info('METRICS MAIN COMPLETED')
        logger.info('[INFO] metrics.py completed successfully')
        return 0
    except Exception as exc:
        # Never expose SQL, DSNs, raw exceptions or sensitive account values.
        logger.error('METRICS_PAPER_GROSS_HELD error_type=%s', type(exc).__name__)
        return 2


def main(argv=None):
    try:
        return _run_metrics(argv)
    except db.disposition.TradePolicyError as exc:
        logger.error('METRICS_POLICY_REFUSED code=%s', str(exc))
        return 2


if __name__ == '__main__':
    sys.exit(main())
