import argparse
import logging

from sqlalchemy import text

from scripts import db

logger = logging.getLogger(__name__)


TABLE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_date DATE PRIMARY KEY,
        started_at TIMESTAMPTZ,
        ended_at TIMESTAMPTZ,
        rc INTEGER,
        summary JSONB,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS screener_candidates (
        run_date DATE NOT NULL,
        timestamp TIMESTAMPTZ,
        symbol TEXT NOT NULL,
        score NUMERIC,
        exchange TEXT,
        close NUMERIC,
        volume BIGINT,
        universe_count INTEGER,
        score_breakdown JSONB,
        entry_price NUMERIC,
        adv20 BIGINT,
        atrp NUMERIC,
        source TEXT,
        created_at TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (run_date, symbol)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS backtest_results (
        run_date DATE NOT NULL,
        symbol TEXT NOT NULL,
        trades INTEGER,
        win_rate NUMERIC,
        net_pnl NUMERIC,
        expectancy NUMERIC,
        profit_factor NUMERIC,
        max_drawdown NUMERIC,
        sharpe NUMERIC,
        sortino NUMERIC,
        created_at TIMESTAMPTZ DEFAULT now(),
        PRIMARY KEY (run_date, symbol)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics_daily (
        run_date DATE PRIMARY KEY,
        total_trades INTEGER,
        win_rate NUMERIC,
        net_pnl NUMERIC,
        expectancy NUMERIC,
        profit_factor NUMERIC,
        max_drawdown NUMERIC,
        sharpe NUMERIC,
        sortino NUMERIC,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS executed_trades (
        trade_id BIGSERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        qty INTEGER,
        entry_time TIMESTAMPTZ,
        entry_price NUMERIC,
        exit_time TIMESTAMPTZ,
        exit_price NUMERIC,
        pnl NUMERIC,
        net_pnl NUMERIC,
        order_id TEXT,
        status TEXT,
        raw JSONB,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    """,
]

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS screener_candidates_symbol_idx ON screener_candidates (symbol);",
    "CREATE INDEX IF NOT EXISTS screener_candidates_run_date_idx ON screener_candidates (run_date);",
    "CREATE INDEX IF NOT EXISTS executed_trades_symbol_idx ON executed_trades (symbol);",
    "CREATE INDEX IF NOT EXISTS executed_trades_entry_time_idx ON executed_trades (entry_time);",
    "CREATE INDEX IF NOT EXISTS executed_trades_exit_time_idx ON executed_trades (exit_time);",
]


def _execute_statement(engine, statement: str) -> bool:
    try:
        execute_fn = getattr(engine, "execute", None)
        if callable(execute_fn):
            execute_fn(text(statement))
        else:
            with engine.begin() as connection:
                connection.execute(text(statement))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_MIGRATE %s", exc)
        return False

    return True


def run_upgrade(engine) -> bool:
    for ddl in TABLE_STATEMENTS + INDEX_STATEMENTS:
        if not _execute_statement(engine, ddl):
            return False
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Database migration utility")
    parser.add_argument("--action", choices=["upgrade"], default="upgrade")
    args = parser.parse_args()

    try:
        if not db.db_enabled():
            logger.warning("[WARN] DB_MIGRATE DATABASE_URL not set; skipping")
            return

        engine = db.get_engine()
        if engine is None:
            logger.warning("[WARN] DB_MIGRATE Engine unavailable; skipping")
            return

        if args.action == "upgrade":
            success = run_upgrade(engine)
            if success:
                logger.info("Database migration upgrade completed.")
            else:
                logger.warning("[WARN] DB_MIGRATE Upgrade encountered issues")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_MIGRATE_MAIN %s", exc)


if __name__ == "__main__":
    main()
