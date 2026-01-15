import argparse
import logging

from scripts import db

logger = logging.getLogger(__name__)


TABLE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS reconcile_state (
        id INTEGER PRIMARY KEY,
        last_after TIMESTAMPTZ,
        last_ran_at TIMESTAMPTZ,
        updated_at TIMESTAMPTZ DEFAULT now()
    );
    """,
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
    """
    CREATE TABLE IF NOT EXISTS order_events (
        event_id BIGSERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        qty NUMERIC,
        order_id TEXT,
        status TEXT,
        event_type TEXT NOT NULL,
        event_time TIMESTAMPTZ NOT NULL,
        raw JSONB,
        created_at TIMESTAMPTZ DEFAULT now()
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
        trade_id BIGSERIAL PRIMARY KEY,
        symbol TEXT NOT NULL,
        qty NUMERIC,
        entry_order_id TEXT UNIQUE,
        entry_time TIMESTAMPTZ,
        entry_price NUMERIC,
        exit_order_id TEXT,
        exit_time TIMESTAMPTZ,
        exit_price NUMERIC,
        realized_pnl NUMERIC,
        exit_reason TEXT,
        status TEXT NOT NULL DEFAULT 'OPEN',
        created_at TIMESTAMPTZ DEFAULT now(),
        updated_at TIMESTAMPTZ DEFAULT now()
    );
    """,
]

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS screener_candidates_symbol_idx ON screener_candidates (symbol);",
    "CREATE INDEX IF NOT EXISTS screener_candidates_run_date_idx ON screener_candidates (run_date);",
    "CREATE INDEX IF NOT EXISTS executed_trades_symbol_idx ON executed_trades (symbol);",
    "CREATE INDEX IF NOT EXISTS executed_trades_entry_time_idx ON executed_trades (entry_time);",
    "CREATE INDEX IF NOT EXISTS executed_trades_exit_time_idx ON executed_trades (exit_time);",
    "CREATE INDEX IF NOT EXISTS idx_order_events_symbol ON order_events(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_order_events_order_id ON order_events(order_id);",
    "CREATE INDEX IF NOT EXISTS idx_order_events_event_time ON order_events(event_time);",
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);",
    "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);",
    "CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);",
]

RECONCILE_STATE_SEED = """
INSERT INTO reconcile_state (id, last_after, last_ran_at)
VALUES (1, NULL, NULL)
ON CONFLICT (id) DO NOTHING;
"""


def _execute_statement(engine, statement: str) -> bool:
    try:
        if engine is None:
            return False
        with engine:
            with engine.cursor() as cursor:
                cursor.execute(statement)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_MIGRATE %s", exc)
        return False


def run_upgrade(engine) -> bool:
    if not _repair_screener_candidates_schema(engine):
        return False

    for ddl in TABLE_STATEMENTS + INDEX_STATEMENTS:
        if not _execute_statement(engine, ddl):
            return False
    if not _execute_statement(engine, RECONCILE_STATE_SEED):
        return False
    return True


def _repair_screener_candidates_schema(engine) -> bool:
    try:
        if engine is None:
            return False
        with engine:
            with engine.cursor() as cursor:
                cursor.execute("SELECT to_regclass('public.screener_candidates')")
                row = cursor.fetchone()
                table_exists = row[0] if row else None
                if not table_exists:
                    return True

                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema=current_schema()
                      AND table_name='screener_candidates'
                    """
                )
                columns = {row[0] for row in cursor.fetchall()}

                if "run_date" in columns:
                    return True

                if "run date" in columns:
                    cursor.execute('ALTER TABLE screener_candidates RENAME COLUMN "run date" TO run_date')
                    return True

                cursor.execute("SELECT COUNT(*) FROM screener_candidates")
                row = cursor.fetchone()
                rowcount = row[0] if row else 0
                if rowcount == 0:
                    logger.warning(
                        "[WARN] DB_MIGRATE screener_candidates missing run_date; recreating empty table"
                    )
                    cursor.execute("DROP TABLE screener_candidates")
                    return True

                logger.warning(
                    "[WARN] DB_MIGRATE screener_candidates missing run_date with existing data; manual repair required"
                )
                return False
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_MIGRATE_SCHEMA_CHECK %s", exc)
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Database migration utility")
    parser.add_argument("--action", choices=["upgrade"], default="upgrade")
    args = parser.parse_args()

    try:
        if not db.db_enabled():
            logger.warning("[WARN] DB_MIGRATE db_disabled; skipping")
            return

        conn = db.get_db_conn()
        if conn is None:
            logger.warning("[WARN] DB_MIGRATE connection unavailable; skipping")
            return

        try:
            if args.action == "upgrade":
                success = run_upgrade(conn)
                if success:
                    logger.info("Database migration upgrade completed.")
                else:
                    logger.warning("[WARN] DB_MIGRATE Upgrade encountered issues")
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_MIGRATE_MAIN %s", exc)


if __name__ == "__main__":
    main()
