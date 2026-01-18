import datetime as dt

import pandas as pd
import pytest

from scripts import db


@pytest.mark.integration
def test_top_candidates_upsert_creates_and_updates_rows():
    """
    Phase 2 validation:
    - Ensures top_candidates table exists
    - Ensures upsert works (insert + idempotent update)
    - Ensures run_date + symbol uniqueness
    """

    # --- Preconditions ---
    assert db.db_enabled(), "Database must be enabled for this test"
    assert db.safe_connect_test(), "Database connection test failed"

    # Ensure table exists (idempotent)
    db.ensure_top_candidates_table()

    run_date = dt.date.today()

    # --- Test data (intentionally floats, DB-safe) ---
    df = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "score": 0.90,
                "win_rate": 55.0,
                "net_pnl": 1200.0,
                "trades": 10,
                "wins": 6,
                "losses": 4,
                "expectancy": 120.0,
                "profit_factor": 1.5,
                "max_drawdown": -200.0,
                "sharpe": 1.2,
                "sortino": 1.8,
                "exchange": "NYSE",
                "entry_price": 50.0,
                "adv20": 5_000_000,
                "atrp": 0.03,
                "source": "metrics",
            },
            {
                "symbol": "BBB",
                "score": 0.80,
                "win_rate": 48.0,
                "net_pnl": 600.0,
                "trades": 8,
                "wins": 4,
                "losses": 4,
                "expectancy": 75.0,
                "profit_factor": 1.2,
                "max_drawdown": -150.0,
                "sharpe": 0.9,
                "sortino": 1.1,
                "exchange": "NASDAQ",
                "entry_price": 32.5,
                "adv20": 3_200_000,
                "atrp": 0.04,
                "source": "metrics",
            },
        ]
    )

    # --- First upsert (INSERT path) ---
    db.upsert_top_candidates(run_date, df)

    # --- Second upsert (UPDATE path, idempotency) ---
    df.loc[df["symbol"] == "AAA", "net_pnl"] = 1300.0
    db.upsert_top_candidates(run_date, df)

    # --- Verification ---
    with db.get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT symbol, net_pnl
                FROM top_candidates
                WHERE run_date = %s
                ORDER BY symbol
                """,
                (run_date,),
            )
            rows = cur.fetchall()

    # Expect exactly 2 rows
    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

    results = {symbol: float(net_pnl) for symbol, net_pnl in rows}

    # Ensure update applied
    assert results["AAA"] == 1300.0
    assert results["BBB"] == 600.0
