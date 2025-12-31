-- Views for Alpaca account snapshots.

CREATE OR REPLACE VIEW v_account_latest AS
SELECT *
FROM alpaca_account_snapshots
WHERE taken_at = (
    SELECT MAX(taken_at) FROM alpaca_account_snapshots
);

CREATE OR REPLACE VIEW v_account_daily_close AS
WITH ordered AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY DATE(taken_at AT TIME ZONE 'utc')
            ORDER BY taken_at DESC
        ) AS rn
    FROM alpaca_account_snapshots
)
SELECT *
FROM ordered
WHERE rn = 1;

CREATE OR REPLACE VIEW v_account_equity_curve AS
SELECT
    taken_at,
    equity,
    cash,
    buying_power,
    portfolio_value,
    account_id
FROM alpaca_account_snapshots
ORDER BY taken_at;
