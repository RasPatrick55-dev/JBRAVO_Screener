CREATE OR REPLACE VIEW v_activities_recent AS
SELECT *
FROM alpaca_activities
ORDER BY transaction_time DESC NULLS LAST, id DESC
LIMIT 200;

CREATE OR REPLACE VIEW v_fills_recent AS
SELECT *
FROM alpaca_activities
WHERE activity_type ILIKE '%FILL%'
ORDER BY transaction_time DESC NULLS LAST, id DESC
LIMIT 200;

CREATE OR REPLACE VIEW v_cash_activity_recent AS
SELECT *
FROM alpaca_activities
WHERE activity_type ILIKE '%FEE%'
   OR activity_type ILIKE '%DIV%'
   OR activity_type ILIKE '%TRANS%'
ORDER BY transaction_time DESC NULLS LAST, id DESC
LIMIT 200;
