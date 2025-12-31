CREATE TABLE IF NOT EXISTS alpaca_activities (
    id BIGSERIAL PRIMARY KEY,
    activity_id TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    transaction_time TIMESTAMPTZ,
    symbol TEXT,
    side TEXT,
    qty NUMERIC,
    price NUMERIC,
    amount NUMERIC,
    order_id TEXT,
    description TEXT,
    raw JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT alpaca_activities_activity_id_type_key UNIQUE (activity_id, activity_type)
);

CREATE INDEX IF NOT EXISTS idx_alpaca_activities_transaction_time
    ON alpaca_activities (transaction_time DESC);

CREATE INDEX IF NOT EXISTS idx_alpaca_activities_symbol
    ON alpaca_activities (symbol);

CREATE INDEX IF NOT EXISTS idx_alpaca_activities_activity_type
    ON alpaca_activities (activity_type);
