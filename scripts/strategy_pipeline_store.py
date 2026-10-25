"""Explicit strategy batch migration and reads. Never executes on import."""

SCHEMA_SQL = """
CREATE TABLE public.screener_strategy_batches_app (
    run_id TEXT PRIMARY KEY CHECK(run_id ~ '^[0-9a-f]{32}$'),
    run_date DATE NOT NULL,
    run_ts_utc TIMESTAMPTZ NOT NULL UNIQUE,
    archive_id TEXT NOT NULL CHECK(archive_id ~ '^[0-9a-f]{64}$'),
    archive_sha256 TEXT NOT NULL CHECK(archive_sha256 ~ '^[0-9a-f]{64}$'),
    candidate_count INTEGER NOT NULL CHECK(candidate_count BETWEEN 0 AND 3),
    payload JSONB NOT NULL CHECK(jsonb_typeof(payload)='object'),
    CHECK((payload->>'candidate_count')::integer=candidate_count),
    CHECK(jsonb_array_length(payload->'proposals')=candidate_count)
);
ALTER TABLE public.screener_strategy_batches_app OWNER TO jbravo_app;
CREATE INDEX screener_strategy_batch_session_idx ON public.screener_strategy_batches_app (run_date DESC,run_ts_utc DESC);
CREATE OR REPLACE VIEW public.latest_screener_candidates AS
WITH latest_batch AS (
    SELECT run_date,run_ts_utc FROM public.screener_strategy_batches_app
    ORDER BY run_date DESC,run_ts_utc DESC LIMIT 1
)
SELECT c.*, NULL::double precision AS model_score_5d, NULL::double precision AS model_score
FROM public.screener_candidates c
JOIN latest_batch b ON c.run_date=b.run_date AND c.run_ts_utc=b.run_ts_utc;
CREATE OR REPLACE VIEW public.latest_top_candidates AS
WITH latest_batch AS (
    SELECT run_date,run_ts_utc FROM public.screener_strategy_batches_app
    ORDER BY run_date DESC,run_ts_utc DESC LIMIT 1
)
SELECT t.*, b.run_ts_utc, NULL::double precision AS model_score_5d, NULL::double precision AS model_score
FROM public.top_candidates t
JOIN latest_batch b ON t.run_date=b.run_date
JOIN public.screener_candidates c ON c.run_date=b.run_date AND c.run_ts_utc=b.run_ts_utc AND c.symbol=t.symbol;
"""


def batch_count(conn, run_date=None):
    from scripts.strategy_pipeline import latest_batch
    with conn.cursor() as cur:
        row = latest_batch(cur, run_date=run_date)
    return (row[2], row[0]) if row else (0, None)


def candidate_frame(conn, run_date, *, limit=None, exact_run_ts=None):
    """Read only an explicitly published batch, including its zero-row timestamp."""
    import pandas as pd
    from scripts.strategy_pipeline import latest_batch
    with conn.cursor() as cur:
        batch = latest_batch(cur, run_date=run_date, exact_run_ts=exact_run_ts)
        if batch is None:
            return pd.DataFrame(), None
        # This timestamp is mandatory even for an empty batch. No legacy fallback.
        query = ('SELECT c.*, NULL::double precision AS model_score_5d, NULL::double precision AS model_score '
            'FROM public.screener_candidates c WHERE c.run_date=%s AND c.run_ts_utc=%s '
            'ORDER BY c.score DESC NULLS LAST,c.symbol')
        params = [batch[0], batch[1]]
        if limit is not None:
            if type(limit) is not int or not 1 <= limit <= 10000:
                raise ValueError('invalid_strategy_candidate_limit')
            query += ' LIMIT %s'; params.append(limit)
        cur.execute(query, params)
        rows, columns = cur.fetchall(), [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=columns), batch[1]
