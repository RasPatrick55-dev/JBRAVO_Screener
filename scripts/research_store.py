"""Explicit PostgreSQL research I/O. No connection or environment loading on import."""
from __future__ import annotations

import os
from pathlib import Path


class StoreError(ValueError):
    """Sanitized connection/transaction diagnostics; never include DSNs or server errors."""


def connect(*, write=False, env=None):
    env = os.environ if env is None else env
    import psycopg2

    dsn = env.get("RESEARCH_DATABASE_URL") or (None if write else env.get("DATABASE_URL"))
    kwargs = {"connect_timeout": 5, "application_name": "jbravo_strategy_research",
              "options": "-c statement_timeout=15000 -c lock_timeout=2000"}
    if not write:
        kwargs["options"] += " -c default_transaction_read_only=on"
    if not dsn:
        if write:
            raise StoreError("Research writes require explicit RESEARCH_DATABASE_URL")
        required = ("DB_HOST", "DB_NAME")
        if not all(env.get(k) for k in required) or not (env.get("DB_USER") or env.get("DB_USERNAME")):
            raise StoreError("Database is not configured in this process")
        kwargs.update(host=env["DB_HOST"], dbname=env["DB_NAME"],
                      user=env.get("DB_USER") or env["DB_USERNAME"],
                      password=env.get("DB_PASSWORD") or env.get("DB_PASS"),
                      port=env.get("DB_PORT", "5432"), sslmode=env.get("DB_SSLMODE", "prefer"))
    conn = None
    try:
        conn = psycopg2.connect(dsn, **kwargs) if dsn else psycopg2.connect(**kwargs)
        conn.set_session(readonly=not write, isolation_level="REPEATABLE READ", autocommit=False)
        return conn
    except Exception as exc:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        message = str(exc).lower()
        category = next((label for needle, label in (
            ("could not translate host name", "DNS_LOOKUP_FAILED"),
            ("name or service not known", "DNS_LOOKUP_FAILED"),
            ("password authentication failed", "AUTHENTICATION_REJECTED"),
            ("connection refused", "CONNECTION_REFUSED"),
            ("timeout", "CONNECTION_TIMEOUT"),
            ("no pg_hba.conf entry", "ACCESS_POLICY_REJECTED"),
            ("ssl", "TLS_CONNECTION_FAILED"),
        ) if needle in message), type(exc).__name__)
        raise StoreError("PostgreSQL connection failed: " + category) from None


def inspect_database(*, env=None):
    conn = connect(env=env)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT current_setting('transaction_read_only'), current_setting('transaction_isolation')")
            read_only, isolation = cur.fetchone()
            cur.execute("SELECT table_schema, table_name FROM information_schema.tables "
                        "WHERE table_name = ANY(%s) ORDER BY table_schema, table_name",
                        (["daily_bars", "ml_artifacts", "candidates", "research_runs"],))
            tables = [dict(schema=s, table=t) for s, t in cur.fetchall()]
            cur.execute("SELECT column_name, data_type FROM information_schema.columns "
                        "WHERE table_schema='public' AND table_name='daily_bars' ORDER BY ordinal_position")
            columns = [dict(name=n, type=t) for n, t in cur.fetchall()]
        return {"authenticated": True, "transaction_read_only": read_only,
                "transaction_isolation": isolation, "tables": tables, "daily_bars_columns": columns}
    except Exception as exc:
        raise StoreError("PostgreSQL inspection failed (" + type(exc).__name__ + ")") from None
    finally:
        conn.rollback()
        conn.close()


def snapshot_daily_bars(symbols, start, end, *, max_rows=250_000, env=None):
    """Bounded immutable-input export; production daily_bars is never altered.

    Feed, adjustment and historical identity are absent from that table's schema.
    This export is explicitly UNQUALIFIED until independently reconciled.
    """
    if not symbols or len(symbols) > 1000 or not 1 <= max_rows <= 250_000:
        raise StoreError("Snapshot symbol/row limit exceeded")
    conn = connect(env=env)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol, date, open, high, low, close, volume FROM public.daily_bars "
                        "WHERE symbol = ANY(%s) AND date BETWEEN %s AND %s "
                        "ORDER BY symbol, date LIMIT %s", (list(symbols), start, end, max_rows + 1))
            rows = cur.fetchall()
        if len(rows) > max_rows:
            raise StoreError("Snapshot row cap exceeded; no truncated snapshot is accepted")
        return {"source": "postgres_public_daily_bars", "research_qualified": False,
                "feed": "UNKNOWN", "adjustment": "UNKNOWN", "currency": "UNKNOWN",
                "transaction": "READ ONLY REPEATABLE READ", "rows": [
                    dict(zip(("symbol", "session", "open", "high", "low", "close", "volume"),
                             [str(v) for v in row])) for row in rows]}
    except StoreError:
        raise
    except Exception as exc:
        raise StoreError("PostgreSQL snapshot failed (" + type(exc).__name__ + ")") from None
    finally:
        conn.rollback()
        conn.close()


SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS jbravo_research;
CREATE TABLE IF NOT EXISTS jbravo_research.research_runs (
    run_id TEXT PRIMARY KEY,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    specification_sha256 CHAR(64) NOT NULL,
    result_sha256 CHAR(64) NOT NULL,
    result JSONB NOT NULL,
    CHECK (jsonb_typeof(result) = 'object' AND result @> '{"research_qualified": false, "execution_authorized": false}'::jsonb)
);
"""


def initialize_database(*, env=None):
    conn = connect(write=True, env=env)
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise StoreError("Research schema initialization failed (" + type(exc).__name__ + ")") from None
    finally:
        conn.close()


def publish_result(path: Path, *, env=None):
    import hashlib
    from scripts.research_dataset import bounded_read, strict_json

    path = Path(path).resolve()
    body = bounded_read(path, limit=16 * 1024 * 1024)
    result = strict_json(body)
    if result.get("research_qualified") is not False or result.get("execution_authorized") is not False:
        raise StoreError("Research result must preserve unqualified/non-execution status")
    if result.get("schema") != "jbravo.strategy-study-result.v1" or result.get("status") != "COMPLETED_EXPLORATORY":
        raise StoreError("Only completed, versioned study results can be archived")
    if result.get("completed_runs") != result.get("planned_runs") or result.get("completed_runs") != len(result.get("runs", [])):
        raise StoreError("Study trial accounting is incomplete")
    manifest = strict_json(bounded_read(path.parent / "manifest.json", limit=1024 * 1024))
    bound_result = False
    if manifest.get("run_id") != result.get("run_id"):
        raise StoreError("Study/result manifest identity mismatch")
    for binding in manifest["bindings"]:
        artifact = (path.parent / binding["path"]).resolve()
        if not artifact.is_relative_to(path.parent):
            raise StoreError("Study artifact escapes the result directory")
        data = bounded_read(artifact)
        if len(data) != binding["bytes"] or hashlib.sha256(data).hexdigest() != binding["sha256"]:
            raise StoreError("Study artifact hash/size mismatch")
        bound_result |= artifact == path
    if not bound_result:
        raise StoreError("Result is not bound by the completed study manifest")
    conn = connect(write=True, env=env)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO jbravo_research.research_runs "
                        "(run_id,specification_sha256,result_sha256,result) VALUES (%s,%s,%s,%s::jsonb)",
                        (result["run_id"], result["specification_sha256"], hashlib.sha256(body).hexdigest(), body.decode()))
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise StoreError("Research result publication failed (" + type(exc).__name__ + ")") from None
    finally:
        conn.close()
