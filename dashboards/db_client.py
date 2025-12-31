import logging
import os
from functools import lru_cache
from typing import Any, Mapping, Optional, Tuple

import dash_bootstrap_components as dbc
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_engine() -> Optional[Engine]:
    """Return a cached SQLAlchemy engine or ``None`` if not configured."""

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return None
    try:
        return create_engine(database_url, pool_pre_ping=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_ENGINE_INIT_FAIL err=%s", exc)
        return None


def db_query_df(sql: str, params: Optional[Mapping[str, Any]] = None) -> Tuple[Optional[pd.DataFrame], Optional[dbc.Alert]]:
    """Execute ``sql`` and return a DataFrame or a Dash ``Alert`` on failure.

    The helper reads ``DATABASE_URL`` from the environment, uses SQLAlchemy for
    connections, and returns a tuple of ``(df, alert)`` where only one will be
    populated on success/failure respectively.
    """

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return None, dbc.Alert(
            "DATABASE_URL is not configured. Account data is unavailable.",
            color="warning",
            className="mb-3",
        )

    engine = _get_engine()
    if engine is None:
        return None, dbc.Alert(
            "Unable to initialize database engine. Check DATABASE_URL.",
            color="danger",
            className="mb-3",
        )

    try:
        with engine.connect() as connection:
            df = pd.read_sql(text(sql), connection, params=params)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_QUERY_FAIL err=%s", exc)
        return None, dbc.Alert(
            f"Database query failed: {exc}",
            color="danger",
            className="mb-3",
        )

    return df, None
