import logging
from typing import Any, Mapping, Optional, Tuple

import dash_bootstrap_components as dbc
import pandas as pd

from scripts import db

logger = logging.getLogger(__name__)


def _fetch_dataframe(sql: str, params: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    conn = db.get_db_conn()
    if conn is None:
        raise RuntimeError("db_unavailable")
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, params or {})
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def db_query_df(sql: str, params: Optional[Mapping[str, Any]] = None) -> Tuple[Optional[pd.DataFrame], Optional[dbc.Alert]]:
    """Execute ``sql`` and return a DataFrame or a Dash ``Alert`` on failure.

    The helper reads Postgres connection settings from the environment and
    returns a tuple of ``(df, alert)`` where only one will be
    populated on success/failure respectively.
    """

    try:
        df = _fetch_dataframe(sql, params=params)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_QUERY_FAIL err=%s", exc)
        return None, dbc.Alert(
            f"Database query failed: {exc}",
            color="danger",
            className="mb-3",
        )

    return df, None
