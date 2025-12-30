import logging
import os
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def db_enabled() -> bool:
    """Return True if DATABASE_URL is present."""

    return bool(os.environ.get("DATABASE_URL"))


def get_engine() -> Optional[Engine]:
    """Return SQLAlchemy engine with pool_pre_ping=True."""

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.warning("[WARN] DB_DISABLED Missing DATABASE_URL")
        return None

    try:
        return create_engine(database_url, pool_pre_ping=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_ENGINE %s", exc)
        return None


def safe_connect_test() -> bool:
    """Return True if DB reachable, else False (log warning)."""

    try:
        engine = get_engine()
        if engine is None:
            return False

        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[WARN] DB_CONNECT_TEST %s", exc)
            return False
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("[WARN] DB_CONNECT_TEST_SETUP %s", exc)
        return False

    return True
