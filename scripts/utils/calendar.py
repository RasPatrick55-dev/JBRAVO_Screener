from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest


def _coerce_date(value: object) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            try:
                return datetime.fromisoformat(value).date()
            except ValueError:
                return None
    return None


def _calendar_entries(calendar: Iterable[object]) -> list[object]:
    return list(calendar or [])


def calc_daily_window(trading_client: TradingClient, days: int) -> Tuple[str, str, str]:
    """Return ISO start/end strings for the most recent ``days`` trading sessions."""

    now = datetime.now(timezone.utc)
    today = now.date()
    lookback_start = today - timedelta(days=4000)
    request = GetCalendarRequest(start=lookback_start, end=today)
    calendar = _calendar_entries(trading_client.get_calendar(request))

    sessions: list[tuple[date, object]] = []
    for entry in calendar:
        session_date = _coerce_date(getattr(entry, "date", None) or getattr(entry, "session", None))
        if session_date is None or session_date > today:
            continue
        close_value = getattr(entry, "close", None) or getattr(entry, "close_time", None)
        if not close_value:
            continue
        sessions.append((session_date, entry))

    if not sessions:
        raise RuntimeError("No closed trading sessions returned by Alpaca calendar")

    sessions.sort(key=lambda item: item[0])
    last_day = sessions[-1][0]

    days = max(1, int(days))
    start_index = max(0, len(sessions) - 1 - days)
    start_day = sessions[start_index][0]

    start_iso = f"{start_day.isoformat()}T00:00:00Z"
    end_iso = f"{last_day.isoformat()}T23:59:59Z"

    return start_iso, end_iso, last_day.isoformat()


__all__ = ["calc_daily_window"]
