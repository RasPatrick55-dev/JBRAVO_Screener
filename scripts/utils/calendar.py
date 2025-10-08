from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest


def calc_daily_window(trading_client: TradingClient, days: int):
    today = datetime.now(timezone.utc).date()
    req = GetCalendarRequest(start=(today - timedelta(days=5000)), end=today)
    cal = trading_client.get_calendar(req)
    sessions = [d for d in cal if getattr(d, "close", None)]
    if not sessions:
        raise RuntimeError("No closed trading sessions returned by Alpaca calendar")
    last = sessions[-1].date
    start_idx = max(0, len(sessions) - 1 - days)
    start = sessions[start_idx].date
    return f"{start}T00:00:00Z", f"{last}T23:59:59Z", last.isoformat()
