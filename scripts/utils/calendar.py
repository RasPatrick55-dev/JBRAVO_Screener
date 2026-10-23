from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from zoneinfo import ZoneInfo

from scripts import signal_handoff


def completed_signal_session(trading_client, *, now=None, provisional=False):
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(ZoneInfo('America/New_York')).date()
    response = trading_client.get_calendar(GetCalendarRequest(
        start=today - timedelta(days=10), end=today + timedelta(days=10)))
    # Freeze the returned calendar, not a weekday approximation. SDK calendar
    # hours are local wall times; reject unexpected representations downstream.
    calendar = [{'date': item.date.isoformat(), 'open': item.open.strftime('%H:%M:%S'),
                 'close': item.close.strftime('%H:%M:%S')} for item in response]
    return signal_handoff.session_contract(calendar, now=now, provisional=provisional)


def calc_daily_window(
    trading_client: TradingClient,
    days: int,
    *,
    end_date: date | None = None,
    session_cutoff: bool = False,
):
    today = end_date or datetime.now(timezone.utc).date()
    req = GetCalendarRequest(start=(today - timedelta(days=5000)), end=today)
    cal = trading_client.get_calendar(req)
    sessions = [d for d in cal if getattr(d, "close", None)]
    if not sessions:
        raise RuntimeError("No closed trading sessions returned by Alpaca calendar")
    last = sessions[-1].date
    start_idx = max(0, len(sessions) - 1 - days)
    start = sessions[start_idx].date
    from scripts.pipeline_postflight import provisional_context
    provisional = provisional_context()
    if provisional is not None:
        if provisional['signal_session'] != last.isoformat():
            raise ValueError('provisional_signal_window_mismatch')
        ny = ZoneInfo('America/New_York')
        lower = datetime.combine(start, datetime.min.time(), ny).astimezone(timezone.utc)
        period_end = datetime.combine(last, datetime.min.time(), ny).replace(hour=20).astimezone(timezone.utc)
        requested_end = min(period_end, datetime.fromisoformat(provisional['started_at']))
        return lower.isoformat(), requested_end.isoformat(), last.isoformat()
    if not session_cutoff:
        return f"{start}T00:00:00Z", f"{last}T23:59:59Z", last.isoformat()
    if end_date is None or last != end_date:
        raise ValueError('signal_window_session_mismatch')
    # Handoff mode uses the explicit completed cutoff, not a future UTC/NY
    # end-of-day. Endpoint filtering selects bar labels; it does not synthesize
    # an extended-hours close or alter Alpaca's trade-condition aggregation.
    ny = ZoneInfo('America/New_York')
    lower = datetime.combine(start, datetime.min.time(), ny).astimezone(timezone.utc)
    upper = datetime.combine(last, datetime.min.time(), ny).replace(hour=20).astimezone(timezone.utc)
    return lower.isoformat(), upper.isoformat(), last.isoformat()
