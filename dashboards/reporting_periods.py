"""Read-only dashboard periods from the broker's US equity session calendar.

Never approximates holidays or changes execution/session eligibility. Open dates
include pre/postmarket completions; the captured observation instant caps results.
"""
from calendar import monthrange
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from alpaca.trading.requests import GetCalendarRequest

from dashboards import strategy_trade_history as H
from scripts import trade_disposition as D

NY = ZoneInfo("America/New_York")


def bounds(now):
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise ValueError("reporting_calendar_invalid")
    today = now.astimezone(NY).date()
    monday = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    month_end = today.replace(day=monthrange(today.year, today.month)[1])
    return {"w": (monday, monday + timedelta(days=4)), "m": (month_start, month_end)}


def read_calendar(client, now):
    periods = bounds(now)
    start = min(p[0] for p in periods.values())
    end = max(p[1] for p in periods.values())
    response = client.get_calendar(GetCalendarRequest(start=start, end=end))
    return validate(response, now)


def validate(response, now):
    periods = bounds(now)
    start = min(p[0] for p in periods.values())
    end = max(p[1] for p in periods.values())
    if type(response) is not list or not 1 <= len(response) <= 35:
        raise ValueError("reporting_calendar_invalid")
    days = []
    for item in response:
        day, opened, closed = item.date, item.open, item.close
        if (type(day) is not date or not start <= day <= end or day.weekday() > 4
                or days and day <= days[-1]
                or not isinstance(opened, (datetime, time))
                or not isinstance(closed, (datetime, time))):
            raise ValueError("reporting_calendar_invalid")
        # Alpaca SDK uses naive local datetime values for regular session hours.
        if isinstance(opened, datetime):
            if opened.date() != day:
                raise ValueError("reporting_calendar_invalid")
            opened = opened.timetz()
        if isinstance(closed, datetime):
            if closed.date() != day:
                raise ValueError("reporting_calendar_invalid")
            closed = closed.timetz()
        if opened.tzinfo is not None or closed.tzinfo is not None or not opened < closed:
            raise ValueError("reporting_calendar_invalid")
        days.append(day)
    month_start, month_end = periods["m"]
    if not any(month_start <= day <= month_end for day in days):
        raise ValueError("reporting_calendar_invalid")
    return Periods(now, tuple(days))


class Periods:
    """One request's validated calendar, bound to its captured observation time."""
    def __init__(self, now, days):
        self.now = now
        requested = bounds(now).values()
        self.request_start = min(p[0] for p in requested).isoformat()
        self.request_end = max(p[1] for p in requested).isoformat()
        self.days = {key: frozenset(day for day in days if start <= day <= end)
                     for key, (start, end) in bounds(now).items()}

    def selected(self, stamp, key):
        return (H.instant(stamp) <= H.instant(self.now)
                and D.aware_time(stamp).astimezone(NY).date() in self.days[key])

    def metadata(self, key):
        days = sorted(self.days[key])
        start, end = (days[0].isoformat(), days[-1].isoformat()) if days else (None, None)
        as_of = self.now.isoformat()
        label = (f"{start} to {end}" if days else "No open trading dates")
        label += f" | America/New_York | period-to-date through {self.now.astimezone(NY).isoformat()}"
        return dict(status="ready", basis="US_EQUITY_OPEN_DATES", source="alpaca_calendar",
                    timezone="America/New_York", start_date=start, end_date=end,
                    request_start_date=self.request_start, request_end_date=self.request_end,
                    coverage_basis="PROVIDER_RETURNED_UNPAGINATED_CALENDAR",
                    as_of=as_of, session_dates=[day.isoformat() for day in days], label=label)
