"""Offline provider fixtures only. No database/provider access or orders."""
from copy import deepcopy
from datetime import datetime, date, time, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from alpaca.trading.models import Calendar
from dashboards import reporting_periods as P, combined_trade_gross as C
import test_combined_trade_gross as F

pytestmark = pytest.mark.alpaca_optional


def now(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def calendar(*days, early=None):
    return [Calendar(date=day, open="09:30", close="13:00" if day == early else "16:00") for day in days]


@pytest.mark.parametrize("at,dates,first,last,included,excluded", [
    ("2026-11-09T17:00:00Z", ["2026-11-02","2026-11-09","2026-11-10","2026-11-13","2026-11-30"],
     "2026-11-09", "2026-11-13", "2026-11-09T09:00:00Z", "2026-11-06T20:00:00Z"),
    ("2026-09-07T17:00:00Z", ["2026-09-01","2026-09-04","2026-09-08","2026-09-11","2026-09-30"],
     "2026-09-08", "2026-09-11", None, "2026-09-07T15:00:00Z"),
    ("2026-04-03T21:00:00Z", ["2026-03-30","2026-04-01","2026-04-02","2026-04-30"],
     "2026-03-30", "2026-04-02", "2026-04-02T23:30:00Z", "2026-04-03T15:00:00Z"),
    ("2026-11-29T23:00:00Z", ["2026-11-02","2026-11-23","2026-11-24","2026-11-25","2026-11-27","2026-11-30"],
     "2026-11-23", "2026-11-27", "2026-11-28T00:30:00Z", "2026-11-26T15:00:00Z"),
    ("2026-03-09T04:00:00Z", ["2026-03-02","2026-03-06","2026-03-09","2026-03-13","2026-03-31"],
     "2026-03-09", "2026-03-13", "2026-03-09T04:00:00Z", "2026-03-09T03:59:59.999999999Z"),
    ("2026-11-02T05:00:00Z", ["2026-11-02","2026-11-06","2026-11-30"],
     "2026-11-02", "2026-11-06", "2026-11-02T05:00:00Z", "2026-11-02T04:59:59.999999999Z"),
])
def test_week_is_current_ny_week_open_dates_through_exact_now(at, dates, first, last, included, excluded):
    periods = P.validate(calendar(*dates, early="2026-11-27"), now(at))
    metadata = periods.metadata("w")
    assert (metadata["start_date"], metadata["end_date"]) == (first, last)
    if included:
        assert periods.selected(included, "w")
    assert not periods.selected(excluded, "w")
    assert not periods.selected(at.replace("Z", "000000001Z") if "." in at else at.replace("Z", ".000000001Z"), "w")


@pytest.mark.parametrize("at,dates,start,end,prior", [
    ("2026-01-01T18:00:00Z", ["2025-12-29","2025-12-30","2025-12-31","2026-01-02","2026-01-30"], "2026-01-02","2026-01-30","2025-12-31T18:00:00Z"),
    ("2026-08-01T15:00:00Z", ["2026-07-27","2026-07-31","2026-08-03","2026-08-31"], "2026-08-03","2026-08-31","2026-07-31T18:00:00Z"),
    ("2028-02-29T19:00:00Z", ["2028-02-01","2028-02-28","2028-02-29","2028-03-01","2028-03-03"], "2028-02-01","2028-02-29","2028-01-31T18:00:00Z"),
    ("2026-12-01T00:30:00Z", ["2026-11-02","2026-11-30","2026-12-01","2026-12-04"], "2026-11-02","2026-11-30","2026-10-30T18:00:00Z"),
])
def test_month_rollover_holidays_leap_and_ny_date(at, dates, start, end, prior):
    periods = P.validate(calendar(*dates), now(at))
    assert periods.metadata("m")["start_date"] == start
    assert periods.metadata("m")["end_date"] == end
    assert not periods.selected(prior, "m")
    if start > now(at).astimezone(P.NY).date().isoformat():
        assert not any(periods.selected(day+"T17:00:00Z", "m") for day in dates)


@pytest.mark.parametrize("bad", [[], [None], calendar("2026-11-09","2026-11-09"),
    calendar("2026-11-10","2026-11-09"), calendar("2026-11-07"), calendar("2026-12-01"),
    [SimpleNamespace(date=date(2026,11,9),open=time(16),close=time(9))],
    [SimpleNamespace(date=date(2026,11,9),open=time(9,tzinfo=timezone.utc),close=time(16))]])
def test_calendar_malformed_never_approximated(bad):
    with pytest.raises((ValueError, AttributeError)):
        P.validate(bad, now("2026-11-09T17:00:00Z"))


def test_provider_request_is_finite_union_and_early_close_is_valid():
    client = Mock(get_calendar=Mock(return_value=calendar("2026-11-02","2026-11-27","2026-11-30","2026-12-04",early="2026-11-27")))
    periods = P.read_calendar(client, now("2026-11-30T17:00:00Z"))
    request = client.get_calendar.call_args.args[0]
    assert request.start == date(2026,11,1) and request.end == date(2026,12,4)
    assert periods.selected("2026-11-28T00:00:00Z", "m")
    assert not periods.selected("2026-11-28T06:00:00Z", "m")


def test_one_calendar_supplies_stats_ranks_and_preserves_history():
    at = now("2026-11-09T17:00:00Z")
    periods = P.validate(calendar("2026-11-02","2026-11-09","2026-11-13","2026-11-30"),at)
    base = F.cohort()["rows"][0]
    rows, economics = [], {}
    for index, stamp in enumerate(["2026-11-09T09:00:00Z", "2026-11-02T17:00:00Z", "2026-11-08T17:00:00Z"]):
        row=deepcopy(base); row.update(row_id=f"legacy:{index}",exit_time=stamp,gross_pnl=str(index+1))
        rows.append(row); economics[row["row_id"]]=Decimal(index+1)
    result=C.summarize(rows,economics,[],0,0,at,"a"*64,periods)
    stats={row["key"]:row for row in result["stats"]}
    assert stats["w"]["tradesCount"] == 1 and stats["w"]["totalPL"] == "1"
    assert stats["m"]["tradesCount"] == 2 and stats["m"]["totalPL"] == "3"
    assert len(result["rows"]) == stats["all"]["tradesCount"] == 3
    assert result["rankings"]["w"]["winners"][0]["pl"] == "1"
    assert result["rankings"]["m"]["winners"][0]["pl"] == "3"
    partial=C.summarize(rows,economics,[],0,0,at,"a"*64,None)
    for key in ("d","y","all"):
        assert next(s for s in partial["stats"] if s["key"]==key)==stats[key]
        assert partial["rankings"][key]==result["rankings"][key]
    assert partial["rows"]==result["rows"] and partial["coverage"]==result["coverage"]
    assert partial["status"] == "partial"
    for metric in partial["stats"][1:3]:
        assert metric["tradesCount"] is metric["totalPL"] is metric["coverage"] is None
        assert metric["period"]["status"] == "unavailable"
        assert partial["rankings"][metric["key"]] is None


def test_calendar_failure_is_bounded_and_does_not_reopen_database(monkeypatch):
    conn=F.connection()
    monkeypatch.setattr(C.E,"READ_PLANS",{**C.E.READ_PLANS,**C.PLANS})
    monkeypatch.setattr(C.D,"read_policy",lambda *a,**kw:F.policy())
    client=SimpleNamespace(get_account=lambda:SimpleNamespace(id=F.S.ACCOUNT),get_calendar=Mock(side_effect=TimeoutError("private")))
    calls=[]
    def read(fn,seconds):
        calls.append(seconds)
        return fn()
    result=C.snapshot(lambda:conn,client,F.IDENTITY,now=F.NOW,read=read)
    assert calls == [2,10,2] and result["status"] == "partial"
    assert result["stats"][-1]["tradesCount"] == 2 and result["rankings"]["w"] is None
    assert "private" not in str(result)
    conn.close.assert_called_once()
