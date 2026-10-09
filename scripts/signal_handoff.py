"""Inert session/candidate contract. No SDK, DB, credential or order side effects."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import hashlib
import json
import math
from numbers import Integral, Real
import re
from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')
MAX_CANDIDATES = 10000
ARRIVAL_MINUTES = 15


def utc(value):
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('ambiguous_timestamp')
    return parsed.astimezone(timezone.utc)


def binding(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def candidate_values(rows, *, ordered=False):
    """Bind full DB values including types, missing values and optional-field presence."""
    def encode(value):
        if value is None:
            return ['null']
        if type(value) is bool:
            return ['bool', value]
        if isinstance(value, datetime):
            return ['datetime', value.isoformat()]
        if isinstance(value, date):
            return ['date', value.isoformat()]
        if isinstance(value, str):
            return ['string', value]
        if isinstance(value, Decimal):
            if not value.is_finite():
                raise ValueError('invalid_candidate_value')
            return ['decimal', str(value)]
        if isinstance(value, Integral):
            return ['integer', str(value)]
        if isinstance(value, Real):
            number = float(value)
            if math.isnan(number):
                return ['missing_float']
            if not math.isfinite(number):
                raise ValueError('invalid_candidate_value')
            return ['float', number.hex()]
        if isinstance(value, (list, tuple)):
            return ['sequence', [encode(item) for item in value]]
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise ValueError('invalid_candidate_field')
            return ['object', [[k, encode(value[k])] for k in sorted(value)]]
        if type(value).__module__ == 'pandas._libs.missing' and type(value).__name__ == 'NAType':
            return ['missing_nullable']
        raise ValueError('unsupported_candidate_value')
    if not isinstance(rows, list) or len(rows) > MAX_CANDIDATES or any(not isinstance(r, dict) for r in rows):
        raise ValueError('invalid_candidate_population')
    population = rows if ordered else sorted(rows, key=lambda row: row.get('symbol', ''))
    data = json.dumps([encode(row) for row in population], separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()
    if len(data) > 4 * 1024 * 1024:
        raise ValueError('candidate_value_binding_oversize')
    return binding(data)


def _at(day, hhmm):
    return datetime.combine(day, time.fromisoformat(hhmm), NY).astimezone(timezone.utc)


def session_contract(calendar, *, now):
    """Latest returned exchange session, never yesterday merely because today is incomplete.

    Regular-session calendar is NOT an extended-hours calendar. Only standard
    09:30-16:00 signal sessions have the explicit 20:00 ET policy here. Early or
    exceptional signal sessions block pending a separately evidenced policy.
    """
    now = utc(now)
    local_now = now.astimezone(NY)
    today = local_now.date()
    # Before premarket opens, today's session has not begun. Yesterday's
    # completed session may still feed today's opening, including after midnight.
    signal_day_limit = today if local_now.time() >= time(4) else today - timedelta(days=1)
    if not isinstance(calendar, list) or not calendar or len(calendar) > 32:
        raise ValueError('invalid_session_calendar')
    normalized = []
    for item in calendar:
        day = date.fromisoformat(item['date'])
        if not today - timedelta(days=10) <= day <= today + timedelta(days=10):
            raise ValueError('calendar_outside_request_window')
        opened, closed = time.fromisoformat(item['open']), time.fromisoformat(item['close'])
        if opened.tzinfo is not None or closed.tzinfo is not None or opened >= closed:
            raise ValueError('invalid_calendar_hours')
        normalized.append({'date': day.isoformat(), 'open': opened.isoformat(), 'close': closed.isoformat()})
    normalized.sort(key=lambda item: item['date'])
    if len({item['date'] for item in normalized}) != len(normalized):
        raise ValueError('duplicate_calendar_session')
    prior = [item for item in normalized if date.fromisoformat(item['date']) <= signal_day_limit]
    if not prior:
        raise ValueError('signal_session_missing')
    signal = prior[-1]
    day = date.fromisoformat(signal['date'])
    following = [item for item in normalized if item['date'] > signal['date']]
    if not following or (today - day).days > 7:
        raise ValueError('next_session_missing_or_calendar_stale')
    next_session = following[0]
    if signal['open'] != '09:30:00' or signal['close'] != '16:00:00' or next_session['open'] != '09:30:00':
        raise ValueError('extended_session_cutoff_unqualified')
    cutoff = _at(day, '20:00')
    ready = cutoff + timedelta(minutes=ARRIVAL_MINUTES)
    opening = _at(date.fromisoformat(next_session['date']), '04:00')
    expires = _at(date.fromisoformat(next_session['date']), '09:30')
    if now < ready:
        raise ValueError('signal_session_not_completed')
    if now >= opening:
        raise ValueError('preparation_missed_opening')
    return {'version': 1, 'signal_session': signal['date'], 'execution_session': next_session['date'],
            'completed_cutoff_utc': cutoff.isoformat(), 'ready_after_utc': ready.isoformat(),
            'entry_open_utc': opening.isoformat(), 'expires_at_utc': expires.isoformat(),
            'arrival_allowance_minutes': ARRIVAL_MINUTES,
            'cutoff_policy': 'standard_session_extended_20_et',
            'calendar_source': 'alpaca_trading_calendar', 'calendar_observed_at': now.isoformat(),
            'calendar': normalized, 'reference_field': 'close',
            'reference_semantics': 'supplied_daily_bar_close_not_guaranteed_20_et_last_trade'}


def validate_session(session, *, started, finished, now, for_entry=False):
    """Recompute policy fields and require preparation completion before opening."""
    if not isinstance(session, dict) or type(session.get('version')) is not int:
        raise ValueError('invalid_session_contract')
    expected = session_contract(session['calendar'], now=utc(session['calendar_observed_at']))
    if session != expected:
        raise ValueError('session_contract_changed')
    started, finished, now = utc(started), utc(finished), utc(now)
    ready, opening, expires = map(utc, (session['ready_after_utc'], session['entry_open_utc'], session['expires_at_utc']))
    if not utc(session['calendar_observed_at']) <= started <= finished < opening or started < ready or finished > now:
        raise ValueError('invalid_session_preparation_window')
    if now >= expires:
        raise ValueError('signal_handoff_expired')
    if for_entry and now < opening:
        raise ValueError('entry_before_session_open')


def freeze_candidates(rows, session, *, run_ts):
    if not rows:
        raise ValueError('empty_signal_candidates')
    references, seen = [], set()
    for row in rows:
        symbol = row.get('symbol')
        if not isinstance(symbol, str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,15}', symbol) or symbol in seen:
            raise ValueError('invalid_or_duplicate_signal_symbol')
        seen.add(symbol)
        run_date = row.get('run_date')
        if isinstance(run_date, date):
            run_date = run_date.isoformat()
        if run_date != session['signal_session'] or utc(row.get('run_ts_utc')) != utc(run_ts):
            raise ValueError('signal_candidate_session_or_batch_mismatch')
        timestamp = utc(row.get('timestamp'))
        if timestamp != _at(date.fromisoformat(session['signal_session']), '00:00'):
            raise ValueError('signal_bar_session_mismatch')
        price = _price(row.get('close'))
        source = row.get('source')
        if source is not None and (not isinstance(source, str) or not re.fullmatch('[a-zA-Z0-9_:-]{1,80}', source)):
            source = 'unrecognized_source_label'
        references.append({'symbol': symbol, 'price': str(price), 'bar_timestamp': timestamp.isoformat(),
                           'field': 'close', 'source': source})
    return {'binding_version': 1, 'query_contract': 'candidate_ranker_join_v1',
            'value_binding': candidate_values(rows), 'count': len(rows),
            'references': sorted(references, key=lambda item: item['symbol'])}


def _price(value):
    if isinstance(value, bool) or not isinstance(value, (str, Decimal, Real)):
        raise ValueError('invalid_signal_price')
    try:
        price = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('invalid_signal_price') from exc
    if not price.is_finite() or price < 1:
        raise ValueError('invalid_signal_price')
    return price


def reference(snapshot, symbol):
    matches = [item for item in snapshot['references'] if item['symbol'] == symbol]
    if len(matches) != 1:
        raise ValueError('unbound_signal_reference')
    return _price(matches[0]['price'])


def price_limit(snapshot, symbol, *, buffer_pct, max_gap_pct):
    """Passive frozen-price limit; no quote substitution or marketable-gap bypass.

    Retain the configured buffer and existing maximum premium. Floor cents to
    stay below the cap through later Alpaca tick normalization. No fill promise.
    """
    try:
        values = [Decimal(str(value)) for value in (buffer_pct, max_gap_pct)]
    except InvalidOperation as exc:
        raise ValueError('invalid_signal_price_policy') from exc
    if any(not value.is_finite() or not 0 <= value <= 100 for value in values):
        raise ValueError('invalid_signal_price_policy')
    anchor = reference(snapshot, symbol)
    premium = min(values)
    try:
        return float((anchor * (1 + premium / 100)).quantize(Decimal('.01'), rounding=ROUND_FLOOR))
    except InvalidOperation as exc:
        raise ValueError('invalid_signal_price_policy') from exc


def price_cap(snapshot, symbol, max_gap_pct):
    return price_limit(snapshot, symbol, buffer_pct=max_gap_pct, max_gap_pct=max_gap_pct)
