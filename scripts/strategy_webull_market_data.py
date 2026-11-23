"""Bounded Webull market-data transport and session-bound price evidence."""
from base64 import b64encode
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from time import monotonic
from urllib.parse import quote
from uuid import uuid4
from zoneinfo import ZoneInfo

from scripts.research_dataset import canonical_hash, strict_json
from scripts.strategy_webull_quotes import QuoteError, SYMBOL, instant, normalize, positive

NY = ZoneInfo('America/New_York')
BARS = '/market-data/stocks/bars/list'
DEPTH = '/market-data/stocks/depths/list'
HOSTS = {'production': 'api.webull.com', 'sandbox': 'api.sandbox.webull.com'}
MAX_FINALISTS = 20
MAX_BAR_AGE = timedelta(minutes=30)
REGULAR_LOOKBACK = timedelta(minutes=15)
QUOTE_AGE = timedelta(seconds=15)
MAX_SPREAD = Decimal('.01')
CAPTURE = 'jbravo.webull-after-hours.v1'
LIMIT_POLICY = 'RAW_AFTER_HOURS_CLOSE_PLUS_0_5_PCT_V1'
ALPACA_IEX = 'alpaca_iex'
WEBULL = 'webull'
ALPACA_DATA = 'https://data.alpaca.markets'


def require(value, code):
    if not value:
        raise ValueError(code)


def require_hosted_qualification(env, environment):
    require(env.get('JBR_WEBULL_DATA_ENV') == environment and
            env.get('JBR_WEBULL_HOSTED_QUALIFIED') == 'true', 'webull_hosted_data_not_qualified')


def sandbox_application():
    directory = Path.home() / '.config/jbravo/webull-market-data'
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(fd)
            require(metadata.st_uid == os.getuid() and stat.S_IMODE(metadata.st_mode) == 0o700,
                    'webull_sandbox_credentials_unavailable')
            file = os.open('app.json', os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd)
            with os.fdopen(file, 'rb') as stream:
                metadata = os.fstat(stream.fileno())
                require(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.getuid() and
                        metadata.st_nlink == 1 and stat.S_IMODE(metadata.st_mode) == 0o600 and
                        0 < metadata.st_size <= 16384, 'webull_sandbox_credentials_unavailable')
                data = stream.read(16385)
        finally:
            os.close(fd)
        require(len(data) <= 16384, 'webull_sandbox_credentials_unavailable')
        app = strict_json(data)
        require(type(app) is dict and set(app) == {'app_key', 'app_secret'} and all(
            type(value) is str and 1 <= len(value) <= 512 and
            all(32 < ord(character) < 127 for character in value) for value in app.values()),
            'webull_sandbox_credentials_unavailable')
    except (OSError, AttributeError, ValueError):
        raise ValueError('webull_sandbox_credentials_unavailable') from None
    return app


def session_hours(row, early_cutoffs):
    """Keep the regular close separate from Webull's published ATH cutoff."""
    day = date.fromisoformat(row['date'])
    opening = time.fromisoformat(row['open'])
    close = time.fromisoformat(row['close'])
    require(opening == time(9, 30), 'unsupported_webull_regular_open')
    if close == time(16):
        require(row['date'] not in early_cutoffs, 'webull_calendar_close_mismatch')
        cutoff = time(20)
    else:
        require(close == time(13) and row['date'] in early_cutoffs,
                'unverified_webull_early_close')
        cutoff = time.fromisoformat(early_cutoffs[row['date']])
        require(cutoff == time(17), 'unverified_webull_early_close')
    return dict(regular_close_at=datetime.combine(day, close, NY).isoformat(),
                after_hours_close_at=datetime.combine(day, cutoff, NY).isoformat())


def signature(path, query, headers, body, secret):
    fields = dict(query, **headers)
    pairs = '&'.join(f'{name}={value}' for name, value in sorted(fields.items()))
    message = f'{path}&{pairs}'
    if body is not None:
        message += '&' + hashlib.md5(body, usedforsecurity=False).hexdigest().upper()
    return b64encode(hmac.digest((secret + '&').encode(), quote(message, safe='').encode(),
                                 'sha1')).decode()


class MarketData:
    """Two fixed market-data routes with no broker or token-management operations."""

    def __init__(self, env):
        import requests
        environment = env.get('JBR_WEBULL_DATA_ENV')
        require(environment in HOSTS, 'webull_data_environment_required')
        require(env.get('JBR_WEBULL_NON_DISPLAY_AUTHORIZED') == 'true',
                'webull_non_display_authorization_required')
        self.environment = environment
        self.host = HOSTS[environment]
        if environment == 'sandbox':
            app = sandbox_application()
            self.key, self.secret = app['app_key'], app['app_secret']
            self.token = None
        else:
            self.key = env.get('JBR_WEBULL_DATA_APP_KEY')
            self.secret = env.get('JBR_WEBULL_DATA_APP_SECRET')
            self.token = env.get('JBR_WEBULL_DATA_ACCESS_TOKEN')
        require(self.key and self.secret, 'webull_data_credentials_unavailable')
        self.session = requests.Session()
        self.session.trust_env = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()

    def _request(self, method, path, query, payload=None):
        headers = {'host': self.host, 'x-app-key': self.key,
                   'x-signature-algorithm': 'HMAC-SHA1', 'x-signature-version': '1.0',
                   'x-signature-nonce': uuid4().hex,
                   'x-timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
        body = json.dumps(payload, separators=(',', ':')).encode() if payload is not None else None
        headers['x-signature'] = signature(path, query, headers, body, self.secret)
        headers.update({'x-version': 'v3', 'Accept': 'application/json',
                        'Accept-Encoding': 'identity'})
        if body is not None:
            headers['Content-Type'] = 'application/json'
        if self.token:
            headers['x-access-token'] = self.token
        try:
            deadline = monotonic() + 6
            with self.session.request(method, 'https://' + self.host + path, params=query,
                    data=body, headers=headers, timeout=(2, 3), stream=True,
                    allow_redirects=False) as response:
                if response.status_code != 200:
                    raise QuoteError('QUOTE_FORBIDDEN' if response.status_code in (401, 403)
                                     else 'QUOTE_HTTP_ERROR')
                raw = bytearray()
                max_bytes = 1024 * 1024 if path == BARS else 64 * 1024
                for part in response.iter_content(8192):
                    raw.extend(part)
                    require(len(raw) <= max_bytes and monotonic() <= deadline,
                            'webull_response_budget_exceeded')
                return strict_json(bytes(raw))
        except QuoteError:
            raise
        except Exception:
            raise QuoteError('QUOTE_TRANSPORT_UNAVAILABLE') from None

    def bars(self, symbols, start, end, session):
        require(1 <= len(symbols) <= MAX_FINALISTS and
                all(SYMBOL.fullmatch(symbol) for symbol in symbols), 'invalid_webull_symbols')
        require(session in {'RTH', 'ATH'}, 'invalid_webull_bar_session')
        # The OpenAPI US SDK uses these fields on the batch bars endpoint.
        # Webull excludes the minute at end_time, so send the exact boundary.
        return self._request('POST', BARS, {}, dict(symbols=list(symbols), category='US_STOCK',
            timespan='M1', count=str(int((end - start).total_seconds() / 60)),
            trading_sessions=session, real_time_required=self.environment == 'production',
            start_time=int(start.timestamp() * 1000), end_time=int(end.timestamp() * 1000)))

    def quote(self, symbol):
        require(SYMBOL.fullmatch(symbol), 'invalid_webull_symbol')
        return self._request('GET', DEPTH, dict(symbol=symbol, category='US_STOCK',
                                               depth='1', overnight_required='false'))


def webull_fallback(resources, env, environment, operation):
    """Open the qualified supplemental client only on the first data gap."""
    from functools import cache

    @cache
    def client():
        try:
            require_hosted_qualification(env, environment)
            return resources.enter_context(MarketData(env))
        except ValueError:
            return None

    def fetch(*args):
        market = client()
        if market is None:
            raise QuoteError('QUOTE_TRANSPORT_UNAVAILABLE')
        return getattr(market, operation)(*args)

    return fetch


def alpaca_bars(provider, symbols, start, end, session):
    """Adapt bounded raw IEX data to the minute-bar validator."""
    from scripts.daily_strategy_scan import pages

    request = dict(
        symbols=','.join(symbols), timeframe='1Min', feed='iex', adjustment='raw',
        currency='USD', sort='asc', limit=10000, start=start.isoformat(),
        end=(end - timedelta(minutes=1)).isoformat())
    collected = {symbol: [] for symbol in symbols}
    for _, value, _ in pages(provider, ALPACA_DATA + '/v2/stocks/bars', request, max_pages=3):
        require(isinstance(value['bars'], dict) and set(value['bars']) <= set(symbols),
                'alpaca_minute_bar_symbol_changed')
        for symbol, rows in value['bars'].items():
            current = collected[symbol]
            if current is None or not isinstance(rows, list) or len(current) + len(rows) > 240:
                collected[symbol] = None
            else:
                current.extend(rows)
    groups = []
    for symbol, rows in collected.items():
        try:
            bars = [dict(time=row['t'], open=row['o'], high=row['h'], low=row['l'],
                         close=row['c'], volume=row['v'], trading_sessions=session) for row in rows]
        except (KeyError, TypeError, ValueError):
            bars = None
        groups.append(dict(symbol=symbol, result=bars))
    return groups


def alpaca_quote(provider, symbol):
    require(SYMBOL.fullmatch(symbol), 'invalid_alpaca_quote_symbol')
    value, _ = provider.get(ALPACA_DATA + '/v2/stocks/quotes/latest',
                            dict(symbols=symbol, feed='iex'), timeout=(2, 3))
    require(isinstance(value['quotes'], dict) and set(value['quotes']) <= {symbol},
            'alpaca_quote_symbol_changed')
    return value['quotes'].get(symbol)


def bar_time(value):
    """Preserve the provider's minute timestamp."""
    if isinstance(value, str) and not value.isdigit():
        # Webull emits offsets like +0000. Python 3.10 requires +00:00.
        if value.endswith('Z'):
            value = value[:-1] + '+00:00'
        elif re.fullmatch(r'[+-][0-9]{4}', value[-5:]):
            value = value[:-2] + ':' + value[-2:]
        return instant(datetime.fromisoformat(value))
    from scripts.strategy_webull_quotes import quote_time
    return quote_time(value)


def checked_bars(rows, start, end, session):
    require(isinstance(rows, list) and len(rows) <= 240, 'invalid_webull_bar_count')
    bars = {}
    for row in rows:
        stamp = bar_time(row['time'])
        tag = row.get('trading_sessions', row.get('trading_session'))
        require(start <= stamp < end and stamp.second == stamp.microsecond == 0 and
                stamp not in bars and tag == session and row.get('trading_session', session) == session,
                'invalid_webull_bar_scope')
        o, h, l, c, v = (positive(row[name]) for name in ('open', 'high', 'low', 'close', 'volume'))
        require(l <= min(o, c) <= max(o, c) <= h, 'invalid_webull_ohlcv')
        bars[stamp] = dict(time=stamp.isoformat(), open=str(o), high=str(h), low=str(l),
                          close=str(c), volume=str(v), trading_sessions=session)
    return [bar for _, bar in sorted(bars.items())]


def bar_groups(fetch, symbols, start, end, session):
    if not symbols:
        return {}
    payload = fetch(symbols, start, end, session)
    if isinstance(payload, dict):
        payload = [payload] if 'symbol' in payload else payload.get('data', payload.get('result'))
    require(isinstance(payload, list) and len(payload) <= len(symbols), 'invalid_webull_bar_envelope')
    groups = {}
    for item in payload:
        symbol = item['symbol']
        require(symbol in symbols and symbol not in groups, 'invalid_webull_bar_symbol')
        try:
            groups[symbol] = checked_bars(item['result'], start, end, session)
        except (KeyError, TypeError, ValueError):
            groups[symbol] = None
    return groups


def reference_status(bars, regular_bar, cutoff):
    if cutoff - bar_time(bars[-1]['time']) > MAX_BAR_AGE:
        return 'STALE'
    return 'AVAILABLE' if regular_bar is not None else 'MISSING_REGULAR_CLOSE'


def capture_window(primary_fetch, fetch, symbols, start, end, session, *, usable=bool):
    def read(source, requested):
        try:
            return bar_groups(source, requested, start, end, session)
        except (KeyError, TypeError, ValueError):
            return dict.fromkeys(requested)

    primary = read(primary_fetch, symbols) if primary_fetch is not None else {}
    gaps = tuple(symbol for symbol in symbols if not usable(primary.get(symbol)))
    supplemental = read(fetch, gaps) if fetch is not None else {}
    result = {}
    for symbol in symbols:
        first, second = primary.get(symbol, []), supplemental.get(symbol, [])
        if second and (usable(second) or not first):
            result[symbol] = (second, WEBULL)
        elif first:
            result[symbol] = (first, ALPACA_IEX)
        else:
            result[symbol] = (None if first is None or second is None else [], None)
    return result


def capture_after_hours(symbols, hours, *, fetch, now, environment='production', primary_fetch=None):
    """Capture the last regular-session minute and after-hours bars for finalists."""
    start, end = (datetime.fromisoformat(hours[key]) for key in
                  ('regular_close_at', 'after_hours_close_at'))
    require(len(symbols) <= MAX_FINALISTS and now() >= end + timedelta(minutes=15),
            'webull_capture_not_ready_or_too_large')
    universe = tuple(sorted(symbols))

    def recent(bars):
        return bool(bars) and end - bar_time(bars[-1]['time']) <= MAX_BAR_AGE

    after_hours = capture_window(primary_fetch, fetch, universe, start, end, 'ATH', usable=recent)
    with_after_hours = tuple(symbol for symbol, (bars, _) in after_hours.items() if recent(bars))
    regular = capture_window(primary_fetch, fetch, with_after_hours, start - REGULAR_LOOKBACK, start, 'RTH')
    data = {}
    for symbol in universe:
        bars, after_hours_source = after_hours[symbol]
        regular_bars, regular_source = regular.get(symbol, (None, None))
        regular_bar = regular_bars[-1] if regular_bars else None
        status = (reference_status(bars, regular_bar, end) if bars else
                  'INVALID' if bars is None else 'MISSING')
        data[symbol] = dict(status=status, bars=bars or [], regular_bar=regular_bar,
                            after_hours_source=after_hours_source, regular_source=regular_source)
    provider = 'alpaca_iex_with_webull_fallback' if primary_fetch is not None else WEBULL
    return dict(schema=CAPTURE, provider=provider, environment=environment, adjustment='raw', interval='M1',
                session=start.date().isoformat(), hours=hours, observed_at=now().isoformat(),
                symbols=data)


def confirm_capture(capture, session, symbols, *, published):
    """Replay bar validation and coverage from the exact archive."""
    hours = session['webull_hours']
    start, end = (datetime.fromisoformat(hours[k]) for k in
                  ('regular_close_at', 'after_hours_close_at'))
    taken = instant(datetime.fromisoformat(capture['observed_at']))
    allowed_sources = (ALPACA_IEX, WEBULL) if capture['provider'] == 'alpaca_iex_with_webull_fallback' else (WEBULL,)
    require(capture['schema'] == CAPTURE and capture['provider'] in ('alpaca_iex_with_webull_fallback', WEBULL) and
            capture.get('environment', 'production') == session.get('webull_data_environment', 'production') and
            capture['adjustment'] == 'raw' and capture['interval'] == 'M1' and
            capture['session'] == session['signal_session'] == start.date().isoformat() and
            capture['hours'] == hours and end + timedelta(minutes=15) <= taken <= published and
            set(capture['symbols']) == set(symbols), 'webull_capture_scope_changed')
    for reference in capture['symbols'].values():
        bars = reference['bars']
        regular_bar = reference['regular_bar']
        for field, present in (('after_hours_source', bool(bars)), ('regular_source', regular_bar is not None)):
            source = reference.get(field, WEBULL if present else None)
            require(source in allowed_sources if present else source is None,
                    'after_hours_source_changed')
        require(checked_bars(bars, start, end, 'ATH') == bars, 'webull_capture_bars_changed')
        if regular_bar is not None:
            require(checked_bars([regular_bar], start - REGULAR_LOOKBACK, start, 'RTH') == [regular_bar],
                    'webull_regular_bar_changed')
        if bars:
            expected = reference_status(bars, regular_bar, end)
        else:
            expected = reference['status']
            require(regular_bar is None and expected in {'MISSING', 'INVALID'},
                    'webull_capture_coverage_changed')
        require(reference['status'] == expected, 'webull_capture_coverage_changed')


def screening_evidence(reference, cutoff):
    sources = dict(after_hours_source=reference.get('after_hours_source', WEBULL if reference['bars'] else None),
                   regular_source=reference.get('regular_source', WEBULL if reference['regular_bar'] else None))
    if reference['status'] != 'AVAILABLE':
        return dict(status=reference['status'], **sources)
    bars = reference['bars']
    last = bars[-1]
    last_close = positive(last['close'])
    regular_bar = reference['regular_bar']
    move = last_close / positive(regular_bar['close']) - 1
    return dict(status='AVAILABLE', confirmed=move > 0, return_since_last_regular_bar=str(move),
                regular_bar_at=regular_bar['time'], regular_bar_close=regular_bar['close'],
                dollar_volume_estimate=str(sum(positive(b['close']) * positive(b['volume']) for b in bars)),
                last_bar_at=last['time'], age_at_cutoff_seconds=int((cutoff - bar_time(last['time'])).total_seconds()),
                bar_count=len(bars), **sources)


def limit_order(proposal, reference, capture, parent, *, selection_id, published, opening):
    last = reference['bars'][-1]
    reference_price = positive(last['close'])
    price = (reference_price * Decimal('1.005')).quantize(Decimal('.01'), rounding=ROUND_FLOOR)
    require(price > 0 and published < opening, 'webull_limit_not_publishable')
    source = reference.get('after_hours_source', WEBULL)
    provider = (dict(provider='alpaca', feed='iex') if source == ALPACA_IEX else
                dict(provider='webull', environment=capture.get('environment', 'production')))
    return dict(schema='jbravo.pipeline-paper-limit-order.v2', primary_run_id=parent['primary_run_id'],
        selection_run_id=selection_id, signal_id=proposal['signal_id'], symbol=proposal['symbol'],
        signal_session=proposal['signal_session'], next_session=proposal['next_session'],
        limit_price=str(price), currency='USD', price_basis=LIMIT_POLICY, price_as_of=last['time'],
        valid_from=published.isoformat(), valid_until=(opening + timedelta(minutes=35)).isoformat(),
        price_provenance=dict(**provider,
            adjustment='raw', reference_price=str(reference_price),
            bars_sha256=canonical_hash(reference['bars']), capture_sha256=canonical_hash(capture),
            captured_at=capture['observed_at'], multiplier='1.005', rounding='ROUND_FLOOR'))


def check_premarket_book(book, *, provider, ticket, opening, observed, premarket=True):
    from scripts.strategy_execution import utc

    stamp = utc(book['timestamp'])
    if not opening <= stamp <= observed < opening + timedelta(minutes=1) or not premarket:
        return dict(status='QUOTE_OUTSIDE_FIRST_PREMARKET_MINUTE', provider=provider)
    if observed - stamp > QUOTE_AGE:
        return dict(status='QUOTE_STALE_OR_FUTURE', provider=provider)
    bid, ask = positive(book['bid']), positive(book['ask'])
    require(bid <= ask, 'crossed_quote')
    spread = (ask - bid) / ((ask + bid) / 2)
    status = ('QUOTE_SPREAD_TOO_WIDE' if spread > MAX_SPREAD else
              'QUOTE_ASK_ABOVE_PIPELINE_LIMIT' if ask > positive(ticket['limit_price']) else 'PASS')
    return dict(status=status, provider=provider, timestamp=stamp.isoformat(),
                observed_at=observed.isoformat(), bid=str(bid), ask=str(ask), is_nbbo=False)


def premarket_decision(fetch, symbol, *, ticket, opening, now, primary_fetch=None):
    """Check an exact symbol's quote at receipt time without changing its ticket."""
    if primary_fetch is not None:
        try:
            quote = primary_fetch(symbol)
            primary = (check_premarket_book(dict(timestamp=quote['t'], bid=quote['bp'], ask=quote['ap']),
                provider=ALPACA_IEX, ticket=ticket, opening=opening, observed=now()) if quote else
                dict(status='QUOTE_MISSING', provider=ALPACA_IEX))
        except (KeyError, TypeError, ValueError):
            primary = dict(status='QUOTE_INVALID_OR_UNAVAILABLE', provider=ALPACA_IEX)
        # A usable Alpaca quote is authoritative, even when it rejects the trade.
        if primary['status'] in ('PASS', 'QUOTE_ASK_ABOVE_PIPELINE_LIMIT', 'QUOTE_SPREAD_TOO_WIDE'):
            return primary
        if not opening <= now() < opening + timedelta(minutes=1):
            return primary
    try:
        payload = fetch(symbol)
        observed = now()
        book = normalize(payload, symbol=symbol, observed=observed)
        # Depth can omit a session label. The request excludes overnight and
        # the provider timestamp itself must fall inside the premarket window.
        premarket = all(payload.get(field, 'PRE') == 'PRE'
                        for field in ('trading_session', 'trading_sessions'))
        decision = check_premarket_book(book, provider=WEBULL, ticket=ticket, opening=opening,
                                        observed=observed, premarket=premarket)
    except QuoteError as error:
        decision = dict(status=str(error), provider=WEBULL)
    except (KeyError, TypeError, ValueError):
        decision = dict(status='QUOTE_INVALID', provider=WEBULL)
    if primary_fetch is not None:
        decision['alpaca_status'] = primary['status']
    return decision
