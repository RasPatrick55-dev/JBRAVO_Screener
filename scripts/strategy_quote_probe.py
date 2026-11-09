"""One fixed ARDT data GET with sanitized output; inert until explicitly called.

No account/order endpoint, Engine, store, database, fallback feed or retry. The
host caller must verify native source and supply its actual configured environment.
"""
from datetime import datetime, timezone
import json
import math
import time

SYMBOL = 'ARDT'
QUOTE_URL = 'https://data.alpaca.markets/v2/stocks/quotes/latest'
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
SIP_MESSAGES = frozenset({
    'subscription does not permit querying recent SIP data',
    'subscription does not permit recent SIP data',
})


class QuoteOnlySession:
    def __init__(self, session, *, utc, now):
        self.session, self.utc, self.now = session, utc, now
        self.calls = 0
        self.http_status = None
        self.observed_at = None
        self.quote_timestamp = None
        self.provider_reason = None
        self.feed = None

    def request(self, method, url, **kwargs):
        params = kwargs.get('params')
        if (self.calls != 0 or method != 'GET' or url != QUOTE_URL or
                type(params) is not dict or set(params) != {'symbols', 'feed'} or
                params['symbols'] != SYMBOL or params['feed'] not in {'sip', 'iex'} or
                kwargs.get('json') is not None or kwargs.get('allow_redirects') is not False):
            raise ValueError('quote_probe_request_scope_denied')
        self.calls += 1
        self.feed = params['feed']
        started = time.monotonic()
        response = self.session.request(method, url, **(kwargs | {
            'timeout': (5, 10), 'allow_redirects': False, 'stream': True}))
        try:
            status = response.status_code
            if type(status) is int and 100 <= status <= 599:
                self.http_status = status
            chunks = []
            size = 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES or time.monotonic() - started > 30:
                    raise ValueError('quote_probe_response_oversize')
                chunks.append(chunk)
            response._content = b''.join(chunks)
            response._content_consumed = True
            self.observed_at = self.utc(self.now())
            try:
                value = json.loads(response.content)
                if status == 200:
                    self.quote_timestamp = self.utc(value['quotes'][SYMBOL]['t'])
                elif status == 403 and self.feed == 'sip' and type(value) is dict:
                    message = value.get('message')
                    if type(message) is str and message in SIP_MESSAGES:
                        self.provider_reason = 'SIP_SUBSCRIPTION_NOT_PERMITTED'
            except (ValueError, KeyError, TypeError):
                pass
            return response
        finally:
            response.close()

    def close(self):
        self.session.close()


def probe_quote(NativeBroker, env, diagnostics, *, session=None, now=None):
    """Use native quote validation for exactly one named symbol; no feed override."""
    now = now or (lambda: datetime.now(timezone.utc))
    if session is None:
        import requests
        session = requests.Session()
        session.trust_env = False
    transport = QuoteOnlySession(session, utc=NativeBroker.quotes.__globals__['utc'], now=now)
    broker = None
    result = dict(symbol=SYMBOL, quote_feed='unknown', quote_status='QUOTE_PROBE_UNAVAILABLE',
                  account_identity_verified=False)
    try:
        broker = NativeBroker(env, session=transport, max_requests=1,
            now=lambda: transport.observed_at or now())
        result['quote_feed'] = diagnostics._feed(broker.quote_feed)
        try:
            quotes = broker.quotes({SYMBOL}, now=now())
            result['quote_status'] = 'QUOTE_AVAILABLE' if SYMBOL in quotes else 'QUOTE_MISSING'
        except Exception as error:
            result['quote_status'] = diagnostics.quote_failure_details(error,
                feed=broker.quote_feed)['quote_failure_code']
        if transport.http_status is not None:
            result['http_status'] = transport.http_status
        if transport.observed_at is not None:
            result['observed_at'] = transport.observed_at.isoformat()
        if transport.quote_timestamp is not None and transport.observed_at is not None:
            age = (transport.observed_at - transport.quote_timestamp).total_seconds()
            if math.isfinite(age):
                result.update(quote_timestamp=transport.quote_timestamp.isoformat(), quote_age_seconds=age)
                if result['quote_status'] == 'QUOTE_STALE_OR_FUTURE':
                    result['quote_status'] = 'QUOTE_FUTURE' if age < -1 else 'QUOTE_STALE'
        if transport.provider_reason is not None:
            result['provider_reason'] = transport.provider_reason
    except Exception:
        # Constructor/context failure is not evidence about the provider account.
        result['quote_status'] = 'QUOTE_PROBE_UNAVAILABLE'
    finally:
        try:
            transport.close()
        except Exception:
            result['quote_status'] = 'QUOTE_PROBE_UNAVAILABLE'
    result.update(data_get_attempts=transport.calls, account_requests=0, order_requests=0,
                  database_operations=0, broker_mutations=0, runtime_mutations=0)
    return result
