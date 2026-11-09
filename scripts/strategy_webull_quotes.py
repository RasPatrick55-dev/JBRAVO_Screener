"""Opt-in Webull depth quotes for the pinned Alpaca paper monitor.

Passive on import. This module does not load/save credentials, initialize an SDK,
authenticate, install packages, submit orders, or change a running monitor. A
reviewed bootstrap must supply an already authorized market-data-only transport.
No snapshots, last trades, formatted MCP text, or alternate-feed fallback.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import math
import re

from scripts import strategy_quote_diagnostics as diagnostics

FEED = 'webull_nasdaq_basic'
SYMBOL = re.compile(r'[A-Z][A-Z0-9.\-]{0,15}')
MAX_SYMBOLS = 4
MAX_RESPONSE_BYTES = 64 * 1024
MANAGE_SHA256 = '7c26244b97053637569daab0eccfab599c2f1b5c27408c39181813c71ddf6043'
CODES = frozenset({
    'QUOTE_MISSING', 'QUOTE_INVALID', 'QUOTE_STALE_OR_FUTURE',
    'QUOTE_HTTP_ERROR', 'QUOTE_FORBIDDEN', 'QUOTE_TRANSPORT_UNAVAILABLE',
    'QUOTE_BUDGET_EXHAUSTED', 'QUOTE_CLOCK_INVALID',
})


class QuoteError(ValueError):
    """Fixed diagnostic code only; never retain provider messages/headers."""
    def __init__(self, code):
        super().__init__(code if code in CODES else 'QUOTE_INVALID')


def need(value, code='QUOTE_INVALID'):
    if not value:
        raise QuoteError(code)


def instant(value):
    need(isinstance(value, datetime) and value.tzinfo is not None and
         value.utcoffset() is not None, 'QUOTE_CLOCK_INVALID')
    return value.astimezone(timezone.utc)


def positive(value):
    need(type(value) in {str, int, float, Decimal})
    need(not isinstance(value, str) or 0 < len(value) <= 64)
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise QuoteError('QUOTE_INVALID') from None
    need(result.is_finite() and result > 0)
    return result


def quote_time(value):
    """Webull quote_time is Unix milliseconds, never reception/last-trade time."""
    need(type(value) is int or (type(value) is str and
         re.fullmatch(r'[0-9]{13}', value) is not None))
    milliseconds = int(value)
    need(1_000_000_000_000 <= milliseconds < 10_000_000_000_000)
    try:
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=milliseconds)
    except (OverflowError, ValueError):
        raise QuoteError('QUOTE_INVALID') from None


def freshness(stamp, observed):
    age = (instant(observed) - stamp).total_seconds()
    need(-1 <= age <= 60, 'QUOTE_STALE_OR_FUTURE')


def normalize(payload, *, symbol, observed):
    """Accept exactly one depth-1 book with explicit provider symbol and time."""
    need(type(symbol) is str and SYMBOL.fullmatch(symbol) is not None)
    need(type(payload) is dict)
    need(payload.get('symbol') == symbol)
    stamp = quote_time(payload.get('quote_time'))
    freshness(stamp, observed)
    levels = []
    for side in ('bids', 'asks'):
        book = payload.get(side)
        need(type(book) is list and len(book) == 1, 'QUOTE_MISSING')
        need(type(book[0]) is dict)
        levels.append((positive(book[0].get('price')), positive(book[0].get('size'))))
    (bid, bid_size), (ask, ask_size) = levels
    need(bid <= ask)
    # The pinned native engine can cancel a protective stop before computing
    # this sell limit. Reject a nonrepresentable/zero result at the data boundary,
    # before any lifecycle effects, without changing its price policy.
    try:
        exit_price = (bid * Decimal('.995')).quantize(Decimal('.01'), rounding=ROUND_FLOOR)
        need(exit_price > 0 and math.isfinite(float(exit_price)) and math.isfinite(float(ask)))
    except (InvalidOperation, OverflowError):
        raise QuoteError('QUOTE_INVALID') from None
    return dict(bid=str(bid), ask=str(ask), timestamp=stamp.isoformat(),
                bid_size=str(bid_size), ask_size=str(ask_size),
                observed_at=instant(observed).isoformat(), provider='webull',
                feed=FEED, venue_scope='single_exchange', is_nbbo=False,
                venue_basis='configured_nasdaq_basic_entitlement',
                timestamp_basis='provider_quote_time_unix_ms')


class DepthTransport:
    """Use only an already initialized SDK MarketData.get_quotes interface.

    Setup is deliberately external: SDK DataClient construction can perform
    authentication/token writes. The bootstrap must disable retries/debug logs,
    bind production US endpoints and bounded timeouts, and qualify the SDK before
    passing its market_data interface. This class never creates those clients.
    The size check limits accepted content, not bytes buffered by the SDK/network.
    """
    def __init__(self, market_data):
        self._market_data = market_data

    def __call__(self, symbol):
        response = None
        try:
            response = self._market_data.get_quotes(symbol=symbol, category='US_STOCK',
                                                    depth=1, overnight_required=False)
            status = response.status_code
            if status != 200:
                raise QuoteError('QUOTE_FORBIDDEN' if status == 403 else 'QUOTE_HTTP_ERROR')
            need(type(response.content) is bytes and len(response.content) <= MAX_RESPONSE_BYTES)
            return response.json()
        except QuoteError:
            raise
        except Exception:
            raise QuoteError('QUOTE_TRANSPORT_UNAVAILABLE') from None
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass


class WebullQuotes:
    """Fail closed per symbol, with bounded calls and no retries or quote cache."""
    def __init__(self, fetch, *, non_display_authorized, now=None, max_requests=96):
        # This flag is a runtime gate, not proof of a license or user consent.
        need(non_display_authorized is True)
        need(callable(fetch) and type(max_requests) is int and 1 <= max_requests <= 96)
        self.fetch = fetch
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.max_requests = max_requests
        self.requests = 0
        self.failures = {}

    def quotes(self, symbols, *, now):
        self.failures = {}
        need(type(symbols) in {set, frozenset, list, tuple} and len(symbols) <= MAX_SYMBOLS)
        need(all(type(s) is str and SYMBOL.fullmatch(s) is not None for s in symbols))
        need(len(set(symbols)) == len(symbols))
        result = {}
        if not symbols:
            return result
        # Caller time cannot make an old quote fresh; use actual receipt time and
        # reject obvious local-clock drift. Recheck early symbols after the batch.
        requested = instant(now)
        start = instant(self.now())
        need(abs((start - requested).total_seconds()) <= 60, 'QUOTE_CLOCK_INVALID')
        previous = start
        for symbol in sorted(symbols):
            try:
                need(self.requests < self.max_requests, 'QUOTE_BUDGET_EXHAUSTED')
                self.requests += 1
                payload = self.fetch(symbol)
                observed = instant(self.now())
                need(observed >= previous, 'QUOTE_CLOCK_INVALID')
                previous = observed
                result[symbol] = normalize(payload, symbol=symbol, observed=observed)
            except QuoteError as error:
                self.failures[symbol] = error.args[0]
            except Exception:
                self.failures[symbol] = 'QUOTE_TRANSPORT_UNAVAILABLE'
        end = instant(self.now())
        if end < previous:
            self.failures = {symbol: 'QUOTE_CLOCK_INVALID' for symbol in symbols}
            return {}
        for symbol in list(result):
            try:
                freshness(datetime.fromisoformat(result[symbol]['timestamp']), end)
            except QuoteError as error:
                del result[symbol]
                self.failures[symbol] = error.args[0]
        return result


def safe_failures(value):
    if type(value) is not dict or len(value) > MAX_SYMBOLS:
        return {}
    return {symbol: code for symbol, code in value.items()
            if type(symbol) is str and SYMBOL.fullmatch(symbol) is not None and
            type(code) is str and code in CODES}


def execution_components(NativeBroker, NativeEngine, *, quote_factory):
    """Explicit monitor-local factory. Native trading/ownership methods untouched.

    The separately reviewed bootstrap owns transport setup and source/permission
    pins. Do not install this factory in an existing task without that rollout.
    """
    diagnostics._verify(NativeBroker, diagnostics.BROKER_METHOD_HASHES)
    diagnostics._verify(NativeEngine, diagnostics.ENGINE_METHOD_HASHES)
    # Price compatibility above is deliberately bound to this exact native
    # calculation/order sequence, not an assumed future implementation.
    diagnostics._verify(NativeEngine, {'manage': MANAGE_SHA256})
    need(callable(quote_factory))

    class WebullPaperBroker(NativeBroker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)  # Original paper/account context.
            self.quote_feed = FEED
            self.webull_quotes = None
            self.quote_failures = {}
            # Defer optional provider construction so loss of data entitlement or
            # provider setup cannot prevent existing stop/order reconciliation.

        def quotes(self, symbols, *, now):
            self.quote_failures = {}
            if not symbols:
                return {}
            try:
                if self.webull_quotes is None:
                    candidate = quote_factory()
                    need(isinstance(candidate, WebullQuotes))
                    self.webull_quotes = candidate
                result = self.webull_quotes.quotes(symbols, now=now)
                self.quote_failures = safe_failures(self.webull_quotes.failures)
                return result
            except Exception as error:
                failure = error.args[0] if isinstance(error, QuoteError) else 'QUOTE_TRANSPORT_UNAVAILABLE'
                self.quote_failures = {s: failure for s in symbols
                                       if type(s) is str and SYMBOL.fullmatch(s)}
                return {}

    class WebullEngine(NativeEngine):
        def note(self, signal, code, **details):
            if code == 'STRATEGY_QUOTES_UNAVAILABLE':
                failures = safe_failures(getattr(self.broker, 'quote_failures', None))
                details = dict(quote_feed=FEED, quote_failure_code='QUOTE_MISSING',
                               quote_failures=failures, quote_is_nbbo=False)
            return super().note(signal, code, **details)

    return WebullEngine, WebullPaperBroker
