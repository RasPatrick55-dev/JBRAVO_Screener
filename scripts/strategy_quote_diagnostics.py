"""Fixed quote observations over pinned native interfaces, with no I/O on import.

Only the supplied monitor's broker and event classes are derived. Native classes,
globals, decision methods, order methods and lifecycle source pins stay unchanged.
"""
import hashlib
import inspect
import textwrap


BROKER_METHOD_HASHES = {
    '__init__': '4d921b5c42cc971405199fda51588358324c58ce020c4af2f08aaa44ef67bdf2',
    'request': 'aa50ad1b914bce58de35c32202eb2ca7ae5136d8b135372f432ce11535b0e4a0',
    'quotes': '5896abeca1c8ba878f278924e9d1a44f238ac590088cba3422d81167e387a016',
}
ENGINE_METHOD_HASHES = {
    'note': '4848c31a57496891601396f13aa2c8ebfb5b1b01a63730369aa866b7e39e3c3e',
    'tick': '2145b985643d358ea75983c3afaf0fab8e61db6ecb4139f734e05d2540d2c38e',
}
CODES = frozenset({'QUOTE_FORBIDDEN', 'QUOTE_HTTP_ERROR', 'QUOTE_STALE_OR_FUTURE', 'QUOTE_MISSING', 'QUOTE_INVALID', 'QUOTE_TRANSPORT_UNAVAILABLE',
    'QUOTE_FAILURE_UNKNOWN'})


def _feed(value):
    return value if type(value) is str and value in {'sip', 'iex'} else 'unknown'


def quote_failure_details(error=None, *, feed=None, missing=False):
    """Return fixed fields only. Never serialize exceptions or provider text."""
    details = dict(quote_failure_code='QUOTE_MISSING' if missing else 'QUOTE_FAILURE_UNKNOWN',
                   quote_feed=_feed(feed))
    status = getattr(error, 'http_status', None)
    if type(status) is int and 100 <= status <= 599:
        details.update(quote_failure_code='QUOTE_FORBIDDEN' if status == 403 else 'QUOTE_HTTP_ERROR',
                       http_status=status)
    elif isinstance(error, (KeyError, TypeError)):
        details['quote_failure_code'] = 'QUOTE_INVALID'
    elif isinstance(error, ValueError) and len(error.args) == 1 and type(error.args[0]) is str:
        code = {
            'strategy_quote_response_unavailable': 'QUOTE_MISSING',
            'strategy_quote_stale_or_future': 'QUOTE_STALE_OR_FUTURE',
            'strategy_quote_symbol_changed': 'QUOTE_INVALID',
            'strategy_quote_crossed': 'QUOTE_INVALID',
            'strategy_broker_response_malformed': 'QUOTE_INVALID',
            'execution_ambiguous_timestamp': 'QUOTE_INVALID',
            'execution_invalid_timestamp': 'QUOTE_INVALID',
            'execution_invalid_number': 'QUOTE_INVALID',
            'strategy_broker_transport_outcome_unknown': 'QUOTE_TRANSPORT_UNAVAILABLE',
        }.get(error.args[0])
        if code:
            details['quote_failure_code'] = code
    return details


def safe_details(value, *, feed=None):
    """Validate copied fields again at the event boundary."""
    result = quote_failure_details(feed=feed, missing=True)
    if type(value) is not dict:
        return result
    code = value.get('quote_failure_code')
    if type(code) is str and code in CODES:
        result['quote_failure_code'] = code
    result['quote_feed'] = _feed(value.get('quote_feed', feed))
    status = value.get('http_status')
    if type(status) is int and 100 <= status <= 599:
        result['http_status'] = status
    return result


def _verify(cls, pins):
    for name, digest in pins.items():
        source = textwrap.dedent(inspect.getsource(getattr(cls, name))).strip()
        if hashlib.sha256(source.encode()).hexdigest() != digest:
            raise ValueError('quote_diagnostics_native_interface_changed')


def execution_components(NativeBroker, NativeEngine):
    """Create monitor-local subclasses. No instances, requests or global patches."""
    _verify(NativeBroker, BROKER_METHOD_HASHES)
    _verify(NativeEngine, ENGINE_METHOD_HASHES)

    class DiagnosticBroker(NativeBroker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.quote_failure = None

        def quotes(self, symbols, *, now):
            self.quote_failure = None
            try:
                result = super().quotes(symbols, now=now)
                if any(symbol not in result for symbol in symbols):
                    self.quote_failure = quote_failure_details(feed=self.quote_feed, missing=True)
                return result
            except (ValueError, KeyError, TypeError) as error:
                self.quote_failure = quote_failure_details(error, feed=self.quote_feed)
                raise

    class DiagnosticEngine(NativeEngine):
        def note(self, signal, code, **details):
            if code == 'STRATEGY_QUOTES_UNAVAILABLE':
                details = safe_details(getattr(self.broker, 'quote_failure', None),
                                       feed=getattr(self.broker, 'quote_feed', None))
            return super().note(signal, code, **details)

    return DiagnosticEngine, DiagnosticBroker
