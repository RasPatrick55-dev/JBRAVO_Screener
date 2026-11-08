"""Digest-bound, monitor-owned entry lifetime; inert without supplied ports.

The submission ticket and native handoff remain immutable. Calendar provenance
must be verified by the caller's GET-only exchange-calendar adapter. This module
checks its structure and agreement with the handoff, not exchange holidays.
"""
from __future__ import annotations

import ast
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import inspect
import json
import textwrap
from zoneinfo import ZoneInfo

SCHEMA = 'jbravo.monitor-owned-entry-lifecycle.v1'
MODEL = 'PREMARKET_0400_TO_EXCHANGE_REGULAR_CLOSE_V1'
OWNER = 'MONITOR_POSITIONS'
NY = ZoneInfo('America/New_York')
_METHOD_HASHES = {
    'sync': '7bfb168c8cc26f22a5a2cf2fc71845901c1258f9b73eb8f0d37be514b0679cfc',
    'tick': '2145b985643d358ea75983c3afaf0fab8e61db6ecb4139f734e05d2540d2c38e',
}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    _require(isinstance(value, datetime) and value.tzinfo is not None and
             value.utcoffset() is not None, 'entry_lifecycle_timestamp_invalid')
    return value.astimezone(timezone.utc)


def marker_for(handoff, calendar_session):
    """Return a fresh JSON marker, bound to the complete unchanged handoff.

    No enrollment, ticket extension, calendar request, or broker call is made.
    Store the returned marker in payload['entry_lifecycle'] before any POST.
    """
    try:
        _require(isinstance(handoff, dict) and isinstance(calendar_session, dict),
                 'entry_lifecycle_context_missing')
        _require(handoff['handoff_id'] == _digest({k: v for k, v in handoff.items()
                                                  if k != 'handoff_id'}),
                 'entry_lifecycle_handoff_digest_changed')
        session = handoff['session']
        day = date.fromisoformat(calendar_session['date'])
        _require(day.isoformat() == calendar_session['date'] == session['next_session']
                 and day.weekday() < 5, 'entry_lifecycle_session_mismatch')
        opening = time.fromisoformat(calendar_session['open'])
        closing = time.fromisoformat(calendar_session['close'])
        _require(opening.tzinfo is None and closing.tzinfo is None and
                 opening == time(9, 30) and opening < closing <= time(16),
                 'entry_lifecycle_calendar_hours_invalid')
        regular_open = datetime.combine(day, opening, NY).astimezone(timezone.utc)
        close = datetime.combine(day, closing, NY).astimezone(timezone.utc)
        premarket = datetime.combine(day, time(4), NY).astimezone(timezone.utc)
        policy = handoff['execution_policy']
        _require(_utc(policy['entry_open_at']) == premarket and
                 _utc(session['next_open_at']) == regular_open and
                 _utc(policy['cancel_at']) == premarket + timedelta(minutes=35),
                 'entry_lifecycle_handoff_window_mismatch')
        marker = dict(schema=SCHEMA, model=MODEL, owner=OWNER,
                      handoff_id=handoff['handoff_id'], handoff_sha256=_digest(handoff),
                      session_date=day.isoformat(), entry_open_at=premarket.isoformat(),
                      regular_open_at=regular_open.isoformat(), cancel_at=close.isoformat(),
                      calendar_session=dict(date=day.isoformat(), open=opening.isoformat(),
                                            close=closing.isoformat()))
        marker['marker_sha256'] = _digest(marker)
        return marker
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('entry_lifecycle_context_invalid') from error


def cancellation_deadline(payload, *, require_marker=False):
    """Validate marker on every use; absent legacy markers retain native policy.

    New-entry callers must pass require_marker=True when absence is corruption.
    A present null/malformed marker NEVER falls back to the legacy 35 minutes.
    """
    try:
        if 'entry_lifecycle' not in payload:
            _require(not require_marker, 'entry_lifecycle_marker_missing')
            return _utc(payload['handoff']['execution_policy']['cancel_at'])
        marker = payload['entry_lifecycle']
        _require(isinstance(marker, dict), 'entry_lifecycle_marker_invalid')
        expected = marker_for(payload['handoff'], marker['calendar_session'])
        _require(marker == expected, 'entry_lifecycle_marker_changed')
        return _utc(marker['cancel_at'])
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('entry_lifecycle_marker_invalid') from error


def monitor_engine(Engine, *, require_marker=False):
    """Derive pinned native Engine, replacing ONLY two deadline expressions.

    Exact source pins fail closed on host drift. Compile copies in private
    globals, never patch native functions/globals or rewrite stored handoffs.
    Everything else (fill quantities, readback, risk exits, stops) stays native.
    """
    methods = {}
    for name, digest in _METHOD_HASHES.items():
        native = getattr(Engine, name)
        source = textwrap.dedent(inspect.getsource(native)).strip()
        _require(hashlib.sha256(source.encode()).hexdigest() == digest,
                 'entry_lifecycle_native_' + name + '_source_changed')
        tree = ast.parse(source)
        expression = ("payload['handoff']['execution_policy']['cancel_at']" if name == 'sync'
                      else "signal['payload']['handoff']['execution_policy']['cancel_at']")
        target = ast.dump(ast.parse(expression, mode='eval').body)
        argument = 'payload' if name == 'sync' else "signal['payload']"

        class Rewrite(ast.NodeTransformer):
            count = 0

            def visit_Subscript(self, node):
                if ast.dump(node) == target:
                    self.count += 1
                    return ast.copy_location(ast.parse('_lifecycle_deadline(' + argument + ')',
                                                       mode='eval').body, node)
                return self.generic_visit(node)

        rewrite = Rewrite()
        tree = rewrite.visit(tree)
        _require(rewrite.count == 1, 'entry_lifecycle_native_deadline_count_changed')
        namespace = dict(native.__globals__)
        namespace['_lifecycle_deadline'] = lambda payload: cancellation_deadline(
            payload, require_marker=require_marker)
        exec(compile(ast.fix_missing_locations(tree), '<monitor-entry-lifecycle>', 'exec'), namespace)
        methods[name] = namespace[name]
    copied_sync = methods['sync']

    def sync(self, signal, *, now):
        # Validate even for already-filled entries, before any lifecycle I/O.
        cancellation_deadline(signal['payload'], require_marker=require_marker)
        return copied_sync(self, signal, now=now)

    methods['sync'] = sync
    methods['__module__'] = __name__
    return type('MonitorOwnedEntryEngine', (Engine,), methods)
