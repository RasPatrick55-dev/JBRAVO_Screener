"""Standalone lifecycle monitor for the reviewed native paper execution release.

No import-time connections. Validation is file-only and never constructs an
Engine, broker or store. Activation requires separate deployment review.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone, time as day_time
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
NATIVE_SHA256 = {
    'rules': '515401c0b8d6652383cd87abf2967004905ca907858fc205213f8ab9d6e29cca',
    'runtime': 'bf392f82c25f175f38934b4de113836717cec62cf6fef0f67e697755bbdef9ad',
    'broker': 'bf6835a7721b0485b8a2ca6528cb8295193c4431c38542659ae8375abe45a931',
    'engine': '8de13a699757c7cbd67f3b5e85245001ff302be98d5c9e9ba565a55d645039f0',
    'store': '7519c0af48d2bed7050ed92144d47774e49567c42dccbb123f3ce594eb3d938e',
    'handoff': 'b4ec384c45a426a2a8ca51d5efc0b699d3cea2bb31c616fb8f1ec9f7bd166a89',
    'pipeline': '319470261384d6bf2aa3c380418bb77abb0e3efda361d74b17f7440a8f05e7f3',
}
STRATEGIES = (
    'trend_pullback_reclaim_qualifying_close_v2', 'bull_flag_breakout_v1',
    'ascending_triangle_breakout_v1', 'double_bottom_breakout_v1',
    'cup_handle_breakout_v1', 'inverse_head_shoulders_breakout_v1',
)
_CALENDAR_CACHE = {}
_COMPLETED_FRAME_CACHE = {}
COMPLETED_REASONS = frozenset({
    'strategy_postflight_stale_or_future', 'completed_report_not_latest_market_session',
    'completed_next_open_calendar_mismatch', 'completed_report_time_outside_bound',
    'completed_calendar_lookback_exhausted', 'completed_next_session_unavailable',
    'completed_inspection_scope_changed', 'completed_handoff_changed',
    'completed_archive_parent_changed', 'completed_strategy_definition_changed',
    'completed_held_symbol_data_unavailable', 'completed_frame_session_gap',
    'execution_bar_not_completed', 'execution_missing_exit_feature',
    'strategy_held_symbol_completed_data_unavailable', 'completed_position_not_reconciled',
})


def require(value, reason):
    if not value:
        raise ValueError(reason)


def completed_failure(error, phase):
    # Only exact fixed source codes, never provider/DB exception text.
    return dict(phase=phase, error_type=type(error).__name__,
                reason=str(error) if type(error) is ValueError and str(error) in COMPLETED_REASONS else 'COMPLETED_INPUT_CHECK_FAILED')


def verify_native(base_dir):
    for name, expected in NATIVE_SHA256.items():
        path = Path(base_dir) / ('scripts/strategy_pipeline.py' if name == 'pipeline' else
                                'scripts/strategy_execution.py' if name == 'rules' else
                                'scripts/strategy_execution_' + name + '.py')
        require(not path.is_symlink() and path.is_file(), 'native_source_missing')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                'native_interface_release_mismatch')


def enabled(kill_file):
    require(os.environ.get('JBR_STRATEGY_EXECUTION') == 'true' and
            os.environ.get('JBR_EXEC_PAPER') == 'true' and
            os.environ.get('APCA_API_BASE_URL') == 'https://paper-api.alpaca.markets' and
            not (Path(kill_file).exists() or Path(kill_file).is_symlink()), 'monitor_disabled')


def origin(signal, account_id):
    payload = signal['payload']
    # These markers never confer sell authority, even if malformed or stale.
    if ('execution_owner' in payload or 'ownership_history' in payload or
            'manual_paper_test' in payload.get('handoff', {})):
        return None
    proposal = payload['proposal']; handoff = payload['handoff']
    sid = proposal.get('strategy_id')
    require(sid in STRATEGIES and handoff['strategy_contract']['strategy_id'] == sid,
            'originating_strategy_identity_missing_or_changed')
    require(payload.get('account_id') == account_id and
            proposal['signal_id'] == signal['signal_id'] and
            proposal in handoff['proposals'], 'owned_signal_lineage_mismatch')
    if sid != STRATEGIES[0]:
        require(proposal.get('pattern_exit', {}).get('pattern') == sid,
                'originating_pattern_exit_missing_or_changed')
    return sid


def lifecycle_guard(runtime, signal, *, base_dir):
    """Validate owned exit code without coupling it to new dashboard inventory.

    The native append-only resolver still validates the entire compatibility
    history. Every required lifecycle pin and originating strategy definition
    must match exactly. Neither original handoff nor durable state is changed.
    """
    handoff = runtime.lifecycle_handoff(signal)
    pins = handoff['source_bindings']
    require(isinstance(pins, dict) and set(runtime.LIFECYCLE_COMPONENTS) <= set(pins),
            'strategy_lifecycle_sources_not_bound')
    for name in runtime.LIFECYCLE_COMPONENTS:
        current = runtime.health._binding(runtime.health._read(
            Path(base_dir) / name, runtime.health.MAX_SOURCE_BYTES))
        require(current == pins[name], 'strategy_lifecycle_source_changed')
    from scripts.strategy_library import get_strategy
    contract = handoff['strategy_contract']
    definition = get_strategy(contract['strategy_id']).describe()
    require(definition['id'] == contract['strategy_id'] and
            definition['version'] == contract['strategy_version'] and
            definition['definition_sha256'] == contract['definition_sha256'],
            'strategy_lifecycle_rule_version_changed')
    return True


def latest_completed_session(runtime, env, *, now):
    """Latest exchange session whose existing20:15NY daily boundary has passed."""
    local = now.astimezone(runtime.rules.NY)
    for offset in range(15):
        day = local.date() - timedelta(days=offset)
        session = calendar_session(env, day=day)
        ready = datetime.combine(day, day_time(20, 15), runtime.rules.NY)
        if session is not None and now >= ready:
            for forward in range(1, 15):
                following = day + timedelta(days=forward)
                next_session = calendar_session(env, day=following)
                if next_session is not None:
                    return day.isoformat(), following.isoformat(), next_session
            raise ValueError('completed_next_session_unavailable')
    raise ValueError('completed_calendar_lookback_exhausted')


def admit_completed_report(runtime, env, record, *, now):
    completed, following, calendar = latest_completed_session(runtime, env, now=now)
    session = record['session']
    require(session['signal_session'] == completed and session['next_session'] == following,
            'completed_report_not_latest_market_session')
    expected_open = datetime.combine(date.fromisoformat(following),
        day_time.fromisoformat(calendar['open']), runtime.rules.NY)
    require(runtime.rules.utc(session['next_open_at']) == expected_open,
            'completed_next_open_calendar_mismatch')
    finished = runtime.rules.utc(record['finished_at'])
    ready = datetime.combine(date.fromisoformat(completed), day_time(20, 15), runtime.rules.NY)
    require(ready <= finished <= now and (now - finished).total_seconds() <= 14 * 86400,
            'completed_report_time_outside_bound')
    return 14 * 86400


def isolated_handoff_reader(native, native_rules, record, *, runtime, env, now):
    """Private function globals change ONLY the two exact12h inspection gates.

    All native validation bodies execute unchanged with actual now. No shared
    module/function globals, new-entry validators or source files are patched.
    The caller must have admitted this exact record against the real calendar.
    """
    age_limit = admit_completed_report(runtime, env, record, now=now)
    pipeline = native.pipeline
    def bounded(call, *args, **kwargs):
        require(args and args[0] == record and kwargs.get('now') == now and
                kwargs.get('inspection') is True and kwargs.get('max_age_seconds') == 43200,
                'completed_inspection_scope_changed')
        return call(*args, **dict(kwargs, max_age_seconds=age_limit))
    facade = SimpleNamespace(STRATEGY=pipeline.STRATEGY,
        assess=lambda *a, **k: bounded(pipeline.assess, *a, **k),
        check_report=lambda *a, **k: bounded(pipeline.check_report, *a, **k))
    def copy_function(function, **overrides):
        clone = FunctionType(function.__code__, dict(function.__globals__, **overrides),
                             function.__name__, function.__defaults__, function.__closure__)
        clone.__kwdefaults__ = dict(function.__kwdefaults__ or {})
        return clone
    prepare = copy_function(native_rules.prepare_handoff, pipeline=facade)
    return copy_function(native.load_handoff, pipeline=facade, prepare_handoff=prepare)


def read_completed_handoff(runtime, env, *, base_dir, now, connection=None):
    from scripts import strategy_execution_handoff as native
    from scripts.research_store import connect
    path = Path(base_dir) / 'reports/pipeline_postflight/latest.json'
    raw = runtime.health._read(path, runtime.health.MAX_REPORT_BYTES)
    record = json.loads(raw, object_pairs_hook=runtime.health._strict_object,
                        parse_constant=runtime.health._invalid_constant)
    reader = isolated_handoff_reader(native, runtime.rules, record, runtime=runtime, env=env, now=now)
    owned = connection is None
    connection = connect(env=env) if owned else connection
    try:
        require(connection.readonly is True and connection.autocommit is False and
                connection.isolation_level == 2, 'completed_readonly_connection_required')
        handoff = reader(connection, base_dir=base_dir, now=now, inspection=True)
        require(runtime.health._read(path, runtime.health.MAX_REPORT_BYTES) == raw and
                handoff['session'] == record['session'] and handoff['inspection_mode'] is True and
                handoff['order_submission_authorized'] is False, 'completed_handoff_changed')
        return handoff
    finally:
        if owned:
            try: connection.rollback()
            finally: connection.close()


def completed_frames(runtime, env, handoff, *, base_dir, symbols, connection=None, cache=True):
    """Bounded archive SELECT and exact native dataset verification; no writes."""
    from scripts.research_store import connect
    from scripts import strategy_pipeline as pipeline
    from scripts.daily_strategy_scan import load_config
    from scripts.strategy_library import get_strategy
    wanted = set(symbols)
    require(0 < len(wanted) <= 4, 'completed_owned_symbol_bound')
    path = Path(base_dir) / 'reports/pipeline_postflight/latest.json'
    raw = runtime.health._read(path, runtime.health.MAX_REPORT_BYTES)
    record = json.loads(raw, object_pairs_hook=runtime.health._strict_object,
                        parse_constant=runtime.health._invalid_constant)
    parent_hash = handoff.get('companion_parent_report_sha256', handoff['primary_report_sha256'])
    require(runtime.rules.digest(record) == parent_hash and record['session'] == handoff['session'],
            'completed_archive_parent_changed')
    key = (parent_hash, runtime.rules.digest(handoff['strategy_contract']), tuple(sorted(wanted)))
    if cache and key in _COMPLETED_FRAME_CACHE: return _COMPLETED_FRAME_CACHE[key]
    owned = connection is None
    connection = connect(env=env) if owned else connection
    try:
        require(connection.readonly is True and connection.autocommit is False and
                connection.isolation_level == 2, 'completed_readonly_connection_required')
        with connection.cursor() as cursor:
            cursor.execute('SELECT specification_sha256,result_sha256,result FROM jbravo_research.research_runs '
                           'WHERE run_id=%s LIMIT 1', (record['batch']['archive_id'],))
            archived = cursor.fetchone()
        value = pipeline.validate_archive(archived, load_config(Path(base_dir) / 'config/strategy_scan_daily.json'),
                                          handoff['session'], base_dir=base_dir)
    finally:
        if owned:
            try: connection.rollback()
            finally: connection.close()
    dataset = pipeline.verified_dataset(value, archived[0], base_dir=base_dir)
    specification = json.loads(runtime.health._read(Path(value['output_root']) / 'scan/specification.json', 2 * 1024 * 1024))
    strategy = get_strategy(handoff['strategy_contract']['strategy_id'])
    definition = strategy.describe()
    require(handoff['strategy_contract'] == dict(strategy_id=definition['id'],
            strategy_version=definition['version'], definition_sha256=definition['definition_sha256']),
            'completed_strategy_definition_changed')
    require(wanted <= set(dataset.raw), 'completed_held_symbol_data_unavailable')
    fold = specification['folds'][0]
    if getattr(dataset, 'market_wide', False):
        frames = {symbol: dataset.prepare(symbol, strategy, fold) for symbol in wanted}
    else:
        frames = {symbol: strategy.prepare(dataset.raw[symbol], dataset.features[symbol], fold) for symbol in wanted}
    require(runtime.health._read(path, runtime.health.MAX_REPORT_BYTES) == raw, 'completed_archive_parent_changed')
    if cache:
        _COMPLETED_FRAME_CACHE.clear(); _COMPLETED_FRAME_CACHE[key] = frames
    return frames


def calendar_session(env, *, day):
    """Bounded GET-only SDK capability; at most32 cached date/account entries."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest
    import requests
    require(env.get('APCA_API_BASE_URL') == 'https://paper-api.alpaca.markets' and
            env.get('JBR_EXEC_PAPER') == 'true', 'calendar_paper_context_required')
    key = (day.isoformat(), env.get('JBR_EXEC_EXPECTED_PAPER_ACCOUNT_ID'))
    cached = _CALENDAR_CACHE.get(key)
    if cached is not None and time.monotonic() - cached[0] < 300:
        return cached[1]

    class CalendarSession(requests.Session):
        def request(self, method, url, **kwargs):
            require(method.upper() == 'GET' and
                    url == 'https://paper-api.alpaca.markets/v2/calendar' and
                    kwargs.get('params') == {'start': day.isoformat(), 'end': day.isoformat()},
                    'calendar_request_scope_invalid')
            kwargs.update(timeout=(5, 10), allow_redirects=False, stream=True)
            response = super().request(method, url, **kwargs)
            try:
                require(response.status_code == 200, 'calendar_response_unavailable')
                chunks = []; size = 0
                for chunk in response.iter_content(4096):
                    size += len(chunk); require(size <= 16384, 'calendar_response_oversize')
                    chunks.append(chunk)
                response._content = b''.join(chunks)
                response._content_consumed = True
                return response
            finally: response.close()

    client = TradingClient(env.get('APCA_API_KEY_ID'), env.get('APCA_API_SECRET_KEY'),
                           paper=True, raw_data=True)
    client._session.close(); client._session = CalendarSession()
    client._session.trust_env = False; client._retry = 0
    try: rows = client.get_calendar(GetCalendarRequest(start=day, end=day))
    finally: client._session.close()
    require(isinstance(rows, list) and len(rows) <= 1, 'calendar_population_invalid')
    session = None
    if rows:
        row = rows[0]
        require(row.get('date') == day.isoformat(), 'calendar_date_mismatch')
        opening = day_time.fromisoformat(row['open']); closing = day_time.fromisoformat(row['close'])
        require(opening.tzinfo is None and closing.tzinfo is None and
                day_time(4) < opening < closing <= day_time(20), 'calendar_hours_invalid')
        session = dict(date=day.isoformat(), open=opening.isoformat(), close=closing.isoformat())
    if len(_CALENDAR_CACHE) >= 32:
        _CALENDAR_CACHE.clear()
    _CALENDAR_CACHE[key] = (time.monotonic(), session)
    return session


def quote_window(runtime, clock, now, session):
    """Do not infer a trading session from a weekday or time of day alone."""
    require(abs((runtime.rules.utc(clock['timestamp']) - now).total_seconds()) <= 60,
            'strategy_monitor_clock_stale')
    local = now.astimezone(runtime.rules.NY)
    if session is None:
        require(clock['is_open'] is False, 'market_clock_session_conflict')
        return False, 'NONTRADING_SESSION_NATIVE_STOPS_ONLY'
    require(session['date'] == local.date().isoformat(), 'calendar_date_mismatch')
    opening = day_time.fromisoformat(session['open']); closing = day_time.fromisoformat(session['close'])
    current = local.time().replace(tzinfo=None)
    require(clock['is_open'] == (opening <= current < closing), 'market_clock_session_conflict')
    if not day_time(4) <= local.time().replace(tzinfo=None) < day_time(20):
        return False, 'OVERNIGHT_NATIVE_STOPS_ONLY'
    if clock['is_open'] is True:
        return True, 'REGULAR_SESSION_QUOTE_OBSERVATION'
    if current < opening:
        return True, 'PREMARKET_QUOTE_OBSERVATION'
    return True, 'AFTERHOURS_QUOTE_OBSERVATION'


class LifecycleBroker:
    """Only exact fresh durable lifecycle claims can authorize a mutation."""
    READS = {'clock', 'positions', 'open_orders', 'quotes', 'account',
             'find_by_client_order_id', 'get_order'}

    def __init__(self, broker, check):
        self._broker = broker; self.check = check; self.grant = None
        self.account_id = broker.account_id; self.paper = broker.paper

    def __getattr__(self, name):
        if name in self.READS:
            return getattr(self._broker, name)
        raise AttributeError(name)

    def authorize(self, operation, descriptor):
        self.check(); self.grant = (operation, descriptor)

    def consume(self, operation, order_id, request):
        self.check(); grant = self.grant; self.grant = None
        require(grant is not None and grant[0] == operation, 'unclaimed_broker_mutation')
        descriptor = grant[1]
        require(descriptor.get('order_id') == order_id, 'unowned_order_mutation')
        expected = descriptor.get('order')
        if operation == 'PATCH':
            expected = {k: v for k, v in expected.items() if k in {'qty', 'stop_price', 'client_order_id'}}
        require(request == expected, 'broker_claim_request_changed')

    def submit_order(self, request):
        require(request.get('side') == 'sell', 'monitor_buy_forbidden')
        self.consume('POST', None, request)
        return self._broker.submit_order(request)

    def replace_order(self, order_id, request):
        self.consume('PATCH', order_id, request)
        return self._broker.replace_order(order_id, request)

    def cancel_order(self, order_id):
        self.consume('DELETE', order_id, None)
        return self._broker.cancel_order(order_id)


class LifecycleStore:
    """Explicit store capability: no enroll, no entry claims, no excluded writes."""
    def __init__(self, store, allowed, broker, check):
        self._store = store; self.allowed = frozenset(allowed)
        self.account_id = store.account_id; self.broker = broker; self.check = check

    def signals(self, *, active=True):
        return [s for s in self._store.signals(active=active) if s['signal_id'] in self.allowed]

    def orders(self, signal_id):
        require(signal_id in self.allowed, 'excluded_signal_access')
        return self._store.orders(signal_id)

    def save(self, signal, **kwargs):
        self.check(); require(signal['signal_id'] in self.allowed, 'excluded_signal_write')
        return self._store.save(signal, **kwargs)

    def transaction(self):
        self.check(); return self._store.transaction()

    def claim(self, **kwargs):
        self.check(); sid = kwargs['signal_id']; role = kwargs['role']
        require(sid in self.allowed and role in {'stop', 'partial', 'exit', 'cancel'},
                'monitor_entry_or_excluded_claim_forbidden')
        operation = kwargs['operation']; descriptor = kwargs['request']
        if operation in {'PATCH', 'DELETE'}:
            owned = [r for r in self.orders(sid) if r.get('observation') and
                     r['observation']['id'] == descriptor.get('order_id')]
            require(len(owned) >= 1, 'unowned_order_mutation')
            if operation == 'PATCH':
                require(role == 'stop' and all(r['role'] == 'stop' for r in owned),
                        'monitor_replace_entry_forbidden')
        if operation != 'DELETE':
            require(descriptor['order'].get('side') == 'sell', 'monitor_buy_forbidden')
        claim = self._store.claim(**kwargs)
        if claim['fresh']:
            self.broker.authorize(operation, descriptor)
        return claim

    def _update(self, method, key, *args):
        self.check()
        require(any(r['client_order_id'] == key for s in self.signals(active=False)
                    for r in self.orders(s['signal_id'])), 'excluded_order_write')
        return getattr(self._store, method)(key, *args)

    def observed(self, key, order): return self._update('observed', key, order)
    def rejected(self, key): return self._update('rejected', key)
    def uncertain(self, key): return self._update('uncertain', key)


def preview_completed_replay(runtime, Engine, signal, frame, completed, following, env, now):
    """Validate an entire bounded catchup in memory before any daily-state write.

    Native completed() supplies the exact strategy semantics both for validation
    and application. Historical bars never produce synthetic broker fills.
    """
    import pandas as pd
    payload = signal['payload']; position = payload.get('position')
    if not position or position['entry_time'] > completed:
        return [], deepcopy(signal)
    last = payload.get('last_completed_session')
    if last == completed: return [], deepcopy(signal)
    require(last is None or last < completed, 'completed_session_regressed')
    start = date.fromisoformat(last) + timedelta(days=1) if last else date.fromisoformat(position['entry_time'])
    end = date.fromisoformat(completed)
    require(0 <= (end - start).days <= 30, 'completed_catchup_exceeds_bound')
    sessions = []
    for offset in range((end - start).days + 1):
        day = start + timedelta(days=offset)
        if calendar_session(env, day=day) is not None: sessions.append(day.isoformat())
    require(sessions and sessions[-1] == completed, 'completed_calendar_session_unverified')
    require(not frame.index.has_duplicates and frame.index.is_monotonic_increasing and
            frame.index[-1] == pd.Timestamp(completed), 'completed_frame_population_invalid')
    observed = [stamp.date().isoformat() for stamp in frame.index
                if start <= stamp.date() <= end]
    require(observed == sessions, 'completed_frame_session_gap')

    class PreviewStore:
        def save(self, candidate, *, payload=None, state=None):
            if payload is not None: candidate['payload'] = deepcopy(payload)
            if state is not None: candidate['state'] = state
            return candidate

    preview = SimpleNamespace(store=PreviewStore())
    candidate = deepcopy(signal); replay = []
    for index, session in enumerate(sessions):
        next_session = sessions[index + 1] if index + 1 < len(sessions) else following
        sliced = frame.loc[:pd.Timestamp(session)].copy()
        Engine.completed(preview, candidate, sliced, session=session, next_session=next_session, now=now)
        replay.append((session, next_session, sliced))
    return replay, candidate


def quote_safe_engine(Engine):
    """Keep native stop maintenance reachable behind a queued no-quote exit.

    Native strategy decisions, queued intent and execution guards are unchanged.
    This copies only the pinned engine's existing stop-maintenance branch, using
    its own numeric/terminal helpers and durable _perform path. No quote exits,
    entry, cancellation, synthetic fill or intent rewrite is added.
    """
    core = Engine.manage.__globals__
    rules = core['rules']; quantity = core['quantity']; terminal = core['TERMINAL']
    Decimal = core['Decimal']; ROUND_FLOOR = core['ROUND_FLOOR']

    class QuoteSafeEngine(Engine):
        def manage(self, signal, records, *, now, quote, regular_open):
            payload = signal['payload']; position = payload.get('position')
            rejected = sum(r['role'] == 'stop' and r['state'] == 'REJECTED' for r in records)
            # Match pinned native partial classification exactly: queued, not
            # protective-stop breached, PARTIAL_5PCT only, and more than1share.
            # Do not cancel existing protection for a partial if native's future
            # missing-stop policy has already exhausted its three-rejection cap.
            competing = any(r['role'] in {'exit','partial'} and r.get('observation') and
                            r['observation']['status'] not in terminal for r in records)
            if position and quote is not None and rejected >= 3 and not competing:
                target = max(position['trailing_stop'] or 0, position['atr_stop'] or 0)
                breached = rules.number(quote['bid'], positive=True) <= Decimal(str(target))
                local = rules.utc(now).astimezone(rules.NY)
                queued = bool(payload['pending_exit_reasons']) and local.date().isoformat() >= payload['pending_exit_session']
                if queued and not breached and payload['pending_exit_reasons'] == ['PARTIAL_5PCT'] and quantity(position['qty']) > 1:
                    self.note(signal, 'STOP_REJECTIONS_REQUIRE_REVIEW'); return
            before = len(self.events)
            super().manage(signal, records, now=now, quote=quote, regular_open=regular_open)
            if quote is not None or len(self.events) == before or self.events[-1]['code'] != 'EXIT_QUOTE_UNAVAILABLE':
                return
            require(self.pipeline_orders is True, 'queued_stop_pipeline_required')
            if any(r['state'] in {'CLAIMED','UNKNOWN'} for r in records):
                self.note(signal, 'ORDER_OUTCOME_UNKNOWN_NO_RESUBMISSION'); return
            position = signal['payload'].get('position')
            require(position is not None and bool(signal['payload']['pending_exit_reasons']), 'queued_stop_intent_required')
            own_qty = quantity(position['qty']); symbol = position['symbol']
            active = [r for r in records if r['role'] in {'stop','exit','partial'} and
                      r.get('observation') and r['observation']['status'] not in terminal]
            require(not any(r['role'] in {'exit','partial'} for r in active), 'queued_stop_competing_exit')
            stops = [r for r in active if r['role'] == 'stop']
            require(len(stops) <= 1, 'strategy_multiple_active_stops')
            target = max(position['trailing_stop'] or 0, position['atr_stop'] or 0)
            desired = Decimal(str(target)).quantize(Decimal('.01'), rounding=ROUND_FLOOR)
            rejected_stops = [r for r in records if r['role'] == 'stop' and r['state'] == 'REJECTED']
            require(all(r.get('operation') in {'POST','PATCH'} for r in rejected_stops), 'stop_rejection_operation_unknown')
            rejected_patches = [r for r in rejected_stops if r['operation'] == 'PATCH']
            require(all(isinstance(r.get('request'), dict) and isinstance(r['request'].get('order_id'), str)
                        and bool(r['request']['order_id']) for r in rejected_patches), 'stop_rejection_target_unknown')
            if stops:
                old = stops[0]['observation']
                remaining = quantity(old['qty']) - quantity(old.get('filled_qty', 0))
                if old['status'] in {'pending_new','pending_cancel','pending_replace','held'}:
                    self.note(signal, 'STOP_PENDING_CONFIRMATION'); return
                if quantity(old.get('filled_qty', 0)):
                    self.note(signal, 'PARTIAL_STOP_FILL_PENDING'); return
                old_price = rules.number(old['stop_price'], positive=True)
                if desired <= old_price and remaining == own_qty:
                    self.note(signal, 'REGULAR_STOP_CONFIRMED_NOT_EXTENDED_HOURS' if not regular_open else 'REGULAR_STOP_OBSERVED')
                    return
                # Reserve one attempt under native's three-rejection missing-stop
                # policy, and never repeat a rejected PATCH of this live stop.
                # Keep all historical POST/PATCH outcomes unchanged and visible.

                if rejected >= 2 or any(r['request']['order_id'] == old['id'] for r in rejected_patches):
                    self.note(signal, 'STOP_REJECTIONS_REQUIRE_REVIEW'); return
                sequence = max(r['sequence'] for r in records if r['role'] == 'stop') + 1
                self._perform(signal, 'stop', sequence, dict(symbol=symbol, qty=own_qty, side='sell', type='stop',
                    stop_price=float(max(old_price, desired)), time_in_force='gtc', extended_hours=False),
                    operation='PATCH', order_id=old['id'])
                return
            if rejected >= 2:
                self.note(signal, 'STOP_REJECTIONS_REQUIRE_REVIEW'); return
            sequence = 1 + max((r['sequence'] for r in records if r['role'] == 'stop'), default=-1)
            require(desired > 0, 'strategy_protective_stop_invalid')
            self._perform(signal, 'stop', sequence, dict(symbol=symbol, qty=own_qty, side='sell', type='stop',
                stop_price=float(desired), time_in_force='gtc', extended_hours=False))
    return QuoteSafeEngine


def manage_owned(runtime, Engine, store, raw_broker, env, *, base_dir, kill_file, now, observe_now=None):
    """All calls occur inside source and account leases held by cycle()."""
    def check():
        enabled(kill_file)
        runtime.entry_account(raw_broker, env)

    observe = observe_now or (lambda: datetime.now(timezone.utc))
    def fresh_window():
        observed = observe()
        try:
            clock = raw_broker.clock()
            session = calendar_session(env, day=observed.astimezone(runtime.rules.NY).date())
            enabled_prices, status = quote_window(runtime, clock, observed, session)
            return observed, enabled_prices, status, clock['is_open']
        except Exception:
            return observed, False, 'SESSION_UNVERIFIED_NATIVE_STOPS_ONLY', False

    def current_completed(completed, following):
        current = latest_completed_session(runtime, env, now=observe())
        require(current[:2] == (completed, following), 'completed_report_not_latest_market_session')

    check()
    groups = {}; excluded = []; blocked = []
    for signal in store.signals():
        try:
            sid = origin(signal, raw_broker.account_id)
            if sid is None:
                excluded.append(signal['signal_id']); continue
            lifecycle_guard(runtime, signal, base_dir=base_dir)
            groups.setdefault(sid, []).append(signal['signal_id'])
        except (KeyError, ValueError, TypeError):
            blocked.append(signal['signal_id'])
    broker = LifecycleBroker(raw_broker, check)
    events = []; protections = []; owned = []; completed_unavailable = False
    _, observe_prices, session_status, regular = fresh_window()
    # Never let adopted/manual positions enter an Engine, including malformed ones.
    recognized = runtime.EntryRecognitionStore(store, excluded + blocked)
    for sid, ids in groups.items():
        scoped = LifecycleStore(recognized, ids, broker, check)
        def guard(kind, signal):
            check(); require(kind == 'lifecycle', 'monitor_entry_forbidden')
            require(origin(signal, broker.account_id) == sid, 'origin_changed')
            return lifecycle_guard(runtime, signal, base_dir=base_dir)
        engine = quote_safe_engine(Engine)(scoped, broker, guard=guard, pipeline_orders=True, now=observe)
        frames = None; completed = None; following = None
        phase = 'HANDOFF'
        try:
            handoff = read_completed_handoff(runtime, env, base_dir=base_dir, now=observe())
            if sid != STRATEGIES[0]:
                handoff = runtime.companion_handoff(env, handoff, base_dir=base_dir,
                                                    inspection=True, strategy_id=sid)
            require(handoff['strategy_contract']['strategy_id'] == sid, 'completed_origin_mismatch')
            require(all(s['payload']['handoff']['strategy_contract'] == handoff['strategy_contract']
                        for s in scoped.signals()), 'completed_strategy_definition_changed')
            completed = handoff['session']['signal_session']; following = handoff['session']['next_session']
            held = [s for s in scoped.signals() if s['payload'].get('position')]
            if any(s['payload']['position']['entry_time'] <= completed and
                   s['payload'].get('last_completed_session') != completed for s in held):
                phase = 'FEATURES'
                frames = completed_frames(runtime, env, handoff, base_dir=base_dir,
                    symbols={s['payload']['proposal']['symbol'] for s in held})
                require(all(s['payload']['proposal']['symbol'] in frames for s in held),
                        'completed_strategy_frames_missing')
        except Exception as error:
            frames = None; completed = None; following = None
            completed_unavailable = True
            engine.note(None, 'COMPLETED_BAR_INPUT_NOT_AVAILABLE', **completed_failure(error, phase))
        # Reconcile broker fills and regular-hours stops first, without quote
        # exits. Validate every missed completed day before applying any daily
        # update. A bad/unready frame cannot abort existing-stop reconciliation.
        engine.tick(now=observe(), observe_prices=False)
        if frames is not None:
            for signal in scoped.signals():
                try:
                    current_completed(completed, following)
                    require(engine.sync(signal, now=observe()) is not None,
                            'completed_position_not_reconciled')
                    replay, preview = preview_completed_replay(runtime, Engine, signal,
                        frames[signal['payload']['proposal']['symbol']], completed, following, env, observe())
                    for session, next_session, frame in replay:
                        current_completed(completed, following)
                        engine.completed(signal, frame, session=session, next_session=next_session, now=observe())
                except Exception as error:
                    completed_unavailable = True
                    engine.note(signal, 'COMPLETED_BAR_INPUT_NOT_AVAILABLE', **completed_failure(error, 'REPLAY'))
        tick_now, observe_prices, session_status, regular = fresh_window()
        if session_status == 'SESSION_UNVERIFIED_NATIVE_STOPS_ONLY': completed_unavailable = True
        result = engine.tick(now=tick_now, observe_prices=observe_prices)
        events.extend(engine.events); owned.extend(result['owned_symbols'])
        protections.append(runtime.entry_protection_state(scoped))
    _, observe_prices, session_status, regular = fresh_window()
    positions = [p for protection in protections for p in protection['positions']]
    confirmed = all(p['confirmed'] for p in protections)
    benign = {'REGULAR_STOP_CONFIRMED_NOT_EXTENDED_HOURS', 'REGULAR_STOP_OBSERVED'}
    failed = bool(blocked or completed_unavailable or not confirmed or
                  (groups and session_status == 'SESSION_UNVERIFIED_NATIVE_STOPS_ONLY') or
                  any(e['code'] not in benign for e in events))
    return dict(status='blocked' if failed else 'ok', paper_only=True, entry_ready=False,
        entry_block_reason='STANDALONE_LIFECYCLE_ONLY', owned_symbols=sorted(set(owned)),
        excluded_signal_count=len(excluded), blocked_signal_count=len(blocked),
        strategy_signal_counts={sid: len(ids) for sid, ids in groups.items()},
        session_status=session_status, quote_observation_enabled=observe_prices,
        native_covered=sum(p['regular_hours_stop_confirmed'] for p in positions) if regular else 0,
        regular_hours_stop_confirmed_count=sum(p['regular_hours_stop_confirmed'] for p in positions),
        extended_hours_native_protection=False, native_stops_trigger_only_regular_hours=True,
        protection=protections, events=runtime.compact_events(events),
        finished_at=datetime.now(timezone.utc).isoformat())


def cycle(*, base_dir=ROOT, kill_file=None):
    from scripts import strategy_execution_runtime as runtime
    from scripts.strategy_execution_engine import Engine
    from scripts.strategy_execution_broker import PaperBroker
    kill_file = kill_file or Path(base_dir) / 'reports/strategy_execution/monitor/STOP'
    broker = None; conn = None
    try:
        with runtime.source_lease(base_dir):
            verify_native(base_dir); enabled(kill_file)
            runtime.load_config(base_dir); env = runtime.environment()
            runtime.expected_paper_account(env)
            broker = PaperBroker(env, max_requests=192)
            runtime.entry_account(broker, env)
            conn, store = runtime.store_connection(env, broker)
            with store.lease():
                value = manage_owned(runtime, Engine, store, broker, env, base_dir=base_dir,
                                     kill_file=kill_file, now=datetime.now(timezone.utc))
        runtime.write_report(value, role='monitor', base_dir=base_dir)
        return value
    except Exception as error:
        value = dict(status='blocked', paper_only=True, entry_ready=False,
                     exit_reason='STRATEGY_ONLY_MONITOR_UNAVAILABLE', error_type=type(error).__name__,
                     native_covered=0, finished_at=datetime.now(timezone.utc).isoformat())
        runtime.write_report(value, role='monitor', base_dir=base_dir)
        return value
    finally:
        if conn is not None:
            try: conn.rollback()
            finally: conn.close()
        if broker is not None: broker.close()


@contextmanager
def singleton(path):
    # Native Linux advisory lock released by the OS on death; never unlink lock.
    import fcntl
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally: os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--validate', action='store_true', help='File-only validation; no writes, DB or broker')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--base-dir', type=Path, default=ROOT,
                        help='Absolute native checkout, required for isolated deployment')
    args = parser.parse_args(argv)
    base_dir = args.base_dir
    require(base_dir.is_absolute() and base_dir.is_dir() and not base_dir.is_symlink(),
            'native_base_directory_invalid')
    require(5 <= args.interval <= 60, 'poll_interval_outside_bounds')
    if args.validate:
        verify_native(base_dir)
        print(json.dumps(dict(status='source_compatible', broker_operations=0, database_operations=0,
                              mutations=0, live_readiness='NOT_CHECKED')))
        return 0
    sys.path.insert(0, str(base_dir))
    directory = base_dir / 'reports/strategy_execution/monitor'
    # Directory must be provisioned by reviewed deployment, not silently created.
    require(directory.is_dir() and not directory.is_symlink(), 'monitor_directory_missing')
    with singleton(directory / 'strategy-only.lock'):
        while True:
            value = cycle(base_dir=base_dir)
            print(json.dumps({k: value[k] for k in ('status', 'finished_at')}), flush=True)
            if args.once: return 0 if value['status'] == 'ok' else 1
            if (directory / 'STOP').exists(): return 1
            time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
