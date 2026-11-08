"""Offline ownership boundary tests; these never contact an account."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.strategy_entry_only import EntryOnlyBroker, EntryOnlyStore, entry_engine, run_entry_only

pytestmark = pytest.mark.alpaca_optional

NOW = datetime(2026, 11, 9, 9, 0, 2, tzinfo=timezone.utc)


class Store:
    account_id = 'paper'

    def __init__(self):
        self.rows = {}; self.outbox = {}; self.locked = False

    def signals(self, active=True):
        return deepcopy(list(self.rows.values()))

    def orders(self, sid):
        return deepcopy([r for r in self.outbox.values() if r['signal_id'] == sid])

    def enroll(self, handoff, proposal, reference):
        sid = proposal['signal_id']
        if sid not in self.rows:
            self.rows[sid] = dict(signal_id=sid, state='PREPARED', revision=0,
                payload=dict(handoff=deepcopy(handoff), proposal=proposal, reference=reference))
        return deepcopy(self.rows[sid])

    def save(self, signal, *, payload=None, state=None):
        signal.update(revision=signal['revision'] + 1,
                      payload=payload or signal['payload'], state=state or signal['state'])
        self.rows[signal['signal_id']] = deepcopy(signal)

    def claim(self, **kwargs):
        self.outbox[kwargs['signal_id']] = dict(kwargs, state='CLAIMED', observation=None,
                                              client_order_id=kwargs['signal_id'])


class Broker:
    paper = True; account_id = 'paper'

    def __init__(self):
        self.posts = []; self.closed = False

    def submit_order(self, request):
        self.posts.append(deepcopy(request))
        return dict(request, status='new', id=request['client_order_id'], filled_qty='0')

    def positions(self): return {}
    def open_orders(self): return []
    def account(self): return {'cash': '100000'}
    def clock(self): return {'timestamp': NOW.isoformat()}
    def close(self): self.closed = True


class NativeEngine:
    def __init__(self, store, broker, *, guard, pipeline_orders, now):
        self.store, self.broker, self.guard, self.now = store, broker, guard, now
        self.events = []

    def enter(self, handoff, references, *, selected_signal_ids, **kwargs):
        sid = selected_signal_ids[0]
        proposal = next(p for p in handoff['proposals'] if p['signal_id'] == sid)
        signal = self.store.enroll(handoff, proposal, references[proposal['symbol']])
        assert 'entry_lifecycle' in signal['payload']
        assert self.guard('entry', signal)
        request = dict(symbol=proposal['symbol'], side='buy', type='limit', time_in_force='day',
                       extended_hours=True, qty=1, limit_price=12.34, client_order_id=sid)
        self.store.claim(signal_id=sid, role='entry', operation='POST', request=request)
        order = self.broker.submit_order(request)
        self.store.store.outbox[sid].update(state='OBSERVED', observation=order)
        self.store.save(signal, state='ENTRY_PENDING')
        self.tick(now=self.now())

    def tick(self, **kwargs):
        raise AssertionError('entry cannot run native lifecycle tick')


def fixture():
    store = Store(); broker = Broker(); reports = []; checks = []
    proposal = dict(signal_id='a' * 64, symbol='AAA')
    handoff = dict(handoff_id='h', source_bindings={}, primary_report_sha256='p',
        session={'next_session': '2026-11-09'}, proposals=[proposal],
        pipeline_limit_order={'limit_price': 12.34},
        execution_policy={'entry_open_at': '2026-11-09T09:00:00+00:00'})
    verified = dict(record={'selection': {}, 'run_id': 'run'}, handoff=handoff)
    @contextmanager
    def lease(*args, **kwargs):
        assert not store.locked
        store.locked = True
        try: yield
        finally: store.locked = False
    @contextmanager
    def source(*args): yield
    def report(value, **kwargs):
        assert not store.locked
        reports.append(value)
    def preflight(*args, exempt_signal_ids=(), **kwargs):
        assert store.locked
        checks.append(set(exempt_signal_ids))
        return {'status': 'pass'}
    conn = SimpleNamespace(rollback=lambda: None, close=lambda: None)
    runtime = SimpleNamespace(load_config=lambda *a: None, environment=lambda: {},
        source_lease=source, readonly_handoff=lambda *a, **k: handoff,
        source_guard=lambda *a, **k: True, unified_handoff=lambda *a, **k: verified,
        pipeline_execution_queue=lambda v: [dict(proposal=p, handoff=handoff) for p in handoff['proposals']],
        rules=SimpleNamespace(utc=datetime.fromisoformat), expected_paper_account=lambda env: 'paper',
        entry_account=lambda b, e: b.account(), store_connection=lambda *a: (conn, store),
        entry_lease=lease, guard_for=lambda *a, **k: lambda *a: True,
        available_entry_budget=lambda *a: 3, compact_events=lambda e: e, write_report=report)
    config = SimpleNamespace(paper_research_entries=True, source='db', reconcile_auto=True,
        reconcile_only=False, diagnostic=False, allocation_pct=.05, max_positions=4,
        max_new_positions=3, disable_open_position_cap=False, min_order_usd=0,
        allow_bump_to_one=False, allow_fractional=False, submit_at_ny='04:00', extended_hours=True,
        limit_buffer_pct=0, cancel_after_min=35, chase_enabled=False, dry_run=False)
    calendar = {'date': '2026-11-09', 'open': '09:30:00', 'close': '16:00:00'}
    marker = {'owner': 'MONITOR_POSITIONS', 'cancel_at': '2026-11-09T21:00:00+00:00'}
    args = dict(runtime=runtime, NativeEngine=NativeEngine,
        calendar_reader=lambda *a, **k: calendar, marker_for=lambda *a: deepcopy(marker),
        preflight=preflight, base_dir='unused', progress={}, now=lambda: NOW,
        broker_factory=lambda *a, **k: broker)
    return config, args, store, broker, reports, checks, handoff


def test_entry_batch_has_no_lifecycle_tick_and_releases_lease_before_report():
    config, args, store, broker, reports, checks, handoff = fixture()
    original = deepcopy(handoff)
    value = run_entry_only(config, **args)
    assert len(broker.posts) == 1 and broker.posts[0]['limit_price'] == 12.34
    assert handoff == original
    assert not store.locked and broker.closed
    assert value['status'] == 'ok' and value['orders_submitted'] == 1
    assert value['cancellation_deadline'] == '2026-11-09T21:00:00+00:00'
    assert value['protection_confirmed'] is False
    assert value['protection_status'] == 'MONITOR_RECONCILIATION_REQUIRED'
    assert checks == [set(), set()]


def test_new_batch_exemption_is_only_newly_enrolled_signals():
    config, args, store, broker, reports, checks, handoff = fixture()
    handoff['proposals'].append({'signal_id': 'b' * 64, 'symbol': 'BBB'})
    run_entry_only(config, **args)
    assert checks == [set(), set(), {'a' * 64}]
    assert len(broker.posts) == 2


def test_batch_preflight_uses_copied_queued_origin_policy():
    config, args, store, broker, reports, checks, handoff = fixture()
    handoff['execution_policy']['model'] = 'legacy'
    handoff['proposals'].append({'signal_id': 'b' * 64, 'symbol': 'BBB'})
    def queue(verified):
        result = []
        for proposal in handoff['proposals']:
            origin = deepcopy(handoff)
            origin['execution_policy']['model'] = 'PIPELINE_LIMIT_MODEL'
            result.append(dict(proposal=proposal, handoff=origin))
        return result
    args['runtime'].pipeline_execution_queue = queue
    def proof(*a, handoff, exempt_signal_ids=(), **kw):
        for sid in exempt_signal_ids:
            assert store.rows[sid]['payload']['handoff']['execution_policy'] == handoff['execution_policy']
        return {'status': 'pass'}
    args['preflight'] = proof
    run_entry_only(config, **args)
    assert len(broker.posts) == 2
    assert handoff['execution_policy']['model'] == 'legacy'


def test_failed_preflight_sends_nothing_and_releases_lease():
    config, args, store, broker, *rest = fixture()
    def fail(*a, **k): raise ValueError('unsafe_pending_replacement')
    args['preflight'] = fail
    with pytest.raises(ValueError, match='pending_replacement'):
        run_entry_only(config, **args)
    assert not broker.posts and not store.rows and not store.locked and broker.closed
    assert config._entry_cancellation_policy == 'monitor_exchange_regular_close'


def test_late_batch_failure_preserves_new_order_receipt_policy():
    config, args, store, broker, reports, checks, handoff = fixture()
    handoff['proposals'].append({'signal_id': 'b' * 64, 'symbol': 'BBB'})
    calls = []
    def preflight(*a, **kw):
        calls.append(1)
        if len(calls) == 3:
            raise ValueError('second_symbol_guard_failed')
        return {'status': 'pass'}
    args['preflight'] = preflight
    with pytest.raises(ValueError, match='second_symbol'):
        run_entry_only(config, **args)
    assert len(broker.posts) == 1 and not store.locked
    assert config._entry_cancellation_policy == 'monitor_exchange_regular_close'


@pytest.mark.parametrize('delta', [-3, 58, 2100])
def test_outside_open_window_cannot_construct_broker(delta):
    config, args, store, broker, *rest = fixture()
    args['now'] = lambda: NOW + timedelta(seconds=delta)
    args['broker_factory'] = lambda *a, **k: pytest.fail('broker constructed')
    with pytest.raises(ValueError, match='submission_window_closed'):
        run_entry_only(config, **args)
    assert not store.rows


def test_calendar_unavailable_prevents_broker_construction():
    config, args, store, broker, *rest = fixture()
    def marker(h, c):
        if c is None: raise ValueError('calendar_unavailable')
    args.update(calendar_reader=lambda *a, **k: None, marker_for=marker,
                broker_factory=lambda *a, **k: pytest.fail('broker constructed'))
    with pytest.raises(ValueError, match='calendar_unavailable'):
        run_entry_only(config, **args)


@pytest.mark.parametrize('method', ['cancel_order', 'replace_order'])
def test_entry_broker_forbids_stop_mutations(method):
    with pytest.raises(ValueError, match='forbidden'):
        getattr(EntryOnlyBroker(Broker()), method)('order')


@pytest.mark.parametrize('change', [{'side': 'sell'}, {'type': 'stop'},
                                  {'time_in_force': 'gtc'}, {'extended_hours': False}])
def test_entry_broker_only_accepts_day_extended_buy_limits(change):
    broker = Broker(); wrapped = EntryOnlyBroker(broker)
    request = dict(side='buy', type='limit', time_in_force='day', extended_hours=True)
    request.update(change)
    with pytest.raises(ValueError, match='buy_limit_required'):
        wrapped.submit_order(request)
    assert not broker.posts


def test_store_refuses_to_rebind_existing_signal_without_marker():
    config, args, store, broker, reports, checks, h = fixture()
    signal = store.enroll(h, h['proposals'][0], {})
    store.save(signal, state='OPEN')
    wrapped = EntryOnlyStore(store, {}, lambda *a: {'cancel_at': 'later'})
    with pytest.raises(ValueError, match='not_rebound'):
        wrapped.enroll(h, h['proposals'][0], {})


def test_store_refuses_lifecycle_claims():
    wrapped = EntryOnlyStore(Store(), {}, lambda *a: {})
    with pytest.raises(ValueError, match='claim_forbidden'):
        wrapped.claim(role='stop', operation='POST')


def test_dry_run_does_not_read_calendar_or_broker():
    config, args, store, broker, reports, checks, h = fixture()
    config.dry_run = True
    args['calendar_reader'] = lambda *a, **k: pytest.fail('calendar read')
    value = run_entry_only(config, **args)
    assert value['status'] == 'skipped' and not store.rows and not broker.posts


def test_direct_sync_and_manage_are_forbidden_on_entry_engine():
    engine = entry_engine(NativeEngine)(None, None, guard=None, pipeline_orders=True, now=None)
    for method in ('sync', 'manage'):
        with pytest.raises(ValueError, match='forbidden'):
            getattr(engine, method)()
