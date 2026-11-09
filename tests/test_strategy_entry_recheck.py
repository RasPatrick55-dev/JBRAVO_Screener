"""Bounded admission rechecks use only synthetic data and deny all networking."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
import socket

import pytest

from scripts.strategy_entry_only import EntryOnlyBroker, admitted_entry_lease, entry_engine, run_entry_only
from scripts.strategy_entry_preflight import broker_entry_preflight
from test_strategy_entry_only import NOW, NativeEngine, fixture
from test_strategy_entry_preflight import add, fixture as protection_fixture
from test_strategy_order_lifecycle import native, native_rules  # pinned deployed fixtures

pytestmark = pytest.mark.alpaca_optional
STOP = 'ENTRY_PREFLIGHT_STOP_COVERAGE_UNCONFIRMED'
WINDOW = 'entry_only_premarket_submission_window_closed'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', deny)


class Clock:
    def __init__(self, now=NOW):
        self.current, self.elapsed, self.sleeps = now, 0., []

    def now(self): return self.current
    def monotonic(self): return self.elapsed

    def advance(self, seconds):
        self.current += timedelta(seconds=seconds)
        self.elapsed += seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)


def harness(now=NOW):
    config, args, store, broker, reports, checks, handoff = fixture()
    clock = Clock(now)
    def sleep(seconds):
        assert not store.locked, 'monitor must be able to acquire the lease'
        assert not store.rows and not store.outbox and not broker.posts
        clock.sleep(seconds)
    args.update(now=clock.now, monotonic=clock.monotonic, wait=sleep)
    return config, args, store, broker, clock, handoff


def fails(code=STOP):
    def fail(*args, **kwargs): raise ValueError(code)
    return fail


def test_transient_stop_rechecks_fresh_inputs_then_submits_only_once():
    config, args, store, broker, clock, handoff = harness()
    calls, revalidated = [], []
    original = args['runtime'].readonly_handoff
    def read(*a, **kw):
        if store.locked:
            assert kw.get('pipeline_limits') is True
            revalidated.append(1)
        return original(*a, **kw)
    args['runtime'].readonly_handoff = read
    def preflight(*a, **kw):
        assert store.locked and not store.rows and not broker.posts
        calls.append(1)
        if len(calls) < 3: raise ValueError(STOP)
        return {'status': 'pass'}
    args['preflight'] = preflight
    result = run_entry_only(config, **args)
    assert len(calls) == 4  # three admission checks plus the unchanged per-symbol check
    assert len(revalidated) == 2 and clock.sleeps == [2., 2.]
    assert result['preflight']['admission_attempts'] == 3
    assert len(broker.posts) == 1 and not store.locked and broker.closed


def test_real_pending_replacement_can_settle_between_unlocked_checks(native):
    config, args, account_store, _, clock, _ = harness()
    store, broker, runtime, handoff = protection_fixture()
    add(store, broker, handoff)
    old = broker.orders['onestop']
    old.update(status='pending_replace', replaced_by='successor')
    record = deepcopy(store.records['one'][1])
    record.update(operation='PATCH', client_order_id='successor', sequence=1)
    record['request']['order_id'] = old['id']
    record['request']['order']['client_order_id'] = 'successor'
    new = dict(old, id='successor', client_order_id='successor', status='new', replaces=old['id'])
    record['observation'] = deepcopy(new)
    store.records['one'].append(record); broker.orders['successor'] = new
    immutable = deepcopy((store.rows, store.records))
    def read():
        assert account_store.locked
        return broker_entry_preflight(runtime, native.Engine, store, broker,
            base_dir='unused', handoff=handoff, proposed_symbols={'LI', 'HUM'})
    def settle(seconds):
        assert not account_store.locked
        clock.sleep(seconds)
        old['status'] = 'replaced'
    with admitted_entry_lease(args['runtime'], account_store, NOW,
            check_window=lambda: True, wait=settle, revalidate=lambda: True,
            preflight=read, progress={}) as proof:
        assert proof['admission_attempts'] == 2
        assert proof['protected_signal_ids'] == ['one'] and account_store.locked
    assert immutable == (store.rows, store.records)
    assert clock.sleeps == [2.] and not account_store.locked


def test_permanent_stop_failure_stops_after_ten_checks_without_mutation():
    config, args, store, broker, clock, _ = harness()
    args['preflight'] = fails()
    with pytest.raises(ValueError, match='^' + STOP + '$'):
        run_entry_only(config, **args)
    assert args['progress']['preflight_attempts'] == 10
    assert clock.sleeps == [2.] * 9
    assert not store.rows and not store.outbox and not broker.posts
    assert not store.locked and broker.closed


@pytest.mark.parametrize('code', ['ENTRY_PREFLIGHT_ACCOUNT_MISMATCH',
    'ENTRY_PREFLIGHT_ORDER_OUTCOME_UNRESOLVED', 'ENTRY_PREFLIGHT_SNAPSHOT_CHANGED',
    'ENTRY_PREFLIGHT_READ_OR_VALIDATION_FAILED', 'ENTRY_PREFLIGHT_UNTRACKED_ORDER',
    'ENTRY_PREFLIGHT_OWNERSHIP_CONFLICT', STOP + '_OTHER'])
def test_other_preflight_failures_are_never_retried(code):
    config, args, store, broker, clock, _ = harness()
    args['preflight'] = fails(code)
    with pytest.raises(ValueError, match='^' + code + '$'):
        run_entry_only(config, **args)
    assert args['progress']['preflight_attempts'] == 1 and not clock.sleeps
    assert not store.rows and not broker.posts and not store.locked


def test_read_timeout_is_not_retried():
    config, args, store, broker, clock, _ = harness()
    def timeout(*a, **kw): raise TimeoutError('read timeout')
    args['preflight'] = timeout
    with pytest.raises(TimeoutError): run_entry_only(config, **args)
    assert not clock.sleeps and not broker.posts and not store.locked


@pytest.mark.parametrize('field', ['session', 'source_bindings', 'pipeline_limit_order', 'proposals'])
def test_changed_handoff_after_pause_blocks_without_second_preflight(field):
    config, args, store, broker, clock, handoff = harness()
    args['preflight'] = fails()
    def wait(seconds):
        assert not store.locked
        clock.sleep(seconds)
        handoff[field] = {'changed': True} if field != 'proposals' else []
    args['wait'] = wait
    with pytest.raises(ValueError, match='entry_only_admission_binding_changed'):
        run_entry_only(config, **args)
    assert args['progress']['preflight_attempts'] == 1
    assert not store.rows and not broker.posts and not store.locked


def test_source_guard_error_matching_retry_code_is_still_not_retried():
    config, args, store, broker, clock, _ = harness()
    args['preflight'] = fails()
    def source(*a, **kw):
        if store.locked: raise ValueError(STOP)
        return True
    args['runtime'].source_guard = source
    with pytest.raises(ValueError, match=STOP): run_entry_only(config, **args)
    assert clock.sleeps == [2.] and args['progress']['preflight_attempts'] == 1
    assert not store.rows and not broker.posts


def test_wait_is_capped_at_original_sixty_second_deadline():
    config, args, store, broker, clock, _ = harness(NOW + timedelta(seconds=57))
    args['preflight'] = fails()
    with pytest.raises(ValueError, match=WINDOW): run_entry_only(config, **args)
    assert clock.sleeps == [1.] and args['progress']['preflight_attempts'] == 1
    assert not store.rows and not broker.posts


def test_slow_successful_preflight_cannot_enroll_after_deadline():
    config, args, store, broker, clock, _ = harness()
    def slow(*a, **kw):
        clock.advance(58.)
        return {'status': 'pass'}
    args['preflight'] = slow
    with pytest.raises(ValueError, match=WINDOW): run_entry_only(config, **args)
    assert not store.rows and not broker.posts and not clock.sleeps


def test_wall_clock_rollback_cannot_extend_the_monotonic_budget():
    config, args, store, broker, clock, _ = harness()
    args['preflight'] = fails()
    def wait(seconds):
        assert not store.locked
        clock.elapsed += 58.
        # Wall clock still says 09:00:02, but the original minute has elapsed.
    args['wait'] = wait
    with pytest.raises(ValueError, match=WINDOW): run_entry_only(config, **args)
    assert args['progress']['preflight_attempts'] == 1 and not broker.posts


def test_slow_core_guard_cannot_return_authority_after_deadline():
    config, args, store, broker, clock, _ = harness()
    def guard(*a):
        clock.advance(58.)
        return True
    args['runtime'].guard_for = lambda *a, **kw: guard
    with pytest.raises(ValueError, match=WINDOW): run_entry_only(config, **args)
    assert not broker.posts and not clock.sleeps and not store.locked


def test_final_broker_boundary_checks_time_even_after_guard_and_claim():
    config, args, store, broker, clock, _ = harness()
    original = store.claim
    def delayed_claim(**kw):
        original(**kw)
        clock.advance(58.)
    store.claim = delayed_claim
    with pytest.raises(ValueError, match=WINDOW): run_entry_only(config, **args)
    assert store.outbox and not broker.posts and not clock.sleeps
    assert not store.locked


@pytest.mark.parametrize('after_post', [False, True])
def test_error_after_possible_submission_never_restarts_admission(after_post):
    config, args, store, broker, clock, _ = harness()
    if after_post:
        class FailingEngine(NativeEngine):
            def enter(self, *a, **kw):
                super().enter(*a, **kw)
                raise ValueError(STOP)
        args['NativeEngine'] = FailingEngine
    else:
        original = store.claim
        def uncertain(**kw):
            original(**kw)
            raise ValueError(STOP)
        store.claim = uncertain
    with pytest.raises(ValueError, match=STOP): run_entry_only(config, **args)
    assert args['progress']['preflight_attempts'] == 1 and not clock.sleeps
    assert len(broker.posts) == int(after_post) and store.outbox and not store.locked


def test_per_symbol_failure_is_not_retried_even_before_its_first_submission():
    config, args, store, broker, clock, _ = harness()
    calls = []
    def preflight(*a, **kw):
        calls.append(1)
        if len(calls) == 2: raise ValueError(STOP)
        return {'status': 'pass'}
    args['preflight'] = preflight
    with pytest.raises(ValueError, match=STOP): run_entry_only(config, **args)
    assert len(calls) == 2 and not clock.sleeps and not store.rows and not broker.posts


def test_second_symbol_failure_after_accepted_order_is_not_retried():
    config, args, store, broker, clock, handoff = harness()
    handoff['proposals'].append({'signal_id': 'b' * 64, 'symbol': 'BBB'})
    calls = []
    def preflight(*a, **kw):
        calls.append(1)
        if len(calls) == 3: raise ValueError(STOP)
        return {'status': 'pass'}
    args['preflight'] = preflight
    with pytest.raises(ValueError, match=STOP): run_entry_only(config, **args)
    assert len(calls) == 3 and not clock.sleeps and len(broker.posts) == 1


def test_lease_acquisition_wait_cannot_overrun_submission_window():
    config, args, store, broker, clock, _ = harness(NOW + timedelta(seconds=57.75))
    @contextmanager
    def busy(*a, wait, **kw):
        wait(.5)
        pytest.fail('lease should not have been acquired after deadline')
        yield
    args['runtime'].entry_lease = busy
    with pytest.raises(ValueError, match=WINDOW): run_entry_only(config, **args)
    assert clock.sleeps == [.25] and not broker.posts


def test_failed_lease_release_aborts_without_sleep_or_retry():
    config, args, store, broker, clock, _ = harness()
    @contextmanager
    def broken(*a, **kw):
        store.locked = True
        try: yield
        finally:
            store.locked = False
            raise ValueError('execution_account_lease_lost')
    args['runtime'].entry_lease = broken
    args['preflight'] = fails()
    with pytest.raises(ValueError, match='execution_account_lease_lost'):
        run_entry_only(config, **args)
    assert not clock.sleeps and not broker.posts


def test_native_claim_then_slow_policy_transaction_cannot_post_late(native):
    from tests.test_strategy_execution_engine import setup
    _, store, broker, handoff = setup()
    signal = store.enroll(handoff, handoff['proposals'][0], {})
    clock = Clock()
    @contextmanager
    def slow_transaction():
        clock.advance(58.)
        yield
    store.transaction = slow_transaction
    def window():
        if clock.elapsed >= 58.: raise ValueError(WINDOW)
    engine = entry_engine(native.Engine)(store, EntryOnlyBroker(broker, before_submit=window),
        guard=lambda *a: True, now=clock.now)
    request = dict(symbol=signal['payload']['proposal']['symbol'], side='buy', type='limit',
        time_in_force='day', extended_hours=True, qty=1, limit_price=100.)
    assert engine._perform(signal, 'entry', 0, request) is None
    assert not broker.posts and len(store.outbox) == 1
    assert next(iter(store.outbox.values()))['state'] == 'UNKNOWN'
    assert any(e['code'] == 'ORDER_OUTCOME_UNKNOWN_NO_RESUBMISSION' for e in engine.events)


def test_native_actionable_handoff_is_identical_across_recheck_times(native_rules):
    from tests.test_strategy_execution import positive, NOW as prepared_at
    record, rows = positive()
    initial = native_rules.prepare_handoff(record, rows,
        now=prepared_at + timedelta(seconds=10), pipeline_limits=True)
    opening = native_rules.utc(initial['execution_policy']['entry_open_at'])
    first = native_rules.prepare_handoff(record, rows,
        now=opening + timedelta(seconds=2), pipeline_limits=True)
    later = native_rules.prepare_handoff(record, rows,
        now=opening + timedelta(seconds=20), pipeline_limits=True)
    assert initial == first == later


def test_other_worker_signal_during_pause_is_never_a_batch_exemption():
    config, args, store, broker, clock, handoff = harness()
    handoff['proposals'].append({'signal_id': 'b' * 64, 'symbol': 'BBB'})
    checks = []
    def preflight(*a, exempt_signal_ids=(), **kw):
        checks.append(set(exempt_signal_ids))
        if len(checks) == 1: raise ValueError(STOP)
        return {'status': 'pass'}
    def wait(seconds):
        assert not store.locked
        clock.sleep(seconds)
        store.rows['other-worker'] = dict(signal_id='other-worker', state='OPEN',
            revision=0, payload={})
    args.update(preflight=preflight, wait=wait)
    run_entry_only(config, **args)
    assert checks == [set(), set(), set(), {'a' * 64}]
    assert len(broker.posts) == 2


def test_freshness_failure_on_reread_cannot_retry_or_use_cached_handoff():
    config, args, store, broker, clock, _ = harness()
    original = args['runtime'].readonly_handoff
    def read(*a, **kw):
        if store.locked: raise ValueError('strategy_postflight_stale_or_future')
        return original(*a, **kw)
    args['runtime'].readonly_handoff = read
    args['preflight'] = fails()
    with pytest.raises(ValueError, match='strategy_postflight_stale_or_future'):
        run_entry_only(config, **args)
    assert clock.sleeps == [2.] and not broker.posts and not store.rows


def test_changed_selection_record_cannot_be_adopted_after_pause():
    config, args, store, broker, clock, _ = harness()
    original = args['runtime'].unified_handoff
    def selection(*a, **kw):
        result = deepcopy(original(*a, **kw))
        if store.locked: result['record']['run_id'] = 'new-run'
        return result
    args['runtime'].unified_handoff = selection
    args['preflight'] = fails()
    with pytest.raises(ValueError, match='entry_only_admission_binding_changed'):
        run_entry_only(config, **args)
    assert clock.sleeps == [2.] and not broker.posts and not store.rows
