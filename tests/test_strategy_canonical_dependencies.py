"""Offline integration against canonical native modules, not retained snapshots."""
from copy import deepcopy
from datetime import timedelta
import socket

import pytest

from scripts import strategy_execution as rules
from scripts import strategy_execution_engine as native
from scripts import strategy_order_lifecycle as lifecycle
from scripts.strategy_execution_broker import PaperBroker
from tests.test_strategy_execution import positive, NOW
from tests.test_strategy_execution_broker import ENV, Session, Response, ACCOUNT
from tests.test_strategy_order_lifecycle import (
    pipeline_fixture, test_native_pipeline_entry_only_to_monitor_ownership as ownership_case,
)

pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('No external I/O in canonical dependency tests')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


@pytest.mark.parametrize('fill_mode', ['unfilled', 'partial', 'full'])
def test_canonical_entry_only_to_pinned_monitor(fill_mode):
    # Exercises exact ticket submission, durable marker-before-claim, 04:35
    # survival, fill reconciliation, GTC protection and regular-close cleanup.
    ownership_case(native, fill_mode)


def test_canonical_monitor_preserves_source_drift_guard():
    class Changed(native.Engine):
        def tick(self, **kwargs):
            return {}
    with pytest.raises(ValueError, match='native_tick_source_changed'):
        lifecycle.monitor_engine(Changed)


def test_pipeline_preparation_is_optional_and_bounded():
    record, rows = positive()
    original = rules.prepare_handoff(record, rows, now=NOW+timedelta(seconds=10))
    opening = rules.utc(original['execution_policy']['entry_open_at'])
    with pytest.raises(ValueError, match='preparation_expired'):
        rules.prepare_handoff(record, rows, now=opening)
    enabled = rules.prepare_handoff(record, rows, now=opening, pipeline_limits=True)
    assert enabled['pipeline_limit_placement_v2'] is True
    assert 'pipeline_limit_placement_v2' not in original
    with pytest.raises(ValueError, match='preparation_expired'):
        rules.prepare_handoff(record, rows, now=opening+timedelta(minutes=35), pipeline_limits=True)
    with pytest.raises(ValueError, match='timing_mode_invalid'):
        rules.prepare_handoff(record, rows, now=opening, pipeline_limits='true')


def plan(handoff, broker, reference=None):
    return rules.build_pipeline_limit_order(handoff,
        signal_id=handoff['proposals'][0]['signal_id'], account_id=broker.account_id,
        cash='100000', reference=reference or handoff['pipeline_limit_order'], occupied_slots=0,
        now=broker.now, clock_next_regular_open=broker.clock()['next_open'], clock_timestamp=broker.now)


def test_pipeline_exact_price_no_buffer_or_quote_and_ticket_immutable():
    _, broker, handoff = pipeline_fixture(native)
    before = deepcopy(handoff)
    result = plan(handoff, broker)
    assert result['request']['limit_price'] == 101.23
    assert result['request']['qty'] == 49
    assert result['sizing']['extra_limit_buffer_pct'] == 0
    assert handoff == before
    with pytest.raises(ValueError, match='pipeline_limit_changed'):
        plan(handoff, broker, dict(handoff['pipeline_limit_order'], limit_price=101.24))
    broker.now = rules.utc(handoff['pipeline_limit_order']['valid_until'])
    with pytest.raises(ValueError, match='submission_not_current'):
        plan(handoff, broker)


def test_pipeline_engine_never_reads_market_data_and_unknown_never_reposts():
    store, broker, handoff = pipeline_fixture(native)
    def no_quotes(*args, **kwargs):
        raise AssertionError('Pipeline entry must not read quotes')
    broker.quotes = no_quotes
    broker.mode = 'lost'
    engine = native.Engine(store, broker, guard=lambda *args: True,
                           pipeline_orders=True, now=lambda: broker.now)
    def enter():
        engine.enter(handoff, {handoff['proposals'][0]['symbol']: handoff['pipeline_limit_order']},
                     now=broker.now, clock=broker.clock(), account=broker.account(),
                     positions={}, open_orders=[])
    enter()
    assert store.signals()[0]['state'] == 'UNKNOWN'
    broker.mode = None
    enter()
    assert len(broker.posts) == 1
    engine.tick(now=broker.now, observe_prices=False)
    assert store.signals()[0]['state'] == 'ENTRY_PENDING'


def test_paper_broker_market_data_disabled_before_transport():
    session = Session(Response(ACCOUNT))
    broker = PaperBroker(ENV, session=session, market_data=False)
    assert broker.account()['id'] == ACCOUNT['id']
    for path in ('/v2/stocks/quotes/latest', '/v2/stocks/bars'):
        with pytest.raises(ValueError, match='entry_market_data_forbidden'):
            broker.request('GET', path, data=True)
    assert len(session.calls) == broker.requests == 1


def test_manual_handoff_cannot_claim_or_submit_native_order():
    store, broker, handoff = pipeline_fixture(native)
    handoff['manual_paper_test'] = True
    signal = store.enroll(handoff, handoff['proposals'][0], handoff['pipeline_limit_order'])
    engine = native.Engine(store, broker, guard=lambda *args: True)
    with pytest.raises(ValueError, match='manual_position_orders_not_authorized'):
        engine._perform(signal, 'entry', 0, {})
    assert not store.outbox and not broker.posts


@pytest.mark.parametrize('change', ['digest', 'inspection', 'paper', 'portfolio', 'ticket', 'marker'])
def test_pipeline_changed_authority_or_contract_cannot_build(change):
    _, broker, handoff = pipeline_fixture(native)
    if change == 'digest':
        handoff['handoff_id'] = '0'*64
    else:
        if change == 'inspection': handoff['inspection_mode'] = True
        elif change == 'paper': handoff['paper_only'] = False
        elif change == 'portfolio': handoff['portfolio']['max_positions'] = 99
        elif change == 'ticket': handoff['pipeline_limit_order']['limit_price'] = 101.234
        else: handoff['pipeline_limit_placement_v2'] = False
        handoff['handoff_id'] = rules.digest({k:v for k,v in handoff.items() if k != 'handoff_id'})
    with pytest.raises(ValueError):
        plan(handoff, broker)


def test_pipeline_handoff_cannot_use_legacy_price_builder_or_default_engine():
    from tests.test_strategy_execution_engine import reference
    store, broker, handoff = pipeline_fixture(native)
    proposal = handoff['proposals'][0]
    with pytest.raises(ValueError, match='execution_handoff_changed'):
        rules.build_entry_order(handoff, signal_id=proposal['signal_id'],
            account_id=broker.account_id, cash='100000', reference=reference(),
            occupied_slots=0, now=broker.now,
            clock_next_regular_open=broker.clock()['next_open'], clock_timestamp=broker.now)
    engine = native.Engine(store, broker, guard=lambda *args: True, now=lambda: broker.now)
    with pytest.raises(ValueError, match='execution_handoff_changed'):
        engine.enter(handoff, {proposal['symbol']: reference()}, now=broker.now,
            clock=broker.clock(), account=broker.account(), positions={}, open_orders=[])
    assert not store.outbox and not broker.posts and not store.rows


def test_historical_manual_position_retains_protection_cancel_and_exit():
    from tests.test_strategy_execution_engine import setup, enter
    engine, store, broker, handoff = setup()
    enter(engine, broker, handoff)
    qty = broker.posts[0]['qty']
    broker.fill('entry', qty)
    signal = store.signals()[0]
    payload = deepcopy(signal['payload'])
    payload['handoff']['manual_paper_test'] = True
    store.save(signal, payload=payload)
    engine.tick(now=broker.now)
    assert broker.posts[-1]['type'] == 'stop'
    assert broker.posts[-1]['side'] == 'sell'
    assert broker.posts[-1]['qty'] == qty
    broker.bid = 90
    engine.tick(now=broker.now)
    assert len(broker.cancels) == 1
    assert broker.posts[-1]['type'] == 'limit'
    assert broker.posts[-1]['side'] == 'sell'
    assert broker.posts[-1]['qty'] == qty
    assert store.signals()[0]['state'] == 'EXIT_PENDING'


@pytest.mark.parametrize('hour', [3, 20, 23])
@pytest.mark.parametrize('pending_exit', [False, True])
def test_pipeline_manage_keeps_canonical_offhours_mutation_guard(hour, pending_exit):
    from datetime import datetime
    from tests.test_strategy_execution_engine import filled_position
    _, store, broker, _ = filled_position()
    signal = store.signals()[0]
    if pending_exit:
        payload = deepcopy(signal['payload'])
        payload.update(pending_exit_reasons=['PARTIAL_5PCT'], pending_exit_session='2026-10-26')
        store.save(signal, payload=payload)
    engine = native.Engine(store, broker, guard=lambda *args: True, pipeline_orders=True)
    before = (deepcopy(broker.posts), deepcopy(broker.patches), deepcopy(broker.cancels),
              deepcopy(store.rows), deepcopy(store.outbox))
    now = datetime(2026, 10, 26, hour, tzinfo=rules.NY)
    # A breached stop or queued partial exit would mutate during permitted hours.
    engine.manage(signal, store.orders(signal['signal_id']), now=now,
                  quote=dict(bid='90', ask='90.01'), regular_open=False)
    assert before == (broker.posts, broker.patches, broker.cancels, store.rows, store.outbox)
