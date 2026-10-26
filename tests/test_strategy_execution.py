from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace
import json
import socket

import pandas as pd
import pytest

from scripts import strategy_execution as ex
from scripts import strategy_pipeline as pipeline, strategy_metrics as metrics
from scripts.strategy_backtest import digest
from scripts.strategy_library import get_strategy
from test_strategy_metrics import source
from test_strategy_pipeline import NOW, report

pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a, **kw):
        raise AssertionError('Execution contract tests cannot reach a provider')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


def positive():
    record = report()
    batch, backtest = source()
    record['run_id'] = record['invocation_id'] = '1'*32
    backtest['strategy_definition'] = json.loads(json.dumps(get_strategy(pipeline.STRATEGY).describe()))
    backtest['result_id'] = digest({k:v for k,v in backtest.items() if k != 'result_id'})
    metric = metrics.build_metrics(batch, backtest, primary_run_id=record['run_id'])
    record.update(batch=dict(batch, backtest=backtest, metrics=metric), backtest=backtest, metrics=metric,
        candidate_run_ts_utc=NOW.isoformat())
    record['handoff']['proposals'] = batch['proposals']
    record['stages'].update(backtest=backtest['status'], metrics=metric['status'])
    p = batch['proposals'][0]
    value = dict(run_date=datetime.fromisoformat(p['signal_session']).date(), symbol=p['symbol'],
        run_ts_utc=NOW, score=p['score'], close=p['signal_close_raw'],
        entry_price=p['signal_close_raw'],timestamp='2026-10-23T04:00:00+00:00',
        source='strategy_pullback_reclaim_v2', passed_gates=True, gate_fail_reason=None, ml_weight_used=0.,
        score_breakdown=dict(schema='jbravo.strategy-candidate-lineage.v1', proposal=deepcopy(p),
            archive_id=batch['archive_id'], archive_sha256=batch['archive_sha256'],
            scan_run_id=batch['scan_run_id'], dataset_sha256=backtest['data']['dataset_sha256'],
            definition_sha256=batch['strategy_contract']['definition_sha256']))
    return record, [value]


def test_empty_batch_is_explicit_and_never_replaced_with_old_candidates():
    value = report();value['candidate_run_ts_utc'] = NOW.isoformat()
    actual = ex.prepare_handoff(value, [], now=NOW+timedelta(seconds=10))
    assert actual['status'] == 'NO_BUY_SIGNALS' and actual['proposals'] == []
    assert actual['order_submission_authorized'] is False and actual['broker_operations'] == 0
    with pytest.raises(ValueError):
        ex.prepare_handoff(value, positive()[1], now=NOW+timedelta(seconds=10))


def test_positive_handoff_keeps_strategy_and_both_result_ids_without_performance_gate():
    record, rows = positive()
    value = ex.prepare_handoff(record, rows, now=NOW+timedelta(seconds=10))
    assert value['status'] == 'PREPARED'
    assert value['metrics_result_id'] == record['metrics']['result_id']
    assert value['backtest_result_id'] == record['backtest']['result_id']
    assert value['proposals'] == record['batch']['proposals']
    assert value['portfolio']['alloc_pct'] == .05 and value['portfolio']['atr_multiple'] == 1.
    # This fixture's completed signal is a loss; no unapproved performance filter.
    assert record['metrics']['rows'][0]['net_pnl'] < 0


@pytest.mark.parametrize('change', ['strategy','version','metrics','candidate','price','session','timestamp','bar_timestamp','entry_price','population','portfolio','definition'])
def test_mixed_or_changed_inputs_block(change):
    record, rows = positive()
    if change == 'strategy': record['batch']['strategy_contract']['strategy_id'] = 'legacy'
    elif change == 'version': record['batch']['proposals'][0]['strategy_version'] = '1'
    elif change == 'metrics': record['metrics']['backtest_result_id'] = 'f'*64
    elif change == 'candidate': rows[0]['score_breakdown']['proposal']['signal_id'] = 'f'*64
    elif change == 'price': rows[0]['close'] += 1
    elif change == 'session': rows[0]['run_date'] = NOW.date()
    elif change == 'timestamp': rows[0]['run_ts_utc'] += timedelta(seconds=1)
    elif change == 'bar_timestamp': rows[0]['timestamp']='2026-10-22T04:00:00+00:00'
    elif change == 'entry_price': rows[0]['entry_price'] += 1
    elif change == 'population': rows.append(deepcopy(rows[0]))
    elif change == 'portfolio': record['backtest']['parameters']['portfolio']['alloc_pct'] = .06
    else: record['backtest']['strategy_definition']['version'] = '1'
    with pytest.raises(ValueError):
        ex.prepare_handoff(record, rows, now=NOW+timedelta(seconds=10))


def test_premarket_uses_calendar_offset_and_next_session_clock_through_dst():
    handoff = dict(session=dict(next_open_at='2026-11-02T14:30:00+00:00', next_session='2026-11-02'),
        execution_policy=dict(model=ex.EXECUTION_MODEL,entry_open_at='2026-11-02T09:00:00+00:00',cancel_at='2026-11-02T09:35:00+00:00'))
    def window(now,next_open='2026-11-02T14:30:00+00:00'):
        return ex.entry_window(handoff,now=now,clock_next_regular_open=next_open,clock_timestamp=now)
    assert window('2026-11-02T08:55:00+00:00') == 'WAIT_PREMARKET_OPEN'
    assert window('2026-11-02T09:00:00+00:00') == 'PREMARKET_OPEN'
    with pytest.raises(ValueError, match='expired'):
        window('2026-11-02T09:35:00+00:00')
    with pytest.raises(ValueError, match='clock_session'):
        window('2026-11-02T09:00:00+00:00',next_open='2026-11-03T14:30:00+00:00')


def reference(bars=None):
    if bars is None:bars=[dict(t='2026-10-23T23:59:00+00:00',o=100.,h=100.,l=100.,c=100.,v=1000.)]
    return ex.after_hours_reference(bars,symbol='SRRK',signal_session='2026-10-23',
        observed_at='2026-10-24T00:15:00+00:00',feed='sip',adjustment='raw',currency='USD',complete=True)


def test_bound_signal_produces_a_premarket_limit_plan_with_explicit_backtest_gap():
    record,rows=positive();handoff=ex.prepare_handoff(record,rows,now=NOW+timedelta(seconds=10))
    signal=handoff['proposals'][0]['signal_id']
    planned=ex.build_entry_order(handoff,signal_id=signal,account_id='paper-account',cash=100000,
        reference=reference(),occupied_slots=0,now='2026-10-26T08:00:02+00:00',
        clock_next_regular_open='2026-10-26T13:30:00+00:00',clock_timestamp='2026-10-26T08:00:02+00:00')
    assert planned['request']['qty']==49 and planned['request']['extended_hours'] is True
    assert planned['request']['type']=='limit' and planned['request']['limit_price']==100.5
    assert planned['request']['time_in_force']=='day'
    assert planned['lineage']['metrics_result_id']==record['metrics']['result_id']
    assert planned['lineage']['backtest_result_id']==record['backtest']['result_id']
    assert planned['lineage']['signal_id']==signal
    assert planned['lineage']['backtest_alignment']=='PREMARKET_REPLAY_PENDING'
    assert planned['order_submission_authorized'] is False


@pytest.mark.parametrize('change',['missing','future','changed','wrong_symbol'])
def test_unavailable_or_changed_after_hours_reference_cannot_create_an_order(change):
    record,rows=positive();handoff=ex.prepare_handoff(record,rows,now=NOW+timedelta(seconds=10))
    ref=reference([]) if change=='missing' else reference()
    if change=='future':ref['observed_at']='2026-10-26T08:00:01+00:00'
    elif change=='changed':ref['price']+=1
    elif change=='wrong_symbol':ref['symbol']='PLUG'
    with pytest.raises(ValueError,match='after_hours_reference'):
        ex.build_entry_order(handoff,signal_id=handoff['proposals'][0]['signal_id'],account_id='paper-account',
            cash=100000,reference=ref,occupied_slots=0,now='2026-10-26T08:00:02+00:00',
            clock_next_regular_open='2026-10-26T13:30:00+00:00',clock_timestamp='2026-10-26T08:00:02+00:00')


def test_reference_keeps_sparse_last_print_and_does_not_relabel_daily_close_as_20_et():
    ref=reference([dict(t='2026-10-23T23:53:00+00:00',o=43.79,h=43.79,l=43.6984,c=43.6984,v=476.)])
    assert ref['last_bar_at']=='2026-10-23T23:53:00+00:00' and ref['price']==43.6984
    with pytest.raises(ValueError,match='outside_session'):
        reference([dict(t='2026-10-24T00:01:00+00:00',o=1,h=1,l=1,c=1,v=100)])


def test_cash_sizing_preserves_five_percent_whole_shares_and_total_position_cap():
    assert ex.size_entry(100000, 101, occupied_slots=0) == 49
    assert ex.size_entry(95000, 101, occupied_slots=1) == 47
    assert ex.size_entry(100000, 100, occupied_slots=4) == 0
    assert ex.size_entry(100000, 100, occupied_slots=1, reserved_cash=5000) == 47
    assert ex.size_entry(100, 101, occupied_slots=0) == 0
    for value in (float('nan'), -1, True):
        with pytest.raises(ValueError): ex.size_entry(value,100,occupied_slots=0)


def position():
    proposal = positive()[0]['batch']['proposals'][0]
    proposal = dict(proposal, signal_atr14_raw=1.5)
    return ex.initial_position(proposal, quantity=10, fill_price=100,
        filled_at='2026-10-26T08:00:01+00:00')


def bars(close=106, high=107, ema=103):
    return pd.DataFrame([dict(open=100.,high=101.,low=99.,close=100.,ATR14=2.,ema20=99.,
        macd=1.,macd_signal=.5,rsi=60.),dict(open=100.,high=high,low=99.,close=close,ATR14=2.,
        ema20=ema,macd=1.,macd_signal=.5,rsi=60.)], index=pd.to_datetime(['2026-10-23','2026-10-26']))


def test_actual_fill_sets_atr_and_trail_then_completed_bar_tightens_and_queues_partial():
    pos = position()
    assert pos['entry_price'] == 100 and pos['atr_stop'] == 98.5 and pos['trailing_stop'] == 97
    changed = ex.completed_bar(pos, bars(), session='2026-10-26', next_session='2026-10-27',
        now='2026-10-27T00:15:00+00:00')
    assert changed['position']['atr_stop'] == 104
    assert changed['position']['trailing_stop'] == pytest.approx(104.86)
    assert changed['pending_exit_reasons'] == ['PARTIAL_5PCT']
    assert changed['partial_quantity'] == 5 and changed['pending_exit_session'] == '2026-10-27'
    assert pos['max_price'] == 100  # caller state is never mutated before durable publication


def test_full_completed_bar_exit_precedes_partial_and_uses_calendar_days():
    pos = position(); pos['entry_time'] = '2026-10-19'
    out = ex.completed_bar(pos, bars(close=106, high=107, ema=108), session='2026-10-26',
        next_session='2026-10-27', now='2026-10-27T00:15:00+00:00')
    assert out['pending_exit_reasons'] == ['EMA20_BREAK','MAX_HOLD']
    assert out['partial_quantity'] == 0


def test_unfinished_future_or_missing_exit_data_cannot_tighten_protection():
    with pytest.raises(ValueError, match='not_completed'):
        ex.completed_bar(position(), bars(), session='2026-10-26', next_session='2026-10-27',
            now='2026-10-26T20:00:00+00:00')
    with pytest.raises(ValueError, match='missing_exit_feature'):
        ex.completed_bar(position(), bars().drop(columns='ATR14'),session='2026-10-26',
            next_session='2026-10-27', now='2026-10-27T00:15:00+00:00')


class Store:
    def __init__(self): self.claims = {};self.events=[]
    def claim(self, key, fingerprint, lineage):
        if key in self.claims:
            assert self.claims[key] == fingerprint
            return False
        self.claims[key] = fingerprint
        self.events.append('durable_claim')
        return True
    def uncertain(self, key): self.events.append('uncertain')
    def observed(self, key, value): self.events.append('observed')


class Broker:
    paper=True
    account_id='paper-account'
    base_url='https://paper-api.alpaca.markets'
    def __init__(self, store, *, timeout=False): self.store=store;self.posts=0;self.orders={};self.timeout=timeout
    def submit_order(self, request):
        assert self.store.events[0] == 'durable_claim'
        self.posts += 1
        value=dict(request, id='broker-order-1',status='accepted')
        self.orders[request['client_order_id']] = value
        if self.timeout: raise TimeoutError('accepted request but lost response')
        return value
    def find_by_client_order_id(self, key): return self.orders.get(key)


def submit(store, broker):
    return ex.submit_claimed(store, broker, account_id='paper-account',signal_id='a'*64,
        request=dict(client_order_id=ex.client_order_id('paper-account','a'*64),side='buy',
            symbol='SRRK',type='limit',limit_price=100.,time_in_force='day',extended_hours=True,qty=10),
        lineage=dict(strategy_contract=dict(strategy_id=pipeline.STRATEGY,strategy_version='2'),
            signal_id='a'*64,symbol='SRRK',handoff_id='b'*64,batch_sha256='c'*64,
            backtest_result_id='d'*64,metrics_result_id='e'*64),paper_research_entries=True)


def test_accepted_timeout_then_restart_recovers_order_without_second_post():
    store=Store(); broker=Broker(store, timeout=True)
    assert submit(store,broker)['status'] == 'UNKNOWN_NEVER_RESUBMIT'
    assert submit(store,broker)['status'] == 'RECOVERED'
    assert broker.posts == 1


def test_durable_claim_without_broker_observation_stays_unknown_never_retries():
    store=Store(); broker=Broker(store,timeout=True)
    submit(store,broker);broker.orders.clear()
    assert submit(store,broker)['status'] == 'UNKNOWN_NEVER_RESUBMIT'
    assert broker.posts == 1


def test_identity_is_signal_based_across_primary_reruns_and_separates_accounts_and_roles():
    key=ex.client_order_id('paper-account','a'*64)
    assert len(key) <= 48 and key == ex.client_order_id('paper-account','a'*64)
    assert key != ex.client_order_id('another-account','a'*64)
    assert key != ex.client_order_id('paper-account','a'*64,role='stop')


@pytest.mark.parametrize('origin',['https://api.alpaca.markets','https://paper-api.alpaca.markets.evil.test'])
def test_live_or_lookalike_origin_is_denied_before_claim_or_post(origin):
    store=Store();broker=Broker(store);broker.base_url=origin
    with pytest.raises(ValueError,match='paper_authority'):
        submit(store,broker)
    assert not store.claims and broker.posts == 0
