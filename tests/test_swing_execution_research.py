"""Intraday evidence refusal, conservative price sensitivity and read-only audits."""
from copy import deepcopy
import json
import socket
from types import SimpleNamespace

import pytest

from scripts import swing_execution_research as execution
from scripts.research_dataset import digest_bytes

pytestmark=pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**kw):raise AssertionError('No research network')
    monkeypatch.setattr(socket.socket,'connect',deny)


def capture(tmp_path):
    value=dict(schema='jbravo.entry-research-capture.v1',provider='alpaca',feed='sip',adjustment='raw',currency='USD',
        size_units='SHARES',availability_basis='PROVIDER_EVENT_TIME_ASSUMPTION',
        quotes=[dict(symbol='AAA',at='2024-01-03T09:00:00+00:00',bid=99.9,ask=100.,ask_size_shares=2),
                dict(symbol='AAA',at='2024-01-03T14:30:00+00:00',bid=100.9,ask=101.,ask_size_shares=100)],
        references=[dict(symbol='AAA',signal_session='2024-01-02',at='2024-01-02T19:55:00-05:00',price=100.)])
    def bind():
        path=tmp_path/'quotes.json';raw=json.dumps(value).encode();path.write_bytes(raw)
        return dict(path=str(path),sha256=digest_bytes(raw))
    return value,bind


def book():
    from test_swing_research import proposal
    return [dict(signal_id='id',symbol='AAA',signal_session='2024-01-02',selected=True,is_exit_owner=True,proposal=proposal())]


def compare(binding):
    return execution.entry_comparison(book(),binding,cash=10000,fee_per_fill=1,slippage_bps=5)


def test_paired_first_quote_caps_displayed_size_and_never_claims_strategy_pnl(tmp_path):
    value,bind=capture(tmp_path);result=compare(bind())
    row=result['comparisons'][0]
    assert row['paired'] and row['plans']['premarket']['quantity']==2
    assert row['plans']['premarket']['price']==pytest.approx(100.05)
    assert row['plans']['premarket']['maximum_fill_price']==100.5
    assert row['premarket_minus_regular_price_bps']<0
    assert result['strategy_pnl'] is None and result['production_executor_parity'] is False
    assert result['broker_operations']==0
    assert execution.instant('2024-01-03T09:00:00Z')==execution.instant('2024-01-03T09:00:00+00:00')


def test_passive_touch_does_not_invent_fill_or_search_later_favorable_quote(tmp_path):
    value,bind=capture(tmp_path)
    value['quotes'][0].update(bid=100.8,ask=101.)
    value['quotes'].insert(1,dict(value['quotes'][0],at='2024-01-03T09:00:30+00:00',bid=99.8,ask=100.))
    result=compare(bind())
    assert not result['comparisons'][0]['paired']
    assert result['comparisons'][0]['plans']['premarket']['status'].startswith('NOT_IMMEDIATELY_MARKETABLE')


@pytest.mark.parametrize('cause',['missing_reference','no_quote','late_quote','zero_size'])
def test_missing_or_unusable_intraday_evidence_never_uses_daily_bar_proxy(tmp_path,cause):
    value,bind=capture(tmp_path)
    if cause=='missing_reference':value['references']=[]
    if cause=='no_quote':value['quotes']=value['quotes'][1:]
    if cause=='late_quote':value['quotes'][0]['at']='2024-01-03T09:02:00+00:00'
    if cause=='zero_size':value['quotes'][0]['ask_size_shares']=0
    result=compare(bind())
    assert result['comparisons'][0]['plans']['premarket']['quantity']==0
    assert result['strategy_pnl'] is None
    assert compare(None)['status']=='UNAVAILABLE_NO_CAPTURED_INTRADAY_QUOTES'


@pytest.mark.parametrize('cause',['hash','crossed','ambiguous_units','naive','future_reference','duplicate'])
def test_unbound_malformed_or_future_entry_inputs_rejected(tmp_path,cause):
    value,bind=capture(tmp_path)
    if cause=='crossed':value['quotes'][0]['bid']=1000
    if cause=='ambiguous_units':value['size_units']='LOTS'
    if cause=='naive':value['quotes'][0]['at']='2024-01-03T09:00:00'
    if cause=='future_reference':value['references'][0]['signal_session']='2024-01-03'
    if cause=='duplicate':value['quotes'].append(deepcopy(value['quotes'][0]))
    binding=bind()
    if cause=='hash':binding['sha256']='f'*64
    with pytest.raises(ValueError):compare(binding)


def snapshot():
    buy=dict(id='buy',side='buy',type='limit',status='filled',filled_qty='10',filled_avg_price='100')
    sell=dict(id='sell',side='sell',type='limit',status='filled',filled_qty='10',filled_avg_price='110')
    return dict(schema='jbravo.paper-research-snapshot.v1',signals=[dict(signal_id='a',symbol='AAA',state='CLOSED',
        payload=dict(proposal=dict(strategy_id='test')))],orders=[
            dict(signal_id='a',role='entry',operation='POST',state='OBSERVED',observation=buy),
            dict(signal_id='a',role='cancel',operation='DELETE',state='OBSERVED',observation=deepcopy(buy)),
            dict(signal_id='a',role='exit',operation='POST',state='OBSERVED',observation=sell)])


def test_paper_audit_deduplicates_cumulative_cancel_observations_and_keeps_net_unknown():
    result=execution.paper_scorecard(snapshot())
    row=result['rows'][0]
    assert row['quantity_bought']==10 and row['gross_realized_pnl']==100
    assert row['fully_exited'] and result['complete_round_trips']==1
    assert row['net_realized_pnl'] is None and result['net_expectancy'] is None
    assert row['continuous_protection_proven'] is False


def test_paper_unknown_claim_is_not_fill_and_excess_exit_is_rejected():
    source=snapshot();source['orders'][0].update(state='UNKNOWN',observation=None)
    source['orders']=source['orders'][:1]
    result=execution.paper_scorecard(source)
    assert result['complete_round_trips']==0 and result['rows'][0]['unresolved_claims']==1
    source=snapshot();source['orders'][-1]['observation']['filled_qty']='11'
    with pytest.raises(ValueError,match='exceeds'):execution.paper_scorecard(source)


def test_paper_snapshot_uses_bounded_selects_and_refuses_writable_or_truncated_connections():
    statements=[]
    class Cursor:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,sql,params=None):
            assert sql.startswith('SELECT ')
            if params:assert params[0]=='paper-account' and params[-1]==2
            statements.append(sql)
        def fetchone(self):return ('deployment','database-id','jbravo','jbravo')
        def fetchall(self):return []
    conn=SimpleNamespace(readonly=True,autocommit=False,isolation_level=2,cursor=Cursor)
    result=execution.paper_snapshot(conn,account_id='paper-account',start='2024-01-01T00:00:00+00:00',end='2024-02-01T00:00:00+00:00',max_rows=1)
    assert len(statements)==3 and result['database_mutations']==result['broker_operations']==0
    conn.readonly=False
    with pytest.raises(ValueError,match='Read-only'):execution.paper_snapshot(conn,account_id='paper-account',start='',end='')
    assert len(statements)==3
    conn.readonly=True
    Cursor.fetchall=lambda self:[1,2]
    with pytest.raises(ValueError,match='cap exceeded'):execution.paper_snapshot(conn,account_id='paper-account',start='2024-01-01T00:00:00+00:00',end='2024-02-01T00:00:00+00:00',max_rows=1)
