"""Synthetic financial contracts only; no database, provider or bot imports."""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
pytestmark=pytest.mark.alpaca_optional
from flask import Flask, request, jsonify
from dashboards import execution_gross_projection as E, recorded_trade_analytics as R
from scripts import trade_disposition as D, trade_allocation as A
from test_public_dashboard_integration import policy, IDENTITY, NOW


def inputs(identifier=9001, symbol='FICT', qty='2'):
    row=dict(trade_id=identifier,status='CLOSED',symbol=symbol,qty=qty,entry_price='10',exit_price='11',
        realized_pnl='2',entry_order_id=f'synthetic-entry-{identifier}',exit_order_id=f'synthetic-exit-{identifier}',
        entry_time=(NOW-timedelta(hours=2)).isoformat(),exit_time=(NOW-timedelta(hours=1)).isoformat(),
        updated_at=NOW.isoformat())
    def fill(key,side,quantity,price,offset):
        order=row['entry_order_id'] if side=='buy' else row['exit_order_id']
        return dict(activity_id=key,activity_type='FILL',order_id=order,symbol=symbol,side=side,
            provider_transaction_time=(NOW-timedelta(hours=1,minutes=offset)).isoformat(),
            qty=quantity,price=price,transaction_time=(NOW-timedelta(hours=1,minutes=offset)).isoformat(),
            provider_id=key,provider_order_id=order,provider_symbol=symbol,provider_side=side,
            provider_qty=quantity,provider_price=price,provider_activity_type='FILL')
    fills=[fill(f'b-{identifier}','buy',qty,'10',60),fill(f's1-{identifier}','sell',str(Decimal(qty)/2),'11',1),
           fill(f's2-{identifier}','sell',str(Decimal(qty)/2),'13',0)]
    owner={key:row[key] for key in ('trade_id','status','entry_order_id','exit_order_id')}
    return row,fills,owner


def project(rows=None,fills=None,owners=None,p=None,**kw):
    row,f,o=inputs();rows=[row] if rows is None else rows
    p=p or policy();execution=E.reconstruct(p,rows,f if fills is None else fills,[o] if owners is None else owners)
    return R.project(p,rows,identity=IDENTITY,now=NOW,execution=execution,**kw)


def test_weighted_execution_basis_is_explicit_preserves_legacy_values_and_hold():
    row,f,o=inputs();original=deepcopy(row)
    result=project([row],f,[o]);legacy=R.project(policy(),[row],identity=IDENTITY,now=NOW)
    assert result['rows'][-1]['totalPL']=='4' and legacy['rows'][-1]['totalPL']=='2'
    assert row==original and result['pnl_basis']==E.BASIS and result['view_scope']==E.SCOPE
    assert result['performance_status']=='EXECUTION_GROSS_ONLY'
    assert result['net_eligible'] is result['entry_ready'] is result['accounting_complete'] is False
    assert result['net_pnl'] is result['fees'] is None and result['net_status']=='UNKNOWN'


@pytest.mark.parametrize('field,value,reason',[
    ('qty','0.5','QUANTITY_MISMATCH'),('symbol','OTHER','SYMBOL_MISMATCH'),
    ('side','sell_short','SIDE_OR_LIFECYCLE_UNRESOLVED'),('provider_side','buy','FILL_ECONOMICS_UNRESOLVED'),
    ('price','NaN','FILL_ECONOMICS_UNRESOLVED'),('provider_quantity','999','FILL_ECONOMICS_UNRESOLVED')])
def test_bad_or_partial_order_never_uses_scalar_fallback(field,value,reason):
    row,f,o=inputs();f[-1][field]=value
    if field in ('qty','symbol','side'):
        f[-1]['provider_'+field]=value
    result=project([row],f,[o])
    assert result['status']=='partial' and result['rows'][-1]['totalPL'] is None
    assert result['coverage']['reasons']=={reason:1} and result['coverage']['unknown']==1


def test_missing_fill_unknown_native_duplicates_idempotent_conflicts_fail():
    row,f,o=inputs()
    assert project([row],f[:1],[o])['coverage']['unknown']==1
    assert project([row],f+deepcopy(f),[o])['rows'][-1]['totalPL']=='4'
    changed=deepcopy(f[0]);changed['price']='9'
    with pytest.raises(ValueError,match='ACTIVITY_CONFLICT'):project([row],f+[changed],[o])


def test_external_scalar_owner_or_revoked_reservation_remains_ambiguous():
    row,f,o=inputs();other=dict(trade_id=9999,status='OPEN',entry_order_id=None,exit_order_id=row['exit_order_id'])
    assert project([row],f,[o,other])['coverage']['reasons']=={'ORDER_OWNER_AMBIGUOUS':1}
    overlay=A.result_read('VALIDATED',(),IDENTITY,2,{row['exit_order_id']:9999},
                          [dict(trade_id=9999,state='REVOKED',allocation_revision=2)])
    p=D.Policy(*IDENTITY,2,(),'VALIDATED_PERFORMANCE',(),overlay._metadata_json)
    assert project([row],f,[o],p)['coverage']['reasons']=={'ORDER_OWNER_AMBIGUOUS':1}
    # A sole external owner without the queried row's own claim is not unique
    # ownership for this row: it is an inconsistent population and must refuse.
    missing=dict(o,exit_order_id=None)
    with pytest.raises(ValueError,match='OWNER_BINDING_CHANGED'):project([row],f,[missing,other])


def test_partial_population_exclusions_symbol_ranking_and_zero_are_conserved():
    a,af,ao=inputs();b,bf,bo=inputs(9002);c,cf,co=inputs(9003,'EXCL');d,df,do=inputs(9004,'ZERO')
    for fill in df:fill['price']=fill['provider_price']='10'
    rows=[a,b,c,d];fills=af+bf+cf+df[:1];p=policy((9003,))
    result=project(rows,fills,[ao,bo,co,do],p)
    assert result['coverage']['included']==2 and result['coverage']['excluded']==1 and result['coverage']['unknown']==1
    leaders=project(rows,fills,[ao,bo,co,do],p,kind='leaderboard')
    assert leaders['rows']==[dict(rank=1,symbol='FICT',pl='8')]
    empty=project([],[],[]);unknown=project([a],[],[ao])
    assert empty['rows'][-1]['totalPL']=='0' and unknown['rows'][-1]['totalPL'] is None
    zero=project([d],df,[do]);assert zero['rows'][-1]['totalPL']=='0' and zero['rows'][-1]['winRatePct']==0


def test_row_revision_or_policy_drift_invalidates_captured_projection():
    row,f,o=inputs();p=policy();projection=E.reconstruct(p,[row],f,[o])
    row['updated_at']=(NOW+timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError,match='BINDING_CHANGED'):R.project(p,[row],identity=IDENTITY,now=NOW,execution=projection)
    row,f,o=inputs();projection=E.reconstruct(p,[row],f,[o]);p2=policy((9001,))
    with pytest.raises(ValueError,match='BINDING_CHANGED'):R.project(p2,[row],identity=IDENTITY,now=NOW,execution=projection)


def test_fractional_native_decimals_utc_and_breakeven_are_not_percent_or_integer_coerced():
    row,f,o=inputs(qty='0.125');row['exit_time']=(NOW-timedelta(hours=1)).astimezone(timezone(timedelta(hours=-5))).isoformat()
    result=project([row],f,[o]);assert result['rows'][-1]['totalPL']=='0.25'
    assert result['rows'][0]['tradesCount']==1
    with pytest.raises(ValueError):E.exact(0.125)


def test_crc110_canonical_allocation_once_with_real_pinned_row_guard_on_synthetic_values(monkeypatch):
    row,_,owner=inputs(110,'CRC','30')
    row.update(entry_price='65',exit_price=None,realized_pnl=None,exit_order_id=None)
    owner['exit_order_id']=None
    allocation=A.Allocation('fictional-allocation',110,'ACTIVE',1,Decimal(30),Decimal(30),Decimal('-63.38'),
        'SYNTHETIC_ONLY',('fictional-fill',),('synthetic-entry-110','fictional-exit-one','fictional-exit-two'),('fictional-entry','fictional-fill'))
    reserved={order:110 for order in allocation.order_ids}
    overlay=A.result_read('VALIDATED',(allocation,),IDENTITY,2,reserved)
    p=D.Policy(*IDENTITY,2,(),'VALIDATED_PERFORMANCE',(allocation,),overlay._metadata_json)
    monkeypatch.setattr(A,'ROW_FINGERPRINT',D.fingerprint(row));monkeypatch.setattr(A,'ROW_VERSION',row['updated_at'])
    monkeypatch.setattr(A,'ENTRY_ORDER',row['entry_order_id'])
    monkeypatch.setattr(A,'FILLS',(
        (1,'fictional-entry','buy',row['entry_order_id'],'30','65',row['entry_time']),
        (2,'fictional-fill','sell','fictional-exit-one','30','62',row['exit_time'])))
    result=project([row],[],[owner],p)
    assert result['coverage']['included']==1 and result['rows'][-1]['totalPL']=='-63.38'
    assert row['realized_pnl'] is None and row['exit_price'] is None
    assert result['net_pnl'] is None and result['net_eligible'] is False
    changed=dict(row,updated_at=(NOW+timedelta(seconds=1)).isoformat())
    with pytest.raises(A.AllocationError,match='ROW_CHANGED'):project([changed],[],[owner],p)


def test_full_population_windows_and_ties_are_not_frontend_page_totals():
    a,af,ao=inputs();b,bf,bo=inputs(9002,'OTHER')
    # Same amounts with an older, offset-equivalent exit; ALL includes both.
    shift=timedelta(days=10)
    for key in ('entry_time','exit_time'):b[key]=(datetime.fromisoformat(b[key])-shift).isoformat()
    for fill in bf:fill['transaction_time']=(datetime.fromisoformat(fill['transaction_time'])-shift).isoformat()
    result=project([a,b],af+bf,[ao,bo])
    assert result['rows'][0]['tradesCount']==1 and result['rows'][-1]['tradesCount']==2
    assert result['rows'][-1]['totalPL']=='8'
    leaders=project([a,b],af+bf,[ao,bo],kind='leaderboard')
    assert [row['symbol'] for row in leaders['rows']]==['FICT','OTHER']


from sql_transport_fixture import Cursor


class Connection:
    autocommit=False
    def __init__(self):
        row,f,o=inputs();self.records=[row];self.fills=f;self.owners=[o];self.calls=[];self.wire=[];self.fetch_sizes=[];self.isolation="read committed";self.snapshots=0;self.source_reads=0;self.policy_reads=0
        self.rollback=Mock();self.close=Mock()
    def get_transaction_status(self):return 0
    def cursor(self):return Cursor(self)


def test_read_transaction_budgets_canonical_and_queries_without_provider(monkeypatch):
    conn=Connection();reader=Mock(return_value=policy());monkeypatch.setattr(D,'read_policy',reader)
    result=R.read_view(lambda:conn,IDENTITY,now=NOW,basis='execution_gross')
    assert result['ok'] and result['rows'][-1]['totalPL']=='4'
    assert conn.calls[0][0]=='SET TRANSACTION ISOLATION LEVEL READ COMMITTED READ ONLY'
    assert any('statement_timeout' in sql for sql,_ in conn.calls)
    assert all(not any(word in sql.upper() for word in ('INSERT ','UPDATE ','DELETE ')) for sql,_ in conn.calls)
    reader.assert_called_once();conn.rollback.assert_called_once();conn.close.assert_called_once()


def test_db_failure_timeout_oversize_saturation_are_unavailable_not_empty(monkeypatch):
    assert R.read_view(lambda:None,IDENTITY,basis='execution_gross')['coverage'] is None
    monkeypatch.setattr(D,'read_policy',lambda *a,**k:policy())
    for mutation in ('oversize','saturation'):
        conn=Connection()
        if mutation=='oversize':conn.records[0]['unknown']='x'*32769
        else:conn.records=conn.records*1001
        result=R.read_view(lambda:conn,IDENTITY,now=NOW,basis='execution_gross')
        assert result['status']=='unavailable' and result['rows']==[] and result['coverage'] is None
        conn.rollback.assert_called_once();conn.close.assert_called_once()
    bounded=E.BoundedCursor(Cursor(Connection()),0,clock=lambda:1)
    with pytest.raises(ValueError,match='DEADLINE'):bounded.execute('SELECT 1')


def test_deadline_after_projection_cannot_publish_fresh_amounts(monkeypatch):
    conn=Connection();monkeypatch.setattr(D,'read_policy',lambda *a,**kw:policy())
    real=E.BoundedCursor
    monkeypatch.setattr(E,'BoundedCursor',lambda raw,deadline:real(raw,deadline,clock=lambda:0))
    ticks=iter([0,9]);monkeypatch.setattr(R.time,'monotonic',lambda:next(ticks))
    result=R.read_view(lambda:conn,IDENTITY,now=NOW,basis='execution_gross')
    assert result['status']=='unavailable' and result['coverage'] is None
    conn.rollback.assert_called_once();conn.close.assert_called_once()


@pytest.mark.parametrize('visibility',[[],[('alpaca_activities','v',False,False,False,False,True),('trades','r',False,False,False,False,True)],
    [('alpaca_activities','r',True,False,False,False,True),('trades','r',False,False,False,False,True)],
    [('alpaca_activities','r',False,False,False,False,False),('trades','r',False,False,False,False,True)]])
def test_invisible_view_rls_or_missing_privileges_cannot_be_successful_empty(monkeypatch,visibility):
    conn=Connection();conn.visibility=visibility;monkeypatch.setattr(D,'read_policy',lambda *a,**kw:policy())
    result=R.read_view(lambda:conn,IDENTITY,now=NOW,basis='execution_gross')
    assert result['status']=='unavailable' and result['coverage'] is None
    conn.rollback.assert_called_once();conn.close.assert_called_once()


def test_real_route_bodies_dispatch_only_explicit_basis_to_read_only_reader(monkeypatch):
    # Real Flask requests through the exact parsed route bodies. Full application
    # startup is intentionally excluded because it owns runtime configuration.
    tree=ast.parse(Path('dashboards/dashboard_app.py').read_text(encoding='utf-8'))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('api_trades_stats','api_trades_leaderboard')]
    assert len(functions)==2
    app=Flask('synthetic-execution-route');conn=Connection();monkeypatch.setattr(D,'read_policy',lambda *a,**k:policy())
    ns=dict(request=request,_json_no_store=jsonify,db=SimpleNamespace(get_db_conn=lambda:conn),
        _execution_gross_db_conn=lambda:conn,_dashboard_read_policy_identity=lambda:IDENTITY,_parse_trades_range=lambda value,default='all':value or default,
        _parse_positive_int=lambda value,**kw:kw['default'])
    for function in functions:
        function.decorator_list=[];exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'route-test','exec'),ns)
        app.add_url_rule('/api/trades/'+('stats' if function.name.endswith('stats') else 'leaderboard'),function.name,ns[function.name])
    client=app.test_client()
    stats=client.get('/api/trades/stats?range=all&basis=execution_gross')
    assert stats.status_code==200 and stats.json['view_scope']==E.SCOPE
    leaders=client.get('/api/trades/leaderboard?range=all&basis=execution_gross')
    assert leaders.status_code==200 and leaders.json['view_scope']==E.SCOPE
