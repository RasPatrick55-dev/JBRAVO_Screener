"""Fictional supplied-data tests; no broker, database, registry or private evidence.

These exercise the public dashboard contracts. They do not qualify any actual
allocation, claim native SQL validation, or replace the private accounting proof.
"""
import ast
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
import sys
import pytest
from dashboards import recorded_trade_analytics as R, source_contracts as S
from scripts import observed_fee_display as O, trade_disposition as D, trade_allocation as A

IDENTITY=('fictional-public-test','00000000-0000-4000-8000-000000000901')
NOW=datetime(2001,2,3,12,tzinfo=timezone.utc)

def policy(excluded=()):
    overlay=A.result_read('VALIDATED',(),IDENTITY,2,{})
    cases=tuple(D.ActiveCase('FICTIONAL-'+str(i),i,'SYNTHETIC_NOT_APPROVAL','TEST_ONLY',('VALIDATED_PERFORMANCE',)) for i in excluded)
    return D.Policy(*IDENTITY,2,cases,'VALIDATED_PERFORMANCE',(),overlay._metadata_json)

def row(identifier=9001,symbol='FICT',gross='1.25',stamp=NOW-timedelta(hours=1)):
    return dict(trade_id=identifier,symbol=symbol,status='CLOSED',qty='0.5',entry_price='10',
        exit_price=str(Decimal(10)+Decimal(gross)/Decimal('0.5')),realized_pnl=None,
        entry_time=(stamp-timedelta(hours=1)).isoformat(),exit_time=stamp.isoformat(),updated_at=stamp.isoformat())

def view(rows,p=None,**kw):return R.project(p or policy(),rows,identity=IDENTITY,now=NOW,**kw)

def test_actual_decimal_population_exclusions_and_hold():
    result=view([row(),row(9002,'FICT','-0.25'),row(9003,'FAKE','0'),row(9004,'EXCL','100')],policy((9004,)))
    assert result['coverage']==dict(included=3,excluded=1,unknown=0,unscoped_unknown=0,unscoped_excluded=0,reasons={})
    assert result['rows'][-1]['totalPL']=='1'
    assert result['rows'][-1]['winRatePct']==pytest.approx(100/3)
    assert result['net_eligible'] is result['accounting_complete'] is result['entry_ready'] is False
    assert result['net_pnl'] is None and result['fees_status']=='UNKNOWN'
    assert view([row(),row(9002,'FICT','-0.25')],kind='leaderboard')['rows']==[dict(rank=1,symbol='FICT',pl='1')]

@pytest.mark.parametrize('field,value',[('qty',None),('qty',True),('entry_price',' '),('exit_price',None),('symbol',''),('exit_time','bad')])
def test_unknown_inputs_never_successful_zero(field,value):
    record=row();record[field]=value;result=view([record])
    assert result['status']=='partial' and result['coverage']['unknown']==1
    assert result['rows'][-1]['totalPL'] is None

def test_true_empty_and_utc_window():
    assert view([])['rows'][-1]['totalPL']=='0'
    assert view([])['rows'][-1]['winRatePct'] is None
    record=row(stamp=NOW-timedelta(days=1));record['exit_time']=(NOW-timedelta(days=1)).astimezone(timezone(timedelta(hours=-5))).isoformat()
    assert view([record])['rows'][0]['tradesCount']==1
    record['exit_time']=(NOW-timedelta(days=1,microseconds=1)).isoformat()
    assert view([record])['rows'][0]['tradesCount']==0

def test_duplicate_cap_policy_revision_and_identity_fail():
    with pytest.raises(ValueError):view([row(),row()])
    with pytest.raises(ValueError):view([row()]*10001)
    broken=policy();object.__setattr__(broken,'database_id','00000000-0000-4000-8000-000000000999')
    with pytest.raises(D.TradePolicyError):view([row()],broken)

class Cursor:
    def __init__(self,connection):self.connection=connection;self.records=[];self.description=[]
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def execute(self,sql,params=None):
        self.connection.calls.append((sql,params))
        assert sql.lstrip().split()[0] in ('SELECT','SET','SAVEPOINT','ROLLBACK','RELEASE')
        if 'SELECT trade_id, symbol, status' in sql:
            names=['trade_id','symbol','status','qty','entry_time','entry_price','exit_time','exit_price','realized_pnl','exit_reason','updated_at']
            self.description=[(name,) for name in names];self.records=[tuple(item.get(name) for name in names) for item in self.connection.rows]
        elif 'to_jsonb' in sql:self.records=[(D.canonical(item),) for item in self.connection.rows]
        else:self.records=[]
    def fetchall(self):return self.records

class Connection:
    autocommit=False
    def __init__(self,rows):self.rows=rows;self.calls=[];self.rollback=Mock();self.close=Mock()
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def cursor(self):return Cursor(self)
    def get_transaction_status(self):return 0

def test_read_route_is_readonly_and_failure_is_not_empty(monkeypatch):
    connection=Connection([row()]);reader=Mock(return_value=policy());monkeypatch.setattr(D,'read_policy',reader)
    result=R.read_view(lambda:connection,IDENTITY,now=NOW)
    assert result['ok'] and result['rows'][-1]['totalPL']=='1.25'
    assert connection.calls[0][0]=='SET TRANSACTION READ ONLY'
    reader.assert_called_once();assert reader.call_args.kwargs['scope']=='VALIDATED_PERFORMANCE'
    connection.rollback.assert_called_once();connection.close.assert_called_once()
    assert R.read_view(lambda:None,IDENTITY)['status']=='unavailable'
    assert R.read_view(lambda:connection,None)['coverage'] is None

def test_missing_private_artifact_preserves_physical_rows_and_hold(tmp_path,monkeypatch):
    monkeypatch.setattr(O,'EVIDENCE_PATH',tmp_path/'private-evidence-not-distributed.json')
    record=row(identifier=110,symbol='FICT');record.update(qty=Decimal('0.5'),entry_time=NOW-timedelta(days=1),exit_time=NOW,updated_at=NOW)
    connection=Connection([record]);result=S.physical_history(lambda:connection,derived_identity=IDENTITY,now=NOW)
    assert result['ok'] and len(result['rows'])==1
    assert result['derived_read_context']['status']=='UNAVAILABLE_OR_INVALIDATED'
    assert result['rows'][0].get('derived_accounting') is None
    assert result['accounting_complete'] is result['entry_ready'] is False
    assert result['fees_status']==result['net_status']=='UNKNOWN'
    assert any('ROLLBACK TO SAVEPOINT history_derived_costs' in sql for sql,_ in connection.calls)
    connection.close.assert_called_once()

def test_native_timestamp_boundary_still_strict():
    assert O.wire_time(NOW)==O.wire_time(NOW.astimezone(timezone(timedelta(hours=-5))).isoformat())
    with pytest.raises(O.N.QualificationError):O.wire_time(NOW.replace(tzinfo=None))

@pytest.mark.parametrize('name',['_execute_latest_candidates_count','_screener_latest_run_scope','api_screener_picks','api_screener_backtest','api_screener_metrics'])
def test_current_main_strategy_enabled_route_is_preserved(name,monkeypatch):
    called=[];snapshot=defaultdict(lambda:'FICTIONAL',rows_final=2)
    monkeypatch.setitem(sys.modules,'scripts.strategy_pipeline',SimpleNamespace(enabled=lambda:True))
    monkeypatch.setitem(sys.modules,'scripts.strategy_dashboard',SimpleNamespace(snapshot=lambda:snapshot,
        api_payload=lambda kind,args:called.append(kind) or dict(ok=True,kind=kind)))
    path=Path(__file__).parents[1]/'dashboards/dashboard_app.py'
    node=next(item for item in ast.parse(path.read_bytes()).body if isinstance(item,ast.FunctionDef) and item.name==name)
    node.decorator_list=[] # Exercise the handler body without starting Flask or importing app configuration.
    namespace=dict(Any=Any,_json_no_store=lambda value:value,request=SimpleNamespace(args={}))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'reviewed-strategy-route','exec'),namespace)
    result=namespace[name]()
    if name.startswith('api_screener_'):assert result==dict(ok=True,kind=name.removeprefix('api_screener_')) and len(called)==1
    elif name=='_execute_latest_candidates_count':assert result==2
    else:assert result['run_date']=='FICTIONAL' and result['status']=='FICTIONAL'
