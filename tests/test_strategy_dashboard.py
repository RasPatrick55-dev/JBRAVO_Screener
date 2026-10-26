"""Execute the actual dashboard entry points without importing/starting the app."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
from scripts import strategy_dashboard as dashboard
from scripts import strategy_pipeline as pipeline

pytestmark=pytest.mark.alpaca_optional
ROOT=Path(__file__).resolve().parents[1]


def extract(path,name):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    fn.decorator_list=[];fn.returns=None
    for a in (*fn.args.args,*fn.args.kwonlyargs):a.annotation=None
    namespace=dict(pd=pd,request=SimpleNamespace(args={}),_json_no_store=lambda value:value)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]


def view(rows,*,status=None):
    return dict(ok=True,status=status or ('COMPLETE' if rows else 'EMPTY'),run_date='2026-10-26',
        run_ts_utc='2026-10-27T03:45:00+00:00',rows=rows,rows_final=len(rows),source='postgres',
        source_detail='strategy_primary:'+('BUY_SIGNALS' if rows else 'NO_BUY_SIGNALS'),entry_ready=False)


def positive():
    return dict(symbol='SRRK',close=51.51,volume=123456,score=8,
        score_breakdown=dict(proposal=dict(signal_id='a'*64,strategy_id=pipeline.STRATEGY,
            strategy_version='2',next_open_at='2026-10-27T13:30:00+00:00')))


@pytest.mark.parametrize('name',['api_screener_picks','api_screener_candidates'])
def test_actual_api_buy_then_zero_never_uses_raw_tables_or_csv(monkeypatch,name):
    monkeypatch.setattr(pipeline,'enabled',lambda:True)
    current=[view([positive()])]
    monkeypatch.setattr(dashboard,'snapshot',lambda:current[0])
    api=extract('dashboards/dashboard_app.py',name)
    assert api()['rows'][0]['signal_id']=='a'*64
    current[0]=view([])
    actual=api()
    assert actual['rows']==[] and actual['status']=='EMPTY'
    assert actual['run_date']=='2026-10-26' and actual['source']=='postgres'


def test_actual_table_scope_count_and_backtest_keep_authoritative_empty(monkeypatch):
    monkeypatch.setattr(pipeline,'enabled',lambda:True)
    current=[view([positive()])]
    monkeypatch.setattr(dashboard,'snapshot',lambda:current[0])
    table=extract('dashboards/data_io.py','screener_table')
    scope=extract('dashboards/dashboard_app.py','_screener_latest_run_scope')
    count=extract('dashboards/dashboard_app.py','_execute_latest_candidates_count')
    backtest=extract('dashboards/dashboard_app.py','api_screener_backtest')
    assert len(table()[0])==1 and count()==1
    current[0]=view([])
    assert table()[0].empty and 'NO_BUY_SIGNALS' in table()[2]
    assert scope()['run_ts_utc']==current[0]['run_ts_utc'] and scope()['status']=='EMPTY'
    assert count()==0 and backtest()['rows']==[]
    assert backtest()['metric_basis']=='STRATEGY_RESEARCH_SIMULATION'


def test_unavailable_strategy_snapshot_never_falls_back_to_legacy(monkeypatch):
    monkeypatch.setattr(pipeline,'enabled',lambda:True)
    monkeypatch.setattr(dashboard,'snapshot',lambda:dict(view([]),ok=False,status='ERROR',run_date=None,run_ts_utc=None))
    result=extract('dashboards/dashboard_app.py','api_screener_candidates')()
    assert result['ok'] is False and result['status']=='ERROR' and not result['rows']
    assert extract('dashboards/data_io.py','screener_table')()[0].empty


def test_equal_utc_timestamp_forms_accept_same_batch(monkeypatch):
    monkeypatch.setattr(dashboard,'snapshot',lambda:view([positive()]))
    assert dashboard.api_payload('candidates',dict(run_ts_utc='2026-10-27T03:45:00Z'))['rows_final']==1
    assert dashboard.api_payload('candidates',dict(run_ts_utc='2026-10-26T03:45:00Z'))['ok'] is False


def test_backtest_endpoint_names_the_screener_strategy_and_bound_result(monkeypatch):
    from test_strategy_pipeline import report
    record=report()
    current=dict(view([]),_strategy_backtest=record['backtest'])
    monkeypatch.setattr(dashboard,'snapshot',lambda:current)
    result=dashboard.api_payload('backtest',{})
    assert result['strategy_id']==pipeline.STRATEGY and result['strategy_version']=='2'
    assert result['backtest_status']=='SKIPPED_NO_CANDIDATES' and result['rows']==[]
    assert result['backtest_result_id']==record['backtest']['result_id']
    assert result['definition_sha256']==record['batch']['strategy_contract']['definition_sha256']
    assert '_strategy_backtest' not in result


def test_backtest_endpoint_does_not_borrow_another_preparation(monkeypatch):
    from test_strategy_pipeline import report
    current=dict(view([]),_strategy_backtest=report()['backtest'])
    monkeypatch.setattr(dashboard,'snapshot',lambda:current)
    result=dashboard.api_payload('backtest',dict(run_ts_utc='2026-10-26T03:45:00Z'))
    assert result['ok'] is False and result['status']=='ERROR' and result['rows']==[]
