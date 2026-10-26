from copy import deepcopy
import json
import socket
import pytest
from scripts import strategy_backtest as bt, strategy_metrics as metrics, strategy_pipeline as pipeline
from scripts import strategy_dashboard as dashboard
from scripts.strategy_library import get_strategy
from test_strategy_backtest import inputs, trace
from test_strategy_pipeline import report

pytestmark=pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def fail(*a,**k):raise AssertionError('Metrics must use bound results without a network')
    monkeypatch.setattr(socket.socket,'connect',fail)
    monkeypatch.setattr(socket,'create_connection',fail)


def source(history=True):
    batch,spec,data=inputs(True);p=batch['proposals'][0]
    current=dict(signal_id=p['signal_id'],symbol=p['symbol'],strategy_id=p['strategy_id'],strategy_version='2',
        signal_session=p['signal_session'],status='UNFILLED_PROPOSAL',bought_quantity=0,sold_quantity=0,fills=[])
    replay=dict(as_of=p['signal_session'],top_candidates=batch['proposals'],strategy=get_strategy(p['strategy_id']).describe(),
        metrics=dict(net_pnl=999999),lifecycle=([trace()] if history else [])+[current],observations=[])
    backtest=bt.report_from_replay(batch,replay,primary_run_id='1'*32,dataset=data,specification=spec)
    return batch,backtest


def test_metrics_use_complete_signal_pnl_and_exact_backtest_identity():
    batch,backtest=source();out=metrics.build_metrics(batch,backtest,primary_run_id='1'*32);r=out['rows'][0]
    assert out['backtest_result_id']==backtest['result_id'] and out['backtest_sha256']==bt.digest(backtest)
    assert out['candidate_signal_ids']==[batch['proposals'][0]['signal_id']]
    assert out['strategy_contract']==batch['strategy_contract']
    assert (r['trades'],r['wins'],r['losses'],r['net_pnl'],r['expectancy'])==(1,0,1,-3,-3)
    assert r['win_rate_pct']==0 and r['fees']==3 and r['profit_factor']==0
    assert r['max_drawdown_cash']==-3 and r['drawdown_units']=='USD_CLOSED_SIGNAL_PNL'
    assert out['input_candidates']==out['output_candidates']==1 and not out['ranking_applied']
    assert out['mode']=='REPORT_ONLY' and out['broker_operations']==0
    assert json.loads(json.dumps(out))==out
    assert bt.digest({k:v for k,v in out.items() if k!='result_id'})==out['result_id']


def test_zero_history_is_unknown_and_current_signal_gets_no_invented_outcome():
    batch,backtest=source(False);r=metrics.build_metrics(batch,backtest,primary_run_id='1'*32)['rows'][0]
    assert r['trades']==0 and r['win_rate_pct'] is None and r['expectancy'] is None
    assert r['profit_factor'] is None and r['sample_status']=='NO_COMPLETED_SIGNALS'
    assert r['current_proposal_state']=='UNFILLED_FUTURE_OUTCOME_UNKNOWN'


def test_empty_pipeline_binds_empty_metrics_without_fake_statistics():
    record=report();out=record['metrics']
    assert out['status']=='SKIPPED_NO_CANDIDATES' and out['rows']==[]
    assert out['strategy_contract']==record['batch']['strategy_contract']
    assert out['backtest_result_id']==record['backtest']['result_id']
    assert pipeline.selected_batch(record['batch'])==pipeline.selected_batch({k:v for k,v in record['batch'].items() if k!='metrics'})


@pytest.mark.parametrize('change',['parent','signal','strategy','version','archive','date','count','nan','win_rate','expectancy','current_closed'])
def test_wrong_or_corrupt_backtest_cannot_become_metrics(change):
    batch,backtest=source();backtest=deepcopy(backtest);row=backtest['results'][0]
    if change=='parent':backtest['primary_run_id']='2'*32
    elif change=='signal':row['current_signal_id']='0'*64
    elif change=='strategy':row['strategy_id']='other'
    elif change=='version':row['strategy_version']='1'
    elif change=='archive':backtest['archive_id']='0'*64
    elif change=='date':row['current_signal_session']='2026-10-22'
    elif change=='count':row['winning_signals']=1
    elif change=='nan':row['closed_trade_net_pnl']=float('nan')
    elif change=='win_rate':row['win_rate_pct']=100
    elif change=='expectancy':row['expectancy_net_per_closed_signal']=123
    else:row['completed_signal_ids']=[row['current_signal_id']]
    if change!='nan':backtest['result_id']=bt.digest({k:v for k,v in backtest.items() if k!='result_id'})
    with pytest.raises(ValueError):metrics.build_metrics(batch,backtest,primary_run_id='1'*32)


def test_a_valid_metric_report_cannot_be_substituted_between_backtests():
    batch,backtest=source();out=metrics.build_metrics(batch,backtest,primary_run_id='1'*32)
    changed=deepcopy(out);changed['backtest_result_id']='0'*64
    changed['result_id']=bt.digest({k:v for k,v in changed.items() if k!='result_id'})
    with pytest.raises(ValueError,match='lineage_or_result'):metrics.validate_metrics(changed,backtest,batch,'1'*32)


def test_pipeline_refuses_metrics_not_from_its_bound_backtest():
    from datetime import timedelta
    from test_strategy_pipeline import NOW
    record=report();record['metrics']=deepcopy(record['metrics']);record['metrics']['backtest_result_id']='f'*64
    with pytest.raises(ValueError):pipeline.assess(record,now=NOW+timedelta(seconds=10))


def test_wrong_metrics_are_rejected_before_any_publication_sql():
    from types import SimpleNamespace
    import pandas as pd
    from test_strategy_pipeline import archive,NOW
    _,session,value=archive(positive=True)
    bars=pd.DataFrame(dict(close=[51.69],volume=[123456]),index=['2026-10-23'])
    rows=pipeline.candidate_rows(value,SimpleNamespace(raw={'SRRK':bars,'SPY':bars}),NOW)
    batch,backtest=source()
    good=metrics.build_metrics(batch,backtest,primary_run_id='1'*32)
    class NoSql:
        autocommit=False
        def set_session(self,**kwargs):pass
        def cursor(self):pytest.fail('mismatched Metrics cannot reach publication SQL')
    with pytest.raises(ValueError,match='strategy_metrics_lineage_or_result_mismatch'):
        pipeline.publish_batch(NoSql(),value,rows,session,'1'*32,NOW,backtest=backtest,metrics=dict(good,backtest_result_id='0'*64))


def test_actual_metrics_api_projects_bound_rows_and_reports_pending_without_fallback(monkeypatch):
    from test_strategy_dashboard import view,extract,positive
    batch,backtest=source();out=metrics.build_metrics(batch,backtest,primary_run_id='1'*32)
    current=[dict(view([positive()]),_strategy_backtest=backtest,_strategy_metrics=out)]
    monkeypatch.setattr(pipeline,'enabled',lambda:True)
    monkeypatch.setattr(dashboard,'snapshot',lambda:current[0])
    api=extract('dashboards/dashboard_app.py','api_screener_metrics')
    actual=api();assert actual['rows']==out['rows'] and actual['backtest_result_id']==backtest['result_id']
    assert actual['metrics_result_id']==out['result_id'] and actual['strategy_version']=='2'
    assert '_strategy_backtest' not in actual and '_strategy_metrics' not in actual
    assert dashboard.api_payload('metrics',dict(filter='no_history'))['rows']==[]
    assert dashboard.api_payload('metrics',dict(filter='high_confidence'))['ok'] is False
    current[0]=view([positive()])
    assert api()['status']=='METRICS_PENDING' and api()['rows']==[] and api()['ok'] is False
    current[0]=dict(view([]),ok=False,status='ERROR')
    assert api()['status']=='ERROR' and not api()['ok']
