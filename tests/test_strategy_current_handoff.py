"""A research portfolio's occupied slots cannot suppress today's handoff."""
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from scripts import strategy_scanner as scanner,strategy_library,strategy_backtest,strategy_metrics
from scripts.research_strategies import CHALLENGER
from test_research_testing import simple_frame

pytestmark=pytest.mark.alpaca_optional


def case(monkeypatch, *, consumed_before=False, symbols=('BBB',)):
    # Prepared rows isolate selection/account state from indicator computation.
    monkeypatch.setattr(strategy_library.Strategy,'prepare',lambda self,raw,split,fold:raw.copy())
    held=simple_frame();held['entry_details']=[dict(signal=True,failures=[],pullback_start='2023-12-28') for _ in held.index]
    data={'AAA':held}
    for i,symbol in enumerate(symbols):
        frame=simple_frame(signal=consumed_before)
        frame.loc[frame.index[-1],'score']=10.-i
        frame['entry_details']=[dict(signal=True,failures=[],pullback_start='2023-12-28') for _ in frame.index]
        data[symbol]=frame
    calendar=pd.DataFrame(dict(open='09:30',close='16:00'),index=pd.bdate_range('2024-01-02',periods=5).strftime('%Y-%m-%d'))
    dataset=SimpleNamespace(raw=data,features=data,calendar=calendar,dividends=(),
        provenance=dict(feed='sip',currency='USD',execution_prices='raw',feature_prices='split',dataset_sha256='d'*64))
    portfolio=dict(initial_cash=1000.,alloc_pct=.5,max_positions=1,top_n=1,trail_pct=.03,max_hold_days=7,atr_multiple=.000001)
    fold=dict(id='test',warmup_start='2023-01-01',test_start='2024-01-02',test_end='2024-01-05')
    cost=dict(id='base',fee_per_fill=0.,slippage_bps=0.)
    spec=dict(schema='jbravo.strategy-study.v1',purpose='exploratory_research',capture_manifest='unused',
        actions_capture=dict(path='unused',sha256='f'*64),universe=dict(symbols=list(data),benchmark='SPY',excluded={}),
        folds=[fold],strategies=[CHALLENGER],cost_cases=[cost],portfolio=portfolio,max_runs=2,max_seconds=60,
        research_qualified=False,execution_authorized=False)
    strategy=strategy_library.get_strategy(CHALLENGER)
    result=scanner.evaluate_strategy(dataset,spec,strategy,fold,cost,fold['test_end'])
    return dataset,spec,strategy,result


def test_current_qualified_signal_survives_full_historical_book(monkeypatch):
    _,_,_,result=case(monkeypatch)
    assert [p['symbol'] for p in result['top_candidates']]==['BBB']
    assert result['current_selection']['historical_terminal_proposals']==0
    assert result['current_selection']['qualifying_symbols']==1
    held=next(t for t in result['lifecycle'] if t['symbol']=='AAA')
    current=next(t for t in result['lifecycle'] if t['symbol']=='BBB')
    assert held['status']=='OPEN' and held['bought_quantity']==5
    assert current['status']=='UNFILLED_PROPOSAL' and current['fills']==[]
    assert current['bought_quantity']==current['sold_quantity']==0
    latest={r['symbol']:r for r in result['latest_observations']}
    assert latest['AAA']['reason']=='HELD' and not latest['AAA']['selected']
    assert latest['BBB']['selected'] and not latest['BBB']['historical_portfolio_selected']
    assert all(f['symbol']=='AAA' for f in result['fills'])


def test_current_ranking_retains_top_n_and_stable_score_order(monkeypatch):
    _,spec,_,result=case(monkeypatch,symbols=('BBB','CCC','DDD'))
    assert len(result['top_candidates'])==spec['portfolio']['top_n']==1
    assert result['top_candidates'][0]['symbol']=='BBB'
    assert result['current_selection']['qualifying_symbols']==3
    assert result['top_candidates'][0]['order_submission_authorized'] is False


def test_current_handoff_does_not_rewrite_historical_fills_or_metrics(monkeypatch):
    dataset,spec,strategy,result=case(monkeypatch)
    historical=scanner.ScannerPortfolio(dataset.raw,strategy=strategy,episode_id=result['episode_id'],
        dividends=(),trade_cost=0.,slippage=0.,**spec['portfolio'])
    historical.run()
    assert result['fills']==historical.fills
    assert result['ledger']==historical.reconciliations
    assert result['metrics']==historical.summary()
    assert not [p for p in historical.intents if p['signal_session']=='2024-01-05']
    assert [p['symbol'] for p in result['top_candidates']]==['BBB']


def test_rearming_and_consumption_still_reject_old_setup(monkeypatch):
    _,_,_,result=case(monkeypatch,consumed_before=True)
    assert result['top_candidates']==[]
    latest=next(r for r in result['latest_observations'] if r['symbol']=='BBB')
    assert latest['reason']=='REARM_REQUIRES_PULLBACK_AFTER_CONSUMED_CLOSE'
    assert not latest['selected'] and not latest['consumed']


def test_positive_current_handoff_reaches_exact_backtest_and_metrics(monkeypatch):
    dataset,spec,strategy,scan=case(monkeypatch)
    description=strategy.describe()
    batch=dict(schema='jbravo.strategy-primary-batch.v1',proposals=scan['top_candidates'],candidate_count=1,
        strategy_contract=dict(strategy_id=strategy.id,strategy_version=strategy.version,definition_sha256=description['definition_sha256']),
        session=dict(signal_session='2024-01-05'),archive_id='a'*64,archive_sha256='b'*64,scan_run_id='c'*64,
        order_submission_authorized=False)
    report=strategy_backtest.run_candidate_backtest(batch,primary_run_id='1'*32,dataset=dataset,specification=spec)
    assert report['status']=='COMPLETED_CANDIDATE_BACKTEST'
    assert report['results'][0]['symbol']=='BBB' and report['results'][0]['closed_signals']==0
    assert report['results'][0]['current_proposal_state']=='UNFILLED_FUTURE_OUTCOME_UNKNOWN'
    metrics=strategy_metrics.build_metrics(batch,report,primary_run_id='1'*32)
    strategy_metrics.validate_metrics(metrics,report,batch,'1'*32)
    assert metrics['input_candidates']==1 and report['strategy_definition']==json.loads(json.dumps(description))
