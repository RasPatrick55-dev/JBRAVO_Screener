"""Primary route: exact strategy identity, explicit empty batches, honest handoff."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from scripts import strategy_pipeline as p
from scripts import daily_strategy_scan as daily
from scripts import pipeline_postflight as health
from scripts.research_dataset import canonical_hash
from scripts.strategy_library import get_strategy
from scripts.strategy_scanner import SOURCE_FILES

pytestmark = pytest.mark.alpaca_optional
NOW = datetime.fromisoformat('2026-10-26T03:45:00+00:00')
ROOT = Path(__file__).resolve().parents[1]


def calendar():
    return [dict(date=d, open='09:30', close='16:00') for d in
        ('2026-10-22', '2026-10-23', '2026-10-26', '2026-10-27')]


def archive(*, positive=False):
    cfg = daily.load_config()
    session = p.session_context(calendar(), NOW)
    day = session['signal_session']
    episode = 'a'*64
    proposals = []
    if positive:
        symbol = 'SRRK'
        proposals = [dict(strategy_id=p.STRATEGY, strategy_version='2', symbol=symbol,
            signal_id=canonical_hash(dict(strategy_id=p.STRATEGY, version='2', episode_id=episode,
                symbol=symbol, signal_session=day)), side='BUY', signal_session=day,
            signal_available_at_assumed=session['signal_available_at'], next_session=session['next_session'],
            next_open_at=session['next_open_at'], score=4.2, signal_close_raw=51.69, signal_atr14_raw=1.5,
            allocation_pct=.05, max_positions=4, execution_model=session['execution_model'],
            status='PAPER_PROPOSAL', paper_only=True, order_submission_authorized=False)]
    summary = dict(status='COMPLETED_PAPER_SIMULATION', as_of=day, run_id='b'*64,
        source_sha256={n:p.digest((ROOT/'scripts'/n).read_bytes()) for n in SOURCE_FILES},
        strategies=[dict(strategy=get_strategy(p.STRATEGY).describe(), as_of=day, episode_id=episode)],
        top_candidates=proposals, research_qualified=False, order_submission_authorized=False,
        dataset_sha256='c'*64, broker_operations=0)
    value = dict(schema='jbravo.daily-strategy-scan.v1', status='COMPLETED_PAPER_SCREEN', pipeline=daily.PIPELINE,
        run_id=canonical_hash(dict(schema='jbravo.daily-strategy-archive-id.v1', pipeline=daily.PIPELINE,
            strategy_id=p.STRATEGY, signal_session=day)), strategy_id=p.STRATEGY, signal_session=day,
        scan_run_id=summary['run_id'], configuration_sha256=canonical_hash(cfg), summary=summary,
        handoff=dict(schema='jbravo.strategy-paper-proposals.v1', as_of=day, scan_run_id=summary['run_id'],
            proposals=proposals, paper_only=True, order_submission_authorized=False),
        research_qualified=False, execution_authorized=False, order_submission_authorized=False)
    return cfg, session, value


def row(value):
    return ('d'*64, p.digest(p.encoded(value)), value)


@pytest.mark.parametrize('positive', [False, True])
def test_validate_exact_archive_and_preserve_signal_identity(positive):
    cfg, session, value = archive(positive=positive)
    actual = p.validate_archive(row(value), cfg, session)
    bars = pd.DataFrame(dict(close=[51.69], volume=[123456]), index=['2026-10-23'])
    dataset = SimpleNamespace(raw={'SRRK':bars, 'SPY':bars}, provenance={})
    rows = p.candidate_rows(actual, dataset, NOW)
    assert len(rows) == int(positive)
    if positive:
        lineage = rows[0]['score_breakdown']
        assert lineage['proposal'] == value['handoff']['proposals'][0]
        assert rows[0]['score'] == 4.2 and rows[0]['close'] == 51.69
        assert lineage['archive_sha256'] == row(value)[1]
        assert rows[0]['ml_weight_used'] == 0


@pytest.mark.parametrize('change', ['session','source','config','signal','next_open','order','population','handoff'])
def test_changed_or_mixed_archives_fail_closed(change):
    cfg, session, value = archive(positive=True)
    if change == 'session': value['signal_session'] = '2026-10-22'
    if change == 'source': value['summary']['source_sha256']['screener.py'] = '0'*64
    if change == 'config': value['configuration_sha256'] = '0'*64
    if change == 'signal': value['handoff']['proposals'][0]['signal_id'] = '0'*64
    if change == 'next_open': value['handoff']['proposals'][0]['next_open_at'] = '2026-10-26T08:00:00+00:00'
    if change == 'order': value['order_submission_authorized'] = True
    if change == 'population': value['handoff']['proposals'].append(deepcopy(value['handoff']['proposals'][0]))
    if change == 'handoff': value['handoff']['scan_run_id'] = '0'*64
    with pytest.raises(ValueError): p.validate_archive(row(value), cfg, session)


def test_hash_failure_and_missing_archive_never_fall_back():
    cfg, session, value = archive()
    with pytest.raises(ValueError, match='not_archived'): p.validate_archive(None,cfg,session)
    with pytest.raises(ValueError, match='hash_mismatch'): p.validate_archive(('d'*64,'0'*64,value),cfg,session)


def test_calendar_keeps_holidays_dst_and_rejects_missed_open():
    session = p.session_context(calendar(), NOW)
    assert session['signal_session'] == '2026-10-23'
    assert session['next_open_at'] == '2026-10-26T13:30:00+00:00'
    holiday = [dict(date=d,open='09:30',close='16:00') for d in ('2026-12-24','2026-12-28')]
    later = p.session_context(holiday,datetime.fromisoformat('2026-12-26T03:45:00+00:00'))
    assert later['next_open_at'] == '2026-12-28T14:30:00+00:00'
    with pytest.raises(ValueError,match='open_already_passed'):
        p.session_context(calendar(), datetime.fromisoformat('2026-10-26T14:00:00+00:00'))


def report():
    cfg,session,value = archive()
    from scripts.strategy_backtest import report_from_replay
    batch = p.selection_batch(value,session)
    spec=dict(universe=dict(symbols=cfg['symbols'],benchmark=cfg['benchmark']),portfolio=cfg['portfolio'],
        cost_cases=[cfg['cost_case']],folds=[])
    data=SimpleNamespace(provenance=dict(feed='sip',currency='USD',execution_prices='raw',feature_prices='split'))
    backtest=report_from_replay(batch,None,primary_run_id='a'*32,dataset=data,specification=spec)
    batch=dict(batch,backtest=backtest)
    return dict(schema=p.SCHEMA,run_id='a'*32,invocation_id='a'*32,started_at=NOW.isoformat(),
        finished_at=(NOW+timedelta(seconds=5)).isoformat(),session=session,batch=batch,
        handoff=value['handoff'],backtest=backtest,pipeline_rc=0,broker_operations=0,entry_ready=False,
        order_submission_authorized=False,research_qualified=False,
        stages=dict(screener='COMPLETED',backtest='SKIPPED_NO_CANDIDATES',
            metrics='RESEARCH_SIMULATION_ONLY',labels='NOT_USED_BY_STRATEGY',ranker_eval='NOT_USED_BY_STRATEGY',
            handoff='COMMITTED_AND_SEPARATE_READONLY_VERIFIED'))


def test_empty_screen_is_healthy_and_not_entry_ready():
    actual = p.assess(report(),now=NOW+timedelta(seconds=10))
    assert actual['status'] == 'ok' and actual['candidate_count'] == 0
    assert actual['outcome'] == 'NO_BUY_SIGNALS' and actual['entry_ready'] is False
    assert actual['entry_block_reason'] == 'strategy_regular_open_executor_not_connected'


@pytest.mark.parametrize('change',['count','outcome','entry','stage','stale'])
def test_postflight_does_not_fake_ml_success_or_fabricate_candidate(change):
    value = report()
    if change == 'count': value['batch']['candidate_count'] = 1
    if change == 'outcome': value['batch']['outcome'] = 'BUY_SIGNALS'
    if change == 'entry': value['entry_ready'] = True
    if change == 'stage': value['stages']['ranker_eval'] = 'COMPLETED'
    now = NOW + timedelta(hours=3) if change == 'stale' else NOW+timedelta(seconds=10)
    with pytest.raises(ValueError): p.assess(value,now=now)


def test_empty_db_batch_keeps_timestamp_and_never_reads_legacy_latest():
    from scripts.strategy_pipeline_store import candidate_frame
    cur=Mock();cur.__enter__=Mock(return_value=cur);cur.__exit__=Mock(return_value=False)
    cur.fetchone.return_value=(date(2026,10,23),NOW,0,dict(proposals=[]))
    cur.fetchall.return_value=[];cur.description=[('symbol',),('score_breakdown',)]
    conn=Mock();conn.cursor.return_value=cur
    frame,stamp=candidate_frame(conn,date(2026,10,23))
    assert frame.empty and stamp == NOW
    assert 'screener_strategy_batches_app' in cur.execute.call_args_list[0].args[0]
    assert cur.execute.call_args_list[1].args[1] == [date(2026,10,23),NOW]
    assert 'MAX(' not in cur.execute.call_args_list[1].args[0]


def test_retried_or_racing_nonempty_batch_cannot_masquerade_as_empty():
    from scripts.strategy_pipeline_store import candidate_frame
    cur=Mock();cur.__enter__=Mock(return_value=cur);cur.__exit__=Mock(return_value=False)
    cur.fetchone.return_value=(date(2026,10,23),NOW,1,dict(proposals=[dict(symbol='SRRK')]))
    cur.fetchall.return_value=[];cur.description=[('symbol',),('score_breakdown',)]
    conn=Mock();conn.cursor.return_value=cur
    with pytest.raises(ValueError,match='population_unavailable'):
        candidate_frame(conn,date(2026,10,23),exact_run_ts=NOW)


def test_premarket_executor_cannot_waive_regular_open_contract(monkeypatch):
    from scripts import executor_health as executor
    monkeypatch.setattr(health,'check_health',lambda *a,**k: p.assess(report(),now=NOW+timedelta(seconds=10)))
    result=executor.preflight(now=NOW+timedelta(seconds=10),require_session=True,
        for_entry=True,paper_research_entries=True,paper_base_url='https://paper-api.alpaca.markets')
    assert result['status']=='blocked'
    assert result['reasons']==['strategy_regular_open_executor_not_connected']


def test_primary_pipeline_dispatches_strategy_before_legacy_scan(monkeypatch):
    from scripts import run_pipeline
    called=[]
    monkeypatch.setattr(p,'main',lambda args: called.append(args) or 0)
    monkeypatch.setattr(run_pipeline,'_refresh_logger',lambda: pytest.fail('legacy pipeline must not run'))
    args=['--handoff-invocation','a'*32]
    assert run_pipeline.main(args)==0 and called==[args]
