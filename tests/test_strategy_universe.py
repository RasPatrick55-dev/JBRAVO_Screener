"""Market-wide coverage, immutable data, cache revisions and portfolio parity."""
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import daily_strategy_scan as daily
from scripts import strategy_universe as u
from scripts import strategy_scanner as scanner
from scripts.research_dataset import load_dataset
from scripts.strategy_library import get_strategy
from test_research_testing import featured_source, simple_frame

pytestmark=pytest.mark.alpaca_optional
NOW=datetime.fromisoformat('2026-10-27T01:20:00+00:00')


def asset(symbol, **overrides):
    return dict(id='asset-'+symbol,symbol=symbol,**{'class':'us_equity'},status='active',
                tradable=True,exchange='NASDAQ',**overrides)


@pytest.fixture
def cfg():
    value=daily.load_config();value['universe']['batch_size']=2
    return value


class MarketProvider:
    def __init__(self, *, end='2026-10-26', rows=None, page_size=10000):
        self.receipts=[];self.requests=[];self.page_size=page_size
        self.assets=rows or [asset('AAA'),asset('BBB'),asset('SPY'),
            {**asset('NOPE'),'tradable':False},asset('NEW'),{**asset('OTCX'),'exchange':'OTC'}]
        days=pd.bdate_range('2023-01-03',end)
        self.calendar=[dict(date=d.strftime('%Y-%m-%d'),open='09:30',close='16:00')
            for d in list(days)+[pd.Timestamp(end)+pd.offsets.BDay(1)]]
        frame=featured_source(len(days))
        self.bars={}
        for symbol in {a['symbol'] for a in self.assets}:
            rows=[]
            for day,row in frame.iterrows():
                rows.append(dict(t=day.tz_localize(daily.NY).tz_convert('UTC').isoformat(),
                    o=float(row.open),h=float(row.high),l=float(row.low),c=float(row.close),v=5000))
            self.bars[symbol]=rows[-100:] if symbol=='NEW' else rows
        self.actions={};self.fail_symbol=None
    def get(self,url,params):
        self.requests.append((url,dict(params)))
        if url==daily.CALENDAR:
            value=self.calendar
        elif url==u.ASSETS:
            value=self.assets
        elif url.endswith('/bars'):
            symbols=params['symbols'].split(',')
            if self.fail_symbol in symbols:
                raise daily.DailyError('PROVIDER_HTTP_429')
            start=datetime.fromisoformat(params['start']);end=datetime.fromisoformat(params['end'])
            values=[(s,r) for s in symbols for r in self.bars[s] if start<=datetime.fromisoformat(r['t'])<=end]
            offset=int(params.get('page_token',0)); page=values[offset:offset+self.page_size]
            bars={}
            for symbol,row in page:
                bars.setdefault(symbol,[]).append(deepcopy(row))
            value=dict(bars=bars,next_page_token=str(offset+self.page_size) if offset+self.page_size<len(values) else None)
        else:
            symbols=set(params['symbols'].split(','))
            value=dict(corporate_actions={category:[e for e in rows if e['symbol'] in symbols]
                for category,rows in self.actions.items()},next_page_token=None)
        raw=json.dumps(value,allow_nan=False).encode()
        self.receipts.append(dict(endpoint=url,parameters=dict(params),**u.binding(raw)))
        return value,raw


def capture(tmp_path,cfg,provider,*,now=NOW,cache=None):
    root=tmp_path/now.astimezone(daily.NY).date().isoformat();root.mkdir()
    calendar=daily.normalize_calendar(provider.calendar)
    day=daily.signal_session(calendar,now)
    return root,u.capture_market(provider,cfg,calendar,day,root,now=now,cache_dir=cache)


def test_default_configuration_has_no_fixed_symbol_list():
    cfg=daily.load_config()
    assert u.market_config(cfg) and 'symbols' not in cfg
    assert cfg['universe']['source']=='alpaca_assets' and cfg['universe']['tradable'] is True


def test_asset_discovery_has_no_pilot_or_hundred_symbol_cap(cfg):
    rows=[asset('S'+str(i)) for i in range(1100)]+[asset('SPY')]
    selected,coverage=u.select_assets(rows,cfg,'2026-10-26',NOW)
    assert len(selected)==1101 and len(coverage['assets'])==1101
    cfg['universe']['max_symbols']=100
    with pytest.raises(ValueError,match='OVER_BUDGET'):
        u.select_assets(rows,cfg,'2026-10-26',NOW)


@pytest.mark.parametrize('change',['duplicate_symbol','duplicate_id','missing_tradable','nonboolean'])
def test_ambiguous_asset_metadata_is_a_failed_scan(cfg,change):
    rows=[asset('AAA'),asset('SPY')]
    if change=='duplicate_symbol':rows.append(deepcopy(rows[0]))
    if change=='duplicate_id':rows[1]['id']=rows[0]['id']
    if change=='missing_tradable':del rows[0]['tradable']
    if change=='nonboolean':rows[0]['tradable']='true'
    with pytest.raises(ValueError):u.select_assets(rows,cfg,'2026-10-26',NOW)


def test_sharded_capture_accounts_for_every_symbol_and_every_page(cfg,tmp_path):
    provider=MarketProvider(page_size=300)
    root,spec=capture(tmp_path,cfg,provider)
    data=load_dataset(spec)
    assert spec['universe']['symbols']==['AAA','BBB']
    assert data.coverage['status']=='COMPLETE' and data.coverage['discovered_count']==6
    assert data.coverage['excluded_count']==3
    assert data.coverage['exclusions']==dict(NEW='INSUFFICIENT_250_SESSION_HISTORY',NOPE='NOT_TRADABLE',OTCX='UNSUPPORTED_SIP_EXCHANGE')
    assert data.coverage['evaluated_count']+data.coverage['excluded_count']+1==6
    assert list(data.raw)==['AAA','BBB','SPY']
    assert data.raw['AAA'].index[-1].strftime('%Y-%m-%d')=='2026-10-26'
    assert any('page_token' in p for url,p in provider.requests if url.endswith('/bars'))
    assert all(len(p['symbols'].split(','))<=2 for url,p in provider.requests if url.endswith('/bars'))
    assert all(p['feed']=='sip' and p['adjustment'] in ('raw','split') for url,p in provider.requests if url.endswith('/bars'))
    assert (root/'coverage-pending.json').is_file()


@pytest.mark.parametrize('failure,reason',[('gap','MISSING_HISTORICAL_SESSIONS'),('current','MISSING_CURRENT_SESSION'),
    ('zero','ZERO_VOLUME_HISTORY'),('split','SHARE_ACTION_ADAPTER_REQUIRED'),('action','UNSUPPORTED_CORPORATE_ACTION_FORWARD_SPLITS')])
def test_data_exclusions_are_explicit_and_never_poison_other_symbols(cfg,tmp_path,failure,reason):
    provider=MarketProvider()
    if failure=='gap':provider.bars['AAA'].pop(300)
    if failure=='current':provider.bars['AAA'].pop()
    if failure=='zero':provider.bars['AAA'][300]['v']=0
    if failure=='action':provider.actions={'forward_splits':[dict(id='A1',symbol='AAA',ex_date='2026-05-01')]}
    if failure=='split':
        original=provider.get
        def get(url,params):
            value,raw=original(url,params)
            if url.endswith('/bars') and params['adjustment']=='split' and 'AAA' in value['bars']:
                for row in value['bars']['AAA'][:300]:
                    for key in ('o','h','l','c'):row[key]/=2
                raw=json.dumps(value).encode()
            return value,raw
        provider.get=get
    root,spec=capture(tmp_path,cfg,provider);data=load_dataset(spec)
    assert data.coverage['exclusions']['AAA']==reason and spec['universe']['symbols']==['BBB']


def test_provider_failure_never_archives_partial_universe(cfg,tmp_path):
    provider=MarketProvider();provider.fail_symbol='BBB';archived=[]
    with pytest.raises(daily.DailyError,match='429'):
        daily.run_daily(cfg,provider,tmp_path,now=NOW,archive=lambda *args:archived.append(args))
    assert not archived and (tmp_path/'2026-10-26/failure.json').is_file()
    assert not (tmp_path/'2026-10-26/completion.json').exists()


def test_cache_incremental_overlap_and_revised_history_force_full_refresh(cfg,tmp_path):
    cache=tmp_path/'cache'
    capture(tmp_path,cfg,MarketProvider(),cache=cache)
    p=MarketProvider(end='2026-10-27')
    # Corrected older prices in the overlap must invalidate cached prefixes.
    changed=p.bars['AAA'][-4]
    for key in ('o','h','l','c'):changed[key]*=1.01
    root,spec=capture(tmp_path,cfg,p,now=datetime.fromisoformat('2026-10-28T01:20:00+00:00'),cache=cache)
    coverage=load_dataset(spec).coverage
    assert coverage['full_refresh_batches']==1
    assert any('revised' in f.name for f in (root/'provider-responses').iterdir())
    assert any(params['start'].startswith(cfg['warmup_start']) for url,params in p.requests if url.endswith('/bars'))


def test_unchanged_cached_history_fetches_only_recent_sessions(cfg,tmp_path):
    cache=tmp_path/'cache';first,_=capture(tmp_path,cfg,MarketProvider(),cache=cache)
    p=MarketProvider(end='2026-10-27')
    root,spec=capture(tmp_path,cfg,p,now=datetime.fromisoformat('2026-10-28T01:20:00+00:00'),cache=cache)
    coverage=load_dataset(spec).coverage
    assert coverage['cache_hits']>=2 and coverage['full_refresh_batches']==0
    aaa=[params for url,params in p.requests if url.endswith('/bars') and 'AAA' in params['symbols'].split(',')]
    assert aaa and all(not p['start'].startswith(cfg['warmup_start']) for p in aaa)
    original=first/'capture/analysis/symbols'/f'{u.canonical_hash("AAA")}-0.json.gz'
    repeated=root/'capture/analysis/symbols'/original.name
    assert original.samefile(repeated)
    object_name=u.binding(original.read_bytes())['sha256']+'.json.gz'
    (cache/'objects'/object_name).unlink()
    # Immutable session links survive eviction of the disposable object name.
    assert original.is_file() and repeated.is_file()
    assert load_dataset(spec).raw['AAA'].index[-1].strftime('%Y-%m-%d')=='2026-10-27'


def test_new_listing_with_250_sessions_can_be_evaluated_without_2023_history(cfg,tmp_path):
    provider=MarketProvider();provider.bars['NEW']=deepcopy(provider.bars['AAA'][-300:])
    root,spec=capture(tmp_path,cfg,provider)
    assert 'NEW' in spec['universe']['symbols']
    result=scanner.run_scan(spec,as_of='2026-10-26',output=root/'scan',now=NOW)
    assert result['coverage']['evaluated_count']==3
    rows=json.loads(gzip.decompress((root/'scan'/cfg['strategy_id']/'latest_observations.json.gz').read_bytes()))
    assert len(rows)==3 and next(r for r in rows if r['symbol']=='NEW')['session']=='2026-10-26'


def test_no_evaluable_data_cannot_be_published_as_no_buy_signals(cfg,tmp_path):
    provider=MarketProvider()
    for symbol in ('AAA','BBB'):provider.bars[symbol]=provider.bars[symbol][-100:]
    archived=[]
    with pytest.raises(ValueError,match='NO_EVALUABLE'):
        daily.run_daily(cfg,provider,tmp_path,now=NOW,archive=lambda *args:archived.append(args))
    assert archived==[]


def test_capture_then_scan_complete_current_decisions_and_real_no_signal_kernel(cfg,tmp_path):
    root,spec=capture(tmp_path,cfg,MarketProvider())
    result=scanner.run_scan(spec,as_of='2026-10-26',output=root/'scan',now=NOW)
    assert result['coverage']['status']=='COMPLETE' and result['top_candidates']==[]
    path=root/'scan'/cfg['strategy_id']/'latest_observations.json.gz'
    rows=json.loads(gzip.decompress(path.read_bytes()))
    assert [r['symbol'] for r in rows]==['AAA','BBB']
    assert all(r['reason']=='NO_QUALIFYING_SETUP' for r in rows)


def test_new_symbol_signal_survives_verified_primary_backtest_and_metrics(cfg,tmp_path,monkeypatch):
    from scripts import strategy_pipeline as pipeline,strategy_backtest,strategy_metrics
    from scripts.research_dataset import canonical_hash
    root,spec=capture(tmp_path,cfg,MarketProvider())
    def prepare(self,raw,split,fold):
        frame=raw.loc[raw.index>=fold['test_start']].copy()
        frame['score']=np.nan;frame['ATR14']=2.;frame['ema20']=1.
        frame['macd']=1.;frame['macd_signal']=0.;frame['rsi']=50.;frame['entry_ready']=True
        frame['entry_details']=[dict(signal=False,pullback_start=None,failures=['fixture']) for _ in frame.index]
        frame.iloc[-1,frame.columns.get_loc('score')]=5.
        frame.iat[-1,frame.columns.get_loc('entry_details')]=dict(signal=True,pullback_start='2026-10-20',failures=[])
        return frame
    # Controlled positive input exercises identity/competition/attribution without
    # manufacturing any real provider signal or submitting an order.
    monkeypatch.setattr(type(get_strategy(cfg['strategy_id'])),'prepare',prepare)
    summary=scanner.run_scan(spec,as_of='2026-10-26',output=root/'scan',now=NOW)
    summary=json.loads(json.dumps(summary))
    session=pipeline.session_context(MarketProvider().calendar,NOW)
    value=dict(schema='jbravo.daily-strategy-scan.v1',status='COMPLETED_PAPER_SCREEN',
        run_id=canonical_hash(dict(schema='jbravo.daily-strategy-archive-id.v1',pipeline=daily.PIPELINE,
            strategy_id=cfg['strategy_id'],signal_session='2026-10-26')),
        scan_run_id=summary['run_id'],signal_session='2026-10-26',pipeline=daily.PIPELINE,
        strategy_id=cfg['strategy_id'],configuration_sha256=canonical_hash(cfg),
        scan_manifest_sha256=u.binding((root/'scan/manifest.json').read_bytes())['sha256'],summary=summary,
        handoff=json.loads((root/'scan/paper_handoff.json').read_bytes()),output_root=str(root),
        research_qualified=False,execution_authorized=False,order_submission_authorized=False)
    spec_sha=u.binding((root/'study.json').read_bytes())['sha256']
    pipeline.validate_archive((spec_sha,canonical_hash(value),value),cfg,session)
    base=tmp_path/'base';(base/'config').mkdir(parents=True)
    (base/'config/strategy_scan_daily.json').write_text(json.dumps(cfg))
    (base/'config/strategy_pipeline.json').write_text(json.dumps(dict(producer_root=str(root.parent))))
    dataset=pipeline.verified_dataset(value,spec_sha,base_dir=base)
    selected=pipeline.selection_batch(value,session)
    assert selected['proposals'][0]['symbol']=='AAA' and selected['coverage']==summary['coverage']
    rows=pipeline.candidate_rows(value,dataset,NOW)
    assert rows[0]['symbol']=='AAA' and rows[0]['universe_count']==2
    backtest=strategy_backtest.run_candidate_backtest(selected,primary_run_id='a'*32,dataset=dataset,specification=spec)
    metrics=strategy_metrics.build_metrics(selected,backtest,primary_run_id='a'*32)
    assert backtest['status']=='COMPLETED_CANDIDATE_BACKTEST' and backtest['input_candidates']==2
    assert metrics['status']=='COMPLETED_STRATEGY_METRICS' and metrics['backtest_result_id']==backtest['result_id']
    assert backtest['parameters']['universe']==summary['coverage']
    assert len(strategy_backtest.canonical(backtest))<24*1024
    assert all(r['symbol'] in {'AAA','BBB'} for r in metrics['rows'])


def test_bound_shard_and_coverage_tampering_are_refused(cfg,tmp_path):
    root,spec=capture(tmp_path,cfg,MarketProvider())
    path=root/'capture/analysis/coverage.json';body=path.read_bytes();path.write_bytes(body+b' ')
    with pytest.raises(ValueError,match='COVERAGE_CHANGED'):load_dataset(spec)
    path.write_bytes(body)
    manifest=json.loads((root/'capture/analysis/manifest.json').read_bytes())
    shard=root/'capture'/manifest['shards']['AAA'];shard.write_bytes(shard.read_bytes()+b' ')
    with pytest.raises(ValueError,match='MARKET_MEMBER_CHANGED'):load_dataset(spec)


def test_sparse_market_replay_preserves_competition_consumption_fills_and_exits(monkeypatch):
    strategy=get_strategy('trend_pullback_reclaim_qualifying_close_v2')
    frame=simple_frame(count=35)
    frame['entry_details']=[dict(signal=True,pullback_start='2023-12-29',failures=[]) for _ in frame.index]
    frame.loc[frame.index[[0,1,20]],'score']=5.
    no_signal=frame.copy(deep=True);no_signal['score']=np.nan
    frames=dict(AAA=frame,BBB=frame.copy(deep=True),CCC=no_signal)
    fold=dict(id='episode',warmup_start='2023-01-03',test_start='2024-01-02',test_end=frame.index[-1].strftime('%Y-%m-%d'))
    spec=dict(universe=dict(symbols=list(frames),benchmark='SPY',excluded={}),
        portfolio=dict(initial_cash=100000.,alloc_pct=.05,top_n=1,max_positions=1,trail_pct=.03,max_hold_days=7,atr_multiple=1.))
    calendar=pd.DataFrame([dict(date=d.strftime('%Y-%m-%d'),open='09:30',close='16:00')
        for d in pd.bdate_range(frame.index[0],frame.index[-1]+pd.offsets.BDay(1))]).set_index('date')
    legacy=SimpleNamespace(raw=frames,features=frames,dividends=(),calendar=calendar)
    monkeypatch.setattr(type(strategy),'prepare',lambda self,raw,*a:raw.copy(deep=True))
    market=SimpleNamespace(market_wide=True,calendar=calendar,coverage=dict(policy={
        'source':'alpaca_assets','status':'active','asset_class':'us_equity','tradable':True}),
        prepare=lambda symbol,*args:frames[symbol].copy(deep=True),member=lambda s:dict(actions=[]))
    cost=dict(id='base',fee_per_fill=1.,slippage_bps=5.)
    normal=scanner.evaluate_strategy(legacy,spec,strategy,fold,cost,fold['test_end'])
    sparse=scanner.evaluate_strategy(market,spec,strategy,fold,cost,fold['test_end'])
    # Episode IDs differ by declared population policy; economics and decisions must match.
    def economics(result):return [{k:v for k,v in f.items() if k!='signal_id'} for f in result['fills']]
    assert economics(normal)==economics(sparse)
    assert normal['metrics']==sparse['metrics']
    assert normal['funnel']==sparse['funnel']
    assert [(r['symbol'],r['reason']) for r in normal['latest_observations']]==[(r['symbol'],r['reason']) for r in sparse['latest_observations']]
    assert any(r['reason']=='REARM_REQUIRES_PULLBACK_AFTER_CONSUMED_CLOSE' for r in sparse['observations'])
