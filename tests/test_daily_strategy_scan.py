"""Daily capture, calendar, replay and immutable research archive contracts."""
from datetime import datetime,timezone
import json
import pandas as pd
import pytest
from scripts import daily_strategy_scan as daily
from test_research_testing import featured_source

pytestmark=pytest.mark.alpaca_optional

@pytest.fixture
def cfg():return daily.load_config()

class CapturedProvider:
    def __init__(self,cfg,*,missing=False,unsupported=False):
        self.receipts=[];self.requests=[];self.missing=missing;self.unsupported=unsupported;self.cfg=cfg
        days=pd.bdate_range('2023-01-03','2026-10-23')
        self.calendar=[dict(date=d.strftime('%Y-%m-%d'),open='09:30:00',close='16:00:00',timezone='America/New_York')
            for d in list(days)+[pd.Timestamp('2026-10-26'),pd.Timestamp('2026-10-27')]]
        source=featured_source(len(days));records=[]
        for symbol in cfg['symbols']+[cfg['benchmark']]:
            for day,row in source.iterrows():
                records.append(dict(symbol=symbol,session=day.strftime('%Y-%m-%d'),o=row.open,h=row.high,l=row.low,c=row.close,v=5000))
        self.frames={kind:pd.DataFrame(records) for kind in ('raw','split')}
        self.actions=dict(corporate_actions={},next_page_token=None)
    def get(self,url,params):
        self.requests.append((url,dict(params)))
        if url==daily.CALENDAR:value=self.calendar
        elif url.endswith('/bars'):
            frame=self.frames[params['adjustment']]
            frame=frame[frame.symbol.isin(self.cfg['symbols']+[self.cfg['benchmark']])]
            if self.missing:frame=frame.drop(frame.index[-1])
            groups={}
            for row in frame.to_dict('records'):
                stamp=datetime.fromisoformat(row['session']).replace(tzinfo=daily.NY).astimezone(timezone.utc).isoformat()
                groups.setdefault(row['symbol'],[]).append(dict(t=stamp,**{k:row[k] for k in ('o','h','l','c','v')}))
            value=dict(bars=groups,next_page_token=None)
        else:
            value=json.loads(json.dumps(self.actions))
            if self.unsupported:value['corporate_actions']['forward_splits']=[dict(id='unexpected',symbol='ALK',ex_date='2026-05-27')]
        raw=json.dumps(value).encode();self.receipts.append(dict(endpoint=url,parameters=params,**daily.binding(raw)))
        return value,raw

def test_config_is_explicit_first_paper_cohort(cfg):
    assert len(cfg['symbols'])==10 and cfg['feed']=='sip'
    assert cfg['order_submission_authorized'] is False
    assert cfg['strategy_id']=='trend_pullback_reclaim_qualifying_close_v2'

@pytest.mark.parametrize('stamp,expected',[
    ('2026-10-27T01:20:00+00:00','2026-10-26'),
    ('2026-11-03T01:20:00+00:00','2026-11-02'),
    ('2026-10-26T01:20:00+00:00',None)])
def test_utc_schedule_is_after_cutoff_across_dst_and_skips_weekend(stamp,expected):
    rows=daily.normalize_calendar([dict(date=d,open='09:30',close='16:00') for d in
        ('2026-10-23','2026-10-26','2026-10-27','2026-11-02','2026-11-03')])
    assert daily.signal_session(rows,datetime.fromisoformat(stamp))==expected

def test_early_session_clock_is_refused():
    rows=daily.normalize_calendar([dict(date=d,open='09:30',close='16:00') for d in ('2026-10-23','2026-10-26','2026-10-27')])
    with pytest.raises(daily.DailyError,match='NOT_FINAL'):
        daily.signal_session(rows,datetime.fromisoformat('2026-10-26T23:00:00+00:00'))
    with pytest.raises(daily.DailyError,match='AWARE'):
        daily.signal_session(rows,datetime(2026,10,26))

def test_daily_fresh_capture_runs_same_kernel_and_archives_valid_zero(cfg,tmp_path):
    provider=CapturedProvider(cfg);archived=[]
    def archive(summary,root,config):
        archived.append(summary)
        assert (root/'scan/manifest.json').is_file()
        assert config==cfg
        return dict(status='TEST_ARCHIVE_VERIFIED')
    value=daily.run_daily(cfg,provider,tmp_path,now=datetime.fromisoformat('2026-10-25T22:30:00+00:00'),bootstrap_latest=True,archive=archive)
    assert value['status']=='DAILY_STRATEGY_SCAN_COMPLETE' and value['candidates']==0
    assert len(archived)==1 and value['provider_gets']==4
    assert archived[0]['strategies'][0]['funnel']['observations']==len(pd.bdate_range(cfg['episode_start'],'2026-10-23'))*len(cfg['symbols'])
    raw=[p for u,p in provider.requests if u.endswith('/bars')]
    assert [p['adjustment'] for p in raw]==['raw','split'] and all(p['feed']=='sip' for p in raw)
    assert all(p['asof']=='2026-10-25' for p in raw)
    assert json.loads((tmp_path/'2026-10-23/scan/paper_handoff.json').read_bytes())['order_submission_authorized'] is False
    again=CapturedProvider(cfg)
    with pytest.raises(FileExistsError):
        daily.run_daily(cfg,again,tmp_path,now=datetime.fromisoformat('2026-10-25T22:30:00+00:00'),bootstrap_latest=True,archive=archive)
    assert len(again.requests)==1 and len(archived)==1

def test_weekend_never_fetches_bars_or_writes_outputs(cfg,tmp_path):
    provider=CapturedProvider(cfg)
    value=daily.run_daily(cfg,provider,tmp_path/'unused',now=datetime.fromisoformat('2026-10-26T01:20:00+00:00'))
    assert value['status']=='SKIPPED_NON_TRADING_DAY' and len(provider.requests)==1
    assert not (tmp_path/'unused').exists()

@pytest.mark.parametrize('problem',['missing','unsupported'])
def test_invalid_data_never_archives_and_keeps_failed_claim(cfg,tmp_path,problem):
    provider=CapturedProvider(cfg,**{problem:True});calls=[]
    with pytest.raises(ValueError):
        daily.run_daily(cfg,provider,tmp_path,now=datetime.fromisoformat('2026-10-25T22:30:00+00:00'),bootstrap_latest=True,archive=lambda *a:calls.append(a))
    assert calls==[] and (tmp_path/'2026-10-23/claim.json').is_file()
    assert (tmp_path/'2026-10-23/failure.json').is_file()

def test_pagination_consumes_all_pages_and_refuses_repeated_token():
    class P:
        def __init__(self):self.calls=[]
        def get(self,url,params):
            self.calls.append(params);return dict(next_page_token='same'),b'{}'
    p=P()
    with pytest.raises(daily.DailyError,match='PAGINATION'):
        list(daily.pages(p,daily.DATA+'/v2/stocks/bars',{}))
    assert len(p.calls)==2 and p.calls[1]['page_token']=='same'

def test_missing_pagination_completion_is_refused():
    class P:
        def get(self,*args):return dict(corporate_actions={}),b'{"corporate_actions":{}}'
    with pytest.raises(daily.DailyError,match='COMPLETION_MISSING'):
        list(daily.pages(P(),daily.DATA+'/v1/corporate-actions',{}))

def test_provider_auth_failure_is_sanitized_and_not_retried(monkeypatch):
    provider=daily.Provider('KEY_FOR_OFFLINE_TEST','SECRET_FOR_OFFLINE_TEST')
    calls=[]
    class Response:
        status_code=403
        def close(self):calls.append('closed')
    def get(*a,**kw):calls.append(kw);return Response()
    monkeypatch.setattr(provider.session,'get',get)
    with pytest.raises(daily.DailyError,match='HTTP_403') as error:
        provider.get(daily.DATA+'/v2/stocks/bars',{})
    assert 'SECRET' not in str(error.value)
    assert len(provider.receipts)==1 and calls[0]['allow_redirects'] is False
    assert calls[-1]=='closed';provider.close()

def test_provider_rejects_order_endpoint_before_network():
    provider=daily.Provider('KEY_FOR_OFFLINE_TEST','SECRET_FOR_OFFLINE_TEST')
    with pytest.raises(daily.DailyError,match='ENDPOINT'):
        provider.get('https://paper-api.alpaca.markets/v2/orders',{})
    assert provider.receipts==[];provider.close()

@pytest.mark.parametrize('fail',[False,True])
def test_archive_is_append_only_and_requires_separate_readonly_proof(cfg,tmp_path,monkeypatch,fail):
    from scripts import research_store,strategy_scanner,strategy_library
    from scripts.research_testing import save_json
    from test_research_testing import make_capture,simple_frame
    spec,*_=make_capture(tmp_path);day=spec['folds'][0]['test_end']
    frame=simple_frame(3);frame.index=pd.bdate_range(end=day,periods=3)
    frame['entry_details']=[dict(signal=True,pullback_start='2023-12-01',failures=[]) for _ in range(3)]
    monkeypatch.setattr(strategy_library.Strategy,'prepare',lambda self,*args:frame.copy(deep=True))
    summary=strategy_scanner.run_scan(spec,as_of=day,output=tmp_path/'scan',cost_case_id='base')
    assert isinstance(summary['strategies'][0]['strategy']['buy_rules'],tuple)
    assert summary['strategies'][0]['metrics']['terminal_open_positions']
    assert any(isinstance(value,(datetime,pd.Timestamp)) for value in summary['strategies'][0]['metrics']['terminal_open_positions'][0].values())
    save_json(tmp_path/'study.json',spec)
    stored={};calls=[];connections=[]
    class Cursor:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execute(self,sql,args):
            calls.append(sql)
            if sql.startswith('INSERT'):
                if fail:raise RuntimeError('private database error must not escape')
                stored.update(run_id=args[0],spec=args[1],digest=args[2],result=json.loads(args[3]))
            self.sql=sql
        def fetchone(self):
            if self.sql.startswith('SELECT specification'):return stored['spec'],stored['digest'],stored['result']
            return stored['digest'],stored['result']
    class Conn:
        def __init__(self,write):self.write=write;self.commits=0;self.rollbacks=0;self.closed=False
        def cursor(self):return Cursor()
        def commit(self):self.commits+=1
        def rollback(self):self.rollbacks+=1
        def close(self):self.closed=True
    def connect(*,write=False,env=None):
        conn=Conn(write);connections.append(conn);return conn
    monkeypatch.setattr(research_store,'connect',connect)
    if fail:
        with pytest.raises(daily.DailyError,match='TRANSACTION_FAILED'):
            daily.archive_scan(summary,tmp_path,cfg,env={})
        assert connections[0].commits==0 and connections[0].rollbacks==1
        assert not (tmp_path/'archive.json').exists()
    else:
        result=daily.archive_scan(summary,tmp_path,cfg,env={})
        assert result['rows']==1 and [c.write for c in connections]==[True,False]
        assert connections[0].commits==1 and connections[1].rollbacks==1
        assert stored['result']['execution_authorized'] is False
    assert all(c.closed for c in connections)
    assert sum(sql.startswith('INSERT') for sql in calls)==1
    assert not any(sql.startswith(('CREATE','UPDATE','DELETE','DROP')) for sql in calls)

def test_changed_unbound_request_spec_is_refused_before_database(cfg,tmp_path,monkeypatch):
    from scripts import research_store
    from scripts.research_testing import save_json
    scan=tmp_path/'scan';scan.mkdir()
    summary=dict(status='COMPLETED_PAPER_SIMULATION',as_of='2026-10-23',run_id='batch',
        research_qualified=False,order_submission_authorized=False,broker_operations=0)
    save_json(scan/'result.json',summary)
    save_json(scan/'paper_handoff.json',dict(scan_run_id='batch',order_submission_authorized=False))
    save_json(scan/'specification.json',dict(schema='actual-scanned-spec'))
    save_json(scan/'manifest.json',dict(bindings=[dict(path=p.name,**daily.binding(p.read_bytes())) for p in scan.iterdir()]))
    save_json(tmp_path/'study.json',dict(schema='changed-after-scan'))
    def reject(**kwargs):raise AssertionError('No database access for mismatched specification')
    monkeypatch.setattr(research_store,'connect',reject)
    with pytest.raises(daily.DailyError,match='SCANNED_SPECIFICATION_CHANGED'):
        daily.archive_scan(summary,tmp_path,cfg,env={})
