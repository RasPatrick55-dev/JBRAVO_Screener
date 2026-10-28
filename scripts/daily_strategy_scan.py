"""Daily paper screening over fresh, bounded Alpaca captures. No order endpoints."""
from __future__ import annotations
from datetime import date,datetime,time,timedelta,timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time as clock
from zoneinfo import ZoneInfo

from scripts.research_dataset import canonical_hash,strict_json
from scripts.research_testing import save_json,validate_spec,json_default
from scripts.strategy_scanner import run_scan
from scripts.strategy_library import get_strategy

NY=ZoneInfo('America/New_York')
DATA='https://data.alpaca.markets'
CALENDAR='https://paper-api.alpaca.markets/v2/calendar'
PIPELINE='pullback-reclaim-daily-paper-v1'
CONFIG=Path(__file__).resolve().parents[1]/'config/strategy_scan_daily.json'

class DailyError(ValueError):pass

def require(ok,code):
    if not ok:raise DailyError(code)

def binding(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def load_config(path=CONFIG):
    cfg=strict_json(Path(path).read_bytes())
    from scripts.strategy_universe import market_config
    require(cfg.get('schema') in ('jbravo.daily-strategy-config.v1','jbravo.daily-strategy-config.v2') and
        cfg.get('enabled') is True,'DAILY_CONFIG_DISABLED_OR_INVALID')
    require(cfg.get('research_qualified') is False and cfg.get('order_submission_authorized') is False,'PAPER_SCREENING_SCOPE_REQUIRED')
    strategy=get_strategy(cfg['strategy_id'])
    require(strategy.paper_scan_enabled,'STRATEGY_NOT_ENABLED')
    if market_config(cfg):
        universe=cfg['universe']
        require('symbols' not in cfg and set(universe)=={'source','status','asset_class','tradable','max_symbols','batch_size'} and
            universe['source']=='alpaca_assets' and universe['status']=='active' and
            universe['asset_class']=='us_equity' and universe['tradable'] is True and
            type(universe['max_symbols']) is int and 1<=universe['max_symbols']<=30000 and
            type(universe['batch_size']) is int and 1<=universe['batch_size']<=25,'INVALID_DAILY_UNIVERSE')
        require(1<=cfg['max_provider_gets']<=12000 and 1<=cfg['max_scan_seconds']<=3600 and
            1<=cfg['max_capture_seconds']<=5400 and 1<=cfg['max_total_seconds']<=7200 and
            1<=cfg['max_provider_bytes']<=16*1024**3,'INVALID_DAILY_DATA_BUDGET')
    else:
        symbols=cfg['symbols']+[cfg['benchmark']]
        require(1<=len(cfg['symbols'])<=100 and len(symbols)==len(set(symbols)) and all(
            isinstance(s,str) and 1<=len(s)<=12 and all(c.isupper() or c.isdigit() or c in '.-' for c in s) for s in symbols),'INVALID_DAILY_UNIVERSE')
        require(1<=cfg['max_provider_gets']<=24 and 1<=cfg['max_scan_seconds']<=600,'INVALID_DAILY_DATA_CONTRACT')
    require(cfg['feed']=='sip' and cfg['currency']=='USD','INVALID_DAILY_DATA_CONTRACT')
    require(date.fromisoformat(cfg['warmup_start'])<date.fromisoformat(cfg['episode_start']),'INVALID_DAILY_EPISODE')
    return cfg

class Provider:
    """Fixed GET endpoints, no redirects/retries; bounded complete responses only."""
    def __init__(self,key,secret,*,max_gets=24,max_seconds=240,max_bytes=32*1024*1024,min_interval=0):
        require(bool(key and secret),'ALPACA_DATA_AUTH_UNAVAILABLE')
        import requests
        self.session=requests.Session();self.session.trust_env=False
        self.session.headers.update({'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,'Accept-Encoding':'identity'})
        self.secrets=(key.encode(),secret.encode());self.max_gets=max_gets
        self.receipts=[];self.total=0;self.deadline=clock.monotonic()+max_seconds
        self.max_bytes=max_bytes
        self.min_interval=min_interval;self.last_request=None

    def get(self,url,params):
        from scripts.strategy_universe import ASSETS
        require(url in (CALENDAR,ASSETS,DATA+'/v2/stocks/bars',DATA+'/v1/corporate-actions'),'PROVIDER_ENDPOINT_REFUSED')
        if self.last_request is not None:
            delay=self.min_interval-(clock.monotonic()-self.last_request)
            if delay>0:clock.sleep(delay)
        require(len(self.receipts)<self.max_gets and clock.monotonic()<self.deadline,'PROVIDER_BUDGET_EXCEEDED')
        self.last_request=clock.monotonic()
        receipt=dict(endpoint=url,parameters=dict(params),status='ATTEMPTED')
        self.receipts.append(receipt)
        response=None
        try:
            response=self.session.get(url,params=params,timeout=(10,15),allow_redirects=False,stream=True)
            require(response.status_code==200,'PROVIDER_HTTP_'+str(response.status_code))
            require(response.headers.get('Content-Encoding','identity') in ('identity',''),'PROVIDER_ENCODING_REFUSED')
            raw=b''
            for part in response.iter_content(65536):
                raw+=part;self.total+=len(part)
                require(len(raw)<=(32*1024*1024 if url==ASSETS else 4*1024*1024) and
                    self.total<=self.max_bytes and clock.monotonic()<self.deadline,'PROVIDER_RESPONSE_BUDGET_EXCEEDED')
            require(not any(secret in raw for secret in self.secrets),'PROVIDER_AUTH_ECHO_REFUSED')
            value=strict_json(raw)
            receipt.update(status='COMPLETE',**binding(raw))
            return value,raw
        except DailyError:raise
        except Exception:raise DailyError('PROVIDER_TRANSPORT_OR_JSON_FAILURE') from None
        finally:
            if response is not None:response.close()

    def close(self):self.session.close()

def normalize_calendar(rows):
    require(isinstance(rows,list) and 1<=len(rows)<=5000,'INVALID_PROVIDER_CALENDAR')
    result=[]
    for row in rows:
        day=date.fromisoformat(row['date']).isoformat()
        opening=time.fromisoformat(row['open']);closing=time.fromisoformat(row['close'])
        require(opening<closing and opening.tzinfo is None and closing.tzinfo is None,'INVALID_PROVIDER_CALENDAR_CLOCK')
        result.append(dict(date=day,open=opening.isoformat(),close=closing.isoformat(),timezone='America/New_York'))
    require([r['date'] for r in result]==sorted({r['date'] for r in result}),'DUPLICATE_OR_UNSORTED_CALENDAR')
    return result

def signal_session(calendar,now,*,bootstrap_latest=False):
    require(now.tzinfo is not None and now.utcoffset() is not None,'AWARE_DAILY_CLOCK_REQUIRED')
    local=now.astimezone(NY)
    completed=[r for r in calendar if datetime.combine(date.fromisoformat(r['date']),time(20,15),NY)<=now]
    require(bool(completed),'NO_COMPLETED_SIGNAL_SESSION')
    latest=completed[-1]['date']
    if not bootstrap_latest:
        # UTC01:20 is20:20/21:20 Eastern, after the same day's20:15 cutoff.
        if latest!=local.date().isoformat():
            if any(r['date']==local.date().isoformat() for r in calendar):
                raise DailyError('SIGNAL_SESSION_NOT_FINAL')
            return None
    require(0<=(local.date()-date.fromisoformat(latest)).days<=7,'STALE_BOOTSTRAP_SESSION')
    require(any(r['date']>latest for r in calendar),'NEXT_EXCHANGE_SESSION_UNAVAILABLE')
    return latest

def pages(provider,url,params,*,max_pages=6):
    token=None;seen=set()
    for page in range(max_pages):
        request=dict(params)
        if token is not None:request['page_token']=token
        value,raw=provider.get(url,request)
        require(isinstance(value,dict),'PROVIDER_ENVELOPE_INVALID')
        require('next_page_token' in value,'PROVIDER_PAGINATION_COMPLETION_MISSING')
        yield page,value,raw
        token=value.get('next_page_token')
        if token is None:return
        require(isinstance(token,str) and 0<len(token)<=4096 and token not in seen,'PROVIDER_PAGINATION_INVALID')
        seen.add(token)
    raise DailyError('PROVIDER_PAGE_LIMIT_EXCEEDED')

def capture(provider,cfg,calendar,session,root,*,now):
    from scripts.strategy_universe import capture_market,market_config
    if market_config(cfg):
        return capture_market(provider,cfg,calendar,session,root,now=now,cache_dir=root.parent/'market-cache')
    symbols=cfg['symbols']+[cfg['benchmark']]
    folder=root/'capture/analysis';folder.mkdir(parents=True,mode=0o700)
    raw_dir=root/'provider-responses';raw_dir.mkdir(mode=0o700)
    pins=[]
    def bind_member(name,raw):
        with (folder/name).open('xb') as h:h.write(raw)
        pins.append(dict(path='analysis/'+name,**binding(raw)))
    bind_member('calendar.json',json.dumps(calendar,sort_keys=True).encode())
    start=datetime.combine(date.fromisoformat(cfg['warmup_start']),time.min,NY).isoformat()
    end=datetime.combine(date.fromisoformat(session),time(20),NY).isoformat()
    for adjustment in ('raw','split'):
        records=[]
        params=dict(symbols=','.join(symbols),timeframe='1Day',start=start,end=end,feed=cfg['feed'],
            currency=cfg['currency'],adjustment=adjustment,asof=now.astimezone(NY).date().isoformat(),limit=10000,sort='asc')
        for page,value,raw in pages(provider,DATA+'/v2/stocks/bars',params):
            (raw_dir/(adjustment+'-'+str(page)+'.json')).write_bytes(raw)
            bars=value.get('bars');require(isinstance(bars,dict) and set(bars)<=set(symbols),'BAR_SYMBOL_ENVELOPE_INVALID')
            for symbol,items in bars.items():
                require(isinstance(items,list),'BAR_LIST_INVALID')
                for bar in items:
                    stamp=datetime.fromisoformat(bar['t'].replace('Z','+00:00'))
                    require(stamp.tzinfo is not None,'NAIVE_BAR_TIMESTAMP')
                    local=stamp.astimezone(NY)
                    require(local.timetz().replace(tzinfo=None)==time.min,'NON_DAILY_BAR_TIMESTAMP')
                    require(cfg['warmup_start']<=local.date().isoformat()<=session,'BAR_OUTSIDE_REQUESTED_SESSIONS')
                    records.append(dict(symbol=symbol,session=local.date().isoformat(),**{k:bar[k] for k in ('o','h','l','c','v')},
                        provider='alpaca',feed=cfg['feed'],currency=cfg['currency'],adjustment=adjustment,
                        asof=params['asof'],quality_flags=''))
                    require(len(records)<=50000,'DAILY_BAR_ROW_CAP')
        require({r['symbol'] for r in records}==set(symbols),'MISSING_DAILY_SYMBOL')
        records.sort(key=lambda r:(r['symbol'],r['session']))
        text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        bind_member('alpaca_'+adjustment+'_daily.csv',text.getvalue().encode())
    bind_member('provider-requests.json',json.dumps(provider.receipts,sort_keys=True).encode())
    save_json(folder/'manifest.json',dict(research_qualified=False,execution_authorized=False,bindings=pins))
    actions={}
    params=dict(symbols=','.join(symbols),start=cfg['warmup_start'],end=now.astimezone(NY).date().isoformat(),limit=1000,sort='asc')
    for page,value,raw in pages(provider,DATA+'/v1/corporate-actions',params):
        (raw_dir/('actions-'+str(page)+'.json')).write_bytes(raw)
        categories=value.get('corporate_actions');require(isinstance(categories,dict),'ACTION_ENVELOPE_INVALID')
        for kind,items in categories.items():
            require(isinstance(items,list),'ACTION_LIST_INVALID');actions.setdefault(kind,[]).extend(items)
    action_file=root/'actions.json'
    save_json(action_file,dict(request=dict(start=params['start'],end=params['end'],symbols=params['symbols']),
        isError=False,content=[dict(type='text',text=json.dumps(dict(data=dict(corporate_actions=actions,next_page_token=None))))]))
    spec=dict(schema='jbravo.strategy-study.v1',purpose='exploratory_research',capture_manifest=str(folder/'manifest.json'),
        actions_capture=dict(path=str(action_file),sha256=binding(action_file.read_bytes())['sha256']),
        universe=dict(symbols=cfg['symbols'],benchmark=cfg['benchmark'],excluded={}),
        folds=[dict(id=cfg['episode_id'],warmup_start=cfg['warmup_start'],test_start=cfg['episode_start'],test_end=session)],
        strategies=[cfg['strategy_id']],cost_cases=[cfg['cost_case']],portfolio=cfg['portfolio'],max_runs=2,
        max_seconds=cfg['max_scan_seconds'],research_qualified=False,execution_authorized=False)
    validate_spec(spec);save_json(root/'study.json',spec)
    return spec

def archive_scan(summary,root,cfg,*,env):
    """Append exactly one completed daily row; existing research schema only."""
    from scripts.research_store import connect
    # Match the scanner's saved JSON representation (tuples, dates and NumPy scalars).
    summary=json.loads(json.dumps(summary,default=json_default,allow_nan=False))
    require(summary.get('status')=='COMPLETED_PAPER_SIMULATION' and summary.get('research_qualified') is False and
        summary.get('order_submission_authorized') is False and summary.get('broker_operations')==0,'COMPLETED_PAPER_SCAN_REQUIRED')
    manifest_raw=(root/'scan/manifest.json').read_bytes();manifest=strict_json(manifest_raw)
    require(strict_json((root/'scan/result.json').read_bytes())==summary,'SCAN_RESULT_CHANGED')
    bound=set()
    for pin in manifest['bindings']:
        path=(root/'scan'/pin['path']).resolve()
        require(path.is_relative_to((root/'scan').resolve()) and pin['path'] not in bound,'SCAN_ARTIFACT_PATH_REJECTED')
        require(binding(path.read_bytes())=={k:pin[k] for k in ('bytes','sha256')},'SCAN_ARTIFACT_CHANGED')
        bound.add(pin['path'])
    require({'result.json','paper_handoff.json','specification.json'}<=bound,'SCAN_MANIFEST_INCOMPLETE')
    spec_raw=(root/'scan/specification.json').read_bytes()
    require(spec_raw==(root/'study.json').read_bytes(),'SCANNED_SPECIFICATION_CHANGED')
    run_id=canonical_hash(dict(schema='jbravo.daily-strategy-archive-id.v1',pipeline=PIPELINE,
        strategy_id=cfg['strategy_id'],signal_session=summary['as_of']))
    handoff=strict_json((root/'scan/paper_handoff.json').read_bytes())
    require(handoff['scan_run_id']==summary['run_id'] and handoff['order_submission_authorized'] is False,'SCAN_HANDOFF_IDENTITY_CHANGED')
    result=dict(schema='jbravo.daily-strategy-scan.v1',status='COMPLETED_PAPER_SCREEN',run_id=run_id,
        scan_run_id=summary['run_id'],signal_session=summary['as_of'],pipeline=PIPELINE,strategy_id=cfg['strategy_id'],
        configuration_sha256=canonical_hash(cfg),scan_manifest_sha256=binding(manifest_raw)['sha256'],summary=summary,handoff=handoff,output_root=str(root),
        research_qualified=False,execution_authorized=False,order_submission_authorized=False)
    body=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)
    expected=binding(body.encode())['sha256'];spec_sha=binding(spec_raw)['sha256']
    conn=connect(write=True,env=env)
    try:
        with conn.cursor() as cur:
            cur.execute('INSERT INTO jbravo_research.research_runs (run_id,specification_sha256,result_sha256,result) '
                'VALUES (%s,%s,%s,%s::jsonb)',(run_id,spec_sha,expected,body))
            cur.execute('SELECT result_sha256,result FROM jbravo_research.research_runs WHERE run_id=%s',(run_id,))
            row=cur.fetchone();require(row is not None and row[0].strip()==expected and row[1]==result,'DAILY_ARCHIVE_PRECOMMIT_READBACK_FAILED')
        conn.commit()
    except Exception:
        conn.rollback();raise DailyError('DAILY_ARCHIVE_TRANSACTION_FAILED_NO_REPLAY') from None
    finally:conn.close()
    conn=connect(env=env)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT specification_sha256,result_sha256,result FROM jbravo_research.research_runs WHERE run_id=%s',(run_id,))
            row=cur.fetchone();require(row is not None and row[0].strip()==spec_sha and row[1].strip()==expected and row[2]==result,'DAILY_ARCHIVE_COMMITTED_READBACK_FAILED')
    finally:conn.rollback();conn.close()
    proof=dict(status='COMMITTED_AND_SEPARATE_READONLY_VERIFIED',run_id=run_id,result_sha256=expected,rows=1)
    save_json(root/'archive.json',proof)
    return proof

def run_daily(cfg,provider,output,*,now=None,bootstrap_latest=False,archive=None):
    now=now or datetime.now(timezone.utc)
    started=clock.monotonic()
    rows,raw=provider.get(CALENDAR,dict(start=cfg['warmup_start'],end=(now.astimezone(NY).date()+timedelta(days=14)).isoformat()))
    calendar=normalize_calendar(rows);session=signal_session(calendar,now,bootstrap_latest=bootstrap_latest)
    if session is None:return dict(status='SKIPPED_NON_TRADING_DAY',pipeline=PIPELINE,broker_operations=0)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    require(not output.is_relative_to(Path(__file__).resolve().parents[1]),'DAILY_OUTPUT_INSIDE_REPOSITORY')
    # An interrupted session never reuses its claim or partially saved input batch.
    root=output/session;root.mkdir(mode=0o700,exist_ok=False)
    save_json(root/'claim.json',dict(status='CLAIMED_NO_REPLAY',signal_session=session,started_at_utc=now.isoformat(),
        configuration=cfg,pipeline=PIPELINE,bootstrap_latest=bootstrap_latest))
    phase = 'CAPTURE'
    try:
        (root/'calendar-response.json').write_bytes(raw)
        spec=capture(provider,cfg,calendar,session,root,now=now)
        from scripts.strategy_universe import market_config
        dataset = None
        if market_config(cfg):
            remaining=cfg['max_total_seconds']-(clock.monotonic()-started)
            require(remaining>=1,'DAILY_TOTAL_BUDGET_EXCEEDED')
            from scripts.research_dataset import load_dataset
            from scripts.strategy_prepared_market import build_prepared
            from scripts.setup_selection import required_strategies
            phase = 'SHARED_MARKET_PREPARATION'
            dataset = load_dataset(spec)
            spec['prepared_market'] = build_prepared(dataset, spec, required_strategies(), root/'prepared',
                deadline=min(started+cfg['max_total_seconds'], clock.monotonic()+cfg['max_scan_seconds']),
                stop_file=root/'STOP')
            # This session is exclusively claimed; freeze the prepared reference
            # before any scan/archive. Preserve the original captured study too.
            (root/'study.json').rename(root/'captured-study.json')
            save_json(root/'study.json', spec)
        phase = 'PRIMARY_SCAN'
        result=run_scan(spec,as_of=session,output=root/'scan',strategy_ids=[cfg['strategy_id']],
            fold_id=cfg['episode_id'],cost_case_id=cfg['cost_case']['id'],now=now,
            total_deadline=started+cfg['max_total_seconds'] if market_config(cfg) else None,
            **({'dataset': dataset} if dataset is not None else {}))
        save_json(root/'provider-requests.json',provider.receipts)
        require(archive is not None,'DAILY_POSTGRES_ARCHIVE_REQUIRED')
        proof=archive(result,root,cfg)
        from scripts.bull_flag_companion import produce
        companion = dict(status='DISABLED')
        phase = 'COMPANION_SCANS'
        if market_config(cfg):
            try:
                companion = produce(root, spec, cfg, now=now, archive=archive,
                                    total_deadline=started+cfg['max_total_seconds'], dataset=dataset)
            except Exception as error:
                companion=dict(status='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED',error_type=type(error).__name__)
        companions={}
        if market_config(cfg):
            from scripts.swing_patterns import STRATEGIES as extra_patterns
            for strategy_id in extra_patterns:
                try:
                    outcome=produce(root,spec,cfg,now=now,archive=archive,
                        total_deadline=started+cfg['max_total_seconds'],strategy_id=strategy_id, dataset=dataset)
                except Exception as error:
                    outcome=dict(status='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED',error_type=type(error).__name__)
                if outcome.get('status')!='DISABLED':companions[strategy_id]=outcome
        if market_config(cfg):
            from scripts.bull_flag_strategy import BULL_FLAG
            outcomes={cfg['strategy_id']:dict(status='COMPLETED_COMPANION_SCREEN'), BULL_FLAG:companion, **companions}
            unavailable=[sid for sid in required_strategies() if
                outcomes.get(sid, {}).get('status') != 'COMPLETED_COMPANION_SCREEN']
            if unavailable:
                save_json(root/'blocked.json', dict(status='DAILY_STRATEGY_SCAN_BLOCKED',
                    unavailable_strategies=unavailable, outcomes=outcomes, broker_operations=0))
                raise DailyError('REQUIRED_STRATEGY_SCREENS_UNAVAILABLE')
        value=dict(status='DAILY_STRATEGY_SCAN_COMPLETE',pipeline=PIPELINE,signal_session=session,
            strategy_id=cfg['strategy_id'],scan_run_id=result['run_id'],candidates=len(result['top_candidates']),
            provider_gets=len(provider.receipts),archive=proof,output_root=str(root),
            research_qualified=False,order_submission_authorized=False,broker_operations=0)
        if market_config(cfg):value['coverage']=result['coverage']
        if companion.get('status') != 'DISABLED':value['companion']=companion
        if companions:value['companions']=companions
        save_json(root/'completion.json',value)
        return value
    except Exception as error:
        from scripts.strategy_runtime_diagnostics import diagnostic
        save_json(root/'failure.json',dict(status='DAILY_STRATEGY_SCAN_FAILED_NO_REPLAY',signal_session=session,
            **dict(diagnostic(error, phase, started, deadline=started+cfg.get('max_total_seconds',cfg['max_scan_seconds'])),
                **({'code':str(error)} if isinstance(error,DailyError) else {})),
            broker_operations=0,provider_gets=len(provider.receipts)))
        raise

def configured_environment():
    """Read the established private configuration only inside the host worker."""
    from dotenv import dotenv_values
    from psycopg2.extensions import make_dsn
    values=dotenv_values(Path.home()/'.config/jbravo/.env',interpolate=False)
    env={**os.environ,**{k:v for k,v in values.items() if v is not None}}
    if not env.get('RESEARCH_DATABASE_URL'):
        if env.get('DATABASE_URL'):env['RESEARCH_DATABASE_URL']=env['DATABASE_URL']
        else:
            env['RESEARCH_DATABASE_URL']=make_dsn(host=env['DB_HOST'],dbname=env['DB_NAME'],
                user=env.get('DB_USER') or env.get('DB_USERNAME'),password=env.get('DB_PASSWORD') or env.get('DB_PASS'),
                port=env.get('DB_PORT','5432'),sslmode=env.get('DB_SSLMODE','prefer'))
    return env

def main(output,*,bootstrap_latest=False):
    provider=None
    try:
        cfg=load_config();env=configured_environment()
        provider=Provider(env.get('APCA_API_KEY_ID'),env.get('APCA_API_SECRET_KEY'),max_gets=cfg['max_provider_gets'],
            max_seconds=cfg.get('max_capture_seconds',240),max_bytes=cfg.get('max_provider_bytes',32*1024*1024),
            min_interval=.35 if cfg.get('schema')=='jbravo.daily-strategy-config.v2' else 0)
        result=run_daily(cfg,provider,output,bootstrap_latest=bootstrap_latest,
            archive=lambda summary,root,cfg:archive_scan(summary,root,cfg,env=env))
        print(json.dumps(result,sort_keys=True));return 0
    except Exception as error:
        print(json.dumps(dict(status='DAILY_STRATEGY_SCAN_FAILED',error_type=type(error).__name__,
            code=str(error) if isinstance(error,DailyError) else 'DETAIL_WITHHELD',broker_operations=0)))
        return 1
    finally:
        if provider is not None:provider.close()
