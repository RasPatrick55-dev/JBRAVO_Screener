"""CI-only isolated native checkpoint qualification. No production/provider access."""
import argparse,builtins,contextlib,hashlib,importlib,io,json,math,os
from pathlib import Path
import re,signal,socket,subprocess,sys,threading,time,unittest
from datetime import date,datetime,timezone,timedelta
SOURCE_HEAD=os.environ.get('GITHUB_SHA','UNPUBLISHED')
DATABASE='qualification_ingestion'
PORT=55432
PASSWORD='EPHEMERAL_INGESTION_CI_ONLY'
USERS=('qualification_admin','qualification_runtime','qualification_rowlock')
ROOT=OUT=WORK=None
PG=DB=M=None
CONNECTIONS=[]
GUARDS=[]
GUARD_EVENTS=[]
CHILD_RECEIPTS={}
CHILDREN=[]
ALLOWED_CHILDREN=[]
SERVER_VERIFIED=False
CLEANING=False
DEADLINE_HIT=False
VERSION_METADATA_FALLBACKS=[]
EXPECTED_SOURCE={'dashboards/dashboard_app.py': 'f8317129ba531386007108edb89c77d56e0025ad1432631362fb41f91aa1f438', 'dashboards/data_io.py': '17bad150ac6d03e2432276f58d9ecc7804dc8b6b3dbc8ef71b80d3fda4daa8f1', 'scripts/account_sync.py': 'b318d9d583f63e5a65b39c18efec6e9b24799dd02c429d8116d6bf285f86307e', 'scripts/activity_ingestion_checkpoint.py': '0a699b5ef529dab269fe46d5f17091c33f353bf11d180987552355908124d2f2', 'scripts/db.py': '02690ab2564c046ba9a834031368d0f39f7313d40bbcdff45e15d53c60029630', 'scripts/db_migrate.py': '931e5d9ded6d2c0f8f596e11e8f84426ece4093450f5f3d63630f2823b34e9ac', 'scripts/db_queries.py': 'a2f5fa9151e5cde302ccf6cd0d3fb49d1260a2c35d993324e0ab7b47da42e3de', 'scripts/execute_trades.py': '2184763656a8283b6fb90cbfacf1ff2d491b002d92013e6aafb08fb27b492ee6', 'scripts/executor_health.py': '0a434a8988013575bf1f26c15b30ec7d8dd5a5acf57c637410ca6dbdf04508fd', 'scripts/feature_generator.py': '655be42470cc4fffb546c6e5f0353c60e4b34ae409c40f5377c60af8bc063ab7', 'scripts/fetch_account_activities.py': '359eccb5b6b6ae5ef7752819ef749cfcc3a3d74110701ba89922055b15821d5d', 'scripts/gross_paper_accounting.py': '1b604350a20d7a0e1007dfd1595d3888c4dc2f23bc8e6a32558a79af25e99e43', 'scripts/label_generator.py': 'faab6698320cd6aed24aa6160a90ea99ebde36dc21ec56f0c54292e8d63b4853', 'scripts/metrics.py': '781573df048b3372b4007b39130670fe7986334e882c1bc602c389fd585fe260', 'scripts/ops/run_canary_smoke.sh': '47ce8170006411851c5ffc0bef70f656177640c7102e0adbce0e62c57dc12eef', 'scripts/ops/run_primary_pipeline.sh': '8c13cc7b41557b34f90ed615a1f17d7c002f7f7101548cac23a72efbc5be2bfb', 'scripts/pipeline_postflight.py': 'cadc6e426a2d16fb3152ea6a2468884285d21fe4d0b0f0ec28f3c5de3722b467', 'scripts/primary_paper_pipeline.py': '52e3e6def70a3e91aaac025966a24c730b5087bfb2cc9d31e28a2e9c70790467', 'scripts/primary_pipeline.py': '9d58560830fb34e6b5fde38de5f564302d078fa1d2d77c8e4d71ff19aa3c790c', 'scripts/production_evidence.py': '11d64959f673b514f2f6ea031eb09f0ef38c400636c32067767e4b7cb624a18c', 'scripts/production_score_capture.py': 'c184891d7500c1960f76ea0b26ad6a846b7c8bbd97c47f8853ed7d07944dcc0e', 'scripts/ranker_eval.py': 'a7c577ab3b861fb1a663b5028a270309c8cf0a56602a0ee4c97f75d8a4bc8bd6', 'scripts/ranker_monitor.py': 'd71965ad9af9cff8f001f307fede7fe12951306e177824b9865e7fa94f548b5e', 'scripts/ranker_predict.py': '1be4ea8883a144b7ec73c02ad8f685d7ae918835bf3a59dba8bef844b4b84730', 'scripts/ranker_strategy_eval.py': 'e0d5af1e14fd432f4c0b7b1e2dd329472db648ee4d472fca365fbf21cc050a78', 'scripts/run_pipeline.py': 'a72647465f1360cba04a6beee0a21501457ce46a11c9370c143938c84d800a95', 'scripts/screener.py': '00e6f32faf597e48aa3407985b0be6d5102b50f810de150fe086a884fa9b6126', 'scripts/signal_handoff.py': '505f7bcd510a684ad7b16356ad180f1665fa92b857a0e45a02f7ba9700871c77', 'scripts/trade_allocation.py': '88f26d7b655ed08e4d519fdd8740347ce52e9bac7b004212fc413e926ba721c0', 'scripts/trade_disposition.py': '6da399754be973323b6a9f859e1df2306c226de11d22d0d15c1f7b5880e5ae85', 'scripts/trade_performance.py': 'd40fe11be566f4e1972d3f5d4d4c3309b74b4459d81f80942c9c456b5cc297e9', 'scripts/utils/calendar.py': 'dd8053bb29024f30ce7e3d82e6a9f96bc35631fb38a94e8b9c885747792e7683', 'scripts/utils/ml_health_guard.py': '12e70e640f48d977caa946488a2ea5c36724dc6ba648c0eb73e97835197d1b21', 'tests/test_activity_ingestion_checkpoint.py': '85d8e4ee91087034f48ae40ba85fcc2314d80cc1b4687b69c45e309c9ff9062f'}

def configure_test_capabilities():
    # This test environment admits only IPv4 loopback PostgreSQL. Advertising
    # IPv6 support would make urllib3 try an unnecessary, forbidden bind probe.
    socket.has_ipv6=False
    original=subprocess.check_output
    def version_metadata(command,*args,**kwargs):
        # Exercise the application's existing unavailable-Git fallback. This
        # supplies no fabricated revision and launches no process. All other
        # process requests still reach the unchanged audit guard.
        if (not args and command==['git','rev-parse','--short','HEAD'] and set(kwargs)=={'cwd'}
                and str(kwargs['cwd'])==str(ROOT)):
            if len(VERSION_METADATA_FALLBACKS)<8:VERSION_METADATA_FALLBACKS.append('VERSION_GIT_UNAVAILABLE')
            raise FileNotFoundError('CI_VERSION_METADATA_UNAVAILABLE')
        return original(command,*args,**kwargs)
    subprocess.check_output=version_metadata

class QualificationDeadline(KeyboardInterrupt):
    """Propagate past unittest's ordinary error capture to bounded cleanup."""

def outer_deadline(signum,frame):
    global DEADLINE_HIT
    DEADLINE_HIT=True
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    if not CLEANING:raise QualificationDeadline('OUTER_DEADLINE')

def safe_error(error):
    codes=[];seen=set();current=error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code=getattr(current,'pgcode',None)
        if isinstance(code,str) and re.fullmatch('[A-Z0-9]{5}',code):codes.append(code)
        current=current.__cause__ or current.__context__
    policy=str(error) if re.fullmatch('(?:TRADE_(?:POLICY|ALLOCATION)|ACT_CHECKPOINT)_[A-Z0-9_]{1,100}',str(error)) else None
    return dict(type=type(error).__name__,sqlstates=sorted(set(codes)),policy_code=policy)

def ds(user='qualification_admin'):
    if user not in USERS:raise RuntimeError('SYNTHETIC_USER')
    return f'postgresql://{user}:{PASSWORD}@127.0.0.1:{PORT}/{DATABASE}?sslmode=disable'

def deny(category,event='fixed_target_check',request_class='REFUSED'):
    if len(GUARDS)<16:GUARDS.append(category)
    if len(GUARD_EVENTS)<8:
        near=[];application=[];frame=sys._getframe(1)
        for _ in range(64):
            if frame is None:break
            module=str(frame.f_globals.get('__name__','UNKNOWN'))
            function=frame.f_code.co_name
            if function not in ('deny','audit','guarded_connect'):
                safe_module=module if re.fullmatch(r'[A-Za-z0-9_.]{1,80}',module) else 'OTHER_MODULE'
                safe_function=function if re.fullmatch(r'[A-Za-z0-9_<>.]{1,80}',function) else 'OTHER_FUNCTION'
                item=dict(module=safe_module,function=safe_function,line=frame.f_lineno)
                if len(near)<3:near.append(item)
                if module.startswith(('scripts.','dashboards.','utils.')) and len(application)<3:application.append(item)
            frame=frame.f_back
        callers=[]
        for item in near+application:
            if item not in callers:callers.append(item)
        GUARD_EVENTS.append(dict(category=category,event=event,request_class=request_class,callers=callers))
    raise RuntimeError('QUALIFICATION_GUARD_'+category)

def install_guards():
    def audit(event,args):
        if event=='socket.getaddrinfo':
            if args[0] not in ('127.0.0.1',) or int(args[1])!=PORT:deny('DNS',event)
        if event=='socket.connect':
            address=args[1]
            if not isinstance(address,tuple) or address[:2]!=('127.0.0.1',PORT):deny('NETWORK',event)
        if event in ('socket.bind','socket.sendto'):
            address=args[1]
            request_class=('LOOPBACK_EPHEMERAL_BIND_REFUSED' if event=='socket.bind' and isinstance(address,tuple)
                and address[:2] in (('127.0.0.1',0),('::1',0)) else 'SERVER_OR_DATAGRAM_REFUSED')
            deny('SERVER_OR_DATAGRAM',event,request_class)
        if event=='subprocess.Popen':
            cmd=list(args[1]) if not isinstance(args[1],str) else args[1]
            if cmd not in ALLOWED_CHILDREN:
                label='READ_ONLY_GIT_VERSION_REFUSED' if cmd==['git','rev-parse','--short','HEAD'] and str(args[2])==str(ROOT) else 'UNAPPROVED_PROCESS_REFUSED'
                deny('PROCESS',event,label)
            ALLOWED_CHILDREN.remove(cmd)
        if event in ('os.system','os.posix_spawn','os.spawn'):deny('PROCESS',event)
        if event=='open' and not isinstance(args[0],int):
            path=Path(args[0]);name=path.name.casefold()
            if name.startswith('.env') or name in ('.pgpass','.netrc','api-token.dpapi') or 'credential' in name:deny('SECRET_FILE',event)
            if path.resolve().is_relative_to(WORK/'qualification-retained-inputs'):deny('CHECKOUT_INPUT_REFUSED',event)
            mode=args[1];flags=args[2] if len(args)>2 else 0
            writing=isinstance(mode,str) and any(c in mode for c in 'wax+') or isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
            if writing:
                resolved=path.resolve()
                if not resolved.is_relative_to(WORK):deny('EXTERNAL_WRITE',event)
                if resolved.is_relative_to(ROOT) and (resolved.suffix in ('.py','.yml','.yaml') or '.git' in resolved.parts):deny('SOURCE_WRITE',event)
    sys.addaudithook(audit)
    import psycopg2
    original=psycopg2.connect
    def guarded_connect(*args,**kwargs):
        params=psycopg2.extensions.parse_dsn(args[0]) if args else {}
        params.update(kwargs)
        database=params.get('dbname',params.get('database'))
        if (database!=DATABASE or params.get('host')!='127.0.0.1' or int(params.get('port',0))!=PORT
                or params.get('user') not in USERS or params.get('password')!=PASSWORD
                or params.get('sslmode')!='disable' or 'service' in params):deny('DATABASE_TARGET')
        params.pop('database',None);params['dbname']=DATABASE
        params['connect_timeout']=3;params['application_name']='qualingest-'+os.environ['GITHUB_RUN_ID']
        connection=original(**params);CONNECTIONS.append(connection)
        return connection
    psycopg2.connect=guarded_connect
    return psycopg2

def connect(user='qualification_admin'):return PG.connect(ds(user))

def sql(statement,params=None,fetch=False,user='qualification_admin'):
    connection=connect(user)
    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout='5000ms'")
                cursor.execute(statement,params)
                return cursor.fetchall() if fetch else None
    finally:connection.close()

class BoundedCapture(io.StringIO):
    def write(self,text):
        if self.tell()<1048576:super().write(text[:1048576-self.tell()])
        return len(text)

class Results(unittest.TextTestResult):
    def __init__(self,*a,**kw):super().__init__(*a,**kw);self.subcases=0
    def addSubTest(self,test,subtest,error):self.subcases+=1;super().addSubTest(test,subtest,error)
    def _exc_info_to_string(self,error,test):return json.dumps(safe_error(error[1]))

def cleanup():
    closed=0
    for process in CHILDREN:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
    for connection in CONNECTIONS:
        if not connection.closed:
            try:connection.rollback()
            except Exception:pass
            finally:connection.close();closed+=1
    remaining=None
    if SERVER_VERIFIED:
        probe=connect()
        try:
            with probe.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM pg_catalog.pg_stat_activity WHERE datname=%s "
                    "AND application_name LIKE 'qualingest-%%' AND pid<>pg_catalog.pg_backend_pid()",(DATABASE,))
                remaining=cursor.fetchone()[0]
        finally:probe.rollback();probe.close()
    return dict(tracked_connections=len(CONNECTIONS),all_connections_closed=all(c.closed for c in CONNECTIONS),
        open_connections_closed_during_cleanup=closed,owned_children=len(CHILDREN),
        all_owned_children_returned=all(p.poll() is not None for p in CHILDREN),owned_backend_sessions_remaining=remaining)

# Appended to the reused, previously qualified CI-only capability/cleanup helpers.
from decimal import Decimal
from unittest import mock
from types import SimpleNamespace
import copy,logging,uuid

I=S=E=F=None
CHILD_APPLICATION_RESULT=None
CHILD_CLASSES={}
U=datetime(2020,1,16,12,tzinfo=timezone.utc)
B=U-timedelta(days=1)
A=U-timedelta(days=14,seconds=300)
REQUIRED=U-timedelta(days=14)
ACCOUNT='00000000-0000-0000-0000-000000000001'
W='a'*32

def verify_source():
    for name,expected in EXPECTED_SOURCE.items():
        path=ROOT/name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):raise RuntimeError('SOURCE_PATH')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('SOURCE_PIN')

def configure_runtime():
    global PG,DB,I,S,M,SERVER_VERIFIED
    sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
    os.environ.update(DATABASE_URL=ds('qualification_runtime'),JBR_EXEC_PAPER='true',
        APCA_API_BASE_URL='https://paper-api.alpaca.markets',APCA_API_KEY_ID='DUMMY_CI_KEY',
        APCA_API_SECRET_KEY='DUMMY_CI_SECRET',DB_DISABLED='0')
    configure_test_capabilities()
    PG=install_guards()
    from scripts import db,activity_ingestion_checkpoint,account_sync,db_migrate
    DB,I,S,M=db,activity_ingestion_checkpoint,account_sync,db_migrate
    server=sql("SELECT pg_catalog.current_database(),pg_catalog.current_setting('server_version_num')",fetch=True)
    if server!=[(DATABASE,'120002')]:raise RuntimeError('SYNTHETIC_SERVER_IDENTITY')
    SERVER_VERIFIED=True

def fixture():
    if GUARDS:raise RuntimeError('GUARD_REFUSAL_STOPS_FIXTURE')
    for c in CONNECTIONS:
        if not c.closed:c.rollback();c.close()
    sql('DROP SCHEMA public CASCADE; CREATE SCHEMA public; GRANT USAGE ON SCHEMA public TO PUBLIC')
    sql('''CREATE TABLE public.trades(trade_id bigint PRIMARY KEY,symbol text,status text,qty numeric,
        entry_price numeric,exit_price numeric,realized_pnl numeric,entry_order_id text,exit_order_id text,
        entry_time timestamptz,exit_time timestamptz,exit_reason text,updated_at timestamptz);
        CREATE TABLE public.trade_disposition_policy(singleton int PRIMARY KEY,deployment_id text,database_id text,
            database_name text,policy_revision bigint);
        CREATE TABLE public.trade_disposition_cases(case_id uuid,trade_id bigint,state text,case_revision bigint,
            request jsonb,row_snapshot jsonb,manifest jsonb);
        CREATE TABLE public.trade_disposition_audit(approval_ref text,request_digest text,request jsonb,
            manifest jsonb,result jsonb,policy_revision bigint);
        CREATE TABLE public.alpaca_activities(id bigserial PRIMARY KEY,activity_id text NOT NULL,
            activity_type text NOT NULL,transaction_time timestamptz,symbol text,side text,qty numeric,
            price numeric,amount numeric,order_id text,description text,raw jsonb,
            UNIQUE(activity_id,activity_type));
        CREATE TABLE public.reconcile_state(id int PRIMARY KEY,last_after timestamptz,last_ran_at timestamptz,
            updated_at timestamptz DEFAULT now());''')
    sql('INSERT INTO public.trade_disposition_policy VALUES(1,%s,%s,%s,2)',
        (*DB._PAPER_DISPOSITION_BINDING[:2],DATABASE))
    sql('INSERT INTO public.reconcile_state(id,last_after) VALUES(1,%s),(2,%s)',(REQUIRED,B))
    c=connect()
    try:
        if not M.ensure_activity_ingestion_schema(c):raise RuntimeError('SYNTHETIC_CHECKPOINT_SETUP_FAILED')
    finally:c.close()
    # These are disposable role grants, never grants on the owner PC/host.
    sql('GRANT SELECT ON public.trade_disposition_policy,public.trade_disposition_cases,public.trade_disposition_audit,public.trades TO qualification_runtime;'
        'GRANT SELECT,INSERT,UPDATE ON public.reconcile_state TO qualification_runtime;'
        'GRANT SELECT,INSERT ON public.alpaca_activities,public.activity_ingestion_checkpoints TO qualification_runtime;'
        'GRANT USAGE ON SEQUENCE public.alpaca_activities_id_seq TO qualification_runtime')

def row(key='synthetic-native'):
    event=dict(id=key,activity_type='FILL',transaction_time=B.isoformat(),symbol='QUALONLY',side='buy',
        qty='2',price='10.5',order_id='00000000-0000-0000-0000-000000000002')
    return dict(activity_id=key,activity_type='FILL',transaction_time=B,symbol='QUALONLY',side='buy',
        qty=Decimal('2'),price=Decimal('10.5'),amount=None,order_id=event['order_id'],description=None,
        raw=json.dumps(event))

def commit_rows(c,**changes):
    kw=dict(workflow_id=W,source=I.source_fingerprint(I.ENDPOINT,ACCOUNT),after=A,until=U,before=B,
        rows=[row()],policy_lock=DB._policy_write_lock)
    kw.update(changes);return I.persist(c,**kw)

def consume_rows(c,**changes):
    kw=dict(workflow_id=W,source=I.source_fingerprint(I.ENDPOINT,ACCOUNT),required_after=REQUIRED,
        required_until=U,now=U+timedelta(seconds=30))
    kw.update(changes);return I.consume(c,**kw)

def counts():
    return tuple(sql('SELECT count(*) FROM public.'+name,fetch=True)[0][0]
        for name in ('alpaca_activities','activity_ingestion_checkpoints'))

class Proxy:
    """Native connection adapter for explicit fault/commit-ack injection only."""
    def __init__(self,c,after_commit=None,before_ledger_insert=None):
        self.c=c;self.after_commit=after_commit;self.before_ledger_insert=before_ledger_insert
    @property
    def autocommit(self):return self.c.autocommit
    def __enter__(self):self.c.__enter__();return self
    def __exit__(self,*args):
        result=self.c.__exit__(*args)
        if args[0] is None and self.after_commit:self.after_commit()
        return result
    def cursor(self,*args,**kwargs):
        cur=self.c.cursor(*args,**kwargs);owner=self
        class Wrapped:
            def __enter__(self):cur.__enter__();return self
            def __exit__(self,*a):return cur.__exit__(*a)
            def execute(self,query,params=None):
                if query.startswith('INSERT INTO public.alpaca_activities') and owner.before_ledger_insert:
                    callback=owner.before_ledger_insert;owner.before_ledger_insert=None;callback()
                return cur.execute(query,params)
            def __getattr__(self,name):return getattr(cur,name)
        return Wrapped()

def synthetic_response(path,params=None,**kwargs):
    """All broker inputs are synthetic; no actual HTTP request is delegated."""
    cutoff=I.utc(os.environ[I.ENV_CUTOFF])
    if path.endswith('/v2/account'):
        payload={'id':ACCOUNT}
    elif path.endswith('/v2/account/activities'):
        if not params or I.utc(params['until'])!=cutoff:raise RuntimeError('SYNTHETIC_ACTIVITY_CUTOFF')
        payload=[] if params.get('page_token') else [json.loads(row()['raw'])]
    elif path.endswith('/v2/orders'):
        if I.utc(params['until'])!=cutoff:raise RuntimeError('SYNTHETIC_ORDER_CUTOFF')
        if os.environ.get('QUALIFICATION_LATE_ORDER')=='true':
            future=(cutoff+timedelta(seconds=1)).isoformat()
            payload=[dict(id='synthetic-late',symbol='QUALONLY',side='sell',status='filled',created_at=B.isoformat(),
                updated_at=future,filled_at=future,submitted_at=B.isoformat(),filled_avg_price='11',filled_qty='1')]
        else:payload=[]
    else:raise RuntimeError('UNEXPECTED_SYNTHETIC_BROKER_REQUEST')
    return SimpleNamespace(ok=True,status_code=200,headers={},json=lambda:copy.deepcopy(payload),
        raise_for_status=lambda:None)

def run_stage_child(mode):
    global CHILD_APPLICATION_RESULT
    logging.basicConfig(level=logging.INFO,format='%(message)s',stream=sys.stdout,force=True)
    # Only dotenv configuration and read-only broker transport are adapters.
    # All ingestion/policy/checkpoint/strict-consumer application code is real.
    if mode=='activities-child':
        from scripts import fetch_account_activities as f
        with mock.patch.object(f,'load_dotenv',lambda:None),mock.patch.object(f.requests,'get',synthetic_response):
            rc=f.main(list(S.STAGES[0].args))
            CHILD_APPLICATION_RESULT=dict(kind='ingestor_main',returned_rc=rc,success=rc==0)
            return rc
    from scripts import execute_trades as e
    executor=e.TradeExecutor.__new__(e.TradeExecutor)
    executor.config=SimpleNamespace(reconcile_use_watermark=True,reconcile_lookback_days=14,reconcile_limit=500,
        reconcile_overlap_secs=300)
    executor.metrics=SimpleNamespace();executor.client=SimpleNamespace(get_all_positions=lambda:[])
    with mock.patch.object(e,'load_env',lambda:None),mock.patch.object(e,'alpaca_http_get',synthetic_response):
        result=e.TradeExecutor.reconcile_closed_trades(executor,lookback_days=14,limit=500)
        rc=0 if e._operational_reconciliation_complete(executor,result) else 1
        state=executor.metrics.reconciliation
        CHILD_APPLICATION_RESULT=dict(kind='reconcile_method',returned_rc=rc,success=result is True,
            reconciliation={key:state.get(key) for key in ('scope','operational_complete','activity_ingestion_complete',
                'recovery_complete','accounting_complete','watermark_advanced','reason')},
            failure_evidence=e.executor_health.reconciliation_failure_payload(state.get('failure_evidence'),failed=rc!=0))
        return rc

def validate_child_receipt(receipt,*,mode,workflow_id,cutoff,exit_code,late):
    allowed={'source_head','python','platform','production_data','provider_calls_authorized','financial_writes',
        'CRC_activation','guards','cleanup','guard_events','deadline_received','server_verified','mode',
        'workflow_id','cutoff_utc','application_result','status','process_rc','finished_at_utc','qualification','child_receipts'}
    if (type(receipt) is not dict or receipt.get('source_head')!=SOURCE_HEAD or receipt.get('mode')!=mode
            or set(receipt)-allowed or receipt.get('platform')!='linux'
            or not str(receipt.get('python','')).startswith('3.10.')
            or any(receipt.get(k) is not False for k in ('production_data','provider_calls_authorized','financial_writes','CRC_activation'))
            or receipt.get('workflow_id')!=workflow_id or I.utc(receipt.get('cutoff_utc'))!=I.utc(cutoff)
            or receipt.get('server_verified') is not True or receipt.get('deadline_received') is not False
            or receipt.get('guards')!=[] or receipt.get('guard_events')!=[] or receipt.get('child_receipts')!={} or 'failure' in receipt
            or type(receipt.get('process_rc')) is not int or receipt['process_rc'] not in (0,1)
            or type(exit_code) is not int or receipt['process_rc']!=exit_code
            or receipt.get('status')!=('PASS' if exit_code==0 else 'FAIL')):
        raise RuntimeError('CHILD_IDENTITY_OR_EXECUTION_INVALID')
    clean=receipt.get('cleanup',{})
    if not (clean.get('all_connections_closed') is True and clean.get('all_owned_children_returned') is True
            and type(clean.get('owned_backend_sessions_remaining')) is int and clean['owned_backend_sessions_remaining']==0):
        raise RuntimeError('CHILD_CLEANUP_UNKNOWN')
    app=receipt.get('application_result')
    if not isinstance(app,dict) or type(app.get('returned_rc')) is not int or app['returned_rc']!=exit_code:
        raise RuntimeError('CHILD_APPLICATION_RESULT_INVALID')
    if mode=='activities-child':
        if app.get('kind')!='ingestor_main' or app.get('success') is not True or exit_code!=0:
            raise RuntimeError('INGESTOR_DID_NOT_COMPLETE')
        return 'INGESTOR_COMPLETED'
    state=app.get('reconciliation');failure=app.get('failure_evidence')
    if (app.get('kind')!='reconcile_method' or not isinstance(state,dict) or state.get('scope')!='ACTIVE_RECONCILIATION'
            or any(state.get(k) is not False for k in ('recovery_complete','accounting_complete','watermark_advanced'))
            or not isinstance(failure,dict) or failure.get('status')!='VALID'
            or failure.get('evidence_complete') is not True or failure.get('truncated') is not False):
        raise RuntimeError('RECONCILIATION_EVIDENCE_INVALID')
    if late:
        expected={'ORDERS_INVALID':1,'ORDERS_READ_FAILED':1,'ORDERS_READ_INCOMPLETE':1}
        if (exit_code!=1 or app.get('success') is not False or state.get('operational_complete') is not False
                or state.get('activity_ingestion_complete') is not False or failure.get('reason_counts')!=expected
                or failure.get('total_count')!=3):
            raise RuntimeError('EXPECTED_LATE_ORDER_REFUSAL_UNPROVED')
        return 'EXPECTED_LATE_ORDER_REFUSAL'
    if (exit_code!=0 or app.get('success') is not True or state.get('operational_complete') is not True
            or state.get('activity_ingestion_complete') is not True
            or state.get('reason')!='DURABLE_INGESTION_PROVED_ACCOUNTING_HELD'
            or failure.get('total_count')!=0 or failure.get('reason_counts')!={}):
        raise RuntimeError('RECONCILIATION_DID_NOT_COMPLETE')
    return 'BOUNDED_RECONCILIATION_COMPLETED'

def admit_child_receipt(receipt,**context):
    try:return validate_child_receipt(receipt,**context)
    except Exception:
        # run_workflow catches ordinary runner errors. A qualification guard is
        # also permanent parent evidence, so a negative case cannot hide it.
        deny('CHILD_EVIDENCE_INVALID')

def child_summaries():
    result={}
    for key in sorted(set(CHILD_CLASSES)|set(CHILD_RECEIPTS)):
        receipt=CHILD_RECEIPTS.get(key)
        if type(receipt) is not dict:
            result[key]={'classification':CHILD_CLASSES.get(key,'MALFORMED_CHILD_RECEIPT'),'missing_or_malformed_receipt':True};continue
        app=receipt.get('application_result');failure=app.get('failure_evidence',{}) if isinstance(app,dict) else {}
        raw_failure=receipt.get('failure');kind=raw_failure.get('type') if isinstance(raw_failure,dict) else None
        result[key]=dict(classification=CHILD_CLASSES.get(key,'CHILD_EVIDENCE_UNVALIDATED'),
            status=receipt.get('status') if receipt.get('status') in ('PASS','FAIL') else 'UNKNOWN',
            process_rc=receipt.get('process_rc') if type(receipt.get('process_rc')) is int else None,
            deadline_received=receipt.get('deadline_received') is True,server_verified=receipt.get('server_verified') is True,
            guard_categories=[x for x in receipt.get('guards',[])[:16] if isinstance(x,str) and re.fullmatch('[A-Z_]{1,64}',x)] if isinstance(receipt.get('guards'),list) else ['INVALID'],
            failure_type=kind if isinstance(kind,str) and re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,63}',kind) else None,
            application_reason_counts={k:v for k,v in failure.get('reason_counts',{}).items() if re.fullmatch('[A-Z_]{1,64}',str(k)) and type(v) is int and 0<=v<=1000} if isinstance(failure,dict) and isinstance(failure.get('reason_counts'),dict) else {})
    return result

def workflow(late=False):
    """Real account-sync + native child processes using a fixed test bootstrap."""
    original=subprocess.Popen;mapped=[]
    def launch(command,*args,**kwargs):
        allowed={tuple([sys.executable,'-m',stage.module,*stage.args]):stage.name for stage in S.STAGES[:2]}
        name=allowed.get(tuple(command))
        if name is None:raise RuntimeError('UNEXPECTED_STAGE_COMMAND')
        mode='activities-child' if name=='activities' else 'reconcile-child'
        target=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--source-root',str(ROOT),'--mode',mode]
        ALLOWED_CHILDREN.append(target)
        process=original(target,*args,**kwargs);CHILDREN.append(process);mapped.append(mode);return process
    def runner(stage,seconds,lock_fd):
        mode='activities-child' if stage.name=='activities' else 'reconcile-child'
        key=mode+':'+stage.workflow_id
        try:
            with mock.patch.object(subprocess,'Popen',launch):result=S.run_stage(stage,min(seconds,60),lock_fd)
            path=OUT/(mode+'-'+stage.workflow_id+'-result.json')
            if not path.is_file() or path.stat().st_size>65536:raise RuntimeError('CHILD_RECEIPT_UNAVAILABLE')
            receipt=json.loads(path.read_text());CHILD_RECEIPTS[key]=receipt
            CHILD_CLASSES[key]=admit_child_receipt(receipt,mode=mode,workflow_id=stage.workflow_id,cutoff=stage.cutoff_utc,
                exit_code=result.get('exit_code'),late=late and mode=='reconcile-child')
            return result
        except Exception:
            CHILD_CLASSES[key]='CHILD_EVIDENCE_INVALID'
            if 'CHILD_EVIDENCE_INVALID' not in GUARDS:deny('CHILD_EVIDENCE_INVALID')
            raise
    with mock.patch.dict(os.environ,{'QUALIFICATION_LATE_ORDER':'true' if late else 'false'}):
        report=S.run_workflow(OUT/('workflow-'+uuid.uuid4().hex),stage_timeout=60,runner=runner)
    if mapped!=['activities-child','reconcile-child']:raise RuntimeError('STAGE_CHAIN_NOT_EXECUTED')
    return report

class NativeCases(unittest.TestCase):
    def setUp(self):fixture()
    def test_01_explicit_ddl_and_constraints(self):
        self.assertEqual(sql("SELECT count(*) FROM information_schema.columns WHERE table_schema='public' AND table_name='activity_ingestion_checkpoints'",fetch=True)[0][0],13)
        self.assertFalse(any('activity_ingestion_checkpoints' in text for text in M.TABLE_STATEMENTS))
        self.assertEqual(sql("SELECT count(*) FROM pg_catalog.pg_indexes WHERE schemaname='public' AND tablename='activity_ingestion_checkpoints'",fetch=True)[0][0],2)
    def test_02_native_commit_roundtrip_and_separate_reader(self):
        c=connect('qualification_runtime');result=commit_rows(c);c.close()
        reader=connect('qualification_runtime')
        with reader:proof=consume_rows(reader)
        reader.close();self.assertEqual(result['inserted'],1);self.assertEqual(proof['rows'],1);self.assertEqual(counts(),(1,1))
    def test_03_actual_trigger_failures_roll_back_all_writes(self):
        for table in ('alpaca_activities','activity_ingestion_checkpoints','reconcile_state'):
            with self.subTest(table=table):
                fixture()
                sql("CREATE FUNCTION public.fail_write() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'SYNTHETIC_WRITE_REFUSAL'; END $$; CREATE TRIGGER refusal BEFORE INSERT ON public."+table+' FOR EACH ROW EXECUTE FUNCTION public.fail_write()')
                c=connect('qualification_runtime')
                with self.assertRaises(PG.Error):commit_rows(c)
                c.close();self.assertEqual(counts(),(0,0));self.assertEqual(sql('SELECT last_after FROM public.reconcile_state WHERE id=2',fetch=True),[(B,)])
        fixture()
        sql("CREATE FUNCTION public.corrupt_readback() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN NEW.source_fingerprint:=repeat('b',64); RETURN NEW; END $$; CREATE TRIGGER mismatch BEFORE INSERT ON public.activity_ingestion_checkpoints FOR EACH ROW EXECUTE FUNCTION public.corrupt_readback()")
        c=connect('qualification_runtime')
        with self.assertRaisesRegex(RuntimeError,'CHECKPOINT_READBACK'):commit_rows(c)
        c.close();self.assertEqual(counts(),(0,0))
    def test_04_protected_duplicates_are_not_reinserted(self):
        c=connect('qualification_runtime');commit_rows(c);c.close()
        sql("CREATE FUNCTION public.protect_original() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'ORIGINAL_INSERT_FORBIDDEN'; END $$; CREATE TRIGGER protected_original BEFORE INSERT ON public.alpaca_activities FOR EACH ROW EXECUTE FUNCTION public.protect_original()")
        c=connect('qualification_runtime');result=commit_rows(c,workflow_id='b'*32,before=U);c.close()
        self.assertEqual(result['inserted'],0);self.assertEqual(result['matched_existing'],1);self.assertEqual(counts(),(1,2))
    def test_05_conflicting_payload_rolls_back_without_overwrite(self):
        c=connect('qualification_runtime');commit_rows(c);c.close();changed=row();changed['price']=Decimal('99')
        c=connect('qualification_runtime')
        with self.assertRaisesRegex(RuntimeError,'LEDGER_CONFLICT'):commit_rows(c,workflow_id='b'*32,before=U,rows=[changed])
        c.close();self.assertEqual(counts(),(1,1));self.assertEqual(sql('SELECT price FROM public.alpaca_activities',fetch=True),[(Decimal('10.5'),)])
    def test_06_two_sessions_lock_timeout_and_cursor_cas(self):
        holder=connect();writer=connect('qualification_runtime')
        try:
            with holder.cursor() as cur:cur.execute("SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('jbravo:activity-ingestion'))")
            with self.assertRaises(PG.Error):commit_rows(writer)
        finally:holder.rollback();holder.close();writer.close()
        sql('UPDATE public.reconcile_state SET last_after=%s WHERE id=2',(B+timedelta(seconds=1),))
        writer=connect('qualification_runtime')
        with self.assertRaisesRegex(RuntimeError,'CURSOR_CHANGED'):commit_rows(writer)
        writer.close();self.assertEqual(counts(),(0,0))
    def test_07_real_concurrent_duplicate_returning_count(self):
        def other_insert():
            values=row();names=','.join(I.FIELDS);slots=','.join('%s' if n!='raw' else '%s::jsonb' for n in I.FIELDS)
            sql('INSERT INTO public.alpaca_activities('+names+') VALUES('+slots+')',tuple(values[n] for n in I.FIELDS))
        c=connect('qualification_runtime');result=commit_rows(Proxy(c,before_ledger_insert=other_insert));c.close()
        self.assertEqual(result['inserted'],0);self.assertEqual(result['matched_existing'],1);self.assertEqual(counts(),(1,1))
    def test_08_consumed_restart_refuses_new_workflow_is_idempotent(self):
        c=connect('qualification_runtime');commit_rows(c)
        with self.assertRaises(PG.Error):commit_rows(c,before=U)
        result=commit_rows(c,workflow_id='b'*32,before=U);c.close();self.assertEqual(result['inserted'],0);self.assertEqual(counts(),(1,2))
    def test_09_commit_ack_loss_is_unknown_and_readable(self):
        def lost():raise OSError('SYNTHETIC_COMMIT_ACK_LOSS')
        c=connect('qualification_runtime')
        with self.assertRaises(OSError):commit_rows(Proxy(c,after_commit=lost))
        c.close();reader=connect('qualification_runtime')
        with reader:proof=consume_rows(reader)
        reader.close();self.assertEqual(proof['rows'],1);self.assertEqual(counts(),(1,1))
    def test_10_tamper_source_recipe_cursor_and_cutoff_refuse(self):
        for column,value in (('source_fingerprint','b'*64),('recipe_sha256','c'*64),('manifest_sha256','d'*64)):
            with self.subTest(column=column):
                fixture();c=connect('qualification_runtime');commit_rows(c);c.close()
                sql('UPDATE public.activity_ingestion_checkpoints SET '+column+'=%s',(value,))
                c=connect('qualification_runtime')
                with self.assertRaises(RuntimeError):
                    with c:consume_rows(c)
                c.close()
        fixture();c=connect('qualification_runtime');commit_rows(c)
        with self.assertRaisesRegex(RuntimeError,'COVERAGE_INSUFFICIENT'):
            with c:consume_rows(c,required_until=U+timedelta(seconds=1))
        c.close();sql('UPDATE public.reconcile_state SET last_after=%s WHERE id=2',(B,))
        c=connect('qualification_runtime')
        with self.assertRaisesRegex(RuntimeError,'CURSOR_REWOUND'):
            with c:consume_rows(c)
        c.close()
    def test_11_zero_row_interval_is_not_global_history(self):
        c=connect('qualification_runtime');commit_rows(c,rows=[])
        with c:proof=consume_rows(c)
        c.close();self.assertEqual(proof['rows'],0);self.assertEqual(proof['scope'],'ACTIVE_RECONCILIATION_CREATED_AT_INTERVAL_V1');self.assertNotIn('recovery_complete',proof)
    def test_12_minimum_runtime_rights_and_no_runtime_ddl(self):
        c=connect('qualification_runtime');commit_rows(c)
        with c:consume_rows(c)
        with self.assertRaises(PG.Error):
            with c:
                with c.cursor() as cur:cur.execute('CREATE TABLE public.forbidden_runtime_table(id int)')
        with self.assertRaises(PG.Error):
            with c:
                with c.cursor() as cur:cur.execute('DELETE FROM public.activity_ingestion_checkpoints')
        c.close();self.assertEqual(counts(),(1,1))
    def test_13_full_native_workflow_and_both_consumers_keep_holds(self):
        report=workflow();self.assertEqual(report['status'],'degraded');self.assertEqual(report['stages'][0]['status'],'ok');stage=report['stages'][1]
        self.assertEqual(stage['completion_contract'],'ACTIVE_RECONCILIATION_INGESTION_V2');self.assertTrue(stage['reconciliation']['activity_ingestion_complete'])
        for key in ('accounting_complete','recovery_complete','watermark_advanced','snapshot_eligible'):self.assertIs(stage['reconciliation'][key],False)
        self.assertEqual(report['stages'][2]['status'],'not_run');self.assertEqual(counts(),(1,1));self.assertEqual(sql('SELECT last_after FROM public.reconcile_state WHERE id=1',fetch=True),[(REQUIRED,)])
    def test_14_full_method_late_order_refuses_without_completion(self):
        report=workflow(late=True);self.assertEqual(report['status'],'failed');self.assertEqual(report['stages'][0]['status'],'ok');self.assertEqual(report['stages'][1]['status'],'failed');self.assertEqual(report['stages'][2]['status'],'not_run')
        stage=report['stages'][1];self.assertEqual(stage['exit_code'],1)
        self.assertEqual(set(stage['failure_markers']),{'RECONCILE_ORDERS_INVALID','RECONCILE_ORDERS_UNAVAILABLE','RECONCILE_WATERMARK_HELD'})
        self.assertEqual(validate_child_receipt(CHILD_RECEIPTS['reconcile-child:'+report['run_id']],mode='reconcile-child',
            workflow_id=report['run_id'],cutoff=report['started_at'],exit_code=1,late=True),'EXPECTED_LATE_ORDER_REFUSAL')

def main():
    global ROOT,OUT,WORK,CLEANING
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',required=True)
    parser.add_argument('--mode',choices=('suite','activities-child','reconcile-child','prepare-check'),default='suite')
    args=parser.parse_args();ROOT=Path(args.source_root).resolve();verify_source()
    if args.mode=='prepare-check':print(json.dumps(dict(status='SOURCE_PINS_VERIFIED',count=len(EXPECTED_SOURCE))));return 0
    if (sys.platform!='linux' or sys.version_info[:2]!=(3,10) or os.environ.get('GITHUB_ACTIONS')!='true'
            or os.environ.get('GITHUB_REPOSITORY')!='RasPatrick55-dev/JBRAVO_Screener'
            or os.environ.get('GITHUB_RUN_ATTEMPT')!='1'
            or os.environ.get('QUALIFICATION_ADMISSION')!='INGESTION_CHECKPOINT_NATIVE_PREP001'
            or not re.fullmatch('[0-9a-f]{40}',SOURCE_HEAD)):
        raise RuntimeError('CI_ONLY_ONE_USE_ADMISSION_REQUIRED')
    WORK=Path(os.environ['GITHUB_WORKSPACE']).resolve();OUT=WORK/'qualification-output';OUT.mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM,outer_deadline)
    record=dict(source_head=SOURCE_HEAD,python=sys.version,platform=sys.platform,production_data=False,
        provider_calls_authorized=False,financial_writes=False,CRC_activation=False,guards=GUARDS)
    rc=1;capture=BoundedCapture()
    try:
        try:
            with contextlib.redirect_stdout(capture) if args.mode=='suite' else contextlib.nullcontext(),contextlib.redirect_stderr(capture) if args.mode=='suite' else contextlib.nullcontext():
                configure_runtime()
                if args.mode!='suite':rc=run_stage_child(args.mode)
                else:
                    for user in USERS[1:]:
                        if sql('SELECT EXISTS(SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=%s)',(user,),fetch=True)[0][0]:raise RuntimeError('SYNTHETIC_ROLE_ALREADY_EXISTS')
                        sql('CREATE ROLE '+user+" LOGIN PASSWORD '"+PASSWORD+"'")
                    native=unittest.TextTestRunner(stream=io.StringIO(),resultclass=Results).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeCases))
                    record['native']=dict(methods=native.testsRun,subcases=native.subcases,failures=len(native.failures),errors=len(native.errors),skipped=len(native.skipped),details=[dict(test=str(t),detail=d) for t,d in native.failures+native.errors])
                    repository=unittest.TextTestRunner(stream=io.StringIO(),resultclass=Results).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_activity_ingestion_checkpoint.py'))
                    record['repository']=dict(methods=repository.testsRun,failures=len(repository.failures),errors=len(repository.errors),skipped=len(repository.skipped),details=[dict(test=str(t),detail=d) for t,d in repository.failures+repository.errors])
                    rc=0 if native.wasSuccessful() and repository.wasSuccessful() and native.testsRun==14 and repository.testsRun==65 and not native.skipped and not repository.skipped and not GUARDS else 1
                verify_source()
        finally:CLEANING=True
    except BaseException as error:record['failure']=safe_error(error);rc=1
    finally:
        try:record['cleanup']=cleanup()
        except BaseException as error:
            record['cleanup']=dict(all_connections_closed=all(c.closed for c in CONNECTIONS),all_owned_children_returned=all(p.poll() is not None for p in CHILDREN),owned_backend_sessions_remaining='UNKNOWN',failure=safe_error(error));rc=1
        record['guard_events']=list(GUARD_EVENTS);record['guards']=list(GUARDS);record['deadline_received']=DEADLINE_HIT
        record.update(server_verified=SERVER_VERIFIED,mode=args.mode,workflow_id=os.environ.get('JBR_ACCOUNT_SYNC_INVOCATION_ID'),cutoff_utc=os.environ.get('JBR_ACCOUNT_SYNC_CUTOFF_UTC'),application_result=CHILD_APPLICATION_RESULT)
        record['child_receipts']=child_summaries()
        if not SERVER_VERIFIED or DEADLINE_HIT or GUARDS or not record['cleanup']['all_connections_closed'] or not record['cleanup']['all_owned_children_returned'] or record['cleanup']['owned_backend_sessions_remaining']!=0:rc=1
        record.update(status='PASS' if rc==0 else 'FAIL',process_rc=rc,finished_at_utc=datetime.now(timezone.utc).isoformat())
        record['qualification']='Native SQL/driver/process tests with synthetic rows. Full reconciliation METHOD executes unchanged, instantiated without executor startup; no complete execute-trades CLI/order path. Account-sync launches real native child adapters that call actual ingestor main/full reconciliation method. Dotenv and broker reads are synthetic adapters. No real provider snapshot, real late-fee finality, production schema/privileges or trading readiness is established.'
        name='suite-result.json' if args.mode=='suite' else args.mode+'-'+os.environ['JBR_ACCOUNT_SYNC_INVOCATION_ID']+'-result.json'
        with (OUT/name).open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    if args.mode=='suite':print(json.dumps(record))
    return rc

if __name__=='__main__':raise SystemExit(main())
