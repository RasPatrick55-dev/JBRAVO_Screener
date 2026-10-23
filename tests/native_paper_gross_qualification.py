"""One CI-only synthetic qualification of fixed application source; never production.

Native PostgreSQL/CLI cases are separate from repository tests using fake adapters.
No captured account rows, provider calls, activation or operating acceptance.
"""
import argparse, builtins, contextlib, hashlib, importlib, io, json, math, os
from pathlib import Path
import re, signal, socket, subprocess, sys, threading, time, unittest
from datetime import date, datetime, timezone, timedelta

SOURCE_HEAD = os.environ.get('GITHUB_SHA','UNPUBLISHED_REPAIR')
EXPECTED_SOURCE = {'dashboards/dashboard_app.py': 'f8317129ba531386007108edb89c77d56e0025ad1432631362fb41f91aa1f438', 'dashboards/data_io.py': 'ffc06966f5b255e3c8e47360f2ad302b5f5aeb22afdfb46ebc8d3e120f851425', 'scripts/db.py': '89d9443e9bd60a721e7471fb8e611a4c43955b6bc8ec3d59bb7d252c8ee87c23', 'scripts/gross_paper_accounting.py': 'b34bb85de27fbb409954943dcd996d0a66d5ae0d0aac6599776db4ca2da569d4', 'scripts/metrics.py': '4d08f31a6deff14aa740489a7faebf55c879fcd5bff10963cc8edcfd159ebc94', 'scripts/pipeline_postflight.py': 'c7d29880164db0ab25563d28a62e6bf4de0313f8025c423536f504e775da0e40', 'scripts/primary_pipeline.py': '1e728cbbc304546b28db9d53f496649716102db6acbd29eec8481ff7de6494a3', 'scripts/run_pipeline.py': '2b022158d7d0d65a6f3dd780f717c3d83dbd078399b5565580c6a03947b0acb9', 'scripts/trade_performance.py': 'd40fe11be566f4e1972d3f5d4d4c3309b74b4459d81f80942c9c456b5cc297e9', 'tests/test_gross_paper_accounting.py': 'de3af06fa9bb2c20ae7670de4319f0a32b15b12260758ec7fc221f247a8a96f1', 'tests/test_paper_gross_integration.py': '9f15f09d58d4fd90d6d36d042af0afe39230161ae7573cc7b112463a905c5a85', 'tests/test_paper_gross_recovery.py': '6e7b9c136fc17258c1617f50da0578c092c0bf1e6dac3b04f773d43056d4d002', 'tests/test_pipeline_postflight.py': '075afb76b7297d46749fb0677eafde0887faee54941f25de4b8e3aee5148d458', 'scripts/db_migrate.py': '6a10b518efc179a64a0a374e57dda8bac07d805034be673bd4033f777708b575'}
DATABASE = 'qualification_745'
PORT = 55432
PASSWORD = 'EPHEMERAL_QUALIFICATION_ONLY_745'
USERS = ('qualification_admin', 'qualification_runtime', 'qualification_rowlock')
DAY = '2000-01-03'
ROOT = OUT = WORK = None
PG = DB = G = D = PD = M = None
CONNECTIONS = []
GUARDS = []
GUARD_EVENTS = []
CHILD_RECEIPTS = {}
CHILDREN = []
ALLOWED_CHILDREN = []
SERVER_VERIFIED = False
CLEANING = False
DEADLINE_HIT = False
VERSION_METADATA_FALLBACKS = []


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
    policy=str(error) if re.fullmatch('TRADE_(?:POLICY|ALLOCATION)_[A-Z0-9_]{1,100}',str(error)) else None
    return dict(type=type(error).__name__,sqlstates=sorted(set(codes)),policy_code=policy)


def verify_source():
    if len(EXPECTED_SOURCE)!=14:raise RuntimeError('SOURCE_PIN_COUNT')
    for name,expected in EXPECTED_SOURCE.items():
        path=ROOT/name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):raise RuntimeError('SOURCE_PATH')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise RuntimeError('SOURCE_PIN')


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
        params['connect_timeout']=3;params['application_name']='qual745-'+os.environ['GITHUB_RUN_ID']
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


def configure_runtime():
    global PG,DB,G,D,PD,M,SERVER_VERIFIED
    sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT))
    (ROOT/'logs').mkdir(exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
    os.environ['DATABASE_URL']=ds()
    configure_test_capabilities()
    PG=install_guards()
    from scripts import db, gross_paper_accounting, trade_disposition, db_migrate
    import pandas
    DB,G,D,PD,M=db,gross_paper_accounting,trade_disposition,pandas,db_migrate
    server=sql('SELECT pg_catalog.current_database(),pg_catalog.current_setting(\'server_version_num\')',fetch=True)
    if server!=[(DATABASE,'120002')]:raise RuntimeError('SYNTHETIC_SERVER_IDENTITY')
    SERVER_VERIFIED=True


def fixture():
    # Only a disposable, fixed-name database on the owned loopback service.
    # No production capture or native CRC activation is used.
    if GUARDS:raise unittest.SkipTest('GUARD_REFUSAL_STOPS_REMAINING_OPERATIONS')
    for connection in CONNECTIONS:
        if not connection.closed:connection.close()
    sql('DROP SCHEMA public CASCADE; CREATE SCHEMA public; GRANT USAGE ON SCHEMA public TO PUBLIC')
    sql('''CREATE TABLE public.trades(trade_id bigint PRIMARY KEY,symbol text,status text,qty numeric,
        entry_price numeric,exit_price numeric,realized_pnl numeric,exit_order_id text,entry_time timestamptz,
        exit_time timestamptz,updated_at timestamptz);
        INSERT INTO public.trades VALUES(9001,'QUALONLY','CLOSED',2,10,12,NULL,'synthetic-exit',
            '2000-01-03T15:00:00Z','2000-01-03T16:00:00Z','2000-01-03T16:00:01Z');
        CREATE TABLE public.order_events(id int PRIMARY KEY,payload jsonb);
        INSERT INTO public.order_events VALUES(1,'{"synthetic":true}');
        CREATE TABLE public.alpaca_activities(id int PRIMARY KEY,payload jsonb);
        INSERT INTO public.alpaca_activities VALUES(1,'{"synthetic":true}');
        CREATE TABLE public.metrics_daily(run_date date PRIMARY KEY,net_pnl double precision);
        CREATE TABLE public.trade_disposition_policy(singleton int PRIMARY KEY,deployment_id text,database_id text,
            database_name text,policy_revision bigint);
        CREATE TABLE public.trade_disposition_cases(case_id uuid,trade_id bigint,state text,case_revision bigint,
            request jsonb,row_snapshot jsonb,manifest jsonb);
        CREATE TABLE public.trade_disposition_audit(approval_ref text,request_digest text,request jsonb,
            manifest jsonb,result jsonb,policy_revision bigint);
        CREATE TABLE public.backtest_results(run_date date,symbol text,trades int,wins int,losses int,
            win_rate double precision,net_pnl double precision,avg_return double precision);
        INSERT INTO public.backtest_results VALUES('2000-01-03','QUALONLY',2,1,1,50,5,2.5);
        CREATE TABLE public.screener_candidates(run_date date,timestamp timestamptz,symbol text,score double precision,
            exchange text,close double precision,volume double precision,universe_count int,score_breakdown jsonb,
            entry_price double precision,adv20 double precision,atrp double precision,source text,final_score numeric,
            sma9 double precision,ema20 double precision,sma180 double precision,rsi14 double precision,passed_gates boolean,
            gate_fail_reason text,ml_weight_used double precision,run_ts_utc timestamptz,created_at timestamptz);
        INSERT INTO public.screener_candidates(run_date,timestamp,symbol,score,exchange,close,volume,universe_count,
            score_breakdown,entry_price,adv20,atrp,source,run_ts_utc,created_at)
            VALUES('2000-01-03',now(),'QUALONLY',1,'SYNTHETIC',10,1000,1,'{}',10,100000,2,'synthetic',now(),now());
        CREATE TABLE public.screener_ranker_scores_app(run_ts_utc timestamptz,symbol text,model_score_5d double precision)''')
    identity=DB._PAPER_DISPOSITION_BINDING
    sql('INSERT INTO public.trade_disposition_policy VALUES(1,%s,%s,%s,0)',(identity[0],identity[1],DATABASE))
    DB.ensure_top_candidates_table()
    owner=connect()
    try:
        if not M.ensure_paper_gross_schema(owner):raise RuntimeError('SYNTHETIC_SCHEMA_SETUP_FAILED')
    finally:owner.close()


def originals():
    return [sql('SELECT pg_catalog.to_jsonb(t)::text FROM public.'+table+' t ORDER BY 1',fetch=True)
        for table in ('trades','order_events','alpaca_activities','metrics_daily')]


def ranked(infinite=False):
    frame=PD.DataFrame([dict(symbol='QUALONLY',score=1.0,win_rate=50.0,net_pnl=5.0,trades=2,
        wins=1,losses=1,entry_price=10.0,exchange='SYNTHETIC',source='synthetic-backtest')])
    if infinite:frame['profit_factor']=float('inf')
    return frame


def publish(infinite=False):
    return DB.publish_paper_gross_pipeline(DAY,ranked(infinite),DB.gross_paper_performance_snapshot())


def run_child(mode,seconds):
    command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--source-root',str(ROOT),'--mode',mode]
    ALLOWED_CHILDREN.append(command)
    cleanup_proved=False
    try:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,start_new_session=True,env=os.environ.copy(),cwd=ROOT)
        CHILDREN.append(process)
        try:output,_=process.communicate(input='',timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:output,_=process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);output,_=process.communicate(timeout=5)
            raise RuntimeError('OWNED_CHILD_TIMEOUT')
        if len(output)>16384:raise RuntimeError('OWNED_CHILD_OUTPUT_CAP')
        result=json.loads(output)
        CHILD_RECEIPTS[mode]=result
        cleaned=result.get('cleanup',{})
        cleanup_proved=(cleaned.get('all_connections_closed') is True and cleaned.get('all_owned_children_returned') is True
                and (cleaned.get('owned_backend_sessions_remaining')==0 or
                     (result.get('server_verified') is False and cleaned.get('tracked_connections')==0)))
        if not cleanup_proved:deny('CHILD_CLEANUP_UNKNOWN')
        if process.returncode!=0:raise RuntimeError('OWNED_CHILD_FAILED_'+str(result.get('failure',{}).get('type','UNKNOWN')))
        return result
    finally:
        if not cleanup_proved and 'CHILD_CLEANUP_UNKNOWN' not in GUARDS:
            try:deny('CHILD_CLEANUP_UNKNOWN')
            except RuntimeError:pass


class NativeCases(unittest.TestCase):
    def setUp(self):fixture()
    def test_01_actual_application_bootstrap_in_native_child(self):
        result=run_child('bootstrap-child',60)
        self.assertIn('scripts.run_pipeline',result['imports'])
        self.assertIn('dashboards.dashboard_app',result['imports'])
    def test_02_real_ddl_commit_original_preservation_and_unknown_net(self):
        before=originals();result=publish();self.assertEqual(originals(),before)
        doc=D.document(sql('SELECT publication::text FROM public.paper_gross_metrics_daily',fetch=True)[0][0])
        self.assertEqual(doc['summary']['gross_pnl'],'4');self.assertIsNone(doc['summary']['net_pnl'])
        self.assertIsNone(doc['summary']['fees']);self.assertFalse(result['entry_ready'])
        self.assertEqual(sql('SELECT net_pnl FROM public.top_candidates',fetch=True),[(5.0,)])
    def test_03_real_statement_failure_rolls_back_both_derived_writes(self):
        publish();before=sql('SELECT publication::text FROM public.paper_gross_metrics_daily',fetch=True)
        top=sql('SELECT pg_catalog.to_jsonb(t)::text FROM public.top_candidates t',fetch=True)
        sql("CREATE FUNCTION public.qual_refuse() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'synthetic refusal'; END $$; "
            'CREATE TRIGGER qualification_refusal BEFORE INSERT OR UPDATE ON public.paper_gross_metrics_daily FOR EACH ROW EXECUTE FUNCTION public.qual_refuse()')
        with self.assertRaises(Exception):publish()
        self.assertEqual(sql('SELECT publication::text FROM public.paper_gross_metrics_daily',fetch=True),before)
        self.assertEqual(sql('SELECT pg_catalog.to_jsonb(t)::text FROM public.top_candidates t',fetch=True),top)
    def test_04_real_read_rejects_trade_change_without_fallback(self):
        publish();sql('UPDATE public.trades SET realized_pnl=999 WHERE trade_id=9001')
        with self.assertRaises(D.TradePolicyError):
            with DB.paper_gross_pipeline_evidence(DAY):pass
    def test_05_real_policy_revision_and_candidate_drift_refuse(self):
        publish();sql('UPDATE public.trade_disposition_policy SET policy_revision=1')
        with self.assertRaises(D.TradePolicyError):
            with DB.paper_gross_pipeline_evidence(DAY):pass
        publish();sql('UPDATE public.top_candidates SET score=0.25')
        with self.assertRaises(D.TradePolicyError):
            with DB.paper_gross_pipeline_evidence(DAY):pass
    def test_06_native_positive_infinity_roundtrip(self):
        proof=publish(True)
        self.assertEqual(sql('SELECT profit_factor FROM public.top_candidates',fetch=True)[0][0],float('inf'))
        with DB.paper_gross_pipeline_evidence(DAY) as current:
            self.assertEqual(proof,current)
    def test_07_native_shared_policy_blocks_publisher_with_original_timeout(self):
        holder=connect()
        with holder.cursor() as cursor:
            cursor.execute("SELECT pg_catalog.pg_advisory_xact_lock_shared(pg_catalog.hashtext('jbravo:trade-disposition:policy'))")
        result=[]
        def worker():
            try:publish()
            except Exception as error:result.append(safe_error(error))
        started=time.monotonic();thread=threading.Thread(target=worker);thread.start();thread.join(8)
        try:
            self.assertFalse(thread.is_alive());self.assertTrue(result)
            self.assertLess(time.monotonic()-started,8)
        finally:holder.rollback();holder.close();thread.join(1)
        publish()
    def test_08_runtime_minimum_privileges_without_policy_or_schema_expansion(self):
        publish()  # Provision the derived table as fixture owner, not runtime.
        for user,extra in (('qualification_runtime',''),
                           ('qualification_rowlock','GRANT UPDATE ON public.trade_disposition_policy TO qualification_rowlock')):
            with self.subTest(runtime=user):
                sql('GRANT USAGE ON SCHEMA public TO '+user+'; GRANT SELECT ON ALL TABLES IN SCHEMA public TO '+user+'; '
                    'GRANT INSERT,UPDATE,DELETE ON public.top_candidates TO '+user+'; '
                    'GRANT INSERT,UPDATE ON public.paper_gross_metrics_daily TO '+user+'; '+extra)
                self.assertEqual(sql('SELECT pg_catalog.has_schema_privilege(%s,\'public\',\'CREATE\')',(user,),fetch=True),[(False,)])
                if user=='qualification_runtime':
                    self.assertEqual(sql('SELECT pg_catalog.has_table_privilege(%s,\'public.trade_disposition_policy\',\'UPDATE\')',(user,),fetch=True),[(False,)])
                before=originals()
                os.environ['DATABASE_URL']=ds(user)
                try:
                    snapshot=DB.gross_paper_performance_snapshot()
                    result=DB.publish_paper_gross_pipeline(DAY,ranked(),snapshot)
                    self.assertTrue(result['gross_input_complete'])
                finally:os.environ['DATABASE_URL']=ds()
                self.assertEqual(originals(),before)
    def test_09_missing_other_row_fields_cannot_be_hidden_by_symbol(self):
        sql("INSERT INTO public.trades VALUES(9002,'QUALONLY','CLOSED',2,10,NULL,NULL,NULL,'2000-01-03T15:00:00Z','2000-01-03T16:00:00Z',now())")
        with self.assertRaises(D.TradePolicyError):DB.gross_paper_performance_snapshot()
        sql('UPDATE public.trades SET exit_price=12 WHERE trade_id=9002')
        sql("INSERT INTO public.trades SELECT 10000+i,'QUALCAP','CLOSED',2,10,12,NULL,'synthetic-exit',"
            "'2000-01-03T15:00:00Z','2000-01-03T16:00:00Z',now() FROM generate_series(1,10000) i")
        with self.assertRaises(D.TradePolicyError):DB.gross_paper_performance_snapshot()
    def test_10_real_daily_reader_retains_shared_policy_until_caller_returns(self):
        publish();other=connect()
        try:
            with DB.paper_gross_pipeline_evidence(DAY):
                with other.cursor() as cursor:
                    cursor.execute("SELECT pg_catalog.pg_try_advisory_xact_lock(pg_catalog.hashtext('jbravo:trade-disposition:policy'))")
                    self.assertEqual(cursor.fetchone(),(False,))
            other.rollback()
            with other.cursor() as cursor:
                cursor.execute("SELECT pg_catalog.pg_try_advisory_xact_lock(pg_catalog.hashtext('jbravo:trade-disposition:policy'))")
                self.assertEqual(cursor.fetchone(),(True,))
        finally:other.rollback();other.close()
    def test_11_actual_metrics_cli_native_process_and_upstream_queries(self):
        before=originals();result=run_child('metrics-child',180)
        self.assertEqual(result['metrics_rc'],0);self.assertEqual(originals(),before)
        with DB.paper_gross_pipeline_evidence(DAY) as proof:
            self.assertTrue(proof['gross_input_complete']);self.assertFalse(proof['net_eligible'])
    def test_12_native_publication_time_and_identity_are_revalidated(self):
        publish()
        with self.assertRaises(D.TradePolicyError):
            with DB.paper_gross_pipeline_evidence(DAY,not_before=datetime.now(timezone.utc)+timedelta(seconds=1)):pass
        sql("UPDATE public.trade_disposition_policy SET database_id='00000000-0000-4000-8000-000000000999'")
        with self.assertRaises(D.TradePolicyError):DB.gross_paper_performance_snapshot()
    def test_13_missing_schema_rolls_back_until_explicit_targeted_setup(self):
        publish();before=originals()
        top=sql('SELECT pg_catalog.to_jsonb(t)::text FROM public.top_candidates t',fetch=True)
        sql('DROP TABLE public.paper_gross_metrics_daily')
        with self.assertRaises(PG.errors.UndefinedTable):publish()
        self.assertEqual(sql('SELECT pg_catalog.to_jsonb(t)::text FROM public.top_candidates t',fetch=True),top)
        self.assertEqual(originals(),before)
        self.assertEqual(sql("SELECT pg_catalog.to_regclass('public.paper_gross_metrics_daily')",fetch=True),[(None,)])
        owner=connect()
        try:self.assertTrue(M.ensure_paper_gross_schema(owner))
        finally:owner.close()
        publish();self.assertEqual(originals(),before)

    def test_14_native_derived_values_beyond_source_number_bounds(self):
        cases=(('1e-1000','1e-1000','2e-1000','0.'+'0'*1999+'1'),
               ('1e999','1','1e999','9'*999+'0'*999))
        for qty,entry,exit_price,expected in cases:
            with self.subTest(qty=qty):
                sql('UPDATE public.trades SET qty=%s::numeric,entry_price=%s::numeric,exit_price=%s::numeric',
                    (qty,entry,exit_price))
                before=originals();proof=publish()
                doc=D.document(sql('SELECT publication::text FROM public.paper_gross_metrics_daily',fetch=True)[0][0])
                self.assertEqual(doc['summary']['gross_pnl'],expected)
                self.assertIsNone(doc['summary']['fees']);self.assertIsNone(doc['summary']['net_pnl'])
                self.assertFalse(doc['summary']['accounting_complete']);self.assertFalse(proof['entry_ready'])
                self.assertEqual(originals(),before)
                self.assertEqual(sql('SELECT net_pnl FROM public.top_candidates',fetch=True),[(5.0,)])
                with DB.paper_gross_pipeline_evidence(DAY) as read:self.assertEqual(read,proof)


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
                    "AND application_name LIKE 'qual745-%%' AND pid<>pg_catalog.pg_backend_pid()",(DATABASE,))
                remaining=cursor.fetchone()[0]
        finally:probe.rollback();probe.close()
    return dict(tracked_connections=len(CONNECTIONS),all_connections_closed=all(c.closed for c in CONNECTIONS),
        open_connections_closed_during_cleanup=closed,owned_children=len(CHILDREN),
        all_owned_children_returned=all(p.poll() is not None for p in CHILDREN),owned_backend_sessions_remaining=remaining)


def main():
    global ROOT,OUT,WORK,CLEANING
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',required=True)
    parser.add_argument('--mode',choices=('suite','bootstrap-child','metrics-child','prepare-check'),default='suite')
    args=parser.parse_args();ROOT=Path(args.source_root).resolve();verify_source()
    if args.mode=='prepare-check':
        print(json.dumps(dict(status='STATIC_SOURCE_PINS_VERIFIED',files=len(EXPECTED_SOURCE),source=SOURCE_HEAD)));return 0
    if (sys.platform!='linux' or sys.version_info[:2]!=(3,10) or os.environ.get('GITHUB_ACTIONS')!='true'
            or os.environ.get('GITHUB_REPOSITORY')!='RasPatrick55-dev/JBRAVO_Screener'
            or os.environ.get('GITHUB_RUN_ATTEMPT')!='1'
            or os.environ.get('QUALIFICATION_ADMISSION')!='PR745_FINAL_REVIEW_CORRECTION_PREP001'
            or not re.fullmatch('[a-f0-9]{40}',SOURCE_HEAD)):
        raise RuntimeError('CI_ONLY_ONE_USE_ADMISSION_REQUIRED')
    WORK=Path(os.environ['GITHUB_WORKSPACE']).resolve();OUT=WORK/'qualification-output';OUT.mkdir(exist_ok=True)
    signal.signal(signal.SIGTERM,outer_deadline)
    capture=BoundedCapture();record=dict(source_head=SOURCE_HEAD,python=sys.version,platform=sys.platform,
        production_data=False,CRC_activation=False,provider_calls_authorized=False,guards=GUARDS)
    rc=1
    try:
        try:
            with contextlib.redirect_stdout(capture),contextlib.redirect_stderr(capture):
                configure_runtime()
                if args.mode=='bootstrap-child':
                    imports=[]
                    for name in ('scripts.db','scripts.metrics','scripts.run_pipeline','scripts.primary_pipeline',
                            'scripts.pipeline_postflight','dashboards.data_io','dashboards.dashboard_app'):
                        record['active_import']=name
                        importlib.import_module(name);imports.append(name)
                        if GUARDS:raise RuntimeError('BOOTSTRAP_GUARD_REFUSAL')
                    record.update(imports=imports);rc=0
                elif args.mode=='metrics-child':
                    from scripts import metrics
                    code=metrics.main(['--paper-gross-pipeline','--run-date',DAY])
                    record.update(metrics_rc=code);rc=code
                else:
                    for user in USERS[1:]:
                        exists=sql('SELECT EXISTS(SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=%s)',(user,),fetch=True)[0][0]
                        if exists:raise RuntimeError('SYNTHETIC_ROLE_ALREADY_EXISTS')
                        sql('CREATE ROLE '+user+" LOGIN PASSWORD '"+PASSWORD+"'")
                    native=unittest.TextTestRunner(stream=io.StringIO(),resultclass=Results).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeCases))
                    record['native']=dict(methods=native.testsRun,subcases=native.subcases,failures=len(native.failures),
                        errors=len(native.errors),skipped=len(native.skipped),details=[dict(test=str(t),detail=d) for t,d in native.failures+native.errors])
                    sys.path.insert(0,str(ROOT/'tests'))
                    suite=unittest.TestSuite()
                    for pattern in ('test_gross_paper_accounting.py','test_paper_gross_integration.py','test_paper_gross_recovery.py','test_pipeline_postflight.py'):
                        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
                    repository=unittest.TextTestRunner(stream=io.StringIO(),resultclass=Results).run(suite)
                    record['repository']=dict(methods=repository.testsRun,subcases=repository.subcases,failures=len(repository.failures),
                        errors=len(repository.errors),skipped=len(repository.skipped),details=[dict(test=str(t),detail=d) for t,d in repository.failures+repository.errors])
                    rc=0 if native.wasSuccessful() and repository.wasSuccessful() and not GUARDS else 1
                verify_source()
        finally:CLEANING=True
    except BaseException as error:
        record['failure']=safe_error(error);rc=1
    finally:
        try:record['cleanup']=cleanup()
        except BaseException as error:
            record['cleanup']=dict(all_connections_closed=all(c.closed for c in CONNECTIONS),
                all_owned_children_returned=all(p.poll() is not None for p in CHILDREN),
                owned_backend_sessions_remaining='UNKNOWN',failure=safe_error(error))
            rc=1
        record['parent_guard_refusals']=list(GUARDS)
        record['guard_events']=list(GUARD_EVENTS)
        record['child_receipts']=dict(CHILD_RECEIPTS)
        record['guards']=list(GUARDS)+[guard for child in CHILD_RECEIPTS.values() for guard in child.get('guards',[])][:32]
        record['server_verified']=SERVER_VERIFIED
        record['version_metadata_fallbacks']=list(VERSION_METADATA_FALLBACKS)
        record['deadline_received']=DEADLINE_HIT
        if (DEADLINE_HIT or record['guards'] or not record['cleanup']['all_connections_closed'] or not record['cleanup']['all_owned_children_returned']
                or record['cleanup']['owned_backend_sessions_remaining'] not in (None,0)):rc=1
        record['status']='PASS' if rc==0 else 'FAIL';record['process_rc']=rc
        record['qualification']='Real native SQL and process cases separated from repository fake-adapter tests; synthetic ordinary-trade fixture only, no native CRC allocation activation or provider-backed full Pipeline execution. Checkout data/tmp/debug inputs are quarantined unread by the workflow and inaccessible to application reads. IPv4-only test capabilities avoid a bind probe; optional version-banner Git unavailability is explicitly adapted, not an OS Git-process qualification. Bootstrap/CLI qualify only if their actual cases pass. PostgreSQL12.2 is a compatibility target, not production-version attestation.'
        record['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        (OUT/(args.mode+'-result.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record));return rc


if __name__=='__main__':raise SystemExit(main())
