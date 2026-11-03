"""Concrete bounded transaction actor. No connection/config discovery on import."""
import hashlib,time
from copy import deepcopy
from datetime import datetime,timezone
from . import owned_exit_repair as O

AUDIT='public.jbravo_owned_exit_correction_audit'
CASE_GUARD_BODY="""
DECLARE binding pg_catalog.jsonb;
BEGIN
 IF TG_OP='UPDATE' AND NEW.group_approval_ref IS DISTINCT FROM OLD.group_approval_ref THEN
  RAISE EXCEPTION 'OWNED_EXIT_GROUP_IMMUTABLE';
 END IF;
 IF NEW.trade_id=110 THEN
  IF NEW.group_approval_ref IS NOT NULL THEN RAISE EXCEPTION 'OWNED_EXIT_LEGACY_SCOPE'; END IF;
  RETURN NEW;
 END IF;
 IF NEW.trade_id<=0 OR NEW.manifest->>'classification' IS DISTINCT FROM 'OWNED_EXIT_COHORT_V1'
   OR COALESCE(NEW.manifest->>'case','') NOT IN ('3','6') OR NEW.group_approval_ref IS NULL THEN
  RAISE EXCEPTION 'OWNED_EXIT_SCOPE';
 END IF;
 SELECT allocation_bindings->NEW.allocation_id::text INTO binding
 FROM public.jbravo_owned_exit_correction_audit WHERE approval_ref=NEW.group_approval_ref;
 IF binding IS NULL OR binding->>'trade_id' IS DISTINCT FROM NEW.trade_id::text
   OR binding->'manifest' IS DISTINCT FROM NEW.manifest
   OR NEW.manifest->>'group_approval_ref' IS DISTINCT FROM NEW.group_approval_ref THEN
  RAISE EXCEPTION 'OWNED_EXIT_AUDIT_BINDING';
 END IF;
 RETURN NEW;
END
"""
FUNCTION_SHA=hashlib.sha256(CASE_GUARD_BODY.encode()).hexdigest()
CREATE_AUDIT="""CREATE TABLE public.jbravo_owned_exit_correction_audit (
 approval_ref text PRIMARY KEY,request_digest text NOT NULL,preview_digest text NOT NULL,
 payload jsonb NOT NULL,allocation_bindings jsonb NOT NULL,receipt jsonb NOT NULL,
 policy_revision bigint NOT NULL,committed_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
 CHECK(request_digest~'^[0-9a-f]{64}$' AND preview_digest~'^[0-9a-f]{64}$'),
 CHECK(octet_length(payload::text)<=1048576 AND octet_length(allocation_bindings::text)<=65536 AND octet_length(receipt::text)<=4096))"""
CREATE_GUARD="CREATE FUNCTION public.trade_owned_exit_case_guard() RETURNS pg_catalog.trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $owned$"+CASE_GUARD_BODY+"$owned$"
SCHEMA_SQL=(CREATE_AUDIT,
 "CREATE TRIGGER jbravo_owned_exit_audit_immutable BEFORE UPDATE OR DELETE OR TRUNCATE ON public.jbravo_owned_exit_correction_audit FOR EACH STATEMENT EXECUTE PROCEDURE public.trade_allocation_immutable()",
 "ALTER TABLE public.trade_allocation_cases ADD COLUMN group_approval_ref text",
 "ALTER TABLE public.trade_allocation_cases ADD CONSTRAINT trade_owned_exit_group_fk FOREIGN KEY(group_approval_ref) REFERENCES public.jbravo_owned_exit_correction_audit(approval_ref) DEFERRABLE INITIALLY DEFERRED",
 CREATE_GUARD,
 "CREATE TRIGGER trade_owned_exit_case_guard BEFORE INSERT OR UPDATE ON public.trade_allocation_cases FOR EACH ROW EXECUTE PROCEDURE public.trade_owned_exit_case_guard()")
SCHEMA_ACTOR_SQL="""SELECT policy.relowner=actor.oid,
 current_user=session_user AND actor.rolsuper,
 owner_role.rolname='jbravo_v7_owner',cases.relowner=policy.relowner,
 pg_catalog.has_schema_privilege(current_user,'public','CREATE'),
 pg_catalog.to_regclass('public.jbravo_owned_exit_correction_audit') IS NULL
 FROM pg_catalog.pg_class policy CROSS JOIN pg_catalog.pg_class cases
 CROSS JOIN pg_catalog.pg_roles actor CROSS JOIN pg_catalog.pg_roles owner_role
 WHERE policy.oid='public.trade_disposition_policy'::regclass
 AND cases.oid='public.trade_allocation_cases'::regclass
 AND actor.rolname=current_user AND owner_role.oid=policy.relowner"""
NEW_AUDIT_OWNER_SQL='ALTER TABLE public.jbravo_owned_exit_correction_audit OWNER TO jbravo_v7_owner'
NEW_GUARD_OWNER_SQL='ALTER FUNCTION public.trade_owned_exit_case_guard() OWNER TO jbravo_v7_owner'

# Readable by runtime roles through catalogs; no SELECT on private audit rows.
ATTEST_SQL="""SELECT 'FUNCTION',pg_catalog.json_build_object('body',CASE WHEN octet_length(p.prosrc)<=16384 THEN p.prosrc END,'owner',p.proowner=policy.relowner,'definer',p.prosecdef,'config',p.proconfig=ARRAY['search_path=pg_catalog']::text[],'language',l.lanname,'trigger',p.prorettype='pg_catalog.trigger'::regtype,'args',p.pronargs)::text
 FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace JOIN pg_catalog.pg_language l ON l.oid=p.prolang CROSS JOIN pg_catalog.pg_class policy
 WHERE policy.oid='public.trade_disposition_policy'::regclass AND n.nspname='public' AND p.proname='trade_owned_exit_case_guard'
 UNION ALL SELECT 'AUDIT',pg_catalog.json_build_object('owner',c.relowner=policy.relowner,'ordinary',c.relkind='r' AND NOT c.relrowsecurity AND NOT c.relforcerowsecurity AND NOT EXISTS(SELECT 1 FROM pg_catalog.pg_inherits i WHERE i.inhrelid=c.oid OR i.inhparent=c.oid),'private',(SELECT count(*)=0 FROM pg_catalog.aclexplode(COALESCE(c.relacl,pg_catalog.acldefault('r',c.relowner))) a WHERE a.grantee<>c.relowner),'columns',(SELECT count(*)=8 AND bool_and(a.attnotnull AND (a.attname,a.atttypid) IN (('approval_ref','pg_catalog.text'::regtype),('request_digest','pg_catalog.text'::regtype),('preview_digest','pg_catalog.text'::regtype),('payload','pg_catalog.jsonb'::regtype),('allocation_bindings','pg_catalog.jsonb'::regtype),('receipt','pg_catalog.jsonb'::regtype),('policy_revision','pg_catalog.int8'::regtype),('committed_at','pg_catalog.timestamptz'::regtype))) FROM pg_catalog.pg_attribute a WHERE a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped),'primary_key',(SELECT count(*)=1 FROM pg_catalog.pg_constraint k WHERE k.conrelid=c.oid AND k.contype='p' AND k.convalidated AND k.conkey=ARRAY[(SELECT a.attnum FROM pg_catalog.pg_attribute a WHERE a.attrelid=c.oid AND a.attname='approval_ref' AND NOT a.attisdropped)]::smallint[]))::text
 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace CROSS JOIN pg_catalog.pg_class policy WHERE policy.oid='public.trade_disposition_policy'::regclass AND n.nspname='public' AND c.relname='jbravo_owned_exit_correction_audit'
 UNION ALL SELECT 'TRIGGER',pg_catalog.json_build_object('name',t.tgname,'enabled',t.tgenabled='O' AND t.tgqual IS NULL AND t.tgnargs=0 AND pg_catalog.octet_length(t.tgargs)=0 AND t.tgconstraint=0 AND NOT t.tgisinternal,'bits',t.tgtype,'function',p.proname,'public_function',p.pronamespace='public'::regnamespace)::text FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace JOIN pg_catalog.pg_proc p ON p.oid=t.tgfoid WHERE n.nspname='public' AND ((c.relname='trade_allocation_cases' AND t.tgname='trade_owned_exit_case_guard') OR (c.relname='jbravo_owned_exit_correction_audit' AND t.tgname='jbravo_owned_exit_audit_immutable'))
 UNION ALL SELECT 'FK',pg_catalog.json_build_object('valid',c.convalidated AND c.condeferrable AND c.condeferred AND c.contype='f' AND c.confrelid='public.jbravo_owned_exit_correction_audit'::regclass AND c.conkey=ARRAY[a.attnum]::smallint[] AND c.confkey=ARRAY[(SELECT aa.attnum FROM pg_catalog.pg_attribute aa WHERE aa.attrelid=c.confrelid AND aa.attname='approval_ref' AND NOT aa.attisdropped)]::smallint[] AND c.confupdtype='a' AND c.confdeltype='a' AND a.attname='group_approval_ref')::text FROM pg_catalog.pg_constraint c JOIN pg_catalog.pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=ANY(c.conkey) WHERE c.conrelid='public.trade_allocation_cases'::regclass AND c.conname='trade_owned_exit_group_fk' ORDER BY 1,2 LIMIT 7"""

def attest_extension(cursor):
    D=O.policy_module();cursor.execute(ATTEST_SQL);rows=O.read_rows(cursor,6)
    grouped={}
    for kind,body in rows:grouped.setdefault(kind,[]).append(D.document(body))
    O.need({k:len(v) for k,v in grouped.items()}=={'FUNCTION':1,'AUDIT':1,'TRIGGER':2,'FK':1},'EXTENSION_NOT_INSTALLED')
    fn=grouped['FUNCTION'][0];O.need(type(fn['body']) is str and hashlib.sha256(fn['body'].encode()).hexdigest()==FUNCTION_SHA
        and fn['owner'] and not fn['definer'] and fn['config'] and fn['language']=='plpgsql' and fn['trigger'] and fn['args']==0,'EXTENSION_FUNCTION')
    audit=grouped['AUDIT'][0]
    O.need(set(audit)=={'owner','ordinary','private','columns','primary_key'} and all(v is True for v in audit.values()) and grouped['FK'][0]['valid'] is True,'EXTENSION_SCHEMA_OR_ACL')
    triggers={t['name']:t for t in grouped['TRIGGER']}
    for name,bits,function in [('trade_owned_exit_case_guard',23,'trade_owned_exit_case_guard'),('jbravo_owned_exit_audit_immutable',58,'trade_allocation_immutable')]:
        O.need(triggers[name]['enabled'] is True and triggers[name]['bits']==bits and triggers[name]['function']==function and triggers[name]['public_function'] is True,'EXTENSION_TRIGGER')

class Budget:
    def __init__(self,raw,*,maximum=256,seconds=8,clock=time.monotonic):
        self.raw=raw;self.maximum=maximum;self.clock=clock;self.started=clock();self.end=self.started+seconds
        self.statements=0;self.rows=0;self.bytes=0;self.parameters=0
    def execute(self,sql,params=None):
        D=O.policy_module();O.need(self.clock()<self.end and self.statements+4<=self.maximum,'TRANSACTION_BUDGET')
        # Canonical readers request 1500ms; the actor only tightens it.
        if sql=="SET LOCAL lock_timeout = '1500ms'":sql="SET LOCAL lock_timeout = '500ms'"
        def parameter_wire(v):
            if isinstance(v,datetime):return v.isoformat()
            if type(v) in (tuple,list):return [parameter_wire(x) for x in v]
            if type(v) is dict:return {k:parameter_wire(x) for k,x in v.items()}
            return v
        self.parameters+=len(D.canonical({'parameters':parameter_wire(params or ())}).encode())
        O.need(self.parameters<=8*1048576,'PARAMETER_BUDGET')
        self.statements+=1;self.raw.execute("SET LOCAL statement_timeout = '"+str(max(1,min(5000,int((self.end-self.clock())*1000))))+"ms'")
        self.statements+=1;self.raw.execute(sql,params)
        return self
    def fetchmany(self,n):
        D=O.policy_module();result=self.raw.fetchmany(n)
        def wire(value):
            if isinstance(value,datetime):return value.isoformat()
            if type(value) in (list,tuple):return [wire(v) for v in value]
            return value
        self.rows+=len(result);self.bytes+=len(D.canonical({'rows':wire(result)}).encode())
        O.need(self.rows<=4096 and self.bytes<=16*1048576 and self.clock()<self.end,'RESULT_BUDGET');return result
    def fetchone(self):
        rows=self.fetchmany(1);return rows[0] if rows else None
    def fetchall(self):return self.fetchmany(4097)
    @property
    def rowcount(self):return self.raw.rowcount

def qident(value):
    O.need(type(value) is str and value and '\x00' not in value and len(value)<=256,'SQL_IDENTIFIER')
    return '"'+value.replace('"','""')+'"'

def closed_failure(error):
    if type(error) is O.RepairError:
        return BaseException.args.__get__(error)[0]
    try:code=error.pgcode
    except Exception:code=None
    if type(code) is not str:code=None
    return {'42501':'SQL_PERMISSION_REFUSED','42703':'SQL_COLUMN_UNAVAILABLE','42P01':'SQL_TABLE_UNAVAILABLE',
        '23505':'SQL_UNIQUENESS_CONFLICT','23503':'SQL_REFERENCE_CONFLICT','23514':'SQL_CONSTRAINT_REFUSED',
        '57014':'SQL_QUERY_CANCELED','55P03':'SQL_LOCK_REFUSED','40P01':'SQL_DEADLOCK',
        '25006':'SQL_READONLY_CONTEXT'}.get(code,'APPLY_REFUSED')

def install_in_transaction(cursor,capsule,request):
    """Called only under the apply actor's exclusive policy lock; never commits."""
    catalog=capsule['catalog']
    mode=request.get('execution_context')
    O.need(mode in ('POLICY_OWNER','EXISTING_UNSWITCHED_ADMIN') and type(catalog.get('policy_owner')) is bool
        and all(catalog.get(k) is True for k in ('schema_create','audit_absent')),'PREVIEW_SCHEMA_PERMISSION_REQUIRED')
    admin=mode=='EXISTING_UNSWITCHED_ADMIN'
    O.need(catalog['policy_owner'] is (not admin) and (not admin or request.get('allow_new_object_owner_transfer') is True),'SCHEMA_ACTOR_AUTHORITY')
    constraints=catalog.get('legacy_constraints')
    O.need(type(constraints) is list and len(constraints)<=8,'LEGACY_SCOPE_CONSTRAINT')
    expected=[r for r in constraints if type(r) is list and len(r)==2 and type(r[1]) is str and ''.join(r[1].split())=='CHECK((trade_id=110))']
    O.need(len(expected)==1,'LEGACY_SCOPE_CONSTRAINT');expected=tuple(expected[0])
    cursor.execute(SCHEMA_ACTOR_SQL);actor=cursor.fetchone()
    O.need(type(actor) is tuple and len(actor)==6 and all(type(v) is bool for v in actor)
        and all(actor[2:]) and actor[0] is (not admin) and (not admin or actor[1]),'CURRENT_SCHEMA_ACTOR_OR_STATE')
    cursor.execute('LOCK TABLE public.trade_allocation_cases IN ACCESS EXCLUSIVE MODE')
    cursor.execute("SELECT conname,pg_catalog.pg_get_constraintdef(oid) FROM pg_catalog.pg_constraint WHERE conrelid='public.trade_allocation_cases'::regclass AND contype='c' ORDER BY conname LIMIT 9")
    current=O.read_rows(cursor,8);O.need(current.count(expected)==1,'LEGACY_SCOPE_CONSTRAINT_CHANGED')
    for sql in SCHEMA_SQL:
        cursor.execute(sql)
        if admin and sql==CREATE_AUDIT:cursor.execute(NEW_AUDIT_OWNER_SQL)
        if admin and sql==CREATE_GUARD:cursor.execute(NEW_GUARD_OWNER_SQL)
    cursor.execute('ALTER TABLE public.trade_allocation_cases DROP CONSTRAINT '+qident(expected[0])+' RESTRICT')
    cursor.execute("ALTER TABLE public.trade_allocation_cases ADD CONSTRAINT trade_owned_exit_scope_check CHECK(trade_id>0 AND (trade_id=110 OR manifest->>'classification'='OWNED_EXIT_COHORT_V1'))")
    cursor.execute("SELECT a.grantee,CASE WHEN a.grantee=0 THEN NULL ELSE r.rolname END FROM pg_catalog.pg_class c CROSS JOIN LATERAL pg_catalog.aclexplode(COALESCE(c.relacl,pg_catalog.acldefault('r',c.relowner))) a LEFT JOIN pg_catalog.pg_roles r ON r.oid=a.grantee WHERE c.oid='public.jbravo_owned_exit_correction_audit'::regclass AND a.grantee<>c.relowner GROUP BY a.grantee,r.rolname ORDER BY a.grantee LIMIT 3")
    recipients=O.read_rows(cursor,2);O.need(len([x for x in recipients if x[0]!=0])<=1,'NEW_AUDIT_ACL_SCOPE')
    O.need(not recipients or request.get('allow_new_audit_acl_cleanup') is True,'NEW_AUDIT_ACL_APPROVAL_REQUIRED')
    for oid,name in recipients:cursor.execute('REVOKE ALL PRIVILEGES ON TABLE '+AUDIT+' FROM '+('PUBLIC' if oid==0 else qident(name))+' RESTRICT')
    attest_extension(cursor)
    return dict(execution_context=mode,new_object_owner_transfers=2 if admin else 0,audit_acl_recipients_removed=len(recipients))

UPDATE_CASE1="""UPDATE public.trades t SET exit_order_id=%s,exit_price=%s,exit_time=%s,realized_pnl=%s,updated_at=%s WHERE trade_id=%s AND xmin::text=%s AND pg_catalog.to_jsonb(t)=%s::jsonb RETURNING pg_catalog.to_jsonb(t)::text,xmin::text"""
AUDIT_INSERT="""INSERT INTO public.jbravo_owned_exit_correction_audit(approval_ref,request_digest,preview_digest,payload,allocation_bindings,receipt,policy_revision) VALUES(%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)"""
CONSUMER_ROWS_SQL="""SELECT pg_catalog.json_build_object('row',CASE WHEN pg_catalog.octet_length(pg_catalog.to_jsonb(t)::text)<=24000 THEN pg_catalog.to_jsonb(t) END,'version',t.xmin::text)::text FROM ONLY public.trades t WHERE trade_id=ANY(%s) ORDER BY trade_id LIMIT 17"""

def qualify_consumers(cursor,plan,after,revision,manifests):
    """Real canonical readers in this fresh process, before the only COMMIT.

    This is not proof an already loaded Web worker has refreshed its imports.
    """
    from scripts import trade_disposition as D,gross_paper_accounting as G
    from dashboards import execution_gross_projection as E,execution_read_transport as T
    policy=D.read_policy(cursor,tuple(plan['identity']),exclusive=True,scope='VALIDATED_PERFORMANCE')
    O.need(policy.revision==revision,'CONSUMER_POLICY_REVISION')
    wanted={m['trade_id']:m['allocation_id'] for m in manifests}
    actual={a.trade_id:a.allocation_id for a in policy.allocations}
    O.need(all(actual.get(tid)==aid for tid,aid in wanted.items()),'CONSUMER_ALLOCATION_MISSING')
    ids=sorted({v['row']['trade_id'] for v in plan['cases'].values()}|set(actual))
    records,versions=E.split_versions(E.documents(cursor,CONSUMER_ROWS_SQL,(ids,),16),'trade_id')
    O.need({r['trade_id'] for r in records}==set(ids),'CONSUMER_ROW_POPULATION')
    all_orders=sorted(set(plan['orders'])|{o for a in policy.allocations for o in a.order_ids})
    fills,_=E.split_versions(E.documents(cursor,E.FILLS_SQL,(all_orders,),E.MAX_FILLS),'activity_id')
    # Current native target rows were just compared in full. Canonical readers
    # also checked every reserved cohort's global owners/claims and native rows.
    owners={r['trade_id']:r for r in records}
    for s in plan['owners']:
        row=s['row'];owners.setdefault(row['trade_id'],after if row['trade_id']==plan['cases']['1']['row']['trade_id'] else row)
    gross=G.build_snapshot(policy,records,expected_identity=tuple(plan['identity']),expected_revision=revision)
    execution=E.reconstruct(policy,records,fills,list(owners.values()),versions=dict(trades=versions))
    target_ids={v['row']['trade_id'] for v in plan['cases'].values()}
    O.need(all(tid in execution.gross and tid not in execution.unknown for tid in target_ids)
        and gross['net_eligible'] is False and gross['fees'] is None and gross['net_pnl'] is None,'CONSUMER_QUALIFICATION')
    # Measure the actual PostgreSQL case-row wire encoding, including JSON-text
    # escaping and canonical xmin columns, rather than estimate manifest size.
    case_plan=next(p for p in T.plan_table().values() if p['sql'].startswith('SELECT c.allocation_id::text,c.state,c.allocation_revision,c.request::text,c.manifest::text,'))
    count=case_plan['wire_arity'];slots=','.join('c'+str(i) for i in range(count));values=','.join('q.c'+str(i) for i in range(count))
    sql='SELECT pg_catalog.octet_length(pg_catalog.json_build_array('+values+')::text) FROM ('+case_plan['wire_sql']+') q('+slots+')'
    cursor.execute(sql);sizes=[r[0] for r in O.read_rows(cursor,16)]
    O.need(len(sizes)==len(policy.allocations) and all(type(n) is int and 0<n<=T.BODY_BYTES for n in sizes),'CONSUMER_CASE_WIRE_BOUND')
    orders=[o for m in manifests for o in m['orders']];activities=[s['row']['activity_id'] for m in manifests for s in O.manifest_fills(m)]
    params=dict(trades=sorted(wanted),orders=orders,activities=activities,allocations=[m['allocation_id'] for m in manifests])
    cursor.execute('SELECT pg_catalog.octet_length(pg_catalog.json_build_array(kind,body)::text) FROM ('+O.STATE_SQL+') wire',params)
    state_sizes=[r[0] for r in O.read_rows(cursor,320)]
    O.need(state_sizes and all(type(n) is int and 0<n<=T.BODY_BYTES for n in state_sizes),'CONSUMER_STATE_WIRE_BOUND')
    return dict(status='FRESH_PROCESS_FOUR_CASE_CONSUMERS_QUALIFIED',cases=4,allocation_groups=2,max_case_wire_bytes=max(sizes),max_state_wire_bytes=max(state_sizes))

def afterimage_matches(actual,expected,transaction_time):
    if set(actual)!=set(expected):return False
    if {k:v for k,v in actual.items() if k not in ('updated_at','exit_time')}!={k:v for k,v in expected.items() if k not in ('updated_at','exit_time')}:return False
    C=O.clocks()
    return C.precise_instant(actual['updated_at']).nanoseconds==C.precise_instant(transaction_time).nanoseconds and C.precise_instant(actual['exit_time']).nanoseconds==C.precise_instant(expected['exit_time']).nanoseconds

def apply_correction(conn,capsule,request,*,verify_sources,rebind):
    """Exact audited apply; source/schema/preview authority are prerequisites.

    rebind(cursor,plan,policy) performs fixed full beforeimage/owner/claim/event reads
    under the locks below. It must return a bound bundle, never an authority
    boolean. No provider callback is accepted in this transaction.
    """
    D=O.policy_module();result=dict(status='STOPPED',phase='REQUEST_BINDING',committed=False,commit_uncertain=False,updated_rows=0,attempted_row_updates=0,allocation_rows=0,schema_attempted=False)
    cursor=None;commit_attempted=False
    try:
        O.need(request.get('action')=='APPLY_OWNED_EXITS' and type(request.get('approval_ref')) is str,'APPLY_AUTHORITY')
        D.text_field(request['approval_ref']);D.digest(request['source_manifest_sha256'])
        O.need(verify_sources() is True,'SOURCE_BINDING')
        now=datetime.now(timezone.utc);created=O.clocks().precise_instant(capsule['created_at']).nanoseconds
        expiry=O.clocks().precise_instant(capsule['expires_at']).nanoseconds
        current=O.clocks().precise_instant(now).nanoseconds
        O.need(created<=current<expiry and expiry-created==7*86400*1000000000,'PREVIEW_EXPIRED')
        preview_digest=D.fingerprint({k:v for k,v in capsule.items() if k!='self_sha256'})
        O.need(preview_digest==capsule['self_sha256']==request['preview_digest'],'PREVIEW_BINDING')
        plan=O.compile_plan(capsule['bundle'],now=datetime.fromisoformat(capsule['created_at']))
        O.need(plan==capsule['plan'] and plan['broker_bound'],'PREVIEW_PLAN_CHANGED')
        O.need(conn.get_transaction_status()==0 and not conn.autocommit,'TRANSACTION_NOT_IDLE')
        result['phase']='POLICY_BINDING'
        cursor=Budget(conn.cursor());cursor.execute('SET TRANSACTION ISOLATION LEVEL READ COMMITTED READ WRITE')
        policy=D.read_policy(cursor,tuple(plan['identity']),exclusive=True,scope='VALIDATED_PERFORMANCE')
        O.need(policy.revision==plan['policy_revision'] and O.stable_policy(policy.metadata())==plan['policy_stable_hash'],'POLICY_CAS')
        O.need(not {s['row']['trade_id'] for s in plan['cases'].values()}.intersection(policy.all_excluded_ids),'TARGET_EXCLUDED_IN_ANY_SCOPE')
        result['phase']='CURRENT_COHORT_REBIND'
        cursor.execute('LOCK TABLE public.order_events IN SHARE MODE')
        cursor.execute('SELECT trade_id FROM public.trades WHERE trade_id=ANY(%s) ORDER BY trade_id FOR UPDATE',([x['row']['trade_id'] for x in plan['cases'].values()],))
        O.need(len(O.read_rows(cursor,4))==4,'TARGET_LOCKS')
        current_bundle=rebind(cursor,plan,policy)
        O.need(current_bundle.get('policy_revision')==policy.revision and current_bundle.get('broker_proof') is None,'REBIND_DOMAIN')
        # Schema and financial effects share this one transaction.
        current_bundle['broker_proof']=deepcopy(capsule['bundle']['broker_proof'])
        O.need(O.compile_plan(current_bundle,now=datetime.fromisoformat(capsule['created_at']))==plan,'BEFOREIMAGE_OR_CLAIM_CHANGED')
        result['phase']='SCHEMA_EXTENSION';result['schema_attempted']=True
        schema=install_in_transaction(cursor,capsule,request);result.update(schema)
        result['schema_created']=True
        cursor.execute('SELECT request_digest FROM '+AUDIT+' WHERE approval_ref=%s',(request['approval_ref'],))
        O.need(cursor.fetchone() is None,'APPROVAL_ALREADY_CONSUMED')
        cursor.execute('SELECT approval_ref FROM public.trade_disposition_audit WHERE approval_ref=%s UNION ALL SELECT approval_ref FROM public.trade_allocation_audit WHERE approval_ref=ANY(%s)',(request['approval_ref'],[request['approval_ref']+'/3',request['approval_ref']+'/6']))
        O.need(not O.read_rows(cursor,3),'APPROVAL_REUSED')
        cursor.execute('SELECT pg_catalog.transaction_timestamp()');transaction_time=cursor.fetchone()[0]
        row=plan['cases']['1'];fields=plan['case1_after_fields']
        result['phase']='NATIVE_UPDATE';result['attempted_row_updates']=1
        cursor.execute(UPDATE_CASE1,(fields['exit_order_id'],fields['exit_price'],fields['exit_time'],fields['realized_pnl'],transaction_time,row['row']['trade_id'],row['version'],D.canonical(row['row'])))
        changed=cursor.fetchone();O.need(cursor.rowcount==1 and changed is not None,'CASE1_CAS');after=D.document(changed[0])
        expected=deepcopy(row['row']);expected.update(fields)
        O.need(afterimage_matches(after,expected,transaction_time),'CASE1_AFTERIMAGE')
        result['updated_rows']=1
        manifests=O.allocation_manifests(capsule['bundle'],plan,approval_ref=request['approval_ref'],preview_digest=preview_digest,allocation_ids=capsule['allocation_ids'])
        final_revision=policy.revision+3
        receipt=dict(approval_ref=request['approval_ref'],updated_rows=1,unchanged_native_rows=3,allocation_rows=2,policy_revision=final_revision,financial_eligible=False,fees=None,net=None)
        bindings={m['allocation_id']:dict(trade_id=m['trade_id'],manifest=m) for m in manifests}
        O.snapshot(dict(row=after,version=changed[1]))
        payload=dict(request=request,before_rows={k:v['row'] for k,v in plan['cases'].items()},
            before_versions={k:v['version'] for k,v in plan['cases'].items()},
            after_rows={k:(after if k=='1' else v['row']) for k,v in plan['cases'].items()},
            after_versions={k:(changed[1] if k=='1' else v['version']) for k,v in plan['cases'].items()},
            owners_before=plan['owners'],policy_before=dict(identity=plan['identity'],revision=plan['policy_revision'],stable_fingerprint=plan['policy_stable_hash']),
            fills=plan['fills'],events=plan['events'],broker_proof=capsule['bundle']['broker_proof'],
            preview_source_pins=plan['source_pins'],apply_source_manifest_sha256=request['source_manifest_sha256'],
            schema_execution_context=schema['execution_context'],new_object_owner_transfers=schema['new_object_owner_transfers'])
        O.need(len(D.canonical(payload).encode())<=1048576,'AUDIT_BOUND')
        result['phase']='AUDIT_AND_ALLOCATIONS'
        cursor.execute(AUDIT_INSERT,(request['approval_ref'],D.fingerprint(request),preview_digest,D.canonical(payload),D.canonical(bindings),D.canonical(receipt),final_revision))
        revision=policy.revision
        cursor.execute('UPDATE public.trade_disposition_policy SET policy_revision=policy_revision+1 WHERE singleton=1 AND policy_revision=%s',(revision,));O.need(cursor.rowcount==1,'POLICY_CAS');revision+=1
        for manifest in manifests:
            aid=manifest['allocation_id'];subref=request['approval_ref']+'/'+str(manifest['case'])
            req=dict(action='APPLY',allocation_id=aid,trade_id=manifest['trade_id'],identity=plan['identity'],manifest_sha256=D.fingerprint(manifest),expected_allocation_revision=0,expected_policy_revision=revision,approval_ref=subref)
            cursor.execute('INSERT INTO public.trade_allocation_cases(allocation_id,trade_id,state,allocation_revision,request,manifest,group_approval_ref) VALUES(%s,%s,\'ACTIVE\',1,%s::jsonb,%s::jsonb,%s)',(aid,manifest['trade_id'],D.canonical(req),D.canonical(manifest),request['approval_ref']))
            for order in manifest['orders']:cursor.execute('INSERT INTO public.trade_allocation_order_claims(order_id,allocation_id) VALUES(%s,%s)',(order,aid))
            for item in O.manifest_fills(manifest):
                f=item['row'];cursor.execute("INSERT INTO public.trade_allocation_fill_claims(activity_id,activity_type,order_id,allocation_id,quantity) VALUES(%s,'FILL',%s,%s,%s)",(f['activity_id'],f['order_id'],aid,f['qty']))
            cursor.execute('UPDATE public.trade_disposition_policy SET policy_revision=policy_revision+1 WHERE singleton=1 AND policy_revision=%s',(revision,));O.need(cursor.rowcount==1,'POLICY_CAS');revision+=1
            subreceipt=dict(allocation_id=aid,trade_id=manifest['trade_id'],state='ACTIVE',allocation_revision=1,policy_revision=revision,approval_ref=subref)
            cursor.execute('INSERT INTO public.trade_allocation_audit(approval_ref,allocation_id,request_digest,request,manifest,result,policy_revision) VALUES(%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)',(subref,aid,D.fingerprint(req),D.canonical(req),D.canonical(manifest),D.canonical(subreceipt),revision))
            result['allocation_rows']+=1
        O.need(revision==final_revision,'FINAL_REVISION')
        result['phase']='EXACT_READBACK'
        cursor.execute('SELECT request_digest,preview_digest,payload::text,allocation_bindings::text,receipt::text,policy_revision FROM '+AUDIT+' WHERE approval_ref=%s',(request['approval_ref'],))
        saved=cursor.fetchone()
        O.need(saved is not None and saved[:2]==(D.fingerprint(request),preview_digest)
            and D.document(saved[2])==payload and D.document(saved[3])==bindings
            and D.document(saved[4])==receipt and saved[5]==final_revision,'AUDIT_READBACK')
        cursor.execute('SELECT trade_id,pg_catalog.to_jsonb(t)::text FROM public.trades t WHERE trade_id=ANY(%s) ORDER BY trade_id',([x['row']['trade_id'] for x in plan['cases'].values()],))
        observed={tid:D.document(body) for tid,body in O.read_rows(cursor,4)}
        O.need(observed=={v['row']['trade_id']:(after if k=='1' else v['row']) for k,v in plan['cases'].items()},'FINAL_NATIVE_READBACK')
        cursor.execute('SELECT allocation_id::text,manifest::text,group_approval_ref FROM public.trade_allocation_cases WHERE allocation_id=ANY(%s::uuid[]) ORDER BY allocation_id',(list(bindings),))
        allocated={aid:(D.document(body),group) for aid,body,group in O.read_rows(cursor,2)}
        O.need(allocated=={aid:(b['manifest'],request['approval_ref']) for aid,b in bindings.items()},'ALLOCATION_READBACK')
        # Reuse the reader's whole-set claim checks, including global collisions.
        all_orders=[o for m in manifests for o in m['orders']];all_ids=[s['row']['activity_id'] for m in manifests for s in O.manifest_fills(m)]
        cursor.execute(O.CLAIMS_SQL,(list(bindings),all_ids,all_orders,list(bindings),all_orders))
        actual_claims=O.read_rows(cursor,64);expected_claims=set()
        for m in manifests:
            for order in m['orders']:expected_claims.add(('ORDER',order,m['allocation_id'],None,None,None))
            for s in O.manifest_fills(m):
                f=s['row'];expected_claims.add(('FILL',f['activity_id'],m['allocation_id'],f['order_id'],'FILL',O.number(f['qty'])))
        normalized=[]
        for kind,body in actual_claims:
            c=D.document(body)
            normalized.append((kind,c['order_id'] if kind=='ORDER' else c['activity_id'],c['allocation_id'],None if kind=='ORDER' else c['order_id'],None if kind=='ORDER' else c['activity_type'],None if kind=='ORDER' else O.number(c['quantity'])))
        O.need(len(normalized)==len(set(normalized)) and set(normalized)==expected_claims,'CLAIM_READBACK')
        cursor.execute('SELECT policy_revision FROM public.trade_disposition_policy WHERE singleton=1')
        O.need(cursor.fetchone()==(final_revision,),'POLICY_READBACK')
        # Deferred FKs are forced before COMMIT so failures remain precommit.
        cursor.execute('SET CONSTRAINTS ALL IMMEDIATE')
        result['phase']='CONSUMER_QUALIFICATION'
        result['consumer_qualification']=qualify_consumers(cursor,plan,after,final_revision,manifests)
        O.need(verify_sources() is True,'SOURCE_CHANGED_BEFORE_COMMIT')
        O.need(cursor.statements+3<=cursor.maximum and cursor.clock()<cursor.end,'TRANSACTION_BUDGET')
        result['phase']='COMMIT';commit_attempted=True;conn.commit();result.update(status='COMMITTED',phase='COMPLETE',committed=True,receipt=receipt)
    except Exception as error:
        reason=closed_failure(error)
        result.update(reason=reason if not commit_attempted else 'COMMIT_OUTCOME_UNCERTAIN',commit_uncertain=commit_attempted)
    finally:
        try:conn.rollback();result['rollback_acknowledged']=True
        except Exception:result['rollback_acknowledged']=False
        try:conn.close();result['connection_closed']=bool(conn.closed)
        except Exception:result['connection_closed']=False
        result['committed_schema_created']=True if result['committed'] else None if result['commit_uncertain'] else False
        result['committed_updated_rows']=result['updated_rows'] if result['committed'] else None if result['commit_uncertain'] else 0
        result['committed_allocation_rows']=result['allocation_rows'] if result['committed'] else None if result['commit_uncertain'] else 0
        result['allocation_adopted']=True if result['committed'] else None if result['commit_uncertain'] else False
        result.update(financial_eligible=False,fees=None,net=None)
        if result['commit_uncertain']:result['status']='COMMIT_UNCERTAIN'
        elif not result['committed'] and result.get('rollback_acknowledged'):result['status']='ROLLED_BACK'
        if cursor:result['usage']=dict(statements=cursor.statements,total_bound=cursor.statements+2,rows=cursor.rows,bytes=cursor.bytes,parameter_bytes=cursor.parameters,elapsed_seconds=round(cursor.clock()-cursor.started,6))
    return result

RECOVERY_SQL="""SELECT a.request_digest,a.preview_digest,
 CASE WHEN pg_catalog.octet_length(a.payload::text)<=1048576 THEN a.payload::text END,
 CASE WHEN pg_catalog.octet_length(a.allocation_bindings::text)<=65536 THEN a.allocation_bindings::text END,
 CASE WHEN pg_catalog.octet_length(a.receipt::text)<=4096 THEN a.receipt::text END,a.policy_revision,
 p.deployment_id,p.database_id,p.database_name=pg_catalog.current_database()
 FROM public.jbravo_owned_exit_correction_audit a CROSS JOIN public.trade_disposition_policy p
 WHERE p.singleton=1 AND a.approval_ref=%s LIMIT 2"""

def reconcile_commit(conn,capsule,request,*,verify_sources):
    """Separate, newly authorized read only. Never called automatically on error.

    An absent audit in one snapshot is not authority to replay a transaction.
    A matching immutable audit confirms commit even if later native rows differ.
    """
    D=O.policy_module();result=dict(status='STOPPED',committed=None,replay_authorized=False);cursor=None
    try:
        O.need(request.get('action')=='APPLY_OWNED_EXITS' and verify_sources() is True,'RECOVERY_BINDING')
        digest=D.fingerprint({k:v for k,v in capsule.items() if k!='self_sha256'})
        O.need(digest==capsule['self_sha256']==request['preview_digest'],'PREVIEW_BINDING')
        O.need(conn.get_transaction_status()==0 and not conn.autocommit,'TRANSACTION_NOT_IDLE')
        cursor=Budget(conn.cursor(),maximum=32,seconds=8)
        cursor.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        cursor.execute("SELECT pg_catalog.to_regclass('public.jbravo_owned_exit_correction_audit') IS NOT NULL")
        exists=cursor.fetchone();O.need(exists in ((True,),(False,)),'RECOVERY_SCHEMA_STATE')
        rows=[]
        if exists==(True,):
            from . import trade_allocation as A
            A.attest_contract(cursor)
            attest_extension(cursor)
            cursor.execute(RECOVERY_SQL,(request['approval_ref'],));rows=O.read_rows(cursor,1)
        if not rows:result.update(status='AUDIT_ABSENT_CURRENT_SNAPSHOT_NO_REPLAY')
        else:
            row=rows[0];O.need(row[0]==D.fingerprint(request) and row[1]==digest and list(row[6:8])==capsule['plan']['identity'] and row[8] is True,'RECOVERY_AUDIT_CONFLICT')
            payload,bindings,receipt=(D.document(row[i]) for i in (2,3,4))
            plan=capsule['plan']
            O.need(payload.get('request')==request and payload.get('before_rows')=={k:v['row'] for k,v in plan['cases'].items()}
                and payload.get('fills')==plan['fills'] and payload.get('events')==plan['events']
                and payload.get('broker_proof')==capsule['bundle']['broker_proof'] and payload.get('preview_source_pins')==plan['source_pins']
                and payload.get('apply_source_manifest_sha256')==request['source_manifest_sha256']
                and payload.get('schema_execution_context')==request['execution_context']
                and payload.get('new_object_owner_transfers')==(2 if request['execution_context']=='EXISTING_UNSWITCHED_ADMIN' else 0)
                and payload.get('before_versions')=={k:v['version'] for k,v in plan['cases'].items()},'RECOVERY_PAYLOAD')
            manifests=O.allocation_manifests(capsule['bundle'],plan,approval_ref=request['approval_ref'],preview_digest=digest,allocation_ids=capsule['allocation_ids'])
            O.need(bindings=={m['allocation_id']:dict(trade_id=m['trade_id'],manifest=m) for m in manifests}
                and receipt==dict(approval_ref=request['approval_ref'],updated_rows=1,unchanged_native_rows=3,allocation_rows=2,policy_revision=row[5],financial_eligible=False,fees=None,net=None),'RECOVERY_RECEIPT')
            expected=payload['after_rows'];case1=deepcopy(plan['cases']['1']['row']);case1.update(plan['case1_after_fields'])
            O.need(afterimage_matches(expected['1'],case1,expected['1']['updated_at'])
                and all(expected[c]==plan['cases'][c]['row'] for c in ('2','3','6')),'RECOVERY_AFTERIMAGE')
            cursor.execute('SELECT trade_id,pg_catalog.to_jsonb(t)::text FROM public.trades t WHERE trade_id=ANY(%s) ORDER BY trade_id',([v['trade_id'] for v in expected.values()],))
            current={tid:D.document(body) for tid,body in O.read_rows(cursor,4)}
            same=current=={v['trade_id']:v for v in expected.values()}
            result.update(status='COMMIT_CONFIRMED_CURRENT_ROWS_MATCH' if same else 'COMMIT_CONFIRMED_CURRENT_ROWS_CHANGED',committed=True,current_native_rows_match=same)
    except Exception:
        result.update(reason='RECOVERY_REFUSED')
    finally:
        try:conn.rollback();result['rollback_acknowledged']=True
        except Exception:result['rollback_acknowledged']=False
        try:conn.close();result['connection_closed']=bool(conn.closed)
        except Exception:result['connection_closed']=False
        if cursor:result['usage']=dict(statements=cursor.statements,total_bound=cursor.statements+2,rows=cursor.rows,bytes=cursor.bytes,parameter_bytes=cursor.parameters,elapsed_seconds=round(cursor.clock()-cursor.started,6))
    return result
