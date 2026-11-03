"""Approval-bound owned-exit planning/reading. No discovery or I/O on import."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from decimal import Decimal,Context,localcontext
from uuid import UUID

CLASSIFICATION='OWNED_EXIT_COHORT_V1'
PREVIEW_SCHEMA='JBRAVO_OWNED_EXIT_PREVIEW_V1'
CASES=(1,2,3,6)
QUANTITIES={1:Decimal(2),2:Decimal(4),3:Decimal(11),6:Decimal(53)}
DIRECT={1:Decimal(4),2:Decimal(4),3:Decimal(3),6:Decimal(28)}
CHILD={1:Decimal(2),3:Decimal(8),6:Decimal(25)}
FILL_COUNTS={1:2,3:1,6:1}
PRIVATE_MAX=1048576
DENIED_KEYS=frozenset({'account_id','account_number','api_key','api_secret','password','secret','authorization','access_token','refresh_token'})

class RepairError(RuntimeError):pass
def need(ok,code):
    if not ok:raise RepairError('OWNED_EXIT_'+code)
def policy_module():
    from scripts import trade_disposition
    return trade_disposition
def clocks():
    from scripts import owned_exit_clock
    return owned_exit_clock
def number(value):
    need(type(value) in (str,int,Decimal),'NUMBER_TYPE')
    try:value=Decimal(value)
    except (ValueError,ArithmeticError):raise RepairError('OWNED_EXIT_NUMBER_INVALID') from None
    need(value.is_finite() and len(value.as_tuple().digits)<=40 and abs(value.as_tuple().exponent)<=18,'NUMBER_INVALID')
    return value
def identifier(value):
    need(type(value) is str and str(UUID(value))==value,'ORDER_ID_INVALID');return value
def snapshot(value):
    D=policy_module();need(type(value) is dict and set(value)=={'row','version'},'SNAPSHOT_SHAPE')
    need(type(value['version']) is str and value['version'].isdigit() and len(value['version'])<=20,'XMIN_INVALID')
    row=D.document(value['row']);need(len(D.canonical(row).encode())<=32768,'ROW_OVERSIZE')
    return dict(row=row,version=value['version'])
def safe_private(value,depth=0):
    need(depth<=20,'PRIVATE_DEPTH')
    if type(value) is dict:
        need(not {str(k).lower() for k in value}.intersection(DENIED_KEYS),'UNNECESSARY_PRIVATE_FIELD')
        for nested in value.values():safe_private(nested,depth+1)
    elif type(value) in (list,tuple):
        for nested in value:safe_private(nested,depth+1)
    elif type(value) is float:need(False,'NATIVE_FLOAT_FORBIDDEN')
def stable_policy(metadata):
    D=policy_module();value=D.document(D.canonical(metadata));value.pop('policy_revision',None)
    overlay=value.get('allocation_overlay',{});overlay.pop('policy_revision',None);overlay.pop('binding_fingerprint',None)
    return D.fingerprint(value)

def event_context(event):
    row=snapshot(event)['row'];raw=row.get('raw');need(type(raw) is dict,'EVENT_RAW')
    nested=raw.get('raw');need(type(nested) is dict,'EVENT_RAW')
    return dict(event_id=row.get('event_id'),event_type=row.get('event_type'),symbol=row.get('symbol'),qty=row.get('qty'),
        order_id=row.get('order_id'),event_time=row.get('event_time'),parent_order_id=nested.get('parent_order_id'),
        envelope_order_id=raw.get('order_id'),envelope_event_type=raw.get('event_type'),metadata_event_type=nested.get('event_type'),
        top_parent_key_present='parent_order_id' in raw,top_parent_order_id=raw.get('parent_order_id'),
        recorded_submitted_at=nested.get('submitted_at'),envelope_event_time_key_present='event_time' in raw,
        envelope_event_time_json_type='string' if type(raw.get('event_time')) is str else 'null' if raw.get('event_time') is None else 'unsupported',
        envelope_event_time=raw.get('event_time'),raw_shape_valid=type(nested.get('parent_order_id')) is str
            and type(raw.get('order_id')) is str and type(raw.get('event_type')) is str and type(nested.get('event_type')) is str)

def fill_fact(item):
    C=clocks();row=snapshot(item)['row'];raw=row.get('raw')
    need(type(raw) is dict and row.get('activity_type')=='FILL' and raw.get('activity_type')=='FILL','FILL_KIND')
    for column,key in [('activity_id','id'),('order_id','order_id'),('symbol','symbol'),('side','side')]:
        need(type(row.get(column)) is str and row[column]==raw.get(key),'FILL_IDENTITY')
    identifier(row['order_id']);need(row['side'] in ('buy','sell'),'FILL_SIDE')
    q,p=number(row.get('qty')),number(row.get('price'))
    need(q>0 and p>0 and q==number(raw.get('qty')) and p==number(raw.get('price')),'FILL_ECONOMICS')
    stored,original=C.precise_instant(row.get('transaction_time')),C.precise_instant(raw.get('transaction_time'))
    need(C.projection_relation(stored,original) in ('EXACT_INSTANT_MATCH','MICROSECOND_TRUNCATION_COMPATIBLE_NOT_EQUAL'),'FILL_CLOCK')
    return dict(activity_id=row['activity_id'],order_id=row['order_id'],symbol=row['symbol'],side=row['side'],
        qty=q,price=p,time=raw['transaction_time'],stored_time=row['transaction_time'],instant=original.nanoseconds)

def _sum(facts,side=None):return sum((f['qty'] for f in facts if side is None or f['side']==side),Decimal(0))
def _value(facts,side):return sum((f['qty']*f['price'] for f in facts if f['side']==side),Decimal(0))

def validate_broker(proof,children,by_order,expected_cases=None):
    C=clocks();need(type(proof) is dict and set(proof)=={'paper_identity_verified','method','orders','observed_at'},'BROKER_PROOF_SHAPE')
    need(proof['paper_identity_verified'] is True and proof['method']=='INDEPENDENT_ACCOUNT_NUMBER_EXACT_MATCH','PAPER_IDENTITY')
    expected=set(CHILD) if expected_cases is None else set(expected_cases)
    need(expected and expected.issubset(CHILD),'BROKER_SCOPE')
    C.precise_instant(proof['observed_at']);need(type(proof['orders']) is list and len(proof['orders'])==len(expected),'BROKER_SCOPE')
    seen=set()
    for row in proof['orders']:
        need(type(row) is dict and set(row)=={'case','order_id','symbol','side','qty','status','filled_qty','fills','complete'},'BROKER_ORDER_SHAPE')
        case=row['case'];need(case in expected and case not in seen and row['order_id']==children[case],'BROKER_ORDER_ID')
        need(row['side']=='sell' and row['symbol']==by_order[children[case]][0]['symbol']
             and number(row['qty'])==QUANTITIES[case],'BROKER_SUBMISSION_CONTEXT')
        need(row['status']==('filled' if case==1 else 'canceled') and row['complete'] is True,'BROKER_TERMINAL_STATUS')
        need(number(row['filled_qty'])==CHILD[case] and type(row['fills']) is list and len(row['fills'])==FILL_COUNTS[case],'BROKER_FILL_COUNT')
        expected_fills={f['activity_id']:f for f in by_order[children[case]]};actual={}
        for fill in row['fills']:
            need(type(fill) is dict and set(fill)=={'id','activity_type','order_id','symbol','side','qty','price','transaction_time'},'BROKER_FILL_SHAPE')
            key=fill['id'];need(key in expected_fills and key not in actual,'BROKER_FILL_ID');f=expected_fills[key]
            need(fill['activity_type']=='FILL' and fill['order_id']==f['order_id'] and fill['symbol']==f['symbol']
                and fill['side']==f['side'] and number(fill['qty'])==f['qty'] and number(fill['price'])==f['price']
                and C.precise_instant(fill['transaction_time']).nanoseconds==f['instant'],'BROKER_FILL_FACT')
            actual[key]=fill
        need(set(actual)==set(expected_fills),'BROKER_FILL_SCOPE');seen.add(case)
    need(seen==expected,'BROKER_SCOPE')

def compile_plan(bundle,*,now=None,allow_unbound=False):
    """Bind complete private beforeimages; never print or discover inputs."""
    with localcontext(Context(prec=160)):
        return _compile_plan(bundle,now=now,allow_unbound=allow_unbound)

def _compile_plan(bundle,*,now=None,allow_unbound=False):
    D=policy_module();C=clocks();now=now or datetime.now(timezone.utc);safe_private(bundle)
    need(type(bundle) is dict and bundle.get('schema')==PREVIEW_SCHEMA,'BUNDLE_SCHEMA')
    D.identity(*bundle['identity']);D.revision(bundle['policy_revision']);D.digest(bundle['policy_stable_hash'])
    need(type(bundle.get('source_pins')) is dict and bundle['source_pins'],'SOURCE_PINS')
    for path,digest in bundle['source_pins'].items():D.text_field(path);D.digest(digest)
    need(type(bundle.get('cases')) is list and [x.get('case') for x in bundle['cases']]==list(CASES),'CASE_SCOPE')
    cases={x['case']:snapshot(x['snapshot']) for x in bundle['cases']}
    ids={case:item['row']['trade_id'] for case,item in cases.items()};need(len(set(ids.values()))==4,'TRADE_ID_COLLISION')
    for case,item in cases.items():
        row=item['row'];need(type(row.get('trade_id')) is int and row['trade_id']>0 and row.get('status')=='CLOSED','TRADE_STATE')
        need(number(row.get('qty'))==QUANTITIES[case],'TRADE_QUANTITY');identifier(row.get('entry_order_id'));identifier(row.get('exit_order_id'))
        C.precise_instant(row.get('entry_time'));C.precise_instant(row.get('exit_time'))
    need(cases[1]['row']['exit_order_id']==cases[2]['row']['exit_order_id'],'SHARED_PAIR_CHANGED')
    need(type(bundle.get('events')) is list and len(bundle['events'])==3,'EVENT_SCOPE')
    events={};children={};clock_reports={}
    for item in bundle['events']:
        case=item['case'];need(case in CHILD and case not in events,'EVENT_SCOPE')
        snap=snapshot(item['snapshot']);event=event_context(snap)
        result=C.trail_context(event,cases[case]['row'],observed_before=now)
        children[case]=identifier(result['child']);events[case]=snap;clock_reports[case]={k:result[k] for k in ('clock_reason','projection_accepted_as_encoding','db_and_original_same_instant')}
    need(set(events)==set(CHILD) and len(set(children.values()))==3,'CHILD_COLLISION')
    need(type(bundle.get('fills')) is list and 1<=len(bundle['fills'])<=128,'FILL_BOUND')
    facts=[];seen=set();by_order={}
    for item in bundle['fills']:
        f=fill_fact(item);key=f['activity_id'];need(key not in seen,'FILL_DUPLICATE');seen.add(key);facts.append(f);by_order.setdefault(f['order_id'],[]).append(f)
    orders={case:[cases[case]['row']['entry_order_id']]+([children[case]] if case==1 else [cases[case]['row']['exit_order_id']]+([children[case]] if case in (3,6) else [])) for case in CASES}
    all_orders=[o for group in orders.values() for o in group]
    need(len(all_orders)==len(set(all_orders))==10,'ORDER_REUSED')
    need(set(by_order)==set(all_orders),'FILL_ORDER_SCOPE')
    for case in CASES:
        row=cases[case]['row'];cohort=[f for o in orders[case] for f in by_order[o]]
        need(all(f['symbol']==row['symbol'] and f['side']==('buy' if f['order_id']==row['entry_order_id'] else 'sell') for f in cohort),'FILL_LOT_CONTEXT')
        need(_sum(cohort,'buy')==_sum(cohort,'sell')==QUANTITIES[case],'CONSERVATION')
        need(max(f['instant'] for f in cohort if f['side']=='buy')<=min(f['instant'] for f in cohort if f['side']=='sell'),'FILL_CHRONOLOGY')
        if case in CHILD:
            need(_sum(by_order[children[case]])==CHILD[case] and len(by_order[children[case]])==FILL_COUNTS[case],'CHILD_FILL_SCOPE')
            submission=C.precise_instant(event_context(events[case])['recorded_submitted_at']).nanoseconds
            need(all(f['instant']>=submission for f in by_order[children[case]]),'CHILD_FILL_PRECEDES_SUBMISSION')
        if case in (2,3,6):need(_sum(by_order[row['exit_order_id']])==DIRECT[case],'DIRECT_FILL_SCOPE')
    need(type(bundle.get('owners')) is list and len(bundle['owners'])<=64,'OWNER_BOUND')
    observed={snapshot(x)['row']['trade_id']:snapshot(x)['row'] for x in bundle['owners']}
    need(len(observed)==len(bundle['owners']),'OWNER_DUPLICATE')
    need(set(ids.values()).issubset(observed),'OWNER_TARGET_MISSING')
    for case in CASES:need(observed[ids[case]]==cases[case]['row'],'OWNER_TARGET_CHANGED')
    observed[ids[1]]=dict(observed[ids[1]],exit_order_id=children[1])
    expected_owners={order:ids[case] for case,group in orders.items() for order in group}
    for trade_id,row in observed.items():
        for key in ('entry_order_id','exit_order_id'):
            order=row.get(key)
            if order in expected_owners:need(trade_id==expected_owners[order],'GLOBAL_OWNER_CONFLICT')
    reserved=bundle.get('reserved_order_owners');need(type(reserved) is dict,'RESERVATIONS_REQUIRED')
    need(not set(all_orders).intersection(reserved),'ORDER_RESERVED')
    need(bundle.get('relevant_fill_claims')==[] and bundle.get('relevant_order_claims')==[],'FILL_OR_ORDER_ALREADY_CLAIMED')
    need(not set(ids.values()).intersection(bundle.get('excluded_trade_ids',[])),'TRADE_EXCLUDED')
    proof=bundle.get('broker_proof');ready=proof is not None
    if ready:
        validate_broker(proof,children,by_order)
        age=C.precise_instant(now).nanoseconds-C.precise_instant(proof['observed_at']).nanoseconds
        need(0<=age<=600*1000000000,'BROKER_PROOF_AGE')
    else:need(allow_unbound,'BROKER_CONTINUITY_REQUIRED')
    incoming=by_order[cases[1]['row']['entry_order_id']];outgoing=by_order[children[1]]
    need(_value(incoming,'buy')/QUANTITIES[1]==number(cases[1]['row']['entry_price']),'CASE1_ENTRY_PRICE_UNSUPPORTED')
    completion=max(outgoing,key=lambda f:f['instant'])['stored_time']
    after=dict(exit_order_id=children[1],exit_price=_value(outgoing,'sell')/QUANTITIES[1],exit_time=completion,
        realized_pnl=_value(outgoing,'sell')-_value(incoming,'buy'))
    allocations=[]
    for case in (3,6):
        cohort=[f for order in orders[case] for f in by_order[order]]
        selected=[deepcopy(item) for item in bundle['fills'] if item['row']['order_id'] in orders[case]]
        allocations.append(dict(case=case,trade_id=ids[case],classification=CLASSIFICATION,row_snapshot=cases[case],
            fills=selected,orders=orders[case],activity_ids=sorted(f['activity_id'] for f in cohort),
            entry_quantity=QUANTITIES[case],exit_quantity=QUANTITIES[case],gross_result=_value(cohort,'sell')-_value(cohort,'buy'),
            financial_eligible=False,fees=None,net=None))
    plan=dict(schema=PREVIEW_SCHEMA,identity=bundle['identity'],policy_revision=bundle['policy_revision'],
        policy_stable_hash=bundle['policy_stable_hash'],cases={str(k):v for k,v in cases.items()},case1_after_fields=after,allocations=allocations,
        events={str(k):v for k,v in events.items()},fills=deepcopy(bundle['fills']),owners=deepcopy(bundle['owners']),orders=all_orders,
        clock_reports={str(k):v for k,v in clock_reports.items()},broker_bound=ready,financial_eligible=False,fees=None,net=None,
        source_pins=deepcopy(bundle['source_pins']))
    safe_private(plan);need(len(D.canonical(plan).encode())<=PRIVATE_MAX,'PRIVATE_PREVIEW_BOUND')
    return plan

MANIFEST_MAX=20480
def allocation_manifests(bundle,plan,*,approval_ref,preview_digest,allocation_ids):
    D=policy_module();need(plan['broker_bound'],'BROKER_CONTINUITY_REQUIRED');D.digest(preview_digest)
    result=[]
    for item in plan['allocations']:
        case=item['case'];identifier(allocation_ids[str(case)])
        proof=deepcopy(bundle['broker_proof']);proof['orders']=[p for p in proof['orders'] if p['case']==case]
        manifest=dict(classification=CLASSIFICATION,case=case,trade_id=item['trade_id'],allocation_id=allocation_ids[str(case)],
            identity=plan['identity'],group_approval_ref=approval_ref,preview_digest=preview_digest,
            row_snapshot=item['row_snapshot'],fills=[dict(snapshot=s['row'],row_fingerprint=D.fingerprint(s['row']),row_version=s['version']) for s in item['fills']],orders=item['orders'],event=plan['events'][str(case)],
            broker_proof=proof,source_pins=plan['source_pins'],financial_eligible=False,fees=None,net=None)
        need(len(D.canonical(manifest).encode())<=MANIFEST_MAX,'RUNTIME_MANIFEST_BOUND')
        validate_group(manifest,plan['identity']);result.append(manifest)
    return result

def manifest_fills(manifest):
    """Match the installed immutable ingestion guard's legacy snapshot envelope."""
    D=policy_module();items=manifest.get('fills');need(type(items) is list and 1<=len(items)<=128,'FILL_BOUND');result=[]
    for item in items:
        need(type(item) is dict and set(item)=={'snapshot','row_fingerprint','row_version'},'FILL_SNAPSHOT_ENVELOPE')
        D.digest(item['row_fingerprint']);row=D.document(item['snapshot'])
        need(D.fingerprint(row)==item['row_fingerprint'],'FILL_SNAPSHOT_FINGERPRINT')
        result.append(snapshot(dict(row=row,version=item['row_version'])))
    return result

def validate_group(manifest,identity):
    with localcontext(Context(prec=160)):
        return _validate_group(manifest,identity)

def _validate_group(manifest,identity):
    D=policy_module();C=clocks();safe_private(manifest)
    need(type(manifest) is dict and manifest.get('classification')==CLASSIFICATION and manifest.get('case') in (3,6),'ALLOCATION_CLASS')
    need(manifest.get('identity')==list(identity) and manifest.get('financial_eligible') is False
         and manifest.get('fees') is None and manifest.get('net') is None,'ALLOCATION_SCOPE')
    identifier(manifest['allocation_id']);D.digest(manifest['preview_digest']);D.text_field(manifest['group_approval_ref'])
    need(len(D.canonical(manifest).encode())<=MANIFEST_MAX,'RUNTIME_MANIFEST_BOUND')
    row=snapshot(manifest['row_snapshot'])['row'];case=manifest['case']
    need(row['trade_id']==manifest['trade_id'] and row['status']=='CLOSED' and number(row['qty'])==QUANTITIES[case],'ALLOCATION_ROW')
    event=event_context(manifest['event']);link=C.trail_context(event,row,observed_before=manifest['broker_proof']['observed_at'])
    orders=[row['entry_order_id'],row['exit_order_id'],link['child']]
    need(manifest['orders']==orders and len(set(orders))==3,'ALLOCATION_ORDERS')
    facts=[];seen=set();by_order={}
    normalized_fills=manifest_fills(manifest)
    for item in normalized_fills:
        fact=fill_fact(item);need(fact['activity_id'] not in seen,'FILL_DUPLICATE');seen.add(fact['activity_id'])
        need(fact['order_id'] in orders and fact['symbol']==row['symbol']
             and fact['side']==('buy' if fact['order_id']==orders[0] else 'sell'),'FILL_LOT_CONTEXT')
        facts.append(fact);by_order.setdefault(fact['order_id'],[]).append(fact)
    need(set(by_order)==set(orders) and _sum(facts,'buy')==_sum(facts,'sell')==QUANTITIES[case],'CONSERVATION')
    need(_sum(by_order[orders[1]])==DIRECT[case] and _sum(by_order[orders[2]])==CHILD[case],'ALLOCATION_COMPONENTS')
    submission=C.precise_instant(event['recorded_submitted_at']).nanoseconds
    need(all(f['instant']>=submission for f in by_order[orders[2]]),'CHILD_FILL_PRECEDES_SUBMISSION')
    need(max(f['instant'] for f in facts if f['side']=='buy')<=min(f['instant'] for f in facts if f['side']=='sell'),'FILL_CHRONOLOGY')
    validate_broker(manifest['broker_proof'],{case:orders[2]},by_order,(case,))
    with localcontext(Context(prec=160)):gross=_value(facts,'sell')-_value(facts,'buy')
    clock_fills=tuple((s['row']['activity_pk'],s['row']['activity_id'],s['row']['side'],s['row']['order_id'],
        str(number(s['row']['qty'])),str(number(s['row']['price'])),s['row']['transaction_time']) for s in normalized_fills)
    return dict(orders=orders,activity_ids=sorted(seen),quantity=QUANTITIES[case],gross=gross,clock_fills=clock_fills)



# Four batch reads, never one query per fill/case. Fixed SQL only.
CURRENT_TRADES_SQL="""SELECT CASE WHEN pg_catalog.octet_length(pg_catalog.to_jsonb(t)::text)<=24000 THEN pg_catalog.to_jsonb(t)::text END,xmin::text FROM public.trades t WHERE trade_id=ANY(%s) OR entry_order_id::text=ANY(%s) OR exit_order_id::text=ANY(%s) ORDER BY trade_id LIMIT 65"""
CURRENT_FILLS_SQL="""SELECT CASE WHEN pg_catalog.octet_length(pg_catalog.to_jsonb(a)::text)<=24000 THEN pg_catalog.to_jsonb(a)::text END,xmin::text FROM public.alpaca_activities a WHERE activity_type='FILL' AND (order_id::text=ANY(%s) OR activity_id=ANY(%s)) ORDER BY activity_id LIMIT 129"""
CLAIMS_SQL="""SELECT 'FILL',pg_catalog.to_jsonb(f)::text FROM public.trade_allocation_fill_claims f WHERE allocation_id=ANY(%s::uuid[]) OR activity_id=ANY(%s) OR order_id::text=ANY(%s) UNION ALL SELECT 'ORDER',pg_catalog.to_jsonb(o)::text FROM public.trade_allocation_order_claims o WHERE allocation_id=ANY(%s::uuid[]) OR order_id::text=ANY(%s) ORDER BY 1,2 LIMIT 65"""
HISTORY_SQL="""SELECT a.approval_ref,a.allocation_id::text,a.request_digest,CASE WHEN pg_catalog.octet_length(a.request::text)<=2048 THEN a.request::text END,a.manifest=c.manifest,CASE WHEN pg_catalog.octet_length(a.result::text)<=2048 THEN a.result::text END,a.policy_revision FROM public.trade_allocation_audit a JOIN public.trade_allocation_cases c ON c.allocation_id=a.allocation_id WHERE a.allocation_id=ANY(%s::uuid[]) ORDER BY a.policy_revision LIMIT 65"""

def _named(sql,names):
    for name in names:sql=sql.replace('%s','%('+name+')s',1)
    need('%s' not in sql,'FIXED_QUERY_BINDING');return sql

# One whole-state query, not four query/recheck pairs. Every permanent claim and
# audit row carries its xmin inside the canonical wire commitment.
STATE_SQL="WITH ct(body,version) AS MATERIALIZED ("+_named(CURRENT_TRADES_SQL,('trades','orders','orders'))+"), cf(body,version) AS MATERIALIZED ("+_named(CURRENT_FILLS_SQL,('orders','activities'))+"), cc(kind,body) AS MATERIALIZED ("+_named(CLAIMS_SQL.replace('pg_catalog.to_jsonb(f)::text',"(pg_catalog.to_jsonb(f)||pg_catalog.jsonb_build_object('source_xmin',f.xmin::text))::text").replace('pg_catalog.to_jsonb(o)::text',"(pg_catalog.to_jsonb(o)||pg_catalog.jsonb_build_object('source_xmin',o.xmin::text))::text"),('allocations','activities','orders','allocations','orders'))+"), ch(approval,allocation,digest,request,equal_manifest,result,revision,av,cv) AS MATERIALIZED ("+_named(HISTORY_SQL.replace('a.policy_revision FROM','a.policy_revision,a.xmin::text,c.xmin::text FROM'),('allocations',))+"), combined AS (SELECT 'TRADE' kind,pg_catalog.json_build_array(body,version)::text body FROM ct UNION ALL SELECT 'FILL',pg_catalog.json_build_array(body,version)::text FROM cf UNION ALL SELECT 'CLAIM',pg_catalog.json_build_array(kind,body)::text FROM cc UNION ALL SELECT 'HISTORY',pg_catalog.json_build_array(approval,allocation,digest,request,equal_manifest,result,revision,av,cv)::text FROM ch) SELECT kind,body FROM combined ORDER BY kind,body LIMIT 321"

def read_rows(cursor,limit):
    rows=cursor.fetchmany(limit+1);need(len(rows)<=limit,'READ_BOUND');return rows

def read_extensions(cursor,records,identity,policy_revision,excluded_ids,*,allocation_type,attest):
    """Called by canonical A under its existing shared policy lock."""
    D=policy_module();need(len(records)<=2,'ALLOCATION_SCOPE')
    if not records:return [],{},[]
    attest(cursor);groups=[];all_orders=[];all_ids=[];seen_trades=set();seen_cases=set();allocation_ids=[]
    for row in records:
        allocation_id,state,revision,request_text,manifest_text,digest,audit_request,audit_result,audit_revision,group_ref=row
        request,manifest=D.document(request_text),D.document(manifest_text)
        need(len(D.canonical(request).encode())<=2048 and type(revision) is int and 1<=revision<=32,'ALLOCATION_REQUEST')
        need(state in ('ACTIVE','REVOKED') and request.get('action')==('APPLY' if state=='ACTIVE' else 'REVOKE')
             and request.get('allocation_id')==allocation_id==manifest.get('allocation_id')
             and request.get('trade_id')==manifest.get('trade_id') and request.get('manifest_sha256')==D.fingerprint(manifest)
             and group_ref==manifest.get('group_approval_ref') and revision==request.get('expected_allocation_revision',-1)+1
             and (revision%2==1)==(state=='ACTIVE'),'ALLOCATION_REQUEST')
        need(request.get('identity')==list(identity) and request.get('expected_policy_revision',-1)+1==audit_revision<=policy_revision
             and digest==D.fingerprint(request) and D.document(audit_request)==request,'ALLOCATION_AUDIT')
        expected_receipt=dict(allocation_id=allocation_id,trade_id=manifest['trade_id'],state=state,allocation_revision=revision,
            policy_revision=audit_revision,approval_ref=request['approval_ref'])
        need(D.document(audit_result)==expected_receipt,'ALLOCATION_AUDIT')
        info=validate_group(manifest,identity);tid=manifest['trade_id']
        need(manifest['case'] not in seen_cases,'ALLOCATION_CASE_REUSED');seen_cases.add(manifest['case'])
        need(tid not in seen_trades and (state!='ACTIVE' or tid not in excluded_ids),'ALLOCATION_COLLISION');seen_trades.add(tid)
        need(not set(info['orders']).intersection(all_orders) and not set(info['activity_ids']).intersection(all_ids),'ALLOCATION_REUSED')
        all_orders.extend(info['orders']);all_ids.extend(info['activity_ids']);allocation_ids.append(allocation_id)
        groups.append((state,revision,request,manifest,info))
    cursor.execute(STATE_SQL,dict(trades=sorted(seen_trades),orders=all_orders,activities=all_ids,allocations=allocation_ids))
    state={k:[] for k in ('TRADE','FILL','CLAIM','HISTORY')}
    for kind,body in read_rows(cursor,320):
        need(kind in state and type(body) is str,'STATE_WIRE')
        values=D.document('{"row":'+body+'}')['row'];need(type(values) is list,'STATE_WIRE');state[kind].append(values)
    need(all(len(state[k])<=limit for k,limit in [('TRADE',64),('FILL',128),('CLAIM',64),('HISTORY',64)]),'STATE_BOUND')
    current={}
    for body,version in state['TRADE']:
        need(body is not None,'ROW_OVERSIZE');row=D.document(body);need(row['trade_id'] not in current,'OWNER_DUPLICATE');current[row['trade_id']]=row
    actual={}
    for body,version in state['FILL']:
        need(body is not None,'ROW_OVERSIZE');row=D.document(body);need(row['activity_id'] not in actual,'FILL_DUPLICATE');actual[row['activity_id']]=row
    expected={s['row']['activity_id']:s['row'] for _,_,_,m,_ in groups for s in manifest_fills(m)}
    need(actual==expected,'FILL_CHANGED')
    fill_claims={};order_claims={}
    for kind,body in state['CLAIM']:
        claim=D.document(body)
        if kind=='FILL':
            key=claim['activity_id'];need(key not in fill_claims,'CLAIM_DUPLICATE');fill_claims[key]=claim
        else:
            need(kind=='ORDER','CLAIM_KIND');key=claim['order_id'];need(key not in order_claims,'CLAIM_DUPLICATE');order_claims[key]=claim
    need(set(fill_claims)==set(all_ids) and set(order_claims)==set(all_orders),'CLAIM_SCOPE')
    histories={aid:[] for aid in allocation_ids}
    for approval,aid,digest,req,equal_manifest,res,rev,av,cv in state['HISTORY']:
        need(type(av) is str and av.isdigit() and type(cv) is str and cv.isdigit(),'STATE_VERSION')
        need(aid in histories and req is not None and res is not None and equal_manifest is True,'ALLOCATION_HISTORY')
        histories[aid].append((approval,digest,D.document(req),D.document(res),rev))
    active=[];reserved={};reservation_cases=[]
    for state,revision,request,manifest,info in groups:
        aid=manifest['allocation_id'];tid=manifest['trade_id'];history=histories[aid]
        need(len(history)==revision,'ALLOCATION_HISTORY');previous=0
        for index,(approval,digest,req,res,rev) in enumerate(history,1):
            expected_state='ACTIVE' if index%2 else 'REVOKED'
            need(req.get('identity')==list(identity) and req.get('allocation_id')==aid and req.get('trade_id')==tid
                 and req.get('manifest_sha256')==D.fingerprint(manifest) and req.get('expected_allocation_revision')==index-1
                 and req.get('action')==('APPLY' if expected_state=='ACTIVE' else 'REVOKE')
                 and req.get('expected_policy_revision',-1)+1==rev and previous<rev<=policy_revision
                 and approval==req.get('approval_ref') and digest==D.fingerprint(req),'ALLOCATION_HISTORY')
            need(res==dict(allocation_id=aid,trade_id=tid,state=expected_state,allocation_revision=index,
                policy_revision=rev,approval_ref=approval),'ALLOCATION_HISTORY');previous=rev
        need(history[-1][2]==request,'ALLOCATION_HISTORY')
        need(tid in current and (state!='ACTIVE' or current[tid]==manifest['row_snapshot']['row']),'ROW_CHANGED')
        for owner,row in current.items():
            if owner!=tid:need(not any(row.get(k) in info['orders'] for k in ('entry_order_id','exit_order_id')),'GLOBAL_OWNER_CONFLICT')
        for order in info['orders']:
            need(order_claims[order].get('allocation_id')==aid,'ORDER_CLAIM');reserved[order]=tid
        for snapshot_ in manifest_fills(manifest):
            f=snapshot_['row'];claim=fill_claims[f['activity_id']]
            need(claim.get('allocation_id')==aid and claim.get('order_id')==f['order_id']
                 and claim.get('activity_type')=='FILL' and number(claim.get('quantity'))==number(f['qty']),'FILL_CLAIM')
        reservation_cases.append(dict(allocation_id=aid,trade_id=tid,state=state,allocation_revision=revision,approval_ref=request['approval_ref']))
        if state=='ACTIVE':active.append(allocation_type(aid,tid,state,revision,info['quantity'],info['quantity'],info['gross'],
            request['approval_ref'],(manifest['group_approval_ref'],),tuple(info['orders']),tuple(info['activity_ids']),clock_fills=info['clock_fills'],
            row_fingerprint=D.fingerprint(manifest['row_snapshot']['row'])))
    return active,reserved,reservation_cases
