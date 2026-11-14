"""Explicit approval-bound GTLB131 activation; never discovers a DB or runs on import.

Caller owns deployment authority, fresh independent broker proof, and exact local
source-pin verification. This actor owns one idle transaction and closes it.
"""
from datetime import datetime, timezone, timedelta
from . import whole_position_allocation as W, owned_exit_repair as O
from . import owned_exit_transaction as X, trade_allocation as A

OLD_SCOPE = "CHECK (((trade_id > 0) AND ((trade_id = 110) OR ((manifest ->> 'classification'::text) = 'OWNED_EXIT_COHORT_V1'::text))))"
NEW_SCOPE = """ALTER TABLE public.trade_allocation_cases ADD CONSTRAINT trade_owned_exit_scope_check
 CHECK ((trade_id>0 AND (trade_id=110 OR manifest->>'classification'='OWNED_EXIT_COHORT_V1'
 OR (trade_id=131 AND manifest->>'classification'='WHOLE_POSITION_MULTI_ORDER_V1' AND manifest->>'case'='GTLB131'))) IS TRUE)"""
SCHEMA = 'JBRAVO_GTLB131_ACTIVATION_V1'


def install_guard(cursor):
    """Existing schema only, exact predecessor. Called under exclusive policy lock."""
    O.need(X.attest_extension(cursor)==X.FUNCTION_SHA,'MULTI_PREDECESSOR_GUARD')
    cursor.execute('LOCK TABLE public.trade_allocation_cases IN ACCESS EXCLUSIVE MODE')
    cursor.execute("SELECT pg_catalog.pg_get_constraintdef(oid) FROM pg_catalog.pg_constraint WHERE conrelid='public.trade_allocation_cases'::regclass AND conname='trade_owned_exit_scope_check'")
    O.need(cursor.fetchone()==(OLD_SCOPE,),'MULTI_PREDECESSOR_SCOPE')
    cursor.execute('CREATE OR REPLACE FUNCTION public.trade_owned_exit_case_guard() RETURNS pg_catalog.trigger LANGUAGE plpgsql SET search_path = pg_catalog AS $multi$'+X.MULTI_CASE_GUARD_BODY+'$multi$')
    cursor.execute('ALTER TABLE public.trade_allocation_cases DROP CONSTRAINT trade_owned_exit_scope_check RESTRICT')
    cursor.execute(NEW_SCOPE)
    O.need(X.attest_extension(cursor)==X.MULTI_FUNCTION_SHA,'MULTI_GUARD_REQUIRED')


def validate_request(request, manifest, identity, *, enforce_expiry=True):
    D=O.policy_module(); W.validate_group(manifest,identity)
    O.need(type(request) is dict and set(request)=={'action','schema','approval_ref','identity','manifest_sha256',
        'expected_policy_revision','expected_policy_fingerprint','source_manifest_sha256','approved_at','expires_at'},'MULTI_REQUEST_SHAPE')
    O.need(request['action']=='APPLY_GTLB131' and request['schema']==SCHEMA and request['identity']==list(identity)
        and request['approval_ref']==manifest['group_approval_ref']
        and request['manifest_sha256']==D.fingerprint(manifest),'MULTI_REQUEST_BINDING')
    D.revision(request['expected_policy_revision']); D.text_field(request['approval_ref'])
    D.digest(request['expected_policy_fingerprint']); D.digest(request['source_manifest_sha256'])
    approved=D.aware_time(request['approved_at']);expires=D.aware_time(request['expires_at'])
    O.need(approved < expires <= approved+timedelta(days=7),'MULTI_APPROVAL_WINDOW')
    if enforce_expiry:O.need(approved <= datetime.now(timezone.utc) < expires,'MULTI_APPROVAL_EXPIRED')


def rebind(cursor, manifest):
    """Exact complete sources and xmin, global ownership, no existing claims."""
    D=O.policy_module()
    cursor.execute(O.CURRENT_TRADES_SQL,([131],list(W.ORDERS),list(W.ORDERS)))
    owners=[O.snapshot(dict(row=D.document(row),version=version)) for row,version in O.read_rows(cursor,64)]
    targets=[s for s in owners if s['row']['trade_id']==131]
    O.need(targets==[manifest['row_snapshot']],'MULTI_SOURCE_TRADE_CHANGED')
    for owner in owners:
        if owner['row']['trade_id']!=131:
            O.need(not any(owner['row'].get(k) in W.ORDERS for k in ('entry_order_id','exit_order_id')),'GLOBAL_OWNER_CONFLICT')
    cursor.execute(O.CURRENT_FILLS_SQL,(list(W.ORDERS),list(W.FILL_PINS)))
    actual=[O.snapshot(dict(row=D.document(row),version=version)) for row,version in O.read_rows(cursor,128)]
    O.need(len(actual)==5 and sorted(actual,key=lambda s:s['row']['activity_id'])==sorted(O.manifest_fills(manifest),key=lambda s:s['row']['activity_id']),'MULTI_SOURCE_FILLS_CHANGED')
    aid=manifest['allocation_id']
    cursor.execute(O.CLAIMS_SQL,([aid],list(W.FILL_PINS),list(W.ORDERS),[aid],list(W.ORDERS)))
    O.need(not O.read_rows(cursor,64),'MULTI_CLAIMS_EXIST')
    cursor.execute('SELECT allocation_id FROM public.trade_allocation_cases WHERE trade_id=131 OR allocation_id=%s',(aid,))
    O.need(not O.read_rows(cursor,2),'MULTI_CASE_EXISTS')
    return owners


def apply(conn, manifest, request, *, verify_sources):
    """No replay: uncertain commits require separate read-only reconciliation.

    verify_sources(pins,source_manifest_sha256) must verify exact deployed bytes,
    including all imported reader prerequisites, without making any changes.
    """
    D=O.policy_module(); result=dict(status='STOPPED',committed=False,commit_uncertain=False,phase='VALIDATE')
    cursor=None; commit_started=False
    try:
        identity=tuple(manifest['identity']); validate_request(request,manifest,identity)
        facts=[O.fill_fact(s) for s in O.manifest_fills(manifest)]
        W.validate_broker(manifest['broker_proof'],facts,now=datetime.now(timezone.utc))
        O.need(verify_sources(manifest['source_pins'],request['source_manifest_sha256']) is True,'SOURCE_BINDING')
        O.need(conn.get_transaction_status()==0 and not conn.autocommit,'TRANSACTION_NOT_IDLE')
        cursor=X.Budget(conn.cursor(),maximum=768,seconds=45)
        cursor.execute('SET TRANSACTION ISOLATION LEVEL READ COMMITTED READ WRITE')
        result['phase']='POLICY_BINDING'
        policy=D.read_policy(cursor,identity,exclusive=True,scope='VALIDATED_PERFORMANCE')
        O.need(policy.revision==request['expected_policy_revision'] and D.fingerprint(policy.metadata())==request['expected_policy_fingerprint'],'MULTI_POLICY_CHANGED')
        O.need(131 not in policy.all_excluded_ids,'MULTI_TARGET_EXCLUDED')
        cursor.execute('LOCK TABLE public.trades,public.alpaca_activities IN SHARE MODE')
        result['phase']='SOURCE_REBIND'; owners=rebind(cursor,manifest)
        ref=request['approval_ref']; subref=ref+'/GTLB131'; aid=manifest['allocation_id']
        cursor.execute('SELECT approval_ref FROM public.jbravo_owned_exit_correction_audit WHERE approval_ref=%s UNION ALL SELECT approval_ref FROM public.trade_disposition_audit WHERE approval_ref=ANY(%s) UNION ALL SELECT approval_ref FROM public.trade_allocation_audit WHERE approval_ref=ANY(%s)',(ref,[ref,subref],[ref,subref]))
        O.need(not O.read_rows(cursor,4),'MULTI_APPROVAL_REUSED')
        result['phase']='GUARD_EXTENSION'; install_guard(cursor)
        revision=policy.revision+1
        req=dict(action='APPLY',allocation_id=aid,trade_id=131,identity=list(identity),manifest_sha256=D.fingerprint(manifest),expected_allocation_revision=0,expected_policy_revision=policy.revision,approval_ref=subref)
        receipt=dict(allocation_id=aid,trade_id=131,state='ACTIVE',allocation_revision=1,policy_revision=revision,approval_ref=subref)
        payload=dict(schema=SCHEMA,request=request,row_snapshot=manifest['row_snapshot'],fills=manifest['fills'],owners=owners,
            broker_proof=manifest['broker_proof'],source_pins=manifest['source_pins'],financial_eligible=False,fees=None,net=None)
        bindings={aid:dict(trade_id=131,manifest=manifest)}
        cursor.execute(X.AUDIT_INSERT,(ref,D.fingerprint(request),manifest['preview_digest'],D.canonical(payload),D.canonical(bindings),D.canonical(receipt),revision))
        cursor.execute("INSERT INTO public.trade_allocation_cases(allocation_id,trade_id,state,allocation_revision,request,manifest,group_approval_ref) VALUES(%s,131,'ACTIVE',1,%s::jsonb,%s::jsonb,%s)",(aid,D.canonical(req),D.canonical(manifest),ref))
        for order in W.ORDERS:
            cursor.execute('INSERT INTO public.trade_allocation_order_claims(order_id,allocation_id) VALUES(%s,%s)',(order,aid))
        for item in O.manifest_fills(manifest):
            f=item['row'];cursor.execute("INSERT INTO public.trade_allocation_fill_claims(activity_id,activity_type,order_id,allocation_id,quantity) VALUES(%s,'FILL',%s,%s,%s)",(f['activity_id'],f['order_id'],aid,f['qty']))
        cursor.execute('UPDATE public.trade_disposition_policy SET policy_revision=policy_revision+1 WHERE singleton=1 AND policy_revision=%s',(policy.revision,))
        O.need(cursor.rowcount==1,'MULTI_POLICY_CAS')
        cursor.execute('INSERT INTO public.trade_allocation_audit(approval_ref,allocation_id,request_digest,request,manifest,result,policy_revision) VALUES(%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)',(subref,aid,D.fingerprint(req),D.canonical(req),D.canonical(manifest),D.canonical(receipt),revision))
        cursor.execute('SET CONSTRAINTS ALL IMMEDIATE')
        result['phase']='CANONICAL_READBACK'
        after=D.read_policy(cursor,identity,scope='VALIDATED_PERFORMANCE')
        O.need(after.revision==revision and after.active==policy.active and after.all_excluded_ids==policy.all_excluded_ids,'MULTI_POLICY_READBACK')
        old={a.trade_id:a for a in policy.allocations}; new={a.trade_id:a for a in after.allocations}
        O.need(set(new)==set(old)|{131} and all(new[k]==v for k,v in old.items()),'MULTI_EXISTING_ALLOCATION_CHANGED')
        O.need(new[131].gross_result==W.Decimal('-20.69') and new[131].entry_quantity==42 and new[131].financial_eligible is False,'MULTI_READBACK')
        A.validate_bound_trade(new[131],manifest['row_snapshot']['row'])
        cursor.execute('SELECT request_digest,payload::text,allocation_bindings::text,receipt::text,policy_revision FROM public.jbravo_owned_exit_correction_audit WHERE approval_ref=%s',(ref,))
        audit=cursor.fetchone()
        O.need(audit is not None and audit[0]==D.fingerprint(request) and D.document(audit[1])==payload
            and D.document(audit[2])==bindings and D.document(audit[3])==receipt and audit[4]==revision,'MULTI_AUDIT_READBACK')
        result['consumer_qualification']=qualify_consumers(cursor,identity,revision,old)
        O.need(verify_sources(manifest['source_pins'],request['source_manifest_sha256']) is True,'SOURCE_CHANGED_BEFORE_COMMIT')
        validate_request(request,manifest,identity)
        W.validate_broker(manifest['broker_proof'],facts,now=datetime.now(timezone.utc))
        result['phase']='COMMIT';commit_started=True;conn.commit()
        result.update(status='COMMITTED',committed=True,phase='COMPLETE',receipt=receipt)
    except Exception as error:
        result.update(status='COMMIT_UNCERTAIN' if commit_started else 'STOPPED',committed=None if commit_started else False,commit_uncertain=commit_started,
            reason='COMMIT_OUTCOME_UNCERTAIN' if commit_started else X.closed_failure(error))
    finally:
        try:conn.rollback();result['rollback_acknowledged']=True
        except Exception:result['rollback_acknowledged']=False
        try:conn.close();result['connection_closed']=bool(conn.closed)
        except Exception:result['connection_closed']=False
        result.update(financial_eligible=False,fees=None,net=None)
    return result


def qualify_consumers(cursor, identity, revision, old_allocations):
    """Actual bounded canonical transport plus both dashboard gross projections."""
    from dashboards import execution_gross_projection as E, recorded_trade_analytics as R
    from scripts import gross_paper_accounting as G
    D=O.policy_module()
    bounded=E.BoundedCursor(cursor,cursor.end,clock=cursor.clock)
    policy=D.read_policy(bounded,identity,scope='VALIDATED_PERFORMANCE')
    rows,execution=E.read_inputs(bounded,policy);bounded.recheck_canonical()
    current={a.trade_id:a for a in policy.allocations}
    O.need(policy.revision==revision and set(current)==set(old_allocations)|{131}
        and all(current[k]==v for k,v in old_allocations.items()),'MULTI_CONSUMER_ALLOCATIONS')
    O.need(execution.gross.get(131)==W.Decimal('-20.69') and 131 not in execution.unknown
        and execution.unknown.get(182)=='QUANTITY_MISMATCH','MULTI_EXECUTION_QUALIFICATION')
    gross=G.build_snapshot(policy,rows,expected_identity=identity,expected_revision=revision)
    O.need(gross['net_eligible'] is False and gross['fees'] is None and gross['net_pnl'] is None
        and gross['accounting_complete'] is False,'MULTI_GROSS_SCOPE')
    now=datetime.now(timezone.utc)
    for projection in (R.project(policy,rows,identity=identity,now=now),
            R.project(policy,rows,identity=identity,now=now,execution=execution)):
        O.need(all(projection.get(k) is False for k in ('net_eligible','accounting_complete','recovery_complete','entry_ready'))
            and projection.get('fees_status')=='UNKNOWN' and projection.get('net_status')=='UNKNOWN','MULTI_SUMMARY_SCOPE')
    return dict(status='CANONICAL_READERS_QUALIFIED',allocations=len(current),rows=len(rows),
        gtlb_gross='-20.69',ntra_execution='QUANTITY_MISMATCH',net_eligible=False,fees=None,net=None,
        tracked_statements=bounded.statements,wire_rows=bounded.rows,wire_bytes=bounded.bytes)


def reconcile(conn,manifest,request,*,verify_sources):
    """Fresh read-only outcome check, never an automatic replay authorization."""
    D=O.policy_module();result=dict(status='UNCONFIRMED',committed=None,replay_authorized=False)
    try:
        identity=tuple(manifest['identity']);validate_request(request,manifest,identity,enforce_expiry=False)
        O.need(verify_sources(manifest['source_pins'],request['source_manifest_sha256']) is True,'SOURCE_BINDING')
        O.need(conn.get_transaction_status()==0 and not conn.autocommit,'TRANSACTION_NOT_IDLE')
        cursor=X.Budget(conn.cursor(),maximum=512,seconds=45)
        cursor.execute('SET TRANSACTION ISOLATION LEVEL READ COMMITTED READ ONLY')
        cursor.execute('SELECT request_digest,payload::text,allocation_bindings::text,receipt::text,policy_revision FROM public.jbravo_owned_exit_correction_audit WHERE approval_ref=%s',(request['approval_ref'],))
        audit=cursor.fetchone()
        if audit is None:
            result['status']='AUDIT_ABSENT_NO_REPLAY'
        else:
            payload=D.document(audit[1]);bindings=D.document(audit[2]);receipt=D.document(audit[3])
            O.need(audit[0]==D.fingerprint(request) and payload.get('schema')==SCHEMA and payload.get('request')==request
                and bindings=={manifest['allocation_id']:dict(trade_id=131,manifest=manifest)}
                and receipt==dict(allocation_id=manifest['allocation_id'],trade_id=131,state='ACTIVE',allocation_revision=1,
                    policy_revision=request['expected_policy_revision']+1,approval_ref=request['approval_ref']+'/GTLB131')
                and audit[4]==request['expected_policy_revision']+1,'MULTI_RECOVERY_AUDIT')
            policy=D.read_policy(cursor,identity,scope='VALIDATED_PERFORMANCE')
            O.need(any(a.trade_id==131 and a.allocation_id==manifest['allocation_id'] for a in policy.allocations),'MULTI_RECOVERY_ACTIVE')
            old={a.trade_id:a for a in policy.allocations if a.trade_id!=131}
            result['consumer_qualification']=qualify_consumers(cursor,identity,policy.revision,old)
            result.update(status='COMMIT_CONFIRMED_CANONICAL_READBACK',committed=True)
    except Exception as error:
        result['reason']=X.closed_failure(error)
    finally:
        try:conn.rollback();result['rollback_acknowledged']=True
        except Exception:result['rollback_acknowledged']=False
        try:conn.close();result['connection_closed']=bool(conn.closed)
        except Exception:result['connection_closed']=False
    return result
