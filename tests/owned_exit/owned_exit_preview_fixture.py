"""Synthetic fixture helpers; no clients or configuration discovery."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from uuid import uuid4
from scripts import owned_exit_repair as O

SNAPSHOT_SQL="""WITH trades AS MATERIALIZED (
 SELECT t.*,t.xmin::text AS private_xmin FROM ONLY public.trades t
 WHERE trade_id=ANY(%(ids)s) OR entry_order_id::text=ANY(%(orders)s) OR exit_order_id::text=ANY(%(orders)s) ORDER BY trade_id LIMIT 65),
 fills AS MATERIALIZED (SELECT a.*,a.xmin::text AS private_xmin FROM ONLY public.alpaca_activities a
 WHERE activity_type='FILL' AND order_id::text=ANY(%(orders)s) ORDER BY activity_id LIMIT 129),
 events AS MATERIALIZED (SELECT e.*,e.xmin::text AS private_xmin FROM ONLY public.order_events e
 WHERE event_id=ANY(%(events)s) OR (event_type='TRAIL_SUBMIT' AND
 ((raw->'raw'->>'parent_order_id')=ANY(%(parents)s) OR (raw->>'parent_order_id')=ANY(%(parents)s))) ORDER BY event_id LIMIT 4),
 fc AS MATERIALIZED (SELECT f.*,f.xmin::text AS private_xmin FROM public.trade_allocation_fill_claims f
 WHERE order_id::text=ANY(%(orders)s) OR activity_id IN(SELECT activity_id FROM fills) ORDER BY activity_id LIMIT 65),
 oc AS MATERIALIZED (SELECT o.*,o.xmin::text AS private_xmin FROM public.trade_allocation_order_claims o
 WHERE order_id::text=ANY(%(orders)s) ORDER BY order_id LIMIT 65),
 combined AS (SELECT 'TRADE' kind,pg_catalog.to_jsonb(t)-'private_xmin' body,private_xmin FROM trades t
 UNION ALL SELECT 'FILL',pg_catalog.to_jsonb(f)-'private_xmin',private_xmin FROM fills f
 UNION ALL SELECT 'EVENT',pg_catalog.to_jsonb(e)-'private_xmin',private_xmin FROM events e
 UNION ALL SELECT 'FILL_CLAIM',pg_catalog.to_jsonb(f)-'private_xmin',private_xmin FROM fc f
 UNION ALL SELECT 'ORDER_CLAIM',pg_catalog.to_jsonb(o)-'private_xmin',private_xmin FROM oc o)
 SELECT kind,CASE WHEN pg_catalog.octet_length(body::text)<=24000 THEN body::text END,private_xmin
 FROM combined ORDER BY kind,body::text LIMIT 329"""

CATALOG_SQL="""SELECT pg_catalog.json_build_object(
 'policy_owner',p.relowner=(SELECT oid FROM pg_catalog.pg_roles WHERE rolname=current_user),
 'schema_create',pg_catalog.has_schema_privilege(current_user,'public','CREATE'),
 'audit_absent',pg_catalog.to_regclass('public.jbravo_owned_exit_correction_audit') IS NULL,
 'legacy_constraints',(SELECT pg_catalog.json_agg(pg_catalog.json_build_array(c.conname,pg_catalog.pg_get_constraintdef(c.oid))) FROM pg_catalog.pg_constraint c WHERE c.conrelid='public.trade_allocation_cases'::regclass AND c.contype='c'),
 'default_nonowner_grantees',(SELECT count(DISTINCT a.grantee) FROM pg_catalog.pg_default_acl d CROSS JOIN LATERAL pg_catalog.aclexplode(d.defaclacl) a WHERE d.defaclrole=(SELECT oid FROM pg_catalog.pg_roles WHERE rolname=current_user) AND d.defaclobjtype='r' AND (d.defaclnamespace=0 OR d.defaclnamespace='public'::regnamespace) AND a.grantee<>d.defaclrole),
 'default_public_grantee',(SELECT count(*)>0 FROM pg_catalog.pg_default_acl d CROSS JOIN LATERAL pg_catalog.aclexplode(d.defaclacl) a WHERE d.defaclrole=(SELECT oid FROM pg_catalog.pg_roles WHERE rolname=current_user) AND d.defaclobjtype='r' AND (d.defaclnamespace=0 OR d.defaclnamespace='public'::regnamespace) AND a.grantee=0)
 )::text FROM pg_catalog.pg_class p WHERE p.oid='public.trade_disposition_policy'::regclass"""

def capture_fixed(cursor,policy,chosen,events,orders,source_pins,*,now=None,read_catalog=True):
    D=O.policy_module();now=now or datetime.now(timezone.utc)
    cursor.execute(SNAPSHOT_SQL,dict(ids=[r['trade_id'] for r in chosen.values()],orders=orders,events=list(events.values()),parents=[chosen[c]['entry_order_id'] for c in O.CHILD]))
    grouped={k:[] for k in ('TRADE','FILL','EVENT','FILL_CLAIM','ORDER_CLAIM')}
    for kind,body,version in O.read_rows(cursor,328):
        O.need(kind in grouped and body is not None,'SNAPSHOT_BOUND')
        grouped[kind].append(O.snapshot(dict(row=D.document(body),version=version)))
    limits=dict(TRADE=64,FILL=128,EVENT=3,FILL_CLAIM=64,ORDER_CLAIM=64)
    O.need(all(len(grouped[k])<=limits[k] for k in limits),'SNAPSHOT_BOUND')
    trades={s['row']['trade_id']:s for s in grouped['TRADE']};event_rows={s['row']['event_id']:s for s in grouped['EVENT']}
    O.need(len(trades)==len(grouped['TRADE']) and len(event_rows)==3,'SNAPSHOT_DUPLICATE')
    catalog={}
    if read_catalog:
        cursor.execute(CATALOG_SQL);wire=O.read_rows(cursor,1);O.need(len(wire)==1,'CATALOG_BOUND');catalog=D.document(wire[0][0])
    metadata=policy.metadata();alloc=D.document(policy.allocation_metadata_json)
    bundle=dict(schema=O.PREVIEW_SCHEMA,identity=[policy.deployment_id,policy.database_id],policy_revision=policy.revision,
        policy_stable_hash=O.stable_policy(metadata),source_pins=deepcopy(source_pins),
        cases=[dict(case=case,snapshot=trades[chosen[case]['trade_id']]) for case in O.CASES],
        owners=grouped['TRADE'],fills=grouped['FILL'],events=[dict(case=c,snapshot=event_rows[events[c]]) for c in O.CHILD],
        relevant_fill_claims=grouped['FILL_CLAIM'],relevant_order_claims=grouped['ORDER_CLAIM'],
        reserved_order_owners=alloc['reserved_order_owners'],excluded_trade_ids=list(policy.excluded_ids),broker_proof=None)
    O.compile_plan(bundle,now=now,allow_unbound=True)
    return dict(bundle=bundle,catalog=catalog)

def rebind(cursor,plan,policy):
    """The apply actor supplies only identities from its protected preview."""
    chosen={int(c):s['row'] for c,s in plan['cases'].items()}
    events={int(c):s['row']['event_id'] for c,s in plan['events'].items()}
    # The original observation is used only as the comparison cutoff. New rows
    # cannot be adopted because every full beforeimage and version is compared.
    return capture_fixed(cursor,policy,chosen,events,plan['orders'],plan['source_pins'],
        now=datetime.now(timezone.utc),read_catalog=False)['bundle']

def capsule(bundle,catalog,*,now=None):
    D=O.policy_module();now=now or datetime.now(timezone.utc);plan=O.compile_plan(bundle,now=now)
    O.need(type(catalog) is dict,'CATALOG_SHAPE')
    doc=dict(schema=O.PREVIEW_SCHEMA,created_at=now.isoformat(),expires_at=(now+timedelta(days=7)).isoformat(),
        bundle=deepcopy(bundle),plan=plan,catalog=deepcopy(catalog),allocation_ids={str(c):str(uuid4()) for c in (3,6)})
    doc['self_sha256']=D.fingerprint(doc);O.safe_private(doc)
    O.need(len(D.canonical(doc).encode())<=O.PRIVATE_MAX,'PRIVATE_PREVIEW_BOUND')
    return doc
