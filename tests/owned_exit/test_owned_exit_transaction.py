import sys,unittest
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts import owned_exit_repair as O,owned_exit_transaction as X
import owned_exit_preview_fixture as P
from test_owned_exit_core import fixture,uid
D=O.policy_module()

def Policy(b,revision):
    from scripts import trade_allocation as A
    overlay=A.result_read('VALIDATED',[],tuple(b['identity']),revision,{},[])
    return D.Policy(*b['identity'],revision,(),scope='VALIDATED_PERFORMANCE',allocation_metadata_json=overlay._metadata_json)

def ready():
    b=fixture();now=datetime.now(timezone.utc);b['broker_proof']['observed_at']=(now-timedelta(seconds=1)).isoformat()
    b['policy_stable_hash']=O.stable_policy(Policy(b,7).metadata());c=P.capsule(b,dict(policy_owner=True,schema_create=True,audit_absent=True,legacy_constraints=[['fixture_scope','CHECK ((trade_id = 110))']]),now=now)
    request=dict(action='APPLY_OWNED_EXITS',approval_ref='PUBLIC_SYNTHETIC_APPROVAL',preview_digest=c['self_sha256'],allow_new_audit_acl_cleanup=True,source_manifest_sha256='a'*64,execution_context='POLICY_OWNER')
    return c,request

class Connection:
    autocommit=False
    def __init__(self,capsule,*,fail_at=None,uncertain=False,stale=False):
        self.b=deepcopy(capsule['bundle']);self.state=dict(rows={s['row']['trade_id']:deepcopy(s) for s in self.b['owners']},audit={},alloc={},fills=[],orders=[],history=[],revision=7,schema=[])
        self.original=deepcopy(self.state);self.fail_at=fail_at;self.uncertain=uncertain;self.stale=stale
        self.committed=False;self.closed=False;self.rollback_count=0;self.calls=[];self.native=Cursor(self)
    def get_transaction_status(self):return 0
    def cursor(self):return self.native
    def commit(self):
        self.committed=True
        if self.uncertain:raise RuntimeError('synthetic lost commit acknowledgment')
    def rollback(self):
        self.rollback_count+=1
        if not self.committed:self.state=deepcopy(self.original)
    def close(self):self.closed=True

class Cursor:
    def __init__(self,conn):self.c=conn;self.pending=[];self.rowcount=-1
    def execute(self,sql,p=None):
        self.c.calls.append(sql);self.pending=[];self.rowcount=-1
        if sql.startswith('SET '):return
        if self.c.fail_at and self.c.fail_at in sql:raise RuntimeError('synthetic partial failure')
        state=self.c.state
        if sql=="SELECT pg_catalog.current_setting('session_replication_role')":self.pending=[('origin',)]
        elif sql.startswith(('SELECT p.proname,CASE WHEN','SELECT c.relname,t.tgname,')):
            from test_owned_exit_legacy import fixture as legacy
            helper=legacy.active_cursor(legacy.manifest(),legacy.request(legacy.manifest()))
            self.pending=helper.function_metadata if sql.startswith('SELECT p.proname,') else helper.trigger_metadata
        elif sql==X.SCHEMA_ACTOR_SQL:self.pending=[getattr(self.c,'actor_context',(True,False,True,True,True,True))]
        elif sql.startswith('SELECT conname,pg_catalog.pg_get_constraintdef'):self.pending=[('fixture_scope','CHECK ((trade_id = 110))')]
        elif sql.startswith('SELECT a.grantee,CASE WHEN a.grantee=0'):self.pending=[(123,'fixture_role')]
        elif sql=="SELECT pg_catalog.to_regclass('public.jbravo_owned_exit_correction_audit') IS NOT NULL":self.pending=[(bool(state['schema']),)]
        elif sql.startswith(('CREATE ','ALTER ','REVOKE ')):state['schema'].append(sql)
        elif sql==X.ATTEST_SQL:
            fn=dict(body=X.CASE_GUARD_BODY,owner=True,definer=False,config=True,language='plpgsql',trigger=True,args=0)
            self.pending=[('FUNCTION',D.canonical(fn)),('AUDIT',D.canonical(dict(owner=True,ordinary=True,private=True,columns=True,primary_key=True))),
                ('TRIGGER',D.canonical(dict(name='trade_owned_exit_case_guard',enabled=True,bits=23,function='trade_owned_exit_case_guard',public_function=True))),
                ('TRIGGER',D.canonical(dict(name='jbravo_owned_exit_audit_immutable',enabled=True,bits=58,function='trade_allocation_immutable',public_function=True))),('FK',D.canonical(dict(valid=True)))]
        elif sql.startswith('LOCK TABLE'):pass
        elif sql.startswith('SELECT trade_id FROM public.trades'):
            self.pending=[(t,) for t in sorted(state['rows'])]
        elif sql==P.SNAPSHOT_SQL:
            self.pending=[('TRADE',D.canonical(s['row']),s['version']) for s in state['rows'].values()]
            self.pending += [('FILL',D.canonical(s['row']),s['version']) for s in self.c.b['fills']]
            self.pending += [('EVENT',D.canonical(e['snapshot']['row']),e['snapshot']['version']) for e in self.c.b['events']]
            if self.c.stale:self.pending[0]=('TRADE',self.pending[0][1],'101')
        elif sql.startswith('SELECT request_digest FROM '+X.AUDIT):
            if p[0] in state['audit']:self.pending=[(state['audit'][p[0]][1],)]
        elif sql.startswith('SELECT approval_ref FROM public.trade_disposition_audit'):pass
        elif sql=='SELECT pg_catalog.transaction_timestamp()':self.pending=[(datetime.now(timezone.utc),)]
        elif sql==X.UPDATE_CASE1:
            order,price,stamp,gross,updated,tid,version,before=p;s=state['rows'][tid]
            if version!=s['version'] or D.document(before)!=s['row']:self.rowcount=0;return
            s['row'].update(exit_order_id=order,exit_price=price,exit_time=stamp,realized_pnl=gross,updated_at=updated.isoformat());s['version']='201'
            self.pending=[(D.canonical(s['row']),'201')];self.rowcount=1
        elif sql==X.AUDIT_INSERT:state['audit'][p[0]]=p;self.rowcount=1
        elif sql.startswith('UPDATE public.trade_disposition_policy'):
            self.rowcount=int(state['revision']==p[0]);state['revision']+=self.rowcount
        elif sql.startswith('INSERT INTO public.trade_allocation_cases'):
            aid,tid,req,manifest,group=p;state['alloc'][aid]=(tid,D.document(req),D.document(manifest),group);self.rowcount=1
        elif sql.startswith('INSERT INTO public.trade_allocation_order_claims'):
            state['orders'].append(dict(order_id=p[0],allocation_id=p[1]));self.rowcount=1
        elif sql.startswith('INSERT INTO public.trade_allocation_fill_claims'):
            state['fills'].append(dict(activity_id=p[0],activity_type='FILL',order_id=p[1],allocation_id=p[2],quantity=p[3]));self.rowcount=1
        elif sql.startswith('INSERT INTO public.trade_allocation_audit'):
            state['history'].append(p);self.rowcount=1
        elif sql.startswith('SELECT request_digest,preview_digest,payload::text'):
            a=state['audit'][p[0]];self.pending=[tuple(a[1:])]
        elif sql==X.RECOVERY_SQL:
            if p[0] in state['audit']:
                a=state['audit'][p[0]];self.pending=[tuple(a[1:])+tuple(self.c.b['identity'])+(True,)]
        elif sql.startswith('SELECT trade_id,pg_catalog.to_jsonb(t)::text'):
            self.pending=[(tid,D.canonical(s['row'])) for tid,s in sorted(state['rows'].items())]
        elif sql==X.CONSUMER_ROWS_SQL:
            self.pending=[(D.canonical(dict(row=s['row'],version=s['version'])),) for tid,s in sorted(state['rows'].items()) if tid in p[0]]
        elif 'SELECT pg_catalog.octet_length(pg_catalog.json_build_array(' in sql:
            self.pending=[(12000,)]*len(state['alloc']) if 'c.allocation_id::text' in sql else [(800,),(700,),(200,)]
        elif sql.startswith('SELECT allocation_id::text,manifest::text,group_approval_ref'):
            self.pending=[(aid,D.canonical(v[2]),v[3]) for aid,v in state['alloc'].items()]
        elif sql==O.CLAIMS_SQL:self.pending=[('FILL',D.canonical(f)) for f in state['fills']]+[('ORDER',D.canonical(o)) for o in state['orders']]
        elif sql=='SELECT policy_revision FROM public.trade_disposition_policy WHERE singleton=1':self.pending=[(state['revision'],)]
        else:
            from dashboards import execution_gross_projection as E
            if sql==E.FILLS_SQL:
                from test_owned_exit_full_consumer import projected_fill
                self.pending=[(D.canonical(dict(row=projected_fill(s),version=s['version'])),) for s in self.c.b['fills'] if s['row']['order_id'] in p[0]]
            else:raise AssertionError('unhandled fixed query '+sql[:100])
    def fetchmany(self,n):r=self.pending[:n];self.pending=self.pending[n:];return r

class Transaction(unittest.TestCase):
    def invoke(self,conn,c,r,source=True):
        def policy(cursor,identity,exclusive,scope=None):
            if scope!='VALIDATED_PERFORMANCE':raise AssertionError('preview policy scope must be explicit and unchanged')
            from scripts import trade_allocation as A
            active=[];reserved={};cases=[]
            for aid,(tid,request,manifest,group) in conn.state['alloc'].items():
                info=O.validate_group(manifest,identity)
                active.append(A.Allocation(aid,tid,'ACTIVE',1,info['quantity'],info['quantity'],info['gross'],request['approval_ref'],(group,),tuple(info['orders']),tuple(info['activity_ids']),clock_fills=info['clock_fills']))
                reserved.update({o:tid for o in info['orders']})
            overlay=A.result_read('VALIDATED',active,identity,conn.state['revision'],reserved,cases)
            return D.Policy(*identity,conn.state['revision'],(),scope,tuple(active),overlay._metadata_json)
        with patch.object(D,'read_policy',side_effect=policy):
            return X.apply_correction(conn,c,r,verify_sources=lambda:source,rebind=P.rebind)
    def test_actual_apply_caller_commit_and_complete_beforeafter_audit(self):
        c,r=ready();conn=Connection(c);result=self.invoke(conn,c,r)
        self.assertEqual(result['status'],'COMMITTED',result);self.assertTrue(result['committed'])
        self.assertEqual((result['committed_updated_rows'],result['committed_allocation_rows']),(1,2))
        self.assertEqual(len(conn.state['history']),2);self.assertEqual(len(conn.state['audit']),1)
        for case in ('2','3','6'):
            tid=c['plan']['cases'][case]['row']['trade_id'];self.assertEqual(conn.state['rows'][tid],conn.original['rows'][tid])
        self.assertTrue(result['rollback_acknowledged']);self.assertTrue(conn.closed)
        self.assertLessEqual(result['usage']['total_bound'],256)
    def test_stale_version_refuses_before_native_update(self):
        c,r=ready();conn=Connection(c,stale=True);res=self.invoke(conn,c,r)
        self.assertEqual(res['status'],'ROLLED_BACK');self.assertEqual(res['committed_updated_rows'],0)
        self.assertNotIn(X.UPDATE_CASE1,conn.calls);self.assertEqual(conn.state,conn.original)
    def test_partial_failure_rolls_back_native_audit_allocations_and_claims(self):
        for statement in ('INSERT INTO public.trade_allocation_fill_claims','SET CONSTRAINTS ALL IMMEDIATE'):
            c,r=ready();conn=Connection(c,fail_at=statement)
            if statement.startswith('SET '):conn.fail_at=None # inject after ordinary SET handling below
            original=conn.native.execute
            def execute(sql,p=None):
                if sql==statement:raise RuntimeError('synthetic constraint failure')
                return original(sql,p)
            if statement.startswith('SET '):conn.native.execute=execute
            res=self.invoke(conn,c,r)
            with self.subTest(statement=statement):
                self.assertEqual(res['status'],'ROLLED_BACK');self.assertFalse(res['committed']);self.assertEqual(conn.state,conn.original)
                self.assertEqual(res['committed_allocation_rows'],0)
    def test_uncertain_commit_is_not_replayed_or_called_rolled_back(self):
        c,r=ready();conn=Connection(c,uncertain=True);res=self.invoke(conn,c,r)
        self.assertEqual(res['status'],'COMMIT_UNCERTAIN');self.assertTrue(res['commit_uncertain'])
        self.assertIsNone(res['committed_updated_rows']);self.assertTrue(conn.committed);self.assertEqual(conn.calls.count(X.UPDATE_CASE1),1)
        self.assertTrue(res['rollback_acknowledged']);self.assertEqual(len(conn.state['audit']),1)
    def test_missing_authority_source_changed_expired_or_modified_preview_no_sql(self):
        for scenario in ('authority','source','expired','modified'):
            c,r=ready();conn=Connection(c)
            if scenario=='authority':r['action']='OBSERVE'
            if scenario=='expired':c['expires_at']='2020-01-01T00:00:00Z'
            if scenario=='modified':c['plan']['case1_after_fields']['exit_price']='999'
            res=self.invoke(conn,c,r,source=scenario!='source')
            with self.subTest(scenario=scenario):self.assertFalse(res['committed']);self.assertEqual(conn.calls,[])
    def test_duplicate_apply_approval_refuses_without_update(self):
        c,r=ready();conn=Connection(c);conn.state['audit'][r['approval_ref']]=(r['approval_ref'],'a'*64)
        res=self.invoke(conn,c,r);self.assertFalse(res['committed']);self.assertNotIn(X.UPDATE_CASE1,conn.calls)
    def test_safe_identifier_quoting_and_strict_budget(self):
        self.assertEqual(X.qident('role"; --'),'"role""; --"')
        with self.assertRaises(O.RepairError):X.qident('bad\x00role')
        c,_=ready();conn=Connection(c);budget=X.Budget(conn.native,maximum=4)
        budget.execute('SET TRANSACTION ISOLATION LEVEL READ COMMITTED')
        with self.assertRaisesRegex(O.RepairError,'TRANSACTION_BUDGET'):budget.execute('SELECT forbidden')

if __name__=='__main__':unittest.main()
