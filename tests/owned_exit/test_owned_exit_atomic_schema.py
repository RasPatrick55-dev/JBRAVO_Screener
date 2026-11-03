import sys,unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts import owned_exit_repair as O,owned_exit_transaction as X,trade_allocation as A
from test_owned_exit_transaction import ready,Connection,D
import test_owned_exit_transaction as transaction_fixture

class AtomicSchema(unittest.TestCase):
    def invoke(self,conn,c,r):return transaction_fixture.Transaction().invoke(conn,c,r)
    def test_one_commit_contains_schema_and_exact_native_and_allocation_effects(self):
        c,r=ready();conn=Connection(c);res=self.invoke(conn,c,r)
        self.assertEqual(res['status'],'COMMITTED');self.assertTrue(res['committed_schema_created'])
        self.assertEqual(len(conn.state['audit']),1);self.assertEqual(len(conn.state['alloc']),2)
        self.assertEqual(conn.state['revision'],10);self.assertEqual(res['committed_updated_rows'],1)
        self.assertTrue(any('CREATE TABLE public.jbravo_owned_exit_correction_audit' in q for q in conn.state['schema']))
    def test_missing_owner_create_or_absence_predicate_refuses_before_ddl_and_native_dml(self):
        for key in ('policy_owner','schema_create','audit_absent'):
            c,r=ready();c['catalog'][key]=False;c['self_sha256']=D.fingerprint({k:v for k,v in c.items() if k!='self_sha256'});r['preview_digest']=c['self_sha256'];conn=Connection(c)
            res=self.invoke(conn,c,r)
            with self.subTest(key=key):
                self.assertFalse(res['committed']);self.assertNotIn(X.UPDATE_CASE1,conn.calls)
                self.assertFalse(any(q.startswith(('CREATE ','ALTER ','REVOKE ')) for q in conn.calls));self.assertEqual(conn.state,conn.original)
    def test_ddl_failure_rolls_back_all_schema_and_no_native_update(self):
        c,r=ready();conn=Connection(c,fail_at='CREATE FUNCTION public.trade_owned_exit_case_guard')
        res=self.invoke(conn,c,r);self.assertEqual(res['status'],'ROLLED_BACK');self.assertEqual(conn.state,conn.original);self.assertNotIn(X.UPDATE_CASE1,conn.calls)
    def test_private_acl_revoke_only_new_table_and_identifier_quoting(self):
        c,r=ready();conn=Connection(c);original=conn.native.execute
        def execute(sql,p=None):
            original(sql,p)
            if sql.startswith('SELECT a.grantee,CASE WHEN a.grantee=0'):conn.native.pending=[(0,None),(123,'role"; --')]
        conn.native.execute=execute;res=self.invoke(conn,c,r);self.assertTrue(res['committed'],res)
        revokes=[q for q in conn.calls if q.startswith('REVOKE ')]
        self.assertEqual(revokes,['REVOKE ALL PRIVILEGES ON TABLE '+X.AUDIT+' FROM PUBLIC RESTRICT','REVOKE ALL PRIVILEGES ON TABLE '+X.AUDIT+' FROM "role""; --" RESTRICT'])
        self.assertFalse(any('GRANT ' in q or 'ALTER DEFAULT PRIVILEGES' in q or 'CASCADE' in q for q in conn.calls))
    def test_two_other_roles_or_missing_acl_approval_roll_back(self):
        for scenario in ('two_roles','unapproved'):
            c,r=ready();conn=Connection(c);original=conn.native.execute
            if scenario=='unapproved':r['allow_new_audit_acl_cleanup']=False
            else:
                def execute(sql,p=None):
                    original(sql,p)
                    if sql.startswith('SELECT a.grantee,CASE WHEN a.grantee=0'):conn.native.pending=[(123,'one'),(124,'two')]
                conn.native.execute=execute
            res=self.invoke(conn,c,r)
            with self.subTest(scenario=scenario):self.assertFalse(res['committed']);self.assertEqual(conn.state,conn.original);self.assertNotIn(X.UPDATE_CASE1,conn.calls)
    def test_exact_legacy_constraint_is_required_without_overwriting_existing_schema(self):
        c,r=ready();conn=Connection(c);original=conn.native.execute
        def execute(sql,p=None):
            original(sql,p)
            if sql.startswith('SELECT conname,pg_catalog.pg_get_constraintdef'):conn.native.pending=[('fixture_scope','CHECK ((trade_id >= 110))')]
        conn.native.execute=execute;res=self.invoke(conn,c,r)
        self.assertFalse(res['committed']);self.assertFalse(any(q.startswith('CREATE ') for q in conn.calls))
    def test_catalog_attestation_refuses_columns_primary_key_public_acl_and_wrong_trigger_namespace(self):
        for kind,key,value in [('AUDIT','columns',False),('AUDIT','primary_key',False),('AUDIT','private',False),('AUDIT','private','true'),('TRIGGER','public_function',False),('FK','valid',False)]:
            c,r=ready();conn=Connection(c);original=conn.native.execute
            def execute(sql,p=None):
                original(sql,p)
                if sql==X.ATTEST_SQL:
                    out=[];changed=False
                    for k,body in conn.native.pending:
                        d=D.document(body)
                        if k==kind and not changed:d[key]=value;changed=True
                        out.append((k,D.canonical(d)))
                    conn.native.pending=out
            conn.native.execute=execute;res=self.invoke(conn,c,r)
            with self.subTest(kind=kind,key=key):self.assertFalse(res['committed']);self.assertEqual(conn.state,conn.original);self.assertNotIn(X.UPDATE_CASE1,conn.calls)
    def test_old_five_function_pins_and_ten_trigger_contract_are_preserved(self):
        from test_owned_exit_legacy import fixture
        self.assertEqual(len(A.FUNCTION_PINS),5);self.assertEqual(len(A.TRIGGER_CONTRACT),10)
        self.assertIn("bound.value->'snapshot'=pg_catalog.to_jsonb(activity)",fixture.FUNCTION_BODIES['trade_allocation_guard_original'])
        c,r=ready();plan=c['plan'];ms=O.allocation_manifests(c['bundle'],plan,approval_ref=r['approval_ref'],preview_digest=r['preview_digest'],allocation_ids=c['allocation_ids'])
        for m in ms:
            for item in m['fills']:
                self.assertEqual(set(item),{'snapshot','row_fingerprint','row_version'})
                self.assertEqual(item['row_fingerprint'],D.fingerprint(item['snapshot']))

if __name__=='__main__':unittest.main()
