"""Full actor and D/A/G/E readers over one SQL-response state model.

This exercises Python callers, counts and state transitions, not native SQL,
PostgreSQL trigger execution, concurrency or performance.
"""
import unittest
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import test_owned_exit_transaction as T
from test_owned_exit_full_consumer import Native,D,O,X
import owned_exit_preview_fixture as P

class Cursor(T.Cursor):
    def execute(self,sql,p=None):
        if 'pg_advisory_xact_lock' in sql:self.c.calls.append(sql);self.pending=[];return
        if sql==X.CONSUMER_ROWS_SQL:
            super().execute(sql,p)
            from test_owned_exit_legacy import fixture as legacy
            if 110 in p[0]:self.pending.append((D.canonical(dict(row=legacy.TRADE,version='100')),))
            return
        from dashboards import execution_gross_projection as E
        if sql==E.FILLS_SQL:
            from test_owned_exit_full_consumer import projected_fill
            self.c.calls.append(sql)
            self.pending=[(D.canonical(dict(row=projected_fill(s),version=s['version'])),) for s in self.c.reader.full_fills if s['row']['order_id'] in p[0]]
            return
        if 'SELECT pg_catalog.octet_length(pg_catalog.json_build_array(' in sql and 'c.allocation_id::text' in sql:
            super().execute(sql,p);self.pending.append((12000,));return
        try:return super().execute(sql,p)
        except AssertionError as error:
            if not str(error).startswith('unhandled fixed query '):raise
        n=self.c.reader;n.revision=self.c.state['revision'];n.groups=[];n.ms=[];n.histories=[]
        n.b=deepcopy(self.c.b);n.b['owners']=deepcopy(list(self.c.state['rows'].values()))
        for aid,(tid,q,m,group) in self.c.state['alloc'].items():
            saved=next(h for h in self.c.state['history'] if h[1]==aid)
            receipt=D.document(saved[5]);revision=saved[6]
            n.groups.append((aid,'ACTIVE',1,D.canonical(q),D.canonical(m),D.fingerprint(q),D.canonical(q),D.canonical(receipt),revision,group))
            n.ms.append(m);n.histories.append((q['approval_ref'],aid,D.fingerprint(q),D.canonical(q),True,D.canonical(receipt),revision))
        self.pending=n.native(sql,p)

class Connection(T.Connection):
    def __init__(self,c,reader):
        super().__init__(c);self.reader=reader;self.native=Cursor(self)

def ready():
    n=Native();n.groups=[];n.ms=[];n.histories=[];n.revision=7
    policy=D.read_policy(X.Budget(n),n.identity,scope='VALIDATED_PERFORMANCE')
    b=T.fixture();b['identity']=list(n.identity)
    b['policy_stable_hash']=O.stable_policy(policy.metadata())
    b['reserved_order_owners']=D.document(policy.allocation_metadata_json)['reserved_order_owners']
    now=datetime.now(timezone.utc);b['broker_proof']['observed_at']=(now-timedelta(seconds=1)).isoformat()
    c=P.capsule(b,dict(policy_owner=True,schema_create=True,audit_absent=True,legacy_constraints=[['fixture_scope','CHECK ((trade_id = 110))']]),now=now)
    r=dict(action='APPLY_OWNED_EXITS',approval_ref='PUBLIC_SYNTHETIC_APPROVAL',preview_digest=c['self_sha256'],allow_new_audit_acl_cleanup=True,source_manifest_sha256='a'*64,execution_context='POLICY_OWNER')
    return c,r,n

class IntegratedApply(unittest.TestCase):
    def test_full_apply_uses_real_policy_readers_preserves_crc_and_fits_budget(self):
        c,r,n=ready();conn=Connection(c,n)
        result=X.apply_correction(conn,c,r,verify_sources=lambda:True,rebind=P.rebind)
        self.assertEqual(result['status'],'COMMITTED',result)
        self.assertEqual((result['committed_updated_rows'],result['committed_allocation_rows']),(1,2))
        self.assertEqual(result['consumer_qualification']['status'],'FRESH_PROCESS_FOUR_CASE_CONSUMERS_QUALIFIED')
        self.assertEqual(result['committed_schema_created'],True)
        self.assertLessEqual(result['usage']['total_bound'],256)
        self.assertEqual(len(conn.state['history']),2)
        print('SYNTHETIC_INTEGRATED_APPLY_BUDGET',result['usage'])

if __name__=='__main__':unittest.main()
