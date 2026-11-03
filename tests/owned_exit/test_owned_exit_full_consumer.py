"""Actual D/A/E/T readers over a supplied SQL-response model, not PostgreSQL.

The closed wire envelope and row versions are modeled independently. This proves
Python caller contracts/query ceilings; it does not parse or execute server SQL.
"""
import sys,unittest,json,re,hashlib,time
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts import trade_disposition as D,trade_allocation as A,owned_exit_repair as O,owned_exit_transaction as X
from dashboards import execution_gross_projection as E,execution_read_transport as T
from test_owned_exit_core import fixture,NOW
from test_owned_exit_legacy import fixture as legacy
from test_owned_exit_readers import records as group_records,Cursor as GroupCursor
from test_owned_exit_transaction import Connection as MutationConnection,ready

def projected_fill(s):
    r=s['row'];raw=r['raw'];out={k:r[k] for k in ('activity_id','activity_type','order_id','symbol','side','qty','price','transaction_time')}
    for target,key in [('provider_id','id'),('provider_activity_id','activity_id'),('provider_activity_type','activity_type'),('provider_transaction_time','transaction_time'),('provider_order_id','order_id'),('provider_symbol','symbol'),('provider_side','side'),('provider_qty','qty'),('provider_quantity','quantity'),('provider_price','price')]:out[target]=raw.get(key)
    return out

class Native:
    def __init__(self):
        self.calls=[];self.wire=[];self.pending=[];self.fetch_sizes=[];self.bad_wire=False
        self.b,self.ms,self.groups,self.histories=group_records()
        self.identity=legacy.IDENTITY;self.revision=11
        self.b['identity']=list(self.identity)
        # Rebind only the synthetic group fixture's identity and its audit digest.
        updated=[];hist=[]
        for record,m in zip(self.groups,self.ms):
            m['identity']=list(self.identity);q=D.document(record[3]);q['identity']=list(self.identity);q['manifest_sha256']=D.fingerprint(m)
            rr=D.document(record[7]);updated.append((record[0],record[1],record[2],D.canonical(q),D.canonical(m),D.fingerprint(q),D.canonical(q),D.canonical(rr),record[8],record[9]))
            hist.append((q['approval_ref'],m['allocation_id'],D.fingerprint(q),D.canonical(q),True,D.canonical(rr),record[8]))
        self.groups=updated;self.histories=hist
        self.crc_manifest=legacy.manifest();self.crc_request=legacy.request(self.crc_manifest)
        self.crc_cursor=legacy.active_cursor(self.crc_manifest,self.crc_request)
        plan=O.compile_plan(dict(self.b,identity=list(self.identity)),now=NOW)
        self.rows=[deepcopy(s['snapshot']['row']) for s in self.b['cases']]+[deepcopy(legacy.TRADE)]
        self.rows[0].update(plan['case1_after_fields'])
        self.owners=deepcopy(self.rows)
        self.full_fills=deepcopy(self.b['fills'])+[dict(row=s['snapshot'],version='100') for s in self.crc_manifest['fills']]
    def execute(self,sql,params=None):
        self.calls.append((sql,params));self.pending=[]
        if sql.startswith(('SET ','LOCK ')) or ('pg_advisory_xact_lock' in sql and not sql.startswith('/*')):return
        match=re.match(r'/\* execution-read:([a-f0-9]{64}) rows:(\d+) bytes:(\d+) \*/ ',sql)
        if match:
            plans=T.plan_table();plans.update(E.READ_PLANS)
            original=next(q for q in plans if hashlib.sha256(q.encode()).hexdigest()==match[1]);plan=plans[original]
            rows=self.native(original,params)
            extra=plan.get('wire_arity',plan['arity'])-plan['arity']
            values=[tuple(row)+('300',)*extra for row in rows]
            bodies=[D.canonical({'row':list(row)})[7:-1] for row in values]
            total=sum(len(body.encode())+128 for body in bodies);count=len(bodies)
            left,byte_left=int(match[2]),int(match[3])
            self.pending=[(len(body.encode())>32640 or count>left or total>byte_left,
                body if len(body.encode())<=32640 and count<=left and total<=byte_left else None,count,total) for body in bodies]
            self.wire.extend(deepcopy(self.pending));return
        self.pending=self.native(sql,params)
    def native(self,sql,p):
        if sql=='SHOW transaction_isolation':return [('read committed',)]
        if sql.startswith('SELECT deployment_id,database_id,') or sql.startswith('SELECT deployment_id, database_id,'):return [(*self.identity,'fixture_db',self.revision,'fixture_db')]
        if 'FROM public.trade_disposition_cases c LEFT JOIN' in sql:return []
        if 'to_regclass(%s)' in sql:return [tuple(A.TABLES)]
        if sql.startswith('SELECT a.atttypid='):return [(True,True)]
        if sql.startswith('SELECT allocation_schema_version'):return [(1,)]
        if sql=="SELECT pg_catalog.current_setting('session_replication_role')":return [('origin',)]
        if sql.startswith('SELECT p.proname,CASE WHEN'):return self.crc_cursor.function_metadata
        if sql.startswith('SELECT c.relname,t.tgname,'):return self.crc_cursor.trigger_metadata
        if sql.startswith('SELECT c.relname,c.relkind,') and sql!=E.VISIBILITY_SQL:
            names=A.TABLES if 'c.relname=ANY' in sql else ('trades','alpaca_activities')
            return [(n,'r',False,False,False,True) for n in names]
        if sql.startswith('SELECT c.allocation_id::text,'):
            q=self.crc_request;m=self.crc_manifest
            crc=(legacy.CASE,'ACTIVE',1,D.canonical(q),D.canonical(m),D.fingerprint(q),D.canonical(q),D.canonical(A.receipt(q)),2,None)
            return [crc]+self.groups
        if sql.startswith('SELECT a.request_digest,CASE WHEN'):return self.crc_cursor.audit_history
        if 'WHERE t.trade_id=110 OR' in sql:return [(D.canonical(legacy.TRADE),)]
        if 'a.activity_pk=ANY' in sql:return [(D.canonical(s['snapshot']),) for s in self.crc_manifest['fills']]
        if sql.startswith('SELECT activity_id,activity_type,order_id,quantity::text'):return legacy.claim_rows()
        if sql.startswith('SELECT order_id,allocation_id::text'):return [(o,legacy.CASE) for o in A.ORDERS]
        if sql==X.ATTEST_SQL:
            c,_=ready();helper=MutationConnection(c).native;helper.execute(sql);return helper.pending
        if sql==O.STATE_SQL:
            helper=GroupCursor(self.b,self.ms,self.histories);helper.execute(sql,p)
            return helper.pending
        if sql==E.VISIBILITY_SQL:return [('alpaca_activities','r',False,False,False,False,True),('trades','r',False,False,False,False,True)]
        if sql==E.SCHEMA_SQL:return sorted((t,n,sorted(kinds)[0]) for t,cols in E.REQUIRED_COLUMNS.items() for n,kinds in cols.items())
        if sql==E.TRADES_SQL:return [(D.canonical(dict(row=r,version='100')),) for r in self.rows]
        if sql==E.OWNERS_SQL:return [(D.canonical(dict(row=r,version='100')),) for r in self.owners]
        if sql==E.FILLS_SQL:return [(D.canonical(dict(row=projected_fill(s),version=s['version'])),) for s in self.full_fills if s['row']['order_id'] in p[0]]
        raise AssertionError('Unmodeled fixed SQL '+sql[:100])
    def fetchmany(self,n):
        self.fetch_sizes.append(n);rows=self.pending[:n];self.pending=self.pending[n:];return rows
    def fetchone(self):r=self.fetchmany(1);return r[0] if r else None
    def fetchall(self):return self.fetchmany(4097)

class FullConsumer(unittest.TestCase):
    def test_real_policy_allocation_transport_and_execution_readers_resolve_four_cases_preserve_crc_and_hold_net(self):
        native=Native();cursor=E.BoundedCursor(native,time.monotonic()+8)
        p=D.read_policy(cursor,native.identity,scope='VALIDATED_PERFORMANCE')
        self.assertEqual({a.trade_id for a in p.allocations},{110,1003,1006})
        rows,execution=E.read_inputs(cursor,p);cursor.recheck_canonical()
        self.assertEqual(execution.unknown,{})
        self.assertEqual(execution.gross[1001],4);self.assertEqual(execution.gross[1002],4)
        self.assertEqual(execution.gross[1003],19);self.assertEqual(execution.gross[1006],78)
        self.assertEqual(execution.gross[110],O.number('-63.38'))
        self.assertLessEqual(cursor.statements+2,130);self.assertLessEqual(cursor.statements,128)
        self.assertLessEqual(max(native.fetch_sizes),16);self.assertTrue(all(not row[0] for row in native.wire))
        self.assertFalse(any(q.startswith(('INSERT ','UPDATE ','DELETE ','ALTER ','CREATE ','REVOKE ')) for q,p in native.calls))
        self.measurement=dict(tracked_statements=cursor.statements,total_with_reserve=cursor.statements+2,rows=cursor.rows,bytes=cursor.bytes)
        print('SYNTHETIC_CONSUMER_BUDGET',self.measurement)

if __name__=='__main__':unittest.main()
