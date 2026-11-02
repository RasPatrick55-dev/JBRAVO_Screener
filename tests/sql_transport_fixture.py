"""Synthetic PostgreSQL transport model; canonical D/A Python logic is real.

This does NOT execute or validate SQL on PostgreSQL. It independently models
the reviewed wire envelope, including server redaction before native fetch.
"""
import hashlib
import json
import re
from copy import deepcopy
from dashboards import execution_gross_projection as E, execution_read_transport as T
from scripts import trade_disposition as D
from test_public_dashboard_integration import IDENTITY


class Cursor:
    description=[]
    def __init__(self,conn):self.conn=conn;self.rows=[]
    def __enter__(self):return self
    def __exit__(self,*a):return False
    def execute(self,sql,params=None):
        self.conn.calls.append((sql,params));self.rows=[]
        if sql.startswith('SET TRANSACTION'):
            self.conn.isolation='repeatable read' if 'REPEATABLE' in sql else 'read committed'
            return
        if sql.startswith('SET ') or sql.startswith('LOCK '):return
        if 'pg_advisory_xact_lock_shared' in sql and not sql.startswith('/*'):return
        match=re.match(r'/\* execution-read:([a-f0-9]{64}) rows:(\d+) bytes:(\d+) \*/ ',sql)
        assert match, 'Unexpected unguarded data query'
        plans=T.plan_table();plans.update(E.READ_PLANS)
        original=next(q for q in plans if hashlib.sha256(q.encode()).hexdigest()==match[1])
        assert 'AS MATERIALIZED' in sql and 'json_build_array(' in sql
        assert 'CASE WHEN body IS NOT NULL' in sql and 'ELSE NULL END AS body' in sql
        assert 'sum(pg_catalog.octet_length(body)+128) OVER ()' in sql
        left,byte_left=int(match[2]),int(match[3])
        native=self.native(original,params)[:left+1]
        # Independently model physical-row versions exposed only to the guard.
        binding_columns={
            ' FROM public.trade_disposition_policy ':('xmin::text',),
            ' FROM public.trade_disposition_cases c ':('c.xmin::text','a.xmin::text','t.xmin::text'),
            ' FROM public.trade_allocation_cases c ':('c.xmin::text','a.xmin::text'),
            ' FROM public.trade_allocation_audit a ':('a.xmin::text','c.xmin::text'),
            ' FROM public.trades t ':('t.xmin::text',),
            ' FROM public.alpaca_activities a ':('a.xmin::text',),
            ' FROM public.trade_allocation_fill_claims ':('xmin::text',),
            ' FROM public.trade_allocation_order_claims ':('xmin::text',)}
        for marker,columns in binding_columns.items():
            if original in T.plan_table() and marker in original:
                assert ','+','.join(columns)+marker in sql
                target=getattr(self.conn,'canonical_version_scope',None)
                changed=self.conn.source_reads>=2 and (getattr(self.conn,'canonical_version_drift',False) or (target is not None and target in original))
                version='301' if changed else '300'
                native=[tuple(row)+(version,)*len(columns) for row in native]
        bodies=[json.dumps(list(row),ensure_ascii=False,separators=(',',':')) for row in native]
        total=sum(len(body.encode())+128 for body in bodies)
        self.rows=[(len(body.encode())>32640 or len(native)>left or total>byte_left,
                    body if len(body.encode())<=32640 and len(native)<=left and total<=byte_left else None,
                    len(native),total) for body in bodies]
        self.conn.wire.extend(deepcopy(self.rows))

    def native(self,sql,params):
        c=self.conn
        if sql in getattr(c,'overrides',{}):return c.overrides[sql]
        if sql=='SHOW transaction_isolation':return [(getattr(c,'force_isolation',c.isolation),)]
        if 'FROM public.trade_disposition_policy WHERE singleton=1' in sql:
            c.policy_reads+=1
            revision=3 if getattr(c,'revision_drift',False) and c.policy_reads>=3 else 2
            return [(*IDENTITY,'fixture_db',revision,getattr(c,'current_db','fixture_db'))]
        if 'FROM public.trade_disposition_cases c LEFT JOIN' in sql:return getattr(c,'cases',[])
        if 'to_regclass' in sql:return [(None,None,None,None)]
        if 'a.attnotnull' in sql:return []
        if sql==E.VISIBILITY_SQL:
            return getattr(c,'visibility',[('alpaca_activities','r',False,False,False,False,True),('trades','r',False,False,False,False,True)])
        if sql==E.SCHEMA_SQL:
            return sorted((table,name,sorted(kinds)[0]) for table,cols in E.REQUIRED_COLUMNS.items() for name,kinds in cols.items())
        if sql in (E.TRADES_SQL,E.OWNERS_SQL,E.FILLS_SQL):
            if sql==E.TRADES_SQL:
                c.source_reads+=1
                if getattr(c,'unrelated_audit_commit',False):
                    c.audit_revision=1;c.database_snapshot='100:201:'
            field='records' if sql==E.TRADES_SQL else 'owners' if sql==E.OWNERS_SQL else 'fills'
            rows=deepcopy(getattr(c,field))
            version='101' if c.source_reads>1 and (getattr(c,'version_drift',False) or getattr(c,'version_drift_table',None)==field) else '100'
            if c.source_reads>1 and getattr(c,'row_drift',False) and field=='records':rows[0]['qty']='3'
            if c.source_reads>1 and getattr(c,'fill_drift',False) and field=='fills':rows[-1]['price']='12'
            if c.source_reads>1 and getattr(c,'owner_drift',False) and field=='owners':rows.append(dict(rows[0],trade_id=9999))
            return [(D.canonical(dict(row=row,version=version)),) for row in rows]
        raise AssertionError('Unimplemented fixed canonical SQL transport')

    def fetchmany(self,size=16):
        assert 0<=size<=16
        self.conn.fetch_sizes.append(size)
        result,self.rows=self.rows[:size],self.rows[size:]
        return result
