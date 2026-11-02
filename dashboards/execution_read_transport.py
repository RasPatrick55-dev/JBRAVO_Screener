"""Closed canonical read plans and whole-row SQL transport limits. No clients.

Canonical D/A source is unchanged. All their data queries, including duplicate
JSON column names, become ordered JSON ARRAYS with a server-side whole-row and
remaining-total sentinel. No arbitrary query is accepted from a request.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
import time

ROW_BYTES=32768
ENVELOPE_BYTES=128
BODY_BYTES=ROW_BYTES-ENVELOPE_BYTES
MAX_ROWS=4096
MAX_BYTES=16*1024*1024
MAX_SQL=128


def need(value, reason):
    if not value:raise ValueError(reason)


def plan_table():
    result={}
    # Extend only fixed, reviewed SELECT lists. Canonical readers still receive
    # their exact original columns; xmin stays internal to this read guard.
    bindings={
        ' FROM public.trade_disposition_policy ':('xmin::text',),
        ' FROM public.trade_disposition_cases c ':('c.xmin::text','a.xmin::text','t.xmin::text'),
        ' FROM public.trade_allocation_cases c ':('c.xmin::text','a.xmin::text'),
        ' FROM public.trade_allocation_audit a ':('a.xmin::text','c.xmin::text'),
        ' FROM public.trades t ':('t.xmin::text',),
        ' FROM public.alpaca_activities a ':('a.xmin::text',),
        ' FROM public.trade_allocation_fill_claims ':('xmin::text',),
        ' FROM public.trade_allocation_order_claims ':('xmin::text',)}
    for item in CANONICAL_PLANS:
        plan=dict(item);sql=item['sql']
        matches=[(marker,columns) for marker,columns in bindings.items() if marker in sql]
        need(len(matches)<=1,'EXECUTION_PLAN_BINDING_AMBIGUOUS')
        if matches:
            marker,columns=matches[0]
            plan['wire_sql']=sql.replace(marker,','+','.join(columns)+marker,1)
            plan['wire_arity']=plan['arity']+len(columns)
        result[sql]=plan
    return result


def envelope(sql, plan, rows_left, bytes_left):
    need(0<=rows_left<=MAX_ROWS and 0<=bytes_left<=MAX_BYTES,'EXECUTION_BUDGET_INVALID')
    original="SELECT pg_catalog.current_setting('transaction_isolation')" if sql=='SHOW transaction_isolation' else plan.get('wire_sql',sql)
    count=plan.get('wire_arity',plan['arity']);slots=','.join('c'+str(i) for i in range(count))
    values=','.join('q.c'+str(i) for i in range(count))
    order=plan.get('order');ordering='ORDER BY '+order if order else ''
    digest=hashlib.sha256(sql.encode()).hexdigest()
    return (f'/* execution-read:{digest} rows:{rows_left} bytes:{bytes_left} */ '
        f'WITH q({slots}) AS MATERIALIZED (SELECT * FROM ({original}) fixed LIMIT {rows_left+1}), '
        'encoded AS MATERIALIZED (SELECT pg_catalog.row_number() OVER ('+ordering+') AS ordinal,'
        'pg_catalog.json_build_array('+values+')::text AS body FROM q), '
        'checked AS MATERIALIZED (SELECT body,ordinal,pg_catalog.count(*) OVER () AS total_rows,'
        f'pg_catalog.sum(pg_catalog.octet_length(body)+{ENVELOPE_BYTES}) OVER () AS total_bytes FROM encoded) '
        f'SELECT (body IS NULL OR pg_catalog.octet_length(body)>{BODY_BYTES} OR total_rows>{rows_left} '
        f'OR total_bytes>{bytes_left}) AS oversize,'
        f'CASE WHEN body IS NOT NULL AND pg_catalog.octet_length(body)<={BODY_BYTES} AND total_rows<={rows_left} '
        f'AND total_bytes<={bytes_left} THEN body ELSE NULL END AS body,total_rows,total_bytes '
        'FROM checked ORDER BY ordinal')


class Cursor:
    def __init__(self, raw, deadline, *, extra_plans, clock=time.monotonic):
        self.raw,self.deadline,self.clock=raw,deadline,clock
        self.rows=self.bytes=0;self.statements=1 # preceding READ COMMITTED/READ ONLY
        self.plans=plan_table()
        self.canonical_sql={sql for sql,plan in self.plans.items() if plan['kind']!='control'}
        for sql,plan in extra_plans.items():
            if sql not in self.plans:self.plans[sql]=plan
            else:need(plan['arity']==self.plans[sql]['arity'],'EXECUTION_PLAN_BINDING_AMBIGUOUS')
        self.canonical_reads=[];self.rechecking=False
        self.pending=deque();self.arity=0;self.query_rows=0;self.query_bytes=0;self.delivered=0

    def check(self):
        need(self.clock()<=self.deadline,'EXECUTION_DEADLINE_OR_SQL_CAP')

    def execute(self, sql, params=None):
        self.check();need(sql in self.plans,'EXECUTION_UNREVIEWED_SQL')
        remaining=self.deadline-self.clock()
        need(remaining>0 and self.statements+2<=MAX_SQL,'EXECUTION_DEADLINE_OR_SQL_CAP')
        # SET has no result row to discard/undercharge; its bounded integer
        # comes only from the local clock, never request/config text.
        millis=max(1,min(3000,int(remaining*1000)))
        self.raw.execute(f"SET LOCAL statement_timeout = '{millis}ms'");self.statements+=1
        plan=self.plans[sql];self.pending.clear();self.arity=plan.get('arity',0)
        self.wire_arity=plan.get('wire_arity',self.arity)
        self.query_rows=self.query_bytes=self.delivered=0
        if plan['kind']=='control':
            self.raw.execute(sql,params);self.statements+=1;self.check();return
        wrapped=envelope(sql,plan,MAX_ROWS-self.rows,MAX_BYTES-self.bytes)
        self.raw.execute(wrapped,params);self.statements+=1;self.check()
        first=self.raw.fetchmany(1)
        need(type(first) is list and len(first)<=1,'EXECUTION_FETCH_SHAPE')
        if not first:
            self._remember(sql,params);return
        row=first[0];self._wire(row)
        _,_,count,size=row
        # Charge the ENTIRE result before another execute, even when a canonical
        # reader consumes only fetchone and discards remaining native rows.
        need(type(count) is int and type(size) is int and count>0 and size>0
             and self.rows+count<=MAX_ROWS and self.bytes+size<=MAX_BYTES,'EXECUTION_SHARED_BUDGET')
        self.rows+=count;self.bytes+=size;self.query_rows=count;self.query_bytes=size;self.pending.append(row)
        # Drain through bounded native batches before yielding canonical values.
        # This captures every returned row, including fetchone-discarded rows.
        while len(self.pending)<count:
            self.check();batch=self.raw.fetchmany(min(16,count-len(self.pending)))
            need(type(batch) is list and 0<len(batch)<=min(16,count-len(self.pending)),'EXECUTION_FETCH_SHAPE')
            for wire in batch:
                self._wire(wire)
                need(wire[2]==count and wire[3]==size,'EXECUTION_QUERY_COUNT_CHANGED')
                self.pending.append(wire)
        self._remember(sql,params)

    def _remember(self,sql,params):
        if sql in self.canonical_sql and not self.rechecking:
            self.canonical_reads.append((sql,deepcopy(params),tuple(row[1] for row in self.pending)))

    def recheck_canonical(self):
        """Exact relevant query/parameter replay, not database-wide activity."""
        need(not self.rechecking,'EXECUTION_RECHECK_REENTRANT')
        self.rechecking=True
        try:
            for sql,params,expected in self.canonical_reads:
                self.execute(sql,params)
                need(tuple(row[1] for row in self.pending)==expected,'EXECUTION_CANONICAL_INPUT_DRIFT')
        finally:self.rechecking=False

    def _wire(self,row):
        need(isinstance(row,(tuple,list)) and len(row)==4 and row[0] is False
             and isinstance(row[1],str),'EXECUTION_SQL_OVERSIZE_OR_NULL')
        need(len(row[1])<=BODY_BYTES and len(row[1].encode('utf-8'))<=BODY_BYTES,'EXECUTION_ROW_OVERSIZE')

    def _decode(self,row):
        self._wire(row)
        need(row[2]==self.query_rows and row[3]==self.query_bytes,'EXECUTION_QUERY_COUNT_CHANGED')
        try:values=json.loads(row[1])
        except (ValueError,RecursionError):raise ValueError('EXECUTION_ROW_JSON_INVALID') from None
        need(type(values) is list and len(values)==self.wire_arity and
             all(type(value) in (str,int,bool,type(None)) for value in values),'EXECUTION_CANONICAL_TYPES')
        return tuple(values[:self.arity])

    def fetchmany(self,size=16):
        need(type(size) is int and 0<=size<=MAX_ROWS,'EXECUTION_FETCH_LIMIT')
        output=[]
        while len(output)<size:
            self.check()
            if self.pending:batch=[self.pending.popleft()]
            else:
                batch=self.raw.fetchmany(min(16,size-len(output)))
                need(type(batch) is list and len(batch)<=min(16,size-len(output)),'EXECUTION_FETCH_SHAPE')
            if not batch:break
            for row in batch:
                output.append(self._decode(row));self.delivered+=1
                need(self.delivered<=self.query_rows,'EXECUTION_QUERY_COUNT_CHANGED')
        return output

    def fetchall(self):
        rows=[]
        while True:
            batch=self.fetchmany(16)
            if not batch:return rows
            rows.extend(batch)

    def fetchone(self):
        rows=self.fetchmany(1);return rows[0] if rows else None

    @property
    def description(self):
        return tuple(('c'+str(i),None,None,None,None,None,None) for i in range(self.arity))

# Exact source-grounded D.read_policy -> A.read_active read plans follow.

CANONICAL_PLANS = [{'sql': 'SET LOCAL search_path = pg_catalog', 'kind': 'control', 'arity': 0, 'order': None},
 {'sql': 'SHOW transaction_isolation', 'kind': 'plain', 'arity': 1, 'order': None},
 {'sql': "SET LOCAL lock_timeout = '1500ms'", 'kind': 'control', 'arity': 0, 'order': None},
 {'sql': "SET LOCAL statement_timeout = '5000ms'", 'kind': 'control', 'arity': 0, 'order': None},
 {'sql': "SET LOCAL TIME ZONE 'UTC'", 'kind': 'control', 'arity': 0, 'order': None},
 {'sql': 'SELECT '
         "pg_catalog.pg_advisory_xact_lock_shared(pg_catalog.hashtext('jbravo:trade-disposition:policy'))",
  'kind': 'control',
  'arity': 0,
  'order': None},
 {'sql': 'SELECT deployment_id, database_id, database_name, policy_revision, pg_catalog.current_database() '
         'FROM public.trade_disposition_policy WHERE singleton=1',
  'kind': 'canonical',
  'arity': 5,
  'order': None},
 {'sql': 'SELECT c.request::text, c.row_snapshot::text, c.manifest::text, c.case_revision, a.request_digest, '
         'a.policy_revision, pg_catalog.to_jsonb(t)::text, a.request::text, a.result::text FROM '
         'public.trade_disposition_cases c LEFT JOIN public.trades t ON t.trade_id=c.trade_id LEFT JOIN '
         "public.trade_disposition_audit a ON a.approval_ref=c.request->>'approval_ref' WHERE "
         "c.state='ACTIVE' ORDER BY c.trade_id",
  'kind': 'canonical',
  'arity': 9,
  'order': "(q.c0::jsonb->>'trade_id')::bigint"},
 {'sql': 'SELECT '
         "pg_catalog.pg_advisory_xact_lock_shared(pg_catalog.hashtext('jbravo:trade-disposition:policy'))",
  'kind': 'control',
  'arity': 0,
  'order': None},
 {'sql': 'SELECT deployment_id,database_id,database_name,policy_revision,pg_catalog.current_database() FROM '
         'public.trade_disposition_policy WHERE singleton=1',
  'kind': 'canonical',
  'arity': 5,
  'order': None},
 {'sql': 'SELECT '
         'pg_catalog.to_regclass(%s),pg_catalog.to_regclass(%s),pg_catalog.to_regclass(%s),pg_catalog.to_regclass(%s)',
  'kind': 'canonical',
  'arity': 4,
  'order': None},
 {'sql': 'SELECT a.atttypid=%s::pg_catalog.regtype,a.attnotnull FROM pg_catalog.pg_attribute a JOIN '
         'pg_catalog.pg_class c ON c.oid=a.attrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace '
         "WHERE n.nspname='public' AND c.relname='trade_disposition_policy' AND "
         "a.attname='allocation_schema_version' AND a.attnum>0 AND NOT a.attisdropped",
  'kind': 'canonical',
  'arity': 2,
  'order': None},
 {'sql': 'SELECT allocation_schema_version FROM public.trade_disposition_policy WHERE singleton=1',
  'kind': 'canonical',
  'arity': 1,
  'order': None},
 {'sql': 'SELECT c.relname,c.relkind,c.relrowsecurity,c.relforcerowsecurity,EXISTS(SELECT 1 FROM '
         'pg_catalog.pg_inherits i WHERE i.inhrelid=c.oid OR '
         "i.inhparent=c.oid),pg_catalog.has_table_privilege(c.oid,'SELECT') FROM pg_catalog.pg_class c JOIN "
         "pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relname=ANY(%s) "
         'ORDER BY c.relname',
  'kind': 'canonical',
  'arity': 6,
  'order': 'q.c0'},
 {'sql': "SELECT pg_catalog.current_setting('session_replication_role')",
  'kind': 'canonical',
  'arity': 1,
  'order': None},
 {'sql': 'SELECT p.proname,CASE WHEN pg_catalog.octet_length(p.prosrc)<=16384 THEN p.prosrc ELSE NULL '
         "END,p.prosecdef,p.proconfig=ARRAY['search_path=pg_catalog']::pg_catalog.text[],p.proowner=policy.relowner,l.lanname,p.prorettype=%s::pg_catalog.regtype,p.prokind,p.proisstrict,p.proleakproof,p.provolatile,p.proparallel,p.pronargs "
         'FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace JOIN '
         'pg_catalog.pg_language l ON l.oid=p.prolang CROSS JOIN pg_catalog.pg_class policy JOIN '
         "pg_catalog.pg_namespace pn ON pn.oid=policy.relnamespace WHERE n.nspname='public' AND "
         "p.proname=ANY(%s) AND p.pronargs=0 AND pn.nspname='public' AND "
         "policy.relname='trade_disposition_policy' ORDER BY p.proname",
  'kind': 'canonical',
  'arity': 13,
  'order': 'q.c0'},
 {'sql': 'SELECT c.relname,t.tgname,p.proname,n.nspname,t.tgtype,t.tgenabled,t.tgqual IS '
         'NULL,t.tgnargs,pg_catalog.octet_length(t.tgargs),t.tgconstraint=0,t.tgattr::text,(SELECT '
         'a.attnum::text FROM pg_catalog.pg_attribute a WHERE a.attrelid=policy.oid AND '
         "a.attname='allocation_schema_version' AND a.attnum>0 AND NOT a.attisdropped) FROM "
         'pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid JOIN pg_catalog.pg_namespace '
         'cn ON cn.oid=c.relnamespace JOIN pg_catalog.pg_proc p ON p.oid=t.tgfoid JOIN '
         'pg_catalog.pg_namespace n ON n.oid=p.pronamespace CROSS JOIN pg_catalog.pg_class policy JOIN '
         "pg_catalog.pg_namespace pn ON pn.oid=policy.relnamespace WHERE cn.nspname='public' AND "
         "c.relname=ANY(%s) AND t.tgname LIKE %s AND pn.nspname='public' AND "
         "policy.relname='trade_disposition_policy' ORDER BY c.relname,t.tgname",
  'kind': 'canonical',
  'arity': 12,
  'order': 'q.c0,q.c1'},
 {'sql': 'SELECT '
         'c.allocation_id::text,c.state,c.allocation_revision,c.request::text,c.manifest::text,a.request_digest,a.request::text,a.result::text,a.policy_revision '
         'FROM public.trade_allocation_cases c LEFT JOIN public.trade_allocation_audit a ON '
         "a.approval_ref=c.request->>'approval_ref' WHERE c.state IN ('ACTIVE','REVOKED') ORDER BY "
         'c.trade_id LIMIT 17',
  'kind': 'canonical',
  'arity': 9,
  'order': "(q.c3::jsonb->>'trade_id')::bigint"},
 {'sql': 'SELECT a.request_digest,CASE WHEN pg_catalog.octet_length(a.request::text)<=8192 THEN '
         'a.request::text ELSE NULL END,a.manifest=c.manifest,CASE WHEN '
         'pg_catalog.octet_length(a.result::text)<=8192 THEN a.result::text ELSE NULL '
         "END,a.policy_revision,a.approval_ref=a.request->>'approval_ref' FROM public.trade_allocation_audit "
         'a JOIN public.trade_allocation_cases c ON c.allocation_id=a.allocation_id WHERE a.allocation_id=%s '
         "ORDER BY (a.result->>'allocation_revision')::pg_catalog.int8 LIMIT 33",
  'kind': 'canonical',
  'arity': 6,
  'order': "(q.c3::jsonb->>'allocation_revision')::bigint"},
 {'sql': 'LOCK TABLE public.trades,public.alpaca_activities IN ACCESS SHARE MODE',
  'kind': 'control',
  'arity': 0,
  'order': None},
 {'sql': 'SELECT c.relname,c.relkind,c.relrowsecurity,c.relforcerowsecurity,EXISTS(SELECT 1 FROM '
         'pg_catalog.pg_inherits i WHERE i.inhrelid=c.oid OR '
         "i.inhparent=c.oid),pg_catalog.has_table_privilege(c.oid,'SELECT') FROM pg_catalog.pg_class c JOIN "
         "pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relname IN "
         "('trades','alpaca_activities') ORDER BY c.relname",
  'kind': 'canonical',
  'arity': 6,
  'order': 'q.c0'},
 {'sql': 'SELECT CASE WHEN pg_catalog.octet_length(pg_catalog.to_jsonb(t)::text)<=32768 THEN '
         'pg_catalog.to_jsonb(t)::text ELSE NULL END FROM public.trades t WHERE t.trade_id=110 OR '
         't.entry_order_id=ANY(%s) OR t.exit_order_id=ANY(%s) ORDER BY t.trade_id LIMIT 1002',
  'kind': 'canonical',
  'arity': 1,
  'order': "(q.c0::jsonb->>'trade_id')::bigint"},
 {'sql': 'SELECT CASE WHEN pg_catalog.octet_length(pg_catalog.to_jsonb(a)::text)<=32768 THEN '
         'pg_catalog.to_jsonb(a)::text ELSE NULL END FROM public.alpaca_activities a WHERE '
         'a.order_id=ANY(%s) OR a.activity_id=ANY(%s) OR a.activity_pk=ANY(%s) ORDER BY a.activity_pk LIMIT '
         '1002',
  'kind': 'canonical',
  'arity': 1,
  'order': "(q.c0::jsonb->>'activity_pk')::bigint"},
 {'sql': 'SELECT activity_id,activity_type,order_id,quantity::text,allocation_id::text FROM '
         'public.trade_allocation_fill_claims WHERE allocation_id=%s OR activity_id=ANY(%s) OR '
         'order_id=ANY(%s) ORDER BY activity_id LIMIT 7',
  'kind': 'canonical',
  'arity': 5,
  'order': 'q.c0'},
 {'sql': 'SELECT order_id,allocation_id::text FROM public.trade_allocation_order_claims WHERE '
         'allocation_id=%s OR order_id=ANY(%s) ORDER BY order_id LIMIT 5',
  'kind': 'canonical',
  'arity': 2,
  'order': 'q.c0'}]
