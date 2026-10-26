"""Explicit PostgreSQL strategy state and order outbox; no import-time I/O.

Install SCHEMA_SQL only through a separately reviewed migration. Transactions
touch these two new tables and read the existing disposition revision. They
never register fees, modify accounting policy or rewrite financial history.
"""
from contextlib import contextmanager
import json
import re

from scripts.strategy_backtest import digest, require

STATES = {'PREPARED','ENTRY_PENDING','OPEN','EXIT_PENDING','CLOSED','EXPIRED','UNKNOWN','DISPUTED'}
ROLES = {'entry','stop','partial','exit','cancel'}
SCHEMA_SQL = """
CREATE TABLE public.strategy_execution_signals_app (
 account_id TEXT NOT NULL,
 signal_id TEXT NOT NULL CHECK(signal_id ~ '^[0-9a-f]{64}$'),
 symbol TEXT NOT NULL CHECK(symbol ~ '^[A-Z][A-Z0-9.\\-]{0,15}$'),
 state TEXT NOT NULL CHECK(state IN ('PREPARED','ENTRY_PENDING','OPEN','EXIT_PENDING','CLOSED','EXPIRED','UNKNOWN','DISPUTED')),
 revision BIGINT NOT NULL DEFAULT 0 CHECK(revision>=0),
 payload JSONB NOT NULL CHECK(jsonb_typeof(payload)='object'),
 created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(account_id,signal_id)
);
CREATE UNIQUE INDEX strategy_execution_symbol_owner_idx
 ON public.strategy_execution_signals_app(account_id,symbol)
 WHERE state NOT IN ('CLOSED','EXPIRED');
CREATE TABLE public.strategy_execution_orders_app (
 client_order_id TEXT PRIMARY KEY CHECK(client_order_id ~ '^jbr2-[0-9a-f]{40}$'),
 account_id TEXT NOT NULL,
 signal_id TEXT NOT NULL,
 role TEXT NOT NULL CHECK(role IN ('entry','stop','partial','exit','cancel')),
 sequence BIGINT NOT NULL CHECK(sequence>=0),
 operation TEXT NOT NULL CHECK(operation IN ('POST','PATCH','DELETE')),
 request_sha256 TEXT NOT NULL CHECK(request_sha256 ~ '^[0-9a-f]{64}$'),
 request JSONB NOT NULL CHECK(jsonb_typeof(request)='object'),
 lineage JSONB NOT NULL CHECK(jsonb_typeof(lineage)='object'),
 state TEXT NOT NULL CHECK(state IN ('CLAIMED','UNKNOWN','OBSERVED','REJECTED')),
 observation JSONB CHECK(observation IS NULL OR jsonb_typeof(observation)='object'),
 claimed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 UNIQUE(account_id,signal_id,role,sequence),
 FOREIGN KEY(account_id,signal_id) REFERENCES public.strategy_execution_signals_app(account_id,signal_id),
 CHECK((state='OBSERVED')=(observation IS NOT NULL))
);
ALTER TABLE public.strategy_execution_signals_app OWNER TO jbravo_app;
ALTER TABLE public.strategy_execution_orders_app OWNER TO jbravo_app;
"""


def encoded(value, limit=128*1024):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    require(len(raw.encode('utf-8'))<=limit,'execution_state_oversize')
    return raw


class Store:
    """Dedicated READ COMMITTED connection; callers retain no unrelated txn.

    Every write takes the established shared policy lock BEFORE any strategy
    row lock, checks the exact revision, and commits before provider I/O.
    """
    def __init__(self,conn,*,account_id,policy_revision,expected_identity):
        require(conn.autocommit is False and conn.readonly is False and conn.isolation_level==1,
                'execution_store_write_context_required')
        require(isinstance(account_id,str) and 1<=len(account_id)<=128 and
            type(policy_revision) is int and policy_revision>=0,'execution_store_identity_invalid')
        require(isinstance(expected_identity,tuple) and len(expected_identity)==3 and
            all(isinstance(v,str) and bool(v) for v in expected_identity),'execution_database_identity_required')
        self.conn,self.account_id,self.policy_revision=conn,account_id,policy_revision
        self.expected_identity=expected_identity

    @contextmanager
    def transaction(self):
        require(self.conn.get_transaction_status()==0,'execution_store_connection_not_idle')
        try:
            from scripts.db import _policy_write_lock
            _policy_write_lock(self.conn)
            with self.conn.cursor() as cur:
                cur.execute('SELECT deployment_id,database_id,database_name,policy_revision,current_database() '
                    'FROM public.trade_disposition_policy WHERE singleton=1')
                require(cur.fetchone()==(*self.expected_identity,self.policy_revision,self.expected_identity[2]),
                        'execution_policy_revision_changed')
                yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    @contextmanager
    def lease(self):
        """One worker per account; no policy transaction spans network calls."""
        require(self.conn.get_transaction_status()==0,'execution_store_connection_not_idle')
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(hashtext('jbravo:strategy-execution'),hashtext(%s))",(self.account_id,))
            acquired=cur.fetchone()==(True,)
        self.conn.commit()
        require(acquired,'execution_account_worker_busy')
        try:
            yield
        finally:
            self.conn.rollback()
            with self.conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(hashtext('jbravo:strategy-execution'),hashtext(%s))",(self.account_id,))
                require(cur.fetchone()==(True,),'execution_account_lease_lost')
            self.conn.commit()

    def enroll(self,handoff,proposal,reference):
        signal_id=proposal['signal_id']
        require(re.fullmatch('[0-9a-f]{64}',signal_id) is not None,'execution_signal_identity_invalid')
        payload=dict(account_id=self.account_id,handoff=handoff,proposal=proposal,reference=reference,position=None,
            last_completed_session=None,pending_exit_reasons=[],pending_exit_session=None,
            rsi_memory={},stop_order_id=None,stop_sequence=0,exit_sequence=0)
        with self.transaction() as cur:
            cur.execute('INSERT INTO public.strategy_execution_signals_app '
                "(account_id,signal_id,symbol,state,payload) VALUES(%s,%s,%s,'PREPARED',%s::jsonb) "
                'ON CONFLICT(account_id,signal_id) DO NOTHING',
                (self.account_id,signal_id,proposal['symbol'],encoded(payload)))
            cur.execute('SELECT state,revision,payload FROM public.strategy_execution_signals_app '
                'WHERE account_id=%s AND signal_id=%s',(self.account_id,signal_id))
            row=cur.fetchone()
            require(row is not None and row[2]['proposal']==proposal,'execution_signal_changed_after_enrollment')
            return dict(signal_id=signal_id,state=row[0],revision=row[1],payload=row[2])

    def signals(self,*,active=True):
        with self.transaction() as cur:
            cur.execute('SELECT signal_id,state,revision,payload FROM public.strategy_execution_signals_app '
                'WHERE account_id=%s'+(" AND state NOT IN ('CLOSED','EXPIRED')" if active else '')+
                ' ORDER BY created_at,signal_id',(self.account_id,))
            return [dict(signal_id=r[0],state=r[1],revision=r[2],payload=r[3]) for r in cur.fetchall()]

    def save(self,signal,*,state=None,payload=None):
        state=state or signal['state'];payload=signal['payload'] if payload is None else payload
        require(state in STATES,'execution_signal_state_invalid')
        with self.transaction() as cur:
            cur.execute('UPDATE public.strategy_execution_signals_app SET state=%s,payload=%s::jsonb,'
                'revision=revision+1,updated_at=clock_timestamp() WHERE account_id=%s AND signal_id=%s '
                'AND revision=%s RETURNING revision',
                (state,encoded(payload),self.account_id,signal['signal_id'],signal['revision']))
            row=cur.fetchone();require(row is not None,'execution_state_compare_and_swap_failed')
        signal.update(state=state,payload=payload,revision=row[0])
        return signal

    def claim(self,*,signal_id,role,sequence,operation,request,lineage):
        from scripts.strategy_execution import client_order_id
        require(role in ROLES and operation in {'POST','PATCH','DELETE'},'execution_order_role_invalid')
        key=client_order_id(self.account_id,signal_id,role=role,sequence=sequence)
        fingerprint=digest(dict(operation=operation,request=request,lineage=lineage))
        with self.transaction() as cur:
            cur.execute('INSERT INTO public.strategy_execution_orders_app '
                '(client_order_id,account_id,signal_id,role,sequence,operation,request_sha256,request,lineage,state) '
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,'CLAIMED') ON CONFLICT DO NOTHING RETURNING client_order_id",
                (key,self.account_id,signal_id,role,sequence,operation,fingerprint,encoded(request,8192),encoded(lineage,48*1024)))
            fresh=cur.fetchone() is not None
            cur.execute('SELECT request_sha256,state,observation FROM public.strategy_execution_orders_app '
                'WHERE client_order_id=%s AND account_id=%s',(key,self.account_id))
            row=cur.fetchone()
            require(row is not None and row[0]==fingerprint,'execution_existing_order_claim_changed')
            return dict(client_order_id=key,fresh=fresh,state=row[1],observation=row[2])

    def orders(self,signal_id):
        with self.transaction() as cur:
            cur.execute('SELECT client_order_id,role,sequence,operation,request,state,observation '
                'FROM public.strategy_execution_orders_app WHERE account_id=%s AND signal_id=%s '
                'ORDER BY claimed_at,client_order_id',(self.account_id,signal_id))
            return [dict(zip(('client_order_id','role','sequence','operation','request','state','observation'),r))
                    for r in cur.fetchall()]

    def uncertain(self,key):
        with self.transaction() as cur:
            cur.execute("UPDATE public.strategy_execution_orders_app SET state='UNKNOWN',updated_at=clock_timestamp() "
                "WHERE account_id=%s AND client_order_id=%s AND state IN ('CLAIMED','UNKNOWN')",
                (self.account_id,key))
            require(cur.rowcount==1,'execution_unknown_order_transition_failed')

    def rejected(self,key):
        """Only an explicit permanent broker refusal with no matching order."""
        with self.transaction() as cur:
            cur.execute("UPDATE public.strategy_execution_orders_app SET state='REJECTED',updated_at=clock_timestamp() "
                "WHERE account_id=%s AND client_order_id=%s AND state IN ('CLAIMED','UNKNOWN')",
                (self.account_id,key))
            require(cur.rowcount==1,'execution_rejected_order_transition_failed')

    def observed(self,key,order):
        from scripts.strategy_execution import number
        require(isinstance(order,dict) and isinstance(order.get('id'),str) and bool(order['id']),
                'execution_order_observation_invalid')
        keep=('id','client_order_id','symbol','side','type','time_in_force','extended_hours','status','qty',
              'filled_qty','filled_avg_price','limit_price','stop_price','submitted_at','updated_at','filled_at',
              'canceled_at','expired_at','replaced_by','replaces')
        snapshot={k:order[k] for k in keep if k in order}
        with self.transaction() as cur:
            cur.execute('SELECT observation FROM public.strategy_execution_orders_app '
                'WHERE account_id=%s AND client_order_id=%s FOR UPDATE',(self.account_id,key))
            row=cur.fetchone();require(row is not None,'execution_order_claim_missing')
            if row[0] is not None:
                require(row[0]['id']==snapshot['id'] and
                    number(snapshot.get('filled_qty',0))>=number(row[0].get('filled_qty',0)),
                    'execution_broker_observation_regressed')
                if row[0]==snapshot:return
            cur.execute("UPDATE public.strategy_execution_orders_app SET state='OBSERVED',observation=%s::jsonb,"
                'updated_at=clock_timestamp() WHERE account_id=%s AND client_order_id=%s',
                (encoded(snapshot,8192),self.account_id,key))
