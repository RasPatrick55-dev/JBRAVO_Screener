"""Real isolated PostgreSQL tests; never connect to a configured bot database."""
from copy import deepcopy
import os

import pytest

from scripts.strategy_execution_store import Store,SCHEMA_SQL

IDENTITY=("isolated-execution-test","00000000-0000-0000-0000-000000000099","jbravo_execution_test")

pytestmark=pytest.mark.alpaca_optional


@pytest.fixture(scope='module')
def database():
    dsn=os.environ.get('JBR_TEST_EXECUTION_DSN')
    if not dsn:pytest.skip('Explicit isolated execution test PostgreSQL is not running')
    import psycopg2
    conn=psycopg2.connect(dsn)
    params=conn.get_dsn_parameters()
    assert params['host']=='127.0.0.1' and params['port']=='55439'
    assert params['dbname']==params['user']=='jbravo_execution_test'
    conn.set_session(readonly=False,isolation_level='READ COMMITTED',autocommit=False)
    with conn.cursor() as cur:
        cur.execute("DO $$ BEGIN IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='jbravo_app') THEN CREATE ROLE jbravo_app; END IF; END $$")
        cur.execute('DROP TABLE IF EXISTS public.trade_disposition_policy')
        cur.execute('CREATE TABLE public.trade_disposition_policy(singleton INTEGER PRIMARY KEY,deployment_id TEXT,database_id TEXT,database_name TEXT,policy_revision BIGINT NOT NULL)')
        cur.execute('DROP TABLE IF EXISTS public.strategy_execution_orders_app,public.strategy_execution_signals_app')
        cur.execute(SCHEMA_SQL)
    conn.commit();conn.close()
    yield dsn


@pytest.fixture
def connection(database):
    import psycopg2
    conn=psycopg2.connect(database)
    conn.set_session(readonly=False,isolation_level='READ COMMITTED',autocommit=False)
    with conn.cursor() as cur:
        cur.execute('TRUNCATE public.strategy_execution_orders_app,public.strategy_execution_signals_app,public.trade_disposition_policy')
        cur.execute('INSERT INTO public.trade_disposition_policy VALUES(1,%s,%s,%s,2)',IDENTITY)
    conn.commit()
    yield conn
    conn.rollback();conn.close()


def signal(store):
    return store.enroll(dict(primary_run_id='a'*32),dict(signal_id='b'*64,symbol='SRRK'),dict(price=52.44))


def claim(store,**changes):
    return store.claim(**(dict(signal_id='b'*64,role='entry',sequence=0,operation='POST',
        request=dict(symbol='SRRK',qty=10),lineage=dict(strategy_id='pullback'))|changes))


def test_claim_is_committed_before_a_second_connection_or_provider_can_observe_it(connection,database):
    import psycopg2
    store=Store(connection,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY);signal(store)
    first=claim(store);assert first['fresh'] is True
    second=psycopg2.connect(database)
    try:
        with second.cursor() as cur:
            cur.execute('SELECT state,request FROM public.strategy_execution_orders_app WHERE client_order_id=%s',
                (first['client_order_id'],))
            assert cur.fetchone()==('CLAIMED',dict(symbol='SRRK',qty=10))
    finally:second.rollback();second.close()
    assert claim(store)['fresh'] is False
    with pytest.raises(ValueError,match='existing_order_claim_changed'):
        claim(store,request=dict(symbol='SRRK',qty=11))


def test_lost_response_persists_unknown_across_a_new_process_connection(connection,database):
    import psycopg2
    store=Store(connection,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY);signal(store)
    order=claim(store);store.uncertain(order['client_order_id'])
    other=psycopg2.connect(database);other.set_session(readonly=False,isolation_level='READ COMMITTED')
    try:
        resumed=Store(other,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY)
        result=claim(resumed)
        assert result['fresh'] is False and result['state']=='UNKNOWN'
    finally:other.close()


def test_signal_ownership_and_compare_and_swap_are_durable(connection):
    store=Store(connection,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY)
    current=signal(store);stale=deepcopy(current)
    store.save(current,state='OPEN')
    with pytest.raises(ValueError,match='compare_and_swap_failed'):store.save(stale,state='CLOSED')
    assert store.signals()[0]['state']=='OPEN'
    # Primary reruns retain the first handoff instead of permitting a second buy.
    again=store.enroll(dict(primary_run_id='c'*32),dict(signal_id='b'*64,symbol='SRRK'),dict(price=52.))
    assert again['payload']['handoff']['primary_run_id']=='a'*32
    with pytest.raises(ValueError,match='signal_changed'):
        store.enroll({},dict(signal_id='b'*64,symbol='PLUG'),{})


def test_policy_revision_drift_rolls_back_before_a_strategy_write(connection):
    store=Store(connection,account_id='paper-account',policy_revision=1,expected_identity=IDENTITY)
    with pytest.raises(ValueError,match='policy_revision_changed'):signal(store)
    with connection.cursor() as cur:
        cur.execute('SELECT count(*) FROM public.strategy_execution_signals_app');assert cur.fetchone()==(0,)
    connection.rollback()


def test_account_lease_blocks_concurrent_workers_and_releases_after_exception(connection,database):
    import psycopg2
    store=Store(connection,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY)
    other=psycopg2.connect(database);other.set_session(readonly=False,isolation_level='READ COMMITTED')
    second=Store(other,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY)
    try:
        with pytest.raises(RuntimeError,match='test-failure'):
            with store.lease():
                with pytest.raises(ValueError,match='worker_busy'):
                    with second.lease():pass
                raise RuntimeError('test-failure')
        with second.lease():pass
    finally:other.close()


def test_observations_cannot_regress_fill_quantity_or_change_broker_identity(connection):
    store=Store(connection,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY);signal(store)
    key=claim(store)['client_order_id']
    order=dict(id='broker-order',client_order_id=key,symbol='SRRK',qty='10',filled_qty='5',status='partially_filled')
    store.observed(key,order)
    for change in (dict(filled_qty='4'),dict(id='different-order')):
        with pytest.raises(ValueError,match='observation_regressed'):store.observed(key,order|change)
    observed=store.orders('b'*64)[0]
    assert observed['observation']['filled_qty']=='5' and observed['observation']['id']=='broker-order'


def test_store_refuses_unrelated_open_transaction_and_readonly_connection(connection):
    store=Store(connection,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY)
    with connection.cursor() as cur:cur.execute('SELECT 1')
    with pytest.raises(ValueError,match='connection_not_idle'):signal(store)
    connection.rollback();connection.set_session(readonly=True)
    with pytest.raises(ValueError,match='write_context_required'):
        Store(connection,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY)


def test_explicit_broker_rejection_is_durable_and_cannot_create_a_fresh_claim(connection):
    store=Store(connection,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY);signal(store)
    key=claim(store)['client_order_id'];store.rejected(key)
    assert claim(store)['fresh'] is False and claim(store)['state']=='REJECTED'


def test_actual_worker_recovers_lost_acceptance_from_postgresql_without_second_buy(connection):
    from tests.test_strategy_execution_engine import Broker,OPEN,positive,NOW
    from scripts import strategy_execution as rules
    from scripts.strategy_execution_engine import Engine
    from tests.test_strategy_execution import reference
    from datetime import timedelta
    store=Store(connection,account_id='paper-account',policy_revision=2,expected_identity=IDENTITY)
    record,rows=positive();handoff=rules.prepare_handoff(record,rows,now=NOW+timedelta(seconds=10))
    broker=Broker();broker.mode='lost'
    worker=Engine(store,broker,guard=lambda *a:True,now=lambda:broker.now)
    with store.lease():
        worker.enter(handoff,{handoff['proposals'][0]['symbol']:reference()},now=OPEN,clock=broker.clock(),
            account=broker.account(),positions={},open_orders=[])
    assert len(broker.posts)==1 and store.signals()[0]['state']=='UNKNOWN'
    broker.mode=None;broker.fill('entry',49)
    resumed=Engine(store,broker,guard=lambda *a:True,now=lambda:broker.now)
    with store.lease():resumed.tick(now=broker.now)
    saved=store.signals()[0]
    assert len(broker.posts)==2 and broker.posts[-1]['type']=='stop' and saved['state']=='OPEN'
    assert saved['payload']['handoff']['metrics_result_id']==handoff['metrics_result_id']
