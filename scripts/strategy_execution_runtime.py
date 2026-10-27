"""Opt-in native paper executor and monitor wiring. No import-time effects."""
from contextlib import contextmanager,ExitStack
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import tempfile
import time

from scripts import strategy_execution as rules
from scripts import pipeline_postflight as health
from scripts.strategy_backtest import require

ROOT=Path(__file__).resolve().parents[1]
CONFIG='config/strategy_execution.json'
COMPONENTS=(CONFIG,'scripts/strategy_execution.py','scripts/strategy_execution_handoff.py',
    'scripts/strategy_execution_store.py','scripts/strategy_execution_broker.py',
    'scripts/strategy_execution_engine.py','scripts/strategy_execution_runtime.py',
    'scripts/execute_trades.py','scripts/monitor_positions.py')
LIFECYCLE_COMPONENTS=tuple(p for p in COMPONENTS if p not in {
    'scripts/strategy_execution_handoff.py','scripts/execute_trades.py'})+(
    'scripts/backtest_core.py','scripts/strategy_library.py')
LIFECYCLE_COMPONENTS += ('config/strategy_companions.json','scripts/bull_flag_strategy.py',
                        'scripts/chart_patterns.py','scripts/bull_flag_companion.py')
_FEATURE_CACHE={}


def load_config(base_dir=ROOT):
    value=json.loads(health._read(Path(base_dir)/CONFIG,4096),object_pairs_hook=health._strict_object,
        parse_constant=health._invalid_constant)
    require(value==dict(schema='jbravo.premarket-paper-execution-config.v1',enabled=True,paper_only=True,
        strategy_id='trend_pullback_reclaim_qualifying_close_v2',execution_model=rules.EXECUTION_MODEL,
        allocation_pct=.05,max_positions=4,limit_buffer_pct=.5,cancel_after_minutes=35,
        reference_feed='sip',reference_adjustment='raw',currency='USD',research_qualified=False),
        'strategy_execution_configuration_changed')
    return value


def environment():
    from scripts.daily_strategy_scan import configured_environment
    # Explicit process scope wins over the private file, as it does in the
    # scheduled task's command. Never edit the private file or infer paper mode.
    value={**configured_environment(),**os.environ}
    require(value.get('APCA_API_BASE_URL')=='https://paper-api.alpaca.markets' and
        value.get('JBR_EXEC_PAPER')=='true','strategy_paper_process_context_required')
    return value


@contextmanager
def source_lease(base_dir=ROOT):
    """Coordinate with the existing admin deployment lock, without holding it
    during the premarket wait. Only the native reviewed tools directory is used.
    """
    import fcntl
    path=Path('/home/RasPatrick/jbravo-research-tools/jbravo-remediation-ops/operation.lock')
    from scripts.executor_health import ordinary
    ordinary(path)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
        yield
    finally:os.close(fd)


def write_report(value,*,role,base_dir=ROOT):
    require(role in {'entry','monitor'},'strategy_report_role_invalid')
    from scripts.executor_health import ordinary
    directory=Path(base_dir)/'reports/strategy_execution'/role
    ordinary(directory);directory.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,sort_keys=True,allow_nan=False).encode()
    require(len(raw)<=128*1024,'strategy_execution_report_oversize')
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=directory,delete=False,prefix='.strategy-',suffix='.json') as handle:
            temporary=Path(handle.name);handle.write(raw);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,directory/'latest.json');temporary=None
    finally:
        if temporary is not None:temporary.unlink()


def source_guard(handoff,*,base_dir=ROOT,lifecycle=False):
    from scripts.strategy_pipeline import EXTRA_SOURCES
    pins=handoff['source_bindings']
    require(set(pins)==set((*health.SOURCE_FILES,*EXTRA_SOURCES)),'strategy_execution_sources_not_bound')
    selected=LIFECYCLE_COMPONENTS if lifecycle else pins
    for name in selected:
        pin=pins[name]
        require(health._binding(health._read(Path(base_dir)/name,health.MAX_SOURCE_BYTES))==pin,
                'strategy_execution_source_changed')
    from scripts.strategy_library import get_strategy
    definition=get_strategy(handoff['strategy_contract']['strategy_id']).describe()
    require(definition['definition_sha256']==handoff['strategy_contract']['definition_sha256'],
            'strategy_execution_rule_version_changed')
    return True


def readonly_handoff(env,*,base_dir=ROOT,inspection=False):
    from scripts.research_store import connect
    from scripts.strategy_execution_handoff import load_handoff
    conn=connect(env=env)
    try:return load_handoff(conn,base_dir=base_dir,now=datetime.now(timezone.utc),inspection=inspection)
    finally:conn.rollback();conn.close()


def companion_handoff(env, parent, *, base_dir=ROOT, inspection=False):
    from scripts.bull_flag_companion import configuration, read_handoff
    if configuration(base_dir) is None:return None
    from scripts.research_store import connect
    connection=connect(env=env)
    try:
        return read_handoff(connection,parent,now=datetime.now(timezone.utc),
                            inspection=inspection,base_dir=base_dir)
    finally:connection.rollback();connection.close()


def available_entry_budget(store, next_session):
    # Count durable entry attempts, including rejected/uncertain attempts, across
    # both strategies. Restarting a worker cannot replenish the daily budget.
    used=sum(any(row['role']=='entry' for row in store.orders(signal['signal_id']))
             for signal in store.signals(active=False)
             if signal['payload']['proposal']['next_session']==next_session)
    return max(0,3-used)


def store_connection(env,broker):
    from scripts.research_store import connect
    from scripts import db
    from scripts.strategy_execution_store import Store
    # Execution writes belong to the application's established database role,
    # never the separate research-publisher credential. Start idle/read-only,
    # then explicitly enable the scoped state-table transaction below.
    conn=connect(env={k:v for k,v in env.items() if k!='RESEARCH_DATABASE_URL'})
    conn.set_session(readonly=False,isolation_level='READ COMMITTED',autocommit=False)
    try:
        # Validate the established database/policy/allocation identity on this
        # exact connection before any strategy write or account lease.
        policy=db.trade_policy(conn,scope='ACTIVE_RECONCILIATION')
        with conn.cursor() as cur:cur.execute('SELECT current_database()');name=cur.fetchone()[0]
        identity=(policy.deployment_id,policy.database_id,name)
        revision=policy.revision;conn.commit()
        return conn,Store(conn,account_id=broker.account_id,policy_revision=revision,expected_identity=identity)
    except Exception:
        conn.rollback();conn.close();raise


def guard_for(env,*,base_dir=ROOT):
    def guard(kind,signal):
        require(os.environ.get('APCA_API_BASE_URL')=='https://paper-api.alpaca.markets' and
            os.environ.get('JBR_EXEC_PAPER')=='true','strategy_paper_process_context_changed')
        handoff=signal['payload']['handoff'];source_guard(handoff,base_dir=base_dir,lifecycle=(kind=='lifecycle'))
        if kind=='entry':
            current=readonly_handoff(env,base_dir=base_dir,inspection=True)
            if handoff['strategy_contract']['strategy_id']=='bull_flag_breakout_v1':
                current=companion_handoff(env,current,base_dir=base_dir,inspection=True)
                require(current is not None,'companion_current_batch_unavailable')
            require(current['primary_report_sha256']==handoff['primary_report_sha256'] and
                current['candidate_values']==handoff['candidate_values'] and
                current['batch_sha256']==handoff['batch_sha256'],'strategy_entry_batch_changed_before_post')
            now=datetime.now(timezone.utc);opening=rules.utc(handoff['execution_policy']['entry_open_at'])
            require(opening<=now<opening+timedelta(seconds=60),'strategy_initial_entry_window_expired')
            monitor_ready(handoff,signal['payload']['account_id'],now=now,base_dir=base_dir)
        return True
    return guard


def monitor_ready(handoff,account_id,*,now,base_dir=ROOT):
    value=json.loads(health._read(Path(base_dir)/'reports/strategy_execution/monitor/latest.json',128*1024))
    require(value.get('status')=='ok' and value.get('entry_ready') is True and value.get('paper_only') is True and
        value.get('account_id')==account_id and
        0<=(rules.utc(now)-rules.utc(value['finished_at'])).total_seconds()<=120 and
        value.get('source_bindings')==handoff['source_bindings'],
        'strategy_monitor_not_fresh_and_bound')


@contextmanager
def entry_lease(store,opening,*,wait=time.sleep):
    """A short monitor cycle must not consume the only entry opportunity."""
    with ExitStack() as stack:
        while True:
            try:
                stack.enter_context(store.lease());break
            except ValueError as error:
                if (str(error)!='execution_account_worker_busy' or
                    datetime.now(timezone.utc)>=opening+timedelta(seconds=45)):
                    raise
                wait(.5)
        yield


def features(env,handoff,*,base_dir=ROOT,symbols=None):
    """Reopen the frozen producer archive; never substitute a recent CSV/bar."""
    from scripts.research_store import connect
    from scripts import strategy_pipeline as pipeline
    from scripts.daily_strategy_scan import load_config
    from scripts.strategy_library import get_strategy
    wanted={p['symbol'] for p in handoff['proposals']} | set(symbols or ())
    key=(handoff['batch_sha256'],handoff['strategy_contract']['definition_sha256'],tuple(sorted(wanted)))
    if key in _FEATURE_CACHE:return _FEATURE_CACHE[key]
    conn=connect(env=env)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT specification_sha256,result_sha256,result FROM jbravo_research.research_runs '
                'WHERE run_id=%s',(json.loads(health._read(Path(base_dir)/'reports/pipeline_postflight/latest.json',health.MAX_REPORT_BYTES))['batch']['archive_id'],))
            archived=cur.fetchone()
        value=pipeline.validate_archive(archived,load_config(),handoff['session'],base_dir=base_dir)
    finally:conn.rollback();conn.close()
    dataset=pipeline.verified_dataset(value,archived[0],base_dir=base_dir)
    spec=json.loads((Path(value['output_root'])/'scan/specification.json').read_bytes())
    fold=spec['folds'][0];strategy=get_strategy(handoff['strategy_contract']['strategy_id'])
    # Only held/proposed symbols need monitor features. A broad market scan must
    # not make each monitor cycle prepare thousands of unrelated securities.
    if getattr(dataset,'market_wide',False):
        require(wanted<=set(dataset.raw),'strategy_held_symbol_completed_data_unavailable')
        rows={s:dataset.prepare(s,strategy,fold) for s in wanted}
    else:
        rows={s:strategy.prepare(dataset.raw[s],dataset.features[s],fold) for s in spec['universe']['symbols']}
    _FEATURE_CACHE.clear();_FEATURE_CACHE[key]=rows
    return rows


def run_entry(config,*,base_dir=ROOT,wait=None):
    """Called by the existing task after its normal reconciliation gate."""
    from scripts.strategy_execution_broker import PaperBroker
    from scripts.strategy_execution_engine import Engine
    load_config(base_dir)
    require(config.paper_research_entries is True and config.source=='db' and
        config.reconcile_auto is True and not config.reconcile_only and not config.diagnostic,
        'strategy_entry_process_options_incompatible')
    require(config.allocation_pct==.05 and config.max_positions==4 and config.max_new_positions==3 and
        config.disable_open_position_cap is False and config.min_order_usd==0 and config.allow_bump_to_one is False and
        config.allow_fractional is False and config.submit_at_ny=='04:00' and config.extended_hours is True and
        config.limit_buffer_pct==.5 and config.cancel_after_min==35 and config.chase_enabled is False,
        'strategy_entry_risk_settings_mismatch')
    env=environment();now=datetime.now(timezone.utc)
    # Inspection can report an honest empty/expired batch at any time; the
    # order planner/guard cannot use an inspection handoff to place a buy.
    with source_lease(base_dir):
        inspected=readonly_handoff(env,base_dir=base_dir,inspection=True)
        source_guard(inspected,base_dir=base_dir)
        try:companion=companion_handoff(env,inspected,base_dir=base_dir,inspection=True)
        except Exception:companion=None
        companion_count=len(companion['proposals']) if companion else 0
        if (not inspected['proposals'] and not companion_count) or config.dry_run:
            value=dict(status='skipped',exit_reason='STRATEGY_DRY_RUN_INPUTS_ONLY' if config.dry_run else 'NO_STRATEGY_BUY_SIGNALS',
                candidate_count=len(inspected['proposals'])+companion_count,
                orders_submitted=0,orders_filled=0,handoff_id=inspected['handoff_id'],
                strategy_contract=inspected['strategy_contract'],paper_only=True,broker_operations=0,
                finished_at=now.isoformat(),source_bindings=inspected['source_bindings'],
                companion_handoff_id=companion['handoff_id'] if companion else None,
                candidate_counts_by_strategy={inspected['strategy_contract']['strategy_id']:len(inspected['proposals']),
                                              'bull_flag_breakout_v1':companion_count})
            write_report(value,role='entry',base_dir=base_dir);return value
        handoff=readonly_handoff(env,base_dir=base_dir)
        try:companion=companion_handoff(env,handoff,base_dir=base_dir)
        except Exception:companion=None
        broker=PaperBroker(env,max_requests=192)
        try:
            account=broker.account();clock=broker.clock()
            monitor_ready(handoff,broker.account_id,now=now,base_dir=base_dir)
            proposals=[*handoff['proposals'],*(companion['proposals'] if companion else [])]
            references=broker.references(sorted({p['symbol'] for p in proposals}),handoff['session']['signal_session'],now=now)
        except Exception:broker.close();raise
    try:
        opening=rules.utc(handoff['execution_policy']['entry_open_at'])
        require(0<=(opening-now).total_seconds()<=4500,'strategy_entry_wait_outside_bound')
        sleeper=wait or time.sleep
        while datetime.now(timezone.utc)<opening:
            sleeper(max(0.,min(30.,(opening-datetime.now(timezone.utc)).total_seconds())))
        with source_lease(base_dir):
            source_guard(handoff,base_dir=base_dir)
            account=broker.account();clock=broker.clock();now=datetime.now(timezone.utc)
            monitor_ready(handoff,broker.account_id,now=now,base_dir=base_dir)
            conn,store=store_connection(env,broker)
            try:
                with entry_lease(store,opening,wait=sleeper):
                    engine=Engine(store,broker,guard=guard_for(env,base_dir=base_dir))
                    before={r['client_order_id'] for s in store.signals(active=False) for r in store.orders(s['signal_id'])}
                    engine.enter(handoff,references,now=now,clock=clock,account=account,
                        positions=broker.positions(),open_orders=broker.open_orders(),
                        max_new_entries=available_entry_budget(store,handoff['session']['next_session']))
                    if companion:
                        source_guard(companion,base_dir=base_dir)
                        engine.enter(companion,references,now=now,clock=broker.clock(),account=broker.account(),
                            positions=broker.positions(),open_orders=broker.open_orders(),
                            max_new_entries=available_entry_budget(store,handoff['session']['next_session']))
                    # Observe fills and attach protection promptly. The always-on
                    # monitor resumes durable pending orders and the 35m deadline.
                    deadline=time.monotonic()+15
                    for attempt in range(5):
                        engine.tick(now=datetime.now(timezone.utc))
                        if time.monotonic()>=deadline:break
                        sleeper(1)
                    records=[r for s in store.signals(active=False) for r in store.orders(s['signal_id'])]
                    new=[r for r in records if r['role']=='entry' and r['client_order_id'] not in before]
                    unresolved=any(r['state'] in {'CLAIMED','UNKNOWN'} for r in records)
                    observations=[r['observation'] for r in new if r['observation']]
                    refused=any(e['code']=='BROKER_REJECTED' for e in engine.events) or any(o['status']=='rejected' for o in observations)
                    value=dict(status='blocked' if unresolved or refused else 'ok',
                        exit_reason='ORDER_OUTCOME_UNKNOWN' if unresolved else 'BROKER_REJECTED' if refused else 'STRATEGY_PAPER_EXECUTION',
                        candidate_count=len(proposals),orders_submitted=len(observations),
                        orders_filled=sum(o['status']=='filled' for o in observations),order_observations=observations,
                        handoff_id=handoff['handoff_id'],strategy_contract=handoff['strategy_contract'],
                        primary_report_sha256=handoff['primary_report_sha256'],source_bindings=handoff['source_bindings'],
                        cancellation_deadline=handoff['execution_policy']['cancel_at'],
                        paper_only=True,events=engine.events,finished_at=datetime.now(timezone.utc).isoformat(),
                        preflight=dict(status='pass',candidate_binding=handoff['candidate_values'],
                            entry_authorization='paper_strategy_test',strategy_contract=handoff['strategy_contract']))
                    value['candidate_counts_by_strategy']={handoff['strategy_contract']['strategy_id']:len(handoff['proposals']),
                        'bull_flag_breakout_v1':len(companion['proposals']) if companion else 0}
                    if companion:
                        value['companion']=dict(handoff_id=companion['handoff_id'],
                            primary_report_sha256=companion['primary_report_sha256'],
                            strategy_contract=companion['strategy_contract'],candidate_binding=companion['candidate_values'])
                    write_report(value,role='entry',base_dir=base_dir);return value
            finally:conn.rollback();conn.close()
    finally:broker.close()


def monitor_cycle(*,base_dir=ROOT):
    """Existing always-on monitor calls this only with explicit process opt-in."""
    if os.environ.get('JBR_STRATEGY_EXECUTION')!='true':return dict(owned_symbols=[],native_covered=0)
    from scripts.strategy_execution_broker import PaperBroker
    from scripts.strategy_execution_engine import Engine
    load_config(base_dir);env=environment();broker=PaperBroker(env,max_requests=96)
    conn=None
    try:
        with source_lease(base_dir):
            broker.account();conn,store=store_connection(env,broker)
            with store.lease():
                signals=store.signals()
                # Current report supplies code bindings even between its entry
                # windows. Existing signals retain their original rule binding.
                record=json.loads(health._read(Path(base_dir)/'reports/pipeline_postflight/latest.json',health.MAX_REPORT_BYTES))
                template=dict(source_bindings=record['source_bindings'],strategy_contract=record['batch']['strategy_contract'])
                entry_sources_ready=True
                try:source_guard(template,base_dir=base_dir)
                except ValueError:entry_sources_ready=False
                # A changed dashboard/producer must block new entries, without
                # interrupting protection using unchanged frozen exit code.
                for signal in signals:source_guard(signal['payload']['handoff'],base_dir=base_dir,lifecycle=True)
                engine=Engine(store,broker,guard=guard_for(env,base_dir=base_dir))
                frame=None;completed=None;following=None
                if signals:
                    try:
                        handoff=readonly_handoff(env,base_dir=base_dir,inspection=True)
                        completed=handoff['session']['signal_session'];following=handoff['session']['next_session']
                        if any(s['payload'].get('position') and s['payload']['position']['entry_time']<=completed and
                            s['payload'].get('last_completed_session')!=completed for s in signals):
                            frame=features(env,handoff,base_dir=base_dir,
                                symbols={s['payload']['proposal']['symbol'] for s in signals if s['payload'].get('position')})
                    except (ValueError,OSError):
                        # Retain prior stops when the next completed daily input
                        # is unavailable; never fabricate or tighten on live bars.
                        engine.note(None,'COMPLETED_BAR_INPUT_NOT_AVAILABLE')
                result=engine.tick(now=datetime.now(timezone.utc),frames=frame,
                    completed_session=completed,next_session=following)
                orders=[r for s in store.signals() for r in store.orders(s['signal_id'])]
                failures=any(r['state'] in {'CLAIMED','UNKNOWN'} for r in orders) or any(
                    e['code'] in {'STRATEGY_QUOTES_UNAVAILABLE','POSITION_OWNERSHIP_NOT_RECONCILED',
                        'STOP_REJECTIONS_REQUIRE_REVIEW','STOP_CANCELLATION_OR_FILL_UNCONFIRMED',
                        'ENTRY_CANCELLATION_UNCONFIRMED','POSITION_CHANGED_BEFORE_EXIT','EXIT_QUOTE_UNAVAILABLE',
                        'PARTIAL_STOP_FILL_PENDING','BROKER_REJECTED','COMPLETED_BAR_INPUT_NOT_AVAILABLE'}
                    for e in engine.events)
                current=store.signals()
                positions={s['signal_id']:s['payload'].get('position') for s in current}
                covered_symbols=set()
                regular=broker.clock()['is_open']
                for signal in current:
                    position=positions[signal['signal_id']]
                    if not position:continue
                    for order in store.orders(signal['signal_id']):
                        o=order['observation']
                        if (regular and order['role']=='stop' and o and o['status'] in {'new','accepted'} and
                            rules.number(o['qty'])-rules.number(o.get('filled_qty',0))==position['qty']):
                            covered_symbols.add(position['symbol'])
                covered=len(covered_symbols)
                value=dict(status='blocked' if failures else 'ok',paper_only=True,account_id=broker.account_id,
                    entry_ready=entry_sources_ready and not failures,
                    source_bindings=record['source_bindings'],owned_symbols=result['owned_symbols'],native_covered=covered,
                    events=engine.events,finished_at=datetime.now(timezone.utc).isoformat(),
                    native_stops_trigger_only_regular_hours=True)
                write_report(value,role='monitor',base_dir=base_dir);return value
    except Exception:
        write_report(dict(status='blocked',paper_only=True,account_id=broker.account_id,
            owned_symbols=[],native_covered=0,exit_reason='STRATEGY_MONITOR_UNAVAILABLE',
            finished_at=datetime.now(timezone.utc).isoformat()),role='monitor',base_dir=base_dir)
        raise
    finally:
        if conn is not None:conn.rollback();conn.close()
        broker.close()
