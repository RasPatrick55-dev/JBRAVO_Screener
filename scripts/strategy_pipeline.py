"""Primary Screener adapter for verified strategy scans; no broker order capability.

The primary scheduler consumes the earlier daily SIP scan, publishes one atomic
candidate batch, and freezes an identity-preserving handoff. An empty batch is a
successful screen, never permission to borrow yesterday's symbols. Strategy
simulation metrics stay in research; they are not broker performance records.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
NY = ZoneInfo('America/New_York')
STRATEGY = 'trend_pullback_reclaim_qualifying_close_v2'
SCHEMA = 'jbravo.primary-strategy-screen.v1'
EXTRA_SOURCES = ('scripts/strategy_pipeline.py', 'config/strategy_pipeline.json',
                 'config/strategy_scan_daily.json', 'scripts/daily_strategy_scan.py',
                 'scripts/strategy_pipeline_store.py', 'scripts/strategy_dashboard.py',
                 'scripts/strategy_backtest.py', 'scripts/strategy_metrics.py')
EXTRA_SOURCES += tuple('scripts/'+name for name in (
    'strategy_scanner.py', 'strategy_library.py', 'research_dataset.py',
    'research_testing.py', 'research_strategies.py', 'research_simulation.py',
    'research_store.py', 'backtest_core.py', 'indicators.py',
    'strategy_universe.py', 'strategy_market_replay.py'))
EXTRA_SOURCES += ('config/strategy_execution.json',)+tuple('scripts/'+name for name in (
    'strategy_execution.py','strategy_execution_handoff.py','strategy_execution_store.py',
    'strategy_execution_broker.py','strategy_execution_engine.py','strategy_execution_runtime.py',
    'execute_trades.py','monitor_positions.py'))
EXTRA_SOURCES += ('config/strategy_companions.json', 'scripts/bull_flag_strategy.py',
                  'scripts/chart_patterns.py', 'scripts/bull_flag_companion.py',
                  'scripts/pattern_strategy.py','scripts/swing_patterns.py','config/chart_pattern_companions.json')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def enabled(base_dir=ROOT):
    path = Path(base_dir) / 'config/strategy_pipeline.json'
    if not path.exists():
        return False
    from scripts.pipeline_postflight import _read, _strict_object, _invalid_constant
    cfg = json.loads(_read(path, 4096), object_pairs_hook=_strict_object, parse_constant=_invalid_constant)
    require(cfg == dict(schema='jbravo.primary-strategy-route.v1', enabled=True,
        strategy_id=STRATEGY,
        producer_root='/home/RasPatrick/jbravo-research-tools/STRATEGY_DAILY_ACTIVATION_20261025_002/runs',
        execution_model='NEXT_SUPPLIED_REGULAR_SESSION_OPEN', order_submission_authorized=False),
        'invalid_primary_strategy_configuration')
    return True


def session_context(calendar, now):
    """Use a fresh exchange calendar, including intervening holidays and DST."""
    from scripts.daily_strategy_scan import normalize_calendar
    rows = normalize_calendar(calendar)
    require(now.tzinfo is not None, 'ambiguous_strategy_clock')
    today = now.astimezone(NY).date()
    available = [r for r in rows if datetime.combine(date.fromisoformat(r['date']), time(20, 15), NY) <= now]
    require(bool(available), 'no_completed_strategy_session')
    day = available[-1]['date']
    following = [r for r in rows if r['date'] > day]
    require(bool(following) and today - timedelta(days=7) <= date.fromisoformat(day) <= today,
            'missing_or_stale_strategy_calendar')
    signal, upcoming = available[-1], following[0]
    require(time.fromisoformat(signal['open']) == time(9, 30) and time.fromisoformat(signal['close']) == time(16),
            'exceptional_signal_session_requires_review')
    require(time.fromisoformat(upcoming['open']) == time(9, 30), 'unsupported_next_regular_open')
    next_open = datetime.combine(date.fromisoformat(upcoming['date']), time(9, 30), NY)
    require(now < next_open, 'strategy_signal_open_already_passed')
    return dict(signal_session=day, next_session=upcoming['date'],
        signal_available_at=datetime.combine(date.fromisoformat(day), time(20, 15), NY).isoformat(),
        next_open_at=next_open.astimezone(timezone.utc).isoformat(),
        execution_model='NEXT_SUPPLIED_REGULAR_SESSION_OPEN')


def validate_archive(row, cfg, session, *, base_dir=ROOT, strategy_id=STRATEGY):
    """Verify the DB envelope before trusting any filesystem path or proposal."""
    from scripts.daily_strategy_scan import PIPELINE
    from scripts.research_dataset import canonical_hash
    from scripts.strategy_library import get_strategy
    strategy = get_strategy(strategy_id)
    from scripts.strategy_scanner import SOURCE_FILES
    require(row is not None and len(row) == 3, 'strategy_scan_not_archived_for_session')
    spec_sha, result_sha, value = row
    require(isinstance(value, dict) and digest(encoded(value)) == result_sha.strip(), 'strategy_archive_hash_mismatch')
    day = session['signal_session']
    expected_id = canonical_hash(dict(schema='jbravo.daily-strategy-archive-id.v1', pipeline=PIPELINE,
        strategy_id=strategy_id, signal_session=day))
    require(value.get('schema') == 'jbravo.daily-strategy-scan.v1' and
        value.get('status') == 'COMPLETED_PAPER_SCREEN' and value.get('pipeline') == PIPELINE and
        value.get('run_id') == expected_id and value.get('signal_session') == day and
        value.get('strategy_id') == strategy_id and value.get('configuration_sha256') == canonical_hash(cfg),
        'strategy_archive_scope_mismatch')
    require(all(value.get(k) is False for k in
        ('research_qualified', 'execution_authorized', 'order_submission_authorized')), 'strategy_archive_authority_mismatch')
    summary, handoff = value['summary'], value['handoff']
    require(summary.get('status') == 'COMPLETED_PAPER_SIMULATION' and summary.get('broker_operations') == 0 and
        summary.get('as_of') == day and summary.get('run_id') == value['scan_run_id'] and
        summary.get('research_qualified') is False and summary.get('order_submission_authorized') is False,
        'incomplete_strategy_scan')
    require(handoff.get('schema') == 'jbravo.strategy-paper-proposals.v1' and handoff.get('as_of') == day and
        handoff.get('scan_run_id') == value['scan_run_id'] and handoff.get('paper_only') is True and
        handoff.get('order_submission_authorized') is False and
        handoff.get('proposals') == summary.get('top_candidates'), 'strategy_handoff_mismatch')
    current_sources = {n: digest((Path(base_dir)/'scripts'/n).read_bytes()) for n in SOURCE_FILES}
    require(summary.get('source_sha256') == current_sources, 'strategy_scan_source_changed')
    strategies = summary.get('strategies')
    require(isinstance(strategies, list) and len(strategies) == 1 and
        encoded(strategies[0]['strategy']) == encoded(strategy.describe()) and
        strategies[0]['as_of'] == day, 'strategy_definition_changed')
    require(re.fullmatch('[0-9a-f]{64}', spec_sha.strip()) is not None, 'invalid_strategy_specification_binding')
    proposals = handoff['proposals']
    require(isinstance(proposals, list) and len(proposals) <= cfg['portfolio']['top_n'], 'strategy_proposal_limit')
    seen = set()
    for p in proposals:
        from scripts.strategy_universe import market_config, SYMBOL
        allowed = (isinstance(p['symbol'], str) and SYMBOL.fullmatch(p['symbol']) is not None
                   if market_config(cfg) else p['symbol'] in cfg['symbols'])
        require(allowed and p['symbol'] not in seen, 'invalid_strategy_symbol_population')
        seen.add(p['symbol'])
        expected_signal = canonical_hash(dict(strategy_id=strategy_id, version=strategy.version,
            episode_id=strategies[0]['episode_id'], symbol=p['symbol'], signal_session=day))
        require(p['strategy_id'] == strategy_id and p['strategy_version'] == strategy.version and
            p['signal_id'] == expected_signal and p['side'] == 'BUY' and p['signal_session'] == day and
            p['next_session'] == session['next_session'] and
            datetime.fromisoformat(p['next_open_at']) == datetime.fromisoformat(session['next_open_at']) and
            datetime.fromisoformat(p['signal_available_at_assumed']) == datetime.fromisoformat(session['signal_available_at']) and
            p['execution_model'] == session['execution_model'] and p['status'] == 'PAPER_PROPOSAL' and
            p['paper_only'] is True and p['order_submission_authorized'] is False and
            p['allocation_pct'] == cfg['portfolio']['alloc_pct'] and p['max_positions'] == cfg['portfolio']['max_positions'],
            'strategy_proposal_identity_mismatch')
        for key in ('score', 'signal_close_raw', 'signal_atr14_raw'):
            require(type(p[key]) in (int, float) and math.isfinite(p[key]) and
                (key == 'score' or p[key] > 0), 'invalid_strategy_proposal_number')
    return value


def verified_dataset(value, spec_sha, *, base_dir=ROOT, strategy_id=STRATEGY):
    from scripts.pipeline_postflight import _read
    from scripts.research_dataset import load_dataset, strict_json
    from scripts.research_testing import validate_spec
    from scripts.daily_strategy_scan import load_config
    from scripts.research_dataset import canonical_hash
    from scripts.strategy_universe import market_config
    cfg = load_config(Path(base_dir)/'config/strategy_scan_daily.json')
    cfg = dict(cfg, strategy_id=strategy_id)
    market = market_config(cfg)
    root = Path(value['output_root'])
    expected = Path(json.loads((Path(base_dir)/'config/strategy_pipeline.json').read_bytes())['producer_root']) / value['signal_session']
    if strategy_id != STRATEGY:
        from scripts.pattern_strategy import PATTERNS
        require(strategy_id in PATTERNS, 'unknown_companion_strategy')
        expected = expected/'companions'/strategy_id
    require(root == expected, 'strategy_archive_path_mismatch')
    raw = _read(root/'scan/manifest.json', 65536)
    require(digest(raw) == value['scan_manifest_sha256'], 'strategy_scan_manifest_changed')
    bindings = strict_json(raw)['bindings']
    require(1 <= len(bindings) <= 100, 'strategy_artifact_count_limit')
    seen, total = set(), 0
    for pin in bindings:
        relative = Path(pin['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and pin['path'] not in seen,
                'strategy_artifact_path_rejected')
        seen.add(pin['path'])
        body = _read(root/'scan'/relative, (64 if market else 16)*1024*1024)
        total += len(body)
        require(total <= (256 if market else 32)*1024*1024 and len(body) == pin['bytes'] and digest(body) == pin['sha256'],
                'strategy_artifact_changed')
    require({'result.json', 'paper_handoff.json', 'specification.json', 'dataset.json'} <= seen,
            'strategy_manifest_incomplete')
    require(strict_json(_read(root/'scan/result.json', 1024*1024)) == value['summary'] and
        strict_json(_read(root/'scan/paper_handoff.json', 1024*1024)) == value['handoff'], 'strategy_saved_result_changed')
    spec_limit = 2*1024*1024 if market else 65536
    spec_raw = _read(root/'scan/specification.json', spec_limit)
    require(digest(spec_raw) == spec_sha.strip() and spec_raw == _read(root/'study.json', spec_limit),
            'strategy_specification_changed')
    spec = strict_json(spec_raw)
    validate_spec(spec)
    if market:
        require(set(spec['universe'])=={'symbols','benchmark','excluded'} and
            spec['universe']['benchmark']==cfg['benchmark'] and spec['universe']['excluded']=={},
            'strategy_dynamic_universe_scope_changed')
    else:
        require(spec['universe']==dict(symbols=cfg['symbols'],benchmark=cfg['benchmark'],excluded={}),
            'strategy_scan_universe_changed')
    require(spec['portfolio'] == cfg['portfolio'] and spec['strategies'] == [strategy_id] and
        spec['cost_cases'] == [cfg['cost_case']] and spec['folds'] == [dict(id=cfg['episode_id'],
        warmup_start=cfg['warmup_start'],test_start=cfg['episode_start'],test_end=value['signal_session'])],
        'strategy_scan_specification_scope_changed')
    capture_root = root if strategy_id == STRATEGY else root.parents[1]
    require(Path(spec['capture_manifest']) == capture_root/'capture/analysis/manifest.json' and
        Path(spec['actions_capture']['path']) == capture_root/'actions.json', 'strategy_capture_path_mismatch')
    dataset = load_dataset(spec)
    if market:
        require(getattr(dataset,'market_wide',False) and dataset.coverage['policy']==cfg['universe'] and
            value['summary'].get('coverage')==dataset.provenance['selection'] and
            all(p['symbol'] in spec['universe']['symbols'] for p in value['handoff']['proposals']),
            'strategy_market_coverage_changed')
    require(dataset.provenance['dataset_sha256'] == value['summary']['dataset_sha256'], 'strategy_dataset_changed')
    expected_scan = canonical_hash(dict(spec=spec, as_of=value['signal_session'], sources=value['summary']['source_sha256'],
        dataset_sha256=dataset.provenance['dataset_sha256'],
        strategies=[value['summary']['strategies'][0]['strategy']['definition_sha256']], cost_case=cfg['cost_case']['id']))
    require(expected_scan == value['scan_run_id'], 'strategy_scan_batch_identity_mismatch')
    return dataset


def candidate_rows(value, dataset, run_ts):
    """Keep strategy score and stable signal identity; never fill from legacy ranks."""
    rows = []
    day = value['signal_session']
    for p in value['handoff']['proposals']:
        raw = dataset.raw[p['symbol']].loc[day]
        require(float(raw['close']) == p['signal_close_raw'], 'strategy_reference_price_changed')
        metadata = dict(schema='jbravo.strategy-candidate-lineage.v1', proposal=p,
            archive_id=value['run_id'], archive_sha256=digest(encoded(value)),
            scan_run_id=value['scan_run_id'], dataset_sha256=value['summary']['dataset_sha256'],
            definition_sha256=value['summary']['strategies'][0]['strategy']['definition_sha256'])
        rows.append(dict(run_date=date.fromisoformat(day), symbol=p['symbol'], score=p['score'],
            final_score=p['score'], exchange=None, close=p['signal_close_raw'], volume=int(raw['volume']),
            universe_count=len(dataset.raw)-1, score_breakdown=metadata, entry_price=p['signal_close_raw'],
            adv20=None, atrp=p['signal_atr14_raw']/p['signal_close_raw']*100,
            source='strategy_pullback_reclaim_v2', timestamp=datetime.combine(date.fromisoformat(day), time.min, NY),
            passed_gates=True, gate_fail_reason=None, ml_weight_used=0.0, run_ts_utc=run_ts))
    return rows


def latest_batch(cursor, *, run_date=None, exact_run_ts=None):
    where, args = [], []
    if run_date is not None:
        where.append('run_date=%s'); args.append(run_date)
    if exact_run_ts is not None:
        where.append('run_ts_utc=%s'); args.append(exact_run_ts)
    cursor.execute('SELECT run_date,run_ts_utc,candidate_count,payload FROM public.screener_strategy_batches_app' +
        (' WHERE '+' AND '.join(where) if where else '')+' ORDER BY run_date DESC,run_ts_utc DESC LIMIT 1', args)
    return cursor.fetchone()


def selection_batch(value, session):
    definition=value['summary']['strategies'][0]['strategy']
    proposals=value['handoff']['proposals']
    result = dict(schema='jbravo.strategy-primary-batch.v1', archive_id=value['run_id'],
        archive_sha256=digest(encoded(value)), scan_run_id=value['scan_run_id'],
        session=session, proposals=proposals, candidate_count=len(proposals),
        strategy_contract=dict(strategy_id=definition['id'],strategy_version=definition['version'],
            definition_sha256=definition['definition_sha256']),
        outcome='BUY_SIGNALS' if proposals else 'NO_BUY_SIGNALS',
        research_qualified=False, order_submission_authorized=False)
    if 'coverage' in value['summary']:result['coverage']=value['summary']['coverage']
    return result


def selected_batch(payload):
    return {k:v for k,v in payload.items() if k not in ('backtest','metrics')}


def publish_batch(conn, value, rows, session, invocation, run_ts, *, backtest, metrics):
    """Single commit for header + canonical candidates + top candidates, even at zero."""
    from scripts import db
    from scripts.research_testing import json_default
    proposals = value['handoff']['proposals']
    require(len(rows) == len(proposals) and
        [(r['symbol'],r['score_breakdown']['proposal']) for r in rows] == [(p['symbol'],p) for p in proposals] and
        all(r['run_ts_utc'] == run_ts and r['run_date'].isoformat() == session['signal_session'] for r in rows),
        'strategy_publication_population_mismatch')
    require(not conn.autocommit, 'strategy_publication_requires_transaction')
    conn.set_session(isolation_level='READ COMMITTED', readonly=False)
    selected=selection_batch(value,session)
    validate_backtest(backtest,selected,invocation)
    from scripts.strategy_metrics import validate_metrics
    validate_metrics(metrics,backtest,selected,invocation)
    payload=dict(selected,backtest=backtest,metrics=metrics)
    try:
        db._policy_write_lock(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('jbravo:primary:strategy-screen'))")
            previous = latest_batch(cur)
            if previous:
                require(previous[0].isoformat() <= value['signal_session'] and previous[1] < run_ts,
                        'strategy_publication_out_of_order')
                if previous[0].isoformat() == value['signal_session']:
                    require(previous[3]['archive_sha256'] == payload['archive_sha256'], 'strategy_session_revision_requires_review')
            for row in rows:
                names = list(row)
                values = [json.dumps(row[n], default=json_default) if n == 'score_breakdown' else row[n] for n in names]
                cur.execute('INSERT INTO public.screener_candidates ('+','.join('"'+n+'"' for n in names)+') VALUES ('+
                    ','.join(['%s']*len(names))+') ON CONFLICT (run_date,symbol) DO UPDATE SET '+
                    ','.join('"'+n+'"=EXCLUDED."'+n+'"' for n in names if n not in ('run_date','symbol')),
                    values)
                cur.execute('INSERT INTO public.top_candidates (run_date,symbol,score,entry_price,adv20,atrp,source) '
                    'VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(run_date,symbol) DO UPDATE SET '
                    'score=EXCLUDED.score,entry_price=EXCLUDED.entry_price,adv20=EXCLUDED.adv20,atrp=EXCLUDED.atrp,source=EXCLUDED.source,'
                    'win_rate=NULL,net_pnl=NULL,trades=NULL,wins=NULL,losses=NULL,expectancy=NULL,profit_factor=NULL,'
                    'max_drawdown=NULL,sharpe=NULL,sortino=NULL',
                    [row[k] for k in ('run_date','symbol','score','entry_price','adv20','atrp','source')])
            cur.execute('INSERT INTO public.screener_strategy_batches_app '
                '(run_id,run_date,run_ts_utc,archive_id,archive_sha256,candidate_count,payload) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)',
                (invocation, value['signal_session'], run_ts, value['run_id'], payload['archive_sha256'], len(rows), encoded(payload).decode()))
            check = latest_batch(cur, run_date=value['signal_session'], exact_run_ts=run_ts)
            require(check is not None and check[2] == len(rows) and check[3] == payload, 'strategy_batch_precommit_readback_failed')
            verify_candidate_readback(cur, rows, value['signal_session'], run_ts)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return payload


def validate_backtest(backtest, selected, invocation):
    from scripts.strategy_backtest import digest as backtest_digest
    require(isinstance(backtest,dict) and backtest.get('schema')=='jbravo.candidate-strategy-backtest.v1',
            'missing_strategy_backtest')
    core={k:v for k,v in backtest.items() if k!='result_id'}
    count=selected['candidate_count']
    require(backtest_digest(core)==backtest.get('result_id') and
        backtest['mode']=='REPORT_ONLY' and backtest['primary_run_id']==invocation and
        backtest['screener_batch_sha256']==backtest_digest(selected) and
        backtest['strategy_contract']==selected['strategy_contract'] and
        backtest['candidate_signal_ids']==[p['signal_id'] for p in selected['proposals']] and
        backtest['input_candidates']==backtest['output_candidates']==count and
        backtest['signal_session']==selected['session']['signal_session'] and
        backtest['status']==('COMPLETED_CANDIDATE_BACKTEST' if count else 'SKIPPED_NO_CANDIDATES') and
        len(backtest['results'])==count and backtest['replay_performed']==bool(count) and
        backtest['order_submission_authorized'] is False and backtest['broker_operations']==0,
        'strategy_backtest_lineage_or_result_mismatch')


def verify_candidate_readback(cur, expected, day, run_ts):
    cur.execute('SELECT symbol,score_breakdown FROM public.latest_screener_candidates '
        'WHERE run_date=%s AND run_ts_utc=%s ORDER BY symbol', (day, run_ts))
    actual = cur.fetchall()
    require(actual == sorted([(r['symbol'], r['score_breakdown']) for r in expected]), 'strategy_candidate_readback_failed')
    cur.execute('SELECT symbol FROM public.latest_top_candidates WHERE run_date=%s AND run_ts_utc=%s ORDER BY symbol', (day, run_ts))
    require(cur.fetchall() == sorted([(r['symbol'],) for r in expected]), 'strategy_top_candidate_readback_failed')


def main(argv, *, base_dir=ROOT):
    """Called from the main pipeline, after its primary invocation is claimed."""
    import argparse
    from scripts import pipeline_postflight as health
    from scripts.daily_strategy_scan import Provider, CALENDAR, configured_environment, load_config
    from scripts.research_store import connect
    from scripts.research_dataset import canonical_hash
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--handoff-invocation', required=True)
    args, unused = parser.parse_known_args(argv)
    from scripts.primary_pipeline import FIXED_ARGS
    require(unused in (list(FIXED_ARGS), [*FIXED_ARGS, '--screener-args',
        '--feed iex --dollar-vol-min 2000000 --reuse-cache true']), 'unsupported_strategy_primary_arguments')
    require(health.provisional_context() is None, 'strategy_primary_provisional_publication_refused')
    require(re.fullmatch('[0-9a-f]{32}', args.handoff_invocation) is not None, 'strategy_primary_invocation_required')
    require(enabled(base_dir), 'strategy_primary_route_disabled')
    started = datetime.now(timezone.utc)
    sources = {n: health._binding(health._read(Path(base_dir)/n, health.MAX_SOURCE_BYTES))
        for n in (*health.SOURCE_FILES, *EXTRA_SOURCES)}
    env = configured_environment(); cfg = load_config()
    provider = Provider(env.get('APCA_API_KEY_ID'), env.get('APCA_API_SECRET_KEY'), max_gets=1)
    try:
        calendar, _ = provider.get(CALENDAR, dict(start=(started.astimezone(NY).date()-timedelta(days=10)).isoformat(),
            end=(started.astimezone(NY).date()+timedelta(days=10)).isoformat()))
    finally:
        provider.close()
    session = session_context(calendar, started)
    archive_id = canonical_hash(dict(schema='jbravo.daily-strategy-archive-id.v1', pipeline='pullback-reclaim-daily-paper-v1',
        strategy_id=STRATEGY, signal_session=session['signal_session']))
    conn = connect(env=env)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT specification_sha256,result_sha256,result FROM jbravo_research.research_runs WHERE run_id=%s', (archive_id,))
            archived = cur.fetchone()
        value = validate_archive(archived, cfg, session, base_dir=base_dir)
    finally:
        conn.rollback(); conn.close()
    dataset = verified_dataset(value, archived[0], base_dir=base_dir)
    rows = candidate_rows(value, dataset, started)
    from scripts.strategy_backtest import run_candidate_backtest
    spec_raw=(Path(value['output_root'])/'scan/specification.json').read_bytes()
    require(digest(spec_raw)==archived[0].strip(),'strategy_backtest_specification_changed')
    spec=json.loads(spec_raw)
    backtest=run_candidate_backtest(selection_batch(value,session),primary_run_id=args.handoff_invocation,
        dataset=dataset,specification=spec)
    from scripts.strategy_metrics import build_metrics
    metrics=build_metrics(selection_batch(value,session),backtest,primary_run_id=args.handoff_invocation)
    conn = connect(write=True, env=env)
    try:
        batch = publish_batch(conn, value, rows, session, args.handoff_invocation, started,backtest=backtest,metrics=metrics)
    finally:
        conn.close()
    # A new read-only transaction verifies the committed candidate population.
    conn = connect(env=env)
    try:
        with conn.cursor() as cur:
            require(latest_batch(cur, exact_run_ts=started)[3] == batch, 'strategy_batch_committed_readback_failed')
            verify_candidate_readback(cur, rows, session['signal_session'], started)
    finally:
        conn.rollback(); conn.close()
    finished = datetime.now(timezone.utc)
    record = dict(schema=SCHEMA, run_id=args.handoff_invocation, invocation_id=args.handoff_invocation,
        started_at=started.isoformat(), finished_at=finished.isoformat(), pipeline_rc=0,
        session=session, batch=batch, candidate_run_ts_utc=started.isoformat(), source_bindings=sources,
        handoff=value['handoff'], backtest=backtest, metrics=metrics,
        stages=dict(screener='COMPLETED', backtest=backtest['status'],
            metrics=metrics['status'], labels='NOT_USED_BY_STRATEGY', ranker_eval='NOT_USED_BY_STRATEGY',
            handoff='COMMITTED_AND_SEPARATE_READONLY_VERIFIED'),
        legacy_screener_options_applied=False, broker_operations=0, entry_ready=False,
        order_submission_authorized=False, research_qualified=False)
    assess(record, now=finished)
    for relative, pin in sources.items():
        require(health._binding(health._read(Path(base_dir)/relative, health.MAX_SOURCE_BYTES)) == pin,
                'strategy_primary_source_changed_during_run')
    health._save_control(base_dir, 'run-'+record['run_id']+'.json', record)
    health._save_control(base_dir, 'latest.json', record)
    from scripts.bull_flag_companion import prepare_primary
    try:prepare_primary(record, env=env, base_dir=base_dir)
    except Exception as error:
        print(json.dumps(dict(status='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED',
                              error_type=type(error).__name__)),flush=True)
    from scripts.swing_patterns import STRATEGIES as extra_patterns
    for strategy_id in extra_patterns:
        try:prepare_primary(record, env=env, base_dir=base_dir, strategy_id=strategy_id)
        except Exception as error:
            print(json.dumps(dict(status='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED',strategy_id=strategy_id,
                                  error_type=type(error).__name__)),flush=True)
    print(json.dumps(dict(status='PRIMARY_STRATEGY_SCREEN_COMPLETE', strategy_id=STRATEGY,
        signal_session=session['signal_session'], candidates=len(rows), outcome=batch['outcome'],
        scan_run_id=value['scan_run_id'], archive_id=archive_id, broker_operations=0)), flush=True)
    return 0


def assess(record, *, now, max_age_seconds=7200, not_before=None, inspection=False):
    """A separate screen/handoff contract; never invent legacy ML or P&L success."""
    from scripts.pipeline_postflight import _utc
    require(record.get('schema') == SCHEMA and re.fullmatch('[0-9a-f]{32}', record['run_id']) is not None,
            'invalid_strategy_postflight')
    started, finished = _utc(record['started_at']), _utc(record['finished_at'])
    require(started <= finished <= now and 0 <= (now-finished).total_seconds() <= max_age_seconds and
        (not_before is None or started >= not_before), 'strategy_postflight_stale_or_future')
    session, batch, handoff = record['session'], record['batch'], record['handoff']
    selected=selected_batch(batch)
    validate_backtest(record['backtest'],selected,record['run_id'])
    require(record['backtest']==batch['backtest'],'strategy_backtest_postflight_mismatch')
    from scripts.strategy_metrics import validate_metrics
    validate_metrics(record['metrics'],record['backtest'],selected,record['run_id'])
    require(record['metrics']==batch['metrics'],'strategy_metrics_postflight_mismatch')
    require(type(inspection) is bool and _utc(session['signal_available_at']) <= started and
        finished < _utc(session['next_open_at']) and
        (inspection or now < _utc(session['next_open_at'])), 'strategy_handoff_expired_or_early')
    proposals = handoff['proposals']
    require(batch['session'] == session and batch['proposals'] == proposals and
        type(batch['candidate_count']) is int and batch['candidate_count'] == len(proposals) and
        batch['outcome'] == ('BUY_SIGNALS' if proposals else 'NO_BUY_SIGNALS') and
        batch['scan_run_id'] == handoff['scan_run_id'] and handoff['as_of'] == session['signal_session'] and
        handoff['paper_only'] is True and handoff['order_submission_authorized'] is False,
        'strategy_postflight_batch_mismatch')
    require(record['pipeline_rc'] == 0 and record['broker_operations'] == 0 and record['entry_ready'] is False and
        record['order_submission_authorized'] is False and record['research_qualified'] is False and
        record['stages'] == dict(screener='COMPLETED', backtest=record['backtest']['status'],
        metrics=record['metrics']['status'], labels='NOT_USED_BY_STRATEGY', ranker_eval='NOT_USED_BY_STRATEGY',
        handoff='COMMITTED_AND_SEPARATE_READONLY_VERIFIED'), 'strategy_postflight_scope_mismatch')
    return dict(status='ok', failures=[], qualifications=[], run_id=record['run_id'],
        route='pullback_reclaim_strategy', pipeline_ready=True, screen_ready=True,
        entry_ready=False, entry_block_reason='strategy_regular_open_executor_not_connected',
        research_qualified=False, operating_acceptance=False, session=session,
        candidate_count=len(proposals), outcome=batch['outcome'])


def check_report(record, data, base_dir, *, now, max_age_seconds, not_before, require_completion, inspection=False):
    from scripts import pipeline_postflight as h
    result = assess(record, now=now, max_age_seconds=max_age_seconds, not_before=not_before,inspection=inspection)
    require(enabled(base_dir), 'strategy_primary_route_disabled')
    for relative in (*h.SOURCE_FILES, *EXTRA_SOURCES):
        require(record['source_bindings'].get(relative) == h._binding(h._read(Path(base_dir)/relative, h.MAX_SOURCE_BYTES)),
                'strategy_postflight_source_changed')
    directory = Path(base_dir)/'reports/pipeline_postflight'
    require(h._read(directory/('run-'+record['run_id']+'.json'), h.MAX_REPORT_BYTES) == data,
            'strategy_postflight_history_changed')
    active = json.loads(h._read(directory/'active-primary.json', 4096))
    require(record['invocation_id'] == active['id'] == record['run_id'] and
        h._utc(record['started_at']) >= h._utc(active['started_at']), 'strategy_primary_invocation_mismatch')
    result['report_binding'] = h._binding(data)
    if require_completion:
        require(json.loads(h._read(directory/'completed-primary.json', 4096)) ==
            dict(id=active['id'], report_binding=result['report_binding']), 'strategy_primary_completion_mismatch')
    return result
