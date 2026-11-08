"""Authoritative strategy batch projection for dashboard readers; no CSV fallback."""
from datetime import date, datetime, time, timezone
from decimal import Decimal
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def failure_code(error):
    """Expose only validation codes, never arbitrary DB/credential error text."""
    value = str(error)
    if isinstance(error, ValueError) and re.fullmatch(
            r'(strategy_|unified_|companion_|execution_)[a-z_]{1,90}', value):
        return value
    return 'strategy_dashboard_read_failed'


def primary_receipt(base_dir=ROOT, *, now=None):
    """Verify an immutable completed run for display, independent of entry age.

    This is historical inspection. Current source and entry freshness remain
    mandatory in check_report/prepare_handoff and are never relaxed there.
    """
    from scripts import pipeline_postflight as health, strategy_pipeline as pipeline
    now = now or datetime.now(timezone.utc)
    folder = Path(base_dir) / 'reports/pipeline_postflight'
    raw = health._read(folder / 'latest.json', health.MAX_REPORT_BYTES)
    primary = json.loads(raw, object_pairs_hook=health._strict_object,
                         parse_constant=health._invalid_constant)
    finished = health._utc(primary['finished_at'])
    pipeline.require(finished <= now, 'strategy_postflight_future')
    # Validate the run's contract as of completion, not as a new order proposal.
    pipeline.assess(primary, now=finished, max_age_seconds=43200, inspection=True)
    pipeline.require(pipeline.enabled(base_dir), 'strategy_primary_route_disabled')
    pipeline.require(health._read(folder / ('run-' + primary['run_id'] + '.json'),
                                  health.MAX_REPORT_BYTES) == raw, 'strategy_postflight_history_changed')
    active = json.loads(health._read(folder / 'active-primary.json', 4096))
    pipeline.require(primary['invocation_id'] == active['id'] == primary['run_id'] and
        health._utc(primary['started_at']) >= health._utc(active['started_at']),
        'strategy_primary_invocation_mismatch')
    binding = health._binding(raw)
    pipeline.require(json.loads(health._read(folder / 'completed-primary.json', 4096)) ==
        dict(id=active['id'], report_binding=binding), 'strategy_primary_completion_mismatch')
    return primary, raw


def display_selection(connection, batch, primary_run_id, *, base_dir=ROOT, now=None):
    """Reconstruct the exact PostgreSQL selection without conferring entry authority."""
    from scripts import setup_selection as selection, strategy_pipeline as pipeline
    from scripts.bull_flag_companion import _row, primary_id
    from scripts.research_strategies import CHALLENGER
    from scripts import pipeline_postflight as health
    now = now or datetime.now(timezone.utc)
    primary, raw = primary_receipt(base_dir, now=now)
    pipeline.require(primary['run_id'] == primary_run_id and primary['batch'] == batch[3] and
        batch[0].isoformat() == primary['session']['signal_session'] and
        batch[1] == datetime.fromisoformat(primary['candidate_run_ts_utc']),
        'unified_dashboard_primary_changed')
    verified = selection.read_selection(connection, selection.parent_context(primary), now=now, base_dir=base_dir)
    control = json.loads(health._read(Path(base_dir) /
        'reports/strategy_companions/unified_setups/latest.json', 65536))
    pipeline.require(control.get('status') == 'COMPLETED_UNIFIED_SETUP_SELECTION' and
        control.get('run_id') == verified['record']['run_id'] and
        control.get('primary_run_id') == primary_run_id and
        control.get('parent_report_sha256') == selection.digest(primary) and
        control.get('result_sha256') == verified['result_sha256'],
        'strategy_postflight_unified_selection_unavailable')
    reports = {CHALLENGER: dict(backtest=primary['backtest'], metrics=primary['metrics'])}
    for sid in verified['handoffs']:
        if sid != CHALLENGER:
            record = _row(connection, primary_id(primary_run_id, sid))[2]
            reports[sid] = dict(backtest=record['backtest'], metrics=record['metrics'])
    pipeline.require(health._read(Path(base_dir) / 'reports/pipeline_postflight/latest.json',
        health.MAX_REPORT_BYTES) == raw, 'execution_primary_changed_after_database_read')
    verified.update(reports=reports, primary=primary)
    return verified


def handoff_readiness(verified, candidate_rows, *, base_dir=ROOT, now=None):
    """Candidate preparation readiness only; broker checks still precede every order."""
    from scripts import strategy_pipeline as pipeline, strategy_execution as execution
    from scripts import pipeline_postflight as health
    now = now or datetime.now(timezone.utc)
    primary = verified['primary']
    opening = datetime.combine(date.fromisoformat(primary['session']['next_session']),
                               time(4), execution.NY).astimezone(timezone.utc)
    reasons = []
    if now >= opening:
        reasons.append('execution_premarket_preparation_expired')
    try:
        raw = health._read(Path(base_dir) / 'reports/pipeline_postflight/latest.json', health.MAX_REPORT_BYTES)
        pipeline.require(json.loads(raw) == primary, 'unified_dashboard_primary_changed')
        pipeline.check_report(primary, raw, base_dir, now=now, max_age_seconds=43200,
                              not_before=None, require_completion=True, inspection=True)
        if not reasons:
            execution.prepare_handoff(primary, candidate_rows, now=now)
    except Exception as error:
        reasons.append(failure_code(error))
    if not verified['record']['selection']['selected_setups']:
        reasons.append('unified_selection_no_buy_signals')
    return dict(selection_ready=True, entry_ready=not reasons,
        entry_block_reason=reasons[0] if reasons else None, entry_block_reasons=reasons,
        entry_readiness_scope='PREPARED_PAPER_HANDOFF_ONLY', broker_preflight_required=True,
        order_submission_authorized=False, paper_only=True)


def clean(value):
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (date,datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def computation_completion(verified):
    """Display proof of completed computation, never research or entry approval.

    Only call after display_selection has checked immutable primary receipts,
    the completed invocation, and the bound PostgreSQL unified selection.
    """
    return dict(completion_status='verified', computation_complete=True,
        completion_scope='STRATEGY_COMPUTATION_ONLY',
        completion_evidence=dict(source='immutable_primary_and_postgres_unified_selection',
            primary_run_id=verified['primary']['run_id'],
            selection_run_id=verified['record']['run_id'],
            finished_at=verified['primary']['finished_at']),
        research_qualified=False, operating_acceptance=False,
        order_submission_authorized=False)


def snapshot(*, base_dir=ROOT, now=None):
    from scripts import db
    from scripts.strategy_pipeline import latest_batch
    from scripts.strategy_pipeline_store import candidate_frame
    result=dict(ok=False,status='ERROR',run_date=None,run_ts_utc=None,rows=[],
        rows_final=0,source='postgres',source_detail='strategy_primary:unavailable',entry_ready=False,
        completion_status='unverified',computation_complete=False,
        completion_scope='STRATEGY_COMPUTATION_ONLY',completion_evidence=None,
        research_qualified=False,operating_acceptance=False,order_submission_authorized=False)
    conn=None
    try:
        conn=db.get_db_conn()
        if conn is None:
            return dict(result, failure_code='strategy_database_unavailable')
        conn.set_session(readonly=True,isolation_level='REPEATABLE READ',autocommit=False)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='5000ms'")
            cur.execute("SET LOCAL lock_timeout='1000ms'")
            batch=latest_batch(cur)
        if batch is None:
            return dict(result,status='PENDING',source_detail='strategy_primary:no_completed_batch')
        frame,stamp=candidate_frame(conn,batch[0],exact_run_ts=batch[1])
        if stamp != batch[1] or len(frame) != batch[2]:
            raise ValueError('strategy_dashboard_population_mismatch')
        rows=clean(frame.to_dict('records'))
        backtest=batch[3].get('backtest')
        if backtest is not None:
            from scripts.strategy_pipeline import validate_backtest, selected_batch
            with conn.cursor() as cur:
                cur.execute('SELECT run_id FROM public.screener_strategy_batches_app WHERE run_ts_utc=%s',(batch[1],))
                primary=cur.fetchone()
            if primary is None:raise ValueError('backtest_primary_batch_missing')
            validate_backtest(backtest,selected_batch(batch[3]),primary[0])
        metrics=batch[3].get('metrics')
        if metrics is not None:
            if backtest is None:raise ValueError('strategy_metrics_backtest_missing')
            from scripts.strategy_metrics import validate_metrics
            validate_metrics(metrics,backtest,selected_batch(batch[3]),primary[0])
        from scripts.setup_selection import dashboard_rows
        unified=display_selection(conn,batch,primary[0],base_dir=base_dir,now=now)
        readiness=handoff_readiness(unified,frame.to_dict('records'),base_dir=base_dir,now=now)
        rows=dashboard_rows(unified)
        return dict(result,ok=True,status='COMPLETE' if rows else 'EMPTY',
            setup_selection=unified['record']['selection'],
            _unified_reports=unified['reports'],_unified_selection=unified['record']['selection'],
            unified_selection_run_id=unified['record']['run_id'],strategy_priority=False,
            run_date=batch[0].isoformat(),run_ts_utc=stamp.isoformat(),rows=rows,rows_final=len(rows),
            source_detail='unified_strategy_screen:BEST_SETUPS',archive_id=unified['record']['run_id'],
            coverage=clean(batch[3].get('coverage')),
            scan_run_id=unified['record']['run_id'],strategy_contract=dict(strategy_id='ALL_STRATEGIES',
                strategy_version='1',definition_sha256=unified['record']['selection']['model_sha256']),
            _strategy_backtest=backtest,_strategy_metrics=metrics,**dict(readiness, **computation_completion(unified)))
    except Exception as error:
        return dict(result, failure_code=failure_code(error))
    finally:
        if conn is not None:
            try:conn.rollback()
            finally:conn.close()


def api_payload(kind, args, *, base_dir=ROOT, now=None):
    """Use a single repeatable snapshot for each endpoint, including empty days."""
    result=dict(snapshot(base_dir=base_dir, now=now))
    backtest=result.pop('_strategy_backtest',None)
    metrics=result.pop('_strategy_metrics',None)
    unified=result.pop('_unified_selection',None)
    unified_reports=result.pop('_unified_reports',None)
    requested=args.get('run_ts_utc')
    if requested:
        try:
            same=datetime.fromisoformat(str(requested).replace('Z','+00:00')) == datetime.fromisoformat(result['run_ts_utc'])
        except (ValueError,TypeError):same=False
        if not same:
            return dict(result,ok=False,status='ERROR',rows=[],rows_final=0,source_detail='strategy_primary:batch_changed',
                completion_status='unverified',computation_complete=False,completion_evidence=None)
    if unified is not None and kind in {'metrics','backtest'}:
        from scripts.setup_selection import dashboard_reports
        return dashboard_reports(kind,result,unified,unified_reports,args)
    if kind=='metrics':
        if not result.get('ok'):
            return dict(result,rows=[],rows_final=0)
        if metrics is None:
            return dict(result,ok=False,status='METRICS_PENDING',rows=[],rows_final=0,
                source_detail='strategy_primary:no_bound_strategy_metrics',
                completion_status='unverified',computation_complete=False,completion_evidence=None)
        contract=metrics['strategy_contract'];rows=list(metrics['rows'])
        query=str(args.get('q') or '').strip().upper()
        if query:rows=[r for r in rows if query in (r['symbol']+' '+r['strategy_id']+' '+r['sample_status']).upper()]
        if args.get('filter')=='with_history':rows=[r for r in rows if r['trades']>0]
        elif args.get('filter')=='no_history':rows=[r for r in rows if r['trades']==0]
        elif args.get('filter') not in (None,'','all'):
            return dict(result,ok=False,status='ERROR',rows=[],rows_final=0,source_detail='strategy_metrics:unsupported_filter')
        try:limit=max(1,min(200,int(args.get('limit') or 50)))
        except (ValueError,TypeError):limit=50
        rows=rows[:limit]
        return dict(result,rows=rows,rows_final=len(rows),metrics_status=metrics['status'],
            metric_basis=metrics['metric_basis'],metrics_result_id=metrics['result_id'],
            backtest_result_id=metrics['backtest_result_id'],strategy_id=contract['strategy_id'],
            strategy_version=contract['strategy_version'],definition_sha256=contract['definition_sha256'],
            evaluation=metrics['evaluation'],source_detail='strategy_metrics:'+metrics['status'])
    if kind=='backtest':
        if backtest is None:
            return dict(result,ok=False,status='BACKTEST_PENDING',rows=[],rows_final=0,
                metric_basis='STRATEGY_RESEARCH_SIMULATION',source_detail='strategy_primary:no_bound_strategy_backtest',
                completion_status='unverified',computation_complete=False,completion_evidence=None)
        contract=backtest['strategy_contract'];rows=[]
        for r in backtest['results']:
            rows.append(dict(r,signal_id=r['current_signal_id'],trades=r['closed_signals'],
                wins=r['winning_signals'],losses=r['losing_signals'],win_rate=r['win_rate_pct'],
                net_pnl=r['closed_trade_net_pnl'],expectancy=r['expectancy_net_per_closed_signal'],
                max_drawdown=r['max_closed_signal_drawdown_cash'],drawdown_units='USD_CLOSED_SIGNAL_PNL'))
        query=str(args.get('q') or '').strip().upper()
        if query:rows=[r for r in rows if query in r['symbol']]
        try:limit=max(1,min(200,int(args.get('limit') or 50)))
        except (ValueError,TypeError):limit=50
        rows=rows[:limit]
        return dict(result,rows=rows,rows_final=len(rows),backtest_status=backtest['status'],
            strategy_id=contract['strategy_id'],strategy_version=contract['strategy_version'],
            definition_sha256=contract['definition_sha256'],backtest_result_id=backtest['result_id'],
            window='STRATEGY_EPISODE',evaluation_folds=backtest['parameters']['folds'],
            evaluation_end=backtest['parameters']['effective_test_end'],
            metric_basis='STRATEGY_RESEARCH_SIMULATION',source_detail='strategy_backtest:'+backtest['status'])
    rows=result['rows'];query=str(args.get('q') or '').strip().upper()
    if query:
        rows=[r for r in rows if query in r['symbol'] or query in str(r['score_breakdown']).upper()]
    if args.get('filter') in ('errors','gate_failures','data_issues','high_confidence'):
        rows=[]  # Only selected valid signals enter this batch.
    try:limit=max(1,min(200,int(args.get('limit') or 50)))
    except (ValueError,TypeError):limit=50
    if args.get('filter')=='top10':limit=min(10,limit)
    projected=[]
    for rank,r in enumerate(rows[:limit],1):
        proposal=r['score_breakdown']['proposal']
        projected.append(dict(r,rank=rank,screened_at_utc=r.get('timestamp'),price=r['close'],
            dollar_volume=r['close']*r['volume'],final_score=r['score'],
            signal_id=proposal['signal_id'],strategy_id=proposal['strategy_id'],
            strategy_version=proposal['strategy_version'],next_open_at=proposal['next_open_at'],
            metric_basis='UNIFIED_SETUP_BUY_SIGNAL' if unified is not None else 'STRATEGY_BUY_SIGNAL',
            matching_strategies=r['score_breakdown'].get('matching_strategies',[proposal['strategy_id']])))
    return dict(result,rows=projected,rows_final=len(projected))
