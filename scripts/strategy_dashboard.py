"""Authoritative strategy batch projection for dashboard readers; no CSV fallback."""
from datetime import date, datetime
from decimal import Decimal
import math


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


def snapshot():
    from scripts import db
    from scripts.strategy_pipeline import latest_batch
    from scripts.strategy_pipeline_store import candidate_frame
    result=dict(ok=False,status='ERROR',run_date=None,run_ts_utc=None,rows=[],
        rows_final=0,source='postgres',source_detail='strategy_primary:unavailable',entry_ready=False)
    conn=db.get_db_conn()
    if conn is None:
        return result
    try:
        conn.set_session(readonly=True,isolation_level='REPEATABLE READ',autocommit=False)
        with conn.cursor() as cur:
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
        return dict(result,ok=True,status='COMPLETE' if rows else 'EMPTY',
            run_date=batch[0].isoformat(),run_ts_utc=stamp.isoformat(),rows=rows,rows_final=len(rows),
            source_detail='strategy_primary:'+batch[3]['outcome'],archive_id=batch[3]['archive_id'],
            scan_run_id=batch[3]['scan_run_id'],strategy_contract=batch[3].get('strategy_contract'),
            _strategy_backtest=backtest,_strategy_metrics=metrics)
    except Exception:
        return result
    finally:
        conn.rollback();conn.close()


def api_payload(kind, args):
    """Use a single repeatable snapshot for each endpoint, including empty days."""
    result=dict(snapshot())
    backtest=result.pop('_strategy_backtest',None)
    metrics=result.pop('_strategy_metrics',None)
    requested=args.get('run_ts_utc')
    if requested:
        try:
            same=datetime.fromisoformat(str(requested).replace('Z','+00:00')) == datetime.fromisoformat(result['run_ts_utc'])
        except (ValueError,TypeError):same=False
        if not same:
            return dict(result,ok=False,status='ERROR',rows=[],rows_final=0,source_detail='strategy_primary:batch_changed')
    if kind=='metrics':
        if not result.get('ok'):
            return dict(result,rows=[],rows_final=0)
        if metrics is None:
            return dict(result,ok=False,status='METRICS_PENDING',rows=[],rows_final=0,
                source_detail='strategy_primary:no_bound_strategy_metrics')
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
                metric_basis='STRATEGY_RESEARCH_SIMULATION',source_detail='strategy_primary:no_bound_strategy_backtest')
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
            metric_basis='STRATEGY_BUY_SIGNAL'))
    return dict(result,rows=projected,rows_final=len(projected))
