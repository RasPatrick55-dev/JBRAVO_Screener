"""One source-bound setup ranking across all enabled strategies; inert on import."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path

from scripts.strategy_backtest import require,digest

ROOT=Path(__file__).resolve().parents[1]
MODEL='COMMON_TECHNICAL_COMPOSITE_V1'
POLICY='UNIFIED_BEST_SETUPS_V1'


def model_sha256():
    return digest({n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest()
                   for n in ('setup_selection.py','backtest_core.py')})


def contract():
    return dict(strategy_id='ALL_STRATEGIES',strategy_version='1',definition_sha256=model_sha256())


def enrich(frame,split):
    """Apply the identical existing technical score, in split-price units.

    Detector-specific scores stay intact for their historical replay. MACD and
    signal are converted back from raw units; all other composite features
    retain the shared indicator preparation's split-price basis.
    """
    from scripts.backtest_core import composite_score
    result=frame.copy();features=frame.copy()
    factor=frame.close/split.close.reindex(frame.index)
    for name in ('open','high','low','close','volume'):
        features[name]=split[name].reindex(frame.index)
    for name in ('macd','macd_signal'):
        features[name]=frame[name]/factor
    result['setup_score']=composite_score(features)
    result['setup_volume_ratio']=split.volume/split.volume.shift(1).rolling(20).mean()
    return result


def evidence(row,session):
    # Legacy synthetic replay fixtures can omit enrichment. They cannot enter
    # a unified selection: rank_setups requires this evidence on every proposal.
    if 'setup_score' not in row:return {}
    value=dict(model=MODEL,model_sha256=model_sha256(),signal_session=session,
               score=float(row['setup_score']),relative_volume=float(row['setup_volume_ratio']),
               feature_basis='COMPLETED_SPLIT_BARS',signal_volume_raw=float(row['volume']))
    require(all(math.isfinite(value[k]) for k in ('score','relative_volume','signal_volume_raw')) and
            value['relative_volume']>0 and value['signal_volume_raw']>0,'setup_ranking_features_invalid')
    return dict(setup_ranking=value)


def key(proposal):
    value=proposal.get('setup_ranking')
    require(isinstance(value,dict) and value.get('model')==MODEL and
        value.get('model_sha256')==model_sha256() and value.get('feature_basis')=='COMPLETED_SPLIT_BARS' and
        value.get('signal_session')==proposal['signal_session'],'setup_ranking_missing_or_changed')
    for name in ('score','relative_volume','signal_volume_raw'):
        require(type(value.get(name)) in (int,float) and math.isfinite(value[name]) and
                (name=='score' or value[name]>0),'setup_ranking_features_invalid')
    # No strategy name, registry order, detector score or reserved family slot.
    # Equal stock-level scores use a deterministic hash for exit-plan ownership.
    return (-value['score'],-value['relative_volume'],proposal['symbol'],
            hashlib.sha256(proposal['signal_id'].encode()).hexdigest())


def rank_setups(proposals,*,limit=3):
    require(type(limit) is int and 0<limit<=3,'setup_selection_limit_invalid')
    require(len({p['signal_session'] for p in proposals})<=1 and
            len({p['next_session'] for p in proposals})<=1,'setup_selection_mixed_session')
    require(len({p['signal_id'] for p in proposals})==len(proposals),'setup_selection_duplicate_signal')
    ordered=sorted(proposals,key=key)
    groups={}
    for proposal in ordered:groups.setdefault(proposal['symbol'],[]).append(proposal)
    ranked=[]
    for symbol,matches in groups.items():
        first=matches[0]
        # The same symbol/session must have the same completed bar features.
        require(all(p['signal_session']==first['signal_session'] and
                    p['signal_close_raw']==first['signal_close_raw'] and
                    p['setup_ranking']==first['setup_ranking'] for p in matches),
                'setup_selection_mixed_symbol_evidence')
        ranked.append(dict(proposal=deepcopy(first),rank=len(ranked)+1,
            matching_strategies=sorted({p['strategy_id'] for p in matches})))
    return dict(policy=POLICY,model=MODEL,model_sha256=model_sha256(),
        qualifying_finalists=len(proposals),unique_symbols=len(ranked),ranked_setups=ranked,
        selected_setups=ranked[:limit],selected_signal_ids=[r['proposal']['signal_id'] for r in ranked[:limit]],
        tie_break='SCORE_DESC_RELATIVE_VOLUME_DESC_SYMBOL_SIGNAL_HASH',strategy_priority=False)


def selection_id(primary_run_id):
    return digest(dict(schema='jbravo.unified-setup-selection-id.v1',primary_run_id=primary_run_id))


def parent_context(primary):
    from scripts import strategy_execution as rules
    from datetime import date,time
    opening=datetime.combine(date.fromisoformat(primary['session']['next_session']),time(4),rules.NY)
    return dict(primary_run_id=primary['run_id'],primary_report_sha256=digest(primary),
        source_bindings=primary['source_bindings'],session=primary['session'],proposals=primary['batch']['proposals'],
        execution_policy=dict(entry_open_at=opening.isoformat()))


def required_strategies(base_dir=ROOT):
    from scripts.research_strategies import CHALLENGER
    from scripts.bull_flag_companion import configuration
    from scripts.bull_flag_strategy import BULL_FLAG
    from scripts.pattern_strategy import enabled_ids
    return sorted([CHALLENGER]+([BULL_FLAG] if configuration(base_dir) else [])+list(enabled_ids(base_dir)))


def inputs(connection,parent,*,now,base_dir=ROOT):
    from scripts.research_strategies import CHALLENGER
    from scripts.bull_flag_companion import read_handoff,_row,primary_id
    handoffs={CHALLENGER:parent};references={CHALLENGER:parent['primary_report_sha256']}
    for strategy_id in required_strategies(base_dir):
        if strategy_id==CHALLENGER:continue
        handoff=read_handoff(connection,parent,now=now,inspection=True,base_dir=base_dir,strategy_id=strategy_id)
        require(handoff is not None,'unified_selection_required_strategy_unavailable')
        handoffs[strategy_id]=handoff
        references[strategy_id]=_row(connection,primary_id(parent['primary_run_id'],strategy_id))[1].strip()
    proposals=[p for h in handoffs.values() for p in h['proposals']]
    require(all(p['strategy_id']==sid for sid,h in handoffs.items() for p in h['proposals']),
            'unified_selection_strategy_lineage_changed')
    return handoffs,references,rank_setups(proposals)


def publish(primary,*,env,base_dir=ROOT):
    """Publish only after every enabled strategy has a complete, exact result."""
    from scripts.research_store import connect
    from scripts.bull_flag_companion import _publish,_status
    now=datetime.now(timezone.utc);parent=parent_context(primary)
    try:
        connection=connect(env=env)
        try:handoffs,references,selection=inputs(connection,parent,now=now,base_dir=base_dir)
        finally:connection.rollback();connection.close()
        specification=dict(policy=POLICY,model_sha256=model_sha256(),required_strategies=sorted(handoffs),
                           strategy_result_sha256=references)
        record=dict(schema='jbravo.unified-setup-selection.v1',status='COMPLETED_UNIFIED_SETUP_SELECTION',
            run_id=selection_id(primary['run_id']),primary_run_id=primary['run_id'],
            parent_report_sha256=digest(primary),source_bindings=deepcopy(primary['source_bindings']),
            session=primary['session'],specification=specification,selection=selection,
            finished_at=now.isoformat(),paper_only=True,research_qualified=False,order_submission_authorized=False)
        from scripts import pipeline_postflight as health
        require(all(health._binding(health._read(Path(base_dir)/n,health.MAX_SOURCE_BYTES))==pin
                    for n,pin in record['source_bindings'].items()),'unified_selection_sources_changed')
        _publish(record,digest(specification),env=env)
        status=dict(status=record['status'],run_id=record['run_id'],result_sha256=digest(record),
                    selected_signal_ids=selection['selected_signal_ids'])
    except Exception as error:
        status=dict(status='UNIFIED_SETUP_SELECTION_UNAVAILABLE',error_type=type(error).__name__)
    _status(status,base_dir,'unified_setups')
    return status


def read_selection(connection,parent,*,now,base_dir=ROOT):
    """Reconstruct the ranking from exact immutable inputs before using it."""
    from scripts.bull_flag_companion import _row
    row=_row(connection,selection_id(parent['primary_run_id']))
    require(row is not None,'unified_selection_not_published')
    spec_sha,result_sha,record=row
    require(result_sha.strip()==digest(record) and record.get('schema')=='jbravo.unified-setup-selection.v1' and
        record.get('status')=='COMPLETED_UNIFIED_SETUP_SELECTION' and
        record['run_id']==selection_id(parent['primary_run_id']) and
        record['primary_run_id']==parent['primary_run_id'] and
        record['parent_report_sha256']==parent['primary_report_sha256'] and
        record['source_bindings']==parent['source_bindings'] and record['session']==parent['session'] and
        record.get('paper_only') is True and record.get('research_qualified') is False and
        record.get('order_submission_authorized') is False and
        datetime.fromisoformat(record['finished_at'])<=now,'unified_selection_lineage_changed')
    handoffs,references,selection=inputs(connection,parent,now=now,base_dir=base_dir)
    specification=dict(policy=POLICY,model_sha256=model_sha256(),required_strategies=sorted(handoffs),
                       strategy_result_sha256=references)
    require(record['specification']==specification and spec_sha.strip()==digest(specification) and
            record['selection']==selection,'unified_selection_ranking_changed')
    return dict(record=record,result_sha256=result_sha.strip(),handoffs=handoffs)


def execution_queue(verified):
    from scripts import strategy_execution as rules
    record=verified['record'];result=[]
    for setup in record['selection']['selected_setups']:
        proposal=setup['proposal'];handoff=deepcopy(verified['handoffs'][proposal['strategy_id']])
        handoff['unified_selection']=dict(run_id=record['run_id'],result_sha256=verified['result_sha256'],
            rank=setup['rank'],selected_signal_ids=record['selection']['selected_signal_ids'])
        handoff.pop('handoff_id',None);handoff['handoff_id']=rules.digest(handoff)
        result.append(dict(handoff=handoff,proposal=proposal,rank=setup['rank']))
    return result


def dashboard_selection(connection,batch,primary_run_id,*,base_dir=ROOT):
    from scripts import pipeline_postflight as health,strategy_pipeline as pipeline
    from scripts.bull_flag_companion import _row,primary_id
    from scripts.research_strategies import CHALLENGER
    raw=health._read(Path(base_dir)/'reports/pipeline_postflight/latest.json',health.MAX_REPORT_BYTES)
    primary=json.loads(raw)
    require(primary['run_id']==primary_run_id and primary['batch']==batch[3],
            'unified_dashboard_primary_changed')
    now=datetime.now(timezone.utc)
    pipeline.check_report(primary,raw,base_dir,now=now,max_age_seconds=43200,not_before=None,
                          require_completion=True,inspection=True)
    verified=read_selection(connection,parent_context(primary),now=now,base_dir=base_dir)
    reports={CHALLENGER:dict(backtest=primary['backtest'],metrics=primary['metrics'])}
    for sid in verified['handoffs']:
        if sid==CHALLENGER:continue
        record=_row(connection,primary_id(primary_run_id,sid))[2]
        reports[sid]=dict(backtest=record['backtest'],metrics=record['metrics'])
    verified.update(reports=reports,primary=primary)
    return verified


def dashboard_rows(verified):
    from scripts.strategy_execution import NY
    from datetime import date,time
    record=verified['record'];result=[];day=record['session']['signal_session']
    for setup in record['selection']['selected_setups']:
        p=setup['proposal'];score=p['setup_ranking']['score']
        result.append(dict(symbol=p['symbol'],score=score,final_score=score,run_date=day,
            run_ts_utc=verified['primary']['candidate_run_ts_utc'],close=p['signal_close_raw'],
            entry_price=p['signal_close_raw'],volume=p['setup_ranking']['signal_volume_raw'],
            timestamp=datetime.combine(date.fromisoformat(day),time.min,NY).isoformat(),
            score_breakdown=dict(proposal=p,setup_ranking=p['setup_ranking'],
                matching_strategies=setup['matching_strategies'],unified_selection_run_id=record['run_id']),
            source='unified_strategy_screen_v1',passed_gates=True,gate_fail_reason=None,ml_weight_used=0.))
    return result


def dashboard_reports(kind,result,selection,reports,args):
    """Each selected setup retains its own Metrics/Backtest identity, never pooled P&L."""
    rows=[]
    for setup in selection['selected_setups']:
        proposal=setup['proposal'];report=reports[proposal['strategy_id']]
        if kind=='metrics':
            matches=[r for r in report['metrics']['rows'] if r['signal_id']==proposal['signal_id']]
        else:
            matches=[dict(r,signal_id=r['current_signal_id'],trades=r['closed_signals'],
                wins=r['winning_signals'],losses=r['losing_signals'],win_rate=r['win_rate_pct'],
                net_pnl=r['closed_trade_net_pnl'],expectancy=r['expectancy_net_per_closed_signal'],
                max_drawdown=r['max_closed_signal_drawdown_cash'],drawdown_units='USD_CLOSED_SIGNAL_PNL')
                for r in report['backtest']['results'] if r['current_signal_id']==proposal['signal_id']]
        require(len(matches)==1,'unified_dashboard_candidate_history_missing')
        rows.append(dict(matches[0],rank=setup['rank'],strategy_id=proposal['strategy_id'],
            strategy_version=proposal['strategy_version'],setup_score=proposal['setup_ranking']['score'],
            backtest_result_id=report['backtest']['result_id'],metrics_result_id=report['metrics']['result_id']))
    query=str(args.get('q') or '').strip().upper()
    if query:rows=[r for r in rows if query in (r['symbol']+' '+r['strategy_id']).upper()]
    if args.get('filter')=='with_history':rows=[r for r in rows if r['trades']>0]
    if args.get('filter')=='no_history':rows=[r for r in rows if r['trades']==0]
    try:limit=max(1,min(200,int(args.get('limit') or 50)))
    except (ValueError,TypeError):limit=50
    rows=rows[:limit]
    return dict(result,rows=rows,rows_final=len(rows),**contract(),
        metric_basis='STRATEGY_RESEARCH_SIMULATION',strategy_priority=False,
        **({'metrics_status':'COMPLETED_UNIFIED_SETUP_METRICS'} if kind=='metrics' else
           {'backtest_status':'COMPLETED_UNIFIED_SETUP_BACKTEST'}))
