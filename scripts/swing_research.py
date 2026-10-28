"""Frozen six-strategy comparison lab. No scheduler, broker or automatic promotion."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import time
import uuid

import pandas as pd

from scripts import setup_selection
from scripts.pattern_strategy import PATTERNS, plan_from_row
from scripts.research_dataset import bounded_read, canonical_hash, digest_bytes, load_dataset, strict_json
from scripts.research_strategies import CHALLENGER
from scripts.research_testing import json_default, save_json, validate_spec
from scripts.strategy_library import select_strategies
from scripts.swing_execution_research import entry_comparison
from scripts.swing_research_portfolio import EXIT_POLICIES, StudyPortfolio, frames_for_book
from scripts.swing_scorecard import benchmark_comparison,portfolio_scorecard,ranking_scorecard,require

ROOT=Path(__file__).resolve().parents[1]
STRATEGIES=tuple(sorted((CHALLENGER,*PATTERNS)))
FAMILIES=('ALL_STRATEGIES',*STRATEGIES)
SOURCE_FILES=('swing_research.py','swing_scorecard.py','swing_research_portfolio.py','swing_execution_research.py',
    'research_dataset.py','research_testing.py','research_simulation.py','research_strategies.py',
    'research_store.py','backtest_core.py','indicators.py','strategy_library.py','strategy_universe.py',
    'strategy_market_replay.py','strategy_backtest.py','strategy_metrics.py','strategy_pipeline.py',
    'bull_flag_strategy.py','chart_patterns.py','pattern_strategy.py','swing_patterns.py','setup_selection.py')


def plain(value):
    return json.loads(json.dumps(value,default=json_default,allow_nan=False))


def outside(path):
    path=Path(path).resolve()
    require(path!=ROOT and not path.is_relative_to(ROOT),'Research artifacts must be outside the production/source tree')
    return path


def prepare_plan(base, *, capture_manifest, axis='exits', entry_capture=None):
    base=deepcopy(base)
    if len(base['cost_cases'])==1:
        case=base['cost_cases'][0]
        base['cost_cases'].append(dict(id='cost_stress' if case['id']!='cost_stress' else 'cost_stress_2',fee_per_fill=max(1.,case['fee_per_fill']*2),
                                      slippage_bps=max(10.,case['slippage_bps']*2)))
    base['portfolio'].update(alloc_pct=.05,top_n=3,max_positions=4,trail_pct=.03,atr_multiple=1.,max_hold_days=7)
    base.update(strategies=list(STRATEGIES),max_runs=len(base['folds'])*len(base['cost_cases'])*7)
    validate_spec(base)
    roles=['DEVELOPMENT'] if len(base['folds'])==1 else ['DEVELOPMENT']+['VALIDATION']*(len(base['folds'])-2)+['HOLDOUT']
    return dict(schema='jbravo.swing-research-plan.v1',purpose='ISOLATED_EXPLORATORY_RESEARCH',study=base,
        experiment_id=uuid.uuid4().hex,data_kind='UNQUALIFIED_MARKET_CAPTURE',
        capture_manifest_sha256=digest_bytes(bounded_read(capture_manifest)),axis=axis,
        exit_policies=list(EXIT_POLICIES) if axis=='exits' else ['existing'],
        entry_capture=entry_capture,
        phases={f['id']:dict(role=role,exposure='PREVIOUSLY_EXPOSED') for f,role in zip(base['folds'],roles)},
        budgets=dict(max_seconds=3600,max_trials=500,max_symbols=250,max_qualifying_signals=2000,bootstrap_samples=400),
        evidence_floors=dict(min_closed_signals=100,min_signal_weeks=20,max_drawdown_pct=10),
        research_qualified=False,execution_authorized=False)


def validate_plan(plan):
    require(set(plan)=={'schema','purpose','experiment_id','data_kind','study','capture_manifest_sha256','axis','exit_policies','entry_capture',
        'phases','budgets','evidence_floors','research_qualified','execution_authorized'},'Unexpected lab plan fields')
    require(plan['schema']=='jbravo.swing-research-plan.v1' and plan['purpose']=='ISOLATED_EXPLORATORY_RESEARCH' and
        plan['research_qualified'] is False and plan['execution_authorized'] is False,'Exploratory research only')
    spec=plan['study'];validate_spec(spec)
    require(isinstance(plan['experiment_id'],str) and re.fullmatch('[0-9a-f]{32}',plan['experiment_id']), 'Frozen experiment identity required')
    require(plan['data_kind'] in {'UNQUALIFIED_MARKET_CAPTURE','SYNTHETIC_TEST_FIXTURE'},'Explicit data-kind declaration required')
    require(all(spec['portfolio'][k]==v for k,v in dict(alloc_pct=.05,top_n=3,max_positions=4,trail_pct=.03,atr_multiple=1.,max_hold_days=7).items()),
            'Freeze the current risk policy while comparing entry/exit policies')
    cost_values=[(c['fee_per_fill'],c['slippage_bps']) for c in spec['cost_cases']]
    require(len(cost_values)>=2 and len(set(cost_values))==len(cost_values) and
        all(f>=cost_values[0][0] and s>=cost_values[0][1] and (f,s)!=cost_values[0] for f,s in cost_values[1:]),
            'Base and strictly stronger cost cases required')
    require(all(isinstance(r['id'],str) and re.fullmatch(r'[A-Za-z0-9_-]{1,48}',r['id']) for r in [*spec['folds'],*spec['cost_cases']]),
            'Safe canonical fold/cost IDs required')
    require(sorted(spec['strategies'])==list(STRATEGIES),'All six strategies must be present')
    require(plan['axis'] in {'entries','exits'},'Choose one experiment axis')
    policies=plan['exit_policies']
    require(policies and policies[0]=='existing' and len(policies)==len(set(policies)) and set(policies)<=set(EXIT_POLICIES),
            'Existing-policy baseline and distinct registered policies required')
    require(plan['axis']=='exits' or policies==['existing'],'Entry study cannot change exits')
    require(plan['axis']=='entries' or plan['entry_capture'] is None,'Exit study cannot change entry model')
    if plan['entry_capture'] is not None:
        require(set(plan['entry_capture'])=={'path','sha256'},'Entry capture requires exact path/hash')
    require(set(plan['phases'])=={f['id'] for f in spec['folds']},'Every fold needs an exposure declaration')
    roles=[]
    for phase in plan['phases'].values():
        require(set(phase)=={'role','exposure'} and phase['role'] in {'DEVELOPMENT','VALIDATION','HOLDOUT'} and
                phase['exposure'] in {'PREVIOUSLY_EXPOSED','DECLARED_UNSEEN'},'Invalid fold phase/exposure')
    roles=[plan['phases'][f['id']]['role'] for f in spec['folds']]
    require(roles==sorted(roles,key=['DEVELOPMENT','VALIDATION','HOLDOUT'].index) and roles[0]=='DEVELOPMENT',
            'Chronological development/validation/holdout ordering required')
    bounds={'max_seconds':(1,3600),'max_trials':(1,500),'max_symbols':(1,1000),
            'max_qualifying_signals':(1,10000),'bootstrap_samples':(100,2000)}
    require(set(plan['budgets'])==set(bounds),'Unexpected experiment budgets')
    for name,(low,high) in bounds.items():
        require(type(plan['budgets'][name]) is int and low<=plan['budgets'][name]<=high,'Invalid '+name)
    floors=plan['evidence_floors']
    require(set(floors)=={'min_closed_signals','min_signal_weeks','max_drawdown_pct'} and
        type(floors['min_closed_signals']) is int and 1<=floors['min_closed_signals']<=100000 and
        type(floors['min_signal_weeks']) is int and 2<=floors['min_signal_weeks']<=1000 and
        type(floors['max_drawdown_pct']) in (int,float) and 0<floors['max_drawdown_pct']<=100,'Invalid evidence floors')
    trials=len(spec['folds'])*len(spec['cost_cases'])*len(FAMILIES)*len(policies)
    require(trials<=plan['budgets']['max_trials'],'Frozen scenario budget exceeded')
    require(len(spec['universe']['symbols'])<=plan['budgets']['max_symbols'],'Universe over budget; no silent truncation')
    return trials


def build_book(dataset, spec, fold, check, *, max_signals=10000):
    """Uncapacitated qualifying-close cohort, retained before any top-three cut."""
    prepared, observations={},[]
    calendar=list(dataset.calendar.index)
    next_days={d:calendar[i+1] for i,d in enumerate(calendar[:-1])}
    for strategy in select_strategies(STRATEGIES):
        definition=strategy.describe()['definition_sha256']
        for symbol in spec['universe']['symbols']:
            check()
            frame=dataset.prepare(symbol,strategy,fold) if getattr(dataset,'market_wide',False) else strategy.prepare(dataset.raw[symbol],dataset.features[symbol],fold)
            prepared[(strategy.id,symbol)]=frame
            consumed=None
            for day,row in frame.iterrows():
                check()
                if pd.isna(row['score']) or not bool(row['entry_ready']):continue
                session=day.strftime('%Y-%m-%d')
                details=row['entry_details']
                if strategy.id==CHALLENGER:
                    if consumed is not None and details['pullback_start']<=consumed:continue
                    consumed=session
                pattern=plan_from_row(symbol,row) if strategy.id in PATTERNS else None
                require(strategy.id not in PATTERNS or pattern is not None,'Pattern signal lacks validated exit plan')
                start=details.get('pullback_start',session)
                structural=pattern['invalidation'] if pattern else float(dataset.raw[symbol].loc[start:session,'low'].min())*.999
                proposal=dict(strategy_id=strategy.id,strategy_version=strategy.version,definition_sha256=definition,symbol=symbol,
                    signal_session=session,next_session=next_days.get(session,session),
                    signal_id=canonical_hash(dict(strategy=strategy.id,version=strategy.version,symbol=symbol,
                        definition_sha256=definition,fold=fold['id'],episode_start=fold['test_start'],signal_session=session)),
                    signal_close_raw=float(row['close']),signal_atr14_raw=float(row['ATR14']),
                    structural_stop=structural,**setup_selection.evidence(row,session))
                if pattern:proposal['pattern_exit']=pattern
                observations.append(dict(signal_id=proposal['signal_id'],strategy_id=strategy.id,symbol=symbol,
                    signal_session=session,proposal=proposal,selected=False,is_exit_owner=False,
                    terminal_session_without_next=session not in next_days))
                require(len(observations)<=max_signals,'Qualifying-signal budget exceeded; no truncation')
    for session in sorted({r['signal_session'] for r in observations}):
        rows=[r for r in observations if r['signal_session']==session]
        global_rank=setup_selection.rank_setups([r['proposal'] for r in rows])
        ranks={r['proposal']['symbol']:r for r in global_rank['ranked_setups']}
        for family in STRATEGIES:
            ordered=sorted([r for r in rows if r['strategy_id']==family],key=lambda r:setup_selection.key(r['proposal']))
            for rank,item in enumerate(ordered,1):item['family_rank']=rank
        for item in rows:
            ranked=ranks[item['symbol']]
            item.update(global_rank=ranked['rank'],selected=ranked['rank']<=3,
                matching_strategies=ranked['matching_strategies'],
                is_exit_owner=item['signal_id']==ranked['proposal']['signal_id'])
    return prepared,observations


def dividends_for(dataset, spec):
    if not getattr(dataset,'market_wide',False):return dataset.dividends
    return tuple(event for symbol in spec['universe']['symbols'] for category,event in dataset.member(symbol)['actions']
                 if category=='cash_dividends')


def simulate(prepared, book, spec, fold, cost, policy, dividends, check, *, family='ALL_STRATEGIES', signal=None, calendar=None):
    check()
    data=frames_for_book(prepared,book,family=family,only_signal=signal)
    if signal is not None:
        data={symbol:frame.iloc[:spec['portfolio']['max_hold_days']+6].copy() for symbol,frame in data.items()}
    if not data:
        frame=next(iter(prepared.values())).copy()
        frame['score']=float('nan');frame['research_proposal']=None
        data={next(iter(prepared))[1]:frame}
    options=dict(spec['portfolio'],trade_cost=cost['fee_per_fill'],slippage=cost['slippage_bps']/10000)
    engine=StudyPortfolio(data,exit_policy=policy,dividends=dividends,**options)
    if calendar is not None:
        first=next(r['signal_session'] for r in book if r['signal_id']==signal) if signal else fold['test_start']
        last=max(f.index[-1] for f in data.values()).strftime('%Y-%m-%d') if signal else fold['test_end']
        engine.dates=[pd.Timestamp(d) for d in calendar if first<=d<=last]
    # Check the lab's own budget before every session, including counterfactuals.
    before=engine._before_session
    def checked(day):check();before(day)
    engine._before_session=checked
    engine.run()
    return engine


def evidence_gate(metrics, phase, floors):
    interval=metrics['expectancy_interval_95']
    checks=dict(sample_floor=metrics['closed_signals']>=floors['min_closed_signals'],
        exit_policy_applicable_at_entries=metrics['inapplicable_exit_policy_attempts']==0,
        signal_week_floor=interval['signal_weeks']>=floors['min_signal_weeks'],
        positive_net_expectancy=metrics['expectancy_net_per_signal'] is not None and metrics['expectancy_net_per_signal']>0,
        positive_cluster_interval=interval['low'] is not None and interval['low']>0,
        drawdown_within_frozen_limit=metrics['max_drawdown_pct']>=-floors['max_drawdown_pct'],
        declared_unseen_evaluation=phase['role'] in {'VALIDATION','HOLDOUT'} and phase['exposure']=='DECLARED_UNSEEN')
    return dict(checks=checks,all_checks_met=all(checks.values()),research_qualified=False,
        disposition='REVIEW_EVIDENCE_ONLY_NO_AUTOMATIC_PROMOTION',
        limits='Sample floors are not statistical proof; adjacent weeks may remain dependent; multiple testing is not corrected')


def report_html(output, result):
    rows=[]
    for run in result['trials']:
        m=run['scorecard']
        rows.append(dict(fold=run['fold'],phase=run['phase']['role'],family=run['family'],exit=run['exit_policy'],cost=run['cost_case'],
            closed=m['closed_signals'],expectancy=m['expectancy_net_per_signal'],return_pct=m['total_return_pct'],
            drawdown_pct=m['max_drawdown_pct'],exposure_pct=m['average_close_exposure_pct']))
    table=pd.DataFrame(rows).to_html(index=False,escape=True,border=0,na_rep='Unknown')
    entry=''
    if result['entry_comparisons']:
        entry='<h2>Entry comparison</h2>'+pd.DataFrame([dict(fold=r['fold'],cost=r['cost_case'],status=r['status'],
            paired_signals=r.get('paired_signals',0)) for r in result['entry_comparisons']]).to_html(index=False,escape=True,border=0)
        entry+='<p>Entry-price sensitivity only. No premarket strategy P&amp;L is calculated. Missing evidence is unavailable, never a daily-bar proxy.</p>'
    text=("<!doctype html><meta charset='utf-8'><title>JBravo swing research</title>"
        "<style>body{font:15px system-ui;margin:32px;color:#172536}td,th{padding:8px;text-align:right;border-bottom:1px solid #ddd}"
        ".notice{padding:16px;background:#fff0cc}table{border-collapse:collapse}</style><h1>Swing research scorecard</h1>"
        "<p class='notice'>Exploratory research. No profitability qualification or automatic trading change. "
        "Premarket entry sensitivity is separate from daily-bar portfolio results. Unknown outcomes remain unknown.</p>"
        +'<p class="notice">Data: '+html.escape(result['data_kind'])+'</p>'
        +table+entry+'<h2>Limitations</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in result['limitations'])+'</ul>')
    (output/'report.html').write_text(text,encoding='utf-8')


def run_lab(plan, output, registry, *, stop_file=None):
    planned=validate_plan(plan)
    output,registry=outside(output),outside(registry)
    require(output!=registry and not output.is_relative_to(registry) and not registry.is_relative_to(output),'Output and registry must be separate')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'plan.json',plan)
    sources={n:digest_bytes((ROOT/'scripts'/n).read_bytes()) for n in SOURCE_FILES}
    environment=dict(python=platform.python_version(),pandas=pd.__version__,numpy=version('numpy'))
    run_id=canonical_hash(dict(plan=plan,source=sources,environment=environment))
    registry.mkdir(parents=True,exist_ok=True)
    claim=registry/run_id
    claim.mkdir(exist_ok=False)
    save_json(claim/'claim.json',dict(run_id=run_id,output=str(output),planned_trials=planned,
        claimed_at=datetime.now(timezone.utc).isoformat(),disposition='SINGLE_USE_NEVER_REPLAY',plan_sha256=canonical_hash(plan)))
    deadline=time.monotonic()+plan['budgets']['max_seconds']
    stop=Path(stop_file) if stop_file else output/'STOP'
    def check():
        if stop.exists() or time.monotonic()>=deadline:raise RuntimeError('Research STOP/deadline reached')
    trials=[]
    try:
        source=output/'source';source.mkdir()
        for n in SOURCE_FILES:
            body=(ROOT/'scripts'/n).read_bytes()
            require(digest_bytes(body)==sources[n],'Source changed before freeze')
            (source/n).write_bytes(body)
        spec=plan['study']
        require(digest_bytes(bounded_read(spec['capture_manifest']))==plan['capture_manifest_sha256'],'Capture manifest changed')
        dataset=load_dataset(spec)
        save_json(output/'dataset.json',dataset.provenance)
        dividends=dividends_for(dataset,spec)
        prepared_count=0
        ranking_reports=[];entry_reports=[]
        for fold in spec['folds']:
            check()
            for boundary in ('warmup_start','test_start','test_end'):
                require(fold[boundary] in dataset.calendar.index,'Uncaptured fold boundary')
            prepared,book=build_book(dataset,spec,fold,check,
                max_signals=plan['budgets']['max_qualifying_signals']-prepared_count)
            prepared_count+=len(book)
            require(prepared_count<=plan['budgets']['max_qualifying_signals'],'Qualifying-signal budget exceeded; no truncation')
            save_json(output/('cohort-'+fold['id']+'.json'),dict(observations=book,
                cohort='ALL_UNCAPACITATED_QUALIFYING_CLOSES',held_pending_account_filters=False,
                ranking_policy=setup_selection.POLICY,source_sha256=sources,execution_authorized=False))
            for cost in spec['cost_cases']:
                if plan['axis']=='entries':
                    entry=entry_comparison(book,plan['entry_capture'],cash=spec['portfolio']['initial_cash'],
                        fee_per_fill=cost['fee_per_fill'],slippage_bps=cost['slippage_bps'])
                    entry_reports.append(dict(fold=fold['id'],cost_case=cost['id'],**entry))
                for policy in plan['exit_policies']:
                    for family in FAMILIES:
                        folder=output/('trial-'+str(len(trials)+1).zfill(4));folder.mkdir()
                        declaration=dict(fold=fold['id'],phase=plan['phases'][fold['id']],cost_case=cost['id'],exit_policy=policy,family=family)
                        save_json(folder/'trial-claim.json',dict(**declaration,cohort_sha256=canonical_hash(book)))
                        engine=simulate(prepared,book,spec,fold,cost,policy,dividends,check,family=family,calendar=dataset.calendar.index)
                        metrics=portfolio_scorecard(engine,bootstrap_samples=plan['budgets']['bootstrap_samples'])
                        benchmark=spec['universe']['benchmark']
                        events=(tuple(e for category,e in dataset.member(benchmark)['actions'] if category=='cash_dividends')
                                if getattr(dataset,'market_wide',False) else tuple(e for e in dataset.dividends if e['symbol']==benchmark))
                        metrics['benchmarks']=benchmark_comparison(engine,dataset.raw[benchmark],events)
                        record=dict(**declaration,scorecard=metrics,evidence=evidence_gate(metrics,plan['phases'][fold['id']],plan['evidence_floors']))
                        save_json(folder/'result.json',record);save_json(folder/'fills.json',engine.fills)
                        save_json(folder/'attempts.json',engine.attempts);save_json(folder/'ledger.json',engine.reconciliations)
                        engine.equity().to_csv(folder/'equity.csv')
                        trials.append(record)
                    # Rejected setups get exactly the same fixed-cash/cost/exit
                    # experiment as selected setups, without shared capital.
                    outcomes={}
                    for item in book:
                        check()
                        engine=simulate(prepared,[item],spec,fold,cost,policy,dividends,check,signal=item['signal_id'],calendar=dataset.calendar.index)
                        metrics=portfolio_scorecard(engine,bootstrap_samples=100)
                        outcome=metrics['episodes'][0] if metrics['episodes'] else dict(signal_id=item['signal_id'],
                            signal_session=item['signal_session'],state='UNFILLED',net_pnl=None)
                        if outcome['state']=='CLOSED':
                            buy=next(f for f in engine.fills if f['side']=='BUY')
                            outcome['net_return_pct']=100*outcome['net_pnl']/(buy['quantity']*buy['raw_price']+buy['fee']+buy['slippage'])
                        outcomes[item['signal_id']]=outcome
                    ranking=dict(fold=fold['id'],cost_case=cost['id'],exit_policy=policy,
                        scorecard=ranking_scorecard(book,outcomes,bootstrap_samples=plan['budgets']['bootstrap_samples']))
                    ranking_reports.append(ranking)
                    save_json(output/('counterfactual-'+str(len(ranking_reports)).zfill(3)+'.json'),dict(**ranking,outcomes=outcomes))
        require(len(trials)==planned,'Scenario accounting mismatch')
        require(all(digest_bytes((ROOT/'scripts'/n).read_bytes())==pin for n,pin in sources.items()),'Source changed during study')
        comparisons=[]
        for trial in trials:
            if trial['exit_policy']=='existing':continue
            baseline=next(t for t in trials if t['family']==trial['family'] and t['fold']==trial['fold'] and
                t['cost_case']==trial['cost_case'] and t['exit_policy']=='existing')
            comparisons.append(dict(family=trial['family'],fold=trial['fold'],cost_case=trial['cost_case'],exit_policy=trial['exit_policy'],
                return_difference_pct=trial['scorecard']['total_return_pct']-baseline['scorecard']['total_return_pct'],
                drawdown_difference_pct=trial['scorecard']['max_drawdown_pct']-baseline['scorecard']['max_drawdown_pct']))
        result=dict(schema='jbravo.swing-research-result.v1',status='COMPLETED_EXPLORATORY',run_id=run_id,
            data_kind=plan['data_kind'],
            environment=environment,generated_at_utc=datetime.now(timezone.utc).isoformat(),
            plan_sha256=canonical_hash(plan),source_sha256=sources,dataset_sha256=dataset.provenance['dataset_sha256'],
            planned_trials=planned,completed_trials=len(trials),trials=trials,baseline_comparisons=comparisons,
            ranking_scorecards=ranking_reports,entry_comparisons=entry_reports,
            strategy_priority=False,research_qualified=False,execution_authorized=False,
            prospective_confirmation='NOT_ESTABLISHED_BY_RETROSPECTIVE_STUDY',
            limitations=list(dataset.provenance['limitations'])+[
                'Frozen chronological evaluation, no parameter fitting or automated winning-strategy selection',
                'Exposure declarations do not prove unseen data; previously exposed folds cannot establish holdout evidence',
                'Uncapacitated candidate cohort measures selection; it is not production held/pending/rearm parity',
                'Standalone counterfactual outcomes overlap and must never be summed into a portfolio',
                'Unfilled and terminal-open signals are censored, not zero returns; paired session comparisons require every owner outcome closed',
                'Portfolio trials share cash across six families only in ALL_STRATEGIES; family returns cannot be added',
                'Daily OHLC results use regular-session opens, not production premarket execution',
                'Excursions use full interior holding sessions and observed fills; values are lower bounds, not exact intraday extrema',
                'Week-cluster intervals are exploratory; dependence across weeks and multiple testing remain unresolved',
                'Fixed costs/slippage are assumptions; terminal holdings remain open; no invented liquidation',
                'No completed study, sample floor or evidence checklist proves future profitability'])
        save_json(output/'result.json',result);report_html(output,result)
        pins=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=digest_bytes(p.read_bytes()))
              for p in sorted(output.rglob('*')) if p.is_file()]
        save_json(output/'manifest.json',dict(run_id=run_id,bindings=pins))
        save_json(claim/'completion.json',dict(run_id=run_id,manifest_sha256=digest_bytes((output/'manifest.json').read_bytes()),
            status=result['status'],completed_trials=len(trials),research_qualified=False))
        return result
    except Exception as exc:
        failure=dict(status='FAILED_NEVER_REPLAY',error_type=type(exc).__name__,completed_trials=len(trials),
                     reason=str(exc)[:300] if isinstance(exc,(ValueError,RuntimeError)) else 'Unexpected failure; inspect local inputs',
                     research_qualified=False,execution_authorized=False)
        save_json(output/'failure.json',failure);save_json(claim/'failure.json',failure)
        raise


def registry_summary(registry):
    result=[]
    for path in sorted(outside(registry).glob('*/claim.json')):
        claim=strict_json(bounded_read(path,1024*1024))
        require(claim['run_id']==path.parent.name and re.fullmatch('[0-9a-f]{64}',claim['run_id']) and
                type(claim['planned_trials']) is int and 1<=claim['planned_trials']<=500,'Corrupt experiment claim')
        complete,failed=path.parent/'completion.json',path.parent/'failure.json'
        require(not (complete.exists() and failed.exists()),'Ambiguous experiment disposition')
        status='COMPLETED' if complete.exists() else 'FAILED' if failed.exists() else 'INCOMPLETE_UNKNOWN'
        count=None
        if complete.exists():
            receipt=strict_json(bounded_read(complete,1024*1024))
            manifest=outside(claim['output'])/'manifest.json'
            require(receipt['run_id']==claim['run_id'] and receipt['completed_trials']==claim['planned_trials'] and
                receipt['manifest_sha256']==digest_bytes(bounded_read(manifest,16*1024*1024)), 'Completed experiment evidence changed')
            count=receipt['completed_trials']
        elif failed.exists():
            count=strict_json(bounded_read(failed,1024*1024))['completed_trials']
        result.append(dict(run_id=claim['run_id'],planned_trials=claim['planned_trials'],completed_trials=count,status=status))
    return dict(experiments=result,attempted_experiments=len(result),planned_trial_count=sum(r['planned_trials'] for r in result),
                completed_trial_count=sum(r['completed_trials'] or 0 for r in result),
                selection_bias_correction='NOT_COMPUTED_KEEP_ALL_ATTEMPTS')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare')
    prepare.add_argument('--base-spec',type=Path,required=True);prepare.add_argument('--output',type=Path,required=True)
    prepare.add_argument('--axis',choices=('entries','exits'),default='exits');prepare.add_argument('--entry-capture',type=Path)
    run=commands.add_parser('run')
    run.add_argument('--plan',type=Path,required=True);run.add_argument('--output',type=Path,required=True)
    run.add_argument('--registry',type=Path,required=True);run.add_argument('--stop-file',type=Path)
    history=commands.add_parser('experiments');history.add_argument('--registry',type=Path,required=True)
    paper=commands.add_parser('paper-snapshot')
    paper.add_argument('--account-id',required=True);paper.add_argument('--start',required=True);paper.add_argument('--end',required=True)
    paper.add_argument('--output',type=Path,required=True);paper.add_argument('--env-file',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command=='prepare':
            base=strict_json(bounded_read(args.base_spec,1024*1024))
            binding=dict(path=str(args.entry_capture.resolve()),sha256=digest_bytes(bounded_read(args.entry_capture,64*1024*1024))) if args.entry_capture else None
            plan=prepare_plan(base,capture_manifest=base['capture_manifest'],axis=args.axis,entry_capture=binding)
            validate_plan(plan);save_json(outside(args.output),plan)
            result=dict(prepared=str(args.output),planned_trials=validate_plan(plan),research_qualified=False)
        elif args.command=='run':
            value=run_lab(strict_json(bounded_read(args.plan,1024*1024)),args.output,args.registry,stop_file=args.stop_file)
            result=dict(status=value['status'],run_id=value['run_id'],completed_trials=value['completed_trials'])
        elif args.command=='experiments':result=registry_summary(args.registry)
        else:
            from scripts.research_store import connect
            from scripts.swing_execution_research import paper_snapshot,paper_scorecard
            if args.env_file:
                from dotenv import load_dotenv
                load_dotenv(args.env_file,override=False)
            folder=outside(args.output);folder.mkdir(parents=True,exist_ok=False)
            application_env=dict(os.environ)
            application_env.pop('RESEARCH_DATABASE_URL',None)
            conn=connect(env=application_env)
            try:snapshot=paper_snapshot(conn,account_id=args.account_id,start=args.start,end=args.end)
            finally:conn.rollback();conn.close()
            save_json(folder/'snapshot.json',snapshot);save_json(folder/'scorecard.json',paper_scorecard(snapshot))
            result=dict(status='READ_ONLY_PAPER_SNAPSHOT_SAVED',database_mutations=0,broker_operations=0)
        print(json.dumps(result,default=json_default,allow_nan=False));return 0
    except Exception as exc:
        print(json.dumps(dict(status='FAILED',error_type=type(exc).__name__,
            detail=str(exc)[:300] if isinstance(exc,(ValueError,RuntimeError,FileExistsError,FileNotFoundError)) else 'Research operation refused; inspect local immutable artifacts.')))
        return 1


if __name__=='__main__':raise SystemExit(main())
