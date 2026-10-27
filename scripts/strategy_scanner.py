"""Daily strategy scanner and traceable paper replay over verified captured inputs.

No broker client, credential loading, order submission or database publication.
State is reconstructed from the declared episode start through the requested
session, using the same kernel as research. Each output root is single use.
"""
from __future__ import annotations

from datetime import date, datetime, time, timezone
import hashlib
import gzip
import html
import json
from pathlib import Path
import time as clock
from zoneinfo import ZoneInfo

import pandas as pd

from scripts.research_dataset import bounded_read, canonical_hash, load_dataset, strict_json
from scripts.research_simulation import ResearchPortfolio, availability_clock
from scripts.research_testing import save_json, validate_spec, json_default
from scripts.strategy_library import select_strategies
from scripts.pattern_strategy import PATTERNS, plan_from_row

SOURCE_FILES = ('screener.py', 'strategy_scanner.py', 'strategy_library.py', 'research_dataset.py',
                'research_testing.py', 'research_strategies.py', 'research_simulation.py',
                'research_store.py', 'backtest_core.py', 'indicators.py',
                'strategy_universe.py', 'strategy_market_replay.py', 'bull_flag_strategy.py', 'chart_patterns.py', 'pattern_strategy.py', 'swing_patterns.py')


def signal_id(strategy, episode_id, symbol, session):
    # Logical identity is stable across daily runs. Revised input bytes remain
    # explicit in batch provenance; a downstream consumer must bind that batch.
    return canonical_hash(dict(strategy_id=strategy.id, version=strategy.version,
                               episode_id=episode_id, symbol=symbol, signal_session=session))


class ScannerPortfolio(ResearchPortfolio):
    def __init__(self, data, *, strategy, episode_id, **kwargs):
        method = strategy.research_method
        if strategy.id in PATTERNS:
            from scripts.research_strategies import BASELINE
            method = BASELINE
        super().__init__(data, method=method, **kwargs)
        self.strategy = strategy
        self.episode_id = episode_id
        self.daily_scans = []
        self.intents = []
        self.active_signal_ids = {}
        self.selected_signal_ids = set()
        self.latest_pool = []

    def _entry_candidates(self, day, rows):
        self._session_observation_start = len(self.daily_scans)
        before = len(self.signals)
        pool = super()._entry_candidates(day, rows)
        self.latest_pool = pool
        for record in self.signals[before:]:
            sid = signal_id(self.strategy, self.episode_id, record['symbol'], record['session'])
            details = rows[record['symbol']].get('entry_details', {})
            observation = dict(record, signal_id=sid, strategy_id=self.strategy.id,
                               strategy_version=self.strategy.version,
                               conditions=details, selected=False)
            self.daily_scans.append(observation)
        return pool

    def _open_position(self, symbol, row, day, signal_row):
        count = len(self.fills)
        super()._open_position(symbol, row, day, signal_row)
        if len(self.fills) > count:
            sid = signal_id(self.strategy, self.episode_id, symbol, signal_row.name.strftime('%Y-%m-%d'))
            if sid not in self.selected_signal_ids:
                raise ValueError('Paper fill has no prior selected scanner signal')
            self.active_signal_ids[symbol] = sid
            self.fills[-1].update(signal_id=sid, strategy_id=self.strategy.id,
                                  strategy_version=self.strategy.version, paper_simulated=True)

    def _record_exit(self, position, quantity, price, day, reason):
        sid = self.active_signal_ids.get(position.symbol)
        if sid is None:
            raise ValueError('Paper exit has no originating scanner signal')
        super()._record_exit(position, quantity, price, day, reason)
        self.fills[-1].update(signal_id=sid, strategy_id=self.strategy.id,
                              strategy_version=self.strategy.version, paper_simulated=True)

    def _after_session(self, day):
        key = day.strftime('%Y-%m-%d')
        # The kernel has applied ranking/capacity. Save selected proposals BEFORE
        # research's terminal-entry cancellation; proposals are never orders.
        selected = {symbol for symbol, row in self.pending_entries.items() if row.name == day}
        for observation in self.daily_scans[getattr(self, '_session_observation_start', 0):]:
            if observation['session'] != key:
                continue
            observation['selected'] = observation['symbol'] in selected
            if observation['selected']:
                sid = observation['signal_id']
                self.selected_signal_ids.add(sid)
                row = self.pending_entries[observation['symbol']]
                self.intents.append(dict(signal_id=sid, strategy_id=self.strategy.id,
                    strategy_version=self.strategy.version, symbol=observation['symbol'], side='BUY',
                    signal_session=key, signal_available_at_assumed=availability_clock(day),
                    paper_only=True, order_submission_authorized=False,
                    signal_close_raw=float(row['close']), signal_atr14_raw=float(row['ATR14']),
                    allocation_pct=self.alloc_pct, max_positions=self.max_positions,
                    execution_model='NEXT_SUPPLIED_REGULAR_SESSION_OPEN', score=observation['score']))
                if self.strategy.id in PATTERNS:
                    from scripts.pattern_strategy import plan_from_row
                    self.intents[-1]['pattern_exit'] = plan_from_row(observation['symbol'], row)
        super()._after_session(day)


def strategy_episode(dataset, spec, strategy, fold, cost_case):
    market = getattr(dataset, 'market_wide', False)
    population = (dict(policy={k:dataset.coverage['policy'][k] for k in
                              ('source','status','asset_class','tradable')}, benchmark=spec['universe']['benchmark'])
                  if market else spec['universe'])
    episode = dict(strategy_id=strategy.id, version=strategy.version, fold_id=fold['id'],
                   warmup_start=fold['warmup_start'], test_start=fold['test_start'],
                   universe=population, portfolio=spec['portfolio'], cost_case=cost_case,
                   definition_sha256=strategy.describe()['definition_sha256'])
    return episode, canonical_hash(episode)


def evaluate_strategy(dataset, spec, strategy, fold, cost_case, as_of, *, deadline=None, stop_file=None):
    effective = dict(fold, test_end=as_of)
    market = getattr(dataset, 'market_wide', False)
    if market:
        from scripts.strategy_market_replay import prepare_market
        data, missing, dividends, observation_count = prepare_market(dataset, spec, strategy, effective,
            deadline=deadline, stop_file=stop_file)
    else:
        data = {symbol: strategy.prepare(dataset.raw[symbol], dataset.features[symbol], effective)
                for symbol in spec['universe']['symbols']}
        dividends = dataset.dividends
    # Daily additions/removals must not change the strategy's logical episode
    # identity. Exact current membership is separately bound in dataset/run hashes.
    episode, episode_id = strategy_episode(dataset, spec, strategy, fold, cost_case)
    portfolio = ScannerPortfolio
    if strategy.id in PATTERNS:
        from scripts.bull_flag_strategy import scanner_portfolio_class
        if strategy.id != 'bull_flag_breakout_v1':
            from scripts.pattern_strategy import scanner_portfolio_class
        portfolio = scanner_portfolio_class(ScannerPortfolio)
    engine = portfolio(data, strategy=strategy, episode_id=episode_id,
        dividends=dividends, deadline=deadline, stop_file=stop_file,
        trade_cost=cost_case['fee_per_fill'], slippage=cost_case['slippage_bps']/10000,
        **spec['portfolio'])
    if market:
        engine.dates = [pd.Timestamp(d) for d in dataset.calendar.index if effective['test_start'] <= d <= as_of]
    engine.run()
    intents = {intent['signal_id']: dict(intent) for intent in engine.intents}
    historical_terminal = {sid for sid, intent in intents.items() if intent['signal_session'] == as_of}
    # Historical positions are research outcomes, not the execution account's
    # current holdings. Preserve that replay, but rank today's qualifying pool
    # before its simulated capacity gate. The executor applies actual cash and
    # position limits; no historical or future fill is manufactured here.
    ranked = sorted(engine.latest_pool, key=lambda item: item[1], reverse=True)[:spec['portfolio']['top_n']]
    today_ids = set()
    for symbol, score, row in ranked:
        sid = signal_id(strategy, episode_id, symbol, as_of)
        today_ids.add(sid)
        if sid not in intents:
            intents[sid] = dict(signal_id=sid, strategy_id=strategy.id, strategy_version=strategy.version,
                symbol=symbol, side='BUY', signal_session=as_of,
                signal_available_at_assumed=availability_clock(pd.Timestamp(as_of)),
                paper_only=True, order_submission_authorized=False,
                signal_close_raw=float(row['close']), signal_atr14_raw=float(row['ATR14']),
                allocation_pct=engine.alloc_pct, max_positions=engine.max_positions,
                execution_model='NEXT_SUPPLIED_REGULAR_SESSION_OPEN', score=float(score))
            if strategy.id in PATTERNS:
                from scripts.pattern_strategy import plan_from_row
                intents[sid]['pattern_exit'] = plan_from_row(symbol, row)
    for observation in engine.daily_scans:
        if observation['session'] == as_of:
            observation['historical_portfolio_selected'] = observation['selected']
            observation['selected'] = observation['signal_id'] in today_ids
    calendar = dataset.calendar
    for intent in intents.values():
        upcoming = calendar.index[calendar.index > intent['signal_session']]
        next_session = str(upcoming[0]) if len(upcoming) else None
        intent['next_session'] = next_session
        intent['next_open_at'] = None
        if next_session:
            opening = str(calendar.loc[next_session, 'open'])
            if time.fromisoformat(opening) != time(9, 30):
                raise ValueError('Shared paper model requires a supplied 09:30 regular open')
            local = datetime.combine(date.fromisoformat(next_session), time.fromisoformat(opening),
                                     ZoneInfo('America/New_York'))
            intent['next_open_at'] = local.astimezone(timezone.utc).isoformat()
            if intent['signal_available_at_assumed'] >= intent['next_open_at']:
                raise ValueError('Intent availability does not precede next supplied open')
        intent['status'] = 'PAPER_PROPOSAL' if next_session else 'WAITING_FOR_NEXT_CALENDAR_SESSION'
    for fill in engine.fills:
        intent = intents[fill['signal_id']]
        if fill['side'] == 'BUY' and fill['session'] != intent['next_session']:
            raise ValueError('Paper entry is not on its supplied next session')
    latest = [record for record in engine.daily_scans if record['session'] == as_of]
    today_intents = [intents[signal_id(strategy, episode_id, symbol, as_of)] for symbol, _, _ in ranked]
    buys = {fill['signal_id'] for fill in engine.fills if fill['side'] == 'BUY'}
    exits = {fill['signal_id'] for fill in engine.fills if fill['side'] == 'SELL'}
    lifecycle = []
    for sid, intent in intents.items():
        related = [fill for fill in engine.fills if fill['signal_id'] == sid]
        bought = sum(fill['quantity'] for fill in related if fill['side'] == 'BUY')
        sold = sum(fill['quantity'] for fill in related if fill['side'] == 'SELL')
        lifecycle.append(dict(signal_id=sid, symbol=intent['symbol'], signal_session=intent['signal_session'],
            status=('CLOSED' if bought and bought == sold else 'OPEN' if bought else 'UNFILLED_PROPOSAL'),
            bought_quantity=bought, sold_quantity=sold,
            fills=related, strategy_id=strategy.id, strategy_version=strategy.version))
    result = dict(strategy=strategy.describe(), episode_id=episode_id, episode=episode, as_of=as_of,
        latest_observations=latest, top_candidates=today_intents,
        current_selection=dict(scope='QUALIFYING_POOL_BEFORE_SIMULATED_PORTFOLIO_CAPACITY',
            qualifying_symbols=len(engine.latest_pool), proposal_limit=spec['portfolio']['top_n'],
            selected_proposals=len(today_intents), historical_terminal_proposals=len(historical_terminal),
            actual_account_limits_enforced_by='PAPER_EXECUTOR'),
        observations=engine.daily_scans, intents=list(intents.values()), fills=engine.fills,
        lifecycle=lifecycle, actions=engine.action_events, ledger=engine.reconciliations,
        metrics=engine.summary(), funnel=dict(observations=len(engine.daily_scans),
            qualifying_closes=sum(record['consumed'] for record in engine.daily_scans),
            selected_proposals=len(intents), paper_entries=len(buys), signals_with_exits=len(exits)),
        research_qualified=False, order_submission_authorized=False)
    if market:
        from scripts.strategy_market_replay import merge_latest
        merge_latest(result, missing, strategy, episode_id)
        result['funnel']['observations'] = observation_count
        result['metrics']['signal_observations'] = observation_count
    return result


def _report(results):
    chunks = ['<!doctype html><meta charset="utf-8"><title>Strategy scanner trace</title>',
              '<h1>Strategy scanner â€” paper simulation</h1>',
              '<p>Captured daily data; modeled next-open fills. No broker orders.</p>']
    for result in results:
        strategy = result['strategy']
        chunks.append('<h2>'+html.escape(strategy['name'])+' Â· '+html.escape(result['as_of'])+'</h2>')
        chunks.append('<p>'+html.escape(json.dumps(result['funnel'], sort_keys=True))+'</p>')
        chunks.append('<details><summary>Buy rules</summary><ul>'+''.join(
            '<li>'+html.escape(rule)+'</li>' for rule in strategy['buy_rules'])+'</ul></details>')
        if not result['top_candidates']:
            chunks.append('<p>No buy signals selected for this session.</p>')
        chunks.append('<h3>Signal to outcome</h3><table border="1"><tr><th>Symbol</th>'
            '<th>Signal date</th><th>Paper entry</th><th>Paper exits</th>'
            '<th>Realized net P&amp;L</th><th>State</th><th>Signal ID</th></tr>')
        for trace in result['lifecycle']:
            entry = '; '.join(f"{fill['session']} @ {fill['raw_price']:.4f}" for fill in trace['fills'] if fill['side'] == 'BUY')
            exits = '; '.join(f"{fill['session']} @ {fill['raw_price']:.4f} ({fill['reason']})"
                              for fill in trace['fills'] if fill['side'] == 'SELL')
            realized = sum(fill['net_pnl'] for fill in trace['fills'] if fill['side'] == 'SELL')
            cells = (trace['symbol'],trace['signal_session'],entry or 'Unfilled proposal',
                     exits or 'No exit',f'{realized:.2f}',trace['status'],trace['signal_id'])
            chunks.append('<tr>'+''.join('<td>'+html.escape(str(cell))+'</td>' for cell in cells)+'</tr>')
        chunks.append('</table>')
        chunks.append('<h3>Latest daily scan</h3><table border="1"><tr>'
                      '<th>Symbol</th><th>Decision</th><th>Failed conditions</th><th>Selected</th></tr>')
        for row in result['latest_observations']:
            cells = (row['symbol'],row['reason'],', '.join(row['conditions'].get('failures',[])),row['selected'])
            chunks.append('<tr>'+''.join('<td>'+html.escape(str(cell))+'</td>' for cell in cells)+'</tr>')
        chunks.append('</table>')
    return '\n'.join(chunks)


def run_scan(spec, *, as_of, output, strategy_ids=None, fold_id=None, cost_case_id='base_cost', now=None,
             total_deadline=None):
    validate_spec(spec)
    if date.fromisoformat(as_of).isoformat() != as_of:
        raise ValueError('Noncanonical scanner session')
    selected = select_strategies(strategy_ids)
    if len(selected) > spec['max_runs'] or any(
            strategy.research_method not in spec['strategies'] for strategy in selected):
        raise ValueError('Scanner strategy is outside the declared study budget/methods')
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now < datetime.fromisoformat(availability_clock(pd.Timestamp(as_of))):
        raise ValueError('Requested daily session is not available at the assumed cutoff')
    folds = [fold for fold in spec['folds'] if fold['test_start'] <= as_of <= fold['test_end']
             and (fold_id is None or fold['id'] == fold_id)]
    if len(folds) != 1:
        raise ValueError('Scanner requires one declared evaluation episode containing the session')
    costs = [case for case in spec['cost_cases'] if case['id'] == cost_case_id]
    if len(costs) != 1:
        raise ValueError('Unknown scanner cost case')
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repo):
        raise ValueError('Scanner output must be outside the source tree')
    # Claim once before loading or running. Never silently overwrite failed runs.
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'claim.json', dict(status='CLAIMED_NO_REPLAY', as_of=as_of,
        strategy_ids=[strategy.id for strategy in selected], broker_operations=0))
    try:
        deadline = min(clock.monotonic()+spec['max_seconds'], total_deadline or float('inf'))
        dataset = load_dataset(spec)
        if as_of not in dataset.calendar.index:
            raise ValueError('Scanner session is absent from supplied exchange calendar')
        source = output/'source'
        source.mkdir()
        source_hashes = {}
        for name in SOURCE_FILES:
            body = (repo/'scripts'/name).read_bytes()
            (source/name).write_bytes(body)
            source_hashes[name] = hashlib.sha256(body).hexdigest()
        save_json(output/'specification.json', spec)
        save_json(output/'dataset.json', dataset.provenance)
        results = []
        for strategy in selected:
            result = evaluate_strategy(dataset, spec, strategy, folds[0], costs[0], as_of,
                                       deadline=deadline, stop_file=output/'STOP')
            results.append(result)
            folder = output/strategy.id
            folder.mkdir()
            for key in ('latest_observations','top_candidates','observations','intents','fills','lifecycle',
                        'actions','ledger','metrics'):
                if getattr(dataset, 'market_wide', False) and key in {
                        'latest_observations','observations','intents','fills','lifecycle','actions'}:
                    body=json.dumps(result[key],sort_keys=True,default=json_default,allow_nan=False,
                                    separators=(',',':')).encode()
                    with (folder/(key+'.json.gz')).open('xb') as handle:
                        handle.write(gzip.compress(body,mtime=0))
                else:
                    save_json(folder/(key+'.json'), result[key])
        if any(hashlib.sha256((repo/'scripts'/name).read_bytes()).hexdigest() != expected
               for name, expected in source_hashes.items()):
            raise ValueError('Scanner source changed during execution')
        public = [dict(strategy=result['strategy'], episode_id=result['episode_id'],
            as_of=as_of, funnel=result['funnel'], metrics=result['metrics'],
            current_selection=result['current_selection']) for result in results]
        if getattr(dataset, 'market_wide', False):
            for item,result in zip(public,results):
                item['observation_storage']=result['observation_storage']
        candidates = [intent for result in results for intent in result['top_candidates']]
        summary = dict(schema='jbravo.strategy-scan.v1', status='COMPLETED_PAPER_SIMULATION',
            as_of=as_of, generated_at_utc=now.astimezone(timezone.utc).isoformat(),
            run_id=canonical_hash(dict(spec=spec, as_of=as_of, sources=source_hashes,
                dataset_sha256=dataset.provenance['dataset_sha256'],
                strategies=[item['strategy']['definition_sha256'] for item in public], cost_case=cost_case_id)),
            strategies=public, top_candidates=candidates, source_sha256=source_hashes,
            dataset_sha256=dataset.provenance['dataset_sha256'], research_qualified=False,
            order_submission_authorized=False, broker_operations=0,
            limitations=['Retrospective selected universe; research acceptance remains pending',
                'Daily snapshot availability assumed 20:15 Eastern; execution simulated at next regular open',
                'Not the existing premarket passive-limit executor; no order submission or scheduler',
                'Future daily runs require a new verified capture/specification and output directory'])
        if getattr(dataset, 'market_wide', False):
            summary['coverage'] = dataset.provenance['selection']
        save_json(output/'result.json', summary)
        save_json(output/'paper_handoff.json', dict(schema='jbravo.strategy-paper-proposals.v1',
            scan_run_id=summary['run_id'], as_of=as_of, proposals=candidates,
            paper_only=True, order_submission_authorized=False))
        columns = ['strategy_id', 'strategy_version', 'signal_id', 'symbol', 'side',
                   'signal_session', 'signal_available_at_assumed', 'next_session',
                   'next_open_at', 'score', 'status', 'paper_only', 'order_submission_authorized']
        pd.DataFrame(candidates, columns=columns).to_csv(output/'top_candidates.csv', index=False)
        display = results
        if getattr(dataset, 'market_wide', False):
            display = [dict(r,lifecycle=r['lifecycle'][:500],latest_observations=r['latest_observations'][:500]) for r in results]
        preview_note = ('<p>Preview limited to 500 rows per table. Complete current decisions and sparse historical '
                        'portfolio traces are in the compressed JSON artifacts.</p>'
                        if getattr(dataset, 'market_wide', False) else '')
        (output/'report.html').write_text(_report(display)+preview_note,encoding='utf-8')
        bindings = []
        for path in sorted(output.rglob('*')):
            if path.is_file():
                body = path.read_bytes()
                bindings.append(dict(path=str(path.relative_to(output)).replace('\\','/'),
                                     bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
        save_json(output/'manifest.json', dict(bindings=bindings))
        return summary
    except Exception as error:
        save_json(output/'failure.json', dict(status='FAILED_NO_REPLAY', error_type=type(error).__name__,
                                             broker_operations=0))
        raise


def run_cli(args):
    from scripts.strategy_library import list_strategies
    if args.mode == 'strategy-library':
        print(json.dumps(dict(strategies=list_strategies()),indent=2))
        return 0
    try:
        if not args.study_spec or not args.as_of or not args.output_dir:
            raise ValueError('Strategy scan requires --study-spec, --as-of and a new --output-dir')
        spec = strict_json(bounded_read(args.study_spec))
        result = run_scan(spec, as_of=args.as_of, output=args.output_dir,
            strategy_ids=args.strategy, fold_id=args.strategy_fold, cost_case_id=args.strategy_cost_case)
        print(json.dumps(dict(status=result['status'],run_id=result['run_id'],
                              candidates=len(result['top_candidates']),broker_operations=0)))
        return 0
    except Exception as error:
        print(json.dumps(dict(status='STRATEGY_SCAN_FAILED',error_type=type(error).__name__,
                              reason=str(error))),file=__import__('sys').stderr)
        return 1


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['strategy-library', 'strategy-scan'], required=True)
    parser.add_argument('--strategy', action='append')
    parser.add_argument('--study-spec', type=Path)
    parser.add_argument('--as-of')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--strategy-fold')
    parser.add_argument('--strategy-cost-case', default='base_cost')
    return run_cli(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
