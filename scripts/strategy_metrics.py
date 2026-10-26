"""Metrics derived exclusively from the current candidate strategy backtest.

Pure projection of completed-signal statistics; no replay, market data, account
performance, ranking, thresholds or orders. Keep unknown statistics nullable.
"""
import json
import math
import re

from scripts.strategy_backtest import canonical, digest, number, require

SCHEMA = 'jbravo.candidate-strategy-metrics.v1'
BASIS = 'STRATEGY_BACKTEST_CLOSED_SIGNALS'


def count(value):
    require(type(value) is int and value >= 0, 'invalid_strategy_metric_count')
    return value


def metric_row(row, proposal, contract, backtest_id):
    require(row['symbol'] == proposal['symbol'] and
        row['strategy_id'] == proposal['strategy_id'] == contract['strategy_id'] and
        row['strategy_version'] == proposal['strategy_version'] == contract['strategy_version'] and
        row['current_signal_id'] == proposal['signal_id'] and
        row['current_signal_session'] == proposal['signal_session'] and
        row['current_proposal_state'] == 'UNFILLED_FUTURE_OUTCOME_UNKNOWN',
        'strategy_metrics_candidate_lineage_mismatch')
    trades = count(row['closed_signals'])
    wins, losses, flat = (count(row[k]) for k in ('winning_signals','losing_signals','breakeven_signals'))
    require(wins + losses + flat == trades, 'strategy_metrics_outcome_count_mismatch')
    ids = row['completed_signal_ids']
    require(len(ids) == len(set(ids)) == trades and all(isinstance(x,str) and
        re.fullmatch('[0-9a-f]{64}', x) and x != proposal['signal_id'] for x in ids),
        'strategy_metrics_closed_signal_population_mismatch')
    require(re.fullmatch('[0-9a-f]{64}', row['closed_signal_ledger_sha256']) is not None,
        'strategy_metrics_missing_ledger_identity')
    net = number(row['closed_trade_net_pnl'])
    fees, slippage = (number(row[k]) for k in ('fees_closed_signals','slippage_closed_signals'))
    drawdown = number(row['max_closed_signal_drawdown_cash'])
    require(fees >= 0 and slippage >= 0 and drawdown <= 0 and row['trade_pnl_excludes_cash_dividends'] is True,
        'strategy_metrics_cost_or_units_mismatch')
    if trades:
        require(row['sample_status'] == 'HAS_COMPLETED_SIGNALS' and
            math.isclose(number(row['win_rate_pct']), wins / trades * 100, abs_tol=1e-9) and
            math.isclose(number(row['expectancy_net_per_closed_signal']), net / trades, abs_tol=1e-9) and
            number(row['average_hold_calendar_days']) >= 0,
            'strategy_metrics_statistic_mismatch')
    else:
        require(row['sample_status'] == 'NO_COMPLETED_SIGNALS' and
            all(row[k] is None for k in ('win_rate_pct','expectancy_net_per_closed_signal','average_hold_calendar_days')) and
            net == fees == slippage == drawdown == 0, 'strategy_metrics_invented_empty_statistics')
    expected_pf = 'DEFINED' if losses else 'UNDEFINED_NO_LOSSES' if trades else 'NO_CLOSED_SIGNALS'
    require(row['profit_factor_status'] == expected_pf and
        (number(row['profit_factor']) >= 0 if losses else row['profit_factor'] is None),
        'strategy_metrics_profit_factor_mismatch')
    for k in ('prior_open_signals','prior_unfilled_proposals','prior_qualifying_setups'):count(row[k])
    return dict(symbol=row['symbol'],signal_id=proposal['signal_id'],signal_session=proposal['signal_session'],
        strategy_id=contract['strategy_id'],strategy_version=contract['strategy_version'],
        definition_sha256=contract['definition_sha256'],backtest_result_id=backtest_id,
        trades=trades,wins=wins,losses=losses,breakeven=flat,win_rate_pct=row['win_rate_pct'],
        net_pnl=net,expectancy=row['expectancy_net_per_closed_signal'],profit_factor=row['profit_factor'],
        profit_factor_status=row['profit_factor_status'],max_drawdown_cash=drawdown,
        average_hold_calendar_days=row['average_hold_calendar_days'],fees=fees,slippage=slippage,
        sample_status=row['sample_status'],prior_open_signals=row['prior_open_signals'],
        prior_unfilled_proposals=row['prior_unfilled_proposals'],current_proposal_state=row['current_proposal_state'],
        closed_signal_ledger_sha256=row['closed_signal_ledger_sha256'],
        currency='USD',drawdown_units='USD_CLOSED_SIGNAL_PNL',trade_pnl_excludes_cash_dividends=True,
        metric_basis=BASIS,disposition='REPORT_ONLY_NO_PERFORMANCE_FILTER')


def build_metrics(selected, backtest, *, primary_run_id):
    from scripts.strategy_pipeline import validate_backtest
    validate_backtest(backtest, selected, primary_run_id)
    contract = selected['strategy_contract']
    definition = backtest.get('strategy_definition', backtest.get('strategy', {}))
    require(definition.get('id') == contract['strategy_id'] and definition.get('version') == contract['strategy_version'] and
        definition.get('definition_sha256') == contract['definition_sha256'],
        'strategy_metrics_backtest_definition_mismatch')
    require(backtest['archive_id'] == selected['archive_id'] and backtest['archive_sha256'] == selected['archive_sha256'] and
        backtest['scan_run_id'] == selected['scan_run_id'] and backtest['research_qualified'] is False,
        'strategy_metrics_backtest_archive_mismatch')
    rows = [metric_row(row, proposal, contract, backtest['result_id'])
        for row, proposal in zip(backtest['results'], selected['proposals'])]
    closed_ids = [x for row in backtest['results'] for x in row['completed_signal_ids']]
    require(len(closed_ids) == len(set(closed_ids)), 'strategy_metrics_duplicate_closed_signal')
    result = dict(schema=SCHEMA,mode='REPORT_ONLY',primary_run_id=primary_run_id,
        strategy_contract=contract,signal_session=backtest['signal_session'],
        screener_batch_sha256=digest(selected),backtest_result_id=backtest['result_id'],backtest_sha256=digest(backtest),
        archive_id=selected['archive_id'],archive_sha256=selected['archive_sha256'],scan_run_id=selected['scan_run_id'],
        candidate_signal_ids=backtest['candidate_signal_ids'],input_candidates=len(rows),output_candidates=len(rows),
        status='COMPLETED_STRATEGY_METRICS' if rows else 'SKIPPED_NO_CANDIDATES',
        rows=rows,metric_basis=BASIS,evaluation=backtest['parameters'],data=backtest['data'],
        symbols_with_completed_history=sum(r['trades'] > 0 for r in rows),
        performance_filter_applied=False,ranking_applied=False,research_qualified=False,
        broker_operations=0,order_submission_authorized=False,
        limitations=['Completed signals include all partial exits and costs from the bound backtest',
            'Cash drawdown is not a portfolio drawdown percentage',
            'Unknown current-proposal outcomes and unavailable sample statistics remain unknown',
            'These research simulation results are separate from broker account performance'])
    result = json.loads(canonical(result))
    result['result_id'] = digest(result)
    require(len(canonical(result)) <= 24*1024, 'strategy_metrics_report_oversize')
    return result


def validate_metrics(metrics, backtest, selected, primary_run_id):
    require(isinstance(metrics,dict) and metrics.get('schema') == SCHEMA, 'missing_strategy_metrics')
    require(metrics == build_metrics(selected,backtest,primary_run_id=primary_run_id),
        'strategy_metrics_lineage_or_result_mismatch')
