"""Pullback/Reclaim execution contract and shared-kernel exit decisions.

Inert on import. This module does not grant order authority, open a connection,
fetch prices, or submit orders. The broker worker must supply verified inputs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
import re

from scripts import strategy_pipeline as pipeline
from scripts.strategy_backtest import digest, require

SCHEMA = 'jbravo.premarket-strategy-execution-handoff.v1'
EXECUTION_MODEL = 'NEXT_PREMARKET_0400_FROZEN_LIMIT_V1'
PORTFOLIO = dict(initial_cash=100000.0, alloc_pct=.05, top_n=3,
                 max_positions=4, trail_pct=.03, max_hold_days=7, atr_multiple=1.0)
NY = pipeline.NY


def utc(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(isinstance(value, datetime) and value.tzinfo is not None and
            value.utcoffset() is not None, 'execution_ambiguous_timestamp')
    return value.astimezone(timezone.utc)


def number(value, *, positive=False):
    require(not isinstance(value, bool), 'execution_invalid_number')
    try:
        result = Decimal(str(value))
    except Exception as error:
        raise ValueError('execution_invalid_number') from error
    require(result.is_finite() and (result > 0 if positive else result >= 0),
            'execution_invalid_number')
    return result


def prepare_handoff(record, rows, *, now):
    """Bind the exact published population, strategy, backtest and Metrics.

    Callers must separately validate the source-bound completed primary report.
    This pure function never turns a report-only research result into authority.
    """
    from scripts.strategy_library import get_strategy
    from scripts.signal_handoff import candidate_values
    now = utc(now)
    pipeline.assess(record, now=now, max_age_seconds=43200)
    batch, backtest, metrics = record['batch'], record['backtest'], record['metrics']
    definition = get_strategy(pipeline.STRATEGY).describe()
    require(batch['strategy_contract'] == dict(strategy_id=definition['id'],
        strategy_version=definition['version'], definition_sha256=definition['definition_sha256']),
        'execution_strategy_definition_changed')
    require(backtest['strategy_definition'] == json.loads(json.dumps(definition)),
            'execution_backtest_definition_changed')
    require(backtest['parameters']['portfolio'] == PORTFOLIO,
            'execution_portfolio_rules_changed')
    proposals = batch['proposals']
    opening=datetime.combine(date.fromisoformat(record['session']['next_session']),
        datetime.min.time().replace(hour=4),NY).astimezone(timezone.utc)
    require(utc(record['finished_at']) <= now < opening, 'execution_premarket_preparation_expired')
    require(isinstance(rows, list) and len(rows) == len(proposals) <= PORTFOLIO['top_n'],
            'execution_candidate_population_changed')
    require(len({p['symbol'] for p in proposals}) == len(proposals) and
            len({p['signal_id'] for p in proposals}) == len(proposals),
            'execution_duplicate_signal')
    by_symbol = {r['symbol']: r for r in rows}
    require(len(by_symbol) == len(rows), 'execution_duplicate_candidate')
    for proposal in proposals:
        require(proposal['symbol'] in by_symbol, 'execution_missing_candidate')
        row = by_symbol[proposal['symbol']]
        metadata = row['score_breakdown']
        require(metadata == dict(schema='jbravo.strategy-candidate-lineage.v1',
            proposal=proposal, archive_id=batch['archive_id'], archive_sha256=batch['archive_sha256'],
            scan_run_id=batch['scan_run_id'], dataset_sha256=backtest['data']['dataset_sha256'],
            definition_sha256=definition['definition_sha256']), 'execution_candidate_lineage_changed')
        stamp = row['run_date'].isoformat() if isinstance(row['run_date'], date) else row['run_date']
        require(stamp == record['session']['signal_session'] and
            utc(row['run_ts_utc']) == utc(record['candidate_run_ts_utc']) and
            utc(row['timestamp']) == utc(datetime.combine(date.fromisoformat(stamp),datetime.min.time(),NY)) and
            float(number(row['close'], positive=True)) == proposal['signal_close_raw'] and
            float(number(row['entry_price'],positive=True)) == proposal['signal_close_raw'] and
            float(row['score']) == proposal['score'] and math.isfinite(float(row['score'])) and
            row['source'] == 'strategy_pullback_reclaim_v2' and row['passed_gates'] is True and
            row['gate_fail_reason'] is None and row['ml_weight_used'] == 0,
            'execution_candidate_values_changed')
        require(proposal['allocation_pct'] == .05 and proposal['max_positions'] == 4 and
            proposal['next_session'] == record['session']['next_session'] and
            utc(proposal['next_open_at']) == utc(record['session']['next_open_at']) and
            proposal['signal_session'] == record['session']['signal_session'] and
            utc(proposal['signal_available_at_assumed']) == utc(record['session']['signal_available_at']) and
            proposal['strategy_id'] == definition['id'] and proposal['strategy_version'] == definition['version'] and
            proposal['side'] == 'BUY' and re.fullmatch('[0-9a-f]{64}', proposal['signal_id']) is not None and
            proposal['paper_only'] is True and proposal['order_submission_authorized'] is False,
            'execution_proposal_scope_changed')
        number(proposal['signal_atr14_raw'], positive=True)
    result = dict(schema=SCHEMA, primary_run_id=record['run_id'],
        primary_report_sha256=digest(record), candidate_run_ts_utc=record['candidate_run_ts_utc'],
        batch_sha256=digest(batch), candidate_values=candidate_values(rows),
        strategy_contract=deepcopy(batch['strategy_contract']),
        backtest_result_id=backtest['result_id'], metrics_result_id=metrics['result_id'],
        session=deepcopy(record['session']), portfolio=deepcopy(PORTFOLIO), proposals=deepcopy(proposals),
        execution_policy=dict(model=EXECUTION_MODEL,entry_open_at=opening.isoformat(),
            cancel_at=(opening+timedelta(minutes=35)).isoformat(),limit_buffer_pct=.5,
            broker_order_type='LIMIT_DAY_EXTENDED',reference='CAPTURED_AFTER_HOURS_CLOSE',
            backtest_entry_model=record['session']['execution_model'],
            backtest_alignment='PREMARKET_REPLAY_PENDING'),
        status='PREPARED' if proposals else 'NO_BUY_SIGNALS', paper_only=True,
        order_submission_authorized=False, research_qualified=False,
        broker_operations=0, qualification_basis='EXPLICIT_PAPER_STRATEGY_TEST')
    result['handoff_id'] = digest(result)
    return result


def entry_window(handoff, *, now, clock_next_regular_open, clock_timestamp):
    """Prepare after the extended close, submit at the next supplied 04:00 ET.

    A closed regular-market clock is normal during premarket. Validate its next
    regular open/session and observation time instead of requiring is_open=True.
    """
    now, opening = utc(now), utc(handoff['execution_policy']['entry_open_at'])
    observed=utc(clock_timestamp)
    require(observed <= now and (now-observed).total_seconds()<=60 and
        utc(clock_next_regular_open)==utc(handoff['session']['next_open_at']),
        'execution_clock_session_unavailable')
    require(opening.astimezone(NY).date().isoformat() == handoff['session']['next_session'] and
        opening.astimezone(NY).strftime('%H:%M:%S') == '04:00:00' and
        handoff['execution_policy']['model']==EXECUTION_MODEL,
        'execution_session_changed')
    if now < opening:
        return 'WAIT_PREMARKET_OPEN'
    require(now < utc(handoff['execution_policy']['cancel_at']), 'execution_entry_window_expired')
    return 'PREMARKET_OPEN'


def size_entry(cash, opening_price, *, occupied_slots, reserved_cash=0):
    """Sequential 5%-cash whole shares, using the maximum allowed fill price."""
    require(type(occupied_slots) is int and 0 <= occupied_slots <= 4,
            'execution_invalid_position_count')
    cash, reserved, price = number(cash), number(reserved_cash), number(opening_price, positive=True)
    require(reserved <= cash, 'execution_reserved_cash_exceeds_balance')
    if occupied_slots >= 4:
        return 0
    return int(((cash - reserved) * Decimal('.05') / price).to_integral_value(rounding=ROUND_FLOOR))


def client_order_id(account_id, signal_id, *, role='entry', sequence=0):
    """Stable across primary reruns and crashes; broker maximum is 48 chars."""
    require(isinstance(account_id, str) and 1 <= len(account_id) <= 128 and
        re.fullmatch('[0-9a-f]{64}', signal_id) is not None and
        role in {'entry', 'stop', 'exit', 'partial'} and
        type(sequence) is int and sequence >= 0, 'execution_order_identity_invalid')
    return 'jbr2-' + hashlib.sha256(json.dumps(
        [account_id, signal_id, role, sequence], separators=(',', ':')).encode()).hexdigest()[:40]


def after_hours_reference(bars, *, symbol, signal_session, observed_at,
                          feed, adjustment, currency, complete):
    """Preserve the latest observed after-hours bar; never invent a 20:00 print."""
    cutoff=datetime.combine(date.fromisoformat(signal_session),datetime.min.time().replace(hour=20),NY)
    start=cutoff-timedelta(hours=4)
    require(feed=='sip' and adjustment=='raw' and currency=='USD' and complete is True and
        utc(observed_at)>=utc(cutoff)+timedelta(minutes=15), 'execution_reference_capture_invalid')
    require(isinstance(bars,list) and len(bars)<=240, 'execution_reference_population_invalid')
    previous=None
    for bar in bars:
        stamp=utc(bar['t'])
        require(utc(start)<=stamp<utc(cutoff) and (previous is None or previous<stamp),
                'execution_reference_bar_outside_session')
        prices=[number(bar[k],positive=True) for k in ('o','h','l','c')]
        require(prices[2]<=min(prices[0],prices[3])<=max(prices[0],prices[3])<=prices[1] and
            number(bar['v'],positive=True)>0, 'execution_reference_bar_invalid')
        previous=stamp
    result=dict(status='AVAILABLE' if bars else 'NO_AFTER_HOURS_REFERENCE',symbol=symbol,
        signal_session=signal_session,price=float(bars[-1]['c']) if bars else None,
        last_bar_at=bars[-1]['t'] if bars else None,observed_at=utc(observed_at).isoformat(),
        cutoff_at=utc(cutoff).isoformat(),feed=feed,adjustment=adjustment,currency=currency,
        complete=True,bars_sha256=digest(bars))
    result['reference_id']=digest(result)
    return result


def build_entry_order(handoff, *, signal_id, account_id, cash, reference,
                      occupied_slots, now, clock_next_regular_open, clock_timestamp, reserved_cash=0):
    """Make a frozen-limit premarket plan, without granting order authority."""
    require(handoff.get('schema') == SCHEMA and
        digest({k:v for k,v in handoff.items() if k!='handoff_id'}) == handoff.get('handoff_id') and
        handoff['portfolio'] == PORTFOLIO and handoff['paper_only'] is True and
        handoff['order_submission_authorized'] is False, 'execution_handoff_changed')
    require(entry_window(handoff,now=now,clock_next_regular_open=clock_next_regular_open,
        clock_timestamp=clock_timestamp) == 'PREMARKET_OPEN', 'execution_entry_before_premarket_open')
    now=utc(now)
    require(now<utc(handoff['execution_policy']['entry_open_at'])+timedelta(seconds=60),
            'execution_initial_submission_window_expired')
    matches=[p for p in handoff['proposals'] if p['signal_id']==signal_id]
    require(len(matches)==1, 'execution_order_signal_not_in_handoff')
    proposal=matches[0]
    require(reference['status']=='AVAILABLE' and reference['symbol']==proposal['symbol'] and
        reference['signal_session']==handoff['session']['signal_session'] and
        reference['feed']=='sip' and reference['adjustment']=='raw' and reference['currency']=='USD' and
        utc(reference['observed_at'])<utc(handoff['execution_policy']['entry_open_at']) and
        digest({k:v for k,v in reference.items() if k!='reference_id'})==reference['reference_id'],
        'execution_after_hours_reference_unavailable_or_changed')
    price=(number(reference['price'],positive=True)*Decimal('1.005')).quantize(Decimal('.01'),rounding=ROUND_FLOOR)
    quantity=size_entry(cash,price,occupied_slots=occupied_slots,reserved_cash=reserved_cash)
    lineage=dict(handoff_id=handoff['handoff_id'],primary_run_id=handoff['primary_run_id'],
        strategy_contract=deepcopy(handoff['strategy_contract']),signal_id=signal_id,
        symbol=proposal['symbol'],backtest_result_id=handoff['backtest_result_id'],
        metrics_result_id=handoff['metrics_result_id'],batch_sha256=handoff['batch_sha256'],
        execution_model=EXECUTION_MODEL,reference_id=reference['reference_id'],
        backtest_alignment=handoff['execution_policy']['backtest_alignment'])
    return dict(status='ORDER_PLANNED' if quantity else 'SKIP_CAPACITY_OR_WHOLE_SHARE_BUDGET',
        request=dict(symbol=proposal['symbol'],qty=quantity,side='buy',type='limit',limit_price=float(price),time_in_force='day',
            extended_hours=True,client_order_id=client_order_id(account_id,signal_id)),lineage=lineage,
        sizing=dict(cash=str(number(cash)),reserved_cash=str(number(reserved_cash)),
            reference_price=str(reference['price']),reference_at=reference['last_bar_at'],
            maximum_fill_price=str(price),reference_required='RAW_SIP_AFTER_HOURS',allocation_pct=.05,occupied_slots=occupied_slots,
            prepared_at=now.isoformat()),order_submission_authorized=False,broker_operations=0)


def checked_order(request, order):
    require(isinstance(order,dict) and order.get('client_order_id')==request['client_order_id'] and
        isinstance(order.get('id'),str) and bool(order['id']) and
        all(order.get(k)==request[k] for k in ('symbol','side','type','time_in_force')) and
        number(order.get('qty'),positive=True)==request['qty'] and
        number(order.get('limit_price'),positive=True)==number(request['limit_price'],positive=True) and
        order.get('extended_hours') is True, 'execution_broker_order_changed')
    filled=number(order.get('filled_qty',0))
    require(filled <= request['qty'], 'execution_broker_fill_exceeds_order')
    if filled:
        require(number(order.get('filled_avg_price'),positive=True)<=number(request['limit_price'],positive=True),
                'execution_broker_fill_exceeds_limit')
    return order


def initial_position(proposal, *, quantity, fill_price, filled_at):
    """Create protection from actual fills and signal-day ATR, never later ATR."""
    from scripts.backtest_core import Position
    import pandas as pd
    require(type(quantity) is int and quantity > 0, 'execution_invalid_fill_quantity')
    price, atr = float(number(fill_price, positive=True)), float(number(proposal['signal_atr14_raw'], positive=True))
    filled_at = utc(filled_at)
    opening = utc(datetime.combine(date.fromisoformat(proposal['next_session']),datetime.min.time().replace(hour=4),NY))
    require(opening <= filled_at and filled_at.astimezone(NY).date().isoformat() == proposal['next_session'],
            'execution_fill_session_mismatch')
    position = Position(symbol=proposal['symbol'], qty=quantity, entry_price=price,
        entry_time=pd.Timestamp(proposal['next_session']), highest_close=price, max_price=price,
        trailing_stop=price * .97, atr_stop=max(0., price - atr))
    return serialize_position(position)


def serialize_position(position):
    value = asdict(position)
    value['entry_time'] = position.entry_time.date().isoformat()
    return value


def completed_bar(position, frame, *, session, next_session, now, rsi_memory=None):
    """Use the exact backtest kernel to advance surviving daily positions.

    Broker fills/stops are reconciled FIRST by the worker. Do not synthesize a
    historical stop fill from OHLC or tighten a broker stop using unfinished bars.
    """
    from scripts.backtest_core import PortfolioBacktester, Position
    import pandas as pd
    now = utc(now)
    require(date.fromisoformat(next_session) > date.fromisoformat(session),
            'execution_next_exit_session_invalid')
    ready = datetime.combine(date.fromisoformat(session), datetime.min.time().replace(hour=20, minute=15), NY)
    require(now >= ready and position['entry_time'] <= session,
            'execution_bar_not_completed')
    day = pd.Timestamp(session)
    require(not frame.index.has_duplicates and frame.index.is_monotonic_increasing and day in frame.index and
        frame.index[-1] == day and len(frame) >= 2, 'execution_incomplete_or_future_bars')
    for column in ('open', 'high', 'low', 'close', 'ATR14', 'ema20', 'macd', 'macd_signal', 'rsi'):
        require(column in frame.columns and math.isfinite(float(frame.loc[day, column])),
                'execution_missing_exit_feature')
    require(float(frame.loc[day, 'ATR14']) > 0, 'execution_invalid_exit_atr')
    values = deepcopy(position); values['entry_time'] = pd.Timestamp(values['entry_time'])
    engine = PortfolioBacktester({position['symbol']: frame}, **PORTFOLIO)
    engine.positions[position['symbol']] = Position(**values)
    engine.rsi_high_memory = deepcopy(rsi_memory or {})
    engine._update_completed_bar(position['symbol'], frame.loc[day], day)
    reasons = engine.pending_exits.get(position['symbol'], [])
    return dict(position=serialize_position(engine.positions[position['symbol']]),
        rsi_memory=engine.rsi_high_memory, pending_exit_reasons=reasons,
        pending_exit_session=next_session if reasons else None,
        completed_session=session, stop_price=engine._protective_stop(engine.positions[position['symbol']])[0],
        partial_quantity=max(1, position['qty']//2) if reasons == ['PARTIAL_5PCT'] and position['qty'] > 1 else 0)


def submit_claimed(store, broker, *, account_id, signal_id, request, lineage,
                   paper_research_entries=False):
    """One durable claim before one POST. Ambiguous outcomes are never retried.

    Store.claim must atomically commit a unique (account,signal,role,sequence)
    row before returning. A repeated claim observes the same client ID; missing
    broker evidence after an uncertain POST remains UNKNOWN and blocks replay.
    No exception message or raw provider response is persisted here.
    """
    from scripts.executor_health import is_paper_endpoint
    require(paper_research_entries is True and getattr(broker, 'paper', None) is True and
        getattr(broker,'account_id',None) == account_id and
        is_paper_endpoint(getattr(broker, 'base_url', None)), 'execution_paper_authority_required')
    request = deepcopy(request)
    expected = client_order_id(account_id, signal_id)
    require(isinstance(lineage,dict) and lineage.get('signal_id')==signal_id and
        lineage.get('symbol')==request.get('symbol') and
        lineage.get('strategy_contract',{}).get('strategy_id')==pipeline.STRATEGY and
        lineage.get('strategy_contract',{}).get('strategy_version')=='2' and
        all(isinstance(lineage.get(k),str) and re.fullmatch('[0-9a-f]{64}',lineage[k]) is not None
            for k in ('handoff_id','batch_sha256','backtest_result_id','metrics_result_id')),
        'execution_order_lineage_invalid')
    require(set(request)=={'client_order_id','side','symbol','type','limit_price','time_in_force','extended_hours','qty'} and
        request.get('client_order_id') == expected and request.get('side') == 'buy' and
        request.get('type') == 'limit' and number(request.get('limit_price'),positive=True)>0 and
        request.get('time_in_force') == 'day' and request.get('extended_hours') is True and
        type(request.get('qty')) is int and request['qty'] > 0,
        'execution_entry_request_invalid')
    fingerprint = digest(dict(request=request, lineage=lineage))
    claimed = store.claim(expected, fingerprint, lineage)
    require(type(claimed) is bool, 'execution_durable_claim_unverified')
    if not claimed:
        observed = broker.find_by_client_order_id(expected)
        if observed is None:
            return dict(status='UNKNOWN_NEVER_RESUBMIT', client_order_id=expected)
        checked_order(request,observed)
        store.observed(expected, observed)
        return dict(status='RECOVERED', order=observed)
    try:
        order = broker.submit_order(request)
    except Exception:
        store.uncertain(expected)
        return dict(status='UNKNOWN_NEVER_RESUBMIT', client_order_id=expected)
    try:
        checked_order(request,order)
    except (ValueError,TypeError,KeyError):
        store.uncertain(expected)
        raise
    store.observed(expected, order)
    return dict(status='OBSERVED', order=order)
