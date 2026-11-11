"""Pure REVIEW-ONLY pricing contract prototype; no imports of app/broker/database.

No publication or execution adapter is installed. Caller supplies inspected native
pure helpers. Returned previews explicitly do not authorize order submission.
"""
from copy import deepcopy
from datetime import datetime, date, timedelta
from types import SimpleNamespace
import re

POLICY = 'RAW_SIP_AH_OR_REGULAR_15M_CLOSE_PLUS_0_5_PCT_V2'


def pricing_contract(native_rules, native_selection):
    require = native_rules.require
    utc, number, digest, NY = native_rules.utc, native_rules.number, native_rules.digest, native_rules.NY
    after_hours_reference = native_rules.after_hours_reference
    selection_id = native_selection.selection_id
    PIPELINE_LIMIT_POLICY = POLICY
    def _minute_bar_stamp(value):
        stamp=utc(value)
        require(isinstance(value,str) and re.fullmatch(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:00(?:\.0{1,9})?(?:Z|[+-]\d{2}:\d{2})',value) is not None,
            'execution_reference_minute_alignment_invalid')
        return stamp

    def regular_close_reference(bars, *, symbol, signal_session, observed_at,
                                feed, adjustment, currency, complete):
        """Latest eligible raw minute close in the same session's final 15 minutes.

        The primary calendar contract rejects exceptional/early-close sessions.
        This is a timestamped minute-bar close, never an invented official close.
        """
        close=datetime.combine(date.fromisoformat(signal_session),datetime.min.time().replace(hour=16),NY)
        start=close-timedelta(minutes=15)
        require(feed=='sip' and adjustment=='raw' and currency=='USD' and complete is True and
            utc(observed_at)>=utc(close)+timedelta(hours=4,minutes=15),
            'execution_regular_reference_capture_invalid')
        require(isinstance(bars,list) and len(bars)<=15,'execution_regular_reference_population_invalid')
        previous=None
        for bar in bars:
            require(isinstance(bar,dict),'execution_regular_reference_bar_invalid')
            stamp=_minute_bar_stamp(bar['t'])
            require(utc(start)<=stamp<utc(close) and
                (previous is None or previous<stamp),'execution_regular_reference_bar_outside_session')
            prices=[number(bar[k],positive=True) for k in ('o','h','l','c')]
            require(prices[2]<=min(prices[0],prices[3])<=max(prices[0],prices[3])<=prices[1] and
                number(bar['v'],positive=True)>0,'execution_regular_reference_bar_invalid')
            previous=stamp
        result=dict(status='AVAILABLE' if bars else 'NO_REGULAR_CLOSE_REFERENCE',symbol=symbol,
            signal_session=signal_session,price=str(number(bars[-1]['c'],positive=True)) if bars else None,
            last_bar_at=bars[-1]['t'] if bars else None,observed_at=utc(observed_at).isoformat(),
            window_start=utc(start).isoformat(),window_end=utc(close).isoformat(),feed=feed,
            adjustment=adjustment,currency=currency,complete=True,bars=deepcopy(bars),bars_sha256=digest(bars))
        result['reference_id']=digest(result)
        return result

    def after_hours_activity(trades, *, symbol, signal_session, observed_at, complete, error=None):
        """Bounded descriptive raw prints, separate from eligible execution prices.

        Official/corrected-close messages and zero-size records are not new trades.
        A failed or incomplete diagnostic never asserts there was no trading.
        """
        cutoff=datetime.combine(date.fromisoformat(signal_session),datetime.min.time().replace(hour=20),NY)
        start=cutoff-timedelta(hours=4)
        require(type(complete) is bool and utc(observed_at)>=utc(cutoff)+timedelta(minutes=15),
            'execution_activity_capture_invalid')
        require(error in {None,'RAW_TRADES_HTTP_ERROR','RAW_TRADES_TRANSPORT_ERROR',
            'RAW_TRADES_RESPONSE_INVALID'},'execution_activity_error_invalid')
        require(isinstance(trades,list) and len(trades)<=100 and
            (error is None or (not trades and not complete)),'execution_activity_population_invalid')
        previous=None;observations=[];administrative=0;odd_lots=0;extended=0;extended_odd_lots=0
        for trade in trades:
            require(isinstance(trade,dict) and set(trade)<={'t','p','s','c','i','x','z'} and
                {'t','p','s','c'}<=set(trade),'execution_activity_trade_invalid')
            stamp=utc(trade['t']);conditions=trade['c']
            require(utc(start)<=stamp<utc(cutoff) and (previous is None or stamp<=previous),
                'execution_activity_trade_outside_session')
            require(isinstance(conditions,list) and len(conditions)<=16 and
                all(isinstance(c,str) and 1<=len(c)<=4 for c in conditions),
                'execution_activity_conditions_invalid')
            price=number(trade['p'],positive=True);size=number(trade['s'])
            previous=stamp
            if {'M','9'} & set(conditions) or size==0:
                administrative+=1
                continue
            is_odd='I' in conditions;is_extended=bool({'T','U'} & set(conditions))
            odd_lots+=is_odd;extended+=is_extended;extended_odd_lots+=is_odd and is_extended
            observations.append(dict(price=str(price),timestamp=trade['t'],size=str(size),conditions=conditions.copy()))
        status=('ACTIVITY_UNKNOWN' if error else 'INCOMPLETE_RAW_ACTIVITY' if not complete else
            'TRADES_WITHOUT_ELIGIBLE_BAR' if observations else
            'ADMINISTRATIVE_MESSAGES_ONLY' if trades else 'NO_RAW_TRADES')
        summary=dict(status=status,complete=complete,error=error,raw_record_count=len(trades),
            observed_trade_count=len(observations),administrative_message_count=administrative,
            odd_lot_trade_count=odd_lots,extended_hours_trade_count=extended,
            extended_hours_odd_lot_count=extended_odd_lots,
            latest_observed_trade=observations[0] if observations else None)
        return dict(symbol=symbol,signal_session=signal_session,observed_at=utc(observed_at).isoformat(),
            feed='sip',currency='USD',window_start=utc(start).isoformat(),window_end=utc(cutoff).isoformat(),
            summary=summary,trades=deepcopy(trades),trades_sha256=digest(trades))

    def pipeline_price_reference(after_hours, regular_close=None, activity=None):
        """Replay a typed reference without changing raw bars or trade eligibility."""
        require(isinstance(after_hours,dict),'execution_pipeline_reference_invalid')
        symbol=after_hours['symbol'];day=after_hours['signal_session']
        checked=after_hours_reference(after_hours['bars'],symbol=symbol,signal_session=day,
            observed_at=after_hours['observed_at'],feed=after_hours['feed'],adjustment=after_hours['adjustment'],
            currency=after_hours['currency'],complete=after_hours['complete'])
        require(checked==after_hours,'execution_pipeline_reference_changed')
        for bar in after_hours['bars']:_minute_bar_stamp(bar['t'])
        reference=after_hours if after_hours['status']=='AVAILABLE' else None
        kind='AFTER_HOURS_BAR' if reference else None
        observed=utc(after_hours['observed_at'])
        if reference:
            require(regular_close is None and activity is None,'execution_pipeline_unneeded_fallback')
        else:
            require(isinstance(regular_close,dict) and isinstance(activity,dict),
                'execution_pipeline_fallback_required')
            regular=regular_close_reference(regular_close['bars'],symbol=symbol,signal_session=day,
                observed_at=regular_close['observed_at'],feed=regular_close['feed'],
                adjustment=regular_close['adjustment'],currency=regular_close['currency'],complete=regular_close['complete'])
            require(regular==regular_close,'execution_pipeline_regular_reference_changed')
            raw=after_hours_activity(activity['trades'],symbol=symbol,signal_session=day,
                observed_at=activity['observed_at'],complete=activity['summary']['complete'],
                error=activity['summary']['error'])
            require(raw==activity,'execution_pipeline_activity_changed')
            observed=max(observed,utc(regular_close['observed_at']),utc(activity['observed_at']))
            if regular['status']=='AVAILABLE':reference=regular;kind='REGULAR_SESSION_FALLBACK'
        result=dict(schema='jbravo.pipeline-price-reference.v1',symbol=symbol,signal_session=day,
            status='AVAILABLE' if reference else 'NO_ELIGIBLE_PRICE_REFERENCE',reference_kind=kind,
            price=str(number(reference['price'],positive=True)) if reference else None,
            reference_at=reference['last_bar_at'] if reference else None,observed_at=observed.isoformat(),
            after_hours=deepcopy(after_hours),regular_close=deepcopy(regular_close),activity=deepcopy(activity))
        result['reference_id']=digest(result)
        return result

    def build_pipeline_pricing(selection,parent,price_inputs,*,published_at):
        """Versioned price decisions for every selected signal, without backfilling."""
        from decimal import Decimal,ROUND_FLOOR
        from datetime import date,time,timedelta
        selected=selection['selected_setups']
        require(isinstance(price_inputs,dict) and set(price_inputs)=={r['proposal']['symbol'] for r in selected},
            'pipeline_limit_reference_population_changed')
        require(len({r['proposal']['signal_id'] for r in selected})==len(selected),
            'pipeline_limit_signal_population_changed')
        published=rules.utc(published_at)
        opening=datetime.combine(date.fromisoformat(parent['session']['next_session']),time(4),rules.NY)
        require(published<opening+timedelta(minutes=35),'pipeline_limit_publication_after_entry_window')
        orders={};dispositions={};executable=[]
        for setup in selected:
            proposal=setup['proposal'];symbol=proposal['symbol'];signal=proposal['signal_id']
            require(proposal['signal_session']==parent['session']['signal_session'] and
                proposal['next_session']==parent['session']['next_session'],'pipeline_limit_proposal_session_changed')
            reference=price_inputs[symbol]
            require(isinstance(reference,dict),'pipeline_limit_reference_unavailable')
            checked=rules.pipeline_price_reference(reference['after_hours'],reference['regular_close'],reference['activity'])
            require(checked==reference and checked['symbol']==symbol and
                checked['signal_session']==parent['session']['signal_session'] and
                rules.utc(checked['observed_at'])<=published,'pipeline_limit_reference_unavailable')
            available=checked['status']=='AVAILABLE';kind=checked['reference_kind']
            activity=(deepcopy(checked['activity']['summary']) if checked['activity'] is not None else
                dict(status='ELIGIBLE_AFTER_HOURS_BARS',bar_count=len(checked['after_hours']['bars'])))
            dispositions[signal]=dict(signal_id=signal,symbol=symbol,
                status='PRICE_AVAILABLE' if available else 'EXCLUDED_NO_ELIGIBLE_REFERENCE',
                reference_kind=kind,reference_price=checked['price'],reference_at=checked['reference_at'],
                reference_id=checked['reference_id'],after_hours_activity=activity,
                reason=('AFTER_HOURS_BAR_AVAILABLE' if kind=='AFTER_HOURS_BAR' else
                    'NO_ELIGIBLE_AFTER_HOURS_BAR' if available else 'NO_ELIGIBLE_AFTER_HOURS_OR_REGULAR_CLOSE_BAR'))
            if not available:continue
            price=(rules.number(checked['price'],positive=True)*Decimal('1.005')).quantize(Decimal('.01'),rounding=ROUND_FLOOR)
            require(price>0,'pipeline_limit_nonpositive')
            chosen=checked['after_hours'] if kind=='AFTER_HOURS_BAR' else checked['regular_close']
            provenance=dict(provider='alpaca',feed='sip',adjustment='raw',currency='USD',
                policy=PIPELINE_LIMIT_POLICY,reference_kind=kind,reference_price=checked['price'],
                reference_at=checked['reference_at'],captured_at=checked['observed_at'],
                reference_id=checked['reference_id'],bars_sha256=chosen['bars_sha256'],
                multiplier='1.005',rounding='ROUND_FLOOR',tick_size='0.01')
            orders[signal]=dict(schema='jbravo.pipeline-paper-limit-order.v2',
                primary_run_id=parent['primary_run_id'],selection_run_id=selection_id(parent['primary_run_id']),
                signal_id=signal,symbol=symbol,signal_session=proposal['signal_session'],next_session=proposal['next_session'],
                limit_price=format(price,'.2f'),currency='USD',price_basis=PIPELINE_LIMIT_POLICY,
                price_as_of=checked['reference_at'],valid_from=published.isoformat(),
                valid_until=(opening+timedelta(minutes=35)).isoformat(),price_provenance=provenance)
            executable.append(signal)
        status=('NO_SELECTED_SIGNALS' if not selected else 'NO_EXECUTABLE_PRICE_REFERENCES' if not executable else
            'PRICED_WITH_EXCLUSIONS' if len(executable)<len(selected) else 'PRICED')
        return dict(execution_orders=orders,execution_price_dispositions=dispositions,
            executable_signal_ids=executable,pricing_status=status)

    rules = SimpleNamespace(number=number, utc=utc, NY=NY, pipeline_price_reference=pipeline_price_reference)

    def prepare_review(parent_record, price_inputs, *, published_at):
        parent = dict(primary_run_id=parent_record['primary_run_id'], session=parent_record['session'])
        value = build_pipeline_pricing(parent_record['selection'], parent, price_inputs, published_at=published_at)
        return dict(schema='jbravo.isolated-pricing-review.v1', review_only=True,
            order_submission_authorized=False, parent_selection_sha256=digest(parent_record),
            parent_selection_run_id=parent_record['run_id'], primary_run_id=parent['primary_run_id'],
            native_source_bindings_sha256=digest(parent_record['source_bindings']),
            session=deepcopy(parent['session']), published_at=published_at,
            price_inputs=deepcopy(price_inputs), pricing=value)

    def entry_preview(parent_record, review, handoffs):
        require(review == prepare_review(parent_record, review['price_inputs'],
            published_at=review['published_at']), 'isolated_pricing_review_changed')
        require(review['review_only'] is True and review['order_submission_authorized'] is False,
            'isolated_pricing_not_an_execution_authority')
        output=[]
        selected=parent_record['selection']['selected_setups']
        for setup in selected:
            proposal=setup['proposal'];sid=proposal['signal_id']
            if sid not in review['pricing']['executable_signal_ids']:continue
            handoff=deepcopy(handoffs[proposal['strategy_id']])
            handoff['unified_selection']=dict(run_id=parent_record['run_id'],
                result_sha256=digest(parent_record),rank=setup['rank'],
                selected_signal_ids=parent_record['selection']['selected_signal_ids'])
            ticket=native_rules.validate_pipeline_limit_order(handoff,proposal,
                review['pricing']['execution_orders'][sid],published_at=review['published_at'])
            output.append(dict(signal_id=sid,rank=setup['rank'],ticket=ticket,
                order_submission_authorized=False))
        return output

    return SimpleNamespace(regular_close_reference=regular_close_reference,
        after_hours_activity=after_hours_activity,pipeline_price_reference=pipeline_price_reference,
        build_pipeline_pricing=build_pipeline_pricing,prepare_review=prepare_review,
        entry_preview=entry_preview)
