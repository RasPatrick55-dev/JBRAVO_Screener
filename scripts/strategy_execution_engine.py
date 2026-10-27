"""Strategy-specific paper execution state machine over durable/broker ports.

No connection or order on import. A runner holds the account lease and supplies
fresh native account/clock/source/policy guards. Prices and fills are observed;
neither an HTTP acknowledgment nor a missing position means a completed trade.
"""
from copy import deepcopy
from datetime import datetime,time,timezone
from decimal import Decimal,ROUND_FLOOR
import hashlib

from scripts.pattern_strategy import PATTERNS
from scripts.strategy_backtest import require
from scripts import strategy_execution as rules

TERMINAL={'filled','canceled','expired','rejected','replaced'}
LIVE={'new','accepted','pending_new','partially_filled','accepted_for_bidding','held','pending_replace','pending_cancel','stopped','suspended','calculated','done_for_day'}


def quantity(value):
    value=rules.number(value)
    require(value==value.to_integral_value(),'strategy_fractional_position_not_supported')
    return int(value)


def available_cash(account):
    values=[rules.number(account[k]) for k in ('cash','buying_power')]
    if account.get('non_marginable_buying_power') is not None:
        values.append(rules.number(account['non_marginable_buying_power']))
    return min(values)


class Engine:
    def __init__(self,store,broker,*,guard,now=None):
        require(callable(guard) and broker.paper is True and broker.account_id==store.account_id,
                'strategy_engine_paper_account_guard_required')
        self.store,self.broker,self.guard=store,broker,guard
        self.events=[]
        self.now=now or (lambda:datetime.now(timezone.utc))

    def note(self,signal,code):
        self.events.append(dict(signal_id=signal['signal_id'] if signal else None,code=code))

    def _check_order(self,record,order):
        if record['operation']=='DELETE':
            require(order.get('id')==record['request']['order_id'],'strategy_cancel_identity_changed')
            return
        request=record['request']['order']
        require(order.get('client_order_id')==record['client_order_id'] and
            order.get('symbol')==request['symbol'] and order.get('side')==request['side'] and
            order.get('type')==request['type'] and rules.number(order.get('qty'),positive=True)==request['qty'] and
            order.get('time_in_force')==request['time_in_force'] and
            order.get('extended_hours') is request['extended_hours'] and
            order.get('status') in TERMINAL|LIVE,'strategy_broker_order_identity_changed')
        filled=rules.number(order.get('filled_qty',0))
        require(filled<=request['qty'],'strategy_broker_fill_quantity_invalid')
        require(order['status']!='filled' or filled==request['qty'],'strategy_broker_terminal_fill_incomplete')
        for field in ('limit_price','stop_price'):
            if field in request:
                require(rules.number(order.get(field),positive=True)==rules.number(request[field],positive=True),
                    'strategy_broker_order_price_changed')
        if filled:
            price=rules.number(order.get('filled_avg_price'),positive=True)
            if request['type']=='limit' and request['side']=='buy':
                require(price<=rules.number(request['limit_price'],positive=True),'strategy_buy_fill_exceeds_limit')

    def _observe(self,record):
        if record['state']=='REJECTED':return None
        if record['operation']=='DELETE':order=self.broker.get_order(record['request']['order_id'])
        else:order=self.broker.find_by_client_order_id(record['client_order_id'])
        if order is None:
            self.note(None,'ORDER_OUTCOME_UNKNOWN_NO_RESUBMISSION')
            return None
        self._check_order(record,order)
        self.store.observed(record['client_order_id'],order)
        return order

    def _perform(self,signal,role,sequence,request,*,operation='POST',order_id=None):
        key=rules.client_order_id(self.store.account_id,signal['signal_id'],role=role,sequence=sequence)
        request=deepcopy(request)
        if operation!='DELETE':request['client_order_id']=key
        descriptor=dict(order_id=order_id,order=request) if operation!='DELETE' else dict(order_id=order_id)
        lineage=dict(handoff_id=signal['payload']['handoff']['handoff_id'],
            signal_id=signal['signal_id'],strategy_contract=signal['payload']['handoff']['strategy_contract'])
        claim=self.store.claim(signal_id=signal['signal_id'],role=role,sequence=sequence,
            operation=operation,request=descriptor,lineage=lineage)
        record=dict(claim,operation=operation,request=descriptor,role=role,sequence=sequence)
        if not claim['fresh']:return self._observe(record)
        require(self.guard('entry' if role=='entry' else 'lifecycle',signal) is True,
                'strategy_submission_guard_failed')
        # The guard may perform read-only I/O after the original claim. Recheck
        # the policy on the same writer connection immediately before sending.
        with self.store.transaction():pass
        try:
            if operation=='DELETE':
                self.broker.cancel_order(order_id)
                order=self.broker.get_order(order_id)
            elif operation=='PATCH':
                fields={k:v for k,v in request.items() if k in {'qty','stop_price','client_order_id'}}
                order=self.broker.replace_order(order_id,fields)
            else:order=self.broker.submit_order(request)
        except Exception as error:
            if getattr(error,'http_status',None) in {400,401,403,404,422} and operation!='DELETE':
                observed=self.broker.find_by_client_order_id(key)
                if observed is not None:
                    self._check_order(record,observed);self.store.observed(key,observed)
                    return observed
                self.store.rejected(key);self.note(signal,'BROKER_REJECTED')
                return None
            self.store.uncertain(key);self.note(signal,'ORDER_OUTCOME_UNKNOWN_NO_RESUBMISSION')
            return None
        self._check_order(record,order)
        self.store.observed(key,order)
        return order

    def enter(self,handoff,references,*,now,clock,account,positions,open_orders,max_new_entries=3):
        require(handoff['paper_only'] is True and handoff['inspection_mode'] is False,
                'strategy_entry_handoff_not_actionable')
        held=set(positions)
        pending={r['symbol'] for r in open_orders if r.get('side')=='buy'}
        occupied=held|pending
        ceiling=available_cash(account);reserved=Decimal(0)
        active={s['signal_id']:s for s in self.store.signals(active=False)}
        occupied |= {s['payload']['proposal']['symbol'] for s in active.values()
                     if s['state'] not in {'CLOSED','EXPIRED'}}
        require(type(max_new_entries) is int and 0<=max_new_entries<=3,'strategy_daily_entry_budget_invalid')
        attempted=0
        if any(r['state'] in {'CLAIMED','UNKNOWN'} for s in active.values() for r in self.store.orders(s['signal_id'])):
            self.note(None,'UNRESOLVED_ACCOUNT_ORDER_BLOCKS_NEW_ENTRIES');return
        for proposal in handoff['proposals']:
            if attempted>=max_new_entries:
                self.note(None,'SHARED_DAILY_ENTRY_CAP');break
            sid,symbol=proposal['signal_id'],proposal['symbol']
            if sid in active:
                self.note(active[sid],'SIGNAL_ALREADY_ENROLLED_NO_SECOND_BUY');continue
            if symbol in occupied:
                self.note(None,'SYMBOL_ALREADY_HELD_OR_PENDING');continue
            if len(occupied)>=4:
                self.note(None,'TOTAL_POSITION_CAP');break
            reference=references.get(symbol)
            if not reference or reference.get('status')!='AVAILABLE':
                self.note(None,'AFTER_HOURS_REFERENCE_UNAVAILABLE');continue
            if not self.broker.eligible_asset(symbol):
                self.note(None,'ASSET_INELIGIBLE');continue
            try:quoted=self.broker.quotes({symbol},now=self.now())
            except (ValueError,KeyError,TypeError):
                self.note(None,'ENTRY_MONITOR_QUOTE_UNAVAILABLE');continue
            if symbol not in quoted:
                self.note(None,'ENTRY_MONITOR_QUOTE_UNAVAILABLE');continue
            fresh_account=self.broker.account()
            cash=min(available_cash(fresh_account),ceiling-reserved)
            try:
                plan=rules.build_entry_order(handoff,signal_id=sid,account_id=self.store.account_id,cash=cash,
                    reference=reference,occupied_slots=len(occupied),now=self.now(),
                    clock_next_regular_open=clock['next_open'],clock_timestamp=clock['timestamp'])
            except ValueError as error:
                if str(error) not in {'bull_flag_entry_outside_levels','pattern_entry_outside_levels'}:raise
                self.note(None,'BULL_FLAG_ENTRY_OUTSIDE_LEVELS' if str(error).startswith('bull_flag') else 'PATTERN_ENTRY_OUTSIDE_LEVELS');continue
            if plan['request']['qty']==0:
                self.note(None,'WHOLE_SHARE_BUDGET_EMPTY');continue
            signal=self.store.enroll(handoff,proposal,reference)
            attempted+=1
            signal['payload']['entry_plan']=plan
            self.store.save(signal,state='ENTRY_PENDING')
            order=self._perform(signal,'entry',0,plan['request'])
            occupied.add(symbol)
            reserved+=rules.number(plan['request']['limit_price'])*plan['request']['qty']
            if order is None:
                rejected=any(r['role']=='entry' and r['state']=='REJECTED'
                    for r in self.store.orders(sid))
                self.store.save(signal,state='EXPIRED' if rejected else 'UNKNOWN')
                # Unknown acceptance makes remaining cash/capacity uncertain.
                break
            # Protect actual fills before attempting another symbol. A pending
            # entry remains durable for the independent monitor to reconcile.
            self.tick(now=self.now())

    def cancel_confirmed(self,signal,order_id):
        current=self.broker.get_order(order_id)
        if current['status'] in TERMINAL:return current
        sequence=int(hashlib.sha256(order_id.encode()).hexdigest()[:12],16)
        return self._perform(signal,'cancel',sequence,{},operation='DELETE',order_id=order_id)

    def sync(self,signal,*,now):
        """Refresh cumulative fills first; position absence alone proves nothing."""
        records=self.store.orders(signal['signal_id'])
        for record in records:
            observation=record['observation']
            if record['state']!='REJECTED' and (observation is None or observation.get('status') not in TERMINAL):
                self._observe(record)
        records=self.store.orders(signal['signal_id'])
        if any(r['state'] in {'CLAIMED','UNKNOWN'} for r in records):
            self.note(signal,'ORDER_OUTCOME_UNKNOWN_NO_RESUBMISSION');return None
        entry=next((r['observation'] for r in records if r['role']=='entry' and r['observation']),None)
        if entry is None:
            if any(r['role']=='entry' and r['state']=='REJECTED' for r in records):
                self.store.save(signal,state='EXPIRED')
            return None
        bought=quantity(entry.get('filled_qty',0))
        sold_ids={}
        for r in records:
            o=r['observation']
            if o and r['role'] in {'stop','exit','partial'}:
                sold_ids[o['id']]=quantity(o.get('filled_qty',0))
        sold=sum(sold_ids.values());remaining=bought-sold
        require(0<=remaining<=bought,'strategy_exit_fills_exceed_owned_quantity')
        payload=deepcopy(signal['payload'])
        if entry['status'] not in TERMINAL and rules.utc(now)>=rules.utc(payload['handoff']['execution_policy']['cancel_at']):
            canceled=self.cancel_confirmed(signal,entry['id'])
            if canceled is None or canceled['status'] not in TERMINAL:
                self.note(signal,'ENTRY_CANCELLATION_UNCONFIRMED')
            # The next sync uses the terminal cumulative fill, never the older
            # pre-cancel quantity for a sale or for flat-account accounting.
            return None
        if remaining==0:
            if entry['status'] in TERMINAL:
                self.store.save(signal,state='CLOSED' if bought else 'EXPIRED')
            else:
                self.store.save(signal,state='ENTRY_PENDING')
            return None
        actual=self.broker.positions().get(payload['proposal']['symbol'])
        if actual is None or quantity(actual.get('qty',0))!=remaining or actual.get('side','long')!='long':
            self.note(signal,'POSITION_OWNERSHIP_NOT_RECONCILED');return None
        position=payload.get('position')
        if position is None:
            fill_time=entry.get('filled_at') or entry.get('updated_at')
            position=rules.initial_position(payload['proposal'],quantity=remaining,
                fill_price=entry['filled_avg_price'],filled_at=fill_time)
            payload['fill_time_basis']='FILLED_AT' if entry.get('filled_at') else 'VERIFIED_PARTIAL_QUANTITY_BROKER_UPDATE'
        else:
            position=deepcopy(position);position['qty']=remaining
            if bought>payload.get('entry_filled_qty',0):
                average=float(rules.number(entry['filled_avg_price'],positive=True))
                position['entry_price']=average
                position['max_price']=max(position['max_price'],average)
                position['highest_close']=max(position['highest_close'],average)
                position['trailing_stop']=max(position['trailing_stop'],average*.97)
                position['atr_stop']=max(position['atr_stop'],average-payload['proposal']['signal_atr14_raw'])
        payload['entry_filled_qty']=bought
        partial_sold=sum(quantity(r['observation'].get('filled_qty',0)) for r in records
            if r['role']=='partial' and r['observation'])
        if payload.get('partial_target_qty') is not None:
            require(partial_sold<=payload['partial_target_qty'],'strategy_partial_exit_exceeds_intent')
            if partial_sold==payload['partial_target_qty']:
                position['partial_taken']=True
                if payload['pending_exit_reasons']==['PARTIAL_5PCT']:
                    payload.update(pending_exit_reasons=[],pending_exit_session=None)
        payload['position']=position
        pending=any(r['role'] in {'exit','partial'} and r['observation'] and
            r['observation']['status'] not in TERMINAL for r in records)
        self.store.save(signal,state='EXIT_PENDING' if pending else 'OPEN',payload=payload)
        return records

    def completed(self,signal,frame,*,session,next_session,now):
        payload=deepcopy(signal['payload'])
        if not payload.get('position') or payload.get('last_completed_session', '')==session:return
        require(payload.get('last_completed_session') in (None,) or payload['last_completed_session']<session,
                'strategy_completed_bar_regressed')
        result=rules.completed_bar(payload['position'],frame,session=session,next_session=next_session,
            now=now,rsi_memory=payload['rsi_memory'])
        if payload['proposal'].get('strategy_id') in PATTERNS:
            plan=payload['proposal']['pattern_exit']
            result['position']['atr_stop']=max(result['position']['atr_stop'] or 0,plan['invalidation'])
            result['pending_exit_reasons']=[r for r in result['pending_exit_reasons'] if r!='PARTIAL_5PCT']
            if not result['pending_exit_reasons']:result['pending_exit_session']=None
        payload.update(position=result['position'],rsi_memory=result['rsi_memory'],last_completed_session=session)
        # An existing full exit intent remains active even if a later bar would
        # look better; only observed fills can complete that lifecycle.
        if not payload['pending_exit_reasons'] or (
            payload['pending_exit_reasons']==['PARTIAL_5PCT'] and
            result['pending_exit_reasons'] not in ([],['PARTIAL_5PCT'])):
            payload.update(pending_exit_reasons=result['pending_exit_reasons'],
                pending_exit_session=result['pending_exit_session'])
        self.store.save(signal,payload=payload)

    def manage(self,signal,records,*,now,quote,regular_open):
        payload=signal['payload'];position=payload.get('position')
        if not position:return
        local=rules.utc(now).astimezone(rules.NY)
        if not time(4)<=local.time().replace(tzinfo=None)<time(20):return
        own_qty=quantity(position['qty']);symbol=position['symbol']
        active_sells=[r for r in records if r['role'] in {'stop','exit','partial'} and r['observation'] and
                      r['observation']['status'] not in TERMINAL]
        if any(r['role'] in {'exit','partial'} for r in active_sells):
            self.note(signal,'EXIT_ORDER_PENDING_CONFIRMATION');return
        stops=[r for r in active_sells if r['role']=='stop']
        require(len(stops)<=1,'strategy_multiple_active_stops')
        stop=stops[0] if stops else None
        target=max(position['trailing_stop'] or 0,position['atr_stop'] or 0)
        breached=quote is not None and rules.number(quote['bid'],positive=True)<=Decimal(str(target))
        if payload['proposal'].get('strategy_id') in PATTERNS:
            plan=payload['proposal']['pattern_exit']
            reached=quote is not None and rules.number(quote['bid'],positive=True)>=Decimal(str(plan['target']))
            if reached and not payload['pending_exit_reasons']:
                payload=deepcopy(payload)
                payload.update(pending_exit_reasons=['PATTERN_TARGET'],pending_exit_session=local.date().isoformat())
                self.store.save(signal,payload=payload);self.note(signal,'PATTERN_TARGET')
            if breached:
                self.note(signal,'PATTERN_INVALIDATION' if target==plan['invalidation'] else 'EXISTING_PROTECTIVE_STOP')
        queued=bool(payload['pending_exit_reasons']) and local.date().isoformat()>=payload['pending_exit_session']
        if queued and not breached and payload['pending_exit_reasons']==['PARTIAL_5PCT'] and own_qty==1:
            payload=deepcopy(payload);payload.update(pending_exit_reasons=[],pending_exit_session=None)
            self.store.save(signal,payload=payload);queued=False
        if breached or queued:
            if any(r['role']=='entry' and r['observation'] and
                r['observation']['status'] not in TERMINAL for r in records):
                # Freeze the cumulative bought quantity before an exit. Future
                # fills must not reopen a position behind its completed sale.
                entry=next(r['observation'] for r in records if r['role']=='entry')
                self.cancel_confirmed(signal,entry['id'])
                self.note(signal,'ENTRY_CANCELLATION_BEFORE_EXIT');return
            if quote is None:
                self.note(signal,'EXIT_QUOTE_UNAVAILABLE');return
            if stop:
                canceled=self.cancel_confirmed(signal,stop['observation']['id'])
                if canceled is None or canceled['status'] not in TERMINAL or quantity(canceled.get('filled_qty',0))>0:
                    self.note(signal,'STOP_CANCELLATION_OR_FILL_UNCONFIRMED');return
            # Refresh exact ownership after cancellation before requesting SELL.
            actual=self.broker.positions().get(symbol)
            if actual is None or quantity(actual['qty'])!=own_qty:
                self.note(signal,'POSITION_CHANGED_BEFORE_EXIT');return
            partial=queued and not breached and payload['pending_exit_reasons']==['PARTIAL_5PCT'] and own_qty>1
            if partial:
                target=payload.get('partial_target_qty')
                if target is None:
                    payload=deepcopy(payload);target=max(1,own_qty//2)
                    payload['partial_target_qty']=target;self.store.save(signal,payload=payload)
                already=sum(quantity(r['observation'].get('filled_qty',0)) for r in records
                    if r['role']=='partial' and r['observation'])
                qty=target-already
                require(0<qty<=own_qty,'strategy_partial_remainder_invalid')
            else:qty=own_qty
            price=(rules.number(quote['bid'],positive=True)*Decimal('.995')).quantize(Decimal('.01'),rounding=ROUND_FLOOR)
            require(price>0,'strategy_exit_limit_invalid')
            role='partial' if partial else 'exit'
            sequence=1+max((r['sequence'] for r in records if r['role']==role),default=-1)
            order=self._perform(signal,role,sequence,dict(symbol=symbol,qty=qty,side='sell',type='limit',
                limit_price=float(price),time_in_force='day',extended_hours=True))
            if order is not None:self.store.save(signal,state='EXIT_PENDING')
            return
        # Fixed GTC levels preserve completed-bar tightening. Native trailing
        # stops would move intraday and would not protect premarket positions.
        if stop:
            old=stop['observation'];remaining=quantity(old['qty'])-quantity(old.get('filled_qty',0))
            if old['status'] in {'pending_new','pending_cancel','pending_replace','held'}:
                self.note(signal,'STOP_PENDING_CONFIRMATION');return
            if quantity(old.get('filled_qty',0)):
                self.note(signal,'PARTIAL_STOP_FILL_PENDING');return
            old_price=rules.number(old['stop_price'],positive=True)
            desired=Decimal(str(target)).quantize(Decimal('.01'),rounding=ROUND_FLOOR)
            if desired<=old_price and remaining==own_qty:
                self.note(signal,'REGULAR_STOP_OBSERVED_EXTENDED_MONITOR_ACTIVE' if not regular_open else 'REGULAR_STOP_OBSERVED')
                return
            sequence=max(r['sequence'] for r in records if r['role']=='stop')+1
            self._perform(signal,'stop',sequence,dict(symbol=symbol,qty=own_qty,side='sell',type='stop',
                stop_price=float(max(old_price,desired)),time_in_force='gtc',extended_hours=False),
                operation='PATCH',order_id=old['id'])
            return
        if sum(r['role']=='stop' and r['state']=='REJECTED' for r in records)>=3:
            self.note(signal,'STOP_REJECTIONS_REQUIRE_REVIEW');return
        sequence=1+max((r['sequence'] for r in records if r['role']=='stop'),default=-1)
        price=Decimal(str(target)).quantize(Decimal('.01'),rounding=ROUND_FLOOR)
        require(price>0,'strategy_protective_stop_invalid')
        self._perform(signal,'stop',sequence,dict(symbol=symbol,qty=own_qty,side='sell',type='stop',
            stop_price=float(price),time_in_force='gtc',extended_hours=False))

    def tick(self,*,now,frames=None,completed_session=None,next_session=None):
        clock=self.broker.clock()
        require(abs((rules.utc(now)-rules.utc(clock['timestamp'])).total_seconds())<=60,
                'strategy_monitor_clock_stale')
        signals=self.store.signals()
        symbols={s['payload']['proposal']['symbol'] for s in signals if s['state'] not in {'PREPARED','EXPIRED','CLOSED'}}
        ready=[]
        for signal in signals:
            if signal['state']=='PREPARED':
                # A crash between enrollment and the durable claim sent no
                # order. Release that abandoned ownership only after expiry.
                if (rules.utc(now)>=rules.utc(signal['payload']['handoff']['execution_policy']['cancel_at']) and
                    not self.store.orders(signal['signal_id'])):
                    self.store.save(signal,state='EXPIRED')
                continue
            require(self.guard('lifecycle',signal) is True,'strategy_lifecycle_guard_failed')
            records=self.sync(signal,now=now)
            if records is None:continue
            ready.append((signal,records))
        # Cancellations/fill reconciliation must still run when quote service
        # is unavailable. Unknown quote data is never a synthetic zero price.
        try:quotes=self.broker.quotes({s['payload']['proposal']['symbol'] for s,r in ready},now=now) if ready else {}
        except (ValueError,KeyError,TypeError):
            quotes={};self.note(None,'STRATEGY_QUOTES_UNAVAILABLE')
        if any(s['payload']['proposal']['symbol'] not in quotes for s,r in ready):
            self.note(None,'STRATEGY_QUOTES_UNAVAILABLE')
        for signal,records in ready:
            if frames is not None and completed_session and signal['payload']['position']['entry_time']<=completed_session:
                self.completed(signal,frames[signal['payload']['proposal']['symbol']],
                    session=completed_session,next_session=next_session,now=now)
            self.manage(signal,records,now=now,quote=quotes.get(signal['payload']['proposal']['symbol']),
                regular_open=clock['is_open'])
        return dict(events=list(self.events),owned_symbols=sorted(symbols))
