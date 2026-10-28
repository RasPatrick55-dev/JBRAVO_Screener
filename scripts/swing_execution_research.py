"""Captured NBBO entry sensitivity and explicit read-only paper lifecycle exports."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_FLOOR
import math
from zoneinfo import ZoneInfo

from scripts.research_dataset import bounded_read, canonical_hash, digest_bytes, strict_json
from scripts.swing_scorecard import require

NY = ZoneInfo('America/New_York')


def instant(value):
    result = datetime.fromisoformat(value[:-1]+'+00:00' if value.endswith('Z') else value)
    require(result.tzinfo is not None, 'Explicit timezone required')
    return result.astimezone(timezone.utc)


def load_quotes(binding):
    raw = bounded_read(binding['path'], 64*1024*1024)
    require(digest_bytes(raw)==binding['sha256'], 'Entry capture hash changed')
    value = strict_json(raw)
    require(set(value)=={'schema','provider','feed','adjustment','currency','size_units','availability_basis','quotes','references'},
            'Unexpected entry capture fields')
    require(value['schema']=='jbravo.entry-research-capture.v1' and
        value['provider']=='alpaca' and value['feed']=='sip' and value['adjustment']=='raw' and
        value['currency']=='USD' and value['size_units']=='SHARES' and
        value['availability_basis']=='PROVIDER_EVENT_TIME_ASSUMPTION', 'Unsupported entry capture conventions')
    require(len(value['quotes'])<=250000 and len(value['references'])<=10000, 'Entry capture row budget exceeded')
    quotes, references, seen = {}, {}, set()
    for row in value['quotes']:
        require(set(row)=={'symbol','at','bid','ask','ask_size_shares'}, 'Malformed captured quote')
        at = instant(row['at'])
        key = row['symbol'], at
        require(key not in seen, 'Duplicate quote instant')
        seen.add(key)
        require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('bid','ask')) and
                0 < row['bid'] <= row['ask'] and type(row['ask_size_shares']) is int and row['ask_size_shares']>=0,
                'Invalid/crossed quote or ambiguous displayed size')
        quotes.setdefault((row['symbol'],at.astimezone(NY).date().isoformat()),[]).append(row)
    for rows in quotes.values():
        require([instant(r['at']) for r in rows]==sorted(instant(r['at']) for r in rows), 'Unsorted quotes')
    for row in value['references']:
        require(set(row)=={'symbol','signal_session','at','price'}, 'Malformed after-hours reference')
        at = instant(row['at']).astimezone(NY)
        require(at.date().isoformat()==row['signal_session'] and time(16)<=at.time()<=time(20) and
                type(row['price']) in (int,float) and math.isfinite(row['price']) and row['price']>0,
                'Reference must be a positive raw after-hours signal-session price')
        key = row['symbol'],row['signal_session']
        require(key not in references, 'Ambiguous frozen reference')
        references[key]=row
    return quotes,references


def entry_comparison(observations, capture, *, cash, fee_per_fill, slippage_bps):
    """Paired first-quote experiments, not daily-bar premarket portfolio simulation.

    Only immediately marketable, displayed-size-limited orders are modeled. A
    passive quote touch never invents a fill or an order-queue position.
    """
    if capture is None:
        return dict(status='UNAVAILABLE_NO_CAPTURED_INTRADAY_QUOTES', comparisons=[],
                    strategy_pnl=None, broker_operations=0)
    quotes,references=load_quotes(capture)
    output=[]
    for item in observations:
        if not item['selected'] or not item['is_exit_owner']:
            continue
        p=item['proposal']; reference=references.get((item['symbol'],item['signal_session']))
        limit = float((Decimal(str(reference['price']))*Decimal('1.005')).quantize(Decimal('.01'),rounding=ROUND_FLOOR)) if reference else None
        plans={}
        for name,clock in (('premarket',time(4)),('regular',time(9,30))):
            target=datetime.combine(date.fromisoformat(p['next_session']),clock,NY).astimezone(timezone.utc)
            rows=[r for r in quotes.get((item['symbol'],p['next_session']),[]) if target<=instant(r['at'])<=target+timedelta(seconds=60)]
            quote=rows[0] if rows else None
            plan=dict(status='MISSING_QUOTE',quantity=0,price=None,fee=0,modeled_cost=None)
            if name=='premarket' and reference is None:
                plan['status']='MISSING_AFTER_HOURS_REFERENCE'
            elif quote:
                price=quote['ask']*(1+slippage_bps/10000)
                pattern=p.get('pattern_exit')
                if pattern and not pattern['invalidation']<price<pattern['target']:
                    plan['status']='QUOTE_OUTSIDE_PATTERN_LEVELS'
                elif name=='premarket' and pattern and not pattern['invalidation']<limit<pattern['target']:
                    plan['status']='FROZEN_LIMIT_OUTSIDE_PATTERN'
                elif name=='premarket' and price>limit:
                    plan['status']='NOT_IMMEDIATELY_MARKETABLE_NO_PASSIVE_FILL_ASSUMPTION'
                else:
                    budget=cash*.05
                    qty=min(max(0,math.floor((budget-fee_per_fill)/price)),quote['ask_size_shares'])
                    plan.update(status='MODELED_MARKETABLE_FILL' if qty else 'NO_DISPLAYED_SIZE_OR_BUDGET',
                        quantity=qty,price=price if qty else None,fee=fee_per_fill if qty else 0,
                        modeled_cost=qty*price+fee_per_fill if qty else None,quote_at=quote['at'],
                        spread_bps=10000*(quote['ask']-quote['bid'])/((quote['ask']+quote['bid'])/2),
                        maximum_fill_price=limit if name=='premarket' else None)
            plans[name]=plan
        paired=all(plans[k]['quantity']>0 for k in plans)
        output.append(dict(signal_id=item['signal_id'],symbol=item['symbol'],signal_session=item['signal_session'],
            next_session=p['next_session'],plans=plans,paired=paired,
            premarket_minus_regular_price_bps=10000*(plans['premarket']['price']/plans['regular']['price']-1) if paired else None))
    return dict(status='COMPLETED_ENTRY_PRICE_SENSITIVITY',comparisons=output,paired_signals=sum(r['paired'] for r in output),
        policies={name:dict(opportunities=len(output),modeled_fills=sum(r['plans'][name]['quantity']>0 for r in output),
            missing_evidence=sum(r['plans'][name]['status'].startswith('MISSING') for r in output)) for name in ('premarket','regular')},
        strategy_pnl=None,independent_fixed_cash_per_signal=True,production_executor_parity=False,
        assumptions=['Provider quote-event time assumes immediate availability; latency is not observed',
            'First quote within 60 seconds of 04:00/09:30 Eastern; no favorable later-quote search',
            'Size is capped by displayed ask shares; queue position, depth replenishment and actual fills remain unknown',
            'Premarket limit is prior signal-session after-hours reference times 1.005 rounded down to cents',
            'Entry-price sensitivity only: no premarket stop/exit path or portfolio profitability is inferred'],broker_operations=0)


def paper_snapshot(conn, *, account_id, start, end, max_rows=10000):
    """Export one enrollment cohort without constructing the writable execution Store."""
    require(conn.readonly is True and conn.autocommit is False and conn.isolation_level==2,
            'Read-only repeatable-read connection required')
    require(instant(start)<instant(end) and 0<len(account_id)<=128 and type(max_rows) is int and 1<=max_rows<=10000,
            'Invalid bounded paper snapshot request')
    with conn.cursor() as cur:
        cur.execute('SELECT deployment_id,database_id,database_name,current_database() '
                    'FROM public.trade_disposition_policy WHERE singleton=1')
        identity=cur.fetchone()
        require(identity is not None and len(identity)==4 and identity[2]==identity[3], 'Application database identity mismatch')
        cur.execute('SELECT signal_id,symbol,state,payload,created_at,updated_at '
            'FROM public.strategy_execution_signals_app WHERE account_id=%s AND created_at >= %s AND created_at < %s '
            'ORDER BY created_at,signal_id LIMIT %s',(account_id,start,end,max_rows+1))
        signals=cur.fetchall()
        require(len(signals)<=max_rows,'Paper signal cap exceeded; snapshot not truncated')
        cur.execute('SELECT o.signal_id,o.role,o.operation,o.state,o.observation,o.claimed_at '
            'FROM public.strategy_execution_orders_app o JOIN public.strategy_execution_signals_app s '
            'ON s.account_id=o.account_id AND s.signal_id=o.signal_id '
            'WHERE o.account_id=%s AND s.created_at >= %s AND s.created_at < %s '
            'ORDER BY o.claimed_at,o.client_order_id LIMIT %s',(account_id,start,end,max_rows+1))
        orders=cur.fetchall()
        require(len(orders)<=max_rows,'Paper order cap exceeded; snapshot not truncated')
    return dict(schema='jbravo.paper-research-snapshot.v1',account_id=account_id,start=start,end=end,
        database_identity=dict(zip(('deployment_id','database_id','database_name'),identity[:3])),
        observed_at=datetime.now(timezone.utc).isoformat(),transaction='READ ONLY REPEATABLE READ',
        cohort='SIGNALS_ENROLLED_DURING_WINDOW_WITH_LATEST_CUMULATIVE_ORDER_OBSERVATIONS',
        signals=[dict(zip(('signal_id','symbol','state','payload','created_at','updated_at'),r)) for r in signals],
        orders=[dict(zip(('signal_id','role','operation','state','observation','claimed_at'),r)) for r in orders],
        broker_operations=0,database_mutations=0)


def paper_scorecard(snapshot):
    require(snapshot['schema']=='jbravo.paper-research-snapshot.v1', 'Unknown paper snapshot')
    require({r['signal_id'] for r in snapshot['orders']}<={r['signal_id'] for r in snapshot['signals']},'Orphan paper order')
    rows=[]
    seen=set()
    for signal in snapshot['signals']:
        sid=signal['signal_id'];require(sid not in seen,'Duplicate paper signal');seen.add(sid)
        records=[r for r in snapshot['orders'] if r['signal_id']==sid]
        by_order={}
        for record in records:
            order=record['observation']
            if order is None:continue
            qty=float(order.get('filled_qty',0))
            require(order['side'] in {'buy','sell'} and math.isfinite(qty) and qty>=0,'Invalid observed paper quantity')
            if qty:
                price=float(order['filled_avg_price'])
                require(math.isfinite(price) and price>0,'Invalid observed paper fill price')
            # CANCEL/PATCH receipts repeat broker IDs and cumulative quantities.
            prior=by_order.get(order['id'])
            if prior:
                require(float(order.get('filled_qty',0))>=float(prior['filled_qty']), 'Cumulative fill regressed')
            by_order[order['id']]=order
        buys=[o for o in by_order.values() if o['side']=='buy' and float(o.get('filled_qty',0))>0]
        sells=[o for o in by_order.values() if o['side']=='sell' and float(o.get('filled_qty',0))>0]
        bought=sum(float(o['filled_qty']) for o in buys);sold=sum(float(o['filled_qty']) for o in sells)
        require(0<=sold<=bought, 'Observed sell quantity exceeds enrolled buys')
        gross=sum(float(o['filled_qty'])*float(o['filled_avg_price']) for o in sells)-sum(float(o['filled_qty'])*float(o['filled_avg_price']) for o in buys) if bought and sold==bought else None
        proposal=signal['payload']['proposal']
        rows.append(dict(signal_id=sid,symbol=signal['symbol'],strategy_id=proposal['strategy_id'],
            state=signal['state'],entry_claimed=any(r['role']=='entry' for r in records),
            entry_observed=any(r['role']=='entry' and r['observation'] for r in records),
            quantity_bought=bought,quantity_sold=sold,fully_exited=bool(bought and sold==bought),
            unresolved_claims=sum(r['state'] in {'CLAIMED','UNKNOWN'} for r in records),
            gross_realized_pnl=gross,net_realized_pnl=None,net_status='UNAVAILABLE_ACTUAL_FEES_AND_EXECUTION_COSTS_NOT_BOUND',
            current_stop_observed=any(o['type']=='stop' and o['status'] in {'new','accepted','partially_filled'} for o in by_order.values()),
            continuous_protection_proven=False))
    return dict(schema='jbravo.paper-research-scorecard.v1',rows=rows,enrolled_signals=len(rows),
        complete_round_trips=sum(r['fully_exited'] for r in rows),net_expectancy=None,
        selection_coverage='ENROLLED_SIGNALS_ONLY_NOT_ALL_SCANNER_QUALIFIERS',
        continuous_protection_status='UNKNOWN_LATEST_SNAPSHOTS_DO_NOT_PROVE_CONTINUOUS_COVERAGE',
        research_qualified=False,execution_authorized=False)
