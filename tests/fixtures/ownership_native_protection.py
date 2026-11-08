"""Exact protection reporter from hash-bound deployed runtime, offline fixture."""
from scripts import strategy_execution as rules


def entry_protection_state(store):
    """Read the last reconciled state; never infer a stop from an entry fill.

    This certifies regular-hours broker stops only. No market data or monitor
    readiness check is part of this execution completion condition.
    """
    from scripts.strategy_execution_engine import TERMINAL
    from decimal import Decimal,ROUND_FLOOR
    positions=[];issues=[]
    for signal in store.signals():
        sid=signal['signal_id'];payload=signal['payload'];symbol=payload['proposal']['symbol']
        records=store.orders(sid)
        if signal['state'] in {'UNKNOWN','DISPUTED','PREPARED','ENTRY_PENDING','EXIT_PENDING'}:
            issues.append(dict(signal_id=sid,symbol=symbol,reason='SIGNAL_NOT_SETTLED'))
        if any(r['state'] in {'CLAIMED','UNKNOWN'} for r in records):
            issues.append(dict(signal_id=sid,symbol=symbol,reason='ORDER_OUTCOME_UNKNOWN'))
        position=payload.get('position')
        if not position:
            # An OPEN signal with no reconciled position is not a flat proof.
            if signal['state']=='OPEN':issues.append(dict(signal_id=sid,symbol=symbol,reason='POSITION_NOT_RECONCILED'))
            continue
        live=[r for r in records if r['role'] in {'stop','exit','partial'} and
            r.get('observation') and r['observation']['status'] not in TERMINAL]
        stops=[r['observation'] for r in live if r['role']=='stop']
        qty=rules.number(position['qty'],positive=True)
        minimum=Decimal(str(max(position.get('trailing_stop') or 0,position.get('atr_stop') or 0))).quantize(
            Decimal('.01'),rounding=ROUND_FLOOR)
        confirmed=(len(live)==1 and len(stops)==1 and stops[0]['status'] in {'new','accepted'} and
            stops[0].get('side')=='sell' and stops[0].get('type')=='stop' and
            stops[0].get('symbol')==symbol and stops[0].get('time_in_force')=='gtc' and
            stops[0].get('extended_hours') is False and
            rules.number(stops[0].get('filled_qty',0))==0 and
            rules.number(stops[0]['qty'],positive=True)==qty and
            rules.number(stops[0]['stop_price'],positive=True)>=minimum)
        positions.append(dict(signal_id=sid,symbol=symbol,qty=str(qty),
            regular_hours_stop_confirmed=confirmed,extended_hours=False,
            stop_order_id=stops[0]['id'] if confirmed else None))
        if not confirmed:issues.append(dict(signal_id=sid,symbol=symbol,reason='FULL_QUANTITY_STOP_NOT_CONFIRMED'))
    return dict(confirmed=not issues,positions=positions,unresolved=issues,
        scope='REGULAR_HOURS_BROKER_STOPS_ONLY')
