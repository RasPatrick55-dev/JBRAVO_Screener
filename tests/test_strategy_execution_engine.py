"""Offline broker races exercise the actual durable worker, with no credentials."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import socket

import pytest

from scripts import strategy_execution as rules
from scripts.strategy_execution_engine import Engine
from scripts.strategy_execution_broker import BrokerError
from tests.test_strategy_execution import positive,reference,NOW,bars

pytestmark=pytest.mark.alpaca_optional
OPEN=datetime(2026,10,26,8,0,2,tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*a,**k):raise AssertionError('No network in order race tests')
    monkeypatch.setattr(socket.socket,'connect',deny)


class MemoryStore:
    account_id='paper-account'
    def __init__(self):self.rows={};self.outbox={}
    @contextmanager
    def transaction(self):yield None
    def enroll(self,handoff,proposal,ref):
        sid=proposal['signal_id']
        if sid not in self.rows:
            self.rows[sid]=dict(signal_id=sid,state='PREPARED',revision=0,payload=dict(
                account_id=self.account_id,handoff=deepcopy(handoff),proposal=deepcopy(proposal),
                reference=deepcopy(ref),position=None,last_completed_session=None,
                pending_exit_reasons=[],pending_exit_session=None,rsi_memory={}))
        return deepcopy(self.rows[sid])
    def signals(self,active=True):
        return deepcopy([s for s in self.rows.values() if not active or s['state'] not in {'CLOSED','EXPIRED'}])
    def save(self,signal,state=None,payload=None):
        assert self.rows[signal['signal_id']]['revision']==signal['revision']
        signal.update(state=state or signal['state'],payload=deepcopy(payload or signal['payload']),revision=signal['revision']+1)
        self.rows[signal['signal_id']]=deepcopy(signal);return signal
    def claim(self,signal_id,role,sequence,operation,request,lineage):
        key=rules.client_order_id(self.account_id,signal_id,role=role,sequence=sequence)
        fresh=key not in self.outbox
        if fresh:self.outbox[key]=dict(client_order_id=key,signal_id=signal_id,role=role,
            sequence=sequence,operation=operation,request=deepcopy(request),state='CLAIMED',observation=None)
        else:assert self.outbox[key]['request']==request
        return dict(deepcopy(self.outbox[key]),fresh=fresh)
    def orders(self,sid):return deepcopy([r for r in self.outbox.values() if r['signal_id']==sid])
    def observed(self,key,order):self.outbox[key].update(state='OBSERVED',observation=deepcopy(order))
    def uncertain(self,key):self.outbox[key]['state']='UNKNOWN'
    def rejected(self,key):self.outbox[key]['state']='REJECTED'


class Broker:
    paper=True;account_id='paper-account'
    def __init__(self):
        self.orders={};self.held={};self.posts=[];self.patches=[];self.cancels=[]
        self.now=OPEN;self.bid=110.;self.mode=None;self.cancel_pending=False;self.quote_error=False
    def clock(self):return dict(timestamp=self.now.isoformat(),is_open=False,
        next_open='2026-10-26T13:30:00+00:00',next_close='2026-10-26T20:00:00+00:00')
    def account(self):return dict(cash='100000',buying_power='200000',non_marginable_buying_power='100000')
    def positions(self):return deepcopy(self.held)
    def open_orders(self):return [deepcopy(o) for o in self.orders.values() if o['status'] in {'new','partially_filled'}]
    def eligible_asset(self,symbol):return True
    def quotes(self,symbols,now):
        if self.quote_error:raise BrokerError('no_quote')
        return {s:dict(bid=self.bid,ask=self.bid+.02,timestamp=self.now.isoformat()) for s in symbols}
    def submit_order(self,request):
        self.posts.append(deepcopy(request))
        if self.mode=='reject':raise BrokerError('refused',http_status=422)
        key=request['client_order_id'];oid=f'{len(self.orders)+1:036d}'
        order=dict(deepcopy(request),id=oid,status='new',filled_qty='0',filled_avg_price=None,
            filled_at=None,updated_at=self.now.isoformat())
        self.orders[key]=order
        if self.mode=='lost':raise BrokerError('transport_unknown')
        return deepcopy(order)
    def find_by_client_order_id(self,key):return deepcopy(self.orders.get(key))
    def get_order(self,oid):return deepcopy(next(o for o in self.orders.values() if o['id']==oid))
    def replace_order(self,oid,request):
        self.patches.append((oid,deepcopy(request)))
        old=next(o for o in self.orders.values() if o['id']==oid);old['status']='replaced'
        value=dict(old,**request,id=f'{len(self.orders)+1:036d}',status='new')
        self.orders[value['client_order_id']]=value;return deepcopy(value)
    def cancel_order(self,oid):
        self.cancels.append(oid)
        for order in self.orders.values():
            if order['id']==oid:order['status']='pending_cancel' if self.cancel_pending else 'canceled'
    def fill(self,role,qty,price=100.,status='filled'):
        order=next(o for o in reversed(list(self.orders.values())) if
            o['side']==('buy' if role=='entry' else 'sell') and
            (o['type']=='stop' if role=='stop' else o['type']=='limit'))
        order.update(status=status,filled_qty=str(qty),filled_avg_price=str(price),
            filled_at=self.now.isoformat() if status=='filled' else None,updated_at=self.now.isoformat())
        symbol=order['symbol']
        bought=sum(int(o['filled_qty']) for o in self.orders.values() if o['side']=='buy')
        sold=sum(int(o['filled_qty']) for o in self.orders.values() if o['side']=='sell')
        if bought>sold:self.held[symbol]=dict(symbol=symbol,qty=str(bought-sold),side='long')
        else:self.held.pop(symbol,None)
        return order


def setup():
    record,rows=positive();handoff=rules.prepare_handoff(record,rows,now=NOW+timedelta(seconds=10))
    store=MemoryStore();broker=Broker();engine=Engine(store,broker,guard=lambda *a:True,now=lambda:broker.now)
    return engine,store,broker,handoff


def enter(engine,broker,handoff,**kw):
    args=dict(now=broker.now,clock=broker.clock(),account=broker.account(),positions=broker.positions(),open_orders=broker.open_orders())
    args.update(kw)
    engine.enter(handoff,{p['symbol']:reference() for p in handoff['proposals']},**args)


def filled_position(qty=None):
    engine,store,broker,handoff=setup()
    if qty==1:broker.account=lambda:dict(cash='2010',buying_power='2010')
    enter(engine,broker,handoff)
    qty=qty or broker.posts[0]['qty'];broker.fill('entry',qty)
    engine.tick(now=broker.now)
    return engine,store,broker,handoff


def test_one_strategy_entry_and_regular_stop_preserve_exact_lineage():
    engine,store,broker,handoff=filled_position()
    assert len(broker.posts)==2
    buy,stop=broker.posts
    assert buy['qty']==49 and buy['limit_price']==100.5 and buy['extended_hours'] is True
    assert stop['type']=='stop' and stop['stop_price']==98.5 and stop['qty']==49
    signal=store.signals()[0]
    assert signal['payload']['handoff']['backtest_result_id']==handoff['backtest_result_id']
    assert signal['payload']['handoff']['metrics_result_id']==handoff['metrics_result_id']
    enter(engine,broker,handoff);assert len(broker.posts)==2


def test_restart_after_accepted_lost_response_never_repeats_post():
    engine,store,broker,handoff=setup();broker.mode='lost';enter(engine,broker,handoff)
    assert store.signals()[0]['state']=='UNKNOWN' and len(broker.posts)==1
    broker.mode=None;resumed=Engine(store,broker,guard=lambda *a:True,now=lambda:broker.now)
    enter(resumed,broker,handoff);assert len(broker.posts)==1
    resumed.tick(now=broker.now);assert store.signals()[0]['state']=='ENTRY_PENDING' # recovered acceptance, still not flat
    broker.fill('entry',49);resumed.tick(now=broker.now)
    assert len(broker.posts)==2 and store.signals()[0]['state']=='OPEN'


def test_explicit_refusal_is_terminal_and_not_retried():
    engine,store,broker,handoff=setup();broker.mode='reject';enter(engine,broker,handoff)
    assert store.signals(active=False)[0]['state']=='EXPIRED'
    enter(engine,broker,handoff);assert len(broker.posts)==1


@pytest.mark.parametrize('reason',['held','pending','cap','no_reference','no_quote','inspection','before','late','guard'])
def test_invalid_or_unavailable_entry_cannot_buy(reason):
    engine,store,broker,handoff=setup();symbol=handoff['proposals'][0]['symbol'];kw={}
    if reason=='held':kw['positions']={symbol:{}}
    elif reason=='pending':kw['open_orders']=[dict(symbol=symbol,side='buy')]
    elif reason=='cap':kw['positions']={str(i):{} for i in range(4)}
    elif reason=='no_reference':
        engine.enter(handoff,{},now=broker.now,clock=broker.clock(),account=broker.account(),positions={},open_orders=[])
        assert broker.posts==[];return
    elif reason=='no_quote':broker.quote_error=True
    elif reason=='inspection':handoff['inspection_mode']=True
    elif reason=='before':broker.now=OPEN-timedelta(seconds=4)
    elif reason=='late':broker.now=OPEN+timedelta(seconds=60)
    elif reason=='guard':engine.guard=lambda *a:False
    if reason in {'inspection','before','late','guard'}:
        with pytest.raises(ValueError):enter(engine,broker,handoff,**kw)
    else:enter(engine,broker,handoff,**kw)
    assert broker.posts==[]


def test_partial_entry_final_average_and_stop_quantity_reconcile():
    engine,store,broker,handoff=setup();enter(engine,broker,handoff)
    broker.fill('entry',10,99.,'partially_filled');engine.tick(now=broker.now)
    assert store.signals()[0]['payload']['position']['entry_price']==99.
    broker.fill('entry',49,100.,'filled');engine.tick(now=broker.now)
    pos=store.signals()[0]['payload']['position']
    assert pos['qty']==49 and pos['entry_price']==100. and pos['atr_stop']==98.5
    assert broker.patches[-1][1]['qty']==49


def test_deadline_cancellation_runs_even_without_quotes_and_waits_for_terminal():
    engine,store,broker,handoff=setup();enter(engine,broker,handoff)
    broker.now=OPEN+timedelta(minutes=35);broker.quote_error=True;broker.cancel_pending=True
    engine.tick(now=broker.now);assert len(broker.cancels)==1 and store.signals()[0]['state']!='EXPIRED'
    engine.tick(now=broker.now);assert len(broker.cancels)==1 # durable cancellation claim
    next(iter(broker.orders.values()))['status']='canceled'
    engine.tick(now=broker.now);assert store.signals(active=False)[0]['state']=='EXPIRED'


def test_breached_stop_never_sells_until_cancellation_and_position_confirmed():
    engine,store,broker,handoff=filled_position();broker.bid=98.;broker.cancel_pending=True
    engine.tick(now=broker.now)
    assert len(broker.posts)==2 and len(broker.cancels)==1
    stop=next(o for o in broker.orders.values() if o['type']=='stop');stop['status']='canceled'
    engine.tick(now=broker.now)
    assert len(broker.posts)==3 and broker.posts[-1]['side']=='sell' and broker.posts[-1]['qty']==49
    broker.fill('exit',49,98.);engine.tick(now=broker.now)
    assert store.signals(active=False)[0]['state']=='CLOSED'


def test_stop_fill_race_closes_from_real_fills_without_second_sale():
    engine,store,broker,handoff=filled_position();broker.fill('stop',49,98.)
    engine.tick(now=broker.now)
    assert len(broker.posts)==2 and store.signals(active=False)[0]['state']=='CLOSED'


def test_missing_position_is_an_ownership_error_not_a_completed_exit():
    engine,store,broker,handoff=filled_position();broker.held={};engine.tick(now=broker.now)
    assert store.signals()[0]['state']=='OPEN' and len(broker.posts)==2
    assert any(e['code']=='POSITION_OWNERSHIP_NOT_RECONCILED' for e in engine.events)


def test_partial_sale_is_not_repeated_after_its_fill_and_pending_sell_blocks_overlap():
    engine,store,broker,handoff=filled_position();signal=store.signals()[0]
    signal['payload'].update(pending_exit_reasons=['PARTIAL_5PCT'],pending_exit_session='2026-10-26')
    store.save(signal);engine.tick(now=broker.now)
    assert broker.posts[-1]['qty']==24
    engine.tick(now=broker.now);assert len(broker.posts)==3
    broker.fill('partial',24,110.);engine.tick(now=broker.now)
    signal=store.signals()[0]
    assert signal['payload']['position']['qty']==25 and signal['payload']['position']['partial_taken'] is True
    assert signal['payload']['pending_exit_reasons']==[]
    engine.tick(now=broker.now)
    assert len([p for p in broker.posts if p['side']=='sell' and p['type']=='limit'])==1


def test_one_share_partial_keeps_share_and_stop():
    engine,store,broker,handoff=filled_position(1);signal=store.signals()[0]
    signal['payload'].update(pending_exit_reasons=['PARTIAL_5PCT'],pending_exit_session='2026-10-26')
    store.save(signal);engine.tick(now=broker.now)
    assert len(broker.posts)==2 and store.signals()[0]['payload']['pending_exit_reasons']==[]


def test_pending_or_partially_filled_stop_is_not_replaced():
    engine,store,broker,handoff=filled_position()
    stop=next(o for o in broker.orders.values() if o['type']=='stop');stop['status']='pending_replace'
    engine.tick(now=broker.now);assert broker.patches==[]
    stop['status']='partially_filled';broker.fill('stop',5,99.,'partially_filled')
    engine.tick(now=broker.now);assert broker.patches==[] and len(broker.posts)==2


def test_completed_daily_bar_advances_once_and_cannot_tighten_before_completion():
    engine,store,broker,handoff=filled_position();signal=store.signals()[0]
    with pytest.raises(ValueError,match='not_completed'):
        engine.completed(signal,bars(),session='2026-10-26',next_session='2026-10-27',now=OPEN)
    engine.completed(signal,bars(),session='2026-10-26',next_session='2026-10-27',now='2026-10-27T00:15:00+00:00')
    assert signal['payload']['position']['atr_stop']==104.
    revision=signal['revision']
    engine.completed(signal,bars(),session='2026-10-26',next_session='2026-10-27',now='2026-10-27T00:16:00+00:00')
    assert signal['revision']==revision


def test_missing_quote_population_is_reported_even_when_a_regular_stop_exists():
    engine,store,broker,handoff=filled_position();broker.quotes=lambda *a,**kw:{}
    result=engine.tick(now=broker.now)
    assert any(e['code']=='STRATEGY_QUOTES_UNAVAILABLE' for e in result['events'])
    assert len(broker.posts)==2


def test_enrollment_crash_without_order_claim_releases_ownership_only_after_deadline():
    engine,store,broker,handoff=setup()
    store.enroll(handoff,handoff['proposals'][0],reference())
    engine.tick(now=broker.now);assert store.signals()[0]['state']=='PREPARED'
    broker.now=OPEN+timedelta(minutes=35);engine.tick(now=broker.now)
    assert store.signals(active=False)[0]['state']=='EXPIRED' and broker.posts==[]
