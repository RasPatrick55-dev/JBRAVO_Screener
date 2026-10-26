"""Bounded native paper broker/data adapter. No connection on import.

Order methods are called only by the durable strategy worker. No generic URL,
live origin, fallback feed, automatic POST retry or account configuration edit.
"""
from datetime import date,datetime,timedelta,time,timezone
import re

from scripts.strategy_backtest import require
from scripts.strategy_execution import after_hours_reference,number,utc,NY
from scripts.executor_health import is_paper_endpoint

PAPER='https://paper-api.alpaca.markets'
DATA='https://data.alpaca.markets'
SYMBOL=re.compile('[A-Z][A-Z0-9.\\-]{0,15}')
ORDER=re.compile('[0-9a-fA-F-]{36}')


class BrokerError(ValueError):
    def __init__(self,reason,*,http_status=None):
        super().__init__(reason)
        self.http_status=http_status


class PaperBroker:
    paper=True
    base_url=PAPER

    def __init__(self,env,*,session=None,max_requests=96,now=None):
        require(is_paper_endpoint(env.get('APCA_API_BASE_URL')) and env.get('JBR_EXEC_PAPER')=='true',
                'strategy_paper_process_context_required')
        key,secret=env.get('APCA_API_KEY_ID'),env.get('APCA_API_SECRET_KEY')
        require(isinstance(key,str) and bool(key) and isinstance(secret,str) and bool(secret),
                'strategy_paper_credentials_unavailable')
        if session is None:
            import requests
            session=requests.Session();session.trust_env=False
        self.session=session
        self._headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret}
        self.account_id=None;self.requests=0;self.max_requests=max_requests
        self.now=now or (lambda:datetime.now(timezone.utc))

    def close(self):
        self.session.close()

    def request(self,method,path,*,params=None,payload=None,data=False,missing_ok=False):
        trade_get=path in {'/v2/account','/v2/clock','/v2/positions','/v2/orders','/v2/orders:by_client_order_id'}
        asset=path.startswith('/v2/assets/') and SYMBOL.fullmatch(path[len('/v2/assets/'):]) is not None
        order=path.startswith('/v2/orders/') and ORDER.fullmatch(path[len('/v2/orders/'):]) is not None
        allowed=(data and method=='GET' and path in {'/v2/stocks/bars','/v2/stocks/quotes/latest'}) or (
            not data and ((method=='GET' and (trade_get or asset or order)) or
            (method=='POST' and path=='/v2/orders') or (method in {'PATCH','DELETE'} and order)))
        require(allowed and self.requests<self.max_requests,'strategy_broker_request_scope_or_budget')
        self.requests+=1
        try:
            response=self.session.request(method,(DATA if data else PAPER)+path,headers=self._headers,
                params=params,json=payload,timeout=(5,10),allow_redirects=False)
        except Exception:
            raise BrokerError('strategy_broker_transport_outcome_unknown') from None
        try:
            if response.status_code==404 and missing_ok:return None
            if response.status_code not in {200,201,204}:
                raise BrokerError('strategy_broker_http_'+str(response.status_code),http_status=response.status_code)
            if response.status_code==204:return None
            require(len(response.content)<=2*1024*1024,'strategy_broker_response_oversize')
            value=response.json()
            require(isinstance(value,(dict,list)),'strategy_broker_response_malformed')
            return value
        finally:
            response.close()

    def account(self):
        value=self.request('GET','/v2/account')
        require(isinstance(value,dict) and isinstance(value.get('id'),str) and
            ORDER.fullmatch(value['id']) is not None and value.get('status')=='ACTIVE' and
            value.get('currency')=='USD' and all(value.get(k) is False for k in
                ('trading_blocked','account_blocked','trade_suspended_by_user')),
            'strategy_paper_account_not_eligible')
        require(self.account_id in (None,value['id']),'strategy_broker_account_changed')
        self.account_id=value['id']
        number(value['cash'])
        number(value['buying_power'])
        return value

    def clock(self):
        value=self.request('GET','/v2/clock')
        require(isinstance(value,dict) and type(value.get('is_open')) is bool,
                'strategy_broker_clock_unavailable')
        utc(value['timestamp']);utc(value['next_open']);utc(value['next_close'])
        return value

    def positions(self):
        rows=self.request('GET','/v2/positions')
        require(isinstance(rows,list) and len(rows)<=1000 and
            len({r.get('symbol') for r in rows})==len(rows),'strategy_positions_unavailable')
        for row in rows:
            require(isinstance(row,dict) and SYMBOL.fullmatch(row.get('symbol','')) is not None,
                    'strategy_position_identity_invalid')
            # Preserve actual shorts in the snapshot; never reinterpret them as
            # owned longs or available entry capacity.
            qty=number(abs(float(row['qty'])),positive=True)
            require(qty>0,'strategy_position_quantity_invalid')
        return {r['symbol']:r for r in rows}

    def open_orders(self):
        rows=self.request('GET','/v2/orders',params=dict(status='open',limit=500,nested=False))
        require(isinstance(rows,list) and len(rows)<500,'strategy_open_order_population_incomplete')
        return rows

    def eligible_asset(self,symbol):
        value=self.request('GET','/v2/assets/'+symbol)
        return (value.get('symbol')==symbol and value.get('status')=='active' and
            value.get('tradable') is True and value.get('class',value.get('asset_class'))=='us_equity')

    def references(self,symbols,signal_session,*,now):
        require(0<len(symbols)<=3 and len(set(symbols))==len(symbols) and
            all(SYMBOL.fullmatch(s) for s in symbols),'strategy_reference_symbol_scope')
        day=date.fromisoformat(signal_session)
        start=datetime.combine(day,time(16),NY);end=datetime.combine(day,time(20),NY)-timedelta(microseconds=1)
        require(utc(now)>=utc(end)+timedelta(minutes=15),'strategy_reference_not_available')
        value=self.request('GET','/v2/stocks/bars',data=True,params=dict(symbols=','.join(symbols),
            start=start.isoformat(),end=end.isoformat(),timeframe='1Min',adjustment='raw',feed='sip',
            currency='USD',sort='asc',limit=1000))
        require(isinstance(value.get('bars'),dict) and value.get('next_page_token') is None and
            set(value['bars'])<=set(symbols),'strategy_reference_response_incomplete')
        return {s:after_hours_reference(value['bars'].get(s,[]),symbol=s,signal_session=signal_session,
            observed_at=utc(self.now()).isoformat(),feed='sip',adjustment='raw',currency='USD',complete=True) for s in symbols}

    def quotes(self,symbols,*,now):
        if not symbols:return {}
        value=self.request('GET','/v2/stocks/quotes/latest',data=True,
            params=dict(symbols=','.join(sorted(symbols)),feed='sip'))
        require(isinstance(value.get('quotes'),dict),'strategy_quote_response_unavailable')
        observed=utc(self.now())
        result={}
        for symbol,quote in value['quotes'].items():
            require(symbol in symbols,'strategy_quote_symbol_changed')
            stamp=utc(quote['t'])
            require(stamp<=observed+timedelta(seconds=1) and
                (observed-stamp).total_seconds()<=60,'strategy_quote_stale_or_future')
            bid,ask=number(quote['bp'],positive=True),number(quote['ap'],positive=True)
            require(bid<=ask,'strategy_quote_crossed')
            result[symbol]=dict(bid=str(bid),ask=str(ask),timestamp=stamp.isoformat())
        return result

    def find_by_client_order_id(self,key):
        return self.request('GET','/v2/orders:by_client_order_id',params=dict(client_order_id=key),missing_ok=True)

    def get_order(self,order_id):
        return self.request('GET','/v2/orders/'+order_id)

    def submit_order(self,request):
        return self.request('POST','/v2/orders',payload=request)

    def replace_order(self,order_id,request):
        return self.request('PATCH','/v2/orders/'+order_id,payload=request)

    def cancel_order(self,order_id):
        self.request('DELETE','/v2/orders/'+order_id)
