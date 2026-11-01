"""Read-only, bounded stored execution gross. No provider, writer or fee policy."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from types import MappingProxyType
import re
from dashboards import execution_read_transport as T
import time
from scripts import trade_disposition as D, gross_paper_accounting as G

BASIS = 'POSTGRES_MATCHED_EXECUTION_GROSS_V1'
SCOPE = 'EXECUTION_RECONSTRUCTED_GROSS'
MAX_CLOSED, MAX_FILLS, MAX_OWNERS = 1000, 4000, 1000


def connect_current_db(db):
    """Same existing resolver/driver/account; a local2s libpq timeout only.

    No config/environment mutation, discovery fallback, exception text or DSN
    logging. DNS/driver behavior is not claimed to be an OS hard deadline.
    """
    try:
        config = db._resolve_db_config()
        if config is None:return None
        require(type(config) is dict, 'EXECUTION_CONNECTION_CONFIG_INVALID')
        options = {key: config.get(key) for key in ('host','port','dbname','user')}
        options['connect_timeout'] = 2
        for key in ('password','sslmode'):
            if config.get(key):options[key]=config[key]
        return db.psycopg2.connect(**options)
    except Exception:
        return None
REQUIRED_COLUMNS = {
    'trades': {'trade_id': {'int4','int8'}, 'status': {'text','varchar'}, 'symbol': {'text','varchar'},
               'qty': {'numeric','float8','int4','int8'}, 'entry_order_id': {'uuid','text','varchar'},
               'exit_order_id': {'uuid','text','varchar'}, 'entry_time': {'timestamp','timestamptz'},
               'exit_time': {'timestamp','timestamptz'}, 'updated_at': {'timestamp','timestamptz'}},
    'alpaca_activities': {'activity_pk': {'int4','int8'}, 'activity_id': {'text','varchar'},
        'activity_type': {'text','varchar'}, 'order_id': {'uuid','text','varchar'}, 'symbol': {'text','varchar'},
        'side': {'text','varchar'}, 'qty': {'numeric','float8','int4','int8'}, 'price': {'numeric','float8'},
        'transaction_time': {'timestamp','timestamptz'}, 'raw': {'json','jsonb'}}}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def exact(value, *, positive=False):
    require(isinstance(value, (str, int, Decimal)) and not isinstance(value, bool)
            and len(str(value)) <= 64, 'EXECUTION_NUMBER_INVALID')
    result = Decimal(value)
    require(result.is_finite() and -12 <= result.as_tuple().exponent <= 15
            and (not result or -12 <= result.adjusted() <= 15), 'EXECUTION_NUMBER_INVALID')
    require(not positive or result > 0, 'EXECUTION_NUMBER_INVALID')
    return result


def order_id(value):
    require(isinstance(value, str) and 0 < len(value) <= 128 and value == value.strip(), 'ORDER_ID_INVALID')
    return value


def aware(value):
    if isinstance(value, datetime):
        require(value.utcoffset() is not None, 'TIME_INVALID')
        return value.astimezone(timezone.utc)
    return D.aware_time(value).astimezone(timezone.utc)


@dataclass(frozen=True)
class Projection:
    gross: object
    unknown: object
    records_hash: str
    policy_hash: str
    source_hash: str

    def verify(self, policy, records):
        require(self.records_hash == D.fingerprint({'rows': records})
                and self.policy_hash == D.fingerprint(policy.metadata()), 'EXECUTION_BINDING_CHANGED')


def reconstruct(policy, records, fills, owner_records, *, versions=None):
    require(type(records) is list and len(records) <= MAX_CLOSED and type(fills) is list
            and len(fills) <= MAX_FILLS and type(owner_records) is list
            and len(owner_records) <= MAX_OWNERS, 'EXECUTION_POPULATION_INVALID')
    ids = [D.positive_id(row.get('trade_id')) for row in records]
    require(len(ids) == len(set(ids)) and all(row.get('status') == 'CLOSED' for row in records), 'EXECUTION_POPULATION_INVALID')
    allocations = {item.trade_id: item for item in policy.allocations}
    allocated_rows = [row for row in records if row['trade_id'] in allocations]
    # Canonical overlay identity/fingerprint/reserved claims remain mandatory.
    snapshot = G.build_snapshot(policy, allocated_rows, expected_identity=(policy.deployment_id, policy.database_id),
                                expected_revision=policy.revision)
    audited = {row['trade_id']: Decimal(row['gross_pnl']) for row in snapshot['rows']}
    reserved = D.document(policy.allocation_metadata_json).get('reserved_order_owners')
    require(isinstance(reserved, dict), 'EXECUTION_RESERVATIONS_UNAVAILABLE')
    owners, by_order, bad_orders = defaultdict(set), defaultdict(list), set()
    for order, identifier in reserved.items():
        owners[order_id(order)].add(D.positive_id(identifier))
    seen_owners = set()
    for row in owner_records:
        identifier = D.positive_id(row.get('trade_id'))
        require(identifier not in seen_owners, 'EXECUTION_OWNER_DUPLICATE')
        seen_owners.add(identifier)
        for key in ('entry_order_id', 'exit_order_id'):
            if row.get(key) not in (None, ''):
                owners[order_id(row[key])].add(identifier)
    # Every physical claim in the selected cohort must be represented, including
    # a conflicting external/non-CLOSED claimant returned by the exact query.
    for row in records:
        require(row['trade_id'] in seen_owners, 'EXECUTION_OWNER_POPULATION_MISSING')
        for key in ('entry_order_id', 'exit_order_id'):
            if row.get(key) not in (None, ''):
                require(row['trade_id'] in owners[order_id(row[key])], 'EXECUTION_OWNER_BINDING_CHANGED')
    seen_fills = {}
    for row in fills:
        require(row.get('activity_type') == 'FILL', 'EXECUTION_ACTIVITY_TYPE_INVALID')
        key = order_id(row.get('activity_id'))
        if key in seen_fills:
            require(seen_fills[key] == row, 'EXECUTION_ACTIVITY_CONFLICT')
            continue
        seen_fills[key] = row
        order = order_id(row.get('order_id'))
        try:
            ids = [row[key] for key in ('provider_id', 'provider_activity_id') if row.get(key) is not None]
            quantities = [row[key] for key in ('provider_qty', 'provider_quantity') if row.get(key) is not None]
            require(ids and all(value == row['activity_id'] for value in ids) and row.get('provider_order_id') == order
                    and row.get('provider_symbol') == row.get('symbol')
                    and row.get('provider_side') == row.get('side'), 'EXECUTION_ALIAS_INVALID')
            require(row.get('provider_activity_type') in (None, 'FILL'), 'EXECUTION_ALIAS_INVALID')
            require(row.get('side') in ('buy', 'sell', 'sell_short'), 'EXECUTION_SIDE_INVALID')
            qty, price = exact(row.get('qty'), positive=True), exact(row.get('price'), positive=True)
            require(quantities and all(qty == exact(value, positive=True) for value in quantities)
                    and price == exact(row.get('provider_price'), positive=True), 'EXECUTION_ALIAS_INVALID')
            by_order[order].append(dict(qty=qty, price=price, side=row['side'],
                                       symbol=row.get('symbol'), stamp=aware(row.get('transaction_time'))))
        except (ValueError, ArithmeticError, D.TradePolicyError, TypeError):
            bad_orders.add(order)  # One unusable fill taints the entire order.
    gross, unknown = {}, {}
    for row in records:
        identifier = row['trade_id']
        if identifier in policy.excluded_ids:
            continue
        if identifier in allocations:
            # The real canonical reader validates current source rows/fills;
            # project() also repeats the full pinned physical-row check.
            D.require_order_owner(policy, identifier, *allocations[identifier].order_ids)
            gross[identifier] = audited[identifier]
            continue
        try:
            entry, exit_ = order_id(row.get('entry_order_id')), order_id(row.get('exit_order_id'))
            reason = 'ORDER_OWNER_AMBIGUOUS' if entry == exit_ or any(owners[o] != {identifier} for o in (entry, exit_)) else None
            if reason is None and any(o in bad_orders for o in (entry, exit_)):
                reason = 'FILL_ECONOMICS_UNRESOLVED'
            incoming, outgoing = by_order[entry], by_order[exit_]
            if reason is None and (not incoming or not outgoing):
                reason = 'ORDER_FILL_MISSING'
            if reason is None and any(f['symbol'] != row.get('symbol') for f in incoming + outgoing):
                reason = 'SYMBOL_MISMATCH'
            if reason is None and ({f['side'] for f in incoming}, {f['side'] for f in outgoing}) != ({'buy'}, {'sell'}):
                reason = 'SIDE_OR_LIFECYCLE_UNRESOLVED'
            with localcontext() as ctx:
                ctx.prec = 80
                qty = exact(row.get('qty'), positive=True)
                if reason is None and (sum((f['qty'] for f in incoming), Decimal(0)) != qty
                                       or sum((f['qty'] for f in outgoing), Decimal(0)) != qty):
                    reason = 'QUANTITY_MISMATCH'
                entered, exited = aware(row.get('entry_time')), aware(row.get('exit_time'))
                if reason is None and (entered > exited or max(f['stamp'] for f in incoming) > min(f['stamp'] for f in outgoing)
                                       or max(f['stamp'] for f in outgoing) > exited):
                    reason = 'EXECUTION_CHRONOLOGY_UNRESOLVED'
                if reason is None:
                    gross[identifier] = sum((f['qty'] * f['price'] for f in outgoing), Decimal(0)) - sum((f['qty'] * f['price'] for f in incoming), Decimal(0))
            if reason:
                unknown[identifier] = reason
        except (ValueError, ArithmeticError, D.TradePolicyError, TypeError):
            unknown[identifier] = 'ORDER_OR_EXECUTION_INPUT_UNAVAILABLE'
    return Projection(MappingProxyType(gross), MappingProxyType(unknown), D.fingerprint({'rows': records}),
                      D.fingerprint(policy.metadata()), D.fingerprint({'fills': fills, 'owners': owner_records, 'versions': versions}))


class BoundedCursor(T.Cursor):
    def __init__(self, raw, deadline, clock=time.monotonic):
        super().__init__(raw, deadline, extra_plans=READ_PLANS, clock=clock)


def documents(cursor, sql, params, cap):
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    require(len(rows) <= cap and all(row[0] is not None for row in rows), 'EXECUTION_SATURATION_OR_OVERSIZE')
    return [D.document(row[0]) for row in rows]


def guarded(body):
    return 'SELECT CASE WHEN pg_catalog.octet_length(body)<=32768 THEN body END FROM (' + body + ') encoded'


VISIBILITY_SQL="SELECT c.relname,c.relkind,c.relrowsecurity,c.relforcerowsecurity,c.relispartition,EXISTS(SELECT 1 FROM pg_catalog.pg_inherits i WHERE i.inhrelid=c.oid OR i.inhparent=c.oid),pg_catalog.has_table_privilege(current_user,c.oid,'SELECT') FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relname IN ('trades','alpaca_activities') ORDER BY c.relname LIMIT 3"
SCHEMA_SQL="SELECT c.relname,a.attname,t.typname FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid=a.attrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace JOIN pg_catalog.pg_type t ON t.oid=a.atttypid WHERE n.nspname='public' AND c.relname IN ('trades','alpaca_activities') AND a.attnum>0 AND NOT a.attisdropped ORDER BY c.relname,a.attname LIMIT 65"
LOCK_SQL='LOCK TABLE public.trades, public.alpaca_activities IN ACCESS SHARE MODE'
TRADES_SQL=guarded("SELECT pg_catalog.jsonb_build_object('row',pg_catalog.to_jsonb(t),'version',t.xmin::text)::text AS body FROM ONLY public.trades t WHERE status='CLOSED' ORDER BY trade_id LIMIT 1001")
OWNERS_SQL=guarded("SELECT pg_catalog.jsonb_build_object('row',pg_catalog.jsonb_build_object('trade_id',trade_id,'status',status,'entry_order_id',entry_order_id::text,'exit_order_id',exit_order_id::text),'version',xmin::text)::text AS body FROM ONLY public.trades WHERE trade_id=ANY(%s) OR entry_order_id::text=ANY(%s) OR exit_order_id::text=ANY(%s) ORDER BY trade_id LIMIT 1001")
FILLS_SQL=guarded("SELECT pg_catalog.jsonb_build_object('row',pg_catalog.jsonb_build_object('activity_id',activity_id,'activity_type',activity_type,'order_id',order_id,'symbol',symbol,'side',side,'qty',qty::text,'price',price::text,'transaction_time',transaction_time,'provider_id',raw::jsonb->>'id','provider_activity_id',raw::jsonb->>'activity_id','provider_activity_type',raw::jsonb->>'activity_type','provider_order_id',raw::jsonb->>'order_id','provider_symbol',raw::jsonb->>'symbol','provider_side',raw::jsonb->>'side','provider_qty',raw::jsonb->>'qty','provider_quantity',raw::jsonb->>'quantity','provider_price',raw::jsonb->>'price'),'version',xmin::text)::text AS body FROM ONLY public.alpaca_activities WHERE activity_type='FILL' AND order_id::text=ANY(%s) ORDER BY activity_id LIMIT 4001")
READ_PLANS={"SET LOCAL lock_timeout = '500ms'":dict(kind='control',arity=0,order=None),
 "SET LOCAL idle_in_transaction_session_timeout = '8000ms'":dict(kind='control',arity=0,order=None),
 VISIBILITY_SQL:dict(kind='canonical',arity=7,order='q.c0'),
 SCHEMA_SQL:dict(kind='canonical',arity=3,order='q.c0,q.c1'),
 LOCK_SQL:dict(kind='control',arity=0,order=None),
 TRADES_SQL:dict(kind='canonical',arity=1,order="(q.c0::jsonb->'row'->>'trade_id')::bigint"),
 OWNERS_SQL:dict(kind='canonical',arity=1,order="(q.c0::jsonb->'row'->>'trade_id')::bigint"),
 FILLS_SQL:dict(kind='canonical',arity=1,order="q.c0::jsonb->'row'->>'activity_id'")}


def source_guard(cursor):
    cursor.execute(VISIBILITY_SQL)
    require(cursor.fetchall() == [('alpaca_activities','r',False,False,False,False,True),
                                 ('trades','r',False,False,False,False,True)], 'EXECUTION_SOURCE_VISIBILITY_UNAVAILABLE')
    cursor.execute(SCHEMA_SQL)
    rows=cursor.fetchall()
    require(len(rows)<=64,'EXECUTION_SCHEMA_CAP')
    found={(table,name):kind for table,name,kind in rows}
    require(len(found)==len(rows) and all(found.get((table,name)) in kinds
        for table,columns in REQUIRED_COLUMNS.items() for name,kinds in columns.items()),'EXECUTION_SOURCE_SCHEMA_UNAVAILABLE')


def split_versions(documents,key):
    rows=[];versions=[]
    for document in documents:
        require(type(document) is dict and set(document)=={'row','version'} and type(document['row']) is dict
            and isinstance(document['version'],str) and re.fullmatch(r'[0-9]{1,10}',document['version']) is not None,
            'EXECUTION_ROW_VERSION_UNAVAILABLE')
        rows.append(document['row']);versions.append([document['row'].get(key),document['version']])
    return rows,versions


def read_inputs(cursor,policy):
    cursor.execute(LOCK_SQL);source_guard(cursor)
    records,trade_versions=split_versions(documents(cursor,TRADES_SQL,None,MAX_CLOSED),'trade_id')
    orders=set()
    for row in records:
        for key in ('entry_order_id','exit_order_id'):
            if row.get(key) not in (None,''):orders.add(order_id(row[key]))
    orders.update(order for allocation in policy.allocations for order in allocation.order_ids)
    require(len(orders)<=2128,'EXECUTION_ORDER_CAP')
    owners,owner_versions=split_versions(documents(cursor,OWNERS_SQL,
        ([row['trade_id'] for row in records],sorted(orders),sorted(orders)),MAX_OWNERS),'trade_id')
    fills,fill_versions=split_versions(documents(cursor,FILLS_SQL,(sorted(orders),),MAX_FILLS),'activity_id')
    versions=dict(trades=trade_versions,owners=owner_versions,fills=fill_versions)
    return records,reconstruct(policy,records,fills,owners,versions=versions)
