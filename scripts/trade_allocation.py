"""Approval-bound CRC110 native-fill cohort overlay; no imports with runtime effects.

This module neither discovers authority/connections nor migrates or activates on
import. Gross economics are observational and never become net financial input.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, Context, localcontext
from types import MappingProxyType
from uuid import UUID
import json
import hashlib

TABLES = ('trade_allocation_cases', 'trade_allocation_audit',
          'trade_allocation_fill_claims', 'trade_allocation_order_claims')
MAX_HISTORY = 32
ROW_FINGERPRINT = '076b62b8ece8ec28a86e3254efad1a86c9c10ad50bdf47bc23bbc9ec9a7800b6'
ROW_VERSION = '2026-04-17T19:05:44.607608+00:00'
ENTRY_ORDER = '7ad70913-5833-4f98-a3cc-d9a590da02a6'
EXIT_ORDERS = ('da569a6d-2be6-4d1e-a3ae-9c639fce3668', '1e248c11-d86f-431c-9c1a-ddd6e36365ff')
ORDERS = (ENTRY_ORDER,) + EXIT_ORDERS
# Stored PK/native identity, side, order, delta quantity/price, native time.
FILLS = (
    (9728, '20260416070011645::d5509770-cd64-4226-bb1f-a450afb2d86f', 'buy', ENTRY_ORDER, '18', '65', '2026-04-16T11:00:11.645273+00:00'),
    (9729, '20260416070012026::b36c54f3-3ce9-4319-86f0-18c362cc8c76', 'buy', ENTRY_ORDER, '12', '65', '2026-04-16T11:00:12.026949+00:00'),
    (9758, '20260417093234756::d1937646-ea57-4025-9def-2841d93e3c02', 'sell', EXIT_ORDERS[0], '1', '63', '2026-04-17T13:32:34.756057+00:00'),
    (9759, '20260417093322458::3e97f2b5-9f86-44c5-8539-d0df12a68c9d', 'sell', EXIT_ORDERS[0], '14', '62.48', '2026-04-17T13:33:22.458261+00:00'),
    (9763, '20260417141233697::e19e2e95-e0e0-41a7-a84d-1d8f4a19019c', 'sell', EXIT_ORDERS[1], '15', '63.26', '2026-04-17T18:12:33.697462+00:00'),
)
ACTIVITY_COLUMNS = frozenset({'activity_pk','activity_id','activity_type','transaction_time','symbol',
                             'side','qty','price','amount','order_id','description','raw','created_at'})
CREATED_AT = ('2026-04-16T11:15:49.858458+00:00', '2026-04-16T11:15:49.858458+00:00',
              '2026-04-17T14:15:42.253092+00:00', '2026-04-17T14:15:42.253092+00:00',
              '2026-04-17T18:15:54.773706+00:00')
SOURCE_PINS = {
    'CRC110-SOURCE-BACKED-RESOLUTION-003.json': '7ac0bb274f4321c896fb352463652b1b0008659ddd9ade051adf4a34551894b0',
    'runtime-readonly-001/trade-110-full-snapshot-003.json': '0f0c84d1ab9d44374994612b74d7b56a5a8fd9ecd66afe5757ef50b464adfbf8',
    'runtime-readonly-001/storage-owner-output-003.txt': 'e6bf64622186101a96508f03d681033651bcabed91cf586bdb8f640083dae303',
}
FUNCTION_PINS = {
    'trade_allocation_immutable': '472794423e0aad3b29ec4a7250e9f174d9971424d6913377d94a8839180da580',
    'trade_allocation_lock_policy': '4597cbcff1398afad0499fb181833b32c7fa1a8187614d9cc612febf54949336',
    'trade_allocation_case_guard': 'aca6becbc868acc8d2bb043ef34ae7bc5703e83ea25b9063cd0c7178a71cf335',
    'trade_allocation_guard_original': '8ba52975ee99bcd6e3b251e62d6ee60f052e1be2256769beb21de083e1dd79d0',
    'trade_allocation_marker_immutable': 'a77c90840c9ec9dfbb213bc3557b471b6a05efb3d0c4accde45380039d8dd3a9',
}
TRIGGER_CONTRACT = {
    ('trade_allocation_audit', 'trade_allocation_audit_immutable'): ('trade_allocation_immutable', 58, False),
    ('trade_allocation_fill_claims', 'trade_allocation_fill_claims_immutable'): ('trade_allocation_immutable', 58, False),
    ('trade_allocation_order_claims', 'trade_allocation_order_claims_immutable'): ('trade_allocation_immutable', 58, False),
    ('trade_allocation_cases', 'trade_allocation_cases_no_truncate'): ('trade_allocation_immutable', 34, False),
    ('trades', 'trade_allocation_trade_statement'): ('trade_allocation_lock_policy', 30, False),
    ('alpaca_activities', 'trade_allocation_activity_statement'): ('trade_allocation_lock_policy', 30, False),
    ('trade_allocation_cases', 'trade_allocation_cases_guard'): ('trade_allocation_case_guard', 31, False),
    ('trades', 'trade_allocation_trade_guard'): ('trade_allocation_guard_original', 31, False),
    ('alpaca_activities', 'trade_allocation_activity_guard'): ('trade_allocation_guard_original', 31, False),
    ('trade_disposition_policy', 'trade_allocation_marker_guard'): ('trade_allocation_marker_immutable', 19, True),
}

class AllocationError(RuntimeError):
    pass

def fail(code):
    raise AllocationError('TRADE_ALLOCATION_' + code)

def require(value, code):
    if not value:
        fail(code)

def policy_module():
    from scripts import trade_disposition
    return trade_disposition

def number(value):
    require(not isinstance(value, (float, bool)), 'DECIMAL_INVALID')
    try:
        result = Decimal(value)
    except (TypeError, ValueError, InvalidOperation):
        fail('DECIMAL_INVALID')
    require(result.is_finite() and len(result.as_tuple().digits) <= 40
            and abs(result.as_tuple().exponent) <= 40, 'DECIMAL_INVALID')
    return result

def uuid(value):
    try:
        require(type(value) is str and str(UUID(value)) == value, 'UUID_INVALID')
    except (ValueError, TypeError, AttributeError):
        fail('UUID_INVALID')
    return value

def bounded_document(value):
    D = policy_module()
    require(not isinstance(value, str) or len(value.encode('utf-8')) <= 196608, 'DOCUMENT_BOUND')
    def no_native_float(item, depth=0):
        require(depth <= 12, 'DOCUMENT_BOUND')
        require(not isinstance(item, float), 'DECIMAL_INVALID')
        if isinstance(item, dict):
            require(len(item) <= 128, 'DOCUMENT_BOUND')
            for nested in item.values():
                no_native_float(nested, depth + 1)
        elif isinstance(item, list):
            require(len(item) <= 128, 'DOCUMENT_BOUND')
            for nested in item:
                no_native_float(nested, depth + 1)
    if not isinstance(value, str):
        no_native_float(value)
    result = D.document(value)
    # The parser preserves PostgreSQL numerics; native float input is forbidden.
    def check(item, depth=0):
        require(depth <= 12, 'DOCUMENT_BOUND')
        if isinstance(item, dict):
            require(len(item) <= 128, 'DOCUMENT_BOUND')
            for key, nested in item.items():
                require(len(key) <= 256, 'DOCUMENT_BOUND')
                check(nested, depth + 1)
        elif isinstance(item, list):
            require(len(item) <= 128, 'DOCUMENT_BOUND')
            for nested in item:
                check(nested, depth + 1)
        elif isinstance(item, str):
            require(len(item.encode('utf-8')) <= 32768, 'DOCUMENT_BOUND')
    check(result)
    require(len(D.canonical(result).encode('utf-8')) <= 196608, 'DOCUMENT_BOUND')
    return result

def validate_trade(snapshot):
    D = policy_module()
    snapshot = bounded_document(snapshot)
    require(D.fingerprint(snapshot) == ROW_FINGERPRINT and snapshot.get('trade_id') == 110
            and snapshot.get('updated_at') == ROW_VERSION, 'ROW_CHANGED')
    require(snapshot.get('status') == 'CLOSED' and snapshot.get('symbol') == 'CRC'
            and snapshot.get('entry_order_id') == ENTRY_ORDER and snapshot.get('exit_order_id') is None
            and snapshot.get('exit_price') is None and snapshot.get('realized_pnl') is None
            and number(snapshot.get('qty')) == 30 and number(snapshot.get('entry_price')) == 65, 'ROW_INVALID')
    return snapshot

def validate_fills(records):
    # Caller Decimal precision/traps must not change the native cohort arithmetic.
    with localcontext(Context(prec=80)):
        return _validate_fills(records)

def _validate_fills(records):
    D = policy_module()
    require(type(records) is list and len(records) == 5, 'FILL_COUNT')
    expected = {item[1]: item for item in FILLS}
    seen, snapshots = set(), {}
    entry_qty = exit_qty = cost = proceeds = Decimal(0)
    times = {'buy': [], 'sell': []}
    for record in records:
        require(type(record) is dict and set(record) == {'snapshot', 'row_fingerprint'}, 'FILL_DOCUMENT')
        snapshot = bounded_document(record['snapshot'])
        require(set(snapshot) == ACTIVITY_COLUMNS, 'FILL_SNAPSHOT_INCOMPLETE')
        require(D.fingerprint(snapshot) == D.digest(record['row_fingerprint']), 'FILL_FINGERPRINT')
        native = snapshot.get('activity_id')
        require(native in expected and native not in seen, 'FILL_DUPLICATE_OR_UNKNOWN')
        pk, _, side, order, qty, price, time = expected[native]
        require(snapshot.get('activity_pk') == pk and snapshot.get('activity_type') == 'FILL'
                and snapshot.get('symbol') == 'CRC' and snapshot.get('side') == side
                and snapshot.get('order_id') == order, 'FILL_IDENTITY')
        require(number(snapshot.get('qty')) == Decimal(qty)
                and number(snapshot.get('price')) == Decimal(price)
                and D.aware_time(snapshot.get('transaction_time')) == D.aware_time(time)
                and D.aware_time(snapshot.get('created_at')) == D.aware_time(CREATED_AT[FILLS.index(expected[native])]), 'FILL_FACT_CHANGED')
        require(snapshot.get('amount') is None and 'raw' in snapshot, 'FILL_SNAPSHOT_INCOMPLETE')
        q, p = number(snapshot['qty']), number(snapshot['price'])
        require(q > 0 and p > 0, 'FILL_DOMAIN')
        if side == 'buy':
            entry_qty += q
            cost += q * p
        else:
            exit_qty += q
            proceeds += q * p
        times[side].append(D.aware_time(time))
        seen.add(native)
        snapshots[native] = snapshot
    require(entry_qty == exit_qty == Decimal(30) and cost == Decimal(1950)
            and proceeds == Decimal('1886.62') and proceeds - cost == Decimal('-63.38')
            and max(times['buy']) < min(times['sell']), 'CONSERVATION_FAILED')
    return snapshots

def validate_request(request, manifest, expected_identity):
    D = policy_module()
    request, manifest = bounded_document(request), bounded_document(manifest)
    required = {'action', 'allocation_id', 'trade_id', 'row_fingerprint', 'row_version', 'deployment_id',
                'database_id', 'expected_policy_revision', 'expected_allocation_revision', 'manifest_sha256',
                'actor', 'approved_by', 'approved_at', 'approval_ref'}
    require(set(request) == required and request['action'] in ('APPLY', 'REVOKE'), 'APPROVAL_INVALID')
    require(D.identity(request['deployment_id'], request['database_id']) == expected_identity, 'IDENTITY_MISMATCH')
    uuid(request['allocation_id'])
    require(type(request['trade_id']) is int and request['trade_id'] == 110
            and request['row_fingerprint'] == ROW_FINGERPRINT and request['row_version'] == ROW_VERSION, 'TARGET_INVALID')
    D.revision(request['expected_policy_revision'])
    D.revision(request['expected_allocation_revision'])
    require(request['expected_allocation_revision'] < MAX_HISTORY, 'HISTORY_BOUND')
    for key in ('actor', 'approved_by', 'approval_ref'):
        D.text_field(request[key])
    require(D.aware_time(request['approved_at']) <= datetime.now(timezone.utc), 'APPROVAL_TIME')
    require(D.digest(request['manifest_sha256']) == D.fingerprint(manifest), 'MANIFEST_DIGEST')
    for key in ('allocation_id', 'trade_id', 'row_fingerprint', 'row_version', 'deployment_id', 'database_id'):
        require(manifest.get(key) == request[key], 'MANIFEST_BINDING')
    require(manifest.get('classification') == 'NATIVE_FILL_COHORT' and manifest.get('resolution_scope') == 'RECONCILIATION_ONLY'
            and manifest.get('financial_eligible') is False and manifest.get('fees') is None
            and manifest.get('net') is None and manifest.get('source_pins') == SOURCE_PINS, 'MANIFEST_SCOPE')
    D.text_field(manifest.get('rationale'))
    evidence = manifest.get('evidence')
    require(isinstance(evidence, list) and 1 <= len(evidence) <= 32, 'EVIDENCE_INVALID')
    for item in evidence:
        require(type(item) is dict and set(item) == {'reference', 'sha256'}, 'EVIDENCE_INVALID')
        D.text_field(item['reference'])
        D.digest(item['sha256'])
    validate_trade(manifest.get('row_snapshot'))
    validate_fills(manifest.get('fills'))
    return request, manifest

@dataclass(frozen=True)
class Allocation:
    allocation_id: str
    trade_id: int
    state: str
    revision: int
    entry_quantity: Decimal
    exit_quantity: Decimal
    gross_result: Decimal
    approval_ref: str
    source_refs: tuple
    order_ids: tuple
    activity_ids: tuple
    fees: None = None
    net: None = None
    financial_eligible: bool = False

    @property
    def allocation_revision(self):
        return self.revision

    def metadata_dict(self):
        return {'allocation_id': self.allocation_id, 'trade_id': self.trade_id, 'state': self.state,
                'allocation_revision': self.revision, 'entry_quantity': str(self.entry_quantity),
                'exit_quantity': str(self.exit_quantity), 'gross_result': str(self.gross_result),
                'fees': None, 'net': None, 'fees_status': 'UNKNOWN', 'net_status': 'UNKNOWN',
                'financial_eligible': False, 'audit_refs': [self.approval_ref], 'source_refs': list(self.source_refs),
                'order_ids': list(self.order_ids), 'activity_ids': list(self.activity_ids)}

@dataclass(frozen=True)
class AllocationRead:
    allocations: tuple
    metadata: object
    projection: object
    reserved_order_owners: object
    _metadata_json: str

    def metadata_dict(self):
        return json.loads(self._metadata_json)

def result_read(status, allocations, expected_identity, policy_revision, reserved_order_owners=None, reservation_cases=()):
    D = policy_module()
    reserved_order_owners = dict(reserved_order_owners or {})
    metadata = {'status': status, 'deployment_id': expected_identity[0], 'database_id': expected_identity[1],
                'policy_revision': policy_revision, 'active_count': len(allocations),
                'allocation_schema_version': 0 if status == 'MIGRATION_PENDING' else 1,
                'allocations': [item.metadata_dict() for item in allocations],
                'reserved_order_owners': reserved_order_owners,
                'reservation_cases': list(reservation_cases),
                'fees_status': 'UNKNOWN', 'net_status': 'UNKNOWN', 'financial_eligible': False}
    metadata['binding_fingerprint'] = D.fingerprint(metadata)
    encoded = D.canonical(metadata)
    # Nested metadata retained here is immutable too; metadata_dict is the export.
    def freeze(value):
        if isinstance(value, dict):
            return MappingProxyType({key: freeze(item) for key, item in value.items()})
        if isinstance(value, list):
            return tuple(freeze(item) for item in value)
        return value
    return AllocationRead(tuple(allocations), freeze(metadata),
                          MappingProxyType({item.trade_id: item for item in allocations}),
                          MappingProxyType(reserved_order_owners), encoded)

def schema_present(cursor):
    cursor.execute('SELECT pg_catalog.to_regclass(%s),pg_catalog.to_regclass(%s),'
                   'pg_catalog.to_regclass(%s),pg_catalog.to_regclass(%s)', tuple('public.' + name for name in TABLES))
    row = cursor.fetchone()
    require(row is not None and len(row) == 4, 'SCHEMA_METADATA')
    cursor.execute('SELECT a.atttypid=%s::pg_catalog.regtype,a.attnotnull '
        'FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid=a.attrelid '
        'JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace '
        "WHERE n.nspname='public' AND c.relname='trade_disposition_policy' "
        "AND a.attname='allocation_schema_version' AND a.attnum>0 AND NOT a.attisdropped", ('pg_catalog.int4',))
    marker = cursor.fetchone()
    if marker is None:
        require(all(item is None for item in row), 'SCHEMA_MARKER_MISSING')
        return False  # Legacy schema has never recorded an allocation installation.
    require(marker == (True, True), 'SCHEMA_MARKER_INVALID')
    cursor.execute('SELECT allocation_schema_version FROM public.trade_disposition_policy WHERE singleton=1')
    version = cursor.fetchone()
    require(version is not None and type(version[0]) is int and version[0] in (0, 1), 'SCHEMA_MARKER_INVALID')
    if version[0] == 0:
        require(all(item is None for item in row), 'PARTIAL_MIGRATION')
        return False
    require(all(item is not None for item in row), 'PARTIAL_MIGRATION')
    cursor.execute('SELECT c.relname,c.relkind,c.relrowsecurity,c.relforcerowsecurity,'
                   'EXISTS(SELECT 1 FROM pg_catalog.pg_inherits i WHERE i.inhrelid=c.oid OR i.inhparent=c.oid),'
                   "pg_catalog.has_table_privilege(c.oid,'SELECT') FROM pg_catalog.pg_class c "
                   'JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace '
                   "WHERE n.nspname='public' AND c.relname=ANY(%s) ORDER BY c.relname", (list(TABLES),))
    metadata = rows(cursor, 4)
    require(len(metadata) == 4 and {item[0] for item in metadata} == set(TABLES)
            and all(kind == 'r' and not rls and not force and not inherited and visible
                    for name, kind, rls, force, inherited, visible in metadata), 'SCHEMA_VISIBILITY_INVALID')
    attest_contract(cursor)
    return True

def attest_contract(cursor):
    """All trigger/source metadata remains internal; no arbitrary bodies are read."""
    cursor.execute("SELECT pg_catalog.current_setting('session_replication_role')")
    require(cursor.fetchone() == ('origin',), 'TRIGGER_SESSION_INVALID')
    cursor.execute('SELECT p.proname,CASE WHEN pg_catalog.octet_length(p.prosrc)<=16384 THEN p.prosrc ELSE NULL END,'
        "p.prosecdef,p.proconfig=ARRAY['search_path=pg_catalog']::pg_catalog.text[],"
        'p.proowner=policy.relowner,l.lanname,p.prorettype=%s::pg_catalog.regtype,'
        'p.prokind,p.proisstrict,p.proleakproof,p.provolatile,p.proparallel,p.pronargs '
        'FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace '
        'JOIN pg_catalog.pg_language l ON l.oid=p.prolang '
        'CROSS JOIN pg_catalog.pg_class policy JOIN pg_catalog.pg_namespace pn ON pn.oid=policy.relnamespace '
        "WHERE n.nspname='public' AND p.proname=ANY(%s) AND p.pronargs=0 "
        "AND pn.nspname='public' AND policy.relname='trade_disposition_policy' ORDER BY p.proname",
        ('pg_catalog.trigger', list(FUNCTION_PINS)))
    functions = rows(cursor, 5)
    require(len(functions) == 5 and {item[0] for item in functions} == set(FUNCTION_PINS), 'TRIGGER_FUNCTION_MISSING')
    for name, body, security_definer, config, owner, language, returns_trigger, kind, strict, leakproof, volatile, parallel, args in functions:
        require(type(body) is str and hashlib.sha256(body.encode('utf-8')).hexdigest() == FUNCTION_PINS[name]
                and not security_definer and config is True and owner
                and language == 'plpgsql' and returns_trigger and kind == 'f' and not strict
                and not leakproof and volatile == 'v' and parallel == 'u' and args == 0, 'TRIGGER_FUNCTION_CHANGED')
    cursor.execute('SELECT c.relname,t.tgname,p.proname,n.nspname,t.tgtype,t.tgenabled,t.tgqual IS NULL,'
        't.tgnargs,pg_catalog.octet_length(t.tgargs),t.tgconstraint=0,t.tgattr::text,'
        '(SELECT a.attnum::text FROM pg_catalog.pg_attribute a WHERE a.attrelid=policy.oid '
        "AND a.attname='allocation_schema_version' AND a.attnum>0 AND NOT a.attisdropped) "
        'FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid '
        'JOIN pg_catalog.pg_namespace cn ON cn.oid=c.relnamespace '
        'JOIN pg_catalog.pg_proc p ON p.oid=t.tgfoid JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace '
        'CROSS JOIN pg_catalog.pg_class policy JOIN pg_catalog.pg_namespace pn ON pn.oid=policy.relnamespace '
        "WHERE cn.nspname='public' AND c.relname=ANY(%s) AND t.tgname LIKE %s "
        "AND pn.nspname='public' AND policy.relname='trade_disposition_policy' ORDER BY c.relname,t.tgname",
        (list(TABLES) + ['trades','alpaca_activities','trade_disposition_policy'], 'trade_allocation_%'))
    triggers = rows(cursor, 10)
    require(len(triggers) == 10 and {(item[0],item[1]) for item in triggers} == set(TRIGGER_CONTRACT), 'TRIGGER_ATTACHMENT_MISSING')
    for table, name, function, namespace, bits, enabled, no_when, nargs, arg_bytes, ordinary, attributes, marker_attribute in triggers:
        expected_function, expected_bits, marker_only = TRIGGER_CONTRACT[(table,name)]
        require(function == expected_function and namespace == 'public' and bits == expected_bits
                and enabled == 'O' and no_when and nargs == 0 and arg_bytes == 0 and ordinary
                and attributes == (marker_attribute if marker_only else '')
                and (not marker_only or marker_attribute is not None), 'TRIGGER_ATTACHMENT_CHANGED')

def rows(cursor, maximum):
    result = cursor.fetchmany(maximum + 1)
    require(len(result) <= maximum, 'POPULATION_CENSORED')
    return result

def json_projection(alias, maximum):
    # Bound every transferred full row, rather than only the final receipt.
    return (f"CASE WHEN pg_catalog.octet_length(pg_catalog.to_jsonb({alias})::text)<={maximum} "
            f"THEN pg_catalog.to_jsonb({alias})::text ELSE NULL END")

def validate_current(cursor, manifest, *, require_current_trade=True):
    D = policy_module()
    cursor.execute('LOCK TABLE public.trades,public.alpaca_activities IN ACCESS SHARE MODE')
    cursor.execute('SELECT c.relname,c.relkind,c.relrowsecurity,c.relforcerowsecurity,'
                   'EXISTS(SELECT 1 FROM pg_catalog.pg_inherits i WHERE i.inhrelid=c.oid OR i.inhparent=c.oid),'
                   "pg_catalog.has_table_privilege(c.oid,'SELECT') FROM pg_catalog.pg_class c "
                   'JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace '
                   "WHERE n.nspname='public' AND c.relname IN ('trades','alpaca_activities') ORDER BY c.relname")
    visible = rows(cursor, 2)
    require(len(visible) == 2 and {item[0] for item in visible} == {'trades','alpaca_activities'}
            and all(kind == 'r' and not rls and not forced and not inherited and readable
                    for name, kind, rls, forced, inherited, readable in visible), 'ORIGINAL_VISIBILITY_INVALID')
    cursor.execute('SELECT ' + json_projection('t', 32768) + ' FROM public.trades t '
                   'WHERE t.trade_id=110 OR t.entry_order_id=ANY(%s) OR t.exit_order_id=ANY(%s) '
                   'ORDER BY t.trade_id LIMIT 1002', (list(ORDERS), list(ORDERS)))
    owner_rows = rows(cursor, 1001)
    owners = []
    for row in owner_rows:
        require(row[0] is not None, 'ROW_OVERSIZE')
        owners.append(bounded_document(row[0]))
    target = [item for item in owners if item.get('trade_id') == 110]
    require(len(target) == 1, 'ROW_CHANGED')
    if require_current_trade:
        require(validate_trade(target[0]) == manifest['row_snapshot'], 'ROW_CHANGED')
    for item in owners:
        if item.get('trade_id') != 110:
            require(item.get('entry_order_id') not in ORDERS and item.get('exit_order_id') not in ORDERS, 'GLOBAL_OWNER_CONFLICT')
    expected = validate_fills(manifest['fills'])
    cursor.execute('SELECT ' + json_projection('a', 32768) + ' FROM public.alpaca_activities a '
                   'WHERE a.order_id=ANY(%s) OR a.activity_id=ANY(%s) OR a.activity_pk=ANY(%s) '
                   'ORDER BY a.activity_pk LIMIT 1002', (list(ORDERS), [item[1] for item in FILLS], [item[0] for item in FILLS]))
    actual = []
    for row in rows(cursor, 1001):
        require(row[0] is not None, 'FILL_OVERSIZE')
        snapshot = bounded_document(row[0])
        # Native identity is composite (activity_id,activity_type). Fee and
        # other non-FILL observations may share an opaque ID or reserved order.
        if snapshot.get('activity_type') == 'FILL':
            actual.append({'snapshot': snapshot, 'row_fingerprint': D.fingerprint(snapshot)})
        else:
            require(snapshot.get('activity_pk') not in {item[0] for item in FILLS}, 'FILL_IDENTITY_CONFLICT')
    require(validate_fills(actual) == expected, 'FILL_CHANGED')

def receipt(request):
    return {'allocation_id': request['allocation_id'], 'trade_id': 110,
            'state': 'ACTIVE' if request['action'] == 'APPLY' else 'REVOKED',
            'allocation_revision': request['expected_allocation_revision'] + 1,
            'policy_revision': request['expected_policy_revision'] + 1, 'approval_ref': request['approval_ref']}

def check_claims(cursor, allocation_id):
    cursor.execute('SELECT activity_id,activity_type,order_id,quantity::text,allocation_id::text FROM public.trade_allocation_fill_claims '
                   'WHERE allocation_id=%s OR activity_id=ANY(%s) OR order_id=ANY(%s) ORDER BY activity_id LIMIT 7',
                   (allocation_id, [item[1] for item in FILLS], list(ORDERS)))
    fills = rows(cursor, 5)
    require(len(fills) == 5 and all(owner == allocation_id for a, b, c, q, owner in fills)
            and {(a, b, c, number(q)) for a, b, c, q, owner in fills}
            == {(item[1], 'FILL', item[3], Decimal(item[4])) for item in FILLS}, 'CLAIMS_INVALID')
    cursor.execute('SELECT order_id,allocation_id::text FROM public.trade_allocation_order_claims '
                   'WHERE allocation_id=%s OR order_id=ANY(%s) ORDER BY order_id LIMIT 5', (allocation_id, list(ORDERS)))
    order_rows = rows(cursor, 3)
    require(len(order_rows) == 3 and all(row[1] == allocation_id for row in order_rows)
            and {row[0] for row in order_rows} == set(ORDERS), 'CLAIMS_INVALID')

def validate_history(cursor, allocation_id, manifest, current_request, case_revision, policy_revision, expected_identity):
    D = policy_module()
    cursor.execute('SELECT a.request_digest,CASE WHEN pg_catalog.octet_length(a.request::text)<=8192 THEN a.request::text ELSE NULL END,'
        'a.manifest=c.manifest,CASE WHEN pg_catalog.octet_length(a.result::text)<=8192 THEN a.result::text ELSE NULL END,'
        "a.policy_revision,a.approval_ref=a.request->>'approval_ref' "
        'FROM public.trade_allocation_audit a JOIN public.trade_allocation_cases c ON c.allocation_id=a.allocation_id '
        "WHERE a.allocation_id=%s ORDER BY (a.result->>'allocation_revision')::pg_catalog.int8 LIMIT 33", (allocation_id,))
    history = rows(cursor, MAX_HISTORY)
    require(1 <= case_revision == len(history) <= MAX_HISTORY, 'AUDIT_HISTORY_INCOMPLETE')
    prior_policy = -1
    approvals = set()
    for index, (digest, saved_request, same_manifest, saved_result, audit_policy, same_approval) in enumerate(history, 1):
        require(saved_request is not None and saved_result is not None and same_manifest and same_approval, 'AUDIT_HISTORY_INVALID')
        request, _ = validate_request(saved_request, manifest, expected_identity)
        require(request['allocation_id'] == allocation_id and request['expected_allocation_revision'] == index - 1
                and request['action'] == ('APPLY' if index % 2 else 'REVOKE')
                and request['approval_ref'] not in approvals and digest == D.fingerprint(request)
                and bounded_document(saved_result) == receipt(request)
                and audit_policy == request['expected_policy_revision'] + 1
                and prior_policy < audit_policy <= policy_revision, 'AUDIT_HISTORY_INVALID')
        approvals.add(request['approval_ref'])
        prior_policy = audit_policy
        if index == case_revision:
            require(request == current_request, 'AUDIT_HISTORY_INVALID')

def read_active(cursor, expected_identity, policy_revision, excluded_ids):
    """Under D's shared policy lock; errors never become an empty allocation set."""
    D = policy_module()
    D.identity(*expected_identity)
    D.revision(policy_revision)
    cursor.execute("SELECT pg_catalog.pg_advisory_xact_lock_shared(pg_catalog.hashtext('jbravo:trade-disposition:policy'))")
    cursor.execute('SELECT deployment_id,database_id,database_name,policy_revision,pg_catalog.current_database() '
                   'FROM public.trade_disposition_policy WHERE singleton=1')
    binding = cursor.fetchone()
    require(binding is not None and tuple(binding[:2]) == expected_identity and binding[2] == binding[4]
            and binding[3] == policy_revision, 'POLICY_BINDING_CHANGED')
    if not schema_present(cursor):
        return result_read('MIGRATION_PENDING', (), expected_identity, policy_revision)
    cursor.execute('SELECT c.allocation_id::text,c.state,c.allocation_revision,c.request::text,c.manifest::text,'
                   'a.request_digest,a.request::text,a.result::text,a.policy_revision '
                   'FROM public.trade_allocation_cases c LEFT JOIN public.trade_allocation_audit a '
                   "ON a.approval_ref=c.request->>'approval_ref' WHERE c.state IN ('ACTIVE','REVOKED') ORDER BY c.trade_id LIMIT 17")
    active = []
    reserved = {}
    reservation_cases = []
    seen = set()
    for case_id, state, case_revision, saved_request, saved_manifest, digest, audit_request, audit_result, audit_revision in rows(cursor, 16):
        request, manifest = validate_request(saved_request, saved_manifest, expected_identity)
        require(request['action'] == ('APPLY' if state == 'ACTIVE' else 'REVOKE')
                and request['allocation_id'] == case_id and state in ('ACTIVE','REVOKED')
                and case_revision == request['expected_allocation_revision'] + 1
                and (case_revision % 2 == 1) == (state == 'ACTIVE') and 110 not in seen
                and (state != 'ACTIVE' or 110 not in excluded_ids), 'ACTIVE_CASE_INVALID')
        require(digest == D.fingerprint(request) and bounded_document(audit_request) == request
                and bounded_document(audit_result) == receipt(request)
                and audit_revision == request['expected_policy_revision'] + 1
                and audit_revision <= policy_revision, 'AUDIT_INVALID')
        validate_history(cursor, case_id, manifest, request, case_revision, policy_revision, expected_identity)
        # Revoke removes the operational snapshot/write overlay. The original
        # approved snapshot remains immutable audit evidence for reserved owners;
        # subsequent legitimate owner110 scalar changes do not transfer claims.
        validate_current(cursor, manifest, require_current_trade=(state == 'ACTIVE'))
        check_claims(cursor, case_id)
        seen.add(110)
        for order in ORDERS:
            require(order not in reserved, 'RESERVED_OWNER_CONFLICT')
            reserved[order] = 110
        reservation_cases.append({'allocation_id':case_id,'trade_id':110,'state':state,
                                  'allocation_revision':case_revision,'approval_ref':request['approval_ref']})
        if state == 'ACTIVE':
            active.append(Allocation(case_id, 110, state, case_revision, Decimal(30), Decimal(30), Decimal('-63.38'),
                                     request['approval_ref'], tuple(item[1] for item in FILLS),
                                     tuple(ORDERS), tuple(item[1] for item in FILLS)))
    return result_read('VALIDATED', active, expected_identity, policy_revision, reserved, reservation_cases)

def change_allocation(conn, request, manifest, identity):
    """Own one idle transaction. Invocation needs separate exact activation authority.

    Implementation/testing approval is not production activation authority.
    Original trades/events/activities are only read and locked, never updated.
    """
    D = policy_module()
    request, manifest = validate_request(request, manifest, identity)
    require(conn is not None and not conn.autocommit and conn.get_transaction_status() == 0, 'TRANSACTION_NOT_IDLE')
    digest = D.fingerprint(request)
    with conn:
        with conn.cursor() as cursor:
            cursor.execute('SET TRANSACTION ISOLATION LEVEL READ COMMITTED')
            policy = D.read_policy(cursor, identity, exclusive=True)
            require(schema_present(cursor), 'MIGRATION_PENDING')
            cursor.execute('SELECT approval_ref FROM public.trade_disposition_audit WHERE approval_ref=%s', (request['approval_ref'],))
            require(cursor.fetchone() is None, 'APPROVAL_REUSED')
            cursor.execute('SELECT request_digest,result::text FROM public.trade_allocation_audit WHERE approval_ref=%s', (request['approval_ref'],))
            previous = cursor.fetchone()
            if previous:
                require(previous[0] == digest and bounded_document(previous[1]) == receipt(request), 'APPROVAL_REUSED')
                cursor.execute('SELECT state,allocation_revision FROM public.trade_allocation_cases WHERE allocation_id=%s',
                               (request['allocation_id'],))
                current = cursor.fetchone()
                require(current == (receipt(request)['state'], receipt(request)['allocation_revision']), 'REPLAY_SUPERSEDED')
                validate_current(cursor, manifest, require_current_trade=(request['action'] == 'APPLY'))
                check_claims(cursor, request['allocation_id'])
                return receipt(request)
            require(policy.revision == request['expected_policy_revision'], 'POLICY_CAS_FAILED')
            # Original writers take the attested shared advisory lock before any
            # row work. Our exclusive counterpart serializes them; ACCESS SHARE
            # in validate_current needs no original-table write privilege.
            validate_current(cursor, manifest)
            cursor.execute('SELECT state,allocation_revision,manifest::text FROM public.trade_allocation_cases '
                           'WHERE allocation_id=%s FOR UPDATE', (request['allocation_id'],))
            case = cursor.fetchone()
            expected_revision = request['expected_allocation_revision']
            genuine_exclusions = {item.trade_id for item in policy.active}
            require(110 not in genuine_exclusions, 'TRADE_EXCLUDED')
            if request['action'] == 'APPLY':
                require((case is None and expected_revision == 0)
                        or (case is not None and case[0] == 'REVOKED' and case[1] == expected_revision
                            and bounded_document(case[2]) == manifest), 'CASE_CAS_FAILED')
                cursor.execute('SELECT allocation_id::text FROM public.trade_allocation_cases WHERE trade_id=110 AND state=%s', ('ACTIVE',))
                require(not rows(cursor, 1), 'CASE_CONFLICT')
                if case is None:
                    cursor.execute('INSERT INTO public.trade_allocation_cases '
                        '(allocation_id,trade_id,state,allocation_revision,request,manifest) VALUES (%s,110,%s,1,%s::pg_catalog.jsonb,%s::pg_catalog.jsonb)',
                        (request['allocation_id'], 'ACTIVE', D.canonical(request), D.canonical(manifest)))
                    for item in FILLS:
                        cursor.execute('INSERT INTO public.trade_allocation_fill_claims '
                            '(activity_id,activity_type,order_id,allocation_id,quantity) VALUES (%s,%s,%s,%s,%s)',
                            (item[1], 'FILL', item[3], request['allocation_id'], item[4]))
                    for order in ORDERS:
                        cursor.execute('INSERT INTO public.trade_allocation_order_claims (order_id,allocation_id) VALUES (%s,%s)',
                                       (order, request['allocation_id']))
                else:
                    check_claims(cursor, request['allocation_id'])
                    cursor.execute('UPDATE public.trade_allocation_cases SET state=%s,allocation_revision=allocation_revision+1,'
                        'request=%s::pg_catalog.jsonb WHERE allocation_id=%s AND state=%s AND allocation_revision=%s',
                        ('ACTIVE', D.canonical(request), request['allocation_id'], 'REVOKED', expected_revision))
                    require(cursor.rowcount == 1, 'CASE_CAS_FAILED')
            else:
                require(case is not None and case[0] == 'ACTIVE' and case[1] == expected_revision
                        and bounded_document(case[2]) == manifest, 'CASE_CAS_FAILED')
                check_claims(cursor, request['allocation_id'])
                cursor.execute('UPDATE public.trade_allocation_cases SET state=%s,allocation_revision=allocation_revision+1,'
                    'request=%s::pg_catalog.jsonb WHERE allocation_id=%s AND state=%s AND allocation_revision=%s',
                    ('REVOKED', D.canonical(request), request['allocation_id'], 'ACTIVE', expected_revision))
                require(cursor.rowcount == 1, 'CASE_CAS_FAILED')
            cursor.execute('UPDATE public.trade_disposition_policy SET policy_revision=policy_revision+1 WHERE singleton=1 AND policy_revision=%s', (policy.revision,))
            require(cursor.rowcount == 1, 'POLICY_CAS_FAILED')
            result = receipt(request)
            cursor.execute('INSERT INTO public.trade_allocation_audit '
                '(approval_ref,allocation_id,request_digest,request,manifest,result,policy_revision) '
                'VALUES (%s,%s,%s,%s::pg_catalog.jsonb,%s::pg_catalog.jsonb,%s::pg_catalog.jsonb,%s)',
                (request['approval_ref'], request['allocation_id'], digest, D.canonical(request), D.canonical(manifest),
                 D.canonical(result), policy.revision + 1))
            return result
