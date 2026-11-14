"""Fixed GTLB131 whole-position proof. No connection, migration or activation on import."""
from decimal import Decimal, Context, localcontext
from . import owned_exit_repair as O

CLASSIFICATION = 'WHOLE_POSITION_MULTI_ORDER_V1'
CASE = 'GTLB131'
ROW_PIN = '4f204c9d45bb5ee9f8acbf201acf216d310d2676fbc600d567ce90b2bb879254'
ENTRY = 'c81944d6-2d33-45e0-82dc-d39897488662'
FIRST_EXIT = 'af466a7d-735f-4d7f-b182-c21454ffd2de'
TERMINAL_EXIT = 'f9df8361-5aee-43a4-84dd-7b2edbf6f599'
ORDERS = (ENTRY, FIRST_EXIT, TERMINAL_EXIT)
FILL_PINS = {
 '20260521070004735::6da0781e-1b17-455e-b3f7-b6929b201997': '5d34b2ef60dac5aeeb218fb1d3b1b6ca2686c1b42924c1a60f02b4d76e1c0e3f',
 '20260521101102081::42410f5e-9573-41cf-8ef3-b028d36dc06f': '37da85e902f397eaec7b139c48dbae61960c1ac7465cb50c66f57163713e6cb6',
 '20260526094609267::1a472940-dc4d-45b2-a357-9e3e64116158': '3af05718cbdaa990dbf764123b5cb8d34255d8c9c0fec8079aa316e64c6545e1',
 '20260526094616712::e4e11237-57e5-43a8-9184-db13c40bd7c4': 'bf6d63f92eb5a7cdc8ce478c24db281afc4074acb8935830b923ca06cb5cbf08',
 '20260526094718374::516d1a8d-1110-4b0f-9dba-87110a0771fa': 'b19323d6c271fb7039229b3bf78c5f243b6362fad01c62ec1f51f3d0cb17bcb5',
}


def validate_group(manifest, identity):
    with localcontext(Context(prec=160)):
        return _validate_group(manifest, identity)


def _validate_group(m, identity):
    D = O.policy_module()
    O.safe_private(m)
    O.need(type(m) is dict and set(m) == {'classification','case','trade_id','allocation_id',
        'identity','group_approval_ref','preview_digest','row_snapshot','fills','orders',
        'broker_proof','source_pins','financial_eligible','fees','net'}, 'MULTI_MANIFEST_SHAPE')
    O.need(m['classification'] == CLASSIFICATION and m['case'] == CASE and type(m['trade_id']) is int
        and m['trade_id'] == 131 and m['identity'] == list(identity), 'MULTI_SCOPE')
    D.identity(*identity)
    O.identifier(m['allocation_id']); D.digest(m['preview_digest']); D.text_field(m['group_approval_ref'])
    O.need(m['financial_eligible'] is False and m['fees'] is None and m['net'] is None, 'MULTI_FINANCIAL_SCOPE')
    O.need(len(D.canonical(m).encode()) <= O.MANIFEST_MAX, 'RUNTIME_MANIFEST_BOUND')
    row = O.snapshot(m['row_snapshot'])['row']
    O.need(D.fingerprint(row) == ROW_PIN, 'MULTI_ROW_CHANGED')
    O.need(m['orders'] == list(ORDERS), 'MULTI_ORDERS')
    O.need(type(m['source_pins']) is dict and 1 <= len(m['source_pins']) <= 32, 'SOURCE_PINS')
    for path, digest in m['source_pins'].items():
        D.text_field(path); D.digest(digest)
    fills = O.manifest_fills(m)
    O.need(len(fills) == 5, 'MULTI_FILL_COUNT')
    seen = set(); facts = []
    for item in fills:
        f = O.fill_fact(item); key = f['activity_id']
        O.need(key not in seen and key in FILL_PINS and D.fingerprint(item['row']) == FILL_PINS[key], 'MULTI_FILL_CHANGED')
        seen.add(key); facts.append(f)
    O.need(seen == set(FILL_PINS), 'MULTI_FILL_SCOPE')
    O.need(O._sum(facts,'buy') == O._sum(facts,'sell') == 42
        and O._value(facts,'buy') == Decimal('1118.88')
        and O._value(facts,'sell') == Decimal('1098.19'), 'MULTI_CONSERVATION')
    validate_broker(m['broker_proof'], facts)
    O.need(m['source_pins'].get(m['broker_proof']['evidence_reference']) == m['broker_proof']['evidence_sha256'], 'MULTI_BROKER_EVIDENCE_PIN')
    return dict(orders=list(ORDERS),activity_ids=sorted(seen),quantity=Decimal(42),gross=Decimal('-20.69'),
        clock_fills=tuple((s['row']['activity_pk'],s['row']['activity_id'],s['row']['side'],s['row']['order_id'],
            str(O.number(s['row']['qty'])),str(O.number(s['row']['price'])),s['row']['transaction_time']) for s in fills))


def validate_broker(proof, facts, *, now=None):
    C = O.clocks()
    O.need(type(proof) is dict and set(proof) == {'account_identity_redacted','method','observed_at','orders','evidence_reference','evidence_sha256','fill_identity_method'}, 'MULTI_BROKER_SHAPE')
    O.need(proof['account_identity_redacted'] is True and proof['method'] == 'EXACT_HISTORICAL_ORDER_UUIDS_AND_COMPLETE_FILLS', 'MULTI_BROKER_METHOD')
    D=O.policy_module();D.text_field(proof['evidence_reference']);D.digest(proof['evidence_sha256'])
    O.need(proof['fill_identity_method']=='UNIQUE_FACT_TUPLE_MATCH_TO_STORED_NATIVE_ID','MULTI_FILL_IDENTITY_METHOD')
    observed = C.precise_instant(proof['observed_at']).nanoseconds
    if now is not None:
        O.need(0 <= C.precise_instant(now).nanoseconds-observed <= 600*1000000000, 'BROKER_PROOF_AGE')
    O.need(type(proof['orders']) is list and len(proof['orders']) == 3, 'MULTI_BROKER_SCOPE')
    expected = {ENTRY:('buy','filled',42,42),FIRST_EXIT:('sell','expired',42,13),TERMINAL_EXIT:('sell','filled',29,29)}
    seen = set()
    for order in proof['orders']:
        O.need(type(order) is dict and set(order) == {'order_id','symbol','side','status','qty','filled_qty','fills','complete'}, 'MULTI_BROKER_ORDER')
        oid = order['order_id']; O.need(oid in expected and oid not in seen, 'MULTI_BROKER_ORDER')
        side,status,qty,filled = expected[oid]
        O.need(order['symbol']=='GTLB' and order['side']==side and order['status']==status and order['complete'] is True
            and O.number(order['qty'])==qty and O.number(order['filled_qty'])==filled, 'MULTI_BROKER_FACT')
        expected_fills = {f['activity_id']:f for f in facts if f['order_id']==oid}
        O.need(type(order['fills']) is list and len(order['fills'])==len(expected_fills), 'MULTI_BROKER_FILLS')
        ids=set()
        for f in order['fills']:
            O.need(type(f) is dict and set(f)=={'id','order_id','symbol','side','qty','price','transaction_time','activity_type'}, 'MULTI_BROKER_FILL')
            key=f['id']; O.need(key in expected_fills and key not in ids, 'MULTI_BROKER_FILL')
            e=expected_fills[key]
            O.need(f['activity_type']=='FILL' and f['order_id']==oid and f['symbol']=='GTLB' and f['side']==side
                and O.number(f['qty'])==e['qty'] and O.number(f['price'])==e['price']
                and C.precise_instant(f['transaction_time']).nanoseconds==e['instant'], 'MULTI_BROKER_FILL')
            ids.add(key)
        seen.add(oid)


def read_extensions(cursor,records,identity,policy_revision,excluded_ids,*,allocation_type):
    from . import owned_exit_transaction as X
    O.need(len(records)<=1,'MULTI_ALLOCATION_SCOPE')
    def attest(current):
        O.need(X.attest_extension(current)==X.MULTI_FUNCTION_SHA,'MULTI_GUARD_REQUIRED')
    return O._read_extensions(cursor,records,identity,policy_revision,excluded_ids,
        allocation_type=allocation_type,attest=attest,validator=validate_group)
