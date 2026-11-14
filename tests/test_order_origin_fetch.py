import hashlib
import json
from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import Mock, patch
from test_lot_incremental_preview import inputs
import test_lot_hourly_phase as handoff
from scripts.strategy_order_origin_fetch import fetch_origins


class OriginFetch(TestCase):
    def setUp(self):
        fingerprint=patch("scripts.activity_ingestion_checkpoint.recipe_fingerprint",return_value="c"*64)
        fingerprint.start(); self.addCleanup(fingerprint.stop)
        self.i = inputs()
        h = handoff.Handoff(); h.setUp()
        link = json.loads(h.receipts['linkage_result.json'])
        link['linkage']['plan_sha256'] = self.i['linked']['plan_sha256']
        h.receipts['linkage_result.json'] = json.dumps(link).encode()
        h.c['native_receipts'] = {n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
                                 for n,b in h.receipts.items()}
        raw = json.dumps(h.c).encode()
        self.responses = [dict(id=self.i['account_id'])] + list(self.i['origins'].values())
        self.get = Mock(side_effect=lambda *a,**kw: Mock(status_code=200,
             headers={'Date':'Thu, 05 Nov 2026 10:00:00 GMT'},
             json=lambda:self.responses.pop(0)))
        self.kw = dict(account_id=self.i['account_id'],orders=self.i['orders'],
             linked=self.i['linked'],completion_bytes=raw,
             expected_completion_sha256=hashlib.sha256(raw).hexdigest(),
             receipt_bytes=h.receipts,expected_slot=h.c['slot'],request_get=self.get,
             headers={'fixture':'not-secret'},
             now=lambda:datetime(2026,11,5,10,tzinfo=timezone.utc),monotonic=lambda:10)

    def test_exact_get_receipts_bind_sync_and_clock(self):
        r=fetch_origins(**self.kw)
        self.assertEqual(len(r['receipts']),2)
        self.assertEqual(r['receipts'][0]['source_run_id'],'a'*32)
        self.assertEqual(len(r['receipts'][0]['sync_receipts']),8)
        self.assertEqual(r['held'],[])
        for call in self.get.call_args_list:
            self.assertFalse(call.kwargs['allow_redirects'])
            self.assertEqual(call.kwargs['timeout'],30)

    def test_snapshot_changed_is_held_not_imported(self):
        self.responses[1]['status']='canceled'
        r=fetch_origins(**self.kw)
        self.assertEqual(len(r['receipts']),1)
        self.assertEqual(r['held'][0]['reason'],'ORIGIN_SNAPSHOT_CHANGED_SINCE_SYNC')

    def test_wrong_account_stops_before_order_fetch(self):
        self.responses[0]['id']='wrong'
        with self.assertRaisesRegex(ValueError,'ORIGIN_ACCOUNT_MISMATCH'):
            fetch_origins(**self.kw)
        self.assertEqual(self.get.call_count,1)

    def test_bad_sync_never_calls_broker(self):
        self.kw['expected_completion_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SYNC_COMPLETION_PIN'):
            fetch_origins(**self.kw)
        self.get.assert_not_called()

    def test_clock_skew_holds(self):
        self.kw['now']=lambda:datetime(2026,11,6,10,tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError,'ORIGIN_SERVER_CLOCK_MISMATCH'):
            fetch_origins(**self.kw)

    def test_missing_created_at_holds(self):
        self.responses[1].pop('created_at')
        r=fetch_origins(**self.kw)
        self.assertEqual(r['held'][0]['reason'],'IMMUTABLE_ORDER_ORIGIN_REQUIRED')


    def test_bounded_full_history_attached_without_checkpoint_fabrication(self):
        from scripts.strategy_order_origin_fetch import attach_bounded_history
        from scripts import strategy_order_origin_evidence as evidence
        receipts=fetch_origins(**self.kw)['receipts']
        fetch=Mock(return_value=self.i['checkpoint_rows'])
        result=attach_bounded_history(receipts,fetch_activities=fetch,
            normalize=lambda row:(row,None),now=self.kw['now'])
        h=result[0]['bounded_history']
        self.assertIsNone(h['checkpoint_id'])
        self.assertEqual(h['fetched_count'],2)
        self.assertEqual(h,result[1]['bounded_history'])
        for p in result:evidence.validate_fetch_receipt(p)
        self.assertEqual(fetch.call_args.args[-2:],(100,1000))

    def test_bounded_history_fetch_failure_never_full(self):
        from scripts.strategy_order_origin_fetch import attach_bounded_history
        receipts=fetch_origins(**self.kw)['receipts']
        with self.assertRaisesRegex(ValueError,'PAGE_CAP'):
            attach_bounded_history(receipts,fetch_activities=Mock(side_effect=ValueError('PAGE_CAP')),
                normalize=lambda row:(row,None),now=self.kw['now'])

    def test_bounded_history_duplicate_rejected(self):
        from scripts.strategy_order_origin_fetch import attach_bounded_history
        receipts=fetch_origins(**self.kw)['receipts']
        with self.assertRaisesRegex(RuntimeError,'DUPLICATE_IDENTITY'):
            attach_bounded_history(receipts,fetch_activities=Mock(return_value=[self.i['checkpoint_rows'][0]]*2),
                normalize=lambda row:(row,None),now=self.kw['now'])

    def test_bounded_history_45day_cap_before_fetch(self):
        from scripts.strategy_order_origin_fetch import attach_bounded_history
        receipts=fetch_origins(**self.kw)['receipts']
        fetch=Mock()
        with self.assertRaisesRegex(ValueError,'HISTORY_WINDOW_LIMIT'):
            attach_bounded_history(receipts,fetch_activities=fetch,normalize=lambda row:(row,None),
                now=lambda:datetime(2027,1,1,tzinfo=timezone.utc))
        fetch.assert_not_called()

    def test_old_cohort_cap_does_not_block_new_cohort(self):
        from copy import deepcopy
        from scripts.strategy_order_origin_fetch import attach_history_by_cohort
        from scripts import strategy_fill_linkage as core
        receipts=fetch_origins(**self.kw)['receipts']
        old=deepcopy(receipts);orders=deepcopy(self.i['orders']);signals=deepcopy(self.i['signals'])
        old_signal=deepcopy(signals[0]);old_signal['signal_id']='old-signal';signals.append(old_signal)
        for n,p in enumerate(old):
            oid=('66666666-6666-4666-8666-666666666666' if n==0 else '77777777-7777-4777-8777-777777777777')
            p['broker_order_id']=oid;p['raw_order']['id']=oid;p['endpoint']='/v2/orders/'+oid
            p['raw_order']['created_at']='2026-08-01T00:00:00Z'
            p['receipt_sha256']=core.digest({k:v for k,v in p.items() if k!='receipt_sha256'})
            claim=deepcopy(orders[n]);claim['signal_id']='old-signal';claim['observation']['id']=oid
            orders.append(claim)
        fetch=Mock(return_value=self.i['checkpoint_rows'])
        result=attach_history_by_cohort(receipts+old,orders=orders,signals=signals,
            fetch_activities=fetch,normalize=lambda x:(x,None),now=self.kw['now'])
        self.assertEqual(len(result['receipts']),2)
        self.assertEqual(result['held'],[dict(signal_id='old-signal',reason='HISTORY_WINDOW_LIMIT')])
        fetch.assert_called_once()

import pytest
pytestmark = pytest.mark.alpaca_optional
