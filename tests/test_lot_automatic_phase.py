from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch
from test_lot_incremental_preview import inputs
from scripts import strategy_lot_automatic_phase as phase

class AutomaticPhase(TestCase):
    def setUp(self):
        self.i=inputs()
        self.i['origin_receipt_sha256s']={oid:'a'*64 for oid in self.i['origins']}
        self.c=MagicMock(readonly=True)
        self.c.get_transaction_status.return_value=0
        self.kw=dict(account_id=self.i['account_id'],connect=Mock(return_value=self.c),
                     load_inputs=Mock(return_value=self.i),policy_lock=Mock(),
                     journal_directory='unused-fixture')

    def invoke(self,result):
        with patch.object(phase,'reconcile_pending',return_value=None), \
             patch.object(phase.lot,'apply_guarded',return_value=result) as writer:
            output=phase.step(**self.kw)
            return output,writer

    def test_unresolved_never_success(self):
        output,writer=self.invoke(dict(status='ABSENT_UNRESOLVED'))
        self.assertEqual(output['status'],'AUTO_LOT_UNRESOLVED')
        self.assertFalse(output['retry_authorized'])
        writer.assert_called_once()

    def test_wrong_digest_never_success(self):
        output,_=self.invoke(dict(status='COMMITTED_VERIFIED',run_sha256='0'*64))
        self.assertEqual(output['status'],'AUTO_LOT_UNRESOLVED')

    def test_exact_verified_success_and_real_recheck(self):
        expected=phase.checked_preview(self.i)['candidates'][0]['packet']
        output,writer=self.invoke(dict(status='COMMITTED_VERIFIED',run_sha256=expected['run_sha256']))
        self.assertEqual(output['status'],'AUTO_LOT_COMMITTED_VERIFIED')
        self.assertEqual(writer.call_args.kwargs['recheck'](Mock()),expected)
        self.assertFalse(output['accounting_complete'])
        self.i['orders'][0]['observation']['status']='canceled'
        with self.assertRaisesRegex(ValueError,'AUTO_COHORT_CHANGED'):
            writer.call_args.kwargs['recheck'](Mock())

    def test_missing_origin_receipt_stops_before_writer(self):
        self.i['origin_receipt_sha256s']={}
        with patch.object(phase,'reconcile_pending',return_value=None), \
             patch.object(phase.lot,'apply_guarded') as writer:
            with self.assertRaisesRegex(ValueError,'DURABLE_ORIGIN_RECEIPTS_REQUIRED'):
                phase.step(**self.kw)
            writer.assert_not_called()

    def test_noop_does_not_write(self):
        self.i['prior_events']=phase.checked_preview(self.i)['candidates'][0]['packet']['events']
        output,writer=self.invoke({})
        self.assertEqual(output['status'],'NO_ELIGIBLE_RESIDUAL_ALLOCATION')
        writer.assert_not_called()

    def test_pending_recovery_precedes_loader_no_write(self):
        pending=dict(status='PRIOR_LOT_RECONCILIATION_ONLY',retry_authorized=False)
        with patch.object(phase,'reconcile_pending',return_value=pending), \
             patch.object(phase.lot,'apply_guarded') as writer:
            self.assertEqual(phase.step(**self.kw),pending)
            self.kw['load_inputs'].assert_not_called()
            writer.assert_not_called()

import pytest
pytestmark = pytest.mark.alpaca_optional
