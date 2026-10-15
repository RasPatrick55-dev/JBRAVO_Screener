"""Paper eligibility keeps real completion/candidate/session guards intact."""
import copy
from datetime import timedelta
from scripts import executor_health as health, pipeline_postflight as post
from tests.test_signal_handoff import BoundHandoffTests


class PaperResearchTests(BoundHandoffTests):
    def prepare_paper(self, **changes):
        self.publish(readiness_policy=post.PIPELINE_ONLY, ml_health={
            'decision':'warn','reasons':['action_investigate','stale_monitor_input_dataset_end'],
            'monitor_contract_version':2,'model_comparison_scope':'cross_model_diagnostic',
            'production_drift_established':False}, **changes)
        self.complete()

    def paper_gate(self, **changes):
        return self.gate(**(dict(paper_research_entries=True,
                                 paper_base_url='https://paper-api.alpaca.markets') | changes))

    def test_explicit_paper_mode_passes_without_relabeling_research(self):
        self.prepare_paper()
        self.assertEqual(self.gate()['status'],'blocked')
        result=self.paper_gate()
        self.assertEqual(result['status'],'pass')
        self.assertEqual(result['entry_authorization'],'paper_research_test')
        self.assertFalse(result['pipeline_entry_ready'])
        self.assertFalse(result['research_qualified'])
        self.assertFalse(result['operating_acceptance'])
        self.assertIn('action_investigate',result['research_reasons'])
        self.assertFalse(post.check_health(self.root,now=self.now)['entry_ready'])

    def test_live_missing_and_lookalike_endpoints_block(self):
        self.prepare_paper()
        for endpoint in (None,'','https://api.alpaca.markets','http://paper-api.alpaca.markets',
                         'https://paper-api.alpaca.markets.evil.test',
                         'https://paper-api.alpaca.markets@evil.test',
                         'https://evil.test/paper-api.alpaca.markets',
                         'https://user@paper-api.alpaca.markets',
                         'https://paper-api.alpaca.markets:444',
                         'https://paper-api.alpaca.markets?redirect=live'):
            with self.subTest(endpoint=endpoint):
                self.assertEqual(self.paper_gate(paper_base_url=endpoint)['status'],'blocked')

    def test_paper_mode_requires_real_completed_bound_session(self):
        self.prepare_paper()
        self.assertEqual(self.paper_gate(require_session=False)['status'],'blocked')
        (self.root/'reports/pipeline_postflight/completed-primary.json').write_text('{}')
        self.assertEqual(self.paper_gate()['status'],'blocked')

    def test_paper_mode_rejects_changed_candidates_and_stale_session(self):
        self.prepare_paper()
        altered=copy.deepcopy(self.rows);altered[0]['close']=999
        self.assertEqual(self.paper_gate(rows=altered)['status'],'blocked')
        self.assertEqual(self.paper_gate(now=self.now+timedelta(days=1))['status'],'blocked')
        (self.root/'scripts/executor_health.py').write_text('changed')
        self.assertEqual(self.paper_gate()['status'],'blocked')

    def test_other_pipeline_failures_cannot_be_waived(self):
        self.prepare_paper()
        for changes in ({'pipeline_rc':1},{'labels_rows':0},
                        {'freshness':self.facts['freshness']|{'stale':True}},
                        {'controls':self.facts['controls']|{'ml_health_guard':False}}):
            with self.subTest(changes=changes):
                self.publish(readiness_policy=post.PIPELINE_ONLY,**changes)
                self.assertEqual(self.paper_gate()['status'],'blocked')

    def test_unknown_and_production_health_alerts_stay_blocking(self):
        self.prepare_paper()
        for ml in (None,{'decision':'block','reasons':['action_investigate']},
                   {'decision':'warn','reasons':['action_investigate'],
                    'monitor_contract_version':2,'model_comparison_scope':'reviewed_fixed_production_model',
                    'production_drift_established':True}):
            with self.subTest(health=ml):
                self.publish(readiness_policy=post.PIPELINE_ONLY,ml_health=ml)
                self.assertEqual(self.paper_gate()['status'],'blocked')

