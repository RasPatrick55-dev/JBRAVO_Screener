"""Computational completion never admits a new entry or qualifies research."""
from datetime import datetime,timezone
from unittest import mock
import unittest

from scripts import pipeline_postflight as post, executor_health as executor
from tests.test_pipeline_postflight import PostflightTests
from tests.test_signal_handoff import BoundHandoffTests
from scripts import primary_pipeline as primary, signal_handoff as signals
import contextlib
import io

class ReadinessOnlyTests(PostflightTests):
    def health(self,**changes):
        return {'decision':'warn','reasons':['action_investigate','stale_monitor_input_dataset_end','stale_monitor_input_recent_end'],
                'monitor_contract_version':2,'model_comparison_scope':'cross_model_diagnostic','production_drift_established':False,**changes}

    def test_research_diagnostics_do_not_block_pipeline_completion(self):
        record=self.publish(ml_health=self.health(),readiness_policy=post.PIPELINE_ONLY)
        result=self.check()
        self.assertEqual(result['status'],'ok');self.assertTrue(result['pipeline_ready'])
        self.assertFalse(result['entry_ready']);self.assertFalse(result['research_qualified'])
        self.assertEqual(result['research_status'],'pending')
        self.assertEqual(record['ml_health']['reasons'],self.health()['reasons'])
        self.assertFalse(result['operating_acceptance'])

    def test_completion_policy_cannot_be_used_by_executor_even_without_session(self):
        self.publish(ml_health=self.health(),readiness_policy=post.PIPELINE_ONLY)
        for require_session in (False,True):
            result=executor.preflight(self.root,now=self.now,require_session=require_session)
            self.assertEqual(result['status'],'blocked')
        self.assertEqual(self.check(require_completion=True)['status'],'failed')
        with mock.patch.object(post,'check_health',return_value={'status':'ok','readiness_policy':post.PIPELINE_ONLY,'entry_ready':True}):
            self.assertEqual(executor.preflight(self.root,now=self.now)['reasons'],['pipeline_new_entries_not_authorized'])

    def test_healthy_monitor_does_not_enable_entries_in_pipeline_only_mode(self):
        self.publish(readiness_policy=post.PIPELINE_ONLY)
        self.assertEqual(self.check()['status'],'ok')
        self.assertFalse(self.check()['entry_ready'])
        self.assertEqual(executor.preflight(self.root,now=self.now,require_session=False)['status'],'blocked')

    def test_unknown_scope_missing_monitor_and_real_alerts_still_block(self):
        for health in (None,self.health(decision='block'),self.health(monitor_contract_version=True),
                       self.health(model_comparison_scope='reviewed_fixed_production_model'),
                       self.health(production_drift_established=True),self.health(reasons=['stale_predictions']),
                       self.health(reasons=['action_investigate','unknown_ml_health_reason']),
                       self.health(reasons=['action_retrain']),self.health(reasons=None)):
            with self.subTest(health=health):
                self.publish(ml_health=health,readiness_policy=post.PIPELINE_ONLY)
                self.assertNotEqual(self.check()['status'],'ok')

    def test_operational_failures_are_still_blocking(self):
        options={'ml_health':self.health(),'readiness_policy':post.PIPELINE_ONLY}
        for change in ({'step_rcs':self.facts['step_rcs']|{'screener':1}}, {'labels_rows':0},
                       {'freshness':self.facts['freshness']|{'stale':True}},
                       {'freshness':self.facts['freshness']|{'pred_compatible':False}},
                       {'coverage':{'total':6,'non_null':3,'pct':50.,'run_ts_utc':self.started.isoformat()}},
                       {'controls':self.facts['controls']|{'strict_predictions_meta':False}}):
            with self.subTest(change=change):
                self.publish(**(options|change));self.assertNotEqual(self.check()['status'],'ok')
        self.publish(**options)
        (self.root/'scripts/run_pipeline.py').write_bytes(b'changed')
        self.assertEqual(self.check()['status'],'failed')

    def test_source_guard_preserves_original_research_verdict(self):
        from tests.test_ml_health_provenance import ProvenanceTests
        fixture=ProvenanceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        _extract_health=fixture.guard._extract_health
        decide_ml_enrichment=fixture.guard.decide_ml_enrichment
        payload={'recommended_action':'investigate','monitor_contract_version':2,
                 'model_comparison':{'scope':'cross_model_diagnostic','production_drift_established':False}}
        extracted=_extract_health(payload)
        result=decide_ml_enrichment({'present':True,**extracted},mode='warn',max_age_days=7,pipeline_run_date=self.now.date(),now=self.now)
        self.assertEqual(result['recommended_action'],'investigate')
        self.assertIn('action_investigate',result['reasons'])
        self.assertEqual(result['monitor_contract_version'],2)
        self.assertEqual(result['model_comparison_scope'],'cross_model_diagnostic')

class PipelineOnlyCompletionTests(BoundHandoffTests):
    def test_real_primary_completes_but_bound_entry_gate_stays_blocked(self):
        started=self.started
        class Clock(datetime):
            @classmethod
            def now(cls,tz=None):return started
        calls=[]
        def run(arguments):
            calls.append(arguments)
            self.assertIn('--pipeline-readiness-only',arguments)
            invocation=arguments[arguments.index('--handoff-invocation')+1]
            rows=[row|{'run_ts_utc':started.isoformat()} for row in self.rows]
            self.publish(finished_at=started,coverage=self.facts['coverage']|{'run_ts_utc':started.isoformat()},
                         frozen_candidates=signals.freeze_candidates(rows,self.session,run_ts=started),invocation_id=invocation,
                         readiness_policy=post.PIPELINE_ONLY,
                         ml_health={'decision':'warn','reasons':['action_investigate','stale_monitor_input_dataset_end'],
                                    'monitor_contract_version':2,'model_comparison_scope':'cross_model_diagnostic',
                                    'production_drift_established':False})
            return 0
        actual_check=post.check_health
        with mock.patch.object(primary,'datetime',Clock),contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(post,'check_health',side_effect=lambda *a,**k:actual_check(*a,now=started,**k)):
            self.assertEqual(primary.main([],pipeline_main=run,base_dir=self.root),0)
        self.assertEqual(len(calls),1)
        self.assertTrue((self.root/'reports/pipeline_postflight/completed-primary.json').is_file())
        health=actual_check(self.root,now=self.now,require_session=True)
        self.assertTrue(health['pipeline_ready']);self.assertFalse(health['entry_ready']);self.assertFalse(health['research_qualified'])
        self.assertEqual(self.gate()['status'],'blocked')
        health=actual_check(self.root,now=self.now,require_session=True,require_completion=True)
        self.assertIn('new_entries_blocked_by_pipeline_only_policy',health['failures'])

    def test_primary_rejects_pipeline_policy_override(self):
        runner=mock.Mock(return_value=0)
        for argv in (['--pipeline-readiness-only'],['--pipeline-readiness'],['--pipeline-readiness-only=false']):
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(primary.main(argv,pipeline_main=runner),1)
        runner.assert_not_called()

if __name__=='__main__':unittest.main()
