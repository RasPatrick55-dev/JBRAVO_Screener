"""Review-only shared pricing handoff. No I/O or installation on import.

A new schema persists original ranking evidence and its executable projection in
one immutable record. All readers reconstruct both; no V1 price relabeling.
"""
from copy import deepcopy
from isolated_pricing_integration import IsolatedPricing, need

SCHEMA='jbravo.shared-priced-selection.v2'

class SharedPricingContract:
    def __init__(self, isolated):
        self.adapter=isolated
        self.rules=isolated.rules
        self.native=isolated.selection

    def compose(self, ranking, pricing):
        review=pricing['review']
        executable=review['pricing']['executable_signal_ids']
        selected=ranking['selection']['selected_setups']
        need(len(set(executable))==len(executable),'shared_duplicate_executable')
        projected=deepcopy(ranking['selection'])
        projected['selected_setups']=[deepcopy(s) for s in selected if s['proposal']['signal_id'] in executable]
        projected['selected_signal_ids']=[s['proposal']['signal_id'] for s in projected['selected_setups']]
        projected['ranked_selected_setups']=deepcopy(selected)
        need(projected['selected_signal_ids']==executable,'shared_projection_order_changed')
        return dict(schema=SCHEMA,status='COMPLETED_UNIFIED_SETUP_SELECTION',
            run_id=ranking['run_id'],primary_run_id=ranking['primary_run_id'],
            parent_report_sha256=ranking['parent_report_sha256'],
            source_bindings=deepcopy(ranking['source_bindings']),session=deepcopy(ranking['session']),
            specification=dict(policy=SCHEMA,ranking_specification=deepcopy(ranking['specification']),
                pricing_specification=deepcopy(pricing['specification'])),
            selection=projected,ranking_record=deepcopy(ranking),pricing_record=deepcopy(pricing),
            execution_orders=deepcopy(review['pricing']['execution_orders']),
            finished_at=ranking['finished_at'],paper_only=True,research_qualified=False,
            execution_authorized=False,order_submission_authorized=False)

    def prepare(self,primary,**kwargs):
        return self.compose(*self.adapter.prepare_pair(primary,**kwargs))

    def read_selection(self,connection,parent,*,row_read,now,base_dir):
        row=row_read(connection,self.native.selection_id(parent['primary_run_id']))
        need(row is not None,'shared_selection_missing')
        spec_sha,result_sha,record=row
        need(record.get('schema')==SCHEMA,'shared_selection_schema_required')
        need(spec_sha.strip()==self.rules.digest(record['specification']) and
             result_sha.strip()==self.rules.digest(record),'shared_selection_hash_changed')
        ranking=record['ranking_record'];pricing=record['pricing_record']
        need(ranking['primary_run_id']==parent['primary_run_id'] and
             ranking['parent_report_sha256']==parent['primary_report_sha256'] and
             ranking['run_id']==self.native.selection_id(parent['primary_run_id']) and
             ranking['source_bindings']==parent['source_bindings'] and ranking['session']==parent['session'] and
             self.rules.utc(ranking['finished_at'])<=now,'shared_selection_lineage_changed')
        handoffs,references,selection=self.native.inputs(connection,parent,now=now,base_dir=base_dir)
        expected_spec=dict(policy=self.native.POLICY,model_sha256=self.native.model_sha256(),
            required_strategies=sorted(handoffs),strategy_result_sha256=references)
        need(ranking['specification']==expected_spec and ranking['selection']==selection,
             'shared_ranking_reconstruction_changed')
        need(ranking.get('schema')=='jbravo.unified-setup-selection.v1' and
             ranking.get('status')=='COMPLETED_UNIFIED_SETUP_SELECTION' and
             ranking.get('paper_only') is True and ranking.get('research_qualified') is False and
             ranking.get('execution_authorized') is False and ranking.get('order_submission_authorized') is False and
             'execution_orders' not in ranking and 'execution_price_inputs' not in ranking,'shared_ranking_shape_changed')
        # Recreate the complete price record, including adapter pin and every diagnostic.
        review=self.adapter.contract.prepare_review(ranking,pricing['review']['price_inputs'],published_at=ranking['finished_at'])
        expected_pricing=dict(schema='jbravo.isolated-pipeline-pricing.v1',status='COMPLETED_ISOLATED_PRICING',
            run_id=self.adapter.pricing_id(ranking['run_id']),
            specification=dict(policy='RAW_SIP_AH_OR_REGULAR_15M_CLOSE_PLUS_0_5_PCT_V2',adapter_binding=deepcopy(self.adapter.binding)),
            review=review,paper_only=True,execution_authorized=False,order_submission_authorized=False)
        need(pricing==expected_pricing,'shared_price_reconstruction_changed')
        self.adapter.contract.entry_preview(ranking,review,handoffs)
        need(record==self.compose(ranking,expected_pricing),'shared_projection_reconstruction_changed')
        return dict(record=deepcopy(record),result_sha256=result_sha.strip(),handoffs=handoffs)

    def execution_queue(self,verified,*,challenger):
        record=verified['record']
        need(self.rules.digest(record)==verified['result_sha256'],'shared_verified_record_changed')
        ranking=record['ranking_record'];pricing=record['pricing_record']
        intermediate=dict(record=ranking,result_sha256=self.rules.digest(ranking),handoffs=verified['handoffs'],
            isolated_pricing=pricing,isolated_pricing_sha256=self.rules.digest(pricing))
        queue=self.adapter.execution_queue(intermediate,native_queue=self.native.execution_queue,challenger=challenger)
        for setup in queue:
            h=setup['handoff']
            h['unified_selection'].update(result_sha256=verified['result_sha256'],
                selected_signal_ids=record['selection']['selected_signal_ids'],
                ranking_result_sha256=self.rules.digest(ranking),contract_schema=SCHEMA)
            h.pop('handoff_id',None);h['handoff_id']=self.rules.digest(h)
        need([s['proposal']['signal_id'] for s in queue]==record['selection']['selected_signal_ids'],
             'shared_queue_population_changed')
        return queue

    def audit(self,verified):
        record=verified['record']
        need(self.rules.digest(record)==verified['result_sha256'],'shared_verified_record_changed')
        return dict(ranked_selection=deepcopy(record['ranking_record']['selection']),
            execution_selection=deepcopy(record['selection']),
            dispositions=deepcopy(record['pricing_record']['review']['pricing']['execution_price_dispositions']))

    def dashboard_rows(self, verified):
        """Show all ranked evidence, with explicit executable/excluded prices.

        The existing readiness consumer uses the executable projection; display
        must retain excluded candidates rather than silently dropping them.
        """
        record = verified['record']
        need(self.rules.digest(record) == verified['result_sha256'],
             'shared_verified_record_changed')
        ranking = record['ranking_record']
        display = dict(verified, record=ranking,
                       result_sha256=self.rules.digest(ranking))
        rows = self.native.dashboard_rows(display)
        pricing = record['pricing_record']['review']['pricing']
        setups = ranking['selection']['selected_setups']
        need(len(rows) == len(setups), 'shared_dashboard_population_changed')
        for row, setup in zip(rows, setups):
            signal = setup['proposal']['signal_id']
            ticket = pricing['execution_orders'].get(signal)
            row.update(execution_eligible=ticket is not None,
                       pricing_disposition=deepcopy(pricing['execution_price_dispositions'][signal]),
                       execution_limit_price=ticket['limit_price'] if ticket else None)
        return rows


    def dashboard_reports(self, kind, result, selection, reports, args):
        """Retained web forwards this projection to native report rendering."""
        need('ranked_selected_setups' in selection, 'shared_ranked_projection_missing')
        display = deepcopy(selection)
        display['selected_setups'] = display.pop('ranked_selected_setups')
        display['selected_signal_ids'] = [s['proposal']['signal_id'] for s in display['selected_setups']]
        return self.native.dashboard_reports(kind, result, display, reports, args)

