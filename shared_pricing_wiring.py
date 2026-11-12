"""Offline shared-contract wiring. No installation or effects on import."""
import json
from types import SimpleNamespace
from isolated_pricing_integration import (PairedResearchStore, need,
    verify_adapter_files, entry_facade, capture_prices)
from pricing_adapter_prototype import pricing_contract
from isolated_pricing_integration import IsolatedPricing
from shared_pricing_contract import SharedPricingContract, SCHEMA


class SharedResearchStore(PairedResearchStore):
    def publish(self, record):
        need(record.get('schema') == SCHEMA, 'shared_schema_required')
        connection = self.connect(write=True)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout='10s'")
                cursor.execute("SET LOCAL lock_timeout='1500ms'")
                cursor.execute('INSERT INTO jbravo_research.research_runs '
                    '(run_id,specification_sha256,result_sha256,result) VALUES (%s,%s,%s,%s::jsonb)',
                    (record['run_id'], self.digest(record['specification']), self.digest(record),
                     json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)))
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        connection = self.connect(write=False)
        try:
            row = self.read(connection, record['run_id'])
            need(row is not None and row[0].strip() == self.digest(record['specification'])
                 and row[1].strip() == self.digest(record) and row[2] == record,
                 'shared_committed_readback_failed')
        finally:
            connection.rollback()
            connection.close()


def wire_shared(*, root, pins, rules, selection, runtime, connect,
                research_connect, market_get, check_sources, report_status,
                base_dir, challenger, clock, broker_error_type=()):
    required = {'shared_pricing_contract.py', 'shared_pricing_wiring.py',
                'isolated_pricing_integration.py', 'pricing_adapter_prototype.py'}
    need(required <= set(pins), 'shared_wiring_inventory_incomplete')
    pins = verify_adapter_files(root, pins)
    pricing = pricing_contract(rules, selection)
    adapter = IsolatedPricing(rules=rules, selection=selection, contract=pricing,
                              adapter_binding=pins, clock=clock)
    shared = SharedPricingContract(adapter)
    store = SharedResearchStore(connect, rules.digest)

    def read(connection, parent, *, now=None, base_dir=base_dir):
        verify_adapter_files(root, pins)
        check_sources(parent['source_bindings'])
        row = store.read(connection, selection.selection_id(parent['primary_run_id']))
        need(row is not None, 'shared_selection_missing')
        if row[2].get('schema') == 'jbravo.unified-setup-selection.v1':
            return selection.read_selection(connection, parent, now=now or clock(), base_dir=base_dir)
        return shared.read_selection(connection, parent, row_read=store.read,
                                     now=now or clock(), base_dir=base_dir)

    def queue(verified):
        verify_adapter_files(root, pins)
        check_sources(verified['record']['source_bindings'])
        if verified['record'].get('schema') == 'jbravo.unified-setup-selection.v1':
            need(rules.digest(verified['record']) == verified['result_sha256'], 'shared_verified_record_changed')
            return runtime.pipeline_execution_queue(verified)
        return shared.execution_queue(verified, challenger=challenger)

    def native_queue(verified):
        verify_adapter_files(root, pins)
        check_sources(verified['record']['source_bindings'])
        need(rules.digest(verified['record']) == verified['result_sha256'], 'shared_verified_record_changed')
        if verified['record'].get('schema') == 'jbravo.unified-setup-selection.v1':
            return selection.execution_queue(verified)
        return shared.execution_queue(verified, challenger=challenger)

    def rows(verified):
        verify_adapter_files(root, pins)
        if verified['record'].get('schema') == 'jbravo.unified-setup-selection.v1':
            need(rules.digest(verified['record']) == verified['result_sha256'], 'shared_verified_record_changed')
            return selection.dashboard_rows(verified)
        return shared.dashboard_rows(verified)

    def reports(kind, result, selected, strategy_reports, args):
        verify_adapter_files(root, pins)
        if 'ranked_selected_setups' not in selected:
            return selection.dashboard_reports(kind, result, selected, strategy_reports, args)
        return shared.dashboard_reports(kind, result, selected, strategy_reports, args)

    def publish(primary, *, inputs):
        try:
            verify_adapter_files(root, pins)
            check_sources(primary['source_bindings'])
            handoffs, references, selected = inputs(primary)
            prices = capture_prices(selected, primary, rules=rules, contract=pricing,
                get=market_get, now=clock, broker_error_type=broker_error_type)
            record = shared.prepare(primary, selection=selected, handoffs=handoffs,
                references=references, price_inputs=prices, published_at=clock().isoformat())
            # Recheck after provider I/O, before any immutable database publication.
            verify_adapter_files(root, pins)
            check_sources(primary['source_bindings'])
            store.publish(record)
            outcome = record['pricing_record']['review']['pricing']
            status = dict(status=record['status'], run_id=record['run_id'],
                result_sha256=rules.digest(record), primary_run_id=primary['run_id'],
                parent_report_sha256=rules.digest(primary),
                selected_signal_ids=record['selection']['selected_signal_ids'],
                ranked_signal_ids=record['ranking_record']['selection']['selected_signal_ids'],
                pricing_status=outcome['pricing_status'],
                executable_signal_ids=outcome['executable_signal_ids'],
                executable_count=len(outcome['execution_orders']),
                excluded_count=len(selected['selected_setups'])-len(outcome['execution_orders']),
                execution_price_dispositions=outcome['execution_price_dispositions'])
        except Exception:
            status = dict(status='UNIFIED_SETUP_SELECTION_UNAVAILABLE',
                primary_run_id=primary['run_id'], parent_report_sha256=rules.digest(primary),
                failure_code='SHARED_PRICING_PUBLICATION_FAILED')
        report_status(status)
        return status

    def unified_handoff(env, parent, *, base_dir=base_dir):
        connection = research_connect(env=env)
        try:
            return read(connection, parent, now=clock(), base_dir=base_dir)
        finally:
            connection.rollback()
            connection.close()

    facade = entry_facade(runtime, read_verified=unified_handoff, queue=queue)
    return SimpleNamespace(shared=shared, store=store, read_selection=read,
        execution_queue=queue, native_execution_queue=native_queue, dashboard_rows=rows, dashboard_reports=reports,
        publish=publish, entry_runtime=facade)



