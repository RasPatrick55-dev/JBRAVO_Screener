"""Review-only isolated primary/entry/dashboard integration.

No I/O on import. Ports are explicit. No broker/order API or connection discovery.
Runtime activation is deliberately separate from this testable integration.
"""
from copy import deepcopy
from datetime import datetime, timezone
from types import FunctionType
import re

SCHEMA='jbravo.isolated-pipeline-pricing.v1'


def need(ok,code):
    if not ok:raise ValueError(code)


def verify_adapter_files(root,binding):
    """Verify independently supplied release pins before wiring any runtime port."""
    from pathlib import Path
    from hashlib import sha256
    root=Path(root).resolve(strict=True)
    need(isinstance(binding,dict) and bool(binding),'isolated_adapter_binding_required')
    for name,pin in binding.items():
        need(isinstance(name,str) and name and not Path(name).is_absolute(),'isolated_adapter_path_invalid')
        path=(root/name).resolve(strict=True)
        need(root in path.parents and path.is_file(),'isolated_adapter_path_invalid')
        content=path.read_bytes()
        need(type(pin.get('bytes')) is int and len(content)==pin['bytes'] and
             sha256(content).hexdigest()==pin.get('sha256'),'isolated_adapter_source_changed')
    return deepcopy(binding)


def clone_function(function,overrides):
    """Private globals only; code, defaults and native module remain unchanged."""
    namespace=dict(function.__globals__);namespace.update(overrides)
    copied=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    copied.__kwdefaults__=deepcopy(function.__kwdefaults__)
    return copied


class PairedResearchStore:
    """Atomic immutable two-row publication; no overwrite or automatic retry."""
    def __init__(self,connect,digest):self.connect,self.digest=connect,digest
    def read(self,connection,run_id):
        with connection.cursor() as c:
            c.execute('SELECT specification_sha256,result_sha256,result FROM jbravo_research.research_runs WHERE run_id=%s',(run_id,))
            return c.fetchone()
    def publish(self,records):
        import json
        need(len(records)==2 and len({r['run_id'] for r in records})==2,'isolated_pair_required')
        c=self.connect(write=True)
        try:
            with c.cursor() as q:
                q.execute("SET LOCAL statement_timeout='10s'")
                q.execute("SET LOCAL lock_timeout='1500ms'")
                for r in records:
                    q.execute('INSERT INTO jbravo_research.research_runs (run_id,specification_sha256,result_sha256,result) VALUES (%s,%s,%s,%s::jsonb)',
                        (r['run_id'],self.digest(r['specification']),self.digest(r),json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)))
            c.commit()
        except BaseException:
            c.rollback();raise
        finally:c.close()
        c=self.connect(write=False)
        try:
            for r in records:
                row=self.read(c,r['run_id'])
                need(row is not None and row[0].strip()==self.digest(r['specification']) and row[1].strip()==self.digest(r) and row[2]==r,
                    'isolated_pair_committed_readback_failed')
        finally:c.rollback();c.close()


class IsolatedPricing:
    def __init__(self,*,rules,selection,contract,adapter_binding,clock=None):
        self.rules,self.selection,self.contract=rules,selection,contract
        need(isinstance(adapter_binding,dict) and adapter_binding and all(isinstance(v,dict) and
             type(v.get('bytes')) is int and v['bytes']>=0 and isinstance(v.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',v['sha256']) is not None
             for v in adapter_binding.values()),'isolated_adapter_binding_required')
        self.binding=deepcopy(adapter_binding)
        self.clock=clock or (lambda:datetime.now(timezone.utc))
    def pricing_id(self,selection_id):
        return self.rules.digest(dict(schema=SCHEMA,selection_run_id=selection_id))
    def prepare_pair(self,primary,*,selection,handoffs,references,price_inputs,published_at):
        native=self.selection
        specification=dict(policy=native.POLICY,model_sha256=native.model_sha256(),
            required_strategies=sorted(handoffs),strategy_result_sha256=references)
        record=dict(schema='jbravo.unified-setup-selection.v1',status='COMPLETED_UNIFIED_SETUP_SELECTION',
            run_id=native.selection_id(primary['run_id']),primary_run_id=primary['run_id'],
            parent_report_sha256=self.rules.digest(primary),source_bindings=deepcopy(primary['source_bindings']),
            session=deepcopy(primary['session']),specification=specification,selection=deepcopy(selection),
            finished_at=published_at,paper_only=True,research_qualified=False,execution_authorized=False,
            order_submission_authorized=False)
        review=self.contract.prepare_review(record,price_inputs,published_at=published_at)
        pricing=dict(schema=SCHEMA,status='COMPLETED_ISOLATED_PRICING',run_id=self.pricing_id(record['run_id']),
            specification=dict(policy='RAW_SIP_AH_OR_REGULAR_15M_CLOSE_PLUS_0_5_PCT_V2',
                adapter_binding=deepcopy(self.binding)),review=review,paper_only=True,
            execution_authorized=False,order_submission_authorized=False)
        return record,pricing
    def publish_primary(self,primary,*,inputs,capture,check_sources,publish_pair,report_status):
        """Only ports supplied by a separately pinned launcher may do I/O."""
        try:
            handoffs,references,selection=inputs(primary)
            prices=capture(selection,primary)
            pair=self.prepare_pair(primary,selection=selection,handoffs=handoffs,references=references,
                price_inputs=prices,published_at=self.clock().isoformat())
            check_sources(primary['source_bindings'])
            publish_pair(pair)
            status=dict(status='COMPLETED_UNIFIED_SETUP_SELECTION',run_id=pair[0]['run_id'],
                result_sha256=self.rules.digest(pair[0]),primary_run_id=primary['run_id'],
                parent_report_sha256=self.rules.digest(primary),selected_signal_ids=selection['selected_signal_ids'],
                pricing_run_id=pair[1]['run_id'],pricing_result_sha256=self.rules.digest(pair[1]),
                pricing_status=pair[1]['review']['pricing']['pricing_status'])
        except Exception:
            # Do not turn a failed/uncertain publication into a completed marker.
            status=dict(status='UNIFIED_SETUP_SELECTION_UNAVAILABLE',primary_run_id=primary['run_id'],
                parent_report_sha256=self.rules.digest(primary),failure_code='ISOLATED_PRICING_PUBLICATION_FAILED')
        report_status(status)
        return status
    def read_verified(self,connection,parent,*,native_read,row_read,base_dir,now=None):
        now=now or self.clock()
        verified=native_read(connection,parent,now=now,base_dir=base_dir)
        record=verified['record']
        need('execution_orders' not in record and 'execution_price_inputs' not in record,
             'isolated_selection_must_be_ranking_only')
        row=row_read(connection,self.pricing_id(record['run_id']))
        need(row is not None,'isolated_pricing_not_published')
        spec_hash,result_hash,pricing=row
        need(pricing.get('schema')==SCHEMA and pricing.get('status')=='COMPLETED_ISOLATED_PRICING' and
            pricing.get('run_id')==self.pricing_id(record['run_id']) and pricing.get('paper_only') is True and
            pricing.get('execution_authorized') is False and pricing.get('order_submission_authorized') is False and
            pricing.get('specification')==dict(policy='RAW_SIP_AH_OR_REGULAR_15M_CLOSE_PLUS_0_5_PCT_V2',adapter_binding=self.binding) and
            spec_hash.strip()==self.rules.digest(pricing['specification']) and result_hash.strip()==self.rules.digest(pricing),
            'isolated_pricing_artifact_changed')
        review=pricing['review']
        need(review.get('published_at')==record['finished_at'],'isolated_pricing_publication_time_changed')
        need(review==self.contract.prepare_review(record,review['price_inputs'],published_at=review['published_at']) and
            self.rules.utc(review['published_at'])<=now,'isolated_pricing_replay_changed')
        # Native validation reconstructs ranking; independent replay reconstructs pricing.
        self.contract.entry_preview(record,review,verified['handoffs'])
        return dict(verified,isolated_pricing=deepcopy(pricing),isolated_pricing_sha256=result_hash.strip())
    def execution_queue(self,verified,*,native_queue,challenger):
        record=verified['record'];pricing=verified['isolated_pricing'];review=pricing['review']
        need(verified['isolated_pricing_sha256']==self.rules.digest(pricing),'isolated_pricing_queue_binding_changed')
        self.contract.entry_preview(record,review,verified['handoffs'])
        executable=review['pricing']['executable_signal_ids'];tickets=review['pricing']['execution_orders']
        original=native_queue(verified);queue=[deepcopy(r) for r in original if r['proposal']['signal_id'] in executable]
        need(set(tickets)=={r['proposal']['signal_id'] for r in queue},'isolated_pricing_queue_population_changed')
        parent=verified['handoffs'][challenger]
        actionable=parent.get('pipeline_limit_placement_v2') is True and parent.get('inspection_mode') is False
        for setup in queue:
            handoff=setup['handoff'];proposal=setup['proposal']
            ticket=self.rules.validate_pipeline_limit_order(handoff,proposal,tickets[proposal['signal_id']],published_at=review['published_at'])
            handoff['unified_selection'].update(pricing_run_id=pricing['run_id'],pricing_result_sha256=verified['isolated_pricing_sha256'])
            handoff['pipeline_limit_order']=ticket;handoff['pipeline_limit_published_at']=review['published_at']
            if actionable:handoff['inspection_mode']=False;handoff['pipeline_limit_placement_v2']=True
            handoff['execution_policy'].update(model=self.rules.PIPELINE_LIMIT_MODEL,reference='PIPELINE_PUBLISHED_FINAL_LIMIT',limit_buffer_pct=0)
            handoff.pop('handoff_id',None);handoff['handoff_id']=self.rules.digest(handoff)
        return queue
    def dashboard(self,verified):
        record=verified['record'];review=verified['isolated_pricing']['review']
        need(verified['isolated_pricing_sha256']==self.rules.digest(verified['isolated_pricing']),'isolated_dashboard_binding_changed')
        self.contract.entry_preview(record,review,verified['handoffs'])
        pricing=review['pricing']
        return dict(pricing_status=pricing['pricing_status'],selected_count=len(record['selection']['selected_setups']),
            executable_count=len(pricing['executable_signal_ids']),excluded_count=len(record['selection']['selected_setups'])-len(pricing['executable_signal_ids']),
            execution_price_dispositions=deepcopy(pricing['execution_price_dispositions']),
            pricing_run_id=verified['isolated_pricing']['run_id'],pricing_result_sha256=verified['isolated_pricing_sha256'],
            entry_ready=False,entry_block_reason='BROKER_AND_ENTRY_WINDOW_NOT_CHECKED',order_submission_authorized=False)


def entry_facade(native_runtime,*,read_verified,queue):
    """Explicit runner dependency; no changes to native module/global state."""
    class Facade:
        unified_handoff=staticmethod(read_verified)
        pipeline_execution_queue=staticmethod(queue)
        guard_for=staticmethod(clone_function(native_runtime.guard_for,
            {'unified_handoff':read_verified,'pipeline_execution_queue':queue}))
        def __getattr__(self,name):return getattr(native_runtime,name)
    return Facade()


def primary_main(native_primary,native_pipeline,*,finish_handoff):
    main=clone_function(native_pipeline.main,{'finish_strategy_handoff':finish_handoff})
    def pipeline(argv):
        need('--handoff-invocation' in argv and native_pipeline.enabled(),'isolated_primary_route_required')
        return main(argv)
    def run(argv):
        need('--manual-provisional-test' not in argv,'isolated_primary_manual_run_forbidden')
        return native_primary.main(argv,pipeline_main=pipeline)
    return run


def finish_handoff_adapter(native_selection,*,prepare_primary,publish,status,digest,challenger):
    def finish(record,*,env,base_dir,dataset=None):
        unavailable=[]
        for strategy_id in native_selection.required_strategies(base_dir):
            if strategy_id==challenger:continue
            try:
                result=prepare_primary(record,env=env,base_dir=base_dir,strategy_id=strategy_id,dataset=dataset)
                if result.get('status')!='COMPLETED_COMPANION_PRIMARY':unavailable.append(strategy_id)
            except Exception:unavailable.append(strategy_id)
        if unavailable:
            result=dict(status='UNIFIED_SETUP_SELECTION_UNAVAILABLE',primary_run_id=record['run_id'],
                parent_report_sha256=digest(record),unavailable_strategies=unavailable)
            status(result,base_dir,'unified_setups');return result
        return publish(record,env=env,base_dir=base_dir)
    return finish


def install_entry_runner(native_runtime,*,facade,NativeEngine,entry,monitor,lifecycle,preflight):
    need(not getattr(native_runtime,'_entry_monitor_ownership_installed',False),'entry_ownership_duplicate_install')
    lifecycle.monitor_engine(NativeEngine)
    def run(config,*,base_dir=native_runtime.ROOT,wait=None,progress):
        return entry.run_entry_only(config,runtime=facade,NativeEngine=NativeEngine,
            calendar_reader=monitor.calendar_session,marker_for=lifecycle.marker_for,
            preflight=preflight.broker_entry_preflight,base_dir=base_dir,wait=wait,progress=progress)
    # Same supported process-local entry hook as the deployed ownership installer.
    # This does not replace any file or affect the separate monitor process.
    native_runtime._run_entry=run
    native_runtime._entry_monitor_ownership_installed=True
    return dict(native_source_files_changed=False,lifecycle_owner='MONITOR_POSITIONS',entry_only=True)


def dashboard_adapter(native_dashboard,adapter,*,row_read,parent_context,allow_legacy_display=False):
    from types import SimpleNamespace
    def display(connection,batch,primary_run_id,*,base_dir=native_dashboard.ROOT,now=None):
        verified=native_dashboard.display_selection(connection,batch,primary_run_id,base_dir=base_dir,now=now)
        if allow_legacy_display and 'execution_orders' in verified['record']:
            return verified  # Already validated by the unchanged deployed display reader.
        return adapter.read_verified(connection,parent_context(verified['primary']),
            native_read=lambda *a,**k:verified,row_read=row_read,base_dir=base_dir,now=now)
    def readiness(verified,candidate_rows,*,base_dir=native_dashboard.ROOT,now=None):
        result=native_dashboard.handoff_readiness(verified,candidate_rows,base_dir=base_dir,now=now)
        if allow_legacy_display and 'execution_orders' in verified['record'] and 'isolated_pricing' not in verified:
            return result
        summary=adapter.dashboard(verified)
        for key in ('entry_ready','entry_block_reason','order_submission_authorized'):summary.pop(key)
        result=dict(result,**summary)
        if summary['selected_count'] and not summary['executable_count']:
            reasons=list(result.get('entry_block_reasons',[]))
            reasons.append('unified_selection_no_priceable_signals')
            result.update(entry_ready=False,entry_block_reasons=reasons,
                entry_block_reason=reasons[0])
        result['order_submission_authorized']=False
        return result
    snapshot=clone_function(native_dashboard.snapshot,{'display_selection':display,'handoff_readiness':readiness})
    api=clone_function(native_dashboard.api_payload,{'snapshot':snapshot})
    return SimpleNamespace(display_selection=display,handoff_readiness=readiness,snapshot=snapshot,api_payload=api)


def capture_prices(selection,parent,*,rules,contract,get,now,broker_error_type=()):
    """Explicit GET-only market-data port, <=5 calls; no account/order methods.

    The launcher's port must bind the existing authorized data account, exact
    data origin, response bounds and timeouts. No provider/credential fallback.
    """
    from datetime import date,time,timedelta
    symbols=sorted({r['proposal']['symbol'] for r in selection['selected_setups']})
    need(len(symbols)<=3,'isolated_price_capture_symbol_budget')
    if not symbols:return {}
    day=parent['session']['signal_session'];close=datetime.combine(date.fromisoformat(day),time(16),rules.NY)
    end=close+timedelta(hours=4)-timedelta(microseconds=1)
    count=0
    def read(path,params):
        nonlocal count
        need(path in ('/v2/stocks/bars','/v2/stocks/trades') and count<5,'isolated_price_capture_request_scope')
        count+=1
        return get(path,params)
    def bars(group,start,end):
        value=read('/v2/stocks/bars',dict(symbols=','.join(group),start=start.isoformat(),end=end.isoformat(),
            timeframe='1Min',adjustment='raw',feed='sip',currency='USD',sort='asc',limit=1000))
        need(isinstance(value,dict) and isinstance(value.get('bars'),dict) and 'next_page_token' in value and
            value['next_page_token'] is None and set(value['bars'])<=set(group),'isolated_price_capture_incomplete')
        return value['bars']
    after=bars(symbols,close,end);observed=now().isoformat()
    ah={s:rules.after_hours_reference(after.get(s,[]),symbol=s,signal_session=day,observed_at=observed,
        feed='sip',adjustment='raw',currency='USD',complete=True) for s in symbols}
    missing=[s for s in symbols if ah[s]['status']=='NO_AFTER_HOURS_REFERENCE']
    regular={};activities={}
    if missing:
        raw=bars(missing,close-timedelta(minutes=15),close-timedelta(microseconds=1));captured=now().isoformat()
        regular={s:contract.regular_close_reference(raw.get(s,[]),symbol=s,signal_session=day,observed_at=captured,
            feed='sip',adjustment='raw',currency='USD',complete=True) for s in missing}
        for s in missing:
            try:
                trades=read('/v2/stocks/trades',dict(symbols=s,start=close.isoformat(),end=end.isoformat(),feed='sip',currency='USD',sort='desc',limit=100))
                need(isinstance(trades,dict) and isinstance(trades.get('trades'),dict) and set(trades['trades'])<={s} and
                    'next_page_token' in trades and (trades['next_page_token'] is None or isinstance(trades['next_page_token'],str)),
                    'isolated_trade_capture_incomplete')
                activities[s]=contract.after_hours_activity(trades['trades'].get(s,[]),symbol=s,signal_session=day,
                    observed_at=now().isoformat(),complete=trades['next_page_token'] is None)
            except (ValueError,KeyError,TypeError) as error:
                reason=('RAW_TRADES_HTTP_ERROR' if isinstance(error,broker_error_type) and error.http_status else
                        'RAW_TRADES_TRANSPORT_ERROR' if isinstance(error,broker_error_type) else 'RAW_TRADES_RESPONSE_INVALID')
                activities[s]=contract.after_hours_activity([],symbol=s,signal_session=day,
                    observed_at=now().isoformat(),complete=False,error=reason)
    return {s:contract.pipeline_price_reference(ah[s],regular.get(s),activities.get(s)) for s in symbols}

