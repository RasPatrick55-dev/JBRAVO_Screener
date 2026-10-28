"""Full six-strategy replay parity, sparse-input integrity and blocked outcomes."""
from copy import deepcopy
from datetime import timedelta
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import strategy_prepared_market as prepared
from scripts import strategy_market_replay as market
from scripts import strategy_pipeline as pipeline
from scripts import daily_strategy_scan as daily
from scripts import strategy_scanner as scanner
from scripts import bull_flag_companion as companion
from scripts import setup_selection
from scripts.research_dataset import load_dataset
from scripts.research_testing import json_default
from scripts.strategy_library import get_strategy
from scripts.swing_patterns import STRATEGIES
from test_strategy_universe import MarketProvider, capture, asset, NOW
from test_swing_patterns import pattern_frame

pytestmark = pytest.mark.alpaca_optional


def json_value(value):
    return json.loads(json.dumps(value, default=json_default, allow_nan=False))


def test_real_six_strategy_preparation_and_portfolio_replays_are_identical(tmp_path, monkeypatch):
    cfg = daily.load_config(); cfg['universe']['batch_size'] = 2
    symbols = ['AAA', 'BBB', 'CCC', 'DDD']
    provider = MarketProvider(rows=[asset(s) for s in symbols+['SPY']])
    for symbol, sid in zip(symbols, STRATEGIES):
        frame = pattern_frame(sid)
        for target, (_, source) in zip(provider.bars[symbol][-len(frame):], frame.iterrows()):
            for key, column in [('o','open'),('h','high'),('l','low'),('c','close'),('v','volume')]:
                target[key] = float(source[column])
    root, spec = capture(tmp_path, cfg, provider)
    dataset = load_dataset(spec)
    strategies = setup_selection.required_strategies()
    descriptions = {s:get_strategy(s).describe() for s in strategies}
    fold, cost = spec['folds'][0], spec['cost_cases'][0]
    original = {sid:scanner.evaluate_strategy(dataset, spec, get_strategy(sid), fold, cost, fold['test_end'])
                for sid in strategies}
    assert all(original[sid]['top_candidates'] for sid in STRATEGIES)
    dataset._cache.clear()
    read = dataset.read; reads=[]
    def count_read(relative):
        reads.append(relative)
        return read(relative)
    monkeypatch.setattr(dataset, 'read', count_read)
    ref = prepared.build_prepared(dataset, spec, strategies, root/'prepared', deadline=time.monotonic()+60, batch_size=2)
    manifest=json.loads((root/'prepared/manifest.json').read_bytes())
    assert manifest['timing']['shared_pattern_group_seconds']>0
    # One symbol history load serves all six unchanged adapters.
    assert sum(relative in dataset.manifest['shards'].values() for relative in reads) == len(symbols)
    cached_spec = dict(spec, prepared_market=ref)
    monkeypatch.setattr(dataset, 'prepare', lambda *a:pytest.fail('Cached replay prepared raw history again'))
    for sid in strategies:
        cached = scanner.evaluate_strategy(dataset, cached_spec, get_strategy(sid), fold, cost, fold['test_end'])
        assert json_value(cached) == json_value(original[sid])
    assert descriptions == {s:get_strategy(s).describe() for s in strategies}
    # Windows checkout CRLF bytes differ from the Linux production rule bytes.
    # Verify the production fingerprint with canonical repository LF blobs too.
    from dataclasses import asdict
    import hashlib
    definition=asdict(get_strategy(pipeline.STRATEGY))
    definition['source_sha256']={name:hashlib.sha256((Path(pipeline.__file__).parent/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        for name in descriptions[pipeline.STRATEGY]['source_sha256'] if name!='strategy_library.py'}
    assert prepared.canonical_hash(definition)=='bf576123697684b933b0f6772ab0121e8546fc9aaa1daa4084dbabcc27de0207'


def test_codec_preserves_float_bits_nested_decisions_and_types():
    frame = pd.DataFrame(dict(price=[np.nextafter(1., 2.), -0., np.nan, np.inf],
        integer=np.asarray([1,2,3,4], dtype='int64'), boolean=[True,False,True,False],
        detail=[{'score':np.nextafter(10.,11.), 'nested':[1, {'x':None}]}]*4),
        index=pd.date_range('2026-01-01',periods=4,name='session'))
    result = prepared.unpack(json_value(prepared.pack(frame)))
    pd.testing.assert_frame_equal(frame, result, check_exact=True, check_freq=False)
    assert np.signbit(result.price.iloc[1])


@pytest.fixture
def snapshot(tmp_path):
    cfg=daily.load_config();cfg['universe']['batch_size']=2
    root,spec=capture(tmp_path,cfg,MarketProvider())
    data=load_dataset(spec); strategy=get_strategy(cfg['strategy_id'])
    ref=prepared.build_prepared(data,spec,[strategy.id],root/'prepared',deadline=time.monotonic()+60)
    return root,dict(spec,prepared_market=ref),data,strategy


@pytest.mark.parametrize('change',['chunk','manifest','source','fold','universe','portfolio','missing'])
def test_bad_or_stale_prepared_inputs_never_fall_back_to_raw(snapshot,monkeypatch,change):
    root,spec,data,strategy=snapshot
    monkeypatch.setattr(data,'prepare',lambda *a:pytest.fail('Invalid cached input fell back'))
    if change=='chunk':
        path=next((root/'prepared').glob('*.gz'));path.write_bytes(path.read_bytes()+b' ')
    if change=='manifest':
        path=root/'prepared/manifest.json';path.write_bytes(path.read_bytes()+b' ')
    if change=='source':monkeypatch.setattr(prepared,'sources',lambda:{})
    if change in {'fold','universe','portfolio'}:
        spec=deepcopy(spec)
        if change=='fold':spec['folds'][0]['test_start']='2024-02-01'
        if change=='universe':spec['universe']['symbols']=spec['universe']['symbols'][:1]
        if change=='portfolio':spec['portfolio']['max_hold_days']+=1
    if change=='missing':next((root/'prepared').glob('*.gz')).unlink()
    with pytest.raises((ValueError,OSError)):
        market.prepare_market(data,spec,strategy,spec['folds'][0])


def test_deadline_has_safe_phase_receipt_and_no_completed_manifest(snapshot):
    root,spec,data,strategy=snapshot
    from scripts.strategy_runtime_diagnostics import StrategyBudgetError,diagnostic
    with pytest.raises(StrategyBudgetError) as error:
        market.prepare_market(data,spec,strategy,spec['folds'][0],deadline=time.monotonic()-1)
    receipt=diagnostic(error.value,'caller',time.monotonic())
    assert receipt['code']=='WALL_CLOCK_BUDGET_EXCEEDED' and receipt['phase']=='PREPARED_INPUT_READBACK'
    assert 'message' not in diagnostic(RuntimeError('secret-value'),'caller',time.monotonic())


def test_required_failed_screen_blocks_daily_completion(tmp_path,monkeypatch):
    cfg=daily.load_config();cfg['universe']['batch_size']=2
    def failed(*a,**kw):
        return dict(status='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED')
    monkeypatch.setattr(companion,'produce',failed)
    with pytest.raises(daily.DailyError,match='REQUIRED_STRATEGY_SCREENS_UNAVAILABLE'):
        daily.run_daily(cfg,MarketProvider(),tmp_path,now=NOW,archive=lambda *a:{'status':'VERIFIED_FIXTURE'})
    root=tmp_path/'2026-10-26'
    assert (root/'archive.json').exists() is False  # fixture archive does not write
    assert not (root/'completion.json').exists()
    blocked=json.loads((root/'blocked.json').read_bytes())
    assert set(blocked['unavailable_strategies'])==set(setup_selection.required_strategies())-{cfg['strategy_id']}
    assert json.loads((root/'failure.json').read_bytes())['phase']=='COMPANION_SCANS'


def test_failed_companion_primary_prevents_selection_publication(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(companion,'_status',lambda *a:None)
    monkeypatch.setattr(companion,'prepare_primary',lambda *a,**k:dict(status='COMPANION_UNAVAILABLE_PRIMARY_PRESERVED'))
    monkeypatch.setattr(setup_selection,'publish',lambda *a,**k:calls.append(1))
    record=dict(run_id='a'*32)
    result=pipeline.finish_strategy_handoff(record,env={},base_dir=Path(__file__).resolve().parents[1])
    assert result['status']=='UNIFIED_SETUP_SELECTION_UNAVAILABLE' and calls==[]
    assert len(result['unavailable_strategies'])==5


def test_successful_strategy_primaries_pass_exact_snapshot_to_selection(monkeypatch):
    calls=[];record=dict(run_id='a'*32);dataset=object()
    def finish(*a,**kw):
        assert kw['dataset'] is dataset
        calls.append(kw['strategy_id'])
        return dict(status='COMPLETED_COMPANION_PRIMARY')
    monkeypatch.setattr(companion,'prepare_primary',finish)
    monkeypatch.setattr(setup_selection,'publish',lambda r,**k:dict(status='COMPLETED_UNIFIED_SETUP_SELECTION'))
    assert pipeline.finish_strategy_handoff(record,env={},dataset=dataset)['status']=='COMPLETED_UNIFIED_SETUP_SELECTION'
    assert len(calls)==5 and len(set(calls))==5


@pytest.mark.parametrize('state',['missing','failed','old_run','wrong_parent','complete'])
def test_postflight_requires_fresh_bound_unified_completion(tmp_path,monkeypatch,state):
    from scripts import pipeline_postflight as health
    from datetime import datetime,timezone
    now=datetime.now(timezone.utc)
    record=dict(run_id='a'*32,invocation_id='a'*32,started_at=now.isoformat(),source_bindings={})
    raw=json.dumps(record).encode();folder=tmp_path/'reports/pipeline_postflight';folder.mkdir(parents=True)
    (folder/('run-'+record['run_id']+'.json')).write_bytes(raw)
    (folder/'active-primary.json').write_text(json.dumps(dict(id=record['run_id'],started_at=now.isoformat())))
    monkeypatch.setattr(pipeline,'assess',lambda *a,**k:{'status':'ok'})
    monkeypatch.setattr(pipeline,'enabled',lambda *a:True)
    monkeypatch.setattr(pipeline,'EXTRA_SOURCES',())
    monkeypatch.setattr(health,'SOURCE_FILES',())
    status=dict(status='COMPLETED_UNIFIED_SETUP_SELECTION',run_id=setup_selection.selection_id(record['run_id']),
                parent_report_sha256=pipeline.digest(pipeline.encoded(record)))
    if state=='failed':status['status']='UNIFIED_SETUP_SELECTION_UNAVAILABLE'
    if state=='old_run':status['run_id']='b'*64
    if state=='wrong_parent':status['parent_report_sha256']='c'*64
    if state!='missing':companion._status(status,tmp_path,'unified_setups')
    def check():
        return pipeline.check_report(record,raw,tmp_path,now=now,max_age_seconds=7200,
            not_before=None,require_completion=False)
    if state=='complete':assert check()['status']=='ok'
    else:
        with pytest.raises((ValueError,OSError)):check()
