"""Hash-bound sparse inputs shared by daily Scanner and independent Backtest.

Build symbol first, strategy second so the bounded dataset LRU reads each raw /
split history once. Rule adapters and portfolio simulation remain unchanged.
No pickle, provider, database or broker operations are supported here.
"""
from pathlib import Path
import gzip
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

from scripts.research_dataset import canonical_hash, bounded_read, strict_json
from scripts.research_testing import json_default, save_json
from scripts.strategy_runtime_diagnostics import checkpoint, StrategyInputError

SOURCES = ('strategy_prepared_market.py', 'strategy_market_replay.py', 'strategy_runtime_diagnostics.py',
           'strategy_library.py', 'strategy_universe.py', 'research_dataset.py',
           'research_testing.py', 'setup_selection.py',
           'research_strategies.py', 'indicators.py', 'backtest_core.py', 'research_simulation.py',
           'bull_flag_strategy.py', 'chart_patterns.py', 'pattern_strategy.py', 'swing_patterns.py')
MAX_CHUNK = 32*1024*1024


def require(ok, code):
    if not ok:
        raise StrategyInputError(code)


def sources():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCES}


def context(spec):
    return {key: spec[key] for key in ('capture_manifest', 'actions_capture', 'universe', 'folds', 'portfolio')}


def reuse_dataset(dataset, spec):
    """Only a verified market snapshot with exactly the same data/risk scope."""
    require(getattr(dataset, 'market_wide', False) and context(dataset.spec) == context(spec),
            'PREPARED_DATASET_CONTEXT_CHANGED')
    return dataset


def pack(frame):
    columns = []
    for name in frame.columns:
        series = frame[name]
        dtype = str(series.dtype)
        require(dtype in {'float64', 'int64', 'bool', 'object'}, 'PREPARED_COLUMN_TYPE_UNSUPPORTED')
        values = series.tolist()
        if dtype == 'float64':
            values = [v if math.isfinite(v) else ('nan' if math.isnan(v) else 'inf' if v > 0 else '-inf') for v in values]
        columns.append(dict(name=name, dtype=dtype, values=values))
    return dict(index=[v.isoformat() for v in frame.index], index_name=frame.index.name, columns=columns)


def unpack(value):
    index = pd.DatetimeIndex(value['index'], name=value['index_name'])
    require(index.is_unique and index.is_monotonic_increasing, 'PREPARED_INDEX_INVALID')
    arrays = {}
    for column in value['columns']:
        dtype = column['dtype']; name = column['name']
        require(dtype in {'float64', 'int64', 'bool', 'object'} and name not in arrays and
                len(column['values']) == len(index), 'PREPARED_COLUMN_INVALID')
        arrays[name] = np.asarray(column['values'], dtype=dtype)
    return pd.DataFrame(arrays, index=index)


def build_prepared(dataset, spec, strategy_ids, output, *, deadline, stop_file=None, batch_size=25):
    from scripts.strategy_library import get_strategy
    from scripts.strategy_market_replay import prepare_symbols
    reuse_dataset(dataset, spec)
    require(len(spec['folds']) == 1 and 'prepared_market' not in spec and
            1 <= batch_size <= 25 and len(strategy_ids) == len(set(strategy_ids)), 'PREPARED_BUILD_SCOPE_INVALID')
    root = Path(output).resolve()
    require(root == Path(spec['capture_manifest']).resolve().parents[2]/'prepared', 'PREPARED_PATH_INVALID')
    root.mkdir(exist_ok=False)
    strategies = [get_strategy(sid) for sid in strategy_ids]
    definitions = {s.id: s.describe() for s in strategies}; pins = sources()
    entries = {s.id: [] for s in strategies}; total_bytes = 0
    started = time.monotonic(); timings = {s.id: 0. for s in strategies}
    pending = {s.id: [] for s in strategies}

    def flush():
        nonlocal total_bytes
        for strategy in strategies:
            rows = pending[strategy.id]
            if not rows:
                continue
            body = json.dumps(rows, sort_keys=True, separators=(',', ':'), default=json_default, allow_nan=False).encode()
            require(len(body) <= MAX_CHUNK, 'PREPARED_CHUNK_OVERSIZE')
            raw = gzip.compress(body, mtime=0)
            total_bytes += len(raw)
            require(total_bytes <= 2*1024**3, 'PREPARED_STORAGE_BUDGET_EXCEEDED')
            name = strategy.id+'-'+str(len(entries[strategy.id]))+'.json.gz'
            with (root/name).open('xb') as stream:
                stream.write(raw)
            entries[strategy.id].append(dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            rows.clear()

    for i, symbol in enumerate(spec['universe']['symbols']):
        for strategy in strategies:
            checkpoint('SHARED_MARKET_PREPARATION', deadline=deadline, stop_file=stop_file)
            strategy_started = time.monotonic()
            data, latest, dividends, observations = prepare_symbols(dataset, spec, strategy, spec['folds'][0],
                [symbol], deadline=deadline, stop_file=stop_file)
            pending[strategy.id].append(dict(symbol=symbol, frame=pack(data[symbol]) if symbol in data else None,
                latest=latest[symbol], dividends=dividends, observations=observations))
            timings[strategy.id] += time.monotonic()-strategy_started
        if (i+1) % batch_size == 0:
            flush()
    flush()
    checkpoint('SHARED_MARKET_PREPARATION', deadline=deadline, stop_file=stop_file)
    require(pins == sources() and definitions == {s.id: s.describe() for s in strategies}, 'PREPARED_SOURCE_CHANGED')
    manifest = dict(schema='jbravo.prepared-market.v1', context_sha256=canonical_hash(context(spec)),
        dataset_sha256=dataset.provenance['dataset_sha256'], sources=pins, strategies=definitions, chunks=entries,
        timing=dict(total_seconds=round(time.monotonic()-started,3),
                    strategy_seconds={sid:round(seconds,3) for sid,seconds in timings.items()}),
        retained_bytes=total_bytes, symbols=len(spec['universe']['symbols']))
    save_json(root/'manifest.json', manifest)
    raw = (root/'manifest.json').read_bytes()
    return dict(path=str(root/'manifest.json'), sha256=hashlib.sha256(raw).hexdigest())


def read_prepared(dataset, spec, strategy, fold, *, deadline=None, stop_file=None):
    checkpoint('PREPARED_INPUT_READBACK', deadline=deadline, stop_file=stop_file)
    reuse_dataset(dataset, spec)
    require(fold == spec['folds'][0] and len(spec['folds']) == 1, 'PREPARED_FOLD_CHANGED')
    ref = spec['prepared_market']; path = Path(ref['path']).resolve()
    require(path == Path(spec['capture_manifest']).resolve().parents[2]/'prepared/manifest.json', 'PREPARED_PATH_INVALID')
    raw = bounded_read(path, 2*1024*1024)
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'PREPARED_MANIFEST_CHANGED')
    value = strict_json(raw)
    require(value['schema'] == 'jbravo.prepared-market.v1' and value['context_sha256'] == canonical_hash(context(spec)) and
        value['dataset_sha256'] == dataset.provenance['dataset_sha256'] and value['sources'] == sources() and
        canonical_hash(value['strategies'][strategy.id]) == canonical_hash(strategy.describe()), 'PREPARED_LINEAGE_CHANGED')
    data, latest, dividends, observations, symbols = {}, {}, [], 0, []
    for pin in value['chunks'][strategy.id]:
        checkpoint('PREPARED_INPUT_READBACK', deadline=deadline, stop_file=stop_file)
        name = pin['path']
        require(Path(name).name == name and name.startswith(strategy.id+'-'), 'PREPARED_CHUNK_PATH_INVALID')
        raw = bounded_read(path.parent/name, MAX_CHUNK)
        require(len(raw) == pin['bytes'] and hashlib.sha256(raw).hexdigest() == pin['sha256'], 'PREPARED_CHUNK_CHANGED')
        # Bounded decompression also rejects oversized/tampered manifests.
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            body = stream.read(MAX_CHUNK+1)
        require(len(body) <= MAX_CHUNK, 'PREPARED_CHUNK_OVERSIZE')
        for row in strict_json(body):
            symbol = row['symbol']; symbols.append(symbol)
            if row['frame'] is not None:
                data[symbol] = unpack(row['frame'])
            latest[symbol] = row['latest']; dividends.extend(row['dividends']); observations += row['observations']
    require(symbols == spec['universe']['symbols'], 'PREPARED_SYMBOL_ACCOUNTING_CHANGED')
    return data, latest, dividends, observations
