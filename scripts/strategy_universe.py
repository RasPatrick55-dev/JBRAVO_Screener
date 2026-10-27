"""Complete broker universes and bounded, hash-bound market data shards.

Only GET endpoints are used. Cache files accelerate captures; immutable capture
shards and the PostgreSQL archive, never the cache, define a completed scan.
"""
from collections import OrderedDict
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time as clock

import numpy as np
import pandas as pd

from scripts.research_dataset import bounded_read, canonical_hash, strict_json
from scripts.research_testing import save_json

ASSETS = 'https://paper-api.alpaca.markets/v2/assets'
SCHEMA = 'jbravo.market-capture.v1'
MAX_MEMBER = 4 * 1024 * 1024
EXCHANGES = {'AMEX', 'ARCA', 'BATS', 'NYSE', 'NASDAQ', 'NYSEARCA'}
# Match the existing paper order-state schema and broker adapter. Anything
# outside that contract receives an explicit exclusion, never a failed insert.
SYMBOL = re.compile(r'[A-Z][A-Z0-9.\-]{0,15}')


def require(ok, code):
    if not ok:
        raise ValueError(code)


def binding(body):
    return dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def market_config(cfg):
    return cfg.get('schema') == 'jbravo.daily-strategy-config.v2'


def select_assets(rows, cfg, session, now):
    """Account for every returned asset. Never cap a universe by slicing it."""
    require(isinstance(rows, list) and 0 < len(rows) <= cfg['universe']['max_symbols'],
            'ASSET_UNIVERSE_EMPTY_OR_OVER_BUDGET')
    selected, decisions, ids, symbols = {}, [], set(), set()
    for row in rows:
        require(isinstance(row, dict) and all(k in row for k in
                ('id', 'symbol', 'class', 'status', 'tradable', 'exchange')), 'ASSET_METADATA_INCOMPLETE')
        sid, symbol = row['id'], row['symbol']
        require(isinstance(sid, str) and sid and sid not in ids and isinstance(symbol, str) and
                symbol not in symbols and type(row['tradable']) is bool, 'AMBIGUOUS_ASSET_IDENTITY')
        ids.add(sid); symbols.add(symbol)
        reason = ('INACTIVE' if row['status'] != 'active' else
                  'NOT_US_EQUITY' if row['class'] != 'us_equity' else
                  'NOT_TRADABLE' if not row['tradable'] else
                  'UNSUPPORTED_SYMBOL' if SYMBOL.fullmatch(symbol) is None else
                  'UNSUPPORTED_SIP_EXCHANGE' if row['exchange'] not in EXCHANGES else
                  'BENCHMARK_ONLY' if symbol == cfg['benchmark'] else 'DATA_PENDING')
        item = {k: row[k] for k in ('id', 'symbol', 'class', 'status', 'tradable', 'exchange')}
        item['reason'] = reason; decisions.append(item)
        if reason in {'DATA_PENDING', 'BENCHMARK_ONLY'}:
            selected[symbol] = item
    require(cfg['benchmark'] in selected, 'TRADABLE_BENCHMARK_UNAVAILABLE')
    require(len(selected) > 1, 'NO_TRADABLE_UNIVERSE')
    return selected, dict(schema='jbravo.market-coverage.v1', signal_session=session,
        discovered_at=now.isoformat(), policy=cfg['universe'], benchmark=cfg['benchmark'],
        assets=sorted(decisions, key=lambda r: r['symbol']), status='INCOMPLETE')


def normalized_bars(items, start, session, ny):
    from dateutil.parser import isoparse
    require(isinstance(items, list), 'INVALID_SYMBOL_BARS')
    result, previous = [], None
    for bar in items:
        stamp = isoparse(bar['t'])
        require(stamp.tzinfo is not None, 'NAIVE_MARKET_BAR')
        local = stamp.astimezone(ny)
        day = local.date().isoformat()
        require(local.time() == time.min and start <= day <= session and
                (previous is None or previous < day), 'DUPLICATE_OR_OUT_OF_RANGE_MARKET_BAR')
        previous = day
        values = [bar[k] for k in ('o', 'h', 'l', 'c', 'v')]
        require(all(type(x) in (int, float) and np.isfinite(x) for x in values), 'NONFINITE_MARKET_BAR')
        o, h, l, c, v = values
        require(min(o, h, l, c) > 0 and l <= min(o, c) <= max(o, c) <= h and
                v >= 0 and int(v) == v, 'INVALID_MARKET_OHLCV')
        result.append(dict(session=day, **dict(zip(('o', 'h', 'l', 'c', 'v'), values))))
    return result


def data_reason(raw, split, events, calendar, cfg, session):
    """Data eligibility only: no price, volume-liquidity or performance filters."""
    if not raw or not split:
        return 'NO_DAILY_DATA'
    for bar in raw + split:
        values = [bar[k] for k in ('o','h','l','c','v')]
        require(all(type(x) in (int,float) and np.isfinite(x) for x in values), 'NONFINITE_MARKET_BAR')
        o,h,l,c,v = values
        require(min(o,h,l,c)>0 and l<=min(o,c)<=max(o,c)<=h and v>=0 and int(v)==v,
                'INVALID_MARKET_OHLCV')
    raw_days = [r['session'] for r in raw]; split_days = [r['session'] for r in split]
    if raw_days != split_days:
        return 'RAW_SPLIT_SESSION_MISMATCH'
    if raw_days[-1] != session:
        return 'MISSING_CURRENT_SESSION'
    if len(raw_days) < 250:
        return 'INSUFFICIENT_250_SESSION_HISTORY'
    expected = [r['date'] for r in calendar if raw_days[0] <= r['date'] <= session]
    if raw_days != expected:
        return 'MISSING_HISTORICAL_SESSIONS'
    if any(r['v'] <= 0 for r in raw + split):
        return 'ZERO_VOLUME_HISTORY'
    ratios = np.array([r['c'] / s['c'] for r, s in zip(raw, split)])
    if not np.allclose(ratios, ratios[0], rtol=1e-7, atol=1e-9):
        return 'SHARE_ACTION_ADAPTER_REQUIRED'
    for field in ('o', 'h', 'l'):
        if not np.allclose([r[field] for r in raw], np.array([s[field] for s in split]) * ratios,
                           rtol=1e-5, atol=1e-5):
            return 'RAW_SPLIT_PRICE_MISMATCH'
    ids = set()
    for category, event in events:
        require(isinstance(event.get('id'), str) and event['id'] not in ids, 'DUPLICATE_ACTION_ID')
        ids.add(event['id'])
        day = event.get('ex_date') or event.get('effective_date') or event.get('process_date')
        try:
            valid_day = isinstance(day, str) and date.fromisoformat(day).isoformat() == day
        except ValueError:
            valid_day = False
        if not valid_day:
            return 'INVALID_ACTION_DATE'
        if not raw_days[0] <= day <= session:
            continue
        if category != 'cash_dividends':
            return 'UNSUPPORTED_CORPORATE_ACTION_' + category.upper()
        # Large distributions may legitimately pay before the ex-date. Their
        # due-bill entitlement/settlement needs a different accounting adapter;
        # exclude this symbol instead of aborting unrelated market coverage.
        if any(event.get(k) is not None for k in ('due_bill_on_date', 'due_bill_off_date')):
            return 'CASH_DIVIDEND_DUE_BILL_ADAPTER_REQUIRED'
        if not (type(event.get('rate')) in (int, float) and
                np.isfinite(event['rate']) and event['rate'] >= 0):
            return 'INVALID_CASH_DIVIDEND_RATE'
        payable = event.get('payable_date')
        try:
            valid_payable = (isinstance(payable, str) and
                date.fromisoformat(payable).isoformat() == payable and payable >= day)
        except ValueError:
            valid_payable = False
        if not valid_payable:
            return 'INVALID_CASH_DIVIDEND_PAYABLE_DATE'
        if day not in expected:
            return 'INVALID_CASH_DIVIDEND_EX_SESSION'
        if not isinstance(event.get('symbol'), str) or not event['symbol']:
            return 'INVALID_CASH_DIVIDEND_SYMBOL'
    return None


def _cache_key(asset, cfg):
    return canonical_hash(dict(asset_id=asset['id'], symbol=asset['symbol'],
        start=cfg['warmup_start'], feed=cfg['feed'], currency=cfg['currency'], schema=SCHEMA))


def _gzip_json(body):
    require(len(body) <= MAX_MEMBER, 'MARKET_COMPRESSED_MEMBER_TOO_LARGE')
    with gzip.GzipFile(fileobj=io.BytesIO(body)) as handle:
        decoded=handle.read(MAX_MEMBER+1)
    require(len(decoded)<=MAX_MEMBER,'MARKET_DECOMPRESSED_MEMBER_TOO_LARGE')
    return strict_json(decoded)


def read_cache(folder, asset, cfg, session):
    path = folder / (_cache_key(asset, cfg) + '.json.gz')
    if not path.exists() or path.is_symlink():
        return None
    # Cache corruption is a cache miss, never a published data exception.
    try:
        value = _gzip_json(bounded_read(path, MAX_MEMBER))
        require(value['sha256'] == canonical_hash(value['data']), 'CACHE_HASH_MISMATCH')
        value = value['data']
        require(value['key'] == _cache_key(asset, cfg) and
                0 < (date.fromisoformat(session) - date.fromisoformat(value['session'])).days <= 7,
                'CACHE_SCOPE_MISMATCH')
        return value
    except (ValueError, KeyError, TypeError, OSError, EOFError):
        return None


def write_cache(folder, asset, cfg, session, raw, split):
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = dict(key=_cache_key(asset, cfg), session=session, raw=raw, split=split)
    body = gzip.compress(json.dumps(dict(data=data, sha256=canonical_hash(data)),
                                   separators=(',', ':'), allow_nan=False).encode(), mtime=0)
    require(len(body) < MAX_MEMBER, 'MARKET_CACHE_MEMBER_TOO_LARGE')
    target = folder / (_cache_key(asset, cfg) + '.json.gz')
    with tempfile.NamedTemporaryFile(dir=folder, delete=False) as handle:
        temporary = Path(handle.name); handle.write(body)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def captured_object(folder, objects, relative, value):
    """Use verified hardlinks to immutable content-addressed historical chunks.

    Every session still has a contained, hash-bound file. Deleting the disposable
    object/cache name cannot remove a completed capture's independent hardlink.
    """
    body=gzip.compress(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),mtime=0)
    pin=binding(body);path=folder/relative
    require(len(body)<MAX_MEMBER,'MARKET_OBJECT_TOO_LARGE')
    objects.mkdir(parents=True,mode=0o700,exist_ok=True)
    source=objects/(pin['sha256']+'.json.gz')
    if not source.exists():
        with tempfile.NamedTemporaryFile(dir=objects,delete=False) as handle:
            temporary=Path(handle.name);handle.write(body);handle.flush();os.fsync(handle.fileno())
        try:
            try:os.link(temporary,source)
            except FileExistsError:pass
        finally:temporary.unlink(missing_ok=True)
    require(source.is_file() and not source.is_symlink() and binding(bounded_read(source,MAX_MEMBER))==pin,
            'MARKET_OBJECT_CHANGED')
    os.link(source,path,follow_symlinks=False)
    return dict(path='analysis/'+relative,**pin)


def capture_market(provider, cfg, calendar, session, root, *, now, cache_dir=None):
    from scripts.daily_strategy_scan import DATA, NY, pages
    from scripts.research_testing import validate_spec
    folder = root / 'capture/analysis'; folder.mkdir(parents=True, mode=0o700)
    responses = root / 'provider-responses'; responses.mkdir(mode=0o700)
    shard_dir = folder / 'symbols'; shard_dir.mkdir(mode=0o700)
    pins, shards = [], {}
    def save_member(relative, body):
        path = folder / relative
        with path.open('xb') as handle:
            handle.write(body)
        pins.append(dict(path='analysis/' + relative, **binding(body)))
    assets, body = provider.get(ASSETS, dict(status='active', asset_class='us_equity'))
    save_member('assets.json', body)
    selected, coverage = select_assets(assets, cfg, session, now)
    save_json(root / 'coverage-pending.json', coverage)
    cache_dir = Path(cache_dir) if cache_dir else None
    objects=(cache_dir/'objects') if cache_dir else root.parent/'market-objects'
    # Group cache hits separately. One new/ineligible listing must not force a
    # full multi-year refresh for every other symbol sharing its batch.
    symbols = sorted(selected,key=lambda s:(not (cache_dir and
        (cache_dir/(_cache_key(selected[s],cfg)+'.json.gz')).is_file()),s))
    dates = [r['date'] for r in calendar if r['date'] <= session]
    overlap = dates[max(0, len(dates) - 20)]
    stats = dict(cache_hits=0, full_refresh_batches=0)
    for number, offset in enumerate(range(0, len(symbols), cfg['universe']['batch_size'])):
        batch = symbols[offset:offset + cfg['universe']['batch_size']]
        cached = {s: read_cache(cache_dir, selected[s], cfg, session) if cache_dir else None for s in batch}
        start = overlap if all(cached.values()) else cfg['warmup_start']
        # Fetch both conventions with the same mapping date, and verify every page.
        def fetch_bars(start_day, suffix):
            found = {kind: {s: [] for s in batch} for kind in ('raw', 'split')}
            for kind in found:
                params = dict(symbols=','.join(batch), timeframe='1Day',
                    start=datetime.combine(date.fromisoformat(start_day), time.min, NY).isoformat(),
                    end=datetime.combine(date.fromisoformat(session), time(20), NY).isoformat(),
                    feed=cfg['feed'], currency=cfg['currency'], adjustment=kind,
                    asof=now.astimezone(NY).date().isoformat(), limit=10000, sort='asc')
                for page, value, raw_body in pages(provider, DATA+'/v2/stocks/bars', params, max_pages=32):
                    with (responses / f'b{number}-{suffix}-{kind}-{page}.json.gz').open('xb') as handle:
                        handle.write(gzip.compress(raw_body,mtime=0))
                    require(isinstance(value.get('bars'), dict) and set(value['bars']) <= set(batch),
                            'MARKET_BAR_POPULATION_MISMATCH')
                    for symbol, items in value['bars'].items():
                        found[kind][symbol].extend(normalized_bars(items, start_day, session, NY))
            for kind in found:
                for rows in found[kind].values():
                    days = [r['session'] for r in rows]
                    require(days == sorted(set(days)), 'MARKET_BAR_PAGINATION_DUPLICATE')
            return found
        found = fetch_bars(start, 'initial')
        if start != cfg['warmup_start']:
            same = all([r for r in cached[s][kind] if start <= r['session'] <= cached[s]['session']] ==
                       [r for r in found[kind][s] if r['session'] <= cached[s]['session']]
                       for s in batch for kind in ('raw', 'split'))
            if not same:
                found = fetch_bars(cfg['warmup_start'], 'revised'); stats['full_refresh_batches'] += 1
            else:
                for s in batch:
                    for kind in ('raw', 'split'):
                        found[kind][s] = [r for r in cached[s][kind] if r['session'] < start] + found[kind][s]
                stats['cache_hits'] += len(batch)
        events = {s: [] for s in batch}; action_ids = set()
        params = dict(symbols=','.join(batch), start=cfg['warmup_start'],
                      end=now.astimezone(NY).date().isoformat(), limit=1000, sort='asc')
        for page, value, raw_body in pages(provider, DATA+'/v1/corporate-actions', params, max_pages=32):
            with (responses / f'b{number}-actions-{page}.json.gz').open('xb') as handle:
                handle.write(gzip.compress(raw_body,mtime=0))
            require(isinstance(value.get('corporate_actions'), dict), 'MARKET_ACTION_ENVELOPE_INVALID')
            for category, items in value['corporate_actions'].items():
                require(isinstance(items, list), 'INVALID_MARKET_ACTION_LIST')
                for event in items:
                    require(isinstance(event, dict) and isinstance(event.get('id'),str) and
                            event['id'] and event['id'] not in action_ids, 'DUPLICATE_ACTION_ID')
                    action_ids.add(event.get('id'))
                    mentioned = {v for k, v in event.items() if k == 'symbol' or k.endswith('_symbol')}
                    require(bool(mentioned & set(batch)), 'ACTION_POPULATION_MISMATCH')
                    for symbol in mentioned & set(batch):
                        events[symbol].append((category, event))
        for symbol in batch:
            raw, split = found['raw'][symbol], found['split'][symbol]
            reason = data_reason(raw, split, events[symbol], calendar, cfg, session)
            selected[symbol]['reason'] = reason or ('BENCHMARK_ONLY' if symbol == cfg['benchmark'] else 'EVALUATED')
            if symbol == cfg['benchmark']:
                require(reason is None, 'BENCHMARK_DATA_INVALID_' + str(reason))
            if reason:
                continue
            chunks=[]
            for part,begin in enumerate(range(0,len(raw),128)):
                member='symbols/'+canonical_hash(symbol)+'-'+str(part)+'.json.gz'
                pin=captured_object(folder,objects,member,dict(symbol=symbol,asset_id=selected[symbol]['id'],
                    raw=raw[begin:begin+128],split=split[begin:begin+128]))
                pins.append(pin);chunks.append(pin['path'])
            value = dict(symbol=symbol, asset_id=selected[symbol]['id'], chunks=chunks,actions=events[symbol])
            relative = 'symbols/' + canonical_hash(symbol) + '.json.gz'
            save_member(relative, gzip.compress(json.dumps(value, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
            shards[symbol] = 'analysis/' + relative
            if cache_dir:
                write_cache(cache_dir, selected[symbol], cfg, session, raw, split)
    # selected values refer to the same records in coverage; no symbol may stay pending.
    require(all(r['reason'] != 'DATA_PENDING' for r in coverage['assets']), 'UNIVERSE_COVERAGE_INCOMPLETE')
    eligible = sorted(s for s in shards if s != cfg['benchmark'])
    excluded = {r['symbol']: r['reason'] for r in coverage['assets'] if r['reason'] not in {'EVALUATED', 'BENCHMARK_ONLY'}}
    coverage.update(status='COMPLETE', evaluated_symbols=eligible, exclusions=excluded,
                    discovered_count=len(assets), evaluated_count=len(eligible), excluded_count=len(excluded), **stats)
    save_member('coverage.json', json.dumps(coverage, sort_keys=True, allow_nan=False).encode())
    save_member('calendar.json', json.dumps(calendar, sort_keys=True).encode())
    # Compact provenance travels through primary/Backtest/Metrics. Full coverage is hash-bound on disk.
    universe = dict(symbols=eligible, benchmark=cfg['benchmark'], excluded={})
    require(bool(eligible),'NO_EVALUABLE_TRADABLE_SYMBOLS')
    compact = dict(source='alpaca_assets', status='COMPLETE', discovered_count=len(assets),
        tradable_count=len(selected)-1,
        evaluated_count=len(eligible), excluded_count=len(excluded), benchmark=cfg['benchmark'],
        coverage_sha256=pins[-2]['sha256'], symbols_sha256=canonical_hash(eligible), policy=cfg['universe'])
    save_json(folder/'manifest.json', dict(schema=SCHEMA, research_qualified=False, execution_authorized=False,
        signal_session=session, asof=now.astimezone(NY).date().isoformat(), selection=compact,
        bindings=pins, shards=shards))
    save_json(root/'actions.json', dict(schema='jbravo.sharded-actions.v1', capture_manifest=str(folder/'manifest.json')))
    spec = dict(schema='jbravo.strategy-study.v1', purpose='exploratory_research',
        capture_manifest=str(folder/'manifest.json'),
        actions_capture=dict(path=str(root/'actions.json'), sha256=binding((root/'actions.json').read_bytes())['sha256']),
        universe=universe, folds=[dict(id=cfg['episode_id'], warmup_start=cfg['warmup_start'],
            test_start=cfg['episode_start'], test_end=session)], strategies=[cfg['strategy_id']],
        cost_cases=[cfg['cost_case']], portfolio=cfg['portfolio'], max_runs=2,
        max_seconds=cfg['max_scan_seconds'], research_qualified=False, execution_authorized=False)
    validate_spec(spec); save_json(root/'study.json', spec)
    return spec


class SymbolFrames(Mapping):
    def __init__(self, source, kind):
        self.source, self.kind = source, kind
    def __iter__(self): return iter(self.source.manifest['shards'])
    def __len__(self): return len(self.source.manifest['shards'])
    def __getitem__(self, symbol): return self.source.member(symbol)[self.kind]


class MarketDataset:
    def __init__(self, spec, manifest, path, pins, coverage):
        self.spec, self.manifest, self.path, self.pins = spec, manifest, path, pins
        self.coverage = coverage; self._cache = OrderedDict()
        self.asset_ids={r['symbol']:r['id'] for r in coverage['assets']}
        self.raw, self.features = SymbolFrames(self, 'raw'), SymbolFrames(self, 'split')
        self.calendar = pd.DataFrame(strict_json(self.read('analysis/calendar.json'))).set_index('date')
        self.dividends = ()
        self.market_wide = True
        self.provenance = dict(capture_manifest=str(path), capture_manifest_sha256=binding(path.read_bytes())['sha256'],
            selection=manifest['selection'], feed='sip', currency='USD', execution_prices='raw', feature_prices='split',
            asof=manifest['asof'], historical_availability='CACHED_HISTORY_WITH_FRESH_OVERLAP_AND_CURRENT_SNAPSHOT',
            research_qualified=False, execution_authorized=False,
            limitations=['Current active/tradable universe; historical survivorship and no untouched holdout',
                'Cached history is overlap-checked; older provider revisions require a full refresh',
                'Share actions, gaps, zero-volume history and insufficient warmup are explicitly excluded'])
        self.provenance['dataset_sha256'] = canonical_hash(self.provenance)

    def read(self, relative):
        require(relative in self.pins, 'UNBOUND_MARKET_MEMBER')
        path = (self.path.parent.parent / relative).resolve()
        require(path.is_relative_to(self.path.parent), 'MARKET_MEMBER_PATH_ESCAPE')
        body = bounded_read(path, 16*1024*1024 if not relative.endswith('.gz') else MAX_MEMBER)
        require(binding(body) == {k:self.pins[relative][k] for k in ('bytes', 'sha256')}, 'MARKET_MEMBER_CHANGED')
        return body

    def member(self, symbol):
        if symbol in self._cache:
            self._cache.move_to_end(symbol); return self._cache[symbol]
        value = _gzip_json(self.read(self.manifest['shards'][symbol]))
        require(value['symbol'] == symbol and value['asset_id']==self.asset_ids[symbol], 'MARKET_SHARD_SYMBOL_CHANGED')
        raw,split=[],[]
        require(isinstance(value['chunks'],list) and 1<=len(value['chunks'])<=64 and
            len(value['chunks'])==len(set(value['chunks'])),'INVALID_MARKET_CHUNK_LIST')
        for relative in value['chunks']:
            chunk=_gzip_json(self.read(relative))
            require(chunk['symbol']==symbol and chunk['asset_id']==self.asset_ids[symbol], 'MARKET_CHUNK_IDENTITY_CHANGED')
            raw.extend(chunk['raw']);split.extend(chunk['split'])
        value.update(raw=raw,split=split)
        calendar=self.calendar.reset_index().to_dict('records')
        require(data_reason(value['raw'],value['split'],value['actions'],calendar,
            {},self.manifest['signal_session']) is None,'MARKET_SHARD_DATA_CHANGED')
        result = dict(actions=value['actions'])
        for kind in ('raw', 'split'):
            rows = value[kind]
            frame = pd.DataFrame(rows).rename(columns=dict(o='open', h='high', l='low', c='close', v='volume'))
            frame.index = pd.to_datetime(frame['session'])
            result[kind] = frame
        self._cache[symbol] = result
        if len(self._cache) > 4:
            self._cache.popitem(last=False)
        return result

    def prepare(self, symbol, strategy, fold):
        raw = self.raw[symbol].loc[fold['warmup_start']:fold['test_end']]
        split = self.features[symbol].loc[fold['warmup_start']:fold['test_end']]
        if len(raw)<250:
            raise ValueError('Insufficient fresh warmup at evaluation boundary')
        # Newer listings start evaluating only once the same 250-session warmup is ready.
        boundary = max(fold['test_start'], raw.index[249].strftime('%Y-%m-%d'))
        return strategy.prepare(raw, split, dict(fold, test_start=boundary))


def load_market_dataset(spec, manifest, path):
    path = Path(path).resolve()
    require(manifest.get('research_qualified') is False and manifest.get('execution_authorized') is False and
            manifest['signal_session'] == spec['folds'][0]['test_end'], 'INVALID_MARKET_MANIFEST')
    pins = {}
    require(3 <= len(manifest['bindings']) <= 500010, 'MARKET_MANIFEST_SIZE')
    for pin in manifest['bindings']:
        relative = pin['path']
        require(isinstance(relative, str) and relative not in pins and relative.startswith('analysis/') and
                not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'MARKET_MANIFEST_PATH')
        pins[relative] = pin
    # First verify small coverage before trusting the full population.
    member = bounded_read(path.parent/'coverage.json',16*1024*1024)
    require(binding(member) == {k:pins['analysis/coverage.json'][k] for k in ('bytes', 'sha256')}, 'COVERAGE_CHANGED')
    coverage = strict_json(member)
    eligible = spec['universe']['symbols']
    require(coverage['status'] == 'COMPLETE' and coverage['signal_session'] == manifest['signal_session'] and
        coverage['evaluated_symbols'] == eligible and eligible == sorted(set(eligible)) and
        set(manifest['shards']) == set(eligible) | {spec['universe']['benchmark']} and
        len(set(manifest['shards'].values())) == len(manifest['shards']) and
        manifest['selection']['coverage_sha256'] == binding(member)['sha256'] and
        manifest['selection']['symbols_sha256'] == canonical_hash(eligible), 'MARKET_UNIVERSE_CHANGED')
    assets=coverage['assets']; names=[r['symbol'] for r in assets]
    excluded={r['symbol']:r['reason'] for r in assets if r['reason'] not in {'EVALUATED','BENCHMARK_ONLY'}}
    require(names==sorted(set(names)) and len(assets)==coverage['discovered_count'] and
        eligible==[r['symbol'] for r in assets if r['reason']=='EVALUATED'] and
        coverage['exclusions']==excluded and len(excluded)==coverage['excluded_count'] and
        coverage['evaluated_count']==len(eligible) and
        [r['symbol'] for r in assets if r['reason']=='BENCHMARK_ONLY']==[spec['universe']['benchmark']] and
        len(assets)==len(eligible)+len(excluded)+1,'INCOMPLETE_MARKET_ACCOUNTING')
    dataset = MarketDataset(spec, manifest, path, pins, coverage)
    for relative in pins:
        dataset.read(relative)
    require(spec['actions_capture']['sha256'] == binding(bounded_read(spec['actions_capture']['path']))['sha256'],
            'MARKET_ACTION_MANIFEST_CHANGED')
    return dataset
