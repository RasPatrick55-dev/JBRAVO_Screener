"""Filesystem-only qualification of forward production scores and later outcomes.

No network, database publication, model loading, training or order dispatch.
Human review is an externally SHA-bound input, not a boolean CLI override.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
import re
import uuid

import numpy as np
import pandas as pd

from scripts.production_score_capture import binding, PRODUCER_FILES

TARGET = 'label_5d_pos_300bp'
SCORE = 'score_5d'
RETURN = 'fwd_ret_5d'
MAX_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
POLICY = {'calibration_min_rows': 2000, 'psi_bins': 10, 'psi_warn': .10,
          'psi_alert': .25, 'calibration_bins': 10, 'calibration_ece_warn': .05,
          'calibration_ece_alert': .10, 'top_k': 25, 'cost_bps': 5.,
          'horizon_sessions': 5, 'rebalance_calendar_days': 5,
          'drift_columns': 'score_binary_target_and_all_captured_features',
          'drift_binning': 'fixed_probability_bins_and_baseline_feature_quantiles_v1',
          'recent_sharpe_warn': .30, 'recent_sharpe_alert': .00}


def utc(value):
    if not isinstance(value, str):
        raise ValueError('invalid_evidence_time')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('ambiguous_evidence_time')
    return parsed.astimezone(timezone.utc)


def day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('invalid_evidence_session')
    return date.fromisoformat(value)


def ordinary(file):
    for part in (file, *file.parents):
        if part.is_symlink() or getattr(part.lstat(), 'st_file_attributes', 0) & 0x400:
            raise ValueError('redirected_evidence_path')


def read(file, maximum=MAX_BYTES):
    ordinary(file)
    with file.open('rb') as handle:
        content = handle.read(maximum + 1)
    if not 0 < len(content) <= maximum:
        raise ValueError('evidence_size_rejected')
    return content


def strict_json(content):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate_evidence_json_key')
            result[key] = value
        return result
    return json.loads(content, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_evidence_json')))


def identity(value, maximum=MAX_BYTES):
    if (not isinstance(value, dict) or set(value) != {'bytes', 'sha256'}
            or type(value['bytes']) is not int or not 0 < value['bytes'] <= maximum
            or not isinstance(value['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', value['sha256'])):
        raise ValueError('invalid_evidence_binding')
    return value


class Reader:
    def __init__(self, root):
        self.root = Path(root).absolute()
        ordinary(self.root)
        self.total_bytes = 0

    def bound(self, item):
        if not isinstance(item, dict) or set(item) != {'path', 'bytes', 'sha256'}:
            raise ValueError('invalid_evidence_reference')
        path = item['path']
        if (not isinstance(path, str) or not re.fullmatch(r'[A-Za-z0-9_.\-/]{1,250}', path)
                or path.startswith('/') or any(part in ('', '.', '..') or part.startswith('.')
                                             for part in path.split('/')) or '\\' in path):
            raise ValueError('evidence_path_outside_packet')
        expected = identity({key: item[key] for key in ('bytes', 'sha256')})
        content = read(self.root / path)
        self.total_bytes += len(content)
        if self.total_bytes > MAX_TOTAL_BYTES:
            raise ValueError('evidence_total_size_rejected')
        if binding(content) != expected:
            raise ValueError('evidence_bytes_changed')
        return content


def approved(record, *, now):
    if (not isinstance(record, dict) or set(record) != {'approval_id', 'reviewer', 'approved_at_utc'}
            or any(not isinstance(record[key], str) or not record[key].strip() or len(record[key]) > 100
                   for key in ('approval_id', 'reviewer'))):
        raise ValueError('independent_review_required')
    when = utc(record['approved_at_utc'])
    if when > now:
        raise ValueError('future_review_rejected')
    return when


def model_identity(value):
    required = {'bytes', 'sha256', 'kind', 'score_column', 'target', 'feature_columns'}
    if (not isinstance(value, dict) or set(value) != required
            or value['kind'] != 'deployed_predictor_bundle_pickle_sha256_v1'
            or value['target'] != TARGET or value['score_column'] != SCORE):
        raise ValueError('production_model_identity_required')
    identity({key: value[key] for key in ('bytes', 'sha256')})
    names = value['feature_columns']
    reserved = {'symbol', 'timestamp', SCORE, TARGET, RETURN, 'capture_id', 'observed_at_utc',
                'signal_session', 'outcome_entry_close', 'outcome_exit_close', 'outcome_timestamp',
                'outcome_available_at_utc'}
    if (not isinstance(names, list) or not names or len(names) > 512 or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,100}', name)
                   or name in reserved or name.startswith(('fwd_ret_', 'label_')) for name in names)):
        raise ValueError('production_feature_identity_required')
    return value


def reviewed_window_bounds(windows, lengths):
    """Lengths are preregistered study inputs, not fixed monitor defaults."""
    names = ('baseline', 'recent')
    if (not isinstance(lengths, dict)
            or set(lengths) != {name + '_calendar_days' for name in names}
            or any(type(value) is not int or value < 1 for value in lengths.values())):
        raise ValueError('reviewed_window_lengths_required')
    expected = {name + suffix for name in names for suffix in ('_start', '_end')}
    if not isinstance(windows, dict) or set(windows) != expected:
        raise ValueError('explicit_disjoint_windows_required')
    bounds = {key: day(value) for key, value in windows.items()}
    if not bounds['baseline_start'] <= bounds['baseline_end'] < bounds['recent_start'] <= bounds['recent_end']:
        raise ValueError('overlapping_or_inverted_windows')
    for name in names:
        if (bounds[name + '_end'] - bounds[name + '_start']).days + 1 != lengths[name + '_calendar_days']:
            raise ValueError('reviewed_window_length_mismatch')
    return bounds


def validate_plan(plan, calendar, *, now):
    if plan.get('schema') != 'jbravo.production-evidence-plan.v1' or plan.get('policy') != POLICY:
        raise ValueError('unchanged_production_monitor_policy_required')
    approval_time = approved(plan.get('approval'), now=now)
    model_identity(plan.get('model_binding'))
    sources = plan.get('producer_sources')
    if not isinstance(sources, dict) or set(sources) != set(PRODUCER_FILES):
        raise ValueError('reviewed_producer_identity_required')
    for value in sources.values():
        identity(value)
    if plan.get('population_policy') != 'all_reviewed_symbols_every_session':
        raise ValueError('complete_reviewed_population_required')
    symbols = plan.get('symbols')
    if (not isinstance(symbols, list) or len(symbols) < POLICY['top_k'] or len(symbols) > 5000
            or len(set(symbols)) != len(symbols)
            or any(not isinstance(s, str) or not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,31}', s) for s in symbols)):
        raise ValueError('reviewed_symbol_population_required')
    bounds = reviewed_window_bounds(plan.get('windows'), plan.get('window_lengths'))
    if calendar.get('schema') != 'jbravo.market-session-calendar.v1' or calendar.get('venue') != 'XNYS':
        raise ValueError('bound_exchange_calendar_required')
    sessions = calendar.get('sessions')
    if not isinstance(sessions, list) or not 10 <= len(sessions) <= 1024:
        raise ValueError('calendar_population_rejected')
    result = []
    for item in sessions:
        if not isinstance(item, dict) or set(item) != {'session', 'open_utc', 'close_utc'}:
            raise ValueError('calendar_session_schema_rejected')
        session, opened, closed = day(item['session']), utc(item['open_utc']), utc(item['close_utc'])
        if opened.date() != session or closed.date() != session or opened >= closed:
            raise ValueError('calendar_session_time_inconsistent')
        if result and (session <= result[-1]['session'] or opened <= result[-1]['close']):
            raise ValueError('calendar_order_or_duplicate_rejected')
        result.append({'session': session, 'open': opened, 'close': closed})
    cohorts = {name: [x['session'] for x in result
                     if bounds[name + '_start'] <= x['session'] <= bounds[name + '_end']]
               for name in ('baseline', 'recent')}
    if any(not values for values in cohorts.values()):
        raise ValueError('calendar_window_missing')
    # Calendar completeness and market holidays are independent review claims;
    # this parser cannot attest an arbitrary supplied calendar is authoritative.
    if approval_time >= next(x['open'] for x in result if x['session'] == cohorts['baseline'][0]):
        raise ValueError('cohort_not_registered_before_scoring')
    indices = {x['session']: i for i, x in enumerate(result)}
    if any(indices[values[-1]] + POLICY['horizon_sessions'] >= len(result) for values in cohorts.values()):
        raise ValueError('calendar_outcome_horizon_missing')
    return bounds, result, cohorts


def read_csv(content):
    # Check literal header too: pandas otherwise silently renames duplicates.
    import csv
    columns = next(csv.reader(io.StringIO(content.decode('utf-8'))), [])
    if len(columns) != len(set(columns)):
        raise ValueError('duplicate_evidence_csv_column')
    return pd.read_csv(io.BytesIO(content), dtype={'symbol': str, 'timestamp': str})


def keyed(frame):
    if not {'symbol', 'timestamp'} <= set(frame):
        raise ValueError('missing_evidence_identity')
    frame = frame.copy()
    # Require aware original timestamps; pandas utc=True would accept naive data.
    frame['timestamp'] = frame['timestamp'].map(lambda x: utc(x).isoformat())
    if frame[['symbol', 'timestamp']].duplicated().any():
        raise ValueError('duplicate_symbol_timestamp')
    return frame.sort_values(['symbol', 'timestamp']).reset_index(drop=True)


def capture_rows(reader, references, plan, calendar, cohorts, now):
    if not isinstance(references, list) or len(references) > 512:
        raise ValueError('capture_population_rejected')
    required = set(cohorts['baseline'] + cohorts['recent'])
    sessions = {item['session']: i for i, item in enumerate(calendar)}
    seen, collected = {}, []
    for ref in references:
        raw = reader.bound(ref)
        manifest = strict_json(raw)
        cid = manifest.get('capture_id')
        if not isinstance(cid, str) or not re.fullmatch('[0-9a-f]{32}', cid):
            raise ValueError('capture_id_rejected')
        if ref['path'] != 'captures/' + cid + '/manifest.json':
            raise ValueError('capture_manifest_path_rejected')
        if manifest.get('schema') != 'jbravo.production-score-observation.v2':
            raise ValueError('unbound_legacy_score_history')
        if (manifest.get('model_binding') != plan['model_binding']
                or manifest.get('producer_sources') != plan['producer_sources']):
            raise ValueError('noncomparable_model_or_producer')
        if (manifest.get('feature_representation') != 'normalized_imputed_predictor_input'
                or any(manifest.get(key) is not False for key in
                       ('outcomes_attached', 'research_qualified', 'operating_acceptance'))):
            raise ValueError('capture_disposition_rejected')
        session = day(manifest['signal_session'])
        if session not in required or session in seen:
            raise ValueError('duplicate_or_outside_cohort_session')
        observed = utc(manifest['observed_at_utc'])
        index = sessions[session]
        # After the signal close, before the very next trading open; no rescored
        # history or observations that already saw subsequent-session trading.
        if not calendar[index]['close'] <= observed < calendar[index + 1]['open'] or observed > now:
            raise ValueError('score_observed_after_future_market_data')
        frames = {}
        for kind in ('features', 'predictions'):
            frames[kind] = keyed(read_csv(reader.bound({
                'path': 'captures/' + cid + '/' + kind + '.csv', **identity(manifest[kind])})))
        features, predictions = frames['features'], frames['predictions']
        if (set(features) != {'symbol', 'timestamp', *plan['model_binding']['feature_columns']}
                or set(predictions) != {'symbol', 'timestamp', SCORE}
                or type(manifest.get('rows')) is not int or manifest['rows'] != len(features)
                or len(features) != len(plan['symbols'])
                or set(features['symbol']) != set(plan['symbols'])
                or set(predictions['symbol']) != set(plan['symbols'])
                or not features[['symbol', 'timestamp']].equals(predictions[['symbol', 'timestamp']])
                or pd.to_datetime(features['timestamp'], utc=True).dt.date.ne(session).any()):
            raise ValueError('incomplete_or_changed_capture_population')
        values = features[plan['model_binding']['feature_columns']].apply(pd.to_numeric, errors='raise')
        probabilities = pd.to_numeric(predictions[SCORE], errors='raise')
        if (not np.isfinite(values.to_numpy()).all() or not np.isfinite(probabilities).all()
                or not probabilities.between(0, 1).all()):
            raise ValueError('invalid_capture_values')
        if pd.to_datetime(features['timestamp'], utc=True).gt(pd.Timestamp(observed)).any():
            raise ValueError('future_capture_input')
        joined = predictions.merge(features, on=['symbol', 'timestamp'], validate='one_to_one')
        joined['capture_id'] = cid
        joined['observed_at_utc'] = observed.isoformat()
        joined['signal_session'] = session.isoformat()
        seen[session] = ref
        collected.append(joined)
    missing = sorted(required - set(seen))
    return (pd.concat(collected, ignore_index=True) if collected else pd.DataFrame(), missing)


def mature_rows(rows, bars, calendar, *, observed_at, cutoff, now):
    if cutoff > now or observed_at > now or observed_at < cutoff:
        raise ValueError('outcome_snapshot_time_ineligible')
    bars = keyed(bars)
    if pd.to_datetime(bars['timestamp'], utc=True).gt(pd.Timestamp(observed_at)).any():
        raise ValueError('future_outcome_bar_timestamp')
    if 'close' not in bars:
        raise ValueError('outcome_close_missing')
    bars['close'] = pd.to_numeric(bars['close'], errors='raise')
    if not np.isfinite(bars['close']).all() or not bars['close'].gt(0).all():
        raise ValueError('outcome_close_ineligible')
    bars['session'] = pd.to_datetime(bars['timestamp'], utc=True).dt.date
    if bars[['symbol', 'session']].duplicated().any():
        raise ValueError('duplicate_daily_outcome_bar')
    index = {item['session']: i for i, item in enumerate(calendar)}
    if not set(bars['session']) <= set(index):
        raise ValueError('bar_outside_reviewed_calendar')
    if any(calendar[index[session]]['close'] > observed_at for session in set(bars['session'])):
        raise ValueError('future_outcome_bar_in_snapshot')
    known = bars.set_index(['symbol', 'session'])
    eligible, missing = [], []
    if rows.empty:
        return pd.DataFrame(), []
    for signal_session, group in rows.groupby('signal_session', sort=True):
        session = day(signal_session)
        start = index[session]
        end = start + POLICY['horizon_sessions']
        horizon = calendar[start:end + 1]
        if horizon[-1]['close'] > cutoff or any(horizon[-1]['close'] <= utc(x) for x in group['observed_at_utc']):
            missing.extend({'capture_id': row['capture_id'], 'symbol': row['symbol'], 'reason': 'outcome_not_mature'}
                           for row in group.to_dict('records'))
            continue
        keys = pd.MultiIndex.from_product([group['symbol'], [item['session'] for item in horizon]],
                                         names=['symbol', 'session'])
        coverage = known.reindex(keys)['close'].notna().groupby(level='symbol').all()
        complete = group['symbol'].map(coverage).fillna(False)
        missing.extend({'capture_id': row['capture_id'], 'symbol': row['symbol'], 'reason': 'outcome_bar_gap'}
                       for row in group.loc[~complete].to_dict('records'))
        group = group.loc[complete].copy()
        if group.empty:
            continue
        entry_keys = pd.MultiIndex.from_arrays([group['symbol'], [session] * len(group)])
        exit_keys = pd.MultiIndex.from_arrays([group['symbol'], [horizon[-1]['session']] * len(group)])
        entries, outcomes = known.reindex(entry_keys), known.reindex(exit_keys)
        if not np.array_equal(entries['timestamp'].to_numpy(), group['timestamp'].to_numpy()):
            raise ValueError('capture_and_entry_bar_timestamp_mismatch')
        entry_price, exit_price = entries['close'].to_numpy(), outcomes['close'].to_numpy()
        forward = exit_price / entry_price - 1
        if not np.isfinite(forward).all():
            raise ValueError('nonfinite_joined_outcome')
        # Preserve captured inputs even if a feature happens to be named close.
        eligible.append(group.assign(outcome_entry_close=entry_price, outcome_exit_close=exit_price,
                         outcome_timestamp=outcomes['timestamp'].to_numpy(),
                         outcome_available_at_utc=horizon[-1]['close'].isoformat(),
                         **{RETURN: forward, TARGET: (forward >= .03).astype(int)}))
    return pd.concat(eligible, ignore_index=True) if eligible else pd.DataFrame(), missing


def production_psi(baseline, recent, *, probability=False):
    """Fixed probability bins; baseline-only feature bins, including constants."""
    ref, cur = np.asarray(baseline, dtype=float), np.asarray(recent, dtype=float)
    if not len(ref) or not len(cur) or not np.isfinite(ref).all() or not np.isfinite(cur).all():
        return {'psi': None, 'level': 'insufficient', 'bins_used': 0}
    if probability:
        if (ref.min() < 0 or cur.min() < 0 or ref.max() > 1 or cur.max() > 1):
            raise ValueError('invalid_probability_drift_input')
        edges = np.linspace(0, 1, POLICY['psi_bins'] + 1)
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, POLICY['psi_bins'] + 1)))
        if len(edges) == 1:
            value = float(edges[0])
            edges = np.array([-np.inf, np.nextafter(value, -np.inf),
                              np.nextafter(value, np.inf), np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / len(ref), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(cur), 1e-6, None)
    ref_pct, cur_pct = ref_pct / ref_pct.sum(), cur_pct / cur_pct.sum()
    value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return {'psi': value, 'level': 'alert' if value >= POLICY['psi_alert'] else
            'warn' if value >= POLICY['psi_warn'] else 'stable', 'bins_used': len(edges) - 1,
            'ref_count': len(ref), 'cur_count': len(cur),
            'bin_edges': [float(x) if np.isfinite(x) else '-inf' if x < 0 else 'inf' for x in edges],
            'baseline_bin_counts': ref_counts.tolist(), 'recent_bin_counts': cur_counts.tolist(),
            'smoothing_epsilon': 1e-6}


def health(rows, bounds):
    # Imported only after external review and complete qualification. These
    # existing numerical routines are deterministic and perform no I/O.
    from scripts import ranker_monitor as monitor
    rows = rows.copy()
    rows['timestamp'] = pd.to_datetime(rows['timestamp'], utc=True)
    windows = {}
    days = pd.to_datetime(rows['timestamp'], utc=True).dt.date
    for name in ('baseline', 'recent'):
        windows[name] = rows.loc[days.between(bounds[name + '_start'], bounds[name + '_end'])].copy()
    drift = {SCORE: production_psi(windows['baseline'][SCORE], windows['recent'][SCORE], probability=True),
             TARGET: monitor._compute_binary_psi(windows['baseline'][TARGET], windows['recent'][TARGET],
                                                warn=POLICY['psi_warn'], alert=POLICY['psi_alert'])}
    # Continuous normalized features retain their captured scale. No outcomes or
    # secondary score columns inherit the selected model identity.
    features = [name for name in rows if name not in {'symbol', 'timestamp', SCORE, RETURN, TARGET,
                'capture_id', 'observed_at_utc', 'signal_session', 'outcome_entry_close',
                'outcome_exit_close', 'outcome_timestamp', 'outcome_available_at_utc'}]
    for name in features:
        drift[name] = production_psi(windows['baseline'][name], windows['recent'][name])
    calibration = monitor._compute_calibration_drift(recent=windows['recent'], baseline=windows['baseline'],
            label_col=TARGET, score_col=SCORE, fwd_ret_col=RETURN, bins=POLICY['calibration_bins'],
            min_rows=POLICY['calibration_min_rows'])
    strategy = {name: monitor._run_strategy_metrics(frame, score_col=SCORE, fwd_ret_col=RETURN,
                top_k=POLICY['top_k'], rebalance_days=POLICY['rebalance_calendar_days'], cost_bps=POLICY['cost_bps'])
                for name, frame in windows.items()}
    reasons = []
    if not calibration['calibration_applicable']:
        reasons.append('insufficient_calibration_population')
    if any(item.get('psi') is None or item.get('bins_used', 2) < 2 for item in drift.values()):
        reasons.append('insufficient_drift_population_or_bins')
    for name, metrics in strategy.items():
        sharpe = metrics.get('sharpe')
        if (metrics.get('error') or not isinstance(sharpe, (int, float)) or not math.isfinite(sharpe)
                or metrics.get('periods', 0) < 2):
            reasons.append('insufficient_strategy_metrics:' + name)
    max_psi = max((item['psi'] for item in drift.values() if item.get('psi') is not None), default=None)
    if reasons:
        action, diagnostic_reasons = 'investigate', reasons
    else:
        action, diagnostic_reasons = monitor._recommend_action(max_psi=max_psi,
            recent_sharpe=strategy['recent']['sharpe'], calibration_applicable=True,
            recent_ece=calibration['calibration']['recent']['ece'], psi_warn=POLICY['psi_warn'],
            psi_alert=POLICY['psi_alert'], calibration_ece_warn=POLICY['calibration_ece_warn'],
            calibration_ece_alert=POLICY['calibration_ece_alert'],
            recent_sharpe_warn=POLICY['recent_sharpe_warn'], recent_sharpe_alert=POLICY['recent_sharpe_alert'])
    # Even valid evidence never authorizes automated production model operations.
    operational_action = 'none' if action == 'none' else 'investigate'
    return {'recommended_action': operational_action, 'diagnostic_recommended_action': action,
            'recommendation_reasons': diagnostic_reasons, 'drift': {'columns': drift, 'max_psi': max_psi},
            **calibration, 'baseline_strategy': strategy['baseline'], 'recent_strategy': strategy['recent']}


def evaluate(root, review_reference, *, now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('aware_evaluation_time_required')
    now = now.astimezone(timezone.utc)
    reader = Reader(root)
    review = strict_json(reader.bound(review_reference))
    if review.get('schema') != 'jbravo.production-evidence-review.v1':
        raise ValueError('production_evidence_review_required')
    reviewed_at = approved(review.get('approval'), now=now)
    claims = {'model_deployment_identity_reviewed', 'causal_capture_reviewed',
              'calendar_completeness_reviewed', 'population_comparability_reviewed',
              'split_adjusted_outcomes_reviewed'}
    if (not isinstance(review.get('attestations'), dict) or set(review['attestations']) != claims
            or any(review['attestations'][key] is not True for key in claims)):
        raise ValueError('independent_evidence_attestations_required')
    plan = strict_json(reader.bound(review['plan']))
    calendar = strict_json(reader.bound(plan['calendar']))
    bounds, sessions, cohorts = validate_plan(plan, calendar, now=now)
    if reviewed_at < utc(plan['approval']['approved_at_utc']):
        raise ValueError('review_predates_plan')
    rows, missing_sessions = capture_rows(reader, review['captures'], plan, sessions, cohorts, now)
    bars_observed = utc(review['bars_observed_at_utc'])
    cutoff = utc(review['outcome_cutoff_utc'])
    if reviewed_at < bars_observed or review.get('bars_adjustment') != 'split':
        raise ValueError('reviewed_split_adjusted_outcome_snapshot_required')
    bars = read_csv(reader.bound(review['bars']))
    mature, missing_outcomes = mature_rows(rows, bars, sessions, observed_at=bars_observed, cutoff=cutoff, now=now)
    counts = {name: int(mature.get('signal_session', pd.Series(dtype=str)).isin(
                       [day.isoformat() for day in days]).sum()) for name, days in cohorts.items()}
    reasons = []
    if missing_sessions:
        reasons.append('missing_production_score_sessions')
    if missing_outcomes:
        reasons.append('incomplete_mature_outcomes')
    if any(count < POLICY['calibration_min_rows'] for count in counts.values()):
        reasons.append('insufficient_comparable_rows')
    data_end = day(mature['signal_session'].max()) if not mature.empty else None
    signal = day(review['signal_session'])
    if signal > now.date() or signal < bounds['recent_end']:
        raise ValueError('monitor_signal_session_ineligible')
    if data_end is None or not 0 <= (signal - data_end).days <= 7:
        reasons.append('stale_or_missing_mature_input')
    diagnostics = health(mature, bounds) if not reasons else {'recommended_action': 'investigate',
                  'diagnostic_recommended_action': None, 'recommendation_reasons': reasons}
    qualified = not reasons
    coverage = {'dataset_start': mature['signal_session'].min() if not mature.empty else None,
                'dataset_end': data_end.isoformat() if data_end else None,
                **plan['windows'], 'baseline_rows': counts['baseline'], 'recent_rows': counts['recent'],
                'population_rows': int(len(mature))}
    report = {'schema': 'jbravo.production-evidence-result.v1', 'monitor_contract_version': 3,
              'run_utc': now.isoformat(), 'run_date': signal.isoformat(), 'target': TARGET,
              'data_source': 'reviewed://production_score_observations', 'windows': coverage,
              'model_binding': plan['model_binding'], 'producer_sources': plan['producer_sources'],
              'policy': POLICY, 'window_lengths': dict(plan['window_lengths']),
              'review_binding': {key: review_reference[key] for key in ('bytes', 'sha256')},
              'plan_binding': {key: review['plan'][key] for key in ('bytes', 'sha256')},
              'calendar_binding': {key: plan['calendar'][key] for key in ('bytes', 'sha256')},
              'bars_binding': {key: review['bars'][key] for key in ('bytes', 'sha256')},
              'capture_manifest_bindings': review['captures'],
              'model_comparison': {'scope': 'reviewed_fixed_production_model',
                  'production_drift_established': qualified, 'score_col': SCORE,
                  'common_model_sha256': [plan['model_binding']['sha256']] if qualified else [],
                  'weighting': 'one_symbol_session_no_silent_deduplication'},
              'qualified_for_monitoring': qualified, 'missing_capture_sessions': [x.isoformat() for x in missing_sessions],
              'incomplete_outcome_rows': len(missing_outcomes), 'outcome_exclusion_reasons':
                  {reason: sum(item['reason'] == reason for item in missing_outcomes)
                   for reason in ('outcome_not_mature', 'outcome_bar_gap')},
              'monitor_candidate_eligible': qualified and diagnostics['recommended_action'] == 'none',
              'publication_approved': False, 'operating_acceptance': False, 'production_model_changed': False,
              **diagnostics}
    return report, mature


def save_result(output_root, report, rows):
    output_root = Path(output_root).absolute()
    # Parents must already exist. Never follow a symlink or replace latest.json.
    ordinary(output_root)
    packet = output_root / uuid.uuid4().hex
    packet.mkdir()
    def save(name, content):
        if not 0 < len(content) <= MAX_OUTPUT_BYTES:
            raise ValueError('qualified_output_size_rejected')
        file = packet / name
        with file.open('xb') as handle:
            handle.write(content)
            handle.flush()
            import os
            os.fsync(handle.fileno())
        if read(file, MAX_OUTPUT_BYTES) != content:
            raise OSError('qualified_output_readback_failed')
        return binding(content)
    report = {**report, 'qualified_rows_binding': save('qualified_rows.csv', rows.to_csv(index=False).encode())}
    saved = save('monitor_candidate.json', (json.dumps(report, sort_keys=True, allow_nan=False) + '\n').encode())
    return {'packet': str(packet), 'monitor_binding': saved, 'monitor_candidate_eligible':
            report['monitor_candidate_eligible'], 'operating_acceptance': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet-root', required=True, type=Path)
    parser.add_argument('--review-path', required=True)
    parser.add_argument('--review-bytes', required=True, type=int)
    parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report, rows = evaluate(args.packet_root, {'path': args.review_path,
            'bytes': args.review_bytes, 'sha256': args.review_sha256})
        saved = save_result(args.output_root, report, rows)
        print(json.dumps({**saved, 'recommendation': report['recommended_action'],
                          'reasons': report['recommendation_reasons']}, allow_nan=False))
        return 0 if report['monitor_candidate_eligible'] else 1
    except Exception:
        # Paths, arbitrary reviewer strings and third-party file data stay private.
        print(json.dumps({'status': 'evidence_rejected', 'operating_acceptance': False}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
