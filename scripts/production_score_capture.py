"""Append-only observations of an actual predictor invocation; no acceptance.

Only the current input session is captured. Historical rescoring is never
retroactively presented as production history. No model is loaded or trained.
"""
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import uuid

import pandas as pd

MAX_BYTES = 64 * 1024 * 1024
PRODUCER_FILES = ('scripts/ranker_predict.py', 'scripts/production_score_capture.py',
                  'scripts/primary_pipeline.py')


def binding(content):
    return {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


def producer_binding():
    """Bind ordinary files from the running producer checkout, not output paths."""
    root = Path(__file__).resolve().parents[1]
    result = {}
    for relative in PRODUCER_FILES:
        file = root / relative
        if any(part.is_symlink() for part in (file, *file.parents)):
            raise ValueError('producer_symlink_rejected')
        with file.open('rb') as handle:
            content = handle.read(MAX_BYTES + 1)
        if not 0 < len(content) <= MAX_BYTES:
            raise ValueError('producer_size_rejected')
        result[relative] = binding(content)
    return result


def scoring_binding(model, scaler, feature_columns, target, calibrator):
    # Serialize trusted, already-loaded objects only. Never unpickle an input.
    content = pickle.dumps({'model': model, 'scaler': scaler, 'features': list(feature_columns),
        'target': target, 'score_column': 'score_5d', 'posthoc_calibrator': calibrator}, protocol=4)
    if not 0 < len(content) <= MAX_BYTES:
        raise ValueError('scoring_bundle_size_rejected')
    return {**binding(content), 'kind': 'deployed_predictor_bundle_pickle_sha256_v1',
            'score_column': 'score_5d', 'target': target, 'feature_columns': list(feature_columns)}


def save(path, content):
    if not 0 < len(content) <= MAX_BYTES:
        raise ValueError('capture_size_rejected')
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('capture_symlink_rejected')
    with path.open('xb') as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    if path.read_bytes() != content:
        raise OSError('capture_readback_failed')
    return binding(content)


def capture(base_dir, features, predictions, *, snapshot_date, model_binding, generated_at=None,
            producer_sources=None):
    day = date.fromisoformat(str(snapshot_date))
    now = generated_at or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None or day > now.astimezone(timezone.utc).date():
        raise ValueError('capture_time_ineligible')
    names = model_binding['feature_columns']
    stamps = pd.to_datetime(features['timestamp'], utc=True, errors='raise')
    # Freeze only rows observed by THIS invocation for its input session.
    current = features.loc[stamps.dt.date.eq(day), ['symbol', 'timestamp', *names]].copy()
    scored_times = pd.to_datetime(predictions['timestamp'], utc=True, errors='raise')
    scores = predictions.loc[scored_times.dt.date.eq(day), ['symbol', 'timestamp', 'score_5d']].copy()
    if current.empty or scores.empty or len(current) > 250000 or len(scores) != len(current):
        raise ValueError('capture_population_ineligible')
    if pd.to_datetime(current['timestamp'], utc=True).gt(pd.Timestamp(now)).any():
        raise ValueError('capture_future_feature_timestamp')
    for frame in (current, scores):
        if any(not isinstance(value, str) or not value.strip() or len(value) > 32
               for value in frame['symbol']):
            raise ValueError('capture_symbol_ineligible')
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True).map(lambda value: value.isoformat())
        if frame[['symbol', 'timestamp']].duplicated().any():
            raise ValueError('capture_duplicate_identity')
        frame.sort_values(['symbol', 'timestamp'], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not current[['symbol', 'timestamp']].equals(scores[['symbol', 'timestamp']]):
        raise ValueError('capture_score_identity_mismatch')
    if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in scores['score_5d']):
        raise ValueError('capture_score_ineligible')
    if any(not math.isfinite(float(x)) for x in current[names].to_numpy().ravel()):
        raise ValueError('capture_features_ineligible')
    source_identity = producer_binding()
    if producer_sources is not None and producer_sources != source_identity:
        raise ValueError('producer_changed_during_prediction')
    directory = Path(base_dir) / 'data' / 'production_score_history'
    for part in (directory, *directory.parents):
        if part.is_symlink():
            raise ValueError('capture_symlink_rejected')
    directory.mkdir(parents=True, exist_ok=True)
    packet = directory / uuid.uuid4().hex
    packet.mkdir()
    manifest = {'schema': 'jbravo.production-score-observation.v2', 'capture_id': packet.name,
        'observed_at_utc': now.astimezone(timezone.utc).isoformat(), 'signal_session': day.isoformat(),
        'rows': len(scores), 'model_binding': model_binding,
        'producer_sources': source_identity,
        'feature_representation': 'normalized_imputed_predictor_input',
        'features': save(packet / 'features.csv', current.to_csv(index=False).encode()),
        'predictions': save(packet / 'predictions.csv', scores.to_csv(index=False).encode()),
        'outcomes_attached': False, 'research_qualified': False, 'operating_acceptance': False,
        'qualification': 'awaiting_independent_model_population_and_mature_outcome_review'}
    identity = save(packet / 'manifest.json', (json.dumps(manifest, sort_keys=True, allow_nan=False) + '\n').encode())
    return {'manifest_path': str(packet / 'manifest.json'), 'manifest_binding': identity,
            'rows': len(scores), 'signal_session': day.isoformat(), 'model_binding': model_binding,
            'producer_sources': source_identity,
            'research_qualified': False, 'operating_acceptance': False}
