"""Run-bound pipeline health recording and an inert, filesystem-only checker.

The producer is called by the primary pipeline. The checker never imports the
pipeline, loads authentication, queries a database or dispatches work.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_STEPS = ('screener', 'backtest', 'metrics', 'labels', 'ranker_eval')
SOURCE_FILES = ('scripts/run_pipeline.py', 'scripts/pipeline_postflight.py',
                'scripts/primary_pipeline.py', 'scripts/ops/run_primary_pipeline.sh',
                'scripts/ops/run_canary_smoke.sh', 'scripts/utils/ml_health_guard.py')
MAX_REPORT_BYTES = 64 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
ML_PROVENANCE_FIELDS = ('monitor_run_date', 'monitor_run_date_source', 'monitor_artifact_run_date',
                        'monitor_generated_at', 'monitor_generation_age_seconds',
                        'monitor_input_coverage', 'monitor_input_age_days', 'monitor_input_source',
                        'monitor_observed_at', 'monitor_input_reference_date', 'max_age_days', 'source', 'mode',
                        'monitor_provenance_required', 'monitor_provenance_reasons')


def _utc(value):
    if not isinstance(value, str):
        raise ValueError('invalid_timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('ambiguous_timestamp')
    return parsed.astimezone(timezone.utc)


def _ordinary(path):
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('symlink_path')


def _read(path, limit):
    _ordinary(path)
    with path.open('rb') as handle:
        data = handle.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError('empty_or_oversize_input')
    return data


def _binding(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('nonfinite_json')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def assess(record, *, now, expected_day, max_age_seconds=7200, not_before=None):
    """Assess declared outcomes; success is not trading or research acceptance."""
    failures, warnings = [], []
    if not isinstance(record, dict) or type(record.get('schema_version')) is not int or record['schema_version'] != 1:
        raise ValueError('unsupported_schema')
    run_id = record.get('run_id')
    if not isinstance(run_id, str) or not re.fullmatch('[0-9a-f]{32}', run_id):
        raise ValueError('invalid_run_id')
    started, finished = _utc(record.get('started_at')), _utc(record.get('finished_at'))
    if started > finished or finished > now or (now - started).total_seconds() > max_age_seconds:
        failures.append('invalid_or_stale_run_window')
    if started.date() != expected_day or finished.date() != expected_day:
        failures.append('wrong_run_day')
    if not_before is not None and started < not_before:
        failures.append('report_predates_invocation')
    if type(record.get('pipeline_rc')) is not int or record['pipeline_rc'] != 0:
        failures.append('pipeline_failed_or_unknown')
    if record.get('degraded') is not False:
        warnings.append('pipeline_degraded_or_unknown')
    steps, stages = record.get('steps'), record.get('stages')
    if not isinstance(steps, list) or any(not isinstance(s, str) for s in steps) or len(set(steps)) != len(steps):
        raise ValueError('invalid_steps')
    if not isinstance(stages, dict):
        raise ValueError('invalid_stages')
    for step in REQUIRED_STEPS:
        outcome = stages.get(step)
        if step not in steps or not isinstance(outcome, dict):
            failures.append('missing_stage_' + step)
        elif type(outcome.get('rc')) is not int or outcome['rc'] != 0:
            failures.append('failed_stage_' + step)
        elif not _number(outcome.get('seconds')) or outcome['seconds'] <= 0:
            failures.append('skipped_or_unmeasured_stage_' + step)
    for step, outcome in stages.items():
        if not isinstance(outcome, dict) or type(outcome.get('rc')) is not int or outcome['rc'] != 0:
            failures.append('auxiliary_stage_failed_or_unknown')
    if type(record.get('labels_rows')) is not int or record['labels_rows'] <= 0:
        failures.append('labels_empty_or_unknown')
    predictions = record.get('predictions')
    if not isinstance(predictions, dict):
        failures.append('predictions_missing')
    else:
        if predictions.get('stale') is not False or predictions.get('compatible') is not True:
            failures.append('predictions_stale_incompatible_or_unknown')
        predict_rc = predictions.get('predict_rc')
        # None means compatible existing predictions were reused, not a failed call.
        if predict_rc is not None and (type(predict_rc) is not int or predict_rc != 0):
            failures.append('prediction_refresh_failed')
        try:
            snapshot_day = date.fromisoformat(predictions.get('snapshot_date'))
            if snapshot_day > started.date():
                failures.append('future_prediction_snapshot')
        except (TypeError, ValueError):
            failures.append('prediction_snapshot_unknown')
    controls = record.get('controls')
    if not isinstance(controls, dict) or any(controls.get(k) is not True for k in
            ('strict_predictions_meta', 'strict_auto_refresh_predictions', 'auto_refresh_features',
             'auto_refresh_predictions', 'use_champion', 'ml_health_guard', 'refresh_predictions_for_candidates')):
        failures.append('required_ml_controls_missing')
    health = record.get('ml_health')
    if not isinstance(health, dict) or health.get('decision') not in ('allow', 'warn', 'block'):
        warnings.append('ml_health_unknown')
    elif health['decision'] != 'allow':
        warnings.append('ml_health_' + health['decision'])
    if isinstance(health, dict):
        reasons = health.get('reasons')
        if not isinstance(reasons, list) or any(not isinstance(r, str) or not re.fullmatch('[a-z0-9_:-]{1,80}', r) for r in reasons):
            warnings.append('ml_health_reasons_unknown')
        elif reasons:
            warnings.extend('ml_health_reason_' + r for r in reasons)
    coverage = record.get('coverage')
    if not isinstance(coverage, dict):
        failures.append('coverage_missing')
    else:
        total, scored, pct = coverage.get('total'), coverage.get('non_null'), coverage.get('pct')
        if type(total) is not int or type(scored) is not int or total < 0 or not 0 <= scored <= total or not _number(pct):
            failures.append('coverage_invalid')
        elif not math.isclose(pct, 100.0 * scored / total if total else 0.0, abs_tol=1e-7):
            failures.append('coverage_inconsistent')
        elif total == 0:
            warnings.append('zero_candidates_no_coverage')
        elif scored != total:
            warnings.append('incomplete_candidate_scores')
        if type(total) is int and total > 0:
            try:
                candidate_time = _utc(coverage.get('run_ts_utc'))
                if not started <= candidate_time <= finished:
                    failures.append('coverage_not_from_this_run')
            except (ValueError, TypeError):
                failures.append('coverage_run_unknown')
    return {'run_id': run_id, 'status': 'failed' if failures else 'degraded' if warnings else 'ok',
            'failures': sorted(set(failures)), 'qualifications': sorted(set(warnings)),
            'research_qualified': False, 'operating_acceptance': False}


def publish_health(base_dir, *, started_at, finished_at, pipeline_rc, steps, step_rcs,
                   stage_times, degraded, labels_rows, freshness, coverage, ml_health, controls):
    """The pipeline writes selected facts from this invocation, not shared-log greps."""
    base_dir = Path(base_dir)
    freshness = freshness if isinstance(freshness, dict) else {}
    compatible = freshness.get('pred_compatible')
    if isinstance(compatible, str) and compatible.lower() in ('true', 'false'):
        compatible = compatible.lower() == 'true'
    health = ml_health if isinstance(ml_health, dict) else {}
    reasons = health.get('reasons')
    if isinstance(reasons, list):
        reasons = [r if isinstance(r, str) and re.fullmatch('[a-z0-9_:-]{1,80}', r)
                   else 'unknown_ml_health_reason' for r in reasons]
    record = {
        'schema_version': 1, 'run_id': uuid.uuid4().hex,
        'started_at': started_at.isoformat(), 'finished_at': finished_at.isoformat(),
        'pipeline_rc': pipeline_rc, 'degraded': degraded, 'steps': list(steps),
        'stages': {name: {'rc': value, 'seconds': stage_times.get(name)} for name, value in step_rcs.items()},
        'labels_rows': labels_rows,
        'predictions': { 'stale': freshness.get('stale'), 'compatible': compatible,
                         'predict_rc': freshness.get('predict_rc'), 'snapshot_date': freshness.get('snapshot_date')},
        'coverage': {key: (coverage or {}).get(key) for key in ('total', 'non_null', 'pct', 'run_ts_utc')},
        'ml_health': {'decision': health.get('decision') if health.get('decision') in ('allow', 'warn', 'block') else None,
                      'reasons': reasons if isinstance(reasons, list) else None,
                      **{field: health[field] for field in ML_PROVENANCE_FIELDS if field in health}},
        'controls': dict(controls), 'source_bindings': {},
    }
    for relative in SOURCE_FILES:
        record['source_bindings'][relative] = _binding(_read(base_dir / relative, MAX_SOURCE_BYTES))
    # Ensure malformed facts cannot silently become successful JSON.
    assess(record, now=finished_at, expected_day=started_at.date())
    data = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError('oversize_report')
    directory = base_dir / 'reports' / 'pipeline_postflight'
    _ordinary(directory)
    directory.mkdir(parents=True, exist_ok=True)
    history = directory / ('run-' + record['run_id'] + '.json')
    with history.open('xb') as handle:
        handle.write(data)
    if _read(history, MAX_REPORT_BYTES) != data:
        raise OSError('history_readback_failed')
    latest = directory / 'latest.json'
    _ordinary(latest)
    temporary = directory / ('latest-' + record['run_id'] + '.tmp')
    with temporary.open('xb') as handle:
        handle.write(data)
    if _read(temporary, MAX_REPORT_BYTES) != data:
        raise OSError('temporary_readback_failed')
    os.replace(temporary, latest)
    if _read(latest, MAX_REPORT_BYTES) != data:
        raise OSError('latest_readback_failed')
    return record


def check_health(base_dir=ROOT, *, now=None, expected_day=None, max_age_seconds=7200, not_before=None):
    now = now or datetime.now(timezone.utc)
    expected_day = expected_day or now.date()
    if not _number(max_age_seconds) or not 0 < max_age_seconds <= 86400:
        raise ValueError('invalid_age_limit')
    base_dir = Path(base_dir)
    path = base_dir / 'reports' / 'pipeline_postflight' / 'latest.json'
    try:
        data = _read(path, MAX_REPORT_BYTES)
        record = json.loads(data.decode('utf-8'), object_pairs_hook=_strict_object, parse_constant=_invalid_constant)
        result = assess(record, now=now, expected_day=expected_day, max_age_seconds=max_age_seconds, not_before=not_before)
        for relative in SOURCE_FILES:
            if record.get('source_bindings', {}).get(relative) != _binding(_read(base_dir / relative, MAX_SOURCE_BYTES)):
                raise ValueError('source_binding_mismatch')
        history = path.parent / ('run-' + record['run_id'] + '.json')
        if _read(history, MAX_REPORT_BYTES) != data:
            raise ValueError('history_binding_mismatch')
        result['report_binding'] = _binding(data)
        # Display provenance only after source/history saved-byte checks pass.
        result['ml_health'] = record['ml_health']
        return result
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {'status': 'failed', 'failures': ['missing_malformed_or_unbound_report'],
                'qualifications': [], 'research_qualified': False, 'operating_acceptance': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only primary pipeline postflight; no repair or refresh')
    parser.add_argument('--expected-run-date', type=date.fromisoformat)
    parser.add_argument('--max-age-seconds', type=float, default=7200)
    args = parser.parse_args(argv)
    try:
        result = check_health(expected_day=args.expected_run_date, max_age_seconds=args.max_age_seconds)
    except ValueError:
        result = {'status': 'failed', 'failures': ['invalid_check_settings']}
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
