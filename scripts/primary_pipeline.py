"""One primary pipeline invocation with ML preparation and run-bound postflight."""
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import sys
import uuid

from scripts import pipeline_postflight

ROOT = Path(__file__).resolve().parents[1]
FIXED_ARGS = (
    '--steps', 'screener,backtest,metrics,labels,ranker_eval',
    '--use-champion', '--ml-health-guard', '--ml-health-guard-mode', 'warn',
    '--enrich-candidates-with-ranker', '--refresh-predictions-for-candidates', 'true',
    '--auto-refresh-features', 'true', '--auto-refresh-predictions', 'true', '--postflight-report',
    '--session-handoff',
    '--pipeline-readiness-only',
    '--metrics-args=--paper-gross-pipeline',
)
FIXED_FLAGS = {arg for arg in FIXED_ARGS if arg.startswith('--')} | {'--allow-no-screener', '--handoff-invocation', '--metrics-args', '--metrics-args-split'}


def build_arguments(argv):
    # The legacy argparse parser accepts unique option abbreviations. Reject
    # those too, so --step cannot override the fixed --steps later in argv.
    if any(arg.startswith('--') and any(flag.startswith(arg.split('=', 1)[0])
           for flag in FIXED_FLAGS) for arg in argv):
        raise ValueError('fixed_primary_contract_override')
    # Retain ordinary primary caller arguments and JBR_* defaults. Do not add
    # the canary's SCREENER_ARGS / --limit 30 or a second screener invocation.
    return [*FIXED_ARGS, *argv]


def _main(argv=None, *, pipeline_main=None, base_dir=ROOT):
    try:
        arguments = build_arguments(list(sys.argv[1:] if argv is None else argv))
        provisional = pipeline_postflight.provisional_context()
        if provisional is not None:
            arguments.remove('--session-handoff')
    except ValueError:
        print(json.dumps({'status': 'failed', 'reason': 'fixed_primary_contract_override'}), flush=True)
        return 1
    os.environ.setdefault('JBR_STRICT_PREDICTIONS_META', 'true')
    os.environ.setdefault('JBR_CAPTURE_PRODUCTION_SCORES', 'true')
    os.environ.setdefault('JBR_STRICT_AUTO_REFRESH_PREDICTIONS', 'true')
    os.environ.setdefault('JBR_RANKER_PREDICT_TIMEOUT_SECS', '900')
    os.environ.setdefault('JBR_RANKER_EVAL_TIMEOUT_SECS', '900')
    started = datetime.now(timezone.utc)
    invocation = None
    phase = 'PIPELINE'

    def failed(reason, *, error=None, rc=None):
        result = dict(status='failed', reason=reason)
        if error is not None:
            # Do not print arbitrary exception classes or messages.
            result['error_type'] = 'Exception'
        if invocation is not None:
            try:
                result['primary_failure'] = pipeline_postflight.fail_primary(base_dir, invocation,
                    phase=phase, reason=reason, error=error, pipeline_rc=rc)
            except Exception:
                result['failure_receipt_status'] = 'FAILED_TO_RECORD'
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 1

    try:
        invocation = pipeline_postflight.begin_primary(base_dir, started)
        if provisional is not None:
            provisional['id'] = invocation['id']
            os.environ[pipeline_postflight.PROVISIONAL_ENV] = json.dumps(provisional, sort_keys=True)
        if pipeline_main is None:
            from scripts.run_pipeline import main as pipeline_main
        arguments.extend(['--handoff-invocation', invocation['id']])
        try:
            rc = pipeline_main(arguments)
        except SystemExit as completion:
            rc = completion.code
        if type(rc) is not int or rc != 0:
            return failed('pipeline_unsuccessful', rc=rc)
    except Exception as failure:
        return failed('pipeline_exception', error=failure)
    phase = 'POSTFLIGHT'
    try:
        from scripts import strategy_pipeline
        strategy_route = strategy_pipeline.enabled(base_dir)
        result = pipeline_postflight.check_health(base_dir, not_before=started,
            require_session=provisional is None, require_paper_gross=not strategy_route,
            allow_provisional=provisional is not None)
        if provisional is not None:
            passed = result['status'] == 'ok' and result.get('functional_pass') is True
            if not passed:
                return failed('postflight_failed')
            result.update(status='provisional_functional_pass',
                          qualified_daily=False, entry_ready=False, operating_acceptance=False)
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
            return 0
        if result['status'] != 'ok':
            return failed('postflight_failed')
        phase = 'COMPLETION'
        try:
            pipeline_postflight.complete_primary(base_dir, invocation, result)
        except (OSError, ValueError):
            return failed('postflight_completion_record_failed')
    except Exception as failure:
        return failed('postflight_exception', error=failure)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result['status'] == 'ok' else 1


def main(argv=None, *, pipeline_main=None, base_dir=ROOT):
    arguments = list(sys.argv[1:] if argv is None else argv)
    flag = '--manual-provisional-test'
    previous = os.environ.get(pipeline_postflight.PROVISIONAL_ENV)
    if flag not in arguments:
        if previous is not None:
            print(json.dumps({'status': 'failed', 'reason': 'provisional_requires_explicit_manual_flag'}), flush=True)
            return 1
        return _main(arguments, pipeline_main=pipeline_main, base_dir=base_dir)
    if (arguments != [flag] or previous is not None
            or os.environ.get('APCA_API_BASE_URL') != 'https://paper-api.alpaca.markets'
            or os.environ.get('JBR_EXEC_PAPER') != 'true'):
        print(json.dumps({'status': 'failed', 'reason': 'manual_provisional_context_refused'}), flush=True)
        return 1
    started = datetime.now(timezone.utc)
    context = dict(mode='MANUAL_PROVISIONAL_PIPELINE_V1', id=uuid.uuid4().hex,
        started_at=started.isoformat(), expires_at=(started + timedelta(minutes=22)).isoformat(),
        signal_session=None, qualified_daily=False, entry_ready=False)
    os.environ[pipeline_postflight.PROVISIONAL_ENV] = json.dumps(context, sort_keys=True)
    previous_handoff = os.environ.get('JBRAVO_SESSION_HANDOFF')
    try:
        return _main([], pipeline_main=pipeline_main, base_dir=base_dir)
    finally:
        os.environ.pop(pipeline_postflight.PROVISIONAL_ENV, None)
        if previous_handoff is None:
            os.environ.pop('JBRAVO_SESSION_HANDOFF', None)
        else:
            os.environ['JBRAVO_SESSION_HANDOFF'] = previous_handoff


if __name__ == '__main__':
    raise SystemExit(main())
