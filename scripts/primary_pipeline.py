"""One primary pipeline invocation with ML preparation and run-bound postflight."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from scripts import pipeline_postflight

ROOT = Path(__file__).resolve().parents[1]
FIXED_ARGS = (
    '--steps', 'screener,backtest,metrics,labels,ranker_eval',
    '--use-champion', '--ml-health-guard', '--ml-health-guard-mode', 'warn',
    '--enrich-candidates-with-ranker', '--refresh-predictions-for-candidates', 'true',
    '--auto-refresh-features', 'true', '--auto-refresh-predictions', 'true', '--postflight-report',
    '--session-handoff',
)
FIXED_FLAGS = {arg for arg in FIXED_ARGS if arg.startswith('--')} | {'--allow-no-screener', '--handoff-invocation'}


def build_arguments(argv):
    # The legacy argparse parser accepts unique option abbreviations. Reject
    # those too, so --step cannot override the fixed --steps later in argv.
    if any(arg.startswith('--') and any(flag.startswith(arg.split('=', 1)[0])
           for flag in FIXED_FLAGS) for arg in argv):
        raise ValueError('fixed_primary_contract_override')
    # Retain ordinary primary caller arguments and JBR_* defaults. Do not add
    # the canary's SCREENER_ARGS / --limit 30 or a second screener invocation.
    return [*FIXED_ARGS, *argv]


def main(argv=None, *, pipeline_main=None, base_dir=ROOT):
    try:
        arguments = build_arguments(list(sys.argv[1:] if argv is None else argv))
    except ValueError:
        print(json.dumps({'status': 'failed', 'reason': 'fixed_primary_contract_override'}), flush=True)
        return 1
    os.environ.setdefault('JBR_STRICT_PREDICTIONS_META', 'true')
    os.environ.setdefault('JBR_STRICT_AUTO_REFRESH_PREDICTIONS', 'true')
    os.environ.setdefault('JBR_RANKER_PREDICT_TIMEOUT_SECS', '900')
    os.environ.setdefault('JBR_RANKER_EVAL_TIMEOUT_SECS', '900')
    started = datetime.now(timezone.utc)
    try:
        invocation = pipeline_postflight.begin_primary(base_dir, started)
        if pipeline_main is None:
            from scripts.run_pipeline import main as pipeline_main
        arguments.extend(['--handoff-invocation', invocation['id']])
        try:
            rc = pipeline_main(arguments)
        except SystemExit as completion:
            rc = completion.code
        if type(rc) is not int or rc != 0:
            print(json.dumps({'status': 'failed', 'reason': 'pipeline_unsuccessful'}), flush=True)
            return 1
    except Exception as failure:
        print(json.dumps({'status': 'failed', 'reason': 'pipeline_exception',
                          'error_type': type(failure).__name__}), flush=True)
        return 1
    result = pipeline_postflight.check_health(base_dir, not_before=started, require_session=True)
    if result['status'] == 'ok':
        try:
            pipeline_postflight.complete_primary(base_dir, invocation, result)
        except (OSError, ValueError):
            result = {'status': 'failed', 'failures': ['postflight_completion_record_failed']}
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
