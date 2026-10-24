"""Sequential account synchronization; fixed stages, no order execution entry point."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from scripts import activity_ingestion_checkpoint as ingestion_checkpoint


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_LIMIT = 2 * 1024 * 1024


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    args: tuple[str, ...]
    required: tuple[str, ...]
    failures: tuple[str, ...]
    workflow_id: str | None = None
    cutoff_utc: str | None = None


STAGES = (
    Stage('activities', 'scripts.fetch_account_activities', ('--lookback-days', '30'),
          ('ACT_START', 'ACT_COVERAGE_COMPLETE', 'ACT_FETCH_OK', 'ACT_DB_OK'),
          ('ACT_FAIL', 'ACT_WATERMARK_FAIL', 'ACT_PAGINATION_INCOMPLETE',
           'ACT_PAYLOAD_INVALID', 'ACT_REQUEST_FAILED', 'ACT_WATERMARK_INVALID', 'ACT_DB_READ_FAILED')),
    Stage('reconciliation', 'scripts.execute_trades',
          ('--reconcile-only', 'true', '--dry-run', 'true', '--submit-at-ny', '',
           '--ignore-market-gate', 'true', '--reconcile-use-watermark',
           'true', '--reconcile-lookback-days', '14', '--reconcile-limit', '500',
           '--reconcile-overlap-secs', '300', '--max-poll-secs', '1'),
          ('RECONCILE_START', 'RECONCILE_COVERAGE_COMPLETE', 'RECONCILE_END', 'RECONCILE_WATERMARK_UPDATE'),
          ('RECONCILE_DB_FAIL', 'RECONCILE_ALPACA_FAIL', 'RECONCILE_WATERMARK_DISABLED',
           'AUTH_FAIL', 'ALPACA_UNAUTHORIZED', 'RECONCILE_POSITIONS_UNAVAILABLE',
           'RECONCILE_ORDERS_UNAVAILABLE', 'RECONCILE_ORDERS_INVALID',
           'RECONCILE_ORDERS_INCOMPLETE', 'RECONCILE_WATERMARK_HELD', 'RECONCILE_DB_INCOMPLETE')),
    Stage('snapshot', 'scripts.fetch_account_snapshot', (),
          ('ACCT_SNAP_START', 'ACCT_SNAP_FETCH_OK', 'ACCT_SNAP_DB_OK'), ('ACCT_SNAP_FAIL',)),
)
GENERIC_FAILURES = ('[ERROR]', 'Traceback (most recent call last):')
INTEGRITY_REASONS = (
    'RECONCILE_POSITIONS_UNAVAILABLE', 'RECONCILE_ORDERS_INVALID',
    'RECONCILE_ORDERS_UNAVAILABLE', 'RECONCILE_ORDERS_INCOMPLETE',
    'ACT_PAGINATION_INCOMPLETE', 'ACT_PAYLOAD_INVALID', 'ACT_REQUEST_FAILED',
    'ACT_WATERMARK_INVALID', 'ACT_WATERMARK_FAIL', 'ACT_DB_READ_FAILED',
    'RECONCILE_DB_INCOMPLETE', 'RECONCILE_DB_FAIL',
    'RECONCILE_WATERMARK_DISABLED', 'RECONCILE_WATERMARK_HELD',
)


class OverlapError(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def workflow_lock(path: Path):
    """OS-owned lock; no stale PID-file deletion. Children inherit it on POSIX."""
    if path.is_symlink():
        raise ValueError('symlink_lock')
    with path.open('a+b') as handle:
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise OverlapError() from None
        elif os.name == 'nt':
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise OverlapError() from None
        else:
            raise RuntimeError('unsupported_lock_platform')
        try:
            yield handle.fileno()
        finally:
            # Closing releases the lock. On POSIX an inherited child descriptor
            # keeps it held if the parent unexpectedly exits before the child.
            pass


def _legacy_scoped_reconciliation_result(text: str) -> dict | None:
    """Recognize only the current complete operational-only publication."""
    if re.search(r'\bRECONCILE_(?:FILL_HELD|DECORATE_HELD|INCOMPLETE|AUTO_FAIL)\b', text):
        return None
    markers = ('RECONCILE_START', 'RECONCILE_OPERATIONAL_COMPLETE',
               'RECONCILE_WATERMARK_HELD', 'RECONCILE_END')
    selected = {}
    for marker in markers:
        token = r'\b' + marker + r'\b'
        if len(re.findall(token, text)) != 1:
            return None
        lines = [line for line in text.splitlines() if re.search(token, line)]
        if len(lines) != 1 or len(lines[0]) > 16384 or any(ord(c) < 32 for c in lines[0]):
            return None
        selected[marker] = lines[0]
    number = r'(?:0|[1-9][0-9]{0,18})'
    timestamp = (r'\[[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]) '
                 r'(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9],[0-9]{3}\] ')
    prefix = r'(?:' + timestamp + r'\[INFO\] )?\[INFO\] '
    warning_prefix = r'(?:' + timestamp + r'\[WARNING\] )?\[WARN\] '
    if not re.fullmatch(prefix + 'RECONCILE_START open_trades=' + number, selected[markers[0]]):
        return None
    match = re.fullmatch(prefix + r'RECONCILE_OPERATIONAL_COMPLETE scope=ACTIVE_RECONCILIATION '
        r'policy_revision=(' + number + r') excluded_count=(' + number +
        r') recovery_pending_case_ids=(\[.*\])', selected[markers[1]])
    if match is None:
        return None
    case = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    cases = match[3]
    if not re.fullmatch(r"\[(?:'" + case + r"'(?:, '" + case + r"')*)?\]", cases):
        return None
    case_ids = re.findall(case, cases)
    if len(case_ids) > 1024 or len(case_ids) != len(set(case_ids)):
        return None
    if not re.fullmatch(warning_prefix + r'RECONCILE_WATERMARK_HELD '
                       r'reason=independent_durable_activity_ingestion_unproved', selected[markers[2]]):
        return None
    if not re.fullmatch(prefix + r'RECONCILE_END closed=' + number + r' decorated=' + number +
        r' operational_complete=true recovery_complete=false accounting_complete=false '
        r'activity_ingestion_complete=false watermark_advanced=false', selected[markers[3]]):
        return None
    if re.search(r'\bRECONCILE_WATERMARK_UPDATE\b', text):
        return None
    positions = [text.index(selected[marker]) for marker in markers]
    if positions != sorted(positions):
        return None
    return dict(scope='ACTIVE_RECONCILIATION', policy_revision=int(match[1]),
                excluded_count=int(match[2]), recovery_pending_case_ids=case_ids,
                operational_complete=True, recovery_complete=False, accounting_complete=False,
                activity_ingestion_complete=False, watermark_advanced=False, snapshot_eligible=False,
                reason='INDEPENDENT_DURABLE_ACTIVITY_INGESTION_UNPROVED')


def _scoped_reconciliation_result(text: str, *, expected_workflow_id: str | None = None,
                                  expected_cutoff: str | None = None) -> dict | None:
    marker = 'RECONCILE_INGESTION_CHECKPOINT'
    count = len(re.findall(r'\b'+marker+r'\b', text))
    if count == 0:
        return _legacy_scoped_reconciliation_result(text)
    if count != 1 or expected_workflow_id is None:
        return None
    lines = [line for line in text.splitlines() if re.search(r'\b'+marker+r'\b',line)]
    if len(lines)!=1 or len(lines[0])>16384 or any(ord(c)<32 for c in lines[0]):
        return None
    prefix=(r'(?:\[[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]) '
            r'(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9],[0-9]{3}\] \[INFO\] )?\[INFO\] ')
    match=re.fullmatch(prefix+r'RECONCILE_INGESTION_CHECKPOINT (\{.*\})',lines[0])
    if match is None:
        return None
    try:
        proof=ingestion_checkpoint.decode_proof(match[1])
        if proof['checkpoint_id']!=ingestion_checkpoint.workflow(expected_workflow_id):return None
        if ingestion_checkpoint.utc(proof['query_until'])!=ingestion_checkpoint.bound_cutoff(
                expected_workflow_id,expected_cutoff,datetime.now(timezone.utc)):return None
    except (RuntimeError,ValueError,TypeError,KeyError):
        return None
    # The same strict legacy grammar validates all remaining fields and markers.
    # Require the original proof marker in sequence and no contradictory flags.
    positions=[text.find(x) for x in ('RECONCILE_OPERATIONAL_COMPLETE',marker,'RECONCILE_WATERMARK_HELD')]
    if min(positions)<0 or positions!=sorted(positions):return None
    old_reason='reason=independent_durable_activity_ingestion_unproved'
    new_reason='reason=durable_ingestion_proved_accounting_held'
    if text.count(new_reason)!=1 or text.count('activity_ingestion_complete=true')!=1:return None
    normalized=text.replace(lines[0]+'\n','').replace(new_reason,old_reason).replace('activity_ingestion_complete=true','activity_ingestion_complete=false')
    result=_legacy_scoped_reconciliation_result(normalized)
    if result is None:return None
    return dict(result,activity_ingestion_complete=True,ingestion_checkpoint=proof,
                reason='DURABLE_INGESTION_PROVED_ACCOUNTING_HELD')


def run_stage(stage: Stage, timeout: float, lock_fd: int) -> dict:
    """Bound local runtime/output. Never expose arbitrary child output or secrets."""
    command = [sys.executable, '-m', stage.module, *stage.args]
    options = {'cwd': ROOT, 'stderr': subprocess.STDOUT, 'shell': False}
    if stage.name in ('activities','reconciliation'):
        from scripts.primary_paper_pipeline import configure_paper_context

        # Declare PAPER only for this child after validating inherited settings.
        child_environment = os.environ.copy()
        configure_paper_context(child_environment)
        if stage.workflow_id is not None:
            ingestion_checkpoint.workflow(stage.workflow_id)
            child_environment[ingestion_checkpoint.ENV_WORKFLOW]=stage.workflow_id
            cutoff=ingestion_checkpoint.bound_cutoff(stage.workflow_id,stage.cutoff_utc,datetime.now(timezone.utc))
            child_environment[ingestion_checkpoint.ENV_CUTOFF]=cutoff.isoformat()
        options['env'] = child_environment
    if os.name == 'posix':
        options.update(start_new_session=True, pass_fds=(lock_fd,))
    reason = None
    started = time.monotonic()
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, stdout=output, **options)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    reason = 'timeout'
                    break
                if os.fstat(output.fileno()).st_size > OUTPUT_LIMIT:
                    reason = 'output_limit'
                    break
                time.sleep(0.05)
            if reason:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)
        except BaseException:
            # Cleanup concerns only this owned child/group. A remote broker or
            # database operation may still have completed; never retry it here.
            try:
                if process.poll() is None:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=5)
            except Exception:
                pass
            raise
        size = os.fstat(output.fileno()).st_size
        if size > OUTPUT_LIMIT:
            reason = reason or 'output_limit'
        output.seek(0)
        text = output.read(OUTPUT_LIMIT).decode('utf-8', errors='replace')
    failures = [marker for marker in (*stage.failures, *GENERIC_FAILURES) if marker in text]
    missing = [marker for marker in stage.required if marker not in text]
    health_reasons = [marker for marker in INTEGRITY_REASONS if marker in failures]
    if health_reasons:
        reason = reason or health_reasons[0]
    qualifications = []
    if stage.name == 'reconciliation' and 'RECONCILE_DECORATE_MISS' in text:
        qualifications.append('RECONCILE_DECORATE_MISS')
    status = 'failed' if reason or process.returncode != 0 or failures or missing else 'ok'
    if status == 'ok' and qualifications:
        status = 'degraded'
    scoped = None
    if stage.name == 'reconciliation':
        observed = _scoped_reconciliation_result(text, expected_workflow_id=stage.workflow_id,
                                                expected_cutoff=stage.cutoff_utc)
        if (reason == 'RECONCILE_WATERMARK_HELD' and process.returncode == 0
                and observed is not None and failures == ['RECONCILE_WATERMARK_HELD']
                and not qualifications and set(missing) <= {
                    'RECONCILE_COVERAGE_COMPLETE', 'RECONCILE_WATERMARK_UPDATE'}):
            status, reason, scoped = 'degraded', 'RECONCILE_RECOVERY_HELD', observed
            qualifications = ['RECONCILE_RECOVERY_HELD']
            # Preserve the hold as a health reason; it is not a fatal DB error.
            failures, missing = [], []
        elif status in ('ok', 'degraded'):
            status, reason = 'failed', 'RECONCILE_SCOPED_RESULT_UNPROVED'
    result = {
        'status': status, 'exit_code': process.returncode, 'reason': reason,
        'elapsed_seconds': time.monotonic() - started, 'output_bytes': size,
        'failure_markers': failures, 'missing_completion_markers': missing,
        'qualifications': qualifications,
        'health_reasons': health_reasons,
    }
    if scoped is not None:
        result['completion_contract'] = ('ACTIVE_RECONCILIATION_INGESTION_V2' if scoped.get('activity_ingestion_complete') is True else 'ACTIVE_RECONCILIATION_OPERATIONAL_V1')
        result['reconciliation'] = scoped
    return result


def run_workflow(state_dir: Path, stage_timeout: float = 300, *, runner=run_stage) -> dict:
    if not math.isfinite(stage_timeout) or not 1 <= stage_timeout <= 900:
        raise ValueError('invalid_stage_timeout')
    state_dir = Path(state_dir)
    for ancestor in (state_dir, *state_dir.parents):
        if ancestor.is_symlink():
            raise ValueError('symlink_state_directory')
    state_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'schema_version': 1, 'run_id': uuid.uuid4().hex,
        'started_at': utc_now(), 'status': 'failed', 'no_order_workflow': True,
        'stage_timeout_seconds': stage_timeout,
        'stages': [{'name': s.name, 'status': 'not_run'} for s in STAGES],
        'source_bindings': {},
    }
    try:
        with workflow_lock(state_dir / 'account-sync.lock') as lock_fd:
            for relative in ('scripts/account_sync.py', 'scripts/db.py', 'scripts/activity_ingestion_checkpoint.py',
                             *(s.module.replace('.', '/') + '.py' for s in STAGES)):
                data = (ROOT / relative).read_bytes()
                report['source_bindings'][relative] = {
                    'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                }
            for index, stage in enumerate(STAGES):
                entry = report['stages'][index]
                entry['started_at'] = utc_now()
                try:
                    scoped_stage=Stage(stage.name,stage.module,stage.args,stage.required,stage.failures,
                                       report['run_id'],report['started_at'])
                    result = runner(scoped_stage, stage_timeout, lock_fd)
                    required_fields = {'status', 'exit_code', 'reason', 'elapsed_seconds',
                                       'output_bytes', 'failure_markers',
                                       'missing_completion_markers', 'qualifications'}
                    if not isinstance(result, dict) or not required_fields <= result.keys():
                        raise ValueError('incomplete_stage_result')
                    entry.update(result)
                    if entry.get('status') not in ('ok', 'failed', 'degraded'):
                        entry.update(status='failed', reason='invalid_stage_result')
                    if entry['status'] == 'ok' and (
                        type(entry['exit_code']) is not int or entry['exit_code'] != 0
                        or entry['reason'] or entry['failure_markers']
                        or entry['missing_completion_markers'] or entry['qualifications']
                    ):
                        entry.update(status='failed', reason='contradictory_stage_result')
                except Exception as failure:
                    entry.update(status='failed', reason='stage_exception', error_type=type(failure).__name__)
                entry['finished_at'] = utc_now()
                if entry['status'] != 'ok':
                    report['status'] = entry['status']
                    break
            else:
                report['status'] = 'ok'
            report['finished_at'] = utc_now()
            save_report(state_dir, report, update_latest=True)
    except OverlapError:
        report.update(status='overlap', finished_at=utc_now())
        # Do not overwrite the active run's latest health with an overlapping run.
        save_report(state_dir, report, update_latest=False)
    return report


def save_report(state_dir: Path, report: dict, *, update_latest: bool) -> None:
    """Keep each run, atomically publish latest, and verify saved JSON bytes."""
    data = (json.dumps(report, indent=2, sort_keys=True) + '\n').encode('utf-8')
    history = state_dir / ('run-' + report['run_id'] + '.json')
    with history.open('xb') as handle:
        handle.write(data)
    if history.read_bytes() != data:
        raise OSError('history_readback_failed')
    if update_latest:
        latest = state_dir / 'latest.json'
        if latest.is_symlink():
            raise ValueError('symlink_latest_report')
        temporary = state_dir / ('latest-' + report['run_id'] + '.tmp')
        with temporary.open('xb') as handle:
            handle.write(data)
        os.replace(temporary, latest)
        if latest.read_bytes() != data:
            raise OSError('latest_readback_failed')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, default=ROOT / 'logs' / 'account_sync')
    parser.add_argument('--stage-timeout-seconds', type=float, default=300)
    args = parser.parse_args(argv)
    try:
        report = run_workflow(args.state_dir, args.stage_timeout_seconds)
    except Exception as failure:
        # Existing files retain primary stage results even if latest recording fails.
        print(json.dumps({'status': 'failed', 'reason': 'workflow_or_recording_failure',
                          'error_type': type(failure).__name__}), flush=True)
        return 1
    print(json.dumps({'run_id': report['run_id'], 'status': report['status'],
                      'stages': [{'name': s['name'], 'status': s['status']} for s in report['stages']]}), flush=True)
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
