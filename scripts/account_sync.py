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
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_LIMIT = 2 * 1024 * 1024


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    args: tuple[str, ...]
    required: tuple[str, ...]
    failures: tuple[str, ...]


STAGES = (
    Stage('activities', 'scripts.fetch_account_activities', ('--lookback-days', '30'),
          ('ACT_START', 'ACT_FETCH_OK', 'ACT_DB_OK'), ('ACT_FAIL', 'ACT_WATERMARK_FAIL')),
    Stage('reconciliation', 'scripts.execute_trades',
          ('--reconcile-only', 'true', '--dry-run', 'true', '--reconcile-use-watermark',
           'true', '--reconcile-lookback-days', '14', '--reconcile-limit', '500',
           '--reconcile-overlap-secs', '300', '--max-poll-secs', '1'),
          ('RECONCILE_START', 'RECONCILE_END', 'RECONCILE_WATERMARK_UPDATE'),
          ('RECONCILE_DB_FAIL', 'RECONCILE_ALPACA_FAIL', 'RECONCILE_WATERMARK_DISABLED',
           'AUTH_FAIL', 'ALPACA_UNAUTHORIZED')),
    Stage('snapshot', 'scripts.fetch_account_snapshot', (),
          ('ACCT_SNAP_START', 'ACCT_SNAP_FETCH_OK', 'ACCT_SNAP_DB_OK'), ('ACCT_SNAP_FAIL',)),
)
GENERIC_FAILURES = ('[ERROR]', 'Traceback (most recent call last):')


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


def run_stage(stage: Stage, timeout: float, lock_fd: int) -> dict:
    """Bound local runtime/output. Never expose arbitrary child output or secrets."""
    command = [sys.executable, '-m', stage.module, *stage.args]
    options = {'cwd': ROOT, 'stderr': subprocess.STDOUT, 'shell': False}
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
    qualifications = []
    if stage.name == 'reconciliation' and 'RECONCILE_DECORATE_MISS' in text:
        qualifications.append('RECONCILE_DECORATE_MISS')
    status = 'failed' if reason or process.returncode != 0 or failures or missing else 'ok'
    if status == 'ok' and qualifications:
        status = 'degraded'
    return {
        'status': status, 'exit_code': process.returncode, 'reason': reason,
        'elapsed_seconds': time.monotonic() - started, 'output_bytes': size,
        'failure_markers': failures, 'missing_completion_markers': missing,
        'qualifications': qualifications,
    }


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
            for relative in ('scripts/account_sync.py', *(s.module.replace('.', '/') + '.py' for s in STAGES)):
                data = (ROOT / relative).read_bytes()
                report['source_bindings'][relative] = {
                    'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                }
            for index, stage in enumerate(STAGES):
                entry = report['stages'][index]
                entry['started_at'] = utc_now()
                try:
                    result = runner(stage, stage_timeout, lock_fd)
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
