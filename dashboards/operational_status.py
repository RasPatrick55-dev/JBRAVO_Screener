"""Current immutable process receipts, separate from financial/entry acceptance."""
import ast
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def _base(source):
    return dict(ok=False, status='UNAVAILABLE', source=source, started_utc=None,
        finished_utc=None, duration_seconds=None, rc=None, stage_rcs={},
        evidence_kind='operational_diagnostic_artifacts',
        financial_readiness='not_established_by_process_receipts')


def _times(record, now):
    from scripts.executor_health import utc
    start, end = utc(record['started_at']), utc(record['finished_at'])
    if not start <= end <= now:
        raise ValueError('invalid_process_receipt_time')
    return dict(started_utc=start.isoformat(), finished_utc=end.isoformat(),
        duration_seconds=(end-start).total_seconds(),
        age_seconds=(now-end).total_seconds())


def pipeline_task(base_dir=ROOT, *, now=None):
    from scripts import pipeline_postflight as health, setup_selection as selection
    from scripts.strategy_dashboard import primary_receipt, failure_code
    now = now or datetime.now(timezone.utc)
    result = _base('primary_receipt')
    try:
        failure = health.current_primary_failure(base_dir)
        if failure is not None:
            return dict(result, ok=True, status='FAILED', run_id=failure['run_id'],
                rc=failure['pipeline_rc'], **_times(failure, now))
        primary, _ = primary_receipt(base_dir, now=now)
        control = json.loads(health._read(Path(base_dir) /
            'reports/strategy_companions/unified_setups/latest.json', 65536))
        if not (control.get('status') == 'COMPLETED_UNIFIED_SETUP_SELECTION' and
                control.get('run_id') == selection.selection_id(primary['run_id']) and
                control.get('primary_run_id') == primary['run_id'] and
                control.get('parent_report_sha256') == selection.digest(primary)):
            raise ValueError('strategy_postflight_unified_selection_unavailable')
        times = _times(primary, now)
        return dict(result, ok=True, status='COMPLETED' if times['age_seconds'] <= 86400 else 'STALE',
            run_id=primary['run_id'], rc=primary['pipeline_rc'], **times,
            stage_rcs=dict(screener=0, backtest=0, metrics=0))
    except Exception as error:
        return dict(result, failure_code=failure_code(error))


def _execution_receipt(base_dir=ROOT, *, now=None):
    """Inspect the actual premarket receipt; never prefer an older task log."""
    from scripts import executor_health as health
    now = now or datetime.now(timezone.utc)
    result = _base('executor_receipt')
    try:
        folder = Path(base_dir) / 'reports/executor/premarket'
        raw = health.read(folder / 'latest.json')
        record = health.decode(raw)
        if not (record.get('schema_version') == 1 and record.get('role') == 'premarket' and
                re.fullmatch('[0-9a-f]{32}', record.get('run_id', '')) and
                health.read(folder / ('run-' + record['run_id'] + '.json')) == raw and
                type(record.get('process_rc')) is int and
                record.get('research_qualified') is False and record.get('operating_acceptance') is False):
            raise ValueError('invalid_executor_receipt')
        outcome, pending, unknown = health.assess_outcome(record['process_rc'], record['metrics'],
            record['order_observations'], record['observations_truncated'], record.get('preflight'), 'premarket')
        if (record['health'], record['pending_terminal_confirmations'], record['unobserved_submissions']) != (outcome, pending, unknown):
            raise ValueError('inconsistent_executor_receipt')
        times = _times(record, now)
        state = 'FAILED' if record['process_rc'] != 0 or outcome == 'failed' else (
            'PENDING' if outcome == 'pending_confirmation' else 'COMPLETED')
        if times['age_seconds'] > 86400:
            state = 'STALE'
        return dict(result, ok=True, status=state, run_id=record['run_id'],
            rc=record['process_rc'], process_health=outcome, **times)
    except Exception:
        return dict(result, failure_code='executor_receipt_missing_or_invalid')


def health_snapshot(base_dir=ROOT):
    from scripts.strategy_dashboard import snapshot
    from dashboards.data_io import load_connection_health, connection_check_snapshot
    task = pipeline_task(base_dir)
    candidates = snapshot(base_dir=base_dir)
    connection = load_connection_health(Path(base_dir) / 'data')
    return dict(last_run_utc=task['finished_utc'], pipeline_rc=task['rc'],
        source='primary_receipt', latest_source='postgres:unified_strategy_screen',
        process_status=task['status'], pipeline_task=task,
        rows_final=candidates.get('rows_final'), selection_ready=candidates.get('selection_ready', False),
        entry_ready=candidates.get('entry_ready', False),
        entry_block_reason=candidates.get('entry_block_reason') or candidates.get('failure_code'),
        source_detail=candidates.get('source_detail'),
        freshness=dict(age_seconds=task.get('age_seconds'), freshness_level=(
            'green' if task['status'] == 'COMPLETED' else 'gray')),
        trading_ok=connection.get('trading_ok'), trading_status=connection.get('trading_status'),
        data_ok=connection.get('data_ok'), data_status=connection.get('data_status'),
        feed=connection.get('feed'), connection_check=connection_check_snapshot(connection),
        evidence_kind='operational_diagnostic_artifacts',
        financial_readiness='not_established_by_health_checks')


def scheduled_attempt(scheduled_log, *, now=None):
    """Sanitize the newest provider invocation, separate from broker receipts.

    A launcher refusal belongs to its historical invocation, not the release
    currently configured in the scheduler. Never emit raw logs or commands.
    """
    now = now or datetime.now(timezone.utc)
    result = dict(_base('pythonanywhere_schedule_log'), task_id=None,
        reason_code=None, orders_submitted=None, release_attribution='unverified_historical_attempt')
    try:
        text, source = scheduled_log
        # Require a specific provider task logfile, not generic pipeline output.
        matched = re.search(r'(?:schedule[-_/](?:log[-_])?|schedule-log-)([0-9]+)(?:[./:]|$)', str(source))
        if not matched:
            return dict(result, failure_code='scheduled_log_source_unverified')
        result['task_id'] = int(matched[1])
        lines = str(text)[-256 * 1024:].splitlines()
        starts = [i for i, line in enumerate(lines) if re.match(
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*Started task', line)]
        if not starts:
            return dict(result, failure_code='scheduled_invocation_start_missing')
        chunk = lines[starts[-1]:]
        def stamp(line):
            return datetime.fromisoformat(line[:19].replace(' ', 'T')).replace(tzinfo=timezone.utc)
        start = stamp(chunk[0])
        if start > now:
            raise ValueError('future')
        result.update(ok=True, status='RUNNING', started_utc=start.isoformat())
        for line in chunk[1:]:
            if 'Completed task' in line:
                end = stamp(line)
                rc = re.search(r'return code was\s+(-?\d+)', line)
                if end < start or end > now or rc is None:
                    raise ValueError('invalid_completion')
                result.update(status='COMPLETED' if int(rc[1]) == 0 else 'FAILED',
                    rc=int(rc[1]), finished_utc=end.isoformat(), duration_seconds=(end-start).total_seconds())
            # Only this structured launcher result proves no submissions.
            if '{' not in line:
                continue
            try:
                value = json.loads(line[line.index('{'):])
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and value.get('status') == 'SPLIT_EXECUTOR_LAUNCHER_REFUSED':
                reason = value.get('reason')
                if isinstance(reason, str) and re.fullmatch(
                        r'(?:executor|strategy|execution|postflight)_[a-z_]{1,90}', reason):
                    result['reason_code'] = reason
                else:
                    result['reason_code'] = 'input_or_source_gate'
                if type(value.get('orders_submitted')) is int and value['orders_submitted'] == 0:
                    result['orders_submitted'] = 0
        if result['reason_code'] and result['orders_submitted'] == 0:
            result['status'] = 'GUARDED_NO_ENTRY'
        return result
    except Exception:
        return dict(_base('pythonanywhere_schedule_log'), task_id=None, reason_code=None,
            orders_submitted=None, release_attribution='unverified_historical_attempt',
            failure_code='scheduled_log_missing_or_invalid')


def execute_task(base_dir=ROOT, *, now=None, scheduled_log=None):
    """Preserve the last immutable execution; disclose newer scheduler attempts."""
    now = now or datetime.now(timezone.utc)
    receipt = _execution_receipt(base_dir, now=now)
    return dict(receipt, last_execution_receipt=dict(receipt),
                last_scheduled_attempt=scheduled_attempt(scheduled_log, now=now))


def _literal(tree, name):
    values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(values) != 1:
        raise ValueError('execution_contract_literal_unavailable')
    return values[0]


def _source_bytes(path):
    # No imports, commands, configuration secrets or code execution.
    from scripts.executor_health import read
    return read(Path(path), 256 * 1024)


def _scheduled_launcher(task):
    """Accept only the explicit hash-checked scheduled Python launcher grammar."""
    if task.get('enabled') is not True:
        return None
    command = str(task.get('command') or '')
    try:
        tokens = shlex.split(command)
        pipe = tokens.index('|')
        if pipe != 3 or tokens[0] != 'printf' or tokens[1] not in ('%s' + chr(92) + 'n', '%s' + chr(92) * 2 + 'n'):
            return None
        if tokens[pipe + 1:pipe + 5] != ['/usr/bin/sha256sum', '--check', '--status', '&&']:
            return None
        tail = tokens[pipe + 5:]
        if tail[:3] == ['/usr/bin/timeout', '--signal=TERM', '--kill-after=10s']:
            if not re.fullmatch(r'[0-9]+s', tail[3]):
                return None
            tail = tail[4:]
        if len(tail) != 5 or tail[1:3] != ['-I', '-B'] or tail[-1] != '--scheduled':
            return None
        launcher = tail[3]
        if not re.fullmatch(r'/home/[A-Za-z0-9_-]+/jbravo-research-tools/[A-Za-z0-9_-]+/executor_boot\.py', launcher):
            return None
        if not re.fullmatch(r'/home/[A-Za-z0-9_-]+/\.virtualenvs/[A-Za-z0-9_-]+/bin/python(?:3(?:\.[0-9]+)?)?', tail[0]):
            return None
        pin = re.fullmatch(r'([0-9a-f]{64})  ' + re.escape(launcher), tokens[2])
        if pin is None:
            return None
        return launcher, pin[1], hashlib.sha256(command.encode()).hexdigest()
    except (ValueError, IndexError):
        return None


def execution_schedule(base_dir=ROOT, *, scheduled_tasks=None, now=None, source_reader=None):
    """Observe active scheduled source pins without importing the trading runtime.

    A current task command binds the isolated launcher and ownership support.
    Times come from that entry contract, never a hardcoded dashboard window or
    the legacy handoff's35-minute cancellation field. Unknown fails closed.
    """
    now = now or datetime.now(timezone.utc)
    result = dict(status='unknown', source='provider_task_and_pinned_entry_contract',
        observed_at=now.isoformat(), task_id=None, command_sha256=None,
        launcher_sha256=None, entry_source_sha256=None, lifecycle_source_sha256=None,
        timezone=None, entry_start_local=None, entry_end_local=None,
        submission_window_seconds=None, lifecycle_owner=None, cancellation_policy=None,
        cancellation_deadline=None, cancellation_deadline_status='unknown',
        legacy_cancel_after_min=None, legacy_cancel_after_min_is_deadline=False,
        entry_open_at=None, entry_close_at=None, in_window=None, session_date=None,
        session_status='unknown', reason='active_scheduled_contract_unverified')
    reader = source_reader or _source_bytes
    try:
        found = [(t, _scheduled_launcher(t)) for t in (scheduled_tasks or []) if isinstance(t, dict)]
        found = [(t, parsed) for t, parsed in found if parsed is not None]
        if len(found) != 1:
            return result
        task, (launcher, pin, command_hash) = found[0]
        result.update(task_id=task.get('id'), command_sha256=command_hash)
        raw = reader(launcher)
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError('launcher_changed')
        tree = ast.parse(raw)
        argv = _literal(tree, 'ARGV')
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        installed = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'install_entry' and isinstance(n.func.value, ast.Call)
            and isinstance(n.func.value.func, ast.Name) and n.func.value.func.id == 'ownership_support'
            for n in ast.walk(main))
        if not installed:
            raise ValueError('active_entry_wiring_unverified')
        support = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'ownership_support')
        pins = []
        for node in ast.walk(support):
            if isinstance(node, ast.Dict):
                try:
                    item = ast.literal_eval(node)
                except (ValueError, TypeError):
                    continue
                if set(item) == {'bytes', 'sha256'}:
                    pins.append(item)
        if len(pins) != 1:
            raise ValueError('support_pin_unavailable')
        directory = launcher.rsplit('/', 1)[0]
        support_raw = reader(directory + '/ownership_boot_support.py')
        if dict(bytes=len(support_raw), sha256=hashlib.sha256(support_raw).hexdigest()) != pins[0]:
            raise ValueError('support_changed')
        support_tree = ast.parse(support_raw)
        bindings = _literal(support_tree, 'PINS')
        install = next(n for n in support_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install_entry')
        loaded = {ast.literal_eval(n.args[0]) for n in ast.walk(install) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'load' and len(n.args) == 1}
        if not {'strategy_entry_only.py', 'strategy_order_lifecycle.py', 'strategy_order_ownership_install.py'} <= loaded:
            raise ValueError('ownership_wiring_unverified')
        parsed = {}
        for name in ('strategy_entry_only.py', 'strategy_order_lifecycle.py', 'strategy_order_ownership_install.py'):
            value = reader(directory + '/' + name)
            if dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest()) != bindings[name]:
                raise ValueError('entry_source_changed')
            parsed[name] = ast.parse(value)
        entry = parsed['strategy_entry_only.py']
        windows = set()
        attributes = {}
        for node in ast.walk(entry):
            if isinstance(node, ast.BinOp) and isinstance(node.left, ast.Name) and node.left.id == 'opening' and isinstance(node.op, ast.Add):
                call = node.right
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'timedelta':
                    for key in call.keywords:
                        if key.arg == 'seconds':
                            windows.add(ast.literal_eval(key.value))
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr in {'_entry_lifecycle_owner', '_entry_cancellation_policy'}:
                        attributes[target.attr] = ast.literal_eval(node.value)
        if len(windows) != 1 or not 0 < next(iter(windows)) <= 3600:
            raise ValueError('entry_window_unverified')
        lifecycle = parsed['strategy_order_lifecycle.py']
        owner = _literal(lifecycle, 'OWNER')
        zones = [ast.literal_eval(n.args[0]) for n in ast.walk(lifecycle)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'ZoneInfo']
        if len(zones) != 1 or owner != attributes['_entry_lifecycle_owner']:
            raise ValueError('lifecycle_owner_unverified')
        start = time.fromisoformat(argv[argv.index('--submit-at-ny') + 1])
        seconds = next(iter(windows))
        end = (datetime.combine(date(2000, 1, 1), start) + timedelta(seconds=seconds)).time()
        result.update(status='verified', reason=None, launcher_sha256=pin,
            entry_source_sha256=bindings['strategy_entry_only.py']['sha256'],
            lifecycle_source_sha256=bindings['strategy_order_lifecycle.py']['sha256'],
            timezone=zones[0], entry_start_local=start.strftime('%H:%M'), entry_end_local=end.isoformat(timespec='seconds' if end.second else 'minutes'),
            submission_window_seconds=seconds, lifecycle_owner=owner,
            cancellation_policy=attributes['_entry_cancellation_policy'],
            legacy_cancel_after_min=int(argv[argv.index('--cancel-after-min') + 1]))
        # Source verification proves the configured contract, not a successful
        # live run or a current broker-calendar close. Never invent16:00.
        try:
            from scripts.strategy_dashboard import primary_receipt
            primary, _ = primary_receipt(base_dir, now=now)
            day = date.fromisoformat(primary['session']['next_session'])
            if abs((now.date() - day).days) > 7:
                raise ValueError('session_stale')
            opening = datetime.combine(day, start, ZoneInfo(zones[0])).astimezone(timezone.utc)
            closing = opening + timedelta(seconds=seconds)
            result.update(session_date=day.isoformat(), session_status='bound_primary_receipt',
                entry_open_at=opening.isoformat(), entry_close_at=closing.isoformat(),
                in_window=opening <= now < closing)
        except Exception:
            pass
        return result
    except Exception:
        return dict(result, status='unknown', reason='active_scheduled_contract_unverified')
