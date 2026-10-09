# Hourly account synchronization

`python -m scripts.account_sync` replaces three separate hourly account jobs with
one fixed sequence: activities ingestion, reconcile-only trades, account snapshot.
It does not import the application modules in the parent process. Each existing
job runs once in a subprocess using the active interpreter and inherited process
environment; no retry or rollback is attempted.

Activities retain `--lookback-days 30`. Reconciliation retains watermark use,
14-day fallback lookback, limit 500, overlap 300 seconds and poll setting 1 second.
Its immutable arguments add `--reconcile-only true --dry-run true`; neither an
arbitrary module nor an order-mode override is accepted by this CLI. Dry-run here
prevents orders; it does **not** make reconciliation read-only: the existing job
still writes trade history and watermark records. The snapshot also writes its
existing database table. This is not a trading workflow or a database transaction
spanning all stages.

The pipeline, pre-market executor, position monitor and usage updater are unchanged.
Existing component log/metrics paths are retained, including reconciliation's
existing executor metrics writes; separating those shared metrics is outside this
increment.

## Health and failure contract

- An OS lock prevents concurrent invocations of this new workflow. It does not lock
  the old individual schedules or the separate executor/monitor. On POSIX the owned
  child inherits the lock descriptor, so loss of the parent cannot release it while
  that child continues. Windows uses a byte-range lock for local offline testing;
  child descriptor inheritance is only implemented on POSIX.
- Each stage defaults to a 300-second local timeout. `--stage-timeout-seconds`
  accepts finite values from 1 through 900. There are three sequential stages plus
  bounded local child cleanup. Polling limits local runtime and child-output storage
  to 2 MiB with possible overshoot between polls; this is not a provider request or
  wire-byte budget. It does not establish remote cancellation or undo partial writes.
- Nonzero exit, timeout, excess output, missing completion markers or known error
  markers makes the run fail and leaves later stages `not_run`. Legacy reconciliation
  can suppress errors; therefore `RECONCILE_START`, `RECONCILE_END` and
  `RECONCILE_WATERMARK_UPDATE` are required, and its broker/database failure or
  watermark-disabled markers fail even with exit zero. A missing trade decoration
  produces `degraded`, also unsuccessful, and stops the sequence.
- Per-run JSON is retained beneath `logs/account_sync/`; `latest.json` is atomically
  replaced under the lock and reopened. An overlapping invocation returns nonzero
  and retains its own record without overwriting the active run's latest result.
- Records contain run ID, UTC timestamps, fixed stage names, exit codes, durations,
  sanitized marker names, remaining-stage disposition and current source-file hashes.
  Arbitrary child output, environment values and exception text are not copied into
  the health JSON. Existing application logs remain governed separately.
- `ok` alone returns CLI zero. `failed`, `degraded`, `overlap` and recording failures
  return nonzero. Consumers must check timestamps: an old `latest.json` is not proof
  of a new successful run. Source hashes bind observed files, not every imported
  dependency, historical continuity or absence of filesystem races.

## Proposed PythonAnywhere cutover — after source review/publication

One hourly task at **15 minutes past the hour**, description `Account synchronization
(activities, reconcile-only, snapshot)`, command:

```bash
bash /home/RasPatrick/jbravo_screener/bin/run_account_sync_task.sh
```

The launcher loads the existing virtualenv and `.config/jbravo/.env` into its process
without displaying credentials. It uses `set -euo pipefail` and `exec` to preserve
failure status. Do not put credentials inline in the schedule command.

1. Review the isolated source diff and offline results; use the existing approved
   commit/PR/merge/canonical-sync workflow before host adoption.
2. Verify the installed source identity and configured interpreter. Inspect the
   three old schedules and their current execution state without copying secrets.
3. Disable (do not delete) old task IDs 1307546 (hourly :05), 1308220 (:15) and
   1308869 (:15), then confirm no old invocation remains in progress. Disabling a
   schedule does not cancel an already running job. Do not kill a job to force cutover.
4. Create the single hourly :15 task. Avoid a period with both sets enabled. Verify
   its exact command and the next run's per-stage record, final status and freshness.
5. If cutover fails, disable the new task, wait for its invocation to finish, and
   restore the three old schedules unchanged. Keep its failed/partial run evidence.

The current MCP lacks schedule listing/mutation capabilities; an authenticated Tasks
page can provide the separately authorized schedule transition. This document is a
reviewable cutover plan, not proof of deployment or permission to start trading.

## Offline verification

Run `python -m unittest discover -s tests -p test_account_sync.py -v`.
Tests use only standard-library mocks and temporary storage, avoiding repository
pytest conftest's Alpaca import. They do not import the three application modules,
load credential files, connect to a provider/database or execute the jobs. Mocked
process tests validate supervision, not PythonAnywhere process-group behavior or
live API/database semantics. Host/virtualenv/shell integration and actual scheduled
execution remain to be verified after publication and controlled cutover.
