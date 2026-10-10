# Hourly account synchronization

`python -m scripts.account_sync` replaces three separate hourly account jobs with
one fixed sequence: activities ingestion, reconcile-only trades, account snapshot.
It does not import the application modules in the parent process. Each existing
job runs once in a subprocess using the active interpreter and inherited process
environment; no retry or rollback is attempted.

Activities retain `--lookback-days 30`. Reconciliation retains watermark use,
14-day fallback lookback, limit 500, overlap 300 seconds and poll setting 1 second.
Its immutable arguments add `--reconcile-only true --dry-run true`,
`--submit-at-ny ''` and `--ignore-market-gate true`. The empty submit time disables
the executor's default 04:00 ET wait; the supported market-gate bypass allows this
no-order reconciliation to run before market hours, on weekends and holidays.
Dry-run enables that bypass without enabling order execution. Neither an
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

### Reconciliation integrity

The activity ingestor distinguishes a valid empty list from malformed responses,
unavailable saved-watermark reads, incomplete pagination and failed requests. Its
incremental path buffers and validates the entire bounded batch before inserting
activities. A page-cap exhaustion, repeated continuation token or full page without
a continuation token fails without batch insertion or watermark advancement.
An invalid activity identity/time also fails before insertion. A failed watermark
write returns an unsuccessful outcome; any preceding committed insert remains.
`ACT_COVERAGE_COMPLETE` means the requested bounded chain passed these checks, not
that every historical activity or the provider's whole account history is proven.
The historical backfill mode is not qualified by this incremental contract.

Reconciliation validates the order response before trade-state writes. An empty
list is usable evidence; unavailable/malformed data or missing identities/timestamps
is not. A response reaching the configured 500-order limit is conservatively
incomplete: descending retrieval does not prove that older orders were included.
It fails without closing/decorating trades, inserting order events or advancing
the reconciliation watermark. No extra provider pages, retries or changed order
submission behavior are introduced. A genuinely terminal exactly-full window also
blocks until completeness can be established separately.

When open trades require position evidence, a failed/malformed position read stops
before writes. A successfully retrieved empty position list retains the existing
`POSITION_CLOSED` rule; this does not invent a fill price. Valid fills retain the
existing matching, price, quantity and exit-reason rules. Strict database readers
separate initial/empty state from read failure and saturated 200-row windows.
Their other callers retain the existing non-strict default.

Unresolved exit decoration, failed database writes/reads or unavailable watermark
state prevents watermark advancement. Earlier committed writes are not rolled back;
the workflow is not a cross-stage transaction or a snapshot of simultaneous broker
and database state. Existing 14-day fallback, two-day decoration lookback, 300-second
overlap and activity request parameters remain unchanged; bounded-window completion
does not establish historical completeness, attribution or matching correctness.

The reconcile-only caller returns nonzero for incomplete reconciliation. Hourly
health requires `ACT_COVERAGE_COMPLETE` / `RECONCILE_COVERAGE_COMPLETE` and retains
allowlisted `health_reasons`, such as `RECONCILE_POSITIONS_UNAVAILABLE`,
`RECONCILE_ORDERS_INCOMPLETE`, `ACT_PAGINATION_INCOMPLETE` and
`RECONCILE_WATERMARK_HELD`. Later stages remain `not_run` after failure. Raw provider
response/error text is not included in these health records. Auto-reconciliation
uses the same safer reconciliation method but its existing dispatcher still ignores
the Boolean result; this change does not impose a new trading-entry gate.

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
  markers make the run fail and leave later stages `not_run`. Legacy reconciliation
  can suppress errors; therefore `RECONCILE_START`, `RECONCILE_END` and
  `RECONCILE_WATERMARK_UPDATE` are required, and its broker/database failure or
  watermark-disabled markers fail even with exit zero. A missing trade decoration
  stops the sequence: the corrected caller returns failure; a legacy zero-exit
  result with this marker is classified `degraded`, also unsuccessful.
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
Also run `python -m unittest discover -s tests -p test_reconciliation_integrity.py -v`.
Integrity tests normally import the executor, activity ingestor and database reader
module with external modules replaced before import. Broker reads and database
operations are inert doubles; socket, process and environment-loading boundaries
are blocked. They exercise actual reconciliation/ingestion logic, strict database
readers and supervisor health, including incomplete-run caller failure. They do not
establish live API pagination semantics, database rollback, remote cancellation or
host integration. Schedules, ranking, sizing and order execution remain unchanged.

### Reconciliation fill retry contract

Strict reconciliation readers (`get_reconcile_state`, `get_open_trades` and
`get_closed_trades_missing_exit`) own one read-only transaction each. They require
an idle, non-autocommit connection; success ends the transaction, and a read error
rolls it back. An already active caller transaction is rejected without committing
or rolling back that caller's work. Non-strict readers preserve their legacy
ownership behavior. The normal reconciliation caller can reuse the same idle
connection for a subsequent fill or watermark write. These separate read
transactions do not establish a simultaneous broker/database snapshot.

For reconciliation of an observed filled sell, `reconcile_sell_fill` records the
`SELL_FILL` event and closes/decorates its trade in one PostgreSQL transaction.
It requires an idle connection with autocommit disabled, and explicitly selects
transaction-local `READ COMMITTED` so a lock waiter sees the previous committed
event. It does not commit a caller's already active transaction. An order-keyed transaction
advisory lock serializes participating reconcilers, and `FOR UPDATE` locks the
trade before checking its existing exit. A bounded lookup of all trades' exit
order bindings runs under that same order lock. Another owner or multiple owners
block without a write. Ownership is persisted as the target trade's
`exit_order_id`, atomically with the event. An identical prior event is reused
only when that order is uniquely bound to this trade; an orphan legacy event
blocks because its fill fields alone cannot identify its owner.
Multiple events or conflicting symbol, quantity, fill status or timestamp block
the update. A conflicting existing trade exit also blocks. Each existing non-null
exit timestamp, price and reason is checked independently, even when other exit
fields are missing. Matching partial exits can be completed; missing fields never
authorize overwriting conflicting existing evidence. Failed insertion or
trade update rolls back both writes. A successful transaction can be repeated
without inserting another event, including when its commit acknowledgement was
lost. The caller holds its watermark on failure; it does not retry immediately.

This is a per-fill transaction, not a transaction spanning the whole hourly job.
Legacy event writers do not take these locks, and existing duplicate/missing
history is not automatically repaired. Position-only closure remains a separate
operation until a fill is available for decoration. PostgreSQL locking, rollback
and concurrent execution have not been verified against a live database: offline
tests exercise the actual helper with an in-memory transaction model, including
insert/update failures, rollback, repeat calls and lost acknowledgement. A shared
connection caller regression also executes the real strict readers and atomic
writer through closure and decoration, including read/completion failures and
active caller transaction rejection. Provider and connection boundaries remain
synthetic; no live database was accessed.
A trade whose exit fields are complete but realized P&L is missing still needs
decoration; the helper computes that value while reusing the matching fill event.
Before writing an event or completing an exit, the helper requires positive,
finite stored trade quantity and entry price, both representable as positive
finite floats. Exit price must meet the same representation requirement.
Realized P&L retains the existing gross formula
`(exit_price - entry_price) * trade_qty`; it is not cost-adjusted. The computed
result must be finite; zero and negative P&L are valid. Missing, malformed,
nonpositive, nonfinite, overflowed or underflowed basis values fail the transaction
without fill/exit writes or caller watermark advancement. An existing non-null
P&L must be finite and equal to the computed value using explicit decimal
representations of the stored value and the float result, without tolerance.
Conflicts are rejected rather than silently corrected. These checks also apply
when a completed exit is retried. They do not change fill attribution, sizing or
the existing gross-P&L assumptions.
The decoration reader also includes trades missing only `exit_time` or
`exit_reason`, even if price, order ID and realized P&L are populated. Conflicts
and missing ownership keep the caller's watermark unchanged.

This ownership rule covers participating reconciliation transactions, not all
legacy writers: there is no database-wide uniqueness constraint in this
increment. The normal migration source declares a nonunique
`idx_trades_exit_order_id` index on `trades(exit_order_id)` with `IF NOT EXISTS`
for the ownership lookup. Offline regressions exercise the normal upgrade and
statement paths through a mock connection, including index failure propagation.
No migration has been applied by this correction; installed index state,
query plans, lock duration and live concurrency remain unverified. Existing
duplicate ownership is rejected, not repaired. The
caller's latest-sell-by-symbol matching heuristic is unchanged; multiple
unbound trades can still require manual attribution. One may complete before
another blocks, because the job is not one transaction. The same fill cannot
complete both through the participating helper. Offline multi-trade caller
regressions model this behavior; live concurrency remains unqualified.

Activity pagination validates every supplied recognized token, including header
and body aliases, before accepting completion. Only absent keys, null and an empty
string mean no continuation. Falsy nonstrings (`0`, `false`, `[]`, `{}`), whitespace
tokens and conflicting nonempty aliases fail as `ACT_PAYLOAD_INVALID`, without
insertion or watermark advancement. Nonempty string tokens are passed unchanged;
the literal string `"0"` remains valid. These rules do not establish provider
history completeness.
Supervisor tests use standard-library mocks and temporary storage, avoiding repository
pytest conftest's Alpaca import. Gate integration tests additionally require pandas,
dateutil and timezone data. They import the executor normally with provider, database,
environment, alerts and telemetry boundaries replaced before import. The real parser,
configuration builder, submit-time helper and dispatcher market gates execute against
synthetic pre-07:00 ET, weekend and holiday clocks. An empty candidate frame verifies
that reconciliation still reaches its branch; candidate loading/ranking, reconciliation
persistence, metrics and broker operations are doubles. Actual dispatcher logs feed the
supervisor's completion checks, and later snapshot-stage reachability is asserted with
a synthetic snapshot result. No credential file, provider/database connection, order or
real child job is used. This does not qualify full initialization, live reconciliation,
PythonAnywhere process groups or actual scheduled execution. Host integration still
requires verification after publication and controlled cutover.
