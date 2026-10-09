# Executor preflight and health contract

This contract retains per-role receipts and entry-readiness checks. The current
ordinary entry policy is the [completed-session signal handoff](session_signal_handoff.md).
It changes the opening target and price anchor/cap policy while preserving ranking,
sizing formulas, protective SELL behavior, reconciliation and schedules. Local
verification is not Board acceptance or proof of deployed behavior.

## Candidate readiness

Ordinary DB-backed entries require a completed-session primary report, matching
immutable history/source bindings, the active primary invocation and its successful
completion-triggered postflight record. Starting a new primary invocation invalidates
old completion even if that invocation fails. Required stages, controls, predictions,
monitor/ML health and full candidate coverage must pass. Legacy reports retain their
same-UTC-day/12-hour diagnostic assessment but cannot satisfy ordinary entry readiness.

The primary freezes the complete typed candidate-value digest and supplied daily
close references from its exact scored batch in a read-only repeatable-read snapshot.
The executor selects only that run date/timestamp; it cannot fall back to an older
batch. Raw values must equal the producer digest. Filtered and ranked frames retain
their separate mutation checks. BUY gates recheck report/selection bindings, the
04:00–09:30 ET execution session and the frozen-price cap. Existing reports are not
retroactively upgraded. See the linked contract for session/cutoff/price semantics.

The loader rolls back and closes its snapshot after materializing its frame; optional
ranker-join recovery retains its savepoint behavior. Standalone diagnostic callers
keep their older-date fallback and previous connection ownership. This is not a DB
lock through submission or independent proof of live contents. The digest does not
archive all underlying features; the supplied source label may be unknown.

Reconciliation-only and diagnostic runs, and path dry-runs, retain their separate
roles. Ordinary path-based entries fail closed. Existing broker-clock, reconciliation
and authentication paths may still run before an entry is rejected: the entry gate
does not make the whole executor offline or free of side effects.

## Run attribution and receipts

Each actual CLI invocation attempts to save an exclusive
`reports/executor/<role>/run-<uuid>.json` and an atomically replaced same-role
`latest.json`. Roles are `premarket` (ordinary entry CLI), `reconciliation`,
`diagnostic`, `dry_run` and `unresolved` (initialization/argument failure).
An explicitly supplied `JBRAVO_SCHEDULER_TASK_ID` is validated as a positive decimal
ID and carried through the normal config builder into `task_attribution`.
`independently_verified` stays false: an environment declaration is not scheduler
attestation. Missing IDs remain explicit and preserve ordinary CLI compatibility.
Inspection with `--expected-task-id 1326478` rejects absent or different IDs.
This source change does not edit task commands; a separately approved cutover must
add that declaration and bind the configured command through provider readback.
Hourly
reconciliation cannot overwrite the entry role's receipt. Existing shared
`data/execute_metrics.json` remains for compatibility and is not task evidence.

Receipts contain UTC start/finish, actual caller return code, bounded current-run
metrics, preflight report identity and reasons, candidate fingerprint, selected
effective settings, existing order observations and source-file byte bindings.
Error snapshots and persisted metrics no longer seed the current invocation
from an older shared snapshot. Authentication headers, credential values, full
candidate rows and raw exceptions are excluded from the new receipt.

Receipts are saved/reopened before publication. A recording failure is logged as
`EXECUTOR_RECEIPT_FAILED` and changes an otherwise successful CLI return to 1;
an existing primary failure retains its return code. Missing/invalid receipts
must never be interpreted as success. A hard kill cannot guarantee a final
receipt, and exclusive history is application convention, not immutable storage.
No retention/deletion policy or deployment is introduced here.

Read-only inspection, from the repository root:

```bash
python -m scripts.executor_health inspect --role premarket
python -m scripts.executor_health inspect --role reconciliation
python -m scripts.executor_health logs
python -m scripts.executor_health inspect --role premarket --expected-task-id 1326478
python -m scripts.executor_health deadlines --role premarket --expected-task-id 1326478
```

Receipt inspection strictly decodes JSON, checks role/history/source bindings,
current UTC day, finish age (12 hours), timing order and recomputed outcome.
`failed` and `pending_confirmation` return 1; `ok`, `skipped` and `no_orders`
return 0 with distinct statuses. Skipped/no-order execution is not a successful
entry or proof that the strategy had executable candidates. These assessments
are limited to recorded caller metrics and observations, not independent broker,
database, protective-order or scheduler acceptance. Research qualification and
operating acceptance remain false.

## Cancellation semantics

`--cancel-after-min` means minutes **from order submission**, not from regular
market open. The help text now describes the existing behavior. The existing
60-second polling detach remains unchanged: it may occur well before the
35-minute deadline. Detachment does not cancel an order or establish eventual
deadline enforcement.

The receipt separates `submitted`, `resting`, `cancel_requested`,
`cancel_failed`, `cancel_unavailable` and observed terminal statuses (`filled`,
`canceled`, `expired`, `rejected`). A cancellation acknowledgment is not a
terminal observation; the legacy `orders_canceled` counter is explicitly not
confirmation. A submitted order with no observable identity/state is also
unconfirmed. Up to 64 observations are retained, with explicit truncation.
Pending or truncated confirmation remains `pending_confirmation`; reported
cancellation failures remain failed.

`deadlines` is a **read-only plan**, not an active cancellation command. It consumes
an eligible entry receipt and preserves the observed order-ID/symbol/deadline
relationships. It reports waiting, due cancel-and-confirm proposals, or confirmation
of an already acknowledged request. A validated `cancel_requested` observation
requires confirmation immediately, even if no deadline was recorded or a supplied
deadline is still in the future. Its proposal is only `confirm_existing_request`,
never another cancellation request. An absent deadline is returned as JSON `null`;
the planner does not invent one. Any supplied deadline is still validated and
preserved, and contradictory deadlines still block the plan. Terminal orders
propose no action. Missing deadlines for unrequested cancellations,
contradictory identities/deadlines, terminal regression, failed receipts,
unknown submissions and truncated observation populations block the plan. It
performs zero broker calls and always reports `cancellation_enforced: false`.
Exit 1 denotes a blocked plan or due action; exit 0 means no action due, not
successful cancellation.

The proposed enforcement increment is one supervised, paper-account-bound worker
that survives executor polling detachment, reads only its registered BUY order IDs,
checks account/side/symbol/current state before touching them, cancels each due
unfilled remainder once, and separately polls terminal state for a finite window.
It must preserve fills/partial fills and existing protective SELL behavior, record
late or absent confirmation as unresolved, use an overlap lock, and recover its
durable registry after interruption. An hourly account job cannot meet a 35-minute
deadline; the worker needs an explicitly reviewed wake-up/deadline contract.
No account-wide cancel sweep, replacement order, active worker or new schedule is
introduced here. Broker failure, worker loss and remote cancellation remain real
failure modes; software cannot guarantee remote completion.

Before activation, the Board must review the mutation policy, finite broker-call
and confirmation limits, identity binding, partial-fill protection and supervised
runtime. Live cancellation and orders remain outside this local increment.

## Bounded log review

`logs` seeks directly to at most the last 65,536 bytes of
`logs/execute_task.log`, drops the first partial line, and returns at most 40
allowlisted event summaries. It does not download, truncate, rewrite or rotate
that log. Output omits raw lines and restricts returned field values; lines with
credential-related markers are discarded. Missing/redirected paths fail.

The result includes byte range/hash, total file size at open, matched count,
sample truncation and whether size/mtime changed during the read. This is a tail
snapshot, not complete log history, scheduler attribution or evidence of task
success. The command's exit zero means successful inspection only. Fresh role
receipts are the preferred run evidence; long logs remain preserved for diagnosis.

## Offline verification boundaries

Synthetic tests import the actual executor module, exercise CLI/main receipt
recording, the actual loader/normalizer/model filter, submission gate,
cancellation acknowledgment and existing polling detach/terminal branches.
Credentials, environment loading, logging setup, DB/provider/SDK boundaries,
alerts and telemetry initialization are mocked before import; import sentinel
execution is disabled. BUY submission is mocked. No task, host, provider,
database or order is exercised. Tests use temporary files and block network and
child processes. Existing postflight tests include source-extracted pipeline
hooks, so they do not establish full normal pipeline initialization.

The suite now carries the repository's `alpaca_optional` marker when pytest is
available; its documented unittest runner remains usable without pytest installed.
The standard pytest credential fixture can therefore run these tests with no
credentials. A new normal-import dry-run case also exercises actual sizing with
synthetic prices and buying power, and asserts the calculated quantity without
submitting an order. PostgreSQL transaction calls are tested through a fake
connection and actual query-helper recovery, not a real database.

End-to-end broker transport/retries, real cancellation completion,
concurrent execution, scheduled invocation, actual database content and live host
integration remain unqualified. Required Board review and separate publication,
merge/deployment authorization precede use on PythonAnywhere.

## Remaining operational acceptance sequence

1. Review this local diff and new offline verification history, then separately
   authorize publication/integration. Preserve all earlier failed attempts.
2. Qualify live initialization without orders using an explicitly approved
   paper-account diagnostic: fixed source/dependencies, bounded account/clock and
   repeatable-read candidate reads, no auto-reconciliation/write paths. This requires
   its own exact request/query and output limits; a dry-run name alone is insufficient.
3. Approve task-ID command cutover without changing its existing order settings or
   schedule. Read back the configured command and bind its ID/hash to the receipt.
4. Inspect the next naturally scheduled receipt and its history/source bindings,
   primary/candidate linkage and return status. A missing receipt is missing evidence,
   not proof the task failed or never ran. An entry receipt remains distinct from
   independent broker terminal-state evidence.
5. Review and qualify the separate cancellation worker before any activation.

This list is a proposed acceptance route, not permission to run host diagnostics,
queries, scheduled tasks, orders or cancellation. Research/operating acceptance
remains false. No full pipeline or live integration is established by mocks.
