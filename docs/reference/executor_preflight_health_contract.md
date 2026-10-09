# Executor preflight and health contract

This contract adds an entry-readiness gate and per-role execution receipts. It
does not change sizing, ranking, order parameters, retries, cancellation timing,
protective SELL behavior, schedules or paper-only safeguards. Local verification
is not Board acceptance or proof of deployed behavior.

## Candidate readiness

An ordinary entry run must use PostgreSQL candidates and a valid
`reports/pipeline_postflight/latest.json`. The executor runs the existing
postflight assessment itself: the primary must have succeeded without degraded
health, with its required stages, controls, predictions, monitor/ML health and
candidate coverage passing. Its immutable history must match and its source
bindings must match current source. This checks the primary's saved health report;
it does not prove that a separately scheduled postflight command ran.

The primary must have started and finished on the current UTC date, with no future
finish, and started no more than **43,200 seconds (12 hours)** earlier. This
executor policy accommodates overnight preparation and the existing 07:00
America/New_York submission wait. It does not change the standalone postflight
CLI's two-hour default. A missing, failed, degraded, stale or unbound report blocks
new entries with a precise reason and unsuccessful outcome. Zero-candidate primary
reports currently assess as degraded and therefore also block entries.

Before normalization/model-score filtering, every loaded DB row must have an
explicit aware `run_ts_utc` equal to the report's candidate-batch timestamp.
The population count must equal its declared total and symbols must be valid
and unique. The loader's existing older-date fallback remains available for
inspection but cannot pass this entry gate with an old preparation timestamp.
A previous-session `run_date` label is permitted only when preparation linkage
passes; a row label alone is not freshness evidence.

Existing score filters and ranking remain unchanged. Retained rows must be a
subset of the validated batch with unchanged symbol/timestamp relationships.
The report is checked before loading, after loading, after submission wait and
immediately before each logical BUY submission. A different latest report during
the run blocks rather than silently rebinding the candidates. BUY symbols must
belong to the retained selection; SELL/protective submissions keep their existing
path. Reconciliation-only and diagnostic runs, and path-based dry-runs, do not
require entry evidence. Ordinary path-based entry runs now fail closed.

Batch linkage and its receipt fingerprint cover symbol/timestamp membership,
not every price/feature field. They do not establish an atomic database snapshot,
independent data quality, historical eligibility or freedom from filesystem/DB
races. The existing broker clock, reconciliation and authentication paths may
still run before an entry is rejected. This gate does not make those paths
offline or side-effect-free.

## Run attribution and receipts

Each actual CLI invocation attempts to save an exclusive
`reports/executor/<role>/run-<uuid>.json` and an atomically replaced same-role
`latest.json`. Roles are `premarket` (ordinary entry CLI), `reconciliation`,
`diagnostic`, `dry_run` and `unresolved` (initialization/argument failure).
These identify CLI roles, not proof of a particular scheduler/task ID. Hourly
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

No new broker polling, cancellation, monitor behavior or provider call is added.
A later terminal-state confirmation and any deadline enforcement after detach
need a separately reviewed operational mechanism. This packet makes that gap
visible rather than claiming that cancellation is now guaranteed.

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

End-to-end sizing, broker transport/retries, real cancellation completion,
concurrent execution, scheduled invocation, actual database content and live host
integration remain unqualified. Required Board review and separate publication,
merge/deployment authorization precede use on PythonAnywhere.
