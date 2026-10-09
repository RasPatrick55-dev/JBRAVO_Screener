# Primary ML preparation and read-only postflight

The primary pipeline owns screening and ML preparation. The former canary
becomes a postflight reader; it must not publish a newer 30-symbol candidate run.
This is a software contract, not research qualification or trading acceptance.

## Primary invocation

Use `bash scripts/ops/run_primary_pipeline.sh` for the daily primary task.
It loads the existing virtualenv and process environment, holds a nonblocking
`flock` for the invocation, and calls `scripts.primary_pipeline` once.
It does not dispatch the executor or hourly account synchronization. The
[session handoff contract](../reference/session_signal_handoff.md) now fixes
completed-session preparation and completion-triggered postflight for this path.
In particular, `bin/run_pipeline_task.sh` is an executor launcher despite its
name and must not be repurposed.

The fixed steps are `screener,backtest,metrics,labels,ranker_eval`. Existing
pipeline dispatch order is retained. Existing freshness machinery prepares
features/predictions when required; it may regenerate dependent labels during
an internal refresh. There is one screener invocation, with the primary
universe and its existing settings, without the old canary's forced limit 30.
No training, tuning, ranker remediation or extra strategy evaluation is added.

Champion selection, candidate enrichment, candidate-scoped prediction refresh,
automatic feature/prediction refresh and the ML health guard are enabled.
The guard retains warn mode: warning does not silently change enrichment or
execution policy. It produces degraded postflight and an unsuccessful wrapper
result. Strict prediction metadata and strict automatic refresh default true;
explicit existing environment values are retained and disabled required
controls fail health. Existing predictor/evaluator timeouts default 900 seconds.

Approved legacy caller arguments such as `--backtest-quick`, `--backtest-args`,
`--metrics-args`, `--screener-args`, export paths and `--reload-web` can be passed
through unchanged. Pipeline defaults, including reload behavior, otherwise
remain unchanged. Fixed preparation flags cannot be overridden. Before host
cutover, inspect and preserve the actual primary task's nonsecret effective
arguments/environment; do not copy inline credentials into commands or reports.
Do not infer a production-universe setting from the smoke task.

## Run-bound health

The primary path additionally requires the completed-session handoff and matching
active invocation. Only its successful same-invocation postflight creates the
completion record consumed by ordinary entries. The standalone scheduled reader
does not create completion. Source, deployment and scheduler changes require review;
the historical schedule below is not an adopted cutover for the new opening policy.

The opt-in `--postflight-report` hook records selected facts directly from the
same pipeline invocation after its normal finalization. Ordinary callers
without that flag keep their existing behavior. Files are under
`reports/pipeline_postflight/`: an immutable `run-<id>.json` and atomic
`latest.json`, reopened against their saved bytes. Recording failures make a
previously successful pipeline unsuccessful and preserve an existing nonzero
failure code. Early pipeline failures may produce no new report; the wrapper
also rejects any report that predates its current invocation.

The checker requires completed measured screener/backtest/metrics/labels/eval
stages, nonempty labels, zero return codes including auxiliary refresh stages,
compatible nonstale predictions with a known snapshot, enabled required ML
controls, and candidate coverage scoped to this invocation. Predictions may be
reused without a new predictor call when already compatible and fresh.

ML warnings/blocks, unknown health, pipeline degradation, zero candidates and
incomplete score coverage never become green. Full score coverage is required
for green; partial coverage is explicitly degraded, even above the earlier
runbook's suggested alert threshold of 80 percent. Zero candidates can be a
valid software outcome but cannot establish candidate-score coverage.
Failed or skipped stages and malformed/absent/inconsistent evidence fail.

### Monitor generation and input coverage

The enrichment guard reports three distinct dates. `monitor_run_date` is the
legacy payload/artifact label; `monitor_run_date_source` identifies the payload
or DB-record fallback, and `monitor_artifact_run_date` retains the DB label.
Neither label is a substitute for the other two clocks:

- `monitor_generated_at` is the existing monitor payload's `run_utc`, normalized
  to UTC only when it is an explicit timezone-aware timestamp.
- `monitor_input_coverage` carries the existing `windows` dataset, baseline and
  recent start/end session dates and row counts. `monitor_input_source` reports
  the supplied input location; it is not a hash binding or proof of provenance.

The guard reports `monitor_observed_at`, generation age in seconds, the pipeline
reference session date, and input age in calendar days. Both use the existing
`JBR_ML_HEALTH_MAX_AGE_DAYS` limit (default seven): generation age is elapsed time;
dataset/recent endpoints are compared with the pipeline reference date. The
legacy artifact-label age check remains, including `stale_monitor`.
Old reports lacking generation/coverage evidence remain explicitly incomplete;
a recent artifact date or freshly generated report cannot make old inputs green.

Health reasons distinguish missing/invalid generation times, future generation,
stale generation, missing/invalid coverage fields, empty/invalid row counts,
inconsistent windows/counts, future/stale dataset/recent endpoints, and input
coverage after generation. Session dates must be exact ISO dates; naive or
malformed generation timestamps are rejected. A nominal recent-window start
before the actual dataset start is allowed, matching the existing monitor's
short-dataset construction. Overlapping baseline/recent windows remain allowed;
this check does not qualify their statistical suitability or OOS provenance.

DB-first loading and warn/block modes remain unchanged. The pipeline explicitly
sets `require_monitor_provenance=True`, recorded as `monitor_provenance_required`.
`monitor_provenance_reasons` reports the distinct generation/coverage assessment.
Warn continues
enrichment with explicit health reasons and degraded postflight; block prevents
enrichment. The shared guard is also used by `ranker_autoremediate`; its existing
default policy and legacy artifact-date parsing remain unchanged. It can inspect
the additional provenance reasons, but they do not change its decision unless
that caller separately opts in. No remediation,
training, monitor refresh or workflow/schedule change is executed by this repair.
Adding monitor preparation to the primary workflow remains a separate decision.

Synthetic provenance tests normally import the guard with the production DB and
utility-package initializer replaced before import. They exercise DB/FS loading
with inert records, date/count validation, and the real saved postflight path.
The pipeline summary assignment is source-extracted; full legacy pipeline import,
real DB contents, provider execution and scheduled host integration remain untested.

Source byte identities for the pipeline, checker and entry launchers, plus the immutable run
record, are verified before acceptance of the report. This is local binding,
not independent attestation or proof of every underlying database write.
Health is not based on shared-log greps or exit zero alone.

## Read-only scheduled postflight

Retain the old canary entry path as a compatibility launcher:
`bash scripts/ops/run_canary_smoke.sh` now invokes only
`python -B -m scripts.pipeline_postflight`.
It activates the existing virtualenv but loads no credential file, imports no
bot/provider/database module, dispatches no application child process, and
creates no reports/logs/caches. The scheduler itself retains stdout. Historic
canary logs are untouched. This check does not repair or refresh anything.

For legacy reports, default checks require a run started and finished on the current UTC day,
no future timestamps, and age at most 7,200 seconds measured from its start.
The current 03:45 UTC primary and 04:20 UTC postflight fit this window after
successful completion. Slow/incomplete/missing primary runs fail; previous-day
success cannot mask them. Explicit `--expected-run-date` and
`--max-age-seconds` are available for scoped inspection, not fallback.
Exit 0 means ok; exit 1 means failed or degraded.

## Review and controlled cutover

Do not deploy only the new canary reader: the old primary lacks its new report
and ML preparation. Review, publish and integrate both changes together.
Then bind hosted source, preserve legacy task parameters, move the primary
task to the dedicated launcher and replace the canary description with
`Primary pipeline postflight (read-only)`. Keep their existing schedule times
unless separately changed. Observe a complete primary run followed by its
postflight; a checker failure never authorizes a rerun, order or repair.
Preserve prior task commands and results for rollback; rollback must restore
both producer and checker coherently. Source acceptance, merge and host cutover
remain separate review gates. No host change occurred in offline verification.

## Offline verification boundaries

Synthetic tests import the new producer/checker and primary caller normally,
use temporary storage, and block network and subprocess dispatch. They exercise
success, failures, skipped stages, stale/unknown/degraded health, coverage,
source/history bindings, strict JSON and saved-byte paths. The source bindings
also cover `scripts/utils/ml_health_guard.py`, whose decision reaches the report.
Existing pipeline
parser and opt-in recording hook are source-extracted for bounded execution;
the complete legacy pipeline import and actual screener/ML/DB/provider/reload
paths are not exercised. Shell contracts are inspected, not executed by these
tests. Host virtualenv, flock, task parameters, performance and live integration
remain unqualified. A passed checker neither validates historical research
data nor authorizes trading.
