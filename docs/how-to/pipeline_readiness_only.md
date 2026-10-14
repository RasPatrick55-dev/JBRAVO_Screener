# Pipeline completion while research qualifies

The fixed primary launcher now uses `--pipeline-readiness-only`. A successful
run reports `pipeline_ready=true`, `entry_ready=false`, `research_qualified=false`
and `operating_acceptance=false`. This permits scheduled computation and a
run-bound completion receipt while collecting prospective model evidence.
It does not grant a paper or live trading entry exception.

Only explicitly identified version2 cross-model diagnostics can be advisory:
`action_investigate`, optionally accompanied by stale diagnostic input dates or
the diagnostic artifact date. The original ML guard still computes its verdict
and seven-day freshness checks. Its warnings remain in the health record and in
`research_reasons`; they are never replaced with a healthy-model assertion.
Missing or malformed health, unknown reasons, stale predictions, block-mode
decisions and production-model alerts remain blocking.

All current operational checks remain required: completed session timing,
successful measured stages, nonempty labels, compatible current predictions,
strict refresh/metadata controls, complete candidate scores, exact frozen
candidate values, current source bindings, immutable report history and a
completion receipt belonging to this invocation. Operational failures still
produce a failed or degraded postflight and a nonzero primary exit.

The executor rejects `pipeline_only_no_new_entries` before candidate admission,
including legacy/non-session checks. A pipeline completion cannot be reused as
entry authorization, even when the research monitor becomes healthy. Existing
protective-position monitoring and reconciliation are unchanged. Restoring new
entries requires a separately reviewed operating-policy change.

Normal direct pipeline callers retain the legacy coupled policy unless the
explicit pipeline-only flag is supplied. The primary prevents overriding its
fixed flag, including abbreviated arguments. Old reports retain their existing
behavior; no failed report or completion is rewritten.

The host research collector pins the deployed commit and producer identities.
Before deployment, prepare a bounded replacement enrollment from its verified
SIP seed, preserving the same355 symbols, fixed March19 model,40/5/25 windows,
dates and unused71-input/65-score budget. The replacement must be registered
before October14 warmup and October15 baseline. Retire only its two unstarted
dispatcher tasks, verify new packet/task bytes, and preserve the old packet and
all earlier failed claims. Changing a pinned packet in place is prohibited.

Verification uses inert unit tests with temporary source/report copies and
mocked DB, SDK and dispatch boundaries:

```text
python -m unittest tests.test_pipeline_readiness_only tests.test_pipeline_postflight tests.test_ml_health_provenance tests.test_executor_preflight_health tests.test_signal_handoff
```

No test grants research qualification, creates a model, dispatches an order or
qualifies future scheduled runtime. Observe the actual next primary and warmup
receipts after they run. Source deployment must use a new exact reviewed plan;
it does not replay a primary pipeline or an old failed study.
