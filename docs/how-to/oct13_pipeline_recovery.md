# October 13 pipeline and research recovery

The October 12 research collection failed with
`producer_existing_or_future_feature_no_overwrite`. Its cycle receipt reports
`scores_collected=false`. The host feature metadata for October 12 was generated
at October 12 04:04 UTC, before that trading session. The feature generator used
the invocation's wall-clock date. That is a source defect; the producer's refusal
to replace an occupied date correctly preserved the existing artifact.

The later primary pipeline failed at October 13 03:58 UTC after prediction
refresh returned `PRODUCTION_SCORE_CAPTURE_FAILED`. Predictions remained dated
October 9. The deployed generator and predictor both removed rows with immature
forward labels. Offline regression reproduces removal of current inference rows.
The host's old generic capture log does not establish its precise exception.
The patch therefore fixes the demonstrated input defects and adds sanitized
capture reason codes; a successful host run remains required.

## Code behavior

- Derive the artifact date from actual daily-bar timestamps, using the UTC
  session-date convention shared with score capture. Reject unavailable explicit
  sessions and future session dates before publication.
- Retain current feature rows when label rows are absent or outcomes are pending.
  Unknown labels stay blank. No outcome placeholder is published.
- Keep model training's existing exclusion of unknown target labels.
- Require current-session feature metadata during pipeline freshness checks and
  pass the canonical pipeline session to both feature-refresh commands.
- Preserve capture-before-publication, model/source identity checks, freshness,
  and executor preflight checks. Capture errors export only fixed reason codes.

## Host deployment and operating verification

1. Review the immutable patch manifest and tests. Approve its governed merge and
   the narrow deployment registry transition. The existing PythonAnywhere
   deployment registry accepts only previously approved PRs 726–728; this new
   patch cannot reuse their identities, previews or approvals.
2. Prepare a fresh governed host inspection. Preserve existing generated outputs
   and refuse unrelated tracked changes. Prepare and approve the exact deployment
   preview before its one-time apply. Read back all deployed source bytes.
3. Prepare a new primary/postflight verification preview for the relevant current
   signal session. Review its database effects and executor coordination before
   approval. Do not replay an earlier failed acceptance plan or invoke trading
   to repair pipeline data. A historical `active-primary.json` is not evidence
   that a process is still running.
4. Verify the actual feature session, current candidate score coverage, prediction
   snapshot, successful pipeline completion record, new handoff and postflight
   report. A safe reason code may identify a separate host capture error; report
   it and stop rather than weakening capture or freshness checks. Do not infer
   qualified research or model acceptance from exit zero.

Normal pipeline feature publication is an upsert. Any repair run that replaces
an existing same-date legacy feature artifact needs its concrete effect reviewed;
this document does not authorize changing stored historical rows.

## Prospective research restart

The old enrollment cannot resume from a failed first input session: each later
producer requires the previous completed journal. The October 12 claim and
failure remain immutable. There is no receipt-bound October 12 journal to reuse.
Do not delete claims, fabricate history, replay collection, backfill scores,
shift the old approved dates, or label an incomplete cohort as qualified.

Prepare a distinct prospective enrollment after deployment and successful input
verification. It needs a freshly preserved seed ending before the new baseline,
exact deployed helper/source/model bindings, all 355 reviewed symbols, and a
new independent readiness comparison. Preserve the 40 baseline / 5 gap-input /
25 recent design, registered data window, DST/early-close handling, finite
provider/database budgets and create-only producer publication. The new baseline
must open after the new human-reviewed registration. The already opened October
13 session cannot be made its prospective first baseline session.

Review the complete new date table, seed, pilot, registration, dispatcher and
monitoring bytes before approving host enrollment. Old source bindings and
first-session failure evidence cannot qualify the changed producer. Schedule
retirement or disabling old tasks is a separate explicit effect; the current
restricted MCP cannot edit the old collection or monitoring tasks. Do not create
parallel conflicting producers or assume the old tasks have been retired.

This runbook is a recovery proposal, not a registered study, host execution pack,
deployment approval, database repair approval or trading authorization.
