# Bounded entry preflight rechecks

## Local candidate, not deployed

This candidate changes only initial entry admission and adds a final submission
time check. It does not run the entry task, place paper orders, change the
monitor, or authorize a host rollout. PostgreSQL remains the source of truth.

The November 9 investigation found the scheduled entry process failed once on
`ENTRY_PREFLIGHT_STOP_COVERAGE_UNCONFIRMED` before submitting LI or HUM. ARDT's
stop replacement settled about fourteen seconds later, still inside the original
04:00-04:01 America/New_York submission window. That observation supports a
bounded read-only recheck; it does not establish that other entry gates would
have passed or that orders would have filled.

## Admission boundary

- Recheck only the initial GET/SELECT broker preflight, before any enrollment,
  durable order claim, or submission from this process.
- Only the exact `ENTRY_PREFLIGHT_STOP_COVERAGE_UNCONFIRMED` result is eligible.
  All stop coverage and identity requirements are unchanged. A genuinely missing
  or inadequate stop still blocks every new order.
- Allow at most ten admission checks with two-second pauses. The last permitted
  failure is raised; the old failure is never converted into a successful proof.
- Release the account advisory lease before each pause. Acquire it again before
  fresh checks so Monitor Positions has an opportunity to reconcile replacement
  stops. Retain the shared source lease, which is compatible with the monitor's
  shared source lease and prevents a coordinated deployment during admission.
- After each pause, reload the actionable PostgreSQL handoff and unified
  selection through the existing validators. Require the complete original
  handoff, selection record and ordered execution queue to remain equal. Recheck
  source and expected paper-account identity. Do not adopt changed prices,
  candidates, sessions, source bindings or selection runs.
- Recompute the baseline of preexisting signals and orders after successful
  admission. A signal created by another worker during the pause cannot become a
  current-batch exemption.
- Keep the original broker request budget. Exhaustion, timeouts, unresolved
  outcomes, inconsistent snapshots, ownership conflicts and all other failures
  stop immediately. No broad retry of broker errors is introduced.

The exception handler surrounds only the read-only preflight call. The context
manager yields to execution outside that handler. Per-symbol checks, quote
validation, enrollment, claims, submission, receipt writing and cleanup are never
retried, including failures with the same error text after a possible submission.
Existing durable claim/no-second-buy behavior is preserved.

## Strict submission cutoff

The original opening plus sixty seconds remains the exclusive deadline. All
pauses are capped by the remaining time. A monotonic budget also prevents a wall
clock rollback from extending that original remaining time. Deadline checks run
after account-lease acquisition, after revalidation/preflight, around the final
source/selection guard, and at the entry broker's submit boundary.

A slow read may finish after the deadline, but it cannot authorize a late POST.
An already-started network call is not canceled and broker acceptance timing is
not guaranteed. If the deadline expires after a durable claim, native conservative
unresolved-outcome handling remains in force; the claim is never reset/replayed.

Selection, strategy identity, published limit prices, quote freshness, sizing,
daily attempt budget, portfolio caps and monitor-owned close cleanup are unchanged.
This patch does not repair quote availability and cannot promise the next entry
will succeed. There is no manual retry after today's window.

## Verification and rollout boundary

Offline tests include settled and permanent stop replacement states, ten-attempt
exhaustion, strict deadline/time rollback, revalidated handoff drift, account
lease release, slow guards and policy transactions, unrelated errors/timeouts,
and no retry after possible submission. The stop checks and final policy race
also exercise the exact deployed Engine fixture.

A later rollout would need a newly reviewed immutable entry release with its
helper/launcher pins rebuilt, exact source verification and authorization for the
existing task's command change. Do not overwrite the installed release or reuse
any old approval or operation claim. No monitor restart or lifecycle-source change
is required by this design. A fresh scheduled run remains the operational proof.
