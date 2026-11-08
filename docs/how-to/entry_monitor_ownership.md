# Entry placement and Monitor Positions ownership

## Status

This is a tested local candidate based on merged PR785 (`2c28287`). It is not an
activated production configuration. The existing hosted entry005 and monitor006
remain unchanged until a coordinated, verified rollout completes.

## Requested behavior

- The pipeline supplies the symbols and final limit prices. Selection, sizing,
  strategy definitions, exit thresholds, paper-account identity and position caps
  remain unchanged.
- The scheduled executor places eligible paper DAY extended-hours buy limits at
  04:00 America/New_York, within a bounded first-minute submission window. It does
  not chase prices, submit stops, replace stops or manage exits.
- The executor releases its account lease after the short submission batch.
  Monitor Positions reconciles fills, places and updates stops, and cancels any
  unfilled entry remainder on its first cycle at or after that session's regular
  exchange close. The close is normally 16:00 New York and is earlier on an
  exchange-calendar early close. There is no guessed close when the calendar is
  unavailable.
- Existing strategy risk exits may cancel a partially filled entry's unfilled
  remainder earlier. That preserves existing protective/exit behavior.

## Separate submission validity from order lifetime

The pipeline's existing v2 ticket and historical handoff retain their original
35-minute submission-validity fields. They are not rewritten or extended.
New signals receive a separate, digest-bound `entry_lifecycle` contract before
the first broker claim. It binds the complete original handoff, calendar date,
regular open/close, 04:00 entry time, and `MONITOR_POSITIONS` owner.

The new monitor derives two exact, source-pinned native methods and substitutes
only the cancellation deadline expressions. It leaves native globals and stored
handoffs unchanged. Old signals without the new contract keep their original
deadline. A present malformed contract fails closed. No active position, trade
history or ownership record is migrated by this source change.

## Independent entry safety

Entry does not read or require a monitor heartbeat, receipt or health status.
Before enrollment, and before each new symbol, it verifies source/account
identity and fresh broker orders/positions against durable ownership. Unknown
outcomes, conflicting ownership, untracked relevant orders, inconsistent fills,
pending stop transitions or inadequate full-quantity coverage on existing owned
positions block all new buys.

Recognition-only manual holdings remain outside automatic stop/exit authority.
Only signals newly enrolled in the current short batch can be exempt from the
existing-position stop requirement. Their identities, outcomes and quantities
are still checked. The monitor obtains ownership after the account lease is
released; the entry receipt never claims protection is confirmed.

ARDT's known original `pending_replace` stop plus its linked new stop must not
be treated as completed replacement. A fresh read must show the original
terminal and exactly one appropriate full-quantity stop before entry admission.

## Operational limits

Alpaca native equity stops trigger only in regular trading hours. A stop order
does not establish extended-hours protection. Fills between monitor polls are
not instantly protected, and a stopped or failed monitor cannot maintain stops
or perform close cleanup. DAY extended-hours orders can remain eligible after
16:00 if close cancellation is not confirmed. The cancellation request itself
is not proof that the remainder was canceled; cumulative fills and terminal
broker readback remain authoritative.

The native CLI receipt is relabeled `monitor_exchange_regular_close` for this
path. Its legacy `cancel_after_min=35` option remains a ticket-compatibility
setting, not the new order-lifetime deadline. Pending entry observations are
not a completed fill/protection assertion.

## Rollout requirements

1. Verify fresh task commands, all69 native source pins, account/source
   ownership and preserved historical receipts through the shared service.
2. Upload a new immutable release. Do not overwrite entry005, monitor006 or any
   consumed operation claim. Independently read back every uploaded byte.
3. Run a new bounded native source/read-only preflight. Do not invoke the
   trading CLI to prove installation or manually place/cancel trades.
4. Quiesce the existing monitor cooperatively, roll its command to the new
   deadline-aware release, resume it and verify advancing actual cycles. Keep
   its existing no-quote stop safeguards and recognition exclusions.
5. Only after monitor deployment/readback succeeds, change the existing
   scheduled entry task command to the entry-only release. Preserve schedules,
   account/risk settings and all unrelated tasks.
6. Verify the next genuine scheduled entry and monitor results. Rollback cannot
   simply restore a marker-unaware monitor once new marked signals exist.

## Local verification

Tests cover direct broker preflight, recognition exclusions, ARDT's two-stop
state, a corrected batch-policy handoff, forbidden entry lifecycle mutations,
full/partial fills, stop resizing, cancellation readback, early close, DST,
malformed markers, source-pinned method installation, history preservation and
receipt labels. These are offline synthetic tests, not a live trading cycle.

Run the focused tests plus native execution regressions and
`python -m scripts.docs_consistency_check` before publication. Pinned native
source fixtures exist only to test the deployed interface without rewriting
the canonical historical implementation.
