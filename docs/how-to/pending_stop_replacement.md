# Pending stop replacement isolation

## Opt-in correction and activation requirements

This opt-in monitor correction uses verified PR786 (`63b39a5`) as its source
baseline. Tests use synthetic account/order ports and exact reviewed native
source fixtures. Source publication and passing tests are not evidence of
hosted activation, a settled replacement, a confirmed stop, or a passing
production cycle. Hosted acceptance requires separate fresh source,
durable-lineage, review, and rollout verification.

## Why the monitor needs a narrow deferral

The pinned native monitor rejects two active stop records with
`strategy_multiple_active_stops`. A broker replacement can temporarily expose
the predecessor as `pending_replace` while the linked successor is `new` or
`accepted`. That can abort management of other held positions in the same
cycle. Neither observation proves that replacement has finished.

The optional `strategy_pending_stop_replacement.monitor_engine` adapter sits
outside the existing lifecycle and quote-safe classes. Installation is:

```python
install_monitor(monitor, NativeEngine, lifecycle,
                pending_replacement=strategy_pending_stop_replacement)
```

The optional composition is validated eagerly at installation, without
constructing an engine, store, or broker. The original three-argument installer
remains backward compatible. The native Engine, native globals, native source
inventory, and entry-only installer are unchanged.

## Exact recognized state

Deferral requires all of the following current, same-signal evidence:

- Exactly two distinct active stop records, both `OBSERVED`, with native
  account/signal/role/sequence-derived client IDs and valid native
  request/observation identities.
- The predecessor is `pending_replace`; the successor is `new` or `accepted`.
  `replaced_by` and `replaces` are reciprocal. The successor's durable operation
  is `PATCH` with `request.order_id` equal to the predecessor's observed ID.
- Sequences advance by exactly one, and the successor is the latest stop
  sequence. A predecessor that is itself a replacement must have an owned,
  terminal historical stop matching its durable target and `replaces` link.
- Both orders are same-symbol sell stops, GTC, not extended-hours, with zero
  fills and the same positive integral quantity as the reconciled owned long
  position. The new price does not loosen the old price.
- A single terminal entry, cumulative bought/sold quantities, payload quantity,
  and a current broker position agree. No same-signal order is `CLAIMED` or
  `UNKNOWN`; no active competing exit, partial sale, or cancel remains.
- A queued partial-sale intent is not already fulfilled by observed partial
  fills. Native sync must reconcile that transition first.
- The signal is an ordinary owned `OPEN` position, not a manual or
  recognition-only holding. Existing PTCT recognition exclusions stay in place.

The pinned store scopes `orders()` by account and signal but does not return
its separate lineage column. The adapter therefore derives client identities
and verifies the durable PATCH target rather than changing the native SELECT.

Malformed, quantity-changing, loosening, unlinked, filled, unknown, ambiguous,
or competing states do not receive this benign deferral. Original management
semantics remain in force. Unexpected multiple stops or unresolved orders
also block completed-bar writes before any save.

## What deferral does and does not do

For the exact verified pair, `manage` records `STOP_REPLACEMENT_PENDING` and
returns. It does not place, replace, cancel, claim, save, clear an exit intent,
or declare protection confirmed. This leaves unrelated valid positions
eligible for their existing management.

The actual monitor calls `completed` between ticks, so that method independently
reads current scoped durable orders on every invocation. The same verified
pair defers its completed bar without advancing `last_completed_session`,
changing pending intent, or suppressing later catch-up. There is no cached
symbol skip flag. Once native sync observes the predecessor terminal, ordinary
manage/completed behavior resumes unchanged, including quote-safe stop upkeep.

Native sync still performs factual fill and broker reconciliation. Both
durable stop records and all history remain intact. The unchanged native
protection report remains unconfirmed while both orders are active. The entry
preflight still blocks new buys on pending stop transitions. Overall monitor
health may therefore remain blocked even though unrelated protection continues.

No strategy definition, risk threshold, position cap, scheduling rule, paper
account restriction, or entry-window policy is changed. No manual broker
operation is part of validation or activation.

## Verification and rollout boundaries

Run the new pending-replacement tests, existing ownership/lifecycle and entry
preflight tests, native execution regressions, and
`python -m scripts.docs_consistency_check` before publication. Added fixtures
hold exact source-extracted quote-safe and protection-report functions with
hash assertions; existing native engine/rules fixtures are unchanged.

A rollout must bind the new module and installer in a fresh immutable release,
retain all 69 native pins, and verify source checks during verify-only startup.
Before activation, independently verify fresh durable records establish the
specific PATCH lineage. After activation, verify genuine monitor cycles,
unchanged recognition exclusions/history, and honest unresolved protection
reporting. Synthetic test success must never be substituted for those checks.
