# Complete strategy screening and unified handoff

The October 28 run exposed two separate problems. Four required pattern screens
failed with `RuntimeError`, but the producer still wrote a completion marker.
The primary published Pullback/Reclaim and its Backtest/Metrics, ignored the
unavailable unified selection, and returned success. The executor correctly
refused the incomplete selection. Those historical exception receipts lack a
reason or phase, so an exact cause cannot be reconstructed from them.

## Shared preparation

The repaired producer loads and verifies one complete market capture. It visits
each symbol once and calls all six existing preparation adapters while that
symbol's raw/split history remains in the bounded dataset cache. This avoids six
full manifest validations and six passes loading the same history. Rule adapters,
sparse-row policy, complete current decisions and portfolio rules stay unchanged.

`<session>/prepared/manifest.json` binds compressed input chunks, full symbol
population, capture provenance, folds, portfolio settings, preparation sources
and strategy definitions. The study's optional `prepared_market` reference binds
that manifest's exact hash. Chunks preserve frame types, full float precision,
nested entry evidence, dividends and current decisions. No pickle is used.

Scanner and Backtest read the same verified sparse **inputs** and independently
run the original portfolio kernel; Backtest does not copy Scanner outcomes.
Missing, modified, differently scoped or source-mismatched inputs fail without
stale-cache or raw-history fallback. Primary companion validation also reuses its
verified market snapshot while separately verifying each archive and scan artifact.

Preparation uses at most 25 symbols per output chunk, a 32 MiB decompressed chunk
ceiling, and a 2 GiB total compressed-input ceiling. It stays within the existing
3,600-second preparation and 7,200-second whole-producer deadlines. Pattern replay
keeps 600 seconds; Bull Flag keeps 1,800 seconds. Manifest timing records actual
per-strategy preparation seconds. Bounds are failure limits, never truncation
rules. This change does not delete history or establish a retention policy.

## Completion and blocked states

Every enabled strategy must finish its scan and archive, including honest empty
results. Otherwise the producer saves `blocked.json` and `failure.json`, returns
nonzero, and does not save `completion.json`. Committed individual archives remain
partial evidence. Failure receipts include phase, elapsed/remaining time and
deadline/STOP codes without retaining arbitrary exception messages.

The primary snapshot identifies its completed stages as
`PRIMARY_STAGES_COMPLETE_PENDING_UNIFIED_SELECTION`. Companion and unified
PostgreSQL records bind its hash, so it cannot be rewritten afterward. The primary
must complete all companion Backtest/Metrics and unified publication before
returning zero. Otherwise it reports `PRIMARY_STRATEGY_SCREEN_BLOCKED` and nonzero.

Postflight requires the unified control receipt's exact selection ID and parent
report hash before reporting success. Only then can the normal wrapper create
`completed-primary.json`. The executor still reconstructs the actual committed
PostgreSQL selection, complete strategy population, ranking and source bindings;
a control receipt alone cannot authorize a purchase.

An entry exception replaces any stale strategy receipt with a fresh blocked one.
Before broker construction it reports zero orders and `NOT_CONTACTED`. After
broker contact it reports unknown counts and `NOT_ESTABLISHED`, requiring
reconciliation. The monitor distinguishes `entry_sources_ready` from
`unified_selection_ready`. Missing selection blocks entry readiness while the
owned-position lifecycle continues under its original source guards.

## Verification and rollout

Tests compare all six original and prepared-input replays, including positive
synthetic examples for all four patterns, with exact candidates, fills, lifecycle
and metrics. They also cover tampering, full symbol accounting, deadline
diagnostics, blocked completion and fresh entry receipts. These are software
tests, not profitable-strategy or completed-purchase evidence.

Native rollout requires a reviewed source package, **66** runtime bindings,
a replacement immutable worker with **28** source/configuration pins,
preservation of current dashboard/model files, and the established flat-account
and empty durable-ownership gates before lifecycle source changes. Existing
schedules and risk settings stay unchanged.

Before activation, test frozen code in an isolated host directory over an existing
complete capture. Measure preparation, all six scans and candidate Backtest/Metrics
against existing deadlines without DB publication or provider/broker mutations.
A synthetic fixture cannot prove full-market timing. Native proof and deployment
must be recorded separately; this guide establishes neither. Use a fresh claim.

After activation, require a new scheduled signal session, six complete screens,
exact-session committed unified publication, completed primary and proper future
entry window. October 27 signals cannot be bought after their October 28 window.
A valid zero-signal scan remains an honest no-buy result. A submitted or canceled
limit order is not a filled purchase.
