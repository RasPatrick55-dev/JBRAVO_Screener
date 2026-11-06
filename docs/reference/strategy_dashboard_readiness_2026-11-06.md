# Strategy dashboard readiness repair — November 6, 2026

The hosted candidate API hid a valid completed PostgreSQL selection after its
12-hour execution freshness limit. It also bound historical display to current
source bytes, so a later compatibility deployment could hide earlier results.
The response initialized `entry_ready=false` even on a successful read. Meanwhile,
the dashboard process cards preferred legacy task logs and displayed October
dates instead of the current primary and premarket executor receipts.

The repaired reader keeps the latest exact PostgreSQL batch, candidate-population
checks, repeatable read-only transaction, immutable run history/completion,
parent/child hashes and reconstructed unified ranking. Control files identify
and verify database records; they never supply substitute candidate rows. A
verified historical selection remains displayable after its entry window expires
or display source files change. Failed/missing/changed lineage still fails closed,
without a CSV, earlier nonempty batch or primary-family fallback.

`selection_ready` means the displayed unified selection verified successfully.
`entry_ready` now describes prepared paper handoff readiness, separately labeled
`PREPARED_PAPER_HANDOFF_ONLY`. Existing source, freshness, population and execution
preparation checks still apply. Expired preparation, changed current sources and
empty selections report explicit block reasons. Broker preflight remains required;
the API never grants order-submission authority. Execution and ranking rules,
model identity, position limits and the 04:00 New York entry policy are unchanged.

For the enabled strategy route, health and process endpoints use current primary
and immutable premarket receipts. They report timestamps, elapsed time, actual
return code and stage results. The UI explicitly distinguishes completed, failed,
pending, stale and unavailable receipts. Missing current evidence cannot borrow
an older legacy log. Process completion remains separate from financial or entry
acceptance; stored connection-check freshness stays independently labeled.

An isolated native read-only diagnostic confirmed the current PostgreSQL candidate
population and unified selection and identified `strategy_postflight_stale_or_future`
as the display failure. A second isolated native process exercised the proposed
readers against that database: candidates/backtest/metrics each returned two
verified selected rows, both current process receipts reported completion on
November 6, and the health timestamp matched the current primary. Entry readiness
remained false with `execution_premarket_preparation_expired`, as required after
the entry window. Neither process wrote to PostgreSQL, called a broker or ran a
producer, primary or executor job.

Local validation: 356 frontend tests and the production TypeScript/Vite build
passed. The focused strategy/readiness/execution/postflight suite passed; two
unrelated dashboard API assertions (unknown prices expected as zero, and legacy
P&L precedence) reproduce against unchanged main and are outside this repair.

The proposed hosted rollout is a new byte-bound overlay, preserving hosted Git
HEAD `5c402be1cbcf674a6f415261729110f86a3b5e0b`, branch and existing overlays.
It requires reviewed source/asset preimages, preserved runtime pins, new fixed
launcher pin manifests, command-only rollover of the existing producer/executor,
one web reload and independent readback. It does not rewrite historical receipts,
publish candidates, run pipeline/trading jobs, change dependencies, alter schedules,
edit private configuration or relax paper-only and PostgreSQL authority.

The exact review and native evidence are retained locally under
`C:/Users/RasPa/PythonAnywhereReviews/2026-11-06-strategy-dashboard-001/`.
