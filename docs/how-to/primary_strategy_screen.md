# Pullback/Reclaim in the primary pipeline

The enabled `config/strategy_pipeline.json` route connects the primary scheduled
pipeline's Screener to `trend_pullback_reclaim_qualifying_close_v2`. The daily
producer runs at 01:20 UTC; the existing primary job runs at 03:45 UTC. The
primary job obtains a fresh exchange calendar and requires the exact completed
session's PostgreSQL research archive. Missing, stale, revised, incomplete or
source-mismatched archives fail without reverting to legacy picks.

The initial cohort remains ALK, FN, HMY, LOGI, MIRM, NVST, PLUG, SKY, SRRK and
STLD, with SPY as benchmark. Data is Alpaca SIP daily data in USD: raw prices
for price references, split-adjusted prices for strategy features. The scan
assumes availability at 20:15 Eastern and models entry at the next supplied
09:30 Eastern regular open. This is a paper testing candidate; research
qualification is still pending. The primary adapter performs no additional bar
requests, ranking, soft-gate relaxation, ML filtering or filler selection.

## Candidate publication and tracing

One transaction appends `public.screener_strategy_batches_app` and publishes the
selected symbols into the existing `screener_candidates` and `top_candidates`
tables. The batch header exists even when there are zero buys. Both canonical
`latest_*_candidates` views select that header's exact date and timestamp;
historical rows cannot reappear because an empty day has no candidate rows.
The migration in `scripts.strategy_pipeline_store.SCHEMA_SQL` must be applied
explicitly by the database owner before enabling the primary route. There is no
automatic DDL during a scan. Preserve the previous view definitions for rollback.

Each candidate's `score_breakdown` contains `jbravo.strategy-candidate-lineage.v1`:
the complete proposal (including stable signal ID, strategy/version, availability
and intended opening time), archive ID/hash, scan batch ID, dataset hash and
strategy definition hash. The immutable batch payload and primary postflight
handoff contain the same proposals. The adapter verifies all scanner artifacts,
the source hashes and the original dataset, then independently reads back the
committed candidate batch on a new read-only connection.

Repeated primary preparations retain the same signal IDs and archive binding.
A revised archive for an already published session requires review. Publication
serializes on an advisory lock; a failed primary invalidates the earlier primary
completion before dispatch. There are no new broker orders during verification.

## Reading the result

`reports/pipeline_postflight/latest.json` uses
`jbravo.primary-strategy-screen.v1`. `BUY_SIGNALS` means the archived proposals
were published; `NO_BUY_SIGNALS` is a successful zero-candidate screen. Backtest
and metric fields refer to the verified archived research simulation. Legacy
label generation and ranker evaluation are explicitly marked unused. Simulation
P&L is never published as actual broker performance.

The postflight separates `screen_ready` from `entry_ready`. The current executor
implements a 04:00 Eastern passive-limit policy, so it explicitly refuses these
next-regular-open proposals, including when paper research entries are enabled.
The signal-to-primary-Screener-to-database-to-handoff connection is available;
broker entry integration requires a matching regular-open paper executor. No
live trading permission or strategy qualification follows from this connection.

## Rollback

Keep the immutable batch history. Under the existing primary lock, restore the
saved candidate-view definitions and the reviewed previous source commit as one
controlled rollback. Do not merely remove the route configuration while leaving
the batch-aware views in place. Preserve the existing dirty dashboard/generated
files, daily producer task, paper executor task and production model bytes.
