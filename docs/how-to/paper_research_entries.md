# Paper strategy entries while research collection is pending

The scheduled executor may opt into `--paper-research-entries` for strategy tests.
This option defaults off and requires the exact HTTPS Alpaca paper trading origin.
Live, missing, credential-bearing and lookalike endpoints are rejected. The
pipeline retains `pipeline_only_no_new_entries`, `entry_ready=false` and the
original research warnings: paper testing does not qualify a model for live use.

The option waives only the intentional pipeline entry-policy refusal after a
successful, source-bound, completed primary run. Missing or mismatched completion,
operational failures, unknown ML alerts, stale predictions, changed candidates,
expired execution sessions and price-cap violations still block entries. The
executor repeats these checks immediately before each buy submission. Existing
allocation, order-count, liquidity, price, buying-power and stop controls apply.

Use the existing scheduled pre-market executor with this one additional flag.
Its 04:00 Eastern opening and session expiration do not change. Deploying new
source invalidates old source-bound reports until the next successful primary;
do not rewrite old reports or replay an expired session to force same-day trades.
Research collection remains independent and must roll to new source identities
before its first session whenever pinned source files change.

Monitor refreshes retain observed entry metadata for continuing positions. After
an exit attempt, newly disappeared symbols are processed with retained metadata
before the refreshed snapshot replaces it, including late fills and concurrent
closures. Missing entry times remain unknown rather than being fabricated.
