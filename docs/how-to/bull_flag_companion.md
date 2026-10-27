# Bull flag companion strategy

`bull_flag_breakout_v1` adds paper testing alongside
`trend_pullback_reclaim_qualifying_close_v2`. Pullback/Reclaim stays the default
scanner, canonical Screener batch and first executor priority. Its predicates,
historical kernel and logical definition are unchanged.

The market-wide daily producer completes and archives Pullback/Reclaim first.
The companion reuses its exact SIP/USD raw and split capture, action evidence,
calendar and universe, without additional provider requests. It writes under
`<producer-root>/<session>/companions/bull_flag_breakout_v1/scan` and uses a
distinct immutable archive ID in the existing `jbravo_research.research_runs`
table. The primary completes its original publication before it verifies,
backtests and publishes a separate companion record there. There is no schema
migration or replacement of canonical candidate rows. Companion failure leaves
the primary result intact and unavailable companion inputs cannot authorize buys.

The detector requires a 3–10 session pole gaining at least 8%, followed by a
3–10 session sideways or falling flag that retraces at most half the pole.
Average flag volume must be at most 80% of pole volume. A confirmed breakout is
a bullish close over resistance by 0.1%, at most 5% extended, with volume at least
1.5 times the preceding 20 sessions. Features use completed split-adjusted bars;
orders and exits use bound raw prices. Each proposal retains its original pattern.

Entry follows the existing next-session paper limit-order workflow. The raw
entry limit must lie strictly between the structural stop and measured target.
The stop is 0.1% below the flag low; the target is confirmation close plus pole
height. Existing stronger protection, trailing, time and indicator exits can
close earlier. Bull flags have a full target exit and no generic percentage
partial. Replay resolves bars touching both stop and target conservatively with
the prior stop first. Runtime targets require a fresh observed bid; protective
stops, cancellation and fill reconciliation use the existing durable engine.

Both strategies share the account lock, durable order claims, four held/pending
symbol slots, and three new entry attempts per session. Pullback/Reclaim runs
first; bull flags receive 5% of remaining available cash per entry. Held or pending
symbols have one owner across strategies. Restarting a worker does not restore
the daily entry budget. Separate handoff hashes, source bindings, Backtest and
Metrics IDs identify bull-flag evidence in execution reports.

`config/strategy_companions.json` controls this additive route. Missing or disabled
configuration omits it; changes require a new source-bound deployment. Existing
signals retain frozen lifecycle bindings: deployment must attest the native paper
account is flat with no pending orders before changing lifecycle code. The daily
launcher must be replaced with a newly pinned launcher, and the existing monitor
must restart. All current pipeline source bindings require a genuinely fresh scheduled
producer and primary before new entries. Old reports and consumed runs cannot be
rebound or replayed. No frontend reload or forced order is needed.

This is an exploratory paper strategy, with `research_qualified=false`. Synthetic
regressions verify integration and execution behavior; they do not establish
profitability or a future live signal. The primary dashboard remains the
Pullback/Reclaim view. Companion publication status is recorded separately at
`reports/strategy_companions/bull_flag_breakout_v1/latest.json`.
