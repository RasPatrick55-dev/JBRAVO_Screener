# Strategy identity in the backtesting stage

The primary pipeline passes a mandatory `strategy_contract` from the verified
Screener archive into `scripts.strategy_backtest.run_candidate_backtest`:

- `strategy_id`: the exact registry key, currently
  `trend_pullback_reclaim_qualifying_close_v2`.
- `strategy_version`: currently `2`.
- `definition_sha256`: the fingerprint of the versioned rules and their source.

Each proposal also retains its strategy ID, version and stable signal ID. The
backtester selects the registered strategy from this contract and verifies the
version, definition, declared study method and every proposal before simulation.
Missing or unknown identities and mixed strategies fail. It never substitutes
legacy composite-score entries or silently chooses a default strategy.

The stage calls the same `evaluate_strategy` and strategy feature preparation
used by the Screener. It replays the declared historical cohort with the original
portfolio, exits, fees and slippage, then attributes completed historical signals
to the current screened symbols. Simulating only today's selected stocks across
the past would change ranking competition, available cash and entry state.

The current declared episode begins January 2, 2026, using a January 2023 warmup
capture, and ends at the signal session. The backtest uses SIP/USD split-adjusted
features and raw execution references. Signals can fill only at the supplied next
regular-session open. No later price bar is used to invent an outcome for the
current buy proposal. Retrospective data and the selected cohort remain research
limitations; this stage does not establish untouched validation or qualification.
For the [market-wide daily capture](market_wide_strategy_screen.md), the full
current population is hash-bound, data exclusions are explicit, and newer
listings start evaluating once the same 250-session feature warmup is ready.
The same sparse replay retains cross-symbol portfolio competition and all rows
that can affect entries/exits. Compact universe provenance keeps reports bounded.

An empty Screener batch resolves and records the strategy but reports
`SKIPPED_NO_CANDIDATES`. A nonempty batch reports
`COMPLETED_CANDIDATE_BACKTEST`. Partial exit fills are grouped into one completed
signal; still-open positions and unfilled proposals stay separate. Unknown win
rate or expectancy remains null when there are no completed signals. Trade P&L
includes modeled fees/slippage and excludes cash dividends, which remain in the
separate cohort portfolio simulation metrics. Per-symbol drawdown is labeled
in USD of closed-signal P&L, not as a standalone account equity percentage.

This first connection reports results without applying new performance filters.
The same candidate/signal IDs continue to Metrics. Backtest results are bound to
the primary invocation, Screener selection hash, archive, strategy contract and
dataset. They are committed alongside the canonical batch in its existing JSONB
payload, and included in primary postflight. No additional database migration is
needed. Missing or mismatched backtest results prevent publication.

[The Metrics stage](strategy_metrics_stage.md) now consumes this exact bound
backtest report, preserves its strategy/signal/result identities and publishes
closed-signal statistics in the same transaction. It does not use the archived
research summary as a substitute.

The dashboard backtest API exposes the actual strategy identity, result ID,
evaluation period and per-candidate report. It does not borrow earlier runs,
legacy backtest rows or CSVs. `STRATEGY_EPISODE` identifies the actual study
window; a UI window parameter does not redefine this frozen replay. Old batches
without a bound strategy backtest remain explicitly pending.

This stage does not submit orders or change the paper executor. The separate
paper execution route retains its explicit premarket policy and pending
premarket performance validation.
