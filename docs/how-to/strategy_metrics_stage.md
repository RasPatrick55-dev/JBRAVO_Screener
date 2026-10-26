# Strategy Metrics stage

The primary Pullback/Reclaim route derives Metrics from its current candidate
backtest. `scripts.strategy_metrics.build_metrics` consumes the exact Screener
selection and `jbravo.candidate-strategy-backtest.v1` result; it never reruns a
strategy, reads legacy `backtest_results`, or uses broker trade performance.

The handoff requires the same primary invocation, signal session, candidate
signal IDs, strategy ID/version/definition fingerprint, research archive and
scan. Metrics binds both the backtest result ID and full backtest content hash.
Every per-stock row must match its proposal and reconcile closed-trade counts,
win rate and expectancy. Partial exits remain one completed signal as defined
by the backtester. A different strategy, stale result, altered report or invented
current-signal outcome stops publication.

Output schema is `jbravo.candidate-strategy-metrics.v1`. Rows report closed
trades, wins/losses, win rate, net trade P&L, expectancy, profit factor, cash
drawdown and costs from the bound backtest. Win rate uses percentages; P&L,
expectancy and drawdown use USD. Cash dividends are excluded from trade P&L;
these figures describe attributed historical cohort trades, not independent
single-stock portfolios or broker account returns. Metrics adds no ranking or
performance rejection rule. Candidate identities and counts pass through.

An empty Screener batch preserves the strategy identity and yields
`SKIPPED_NO_CANDIDATES` with no metric rows. A stock without completed history
has zero closed trades and nullable win rate/expectancy/profit factor. A current
proposal remains unfilled with unknown future outcome. Profit factor with no
losses remains undefined rather than infinity. Portfolio Sharpe, Sortino and
percentage drawdown are not fabricated from closed-signal cash P&L.

The selection, backtest and Metrics reports are published in one transaction in
the existing `public.screener_strategy_batches_app.payload` JSONB field. No
additional table or DDL is needed. PostgreSQL is authoritative; the legacy
`metrics_daily` account-performance records and unversioned top-candidate
performance columns are separate. No CSV fallback is used.

The primary report includes the same Metrics object and its actual stage
status. Source binding includes `scripts/strategy_metrics.py`. Postflight and
the dashboard revalidate the report against its bound backtest. Earlier batches
without Metrics are explicitly pending and cannot borrow another run's data.

`/api/screener/metrics` exposes the strategy contract, backtest/metrics result
IDs and current rows. It accepts `all`, `with_history` and `no_history` display
filters plus symbol/strategy/history text search; display filters do not change
the published candidate population. The Metrics card shows historical performance
and retains unknown values as `--`. It does not infer gate passes or confidence.

This remains paper-only research reporting. Strategy qualification, production
model promotion, schedules and broker order authority are unchanged. The existing
premarket executor continues refusing next-regular-open proposals.
