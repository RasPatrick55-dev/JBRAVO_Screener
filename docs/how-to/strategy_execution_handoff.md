# Pullback/Reclaim premarket execution handoff

The intended cycle is to collect completed data after 20:00 Eastern, retain the
data-arrival allowance through 20:15, and submit at the next trading session's
04:00 premarket open. The existing executor task already targets 04:00.

This candidate adds a read-only input adapter and pure planning contracts.
**The production strategy worker is not connected or activated yet.** The
current v2 backtest models regular-session opening fills; it remains preserved
as a reference result and is explicitly marked `PREMARKET_REPLAY_PENDING` in
the new execution handoff. It does not establish premarket performance.

## Input and price contract

`scripts.strategy_execution_handoff.load_handoff` verifies the source-bound,
completed primary report and reads the exact latest published PostgreSQL batch
in one read-only REPEATABLE READ transaction. It binds the candidate values,
strategy ID/version/definition, archive, primary run, backtest and Metrics IDs.
Empty batches stay empty. There is no CSV fallback, newest-nonempty lookup,
legacy ML reranking or unapproved performance filter. Imports do not connect.

The standalone module prints handoff/result identities and submits no orders.
PythonAnywhere runs must use the shared MCP's reviewed operation contract and a
clear serial window.

The premarket planner requires a separately captured after-hours reference,
observed after 20:15. Reference bars must be complete historical SIP/USD data
with raw prices, timestamped from 16:00 through 19:59 Eastern for the signal
session. The latest observed close retains its actual timestamp: a 19:53 print
is not relabeled as a 20:00 trade. Missing bars remain missing. The strategy's
daily-bar close is not silently substituted for this execution reference.

The candidate plan uses an extended-hours DAY limit, capped at 0.5% above that
frozen reference and rounded down to cents. This preserves the existing
executor's default limit-buffer policy. Quantity uses 5% of remaining cash,
whole shares, and at most four total positions. It reserves at the limit price;
it cannot know the future opening price. New submissions are confined to the
first minute after the supplied 04:00 opening; existing unfilled orders expire
under the retained 35-minute cancellation policy. Fills are not guaranteed.

Calendar session identity and Eastern offsets govern timing, including DST.
The regular-market clock is normally closed at 04:00: validate its timestamp
and next regular-open session instead of requiring `is_open=true`.

## Exit and recovery contracts

Initial ATR/3% trailing levels use actual fills and signal-day ATR. Completed
daily bars advance protection and queue full/partial exits using the unchanged
`PortfolioBacktester` kernel. The owner selected this sizing and exit policy;
changing entry timing does not authorize a different indicator or exit policy.

Alpaca's native trailing stops do not trigger outside regular market hours.
Premarket protection therefore needs explicit strategy-aware monitor handling,
confirmed cancellation before replacement sells, actual position ownership,
partial-fill accounting and durable restart recovery. A native trailing-stop
acknowledgment must not be reported as premarket stop coverage. Reference:
[Alpaca order documentation](https://docs.alpaca.markets/us/docs/orders-at-alpaca).

`submit_claimed` defines a future worker's order port. It requires the explicit
matching paper account and canonical paper origin. A durable unique
account/signal claim must commit before one POST. An uncertain result is
recovered by the same broker client ID without another POST. This helper does
not install a production outbox, schedule or trading authority.

## Errors repaired in this candidate

The existing executor stops automatic retries after ambiguous order POSTs and
blocks further buys in that process. Confirmed sub-penny validation rejection
can still be corrected and retried with fresh asset eligibility. Durable
restart-safe handling remains part of the pending strategy worker.

The monitor raises an unavailable-snapshot error on failed or malformed broker
position reads. Those failures cannot publish a flat-account snapshot or record
false closures. A successfully observed empty list remains valid.

## Activation requirements

Add the after-hours reference capture to the producer, implement a versioned
premarket limit-fill replay and connect its results to Metrics, then integrate
durable intents/orders/positions and the strategy-aware monitor. Preserve old
regular-session results instead of relabeling them as premarket tests. Missing
liquidity and unfilled/partial orders must remain explicit in research results.

Existing policy locks, recovery exclusions, account/cash limits and explicit
paper testing authority remain in force. Neither research qualification nor
live entry readiness is promoted by this adapter. Deploy in the serial window
released by the dashboard/accounting recovery coordinator. Reviewing/importing
these helpers does not apply a migration, restart a task or submit an order.
