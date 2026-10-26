# Pullback/Reclaim execution handoff

This is an execution review and a read-only input adapter. Production activation
is pending a strategy-aware regular-session worker and durable lifecycle store.
The existing 04:00 Eastern passive-limit executor remains blocked for these
signals. Do not remove its guard to activate this adapter.

`scripts.strategy_execution_handoff.load_handoff` verifies the source-bound,
completed primary report and reads the exact latest published PostgreSQL batch
in one read-only REPEATABLE READ transaction. It requires identical current
candidate values, strategy ID/version/definition, archive, primary run, backtest
result and Metrics result. An empty batch stays empty. There is no CSV fallback,
newest-nonempty lookup, legacy model reranking or unapproved performance filter.

The standalone module prints the handoff identity and stage result identities;
it never submits orders, changes tables or enables a scheduled task. Imports
perform no connection or environment discovery. PythonAnywhere execution must
use the shared MCP's reviewed operation contract and a clear serial window.

## Backtest rule contract

The owner selected the backtest's sizing and exits for eventual paper execution:

- Sequential 5% of remaining cash, whole shares, at most four total positions.
- Next supplied regular-session open; signal-session and DST offsets come from
  the calendar. Premarket references and expired signals cannot size entries.
- Initial protection uses actual fill price, 3% trail and signal-day ATR.
- Completed daily bars advance ATR/trailing protection and queue full or partial
  exits using the unchanged `PortfolioBacktester` kernel. Close-derived signals
  cannot fill on that same completed bar. Full exits take priority over partials.

The pure order planner requires a fresh regular-session price within five
seconds and restricts attempts to the first minute of the supplied session.
This is a proposed execution approximation: an actual broker fill cannot be
guaranteed to equal the backtest's OHLC opening price. Premarket MOO quantity
sizing cannot know that future price. Record actual timing, reference, quantity
and fills alongside modeled results; do not claim exact auction-price parity.
Provider conventions for future price references are raw SIP/USD equities,
with actual observation time retained; no delay/feed substitution is allowed.

`submit_claimed` is an inert port contract for a future worker. It requires an
explicit matching paper account and canonical paper origin. The store must
commit a unique account/signal claim **before** one order POST. On uncertain
transport, a later call observes the same broker client ID and never sends a
second POST. The same signal ID survives primary reruns. This port does not
implement a production database outbox or grant trading authority.

## Errors repaired in this candidate

The existing executor no longer automatically repeats ambiguous order POSTs.
A timeout or server error may conceal an accepted order, so it records an
unknown outcome and blocks further buys in that process. Confirmed sub-penny
validation rejection can still be corrected and retried with fresh eligibility.
Restart-safe recovery still requires the durable strategy outbox.

The position monitor now raises an unavailable-snapshot error when its broker
lookup fails or returns a malformed population. It cannot treat that failure
as a flat account, publish a healthy empty snapshot, or record false closures.
A successfully observed empty list remains valid.

## Activation requirements

Integrate a durable PostgreSQL intent/order/position lifecycle with the entry
task and the always-on monitor, including partial fills, confirmed stop coverage,
cancel-before-sell confirmation, actual position ownership, completed-bar input
refresh, restart recovery, and calendar-based scheduling. Existing policy locks,
accounting/recovery exclusions, cash limits and explicit paper testing authority
must remain in force. Research qualification and live entry readiness remain
separate; neither is promoted by this handoff.

Deploy only in the serial window released by the dashboard/accounting recovery
coordinator. No migration, task change, monitor restart, broker submission or
replay is authorized merely by importing or reviewing these helpers.
