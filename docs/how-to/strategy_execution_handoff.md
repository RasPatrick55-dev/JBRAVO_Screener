# Pullback/Reclaim premarket execution handoff

The intended cycle is to collect completed data after 20:00 Eastern, retain the
data-arrival allowance through 20:15, and submit at the next trading session's
04:00 premarket open. The existing executor task already targets 04:00.

The existing executor and monitor include an opt-in strategy worker, with
PostgreSQL ownership and a durable order outbox. Activation of both tasks was
verified on October 26 and their enabled definitions were rechecked October 27.
**An activated task does not mean its current inputs are entry-ready.** The
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

`strategy_execution_engine.Engine` uses `strategy_execution_store.Store` and
the bounded native `PaperBroker`. The actual native account ID is verified;
an account from a separate MCP connection is not assumed to be the same account.
Each unique account/signal/role/sequence claim commits before one POST, PATCH
or DELETE. Uncertain submissions are recovered by client ID without repeating
the mutation. Explicit permanent refusals remain durable and cannot become a
fresh buy. Position absence does not close a signal; cumulative verified fills
and the broker position must reconcile. Entry deadlines are reconciled before
quotes, so a quote outage cannot prevent cancellation.

The monitor owns only enrolled strategy positions, bypassing the legacy exit
and continuous trailing rules for those symbols. It advances the shared kernel
once per completed daily session, confirms cancellation before selling, and
keeps partial-exit targets across restarts. Fixed GTC stops cover regular hours;
fresh SIP quotes drive extended-hours limit exits. Limits can remain unfilled.
Reports count native coverage only during regular hours and only for the exact
remaining owned quantity. With strategy monitoring enabled, the existing loop
sleeps at most ten seconds during 04:00–20:00 and thirty seconds outside it.

The executor requires a fresh monitor receipt for the same native account and
source bindings before buying. Database policy identity/revision and source
pins are checked again before broker mutations. New entries bind all producer
sources; existing positions bind their frozen exit implementation and strategy
definition, so unrelated dashboard/producer edits cannot pause protection.
Existing schema/accounting
policies are preserved. The two new strategy tables retain lifecycle state,
strategy/backtest/Metrics identities, raw reference bars and broker observations;
these records do not assert fee-complete financial performance.

## Errors repaired in this candidate

The existing executor stops automatic retries after ambiguous order POSTs and
blocks further buys in that process. Confirmed sub-penny validation rejection
can still be corrected and retried with fresh asset eligibility. The opt-in
worker additionally retains claims across process restarts.

The monitor raises an unavailable-snapshot error on failed or malformed broker
position reads. Those failures cannot publish a flat-account snapshot or record
false closures. A successfully observed empty list remains valid.

## Activation requirements

The initial deployment installed both strategy state tables. Preserve that
schema and its historical state; do not rerun the migration to refresh inputs.
The following requirements also apply to future reviewed activations.

Install the two new tables through a reviewed migration; deploy the exact
reviewed source bytes while preserving shared native edits and dashboard assets.
The primary must then produce a new completed report binding all 56 sources
required by the current market-wide route. The earlier 54-source pilot report
does not satisfy the deployed source guard.
Do not rewrite an existing report or backdate a run to make expired input valid.
An after-09:30 run cannot renew the prior signal session; wait for the next
completed session and its normal primary run.

Start the existing monitor with explicit process-scoped paper settings and
`JBR_STRATEGY_EXECUTION=true`, and verify its fresh bound receipt before enabling
`--strategy-premarket-entries` on the executor. Keep explicit
`--paper-research-entries`, DB source, 5% allocation, four total positions,
three new positions, whole shares, no minimum-order bump, no chase, 0.5% limit
buffer, 35-minute cancellation and 04:00 Eastern submission. Existing normal
start/end reconciliation remains required. No private environment-file edit
is needed. A dry run or zero-signal inspection opens no strategy broker adapter
or writable strategy store; normal executor reconciliation is separate.

Premarket entry performance remains unqualified until a versioned minute-data
limit-fill replay is connected to Backtest and Metrics. The owner's explicit
paper strategy-testing mode permits collecting prospective paper observations;
it does not relabel the preserved regular-session reference results or qualify
the strategy. Missing liquidity and unfilled/partial orders remain explicit.

Existing policy locks, recovery exclusions, account/cash limits and explicit
paper testing authority remain in force. Neither research qualification nor
live entry readiness is promoted by this adapter. Deploy in the serial window
released by the dashboard/accounting recovery coordinator. Reviewing/importing
these helpers does not apply a migration, restart a task or submit an order.

## Inspect entry and monitor evidence

Read these bounded, nonsecret files under `/home/RasPatrick/jbravo_screener/`
through the shared PythonAnywhere MCP:

| Evidence | Path | Interpretation |
| --- | --- | --- |
| Current production pipeline | `reports/pipeline_postflight/latest.json` and `completed-primary.json` in the same directory | Exact-session completed primary, matching invocation, batch and Backtest/Metrics |
| Strategy entry result | `reports/strategy_execution/entry/latest.json` | Strategy identity, candidate count, order observations and entry disposition |
| Entry process receipt | `reports/executor/premarket/latest.json` and `run-<observed-run-id>.json` in the same directory | Process return code, effective risk settings and immutable matching history |
| Strategy monitor heartbeat | `reports/strategy_execution/monitor/latest.json` | Fresh account/source binding, owned symbols, events, entry readiness and regular-hours native coverage |
| Reconciliation process | `reports/executor/reconciliation/latest.json` | Separate accounting/reconciliation role; not evidence of another entry attempt |

`data/execute_metrics.json` is shared by executor roles and can be overwritten
by later reconciliation. Use the per-role receipts to identify the last buy
attempt. A declared scheduler ID is process attribution; its receipt explicitly
does not claim independent scheduler execution verification. Correlate it with
the installed task definition and actual times when reviewing a scheduled run.

Inspect the strategy state in `public.strategy_execution_signals_app` and
`public.strategy_execution_orders_app` only through an authorized read-only
database connection. Use the exact signal, strategy, account and client-order
identities; an unrelated MCP account's orders cannot prove the native task's
behavior. No DDL, historical financial edits or broker writes follow from
this inspection.

The [dated October 27 record](../reference/pullback_reclaim_verification_2026-10-27.json)
records a successful empty entry path, a healthy flat monitor, and a separate
positive isolated pipeline audit. A fresh production candidate followed by an
observed paper fill, protection and exit still requires its own evidence. The
department's pending follow-up checks the next scheduled cycle and does not
force a candidate, retry a task or submit an order itself.
