# Completed-session signal handoff

This source contract connects the primary workflow to ordinary DB-backed entry
orders. Ranking, score filters, allocation/ATR sizing formulas, position limits,
paper/live account selection, protective sells and reconciliation are unchanged.
This is local source verification, not deployment or operational acceptance.

## Producer and completion

`python -m scripts.primary_pipeline` fixes `--session-handoff` and starts a new
explicit invocation before importing/dispatching the pipeline. Its saved active
invocation invalidates the previous completion, including if the new run fails
or stops before producing a report. The existing single-flight shell launcher
remains required. Direct session-mode pipeline calls cannot create an eligible
entry handoff without this primary-controller completion.

The producer requests the exchange calendar for ten days before/after its NY
date and freezes the returned sessions. It chooses the latest returned session
on or before that date, **not** yesterday because today's begun market is unfinished.
Before 04:00 ET, today's session has not begun, so the prior completed session
can still supply today's opening across local midnight.
For a standard 09:30–16:00 ET regular session the declared completed extended
cutoff is 20:00 ET, with a fixed 15-minute data-arrival allowance. The producer
must start at/after 20:15 ET and finish before 04:00 ET of the next returned
exchange session. Weekends, returned holidays and DST follow calendar dates
and `America/New_York`, rather than a weekday approximation or fixed UTC hour.
Early-close/nonstandard signal sessions block with
`extended_session_cutoff_unqualified`; this implementation does not guess their
extended-hours cutoff. Missing/duplicate/unsupported calendar data also blocks.
The returned calendar is provider evidence, not independently verified complete
exchange-calendar coverage. The allowance is a policy, not a finality guarantee.

The frozen signal date is passed to screener, backtest and metrics through the
existing run-date injection. In handoff mode daily-window start is NY midnight
and end is that session's 20:00 ET, converted to UTC. Calendar/window failures
cannot use the screener's weekday/UTC fallback. Non-handoff data-window callers
retain their legacy endpoints. Feed, adjustment, strategy and ranking settings
are not changed. The existing SDK/provider data path and its limitations remain.

After all stages finish, the producer reads the **specific scored candidate
batch** via the shared DB query in a repeatable-read, read-only transaction.
It binds all loaded fields/types and optional-field presence with a SHA-256
digest and freezes each symbol's supplied daily-bar `close`, timestamp, field
and available bounded source label (unrecognized labels remain explicit).
Candidate run date must equal the signal session; the daily bar must carry its
NY-midnight label, rather than an intraday/future timestamp. Batch timestamp and count must match this invocation's
coverage. Duplicate symbols, missing/stale bars and unusable prices block.
No ranking/filtering operation is introduced by freezing. A digest is retained,
not a complete archival copy of the DB rows; later DB changes block their reuse.

The pipeline saves its immutable health report/history. **Immediately after
successful pipeline completion**, the primary wrapper runs postflight against
that invocation, checking required stages, model/monitor controls, coverage,
source bindings, session window and snapshot. Only an `ok` result produces a
saved completion record bound to the exact report. A previously scheduled
standalone postflight remains a read-only secondary check; its scheduled time
does not establish producer completion or unlock execution. Recording failures
cannot produce an eligible handoff. Control files use readback and atomic
replacement; filesystem races and cross-host delivery are not qualified here.

## Executor opening and price policy

Ordinary entries require the session report, immutable history, matching active
invocation and completed postflight. Legacy same-UTC-day/12-hour reports remain
inspectable for historical diagnostics but cannot authorize ordinary entries.
The bound report replaces wall-clock same-day freshness: a Friday handoff can
target Monday, while expiring at that target session's regular open (09:30 ET).
Execution must be at/after 04:00 ET and before expiry. Preload may occur before
04:00 ET; its wait uses the **bound execution date**, not simply today's time.
Default `--submit-at-ny` is `04:00`, premarket bounds are 04:00–09:30 ET, and
default `--price-source` is `signal`. Ordinary entry overrides requiring other
opening/price/window policies or disabling extended eligibility block explicitly.
Diagnostic, reconciliation and path dry-run gates retain their existing roles;
legacy price modes remain explicitly available for those diagnostic paths.

Candidate loading selects only the handoff's run date and batch timestamp.
It cannot select the latest older batch. Full typed values must equal the
producer binding before filtering/ranking; raw, filtered and ranked-frame
bindings are rechecked through the existing gates. Submission checks again at
the actual BUY boundary, including direct and replacement/chase callers.

The BUY anchor is the frozen calculation `close`, never a newly fetched previous
close, quote or trade. Its initial passive limit is
`floor_to_cent(close * (1 + min(limit_buffer_pct, max_gap_pct) / 100))`.
Defaults retain the existing 0.5% buffer and 3% maximum premium; they are not
calibrated execution costs or a promise of a fill. Finite 0–100% policy inputs
are required. A $10 reference gives $10.05 initially and a $10.30 maximum;
an opening ask above the limit may leave the order unfilled. Marketable-gap and
tick-correction paths cannot bypass the frozen cap. Every BUY/replacement is
checked against the cap and session window after tick normalization. Existing
chase decisions can therefore be rejected instead of replacing above the cap.
Changing limit price can change computed quantity, but sizing logic is unchanged.
SELL/protective/reconciliation behavior is not changed by these entry checks.

The existing submission-relative cancellation timer and polling detach remain.
This change does not establish cancellation completion after polling detaches.
It does not cancel already resting orders at the handoff expiry; expiry prevents
new entry/replacement submissions. Exact 04:00 dispatch, fill price, liquidity,
SIP entitlement and live broker behavior remain unqualified.

## Daily close meaning and deployment gates

Alpaca daily bars use NY-day labels and trade-condition rules. Extended-hours
trades can update volume without updating daily close. A 20:00 request boundary
selects bar labels; it does not reconstruct intrabar OHLC or prove last-trade
coverage. The frozen value is expressly `supplied_daily_bar_close`, **not an
observed 20:00 last trade**. Late corrections, historical actions, model/data
provenance and research readiness remain qualified. A strategy using an actual
extended-hours closing price requires a separately reviewed data/feature change.

Sources: [Alpaca aggregation rules](https://docs.alpaca.markets/us/docs/market-data-faq)
and [extended-hours order sessions](https://docs.alpaca.markets/us/docs/orders-at-alpaca).

No PythonAnywhere schedule is changed by this packet. A future cutover must
prime execution before 04:00 ET (08:00 UTC in daylight time, 09:00 UTC in standard
time), account for UTC scheduler/DST behavior, and measure actual invocation and
completion. Source review, publication, merge, deployment and schedule cutover
remain subsequent gates. No source self-test is independent acceptance or
permission to place orders.

## Offline verification boundary

`tests/test_signal_handoff.py`, `tests/test_pipeline_postflight.py` and
`tests/test_executor_preflight_health.py` use synthetic calendars/candidates,
temporary storage and blocked external transports/processes. They exercise the
actual inert modules, primary import/caller, normal imported executor, DB query
selection with mocked connections, sizing/price planning without orders, and
submission gates with mocked transport. The legacy pipeline's publication hook
and parser are extracted from its actual AST; its complete initialization and
real ML stages are **not** exercised. No provider, production DB, host scheduler,
strategy evaluation or broker order is executed.
