# Configuration-versioned daily research archives

Daily archive v2 identity includes the full canonical scanner configuration hash,
strategy and completed signal session. Captures live below
`runs/<session>/revisions/<primary-configuration-sha256>/`. Old session claims and
SIP research rows remain immutable. Same-configuration retries refuse their
existing revision; database publication remains insert-only, never an upsert.

The primary consumer requires the exact current configuration identity. There is
no legacy archive fallback, feed relabeling, synthetic bar substitution or relaxed
freshness gate. Logical signal/episode IDs and order duplicate protection remain
unchanged. The historical scorecard supports both archive identities, selecting
one latest completed snapshot per strategy using existing recorded-at ordering.

The new research-only `strategy_archive_companions.py` carries the existing
companion preparation body with only its archive lookup changed. The original
`bull_flag_companion.py` is byte-for-byte unchanged because it is part of frozen
position lifecycle source bindings. Its producer, handoff reader and trading
logic are not replaced or monkey-patched. The new modules belong in the full
primary/entry source inventory, not the lifecycle compatibility whitelist.

POSIX capture creation walks directories with no-follow directory descriptors;
symlinked session/revisions parents are refused. The final configuration revision
is created exclusively. This does not claim protection against all concurrent
renames after returning a Path; normal exclusive producer/source coordination
still applies. The existing explicit output/market-cache location is preserved.

Deployment must coordinate producer, executor, monitor and hourly frozen source
pins and task checksums. Entry remains disabled until fresh source-bound research
and selection handoff, account/position admission, quote freshness and opening
window checks pass. Previously adopted entry-only positions require compatibility
against the new full release bindings. Installing this patch does not authorize
orders, restart an order-capable task or replay a pipeline.

Any one-use recovery scan is separate: fixed session, IEX configuration, source
identity, expiry, provider budgets and insert-only archive scope. Preserve prior
failures and ambiguous commit evidence; never automatically replay an uncertain
capture/archive operation.
