# Dashboard source and performance boundaries

Financial views read the configured Alpaca account or PostgreSQL ledger. Missing
or failed reads remain unknown; a successful empty response is a separate state.
Research strategy results retain their per-strategy/report identities and do not
qualify broker-account performance. Recorded gross summaries are before fees,
use a bounded complete closed-row query, honor policy exclusions, and disclose
unknown coverage. Physical history remains visible independently of performance
qualification. Source freshness is separate from business validity.

The optional CRC110 observed-cost display consumes a separately reviewed,
source-pinned **private runtime artifact** at `scripts/crc110_observed_fees.json`.
That broker-evidence file is intentionally absent from the public repository and
ignored by Git. Do not reconstruct it from examples, commit broker captures,
copy credentials, change the expected hashes, or treat missing evidence as zero
cost. The existing private installation is managed separately. When unavailable
or invalidated, the read fails closed for scoped fees/net while physical history
and its explicit accounting hold remain. Source publication does not install,
register, reconcile, qualify, or alter any financial record or trading control.

Public UI fixtures use fictional UUIDs and row/report fingerprints. Three
literal source/rule pins exercise transport-contract validation only; those
fixtures contain no broker response or private proof bytes and cannot establish
financial eligibility. Public backend tests use supplied fictional records and
cursors, including the absent-private-artifact path; they do not execute a bot,
make network calls, query a database, install a registry, or prove native SQL.
Private accounting-positive verification is not reproducible from this public
tree and is not claimed by these tests.

The compiled frontend is built from the reviewed application sources. Local
preview interceptors, synthetic banners, evidence packets and owner-run server
scripts are not production assets. This source publication is separate from a
production deployment or reload.

## Operational display contracts

Dashboard labels keep evidence scopes separate:

- The 24-hour buy-fill execution count and notional come from the same bounded
  broker FILL-activity observation. A last-run submission count is never a fill
  count, and one partially filled order can have several fill executions.
- Order rows preserve raw broker status and replacement links. Cancellation,
  rejection, a pending replacement and a session-ended order do not prove a
  fill or a triggered stop. A successor order ID alone does not confirm that
  a pending replacement completed or that protection is effective.
- Monitor rows identify their source and timestamp meaning. Research replay
  output is not a live position event. Receipt observation time is not an
  invented broker-event time, and event logs do not override current positions.
- Process completion, artifact presence, freshness, research qualification and
  permission to submit orders are distinct. Historical receipt success is not
  proof that a newer scheduled invocation succeeded.
- Schedule labels require current source/configuration evidence. Unknown
  windows stay unknown; a historical in-window flag does not describe now.
- Resource snapshots retain their capture time, source and used/quota values.
  Missing data stays unknown, and an over-quota percentage is not hidden by
  clipping the text to a full visual progress bar.

These are presentation and read-only evidence contracts. They neither add
public trade controls nor authorize changes to broker orders, trading risk,
strategy qualification, source guards or immutable historical receipts.

### Trailing-24-hour observed buy fills

`GET /api/execute/orders-summary` reads the configured paper broker's
`GET /v2/account/activities/FILL` endpoint after the existing paper-context
validation. It never queries orders to infer this window: order `after`/`until`
filters refer to submission time and cannot capture an older GTC order filled
today or all still-open partial fills.

The response uses `fill_count` with `count_unit=buy_fill_activity_events`.
Each unique activity ID with side `buy` is one execution, including both `fill`
and `partial_fill` activity types. `total_value` is the sum of each execution's
`qty * price`, with `value_unit=USD_buy_fill_notional`. Cumulative quantities,
order quantities and order status do not determine either value. The deprecated
`orders_filled` field remains null rather than silently changing units.

The exact displayed window is `since_utc <= transaction_time < until_utc`, spans
24 hours, and is fixed before reading the first page. It is declared by
`window_basis=transaction_time` and
`window_boundary=since_inclusive_until_exclusive`. Provider query bounds are
padded outward by one second because filter endpoint inclusivity is unspecified;
well-formed observations outside the exact local window are excluded. Missing,
naive or invalid execution times, invalid sides, missing IDs, and nonpositive or
nonfinite quantities/prices prevent complete totals.

Pagination retains the same time bounds and ascending direction, sends the
last returned activity ID as the exclusive `page_token` after **every nonempty
page**, and stops successfully only after an explicit empty page. A short page
does not establish exhaustion. Identical IDs are counted once; conflicting
facts for one ID, a repeated/nonadvancing cursor, malformed pages or failed
reads fail closed. Reads are capped at 100 activities per page, 10 requests,
three seconds per request and an eight-second pagination budget. The existing
bounded reader also imposes an 8.5-second response deadline.

Only an exhausted, valid observation returns `ok=true`, status `ready` or
`empty`, and `coverage=requested_24h_fill_activity_window`. Otherwise coverage
is `incomplete`, both `fill_count` and `total_value` are null, and any
`observed_fill_count` is only the validated lower bound seen before failure.
Frontend headline values require the complete explicit contract; they never
fall back to legacy counts or an incomplete sample. A successful empty read is
distinct from a failure and may truthfully report zero.

Coverage is explicitly `observed_broker_fill_activities`, with
`accounting_final=false`. This describes broker activities observed during the
request, not final settlement or a guarantee against late-reported fills;
later refreshes can change a historical window. It does not qualify strategy
performance or authorize any trading action.

API references: [account activity fields and pagination](https://docs.alpaca.markets/us/docs/account-activities),
[FILL activity endpoint](https://docs.alpaca.markets/us/reference/getaccountactivitiesbyactivitytype-1),
and [order query filters](https://docs.alpaca.markets/us/reference/getallorders-1).
