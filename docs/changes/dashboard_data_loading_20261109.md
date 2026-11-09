# Dashboard display latency review, November 9, 2026

## Observed behavior

The public release009 app returned valid data while some browser panels showed
unavailable or timed out. Sequential public observations on November 9 included:

- Account overview: HTTP 200 with broker data in 11.85 seconds, exceeding the
  Dashboard's 10-second client deadline.
- Execute state: HTTP 200 with orders, trailing-stop observations and logs in
  26.805 seconds, exceeding its 15-second fallback deadline.
- Strategy scorecard: HTTP 200 in 20.54 seconds. Its separate 35-second client
  budget accommodates this observed cold refresh; this is not proof of failure.
- Execution-gross stats: HTTP 200 after 29.692 seconds, but the response itself
  was unavailable with `CURRENT_EXECUTION_GROSS_INPUTS_UNAVAILABLE`.
- Gross leaderboard, physical history and position logs each exceeded the
  independent review client's 35-second timeout. That does not prove the server
  stopped processing those requests.

The underlying pipeline receipt completed at 03:47:33 UTC and its PostgreSQL
selection still contained two setups. The diagnostic score's two-hour age gate
and the process receipt's 24-hour age gate are different contracts. A diagnostic
`Unknown` label does not establish that the pipeline failed or its rows vanished.

The frontend also opened an unbounded Execute state SSE response while polling
the same aggregate every 20 seconds. The server builds its summary, orders,
trailing-stop observations and three log panels sequentially. This creates
avoidable load and can occupy WSGI capacity. Actual worker counts, queue delay
and the cause of every slow request were not measured, so starvation remains a
supported risk rather than a proved sole cause. The audit-stream component has
no current source caller and is not counted as a second active browser stream.

## Narrow backend change

The ordinary trade stats, leaderboard and latest endpoints previously opened a
second database connection and inserted into `trades_api_requests` before
returning their display response. That telemetry INSERT had no local statement
timeout. The response's `recorded_to_db: true` confirms this occurred in the
observed stats GET; the public review must not be described as zero database
writes. This was existing application behavior, not a ledger mutation.

`_record_trades_api_request` now retains its existing callable contract but
returns false without connecting or inserting. Existing telemetry rows remain
untouched. The endpoints return the same financial payload, apart from truthful
`recorded_to_db: false`. Current frontend code does not depend on a true value.
This removes unnecessary display-path work; no measured speedup is claimed
before controlled deployment verification.

Non-streaming API responses also receive a `Server-Timing` field with exactly
the metric `web_handler` and a finite, nonnegative millisecond duration. Flask's
request-local state prevents cross-request reuse. The metric measures execution
between Flask's before-request and after-request hooks. It excludes upstream
queueing, network transfer and deferred streaming work. Streaming and non-API
responses omit it. It contains no request parameters, SQL, exception text,
credentials, account identifiers or source paths.

## Retire the legacy aggregate stream

Only `/api/execute/state/stream` now returns an empty HTTP 204 response with
no-store headers, without constructing its payload or starting a generator.
EventSource treats HTTP 204 as a permanent stop instead of reconnecting. Merely
limiting stream duration or iterations would reconnect cached clients and could
increase load while their separate fallback polling continued.

The finite `/api/execute/state` route and all other routes remain unchanged.
The release009 client already responds to EventSource errors by starting its
finite fallback, and independently polls every 20 seconds. Its old 15-second
deadline remains a limitation for cached clients; updating the frontend is still
needed to obtain the new loading behavior. This endpoint retirement is an
explicit compatibility change for any other consumer of the aggregate stream.
It prevents that stream from retaining a worker but does not prove it caused
every observed timeout or guarantee a particular queue delay.

## Preserved limits and unresolved evidence

No native source or financial-reader change is part of this patch. The source
contracts, policy reader, execution-gross projection and transport remain
unchanged. In particular, the eight-second consistency bound, PostgreSQL
read-only checks, policy exclusions and canonical rechecks are not relaxed.

Ordinary net-performance stats correctly returned held values with validated
policy and incomplete accounting inputs. The four unresolved trade IDs were
110, 125, 131 and 188. This is separate from the execution-gross failure.
Execution-gross catches failures and returns a fixed generic reason without
logging the underlying exception. Current evidence therefore cannot distinguish
a consistency/input failure from a query or resource limit. No new diagnostic
was executed on the host. A future investigation needs narrowly scoped,
sanitized evidence; changing financial gates is not a display repair.

## Verification and release boundary

Offline tests extract only the changed handlers and use synthetic payloads and
a local Flask test client. They prove zero telemetry database calls, unchanged
ready/held financial fields, truthful acknowledgement, request-scoped timing,
and omission for streaming, non-API, missing-start and invalid-clock cases.
Separate route tests prove the retired endpoint returns an empty no-store 204
without invoking its payload factory, stream helper, database or provider, and
the finite route still returns its exact financial payload from one factory call.

This commit is preparation only. It does not upload source, reload the web app,
change WSGI, run jobs, change schedules or alter trading. A release009-style
package review must explicitly admit this backend delta; the old assertion
that only its frontend directory literal changes is no longer sufficient.
