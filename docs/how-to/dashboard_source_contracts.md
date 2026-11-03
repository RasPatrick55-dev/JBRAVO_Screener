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
