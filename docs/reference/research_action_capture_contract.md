# Corporate-action research collection contract

This isolated standard-library REST collector collects observations. It does not
create an action ledger, interpret ratios, adjust prices/share units, credit
dividends, or authorize research/replay. Every returned outcome and manifest has
`qualified_for_research: false`. Production fetching, storage, schedules,
ranking, paper execution and the simulator are unchanged. Imports perform no I/O.

## Retained request contract and unresolved schema

Acquisition specification: `JBRAVO_ACTION_DATA_SOURCE_CONTRACT_001`, retained
`acquisition-specification.md`, 16,694 bytes, SHA-256
`6c1bba3f657e01d47a5135dd72842bd1bd68d9c844c3ee339c9c9ae18326fb74`.
The retained official reference is
<https://docs.alpaca.markets/us/reference/corporateactions-1>.
No new public or authenticated request was made for this source packet.

Only `GET https://data.alpaca.markets/v1/corporate-actions` is enabled. Each chain
freezes inclusive `start`/`end`, `region=all`, `data_quality=all`, `sort=asc`,
`limit=1000` and one comma-separated selector. Only `page_token` changes. Dates
filter **process_date**, not effective/ex/record/payable dates. `types` is omitted
to request all types; `ids`, `asof`, bars settings and arbitrary extra parameters
are not exposed. There is no alternate endpoint, feed, entitlement probe or
parameter-removal fallback.

The default symbol labels are sorted: ALK, FN, HBAN, HMY, LOGI, MIRM, NBIS, NVST,
PLUG, RGLD, SKY, SPY, SRRK, STLD, VSCO, VSXY, YNDX. These are selected research
labels plus continuity aliases, not historical universe membership or verified
security identifiers. Explicit alternate sorted labels can be frozen through
the CLI. The caller must supply the collection end date; it is never silently
replaced by today. The specification's 2022 padding is not proof that all events
affecting the captured period are included.

An optional second `cusips` chain requires a separately SHA-256-bound JSON list
of identifier records. Each record contains `cusip`, internal `security_key`,
`source_reference`, `source_sha256`, `valid_from`, `valid_through`, and Boolean
`independently_bound: true`. These declarations and the manifest pin must be
supplied from independent evidence by the authorizing owner. The collector checks
format/binding (and records the CLI's verified manifest SHA-256); it does not independently adjudicate the evidence's truth,
security identity, interval completeness or coverage. Programmatic callers supply
the same independently sourced records directly; this assertion is explicitly
labelled owner-supplied rather than independently adjudicated. No IDs are inferred from
tickers, and applicable intervals are retained rather than substituted for the
process-date filter. Both chains use the same dates and common parameters.

**Provider response schema is unresolved.** The parser is explicitly provisional:
it recognizes an object with a `corporate_actions` object whose values are event
arrays and an explicit `next_page_token` that is null or a nonempty string. It
requires nonempty string event IDs for duplicate accounting. These structural
assumptions are not promoted to an independently established endpoint contract.
Absent/unsupported envelope, token or ID shapes produce explicit unresolved
schema failures after raw-body preservation. Unknown category/field names remain
raw and unqualified. Retained documentation does not establish all event fields,
currencies, ratio direction, revisions/deletions, US-listed entitlements (including
HMY ADRs/LOGI), historical coverage, latency or a pagination snapshot guarantee.
Live response compatibility remains untested; further contract evidence is needed.

## Limits and local timing

Hard ceilings are twelve GET attempts overall, six pages per chain, 4 MiB per
response, 32 MiB aggregate completed bodies, 600 seconds local acceptance and
15 seconds local wait per request. Programmatic lower ceilings are permitted;
higher/invalid ceilings are rejected before output creation. A terminal page on
page six is allowed; a continuation token stops without requesting page seven.
Identical duplicates are observations, not overwritten events. Repeated tokens
stop. Requests count before dispatch, including failure/uncertain attempts.

The standard-library transport disables redirects and implements no retries.
Authentication uses only process environment `APCA_API_KEY_ID` and
`APCA_API_SECRET_KEY`; it does not read dotenv files or credential stores. Its
daemon worker performs one GET and cannot write evidence, retry, or submit another
request. A local wait timeout stops collection; a late worker may still run until
network completion. Remote cancellation is not established. No compressed content
decoding is attempted: `Accept-Encoding: identity` is requested; unexpected
encoding stops. Bounds cover retained application bodies, not wire traffic.

Oversized responses retain only a bounded prefix with `transfer_complete: false`;
these are not represented as complete original bodies or included in completed
body-byte totals. Incomplete-transfer totals remain UNKNOWN. Timeout/read failure
may leave no body evidence. A late complete body can be retained, but cannot
produce an eligible collection. The deadline includes saved-output verification;
a final deadline/recording failure returns unsuccessful without rewriting bytes.

## Exact preservation and failure recording

Each complete original body is saved and reopened before parsing. All recognized
event payload slices on a page are saved before ID accounting or summaries.
Strict UTF-8 JSON parsing rejects duplicate keys, malformed syntax and non-JSON
numeric constants. The slice preserves event whitespace, UTF-8 bytes and numeric
lexemes such as `1.2300` and `1e-7`; missing and null are retained distinctly.
Raw payloads, not floating-point summaries, are authoritative. Dates/amounts are
not normalized or interpreted. Inventory records identify page, chain, category,
event position, payload binding, supplied ID, present fields and null fields.

Same ID plus identical payload bytes/category is recorded as a duplicate.
Different bytes/category preserves both variants and stops. This deliberately
conservative conflict rule can flag formatting/numeric-spelling differences;
it does not claim an economic contradiction. No semantic-equivalence rule or
last-write-wins merge is introduced. Later payloads on the conflicting page remain
saved even if ID accounting stops before reaching them.

Requests contain no authentication headers. Arbitrary exception text is not
recorded. If a body echoes an exact authentication value, raw evidence is withheld
and collection fails explicitly; secret safety takes precedence over raw-body
retention. This is not permission to save a redacted body as original bytes.

Output must be an absolute external path with no parent traversal or existing
symlink/reparse ancestor; the final destination is also checked. Existing capture
directories are rejected. Files are created exclusively, reopened and hashed.
These checks do not establish freedom from concurrent filesystem races. No
historical file is repaired or overwritten. Partial artifacts remain on failure.

Outputs are `frozen-request.json`, response/event files, `request-receipts.json`,
`event-inventory.json`, optional `failure.json`, and final `capture-manifest.json`.
The manifest binds artifacts and is itself reopened and bound in the returned
result (no circular self-hash). A recording failure remains separate from an
existing primary failure. Final manifest/write/readback failure returns nonzero
and preserves partial bytes without retry. An on-disk manifest from an incomplete
finalization is not proof of successful final verification; the actual return is
also required.

The frozen request also binds the collector's source bytes, Python implementation
and version, transport identity, retained specification identity, identifier
provenance declarations and (for CLI identifier input) verified manifest hash.
No SDK is used. The identifier manifest is limited to 64 KiB and must contain a
nonempty, duplicate-key-free list; an empty optional query is not silently enabled.

`COLLECTED_UNQUALIFIED`/CLI zero means all configured chains reached observed
terminal pages within local bounds and saved artifacts verified. Empty responses
have the same limited meaning. Neither result establishes historical completeness,
point-in-time availability, entitlement/currency correctness, research suitability
or any operating permission. `FAILED`/CLI one preserves bounded available evidence.

## CLI and offline verification boundary

Example only; execution requires separate authorization:

```text
python scripts/research_action_capture.py --start 2022-01-01 --end YYYY-MM-DD --output-base EXTERNAL_ABSOLUTE_DIRECTORY --capture-id NEW_ID
```

Optional independently bound input uses `--identifier-manifest PATH` together
with `--identifier-manifest-sha256 SHA256`. An accepted input binding is not a
corporate-action qualification. Current capture files and previous failed captures
are not read by this implementation packet.

Tests normally import the whole module and exercise capture and CLI using synthetic
transports and temporary storage. They block sockets, provider opening and external
process creation. They do not qualify actual HTTP/provider behavior, entitlement,
remote cancellation, host credentials, production integration, historical action
completeness or action-ledger correctness. No ratios or dividends are applied.
