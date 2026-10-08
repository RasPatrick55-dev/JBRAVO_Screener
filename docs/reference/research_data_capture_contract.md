# Selected-symbol research capture contract (proposed; no retrieval authorized)

Source baseline: `d391216b5f41be4312e37149b9cc1401f991bb01`.
Implementation: `scripts/research_data_capture.py`. Production fetching, database
storage, ranking, execution and schedules are unchanged. Import is inert.

## Universe and provenance

Sorted manifest: **ALK, FN, HBAN, HMY, LOGI, MIRM, NBIS, NVST, PLUG, RGLD, SKY,
SPY, SRRK, STLD, VSCO**. Fourteen symbols come from canonical
`data/history_cache/*.csv` filenames inspected on 2026-10-08; SPY is added only
as the proposed benchmark. Their current file hashes are embedded in
`CACHE_SHA256` and emitted in `symbols.json`. Those hashes bind this selection;
they do not establish cache provenance or point-in-time eligibility.

This is a **selected-symbol research universe**, not historical eligible
membership. Current identifiers and the explicit provider mapping date cannot
recover delisted securities, eliminate survivorship bias, reconstruct original
requests or establish an untouched evaluation period. Existing retained model
evaluation records also prevent assuming prior periods were unseen.

## Proposed settings for Board review

| Field | Proposal |
|---|---|
| Daily session dates | 2023-01-01 through 2026-10-07 inclusive |
| UTC request boundaries | `2023-01-01T00:00:00Z` to `2026-10-08T00:00:00Z` |
| Client boundary rule | Start inclusive, end exclusive; unexpected boundary bars fail validation |
| Timeframe / ordering | `1Day`, `asc` |
| Feed | `sip`; entitlement is unverified; failure stops, never falls back to IEX |
| Adjustment | `raw`; no silent corporate-action adjustment. This does not make raw bars suitable for returns across splits/dividends |
| Currency | Explicit `USD` |
| Mapping | Explicit `asof=2026-10-08`, literal symbols, no alias substitution |
| Calendar | Alpaca paper endpoint `/v2/calendar`, same inclusive date range; exchange times interpreted in America/New_York |
| Output | `%USERPROFILE%/jbravo-research-data/<new-capture-id>/` |

Every setting is required on the CLI; environment feed/date/default settings are
not inherited. These values are proposals, not adopted evaluation settings.
The local alpaca-py request source documents feed, adjustment, currency and asof;
this collector instead uses direct GET requests so SDK pagination and parameter
removal cannot occur. Live endpoint behavior and entitlements remain untested.

## Exact proposed future retrieval budget

Maximum **41 GET requests total**: one calendar request, then at most 40 bar
pages. Each bar page requests at most 10,000 rows across the 15 symbols. There
are **zero retries, redirects, feed fallbacks or alternate endpoints**.
Maximum successful/in-progress decoded response-body read: **4 MiB per request** and
**32 MiB aggregate**, with **120 seconds of local capture eligibility** and
**15 seconds maximum local wait for any request**, shortened by remaining time.
Limits can only be reduced programmatically; the CLI cannot increase them.

The adapter reads at most its supplied byte capacity and rejects an exactly-full
stream when EOF cannot be proved within that capacity. A daemon worker isolates
blocking HTTP work; a timeout ends local acceptance and prevents another request.
It does not prove physical termination of a remote request. Incomplete transport
byte totals remain UNKNOWN, separately from completed decoded response-byte totals.
These limits and body hashes do not measure compressed wire traffic or HTTP headers.
Failure-record finalization may occur after the eligibility deadline; no late
response becomes an accepted dataset. No future capture is authorized here.

## Receipts and validation

Each attempted request records its fixed endpoint, sanitized effective parameters,
UTC submission/receipt times and disposition. Pagination tokens are represented
by SHA-256 and byte length in saved evidence. Complete responses have raw-body
size/hash bindings; raw responses are not archived. Validated page files retain
the canonical OHLCV projection, optional trade count/VWAP, and next-token hash.
Original numeric decimals are parsed without binary float rounding; equivalent
numeric trailing-zero representations canonicalize identically. Authentication
headers and raw error messages/bodies are never serialized.

`request-contract.json` binds settings, limits, collector hash, Python/requests
client identity and the fact no Alpaca SDK runs. The prepared HTTP URL parameters
must match the complete intended parameters. Explicit terminal pagination state
is required; an omitted next-page field is not assumed to mean completion.
Repeated/malformed tokens, malformed
JSON, duplicate JSON keys, unknown symbols and oversized pages fail closed.

Bars require aware timestamps, exchange-local midnight daily sessions, requested
UTC coverage, finite positive OHLC prices, consistent high/low, nonnegative
integral volume, and ordered timestamps per symbol. Fractional timestamp precision
beyond microseconds is rejected, not rounded. Identical adjacent duplicates are
counted and deduplicated; conflicts or backward order fail. Dataset output is
sorted by symbol/session; source ordering is checked before that derived sort.

The calendar must have unique ordered dates and valid open/close times. Weekends
and calendar-excluded weekdays (holidays or closures, not independently named)
are distinguished from absent bars on returned sessions. Missing symbol eligibility
remains UNKNOWN; no IPO/delisting inference is made. Calendar completeness is a
provider claim, not independently qualified. SPY must cover every returned session;
missing SPY makes the capture FAILED. Other missing bars yield
`CAPTURE_COMPLETE_WITH_COVERAGE_GAPS`, never full data qualification.

Output paths must be new, outside the source tree, and without known symlink or
Windows reparse redirection. Parent-traversal (`..`) components are rejected
before absolute-path conversion. Both the base and final capture destination
are checked against the source tree before any directory creation or request;
the final destination and all ancestors are inspected for redirection without
resolving away symlinks. No files are overwritten. Saved bytes are reopened
and bound by size/hash. `capture-manifest.json` is written last, excludes itself,
and its final measured identity is returned. Partial receipts remain under a
FAILED manifest and must not be used as a completed dataset.

## Offline verification and untested paths

`tests/test_research_data_capture.py` imports the real collector and exercises
synthetic multipage responses, token cycles, duplicate conflicts, missing SPY,
coverage gaps, limits, partial failure and actual CLI parameter delivery. The
requests adapter uses a mocked Session response with real prepared URL encoding.
Sockets and child processes are forbidden; all output uses pytest temporary
directories. No provider, production DB, historical evaluation or order runs.
Live TLS/authentication, SIP entitlement, provider calendar completeness, mapping
semantics, actual response compatibility and OS/network cancellation remain
unqualified. Dataset quality and profitability are not established by these tests.

## Development verification history

All four newly authorized invocations used the existing Miniconda interpreter,
`-B -m pytest --noconftest -p no:cacheprovider tests/test_research_data_capture.py -q`
(attempts 2-4 also used `--tb=short`). Plugin autoload and bytecode writes were
disabled; credentials/provider/DB environment variables were excluded from the
isolated process environment. Each process had an enforced 180-second timeout.

| Attempt | Actual result | Process duration |
|---|---|---|
| 1 | 25 passed, 1 failed: socket guard replaced a class with a function, breaking SSL import in the adapter test | 1.782893 s |
| 2 | 30 passed after correcting the guard and adding timeout/calendar checks | 1.8425418 s |
| 3 | 31 passed, 1 skipped after receipt bindings and numeric canonicalization checks | 1.8912299 s |
| 4 | 33 passed, 1 skipped after explicit pagination-state and inert reparse-attribute checks | 2.1186686 s |

The remaining skip is actual symlink creation denied by Windows. The Windows
reparse-bit rejection branch is separately verified using inert metadata; this
does not establish resistance to concurrent filesystem redirection. No timeout
occurred. **4/4 consumed; no additional test allowance remains.** Earlier packet
counters are unchanged. Final code is uncommitted and unaccepted.

### Containment revision verification

`JBRAVO_RESEARCH_CAPTURE_CONTAINMENT_REVISION_001` grants two separate test
invocations; it does not replenish the original exhausted 4/4 allowance.
Revision invocation 1 used the same isolated command and 180-second limit:
**37 passed, 1 skipped**, exit 0, process duration 2.1748586 seconds.
The existing Windows symlink-creation skip remains; inert reparse metadata now
covers both the base and final destination. New regressions cover traversal
resolving into the source tree, an ordinary external capture path, and a final
destination inside the source tree. Rejections assert zero dispatched requests
and no new output. **Revision allowance: 1/2 consumed.** The original exported
patch remains unchanged; the revised patch is a separate review artifact.
