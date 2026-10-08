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
| Mapping | Explicit `asof=2026-10-08` enables provider symbol mapping; literal request symbols, no local alias fallback; not historical eligibility proof |
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

### Bounded bar-validation diagnostics

Bar-level field, duplicate and ordering failures additionally attempt to save
`bar-validation-diagnostic.json` (schema `jbravo.bar-validation-diagnostic.v1`).
It records the original `primary_error_code`, `usable_dataset: false`, exact
field and validation rule, symbol, zero-based index within that symbol's page,
and supplied timestamp. Request ordinals are one-based including the calendar;
bar-page ordinals are one-based excluding the calendar. The response-body byte
length and SHA-256 bind the failing page without archiving its response body.
Only the first rejected bar is reported; preceding validated pages remain
partial evidence, and the failing page is not saved as a validated page.

An offending value includes its parsed Python type (JSON decimals are `Decimal`),
a scalar rendering capped at 128 characters, and explicit truncation information.
Only numeric-looking strings and timestamp-shaped strings may be rendered.
Arbitrary text and containers are omitted with a marker, not serialized or
coerced into accepted bars. Numeric/timestamp representations are diagnostic
data, not authentication data; headers, credentials, raw exceptions and complete
responses are never included. The entire diagnostic JSON is capped at 4,096 bytes.
Rules, accepted values, effective requests, pagination and capture budgets remain
unchanged. Missing required fields identify the missing field rather than inventing
a supplied value.

`failure.json` is saved first with the original code and unusable status. A
secondary diagnostic-write/readback failure cannot replace that primary outcome:
the manifest records `diagnostic_recording.status: FAILED` and only the secondary
exception type. A readable partial diagnostic of at most 4,096 bytes is bound as
`PARTIAL_UNVERIFIED`, never accepted as valid diagnostic JSON; otherwise the
possible unbound partial is explicit. No write is retried. Primary-failure,
receipt or manifest storage failures can still prevent a complete saved outcome.
Existing filesystem-race and remote-cancellation limitations remain.

### Provider capture versus bar quality

`JBRAVO_RESEARCH_BAR_QUALITY_CONTRACT_001`, based on
`b9ceaca57b310bac7b0be78cddcb794a58bfbbde`, changes only the treatment of numeric
zero VWAP and explicit quality reporting. Supplied negative, nonfinite, malformed,
Boolean, null or text VWAP remains rejected. Mandatory OHLC prices remain finite
and strictly positive with the existing range checks; volume remains finite,
nonnegative and integral. Trade-count and timestamp checks are unchanged.

A supplied numeric zero VWAP is retained exactly as zero in the existing
exact-decimal string projection (`vw: "0"`; negative zero preserves its sign).
Numeric formatting already canonicalizes trailing zeroes; the dataset is not a
byte-for-byte archive of provider JSON. Zero is never converted to an OHLC price,
invented VWAP, positive usable price or absent field. Each affected page/dataset
record carries `ZERO_VWAP_UNQUALIFIED` in `quality_flags`. Zero volume is preserved
as integer zero and independently flagged `ZERO_VOLUME_UNQUALIFIED`. A bar can
carry both flags. Neither flag explains its cause or establishes liquidity,
trading eligibility, corporate-action treatment or suitability for evaluation.
An absent optional VWAP remains absent; it is not filled in or counted as zero.

`quality-summary.json` (schema `jbravo.bar-quality.v1`) is saved even for clean or
failed captures and hash-bound in both the manifest's `files` and `quality_summary`.
It records unique validated-bar and flagged-bar counts, exact per-code/per-field
counts, and up to 100 deterministic per-field finding samples in encounter order.
Each sample has bounded supplied-value/timestamp renderings, field/rule, symbol,
session, request/page/bar ordinals and the completed response body's size/hash.
`omitted_findings` and `sample_truncated` explicitly describe the sample limit;
aggregate counts include all observed unique bars, and all anomalies in captured
bars remain flagged in page/dataset records. Identical duplicates are counted
once for quality, just as they are deduplicated from the dataset. On failure,
quality coverage means only unique bars validated before the stop, potentially
including bars from an unsaved failing page; it does not describe a full dataset.
The existing sanitized diagnostic renderer bounds each value to 128 characters;
fixed fields and bounded identifiers keep each sample below 4,096 bytes.

A successfully completed capture with either flag reports
`CAPTURE_COMPLETE_UNQUALIFIED_DATA`, taking precedence over the coverage-gap
completion label. `coverage.json` still records those gaps, and missing SPY or
mandatory validation/storage/deadline errors still yield `FAILED`. No rejected
mandatory field is skipped or repaired. `quality_disposition: UNQUALIFIED_DATA`
is separate from successful transport/pagination. An anomaly-free capture retains
its existing completion label but reports only `NO_LISTED_ANOMALIES_OBSERVED`.
Every manifest and quality summary has `qualified_for_research: false`.
CLI exit 0 means capture completion, including unqualified data, never research
readiness or evaluation permission. Consumers must inspect the disposition and
quality artifacts rather than use process success as acceptance. Quality-summary
storage failures, like other mandatory evidence-storage failures, can prevent a
final manifest; they are not silently ignored or retried.

Local consumer inspection: `scripts/backtest.py:load_bars_from_db` selects date,
open, high, low, close and volume; `_normalize_api_bars` projects the same OHLCV
fields. `prepare_series`, `compute_atr`, `composite_score`, `PortfolioBacktester`
and `scripts/indicators.py:compute_indicators` use OHLCV/derived indicators, not
VWAP. Repository Python searches found no VWAP consumer outside this collector
and its tests. Thus VWAP is not required by the intended current backtest, while
volume is used in scoring/OBV. There is no existing research `dataset.json` to
backtest loader: this inspection is a static dependency finding, not integration
qualification or authority to evaluate flagged data. Production fetching,
storage, ranking, paper execution and request settings/limits remain unchanged.

Neither failed real capture is read, altered or reclassified by this offline
packet. As-of provider mapping cannot prove historical universe membership;
raw corporate actions, provenance, unseen performance and profitability remain
unqualified. Any future retrieval or evaluation requires separate Board authority.

This additive source change does not diagnose the previously failed real capture's
specific offending field: its retained `INVALID_OHLCV` code alone cannot do so.
That capture remains FAILED with its one execution consumed and two provider
requests recorded; unused request capacity does not authorize another execution.

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

### Validation-diagnostics revision verification

`JBRAVO_CAPTURE_VALIDATION_DIAGNOSTICS_001` is additive to integrated canonical
commit `75822b595519e93bd3b9917d314cb32162270399` and grants two new offline test
invocations without replenishing previous allowances. Invocation 1 used:

```text
C:\Users\RasPa\miniconda3\python.exe -B -m pytest --noconftest -p no:cacheprovider tests/test_research_data_capture.py -q --tb=short
```

Result: **55 passed, 1 skipped**, exit 0, process duration **3.5231076 seconds**,
with no timeout under the 180-second ceiling. Plugin autoload and bytecode writes
were disabled, and the process environment excluded provider/DB credentials.
**New allowance: 1/2 consumed; no failed invocation.** The skip remains Windows
denying actual symlink creation, with the existing inert reparse checks passing.

The tests use normal collector import, synthetic provider responses and temporary
storage. New cases distinguish price, volume, trade-count, VWAP, range, timestamp,
missing-field, duplicate and ordering failures; verify second-page/within-symbol
location and response-body bindings; bound or omit offending values; and force
diagnostic-write failures both before writing and after a partial write. They
verify that primary failure codes, unusable status and saved artifact/manifest
bindings survive those secondary failures. Ordinary successful validation emits
no failure diagnostic. Sockets and child processes are forbidden in the fixtures.

No real provider, credential, remote filesystem or production database path was
exercised. Live compatibility, remote cancellation, concurrent filesystem races
and failures storing the primary failure/receipts/manifest remain unqualified.
The original failed capture, patches and verification history remain unchanged;
this revision is uncommitted and awaits Board review. No retrieval is authorized.

### Bar-quality contract verification

This packet grants two new offline invocations, independently of all prior
consumed allowances. Both used the existing Python 3.13.2 Miniconda interpreter:

```text
C:\Users\RasPa\miniconda3\python.exe -B -m pytest --noconftest -p no:cacheprovider tests/test_research_data_capture.py -q --tb=short
```

| Invocation | Actual outcome | Measured process duration |
|---|---|---|
| 1 | Exit 1 before collection: `No module named pytest`; no tests executed | 0.0448665 s |
| 2 | Exit 0: **75 passed, 1 skipped** | 3.6841632 s |

The isolated first environment did not expose the already installed user-site
pytest. Invocation 2 explicitly bound the existing
`C:\Users\RasPa\AppData\Roaming\Python\Python313\site-packages` via `PYTHONPATH`
and retained `APPDATA`; no installation or package change occurred. Both process
environments excluded provider/database authentication variables, disabled plugin
autoload and bytecode writes, and enforced a 180-second timeout. Neither timed
out. **2/2 consumed**, with the failed invocation retained separately. The skip
is Windows denying test symlink creation; inert reparse-attribute checks passed.

Tests use normal collector import, synthetic provider responses, existing socket
and child-process prohibitions, and temporary output only. Added regressions
cover supported zero representations, positive/absent VWAP, negative/nonfinite/
malformed VWAP rejection, independent zero-volume flags, exact quality counts
with duplicate removal, sample truncation, coverage-gap precedence, mandatory
failure precedence, response-body and manifest bindings, final tamper detection,
and CLI completion without research qualification. Existing mandatory-field,
pagination, limits, containment and diagnostic failure tests remain in the suite.

No backtest or provider integration was executed. Actual concurrent filesystem
redirection, remote cancellation, live provider compatibility and evidence-storage
failures remain unqualified. The source increment is uncommitted and awaits Board
review; no capture, evaluation, publication or operational permission follows.
