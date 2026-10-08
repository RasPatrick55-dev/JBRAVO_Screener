# Research security status ledger

Board packet: `JBRAVO_RESEARCH_SECURITY_STATUS_LEDGER_001`.
Source baseline: `43d78ce84e5482160947e895b574e8805ffbadbf`.
Module: `scripts/research_security_status.py`; standard library only, inert normal
import, no CLI, file reads/writes, credentials, provider, database or backtester.

## Classification, not simulator enforcement

`classify_records(records, policy=DEFAULT_POLICY)` accepts in-memory collector-shaped
records (`symbol`, ISO `session`, mandatory OHLCV, optional VWAP `vw` and trade-count
`n`). It returns
detached original-record copies, including supplied nested provenance, plus derived
status/reasons. It does not add evidence to, repair, or rewrite input records.

Internal `research:` keys are policy comparison keys, **not verified CUSIP/ISIN or
proof of economically unchanged share entitlements**. `verified_external_identifiers`
is always empty. Unknown securities remain unresolved; matching tickers alone never
join them. Even positive, internally consistent bars remain `research_qualified: false`.
`executable` and `available_for_warmup` are false when blocked and null when unresolved;
null must not be interpreted as permission. This API has no positive readiness state.

No indicator reset, signal expiry, pending-action handling, position change or fill
is implemented. Returned policy declarations describe the adopted future enforcement
contract only: halted bars cannot supply fills, liquidity or warm-up; prolonged halts
require fresh qualified warm-up; pending entries are invalidated at halts/unresolved
interruptions; position/exit obligations remain unresolved without invented fills;
the first qualified executable bar uses existing causal gap/exit precedence. Ordinary
calendar closures are distinct from missing data; carried valuations remain uncertain.
Simulator integration requires separate authority and verification.

## Evidence and interval boundaries

All session intervals are start-inclusive/end-exclusive. Calendar authority and actual
security tradability remain unqualified. Session dates cannot establish intraday order.

| Policy | Representation and limitation |
|---|---|
| NBIS/YNDX continuity | Both provider labels map to `research:nebius-listed-line`. Historical ticker is null because the precise rename-effective date is not established by the resumption notice. Issuer continuity does not qualify restructuring, capitalization or share entitlements. |
| Documented halt | `[2022-02-28, 2024-10-21)`, `HALTED`; all such records are nonexecutable and unavailable for warm-up irrespective of supplied positive OHLC. |
| Resumption | From 2024-10-21, `RESUMPTION_DOCUMENTED`, still unresolved eligibility with fresh-warm-up/qualification reason. The notice specifies 09:00 ET, not a daily-bar fill. |
| VSCO/VSXY | Historical ticker VSCO before 2026-06-02, VSXY from that date. Both provider labels can appear in either interval because provider mapping may relabel history. This does not assert that a particular request actually mapped correctly. |

The default event metadata preserves the documented halt instant
`2022-02-28T06:38:00-05:00` and resumption instant `2024-10-21T09:00:00-04:00`.
These are public-event annotations; custom policy intervals do not rewrite those events.

Sources, carried as references rather than fetched or independently verified by code:

- [Nasdaq's October 18, 2024 notice](https://www.globenewswire.com/news-release/2024/10/18/2965726/6948/en/nasdaq-resumes-trading-in-nebius-group-n-v.html).
- [Victoria's Secret's May 21, 2026 announcement](https://www.sec.gov/Archives/edgar/data/1856437/000185643726000007/ex991vscomay212026pressrel.htm): ticker effective June 2, unchanged CUSIP.
- [Confirming 8-K, Item 8.01](https://www.sec.gov/Archives/edgar/data/1856437/000185643726000007/vsco-20260521.htm).
- [Alpaca historical bars/asof](https://docs.alpaca.markets/us/reference/stockbars): entity/name mapping, not historical eligibility. `raw` means no adjustments.

`validate_policy` rejects empty/reversed intervals, unsupported status, missing evidence,
external-looking keys, duplicate aliases, overlapping mappings for an alias or security
key, unbound status keys and overlapping status intervals. Adjacent boundaries are valid.
It checks internal policy consistency, not authenticity of supplied evidence references.

## Explicit overlap window

`compare_records(left, right, sessions=[...], policy=DEFAULT_POLICY)` requires a nonempty,
unique sorted caller-supplied session list. Only records within that window participate.
It compares policy key/session identities and `o,h,l,c,v,n,vw` using finite exact Decimal
values, permitting equivalent numeric representations without rounding/tolerance. Optional
`n` or `vw` absent on both sides matches; presence on only one side conflicts for that
field. An absent VWAP stays absent: it generates neither an invalid-field nor a zero-VWAP
reason, and no value is imputed. Supplied null/malformed/nonfinite VWAP remains invalid;
supplied numeric zero remains preserved and flagged `ZERO_VWAP_UNQUALIFIED`.
Mandatory missing/invalid
fields, nonpositive OHLC, inconsistent OHLC, negative VWAP or nonintegral/negative counts
are flagged. Zero VWAP/volume remain supplied values with explicit unqualified reasons.

Outcomes: `MATCH`, `CONFLICT` with differing fields, `CONFLICT_INVALID_RECORD`,
`CONFLICT_DUPLICATE_IDENTITY`, `MISSING_LEFT`, `MISSING_RIGHT`; separate `MISSING_BOTH`
records and unresolved identities. Duplicate aliases cannot silently choose a record.
Missing-both reporting covers only mapped securities observed somewhere in the window:
the module cannot invent an entirely unseen security/universe. Provider timestamps and
all provenance remain copied, not equated or externally verified; comparison is session
OHLCV/count/VWAP comparison, not complete byte equality. Matches, including identical
halt-period zero values, do not promote qualification or create a stitched dataset.

## Verification and remaining gates

Synthetic tests use normal module import with sockets/process spawning forbidden.
Run only the scoped offline tests; no captures, production inputs or strategy evaluation.
The test command disables repository conftest/plugin loading because global conftest
imports the unrelated Alpaca SDK. This does not qualify production import integration.

Real capture 003 has not been accessed by this source packet. Whole-dataset corporate
actions, provider mapping effectiveness, calendar completeness, historical universe,
external security identifiers, raw-price comparability, warm-up adequacy and research
readiness remain open. A supplemental VSXY capture and overlap audit require separate
retrieval authority. Publication, simulator enforcement, evaluation and trading remain
separately gated. No earlier artifacts or consumed allowances are replenished.
