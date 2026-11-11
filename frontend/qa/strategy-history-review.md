# Strategy trade history presentation review

Base: `83d4e60f2a36bb8d5dd40505b0ee79e2dd51f860` (PR 799).

## Scope

Source-only frontend change. Added a separate card above the existing Trades
layout; the recorded ledger, gross summary, ranking, allocation derivations and
their requests remain unchanged. No generated bundles, database writes, broker
actions, host reloads or deployments were performed.

The new independent request uses `/api/trades/strategy-history?limit=25` and the
shared FIFO, four-active-read ceiling, 45-second dispatch-to-body deadline and
completion-based 20-second polling. Tab disposal aborts queued/dispatched reads.

Presentation checks the `jbravo.strategy-history.v1` contract, provenance and
paper/accounting qualifiers. Native lifecycle state is not inferred from fills.
Complete, partially exited, no-exit, observed unfilled and incomplete records are
distinct. Unknown fees/net stay unknown; incomplete and partial rows cannot show
completed gross. No sums or accounting mutations occur in TypeScript.

Historical ownership is allowlisted and visibly qualified. Entry-recognition-only
scope remains separate from strategy enrollment or manual paper tests; none
assert current position-management authority. Source receipts, stored data times,
bounded history and legacy overlap exclusions are visible.

## Automated checks

- `npm ci --offline --ignore-scripts --no-audit --no-fund`: passed.
- `npm test -- --reporter=dot`: 764 tests passed in 39 files.
- New coverage: 62 contract/render tests and 7 mounted lifecycle tests.
- `npx tsc --noEmit -p tsconfig.json`: passed.
- `git diff --check`: passed.
- `python -B -m scripts.docs_consistency_check`: passed.
- No lint script exists in the frontend package.
- Production bundle generation is deferred to integration, as instructed.

Coverage includes backend-produced synthetic wire data, source mismatches,
malformed identities/numbers/times, native symbol validation, equal completed
entry/exit quantities, ordered fill timestamps, nonempty distinct entry/exit
order identities, unknown vs genuine zero, partial/unfilled
states, recognition-only provenance, retained stale receipts, first-read timeout,
business-failure row clearing, 27-second response followed by completion-based
polling, four active requests, tab disposal, late responses and unchanged legacy
rows. Static/server tests retain the pre-existing React useLayoutEffect warning.

`fixtures/strategyHistoryWire.synthetic.json` was produced by the backend
projector's entirely fictional fixture: SYNTH, 40 shares at $10 entry and $12
exit, $80 gross, October 1/5 fill times and synthetic signal/order identities.
It is not live account evidence. Runtime source contains no symbol-specific
handling. The new lifecycle tests also use an entirely fictional legacy row.
The Windows working-copy fixture matches the reviewed backend export byte for byte:
1,768 bytes; SHA256
`9dc479b600369e808be7121d6615ece24fa2dbaf8bb77e0a9d66c82bf6db8e4c`.
Git's existing line-ending policy stores the same JSON with LF line endings:
1,715 bytes; SHA256
`d6321423f382ea87b699acb005e3d39bc82ce4f7481b816bc66b77667afd0478`.
After fixture cleanup and consistency guards, all 764 frontend tests and the
TypeScript check were rerun. The presentation layout is unchanged; observations below
come from the original component review, not a second browser run.

## Browser review

Used the supported Edge browser extension against the standalone loopback Vite
page `/qa/strategyHistoryPreview.html`, which makes no API/database/broker reads.

- 1240 x 900: light and dark table screenshots reviewed, page width 1240.
- 390 x 844: dark mobile cards reviewed; client/page width 375 (vertical
  scrollbar), no horizontal page overflow.
- 320 x 740: light mobile cards reviewed; client/page width 305. Expanded long
  strategy/signal evidence had no overflowing article or details element.
- 768 x 900: table width 920 contained in its 710-wide horizontal scroller;
  page/client width 768.
- Verified explicit unfilled, incomplete, recognition-only and unavailable text.
- Temporary viewport override reset after the checks.

The browser review uses synthetic component scenarios. It does not establish
deployed behavior or real broker/database state. Backend integration and host
release acceptance remain separate. No screenshot binaries are committed.
