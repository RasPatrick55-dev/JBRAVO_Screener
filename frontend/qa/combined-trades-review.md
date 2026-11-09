# Combined Trades UI review

## Scope

One read-only `/api/trades/combined-gross` source drives the actual Trades tab.
The separately qualified `/api/trades/overview` and account equity consumers are
unchanged. Python supplies lifecycle inclusion, deduplication, exact gross math,
rolling UTC ranges and rankings. TypeScript validates the wire and formats it.

Synthetic source evidence is `fixtures/combinedGrossWire.synthetic.json`, schema
`jbravo.trades-overview.v1`. It contains no real account or order identifiers.
Three known contributions are 4 + 80 - 40 = 44, with one excluded candidate,
one unknown candidate, one open recognition-only signal and one unfilled signal.

## Automated checks

- `npm test`: 855 tests in 43 files passed on the final backend fixture.
- 75 new tests in `CombinedTradesOverview.test.tsx` exercise the real production
  component/adapter and its shared request lifecycle, not the historical fixture.
- 81 old composition tests in seven suites continue through a QA-only frozen
  `LegacyTradesTab`. These preserve legacy contract regressions and are not
  acceptance of the new production composition.
- 699 other existing tests remain in the full suite.
- `npm run build`: TypeScript and Vite pass. Existing large-bundle and outdated
  Browserslist-data warnings remain. Test logs also contain existing SSR
  useLayoutEffect warnings; no test is skipped for them.

New checks cover backend-wire admission, source/account/fee/net holds, gross
zero versus unknown, malformed/duplicate identities and rankings, coverage
limits, unknown-only and empty populations, recognition-only/unfilled signals,
coherent precomputed filter changes, same-snapshot pagination, new-cohort page
reset, first-read failure, business-failure clearing, pending read deadline,
whole-cohort stale retention, successful recovery, unmount cancellation, and
unchanged qualified overview/account source contracts.

## Local visual QA

Run `npm run dev -- --host 127.0.0.1 --port 5181 --strictPort`, then open
`http://127.0.0.1:5181/qa/combinedTradesPreview.html`.

The preview imports a static fixture and does not mount any source hooks.
CSP blocks external origins; only same-origin assets and the local Vite HMR
connection are allowed. No API, database or broker reads, provider writes or
credentials are used. Scenario controls: populated, paginated, empty, unavailable, loading,
stale and malformed. Theme toggle and the real leaderboard controls are active.
The immutable backend fixture has five candidates. A clearly labeled QA-only
paginated scenario adds 25 synthetic excluded identities and matching coverage
counts; the eligible server gross values stay unchanged. Positive paging is also
covered in the actual component test.

A separate local visual pass accepted 320/390-pixel light/dark layouts and
desktop with no page overflow. Expanded source details and long identifiers fit.
Positive Next/Previous paging and changing cohorts from page 2 reset to page 1
were verified. Loading and malformed-source wording fixes were rechecked.
Viewport captures succeeded; the full-page screenshot operation timed out.
The live app was not touched. The final backend wire was confirmed byte-identical
to the visually reviewed fixture: 16,978 bytes, SHA256
`6b31f42698e5e8e4af8a81c5a7802f1c7453003df5f7536685d34fa27ec9aecf`.
It is reproducible through `tests/test_combined_trade_gross.py::synthetic_wire`
after backend integration.

## Verification history and limits

The first full run found 55 failures in old tests whose fixture routing assumed
the retired five-endpoint tab. Their historical composition is now explicit;
they have not been counted as new-UI acceptance. The first new-suite pass was
69/71: its static-file URL and recovery timestamp harness assumptions were fixed,
then all 75 expanded current-UI cases passed. No tests were disabled or skipped.
The visual review caught loading copy that said unavailable before settlement;
that wording and malformed-source account-binding copy were corrected and
regression assertions added. The offline preview blocks external web fonts,
so visual measurements use the existing fallback font stacks.
