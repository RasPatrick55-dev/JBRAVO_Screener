# Source Logs Board mobile width correction

## Scope

The release008 live acceptance found an existing Source Logs Board overflow at
390×844. The relevant LogsBoard, LogsPanel, LogSourceReceipt and Tailwind files
were unchanged from release007. This follow-up starts from merge 067c9da and does
not change the deployed release008 source worktree, compiled assets or packet.

The production change is one `min-w-0` class on the LogsPanel section, which is
the grid item inside LogsBoard. A long receipt source contributed an automatic
min-content width; `break-word` alone did not let the item shrink. Setting its
minimum width to zero permits the existing wrapping rules to work. There are no
new global styles, hidden overflow, palette changes, request changes or data
changes.

## Regression fixture and tests

`sourceLogsLongReceiptFixture.json` reconstructs the exact source and receipt
fields visible in the live DOM on 2026-11-08 at approximately 23:45 UTC. It has
two screener receipts and three each for Backtest and Metrics. No new API request
was made to obtain it. Log rows are deliberately omitted: the regression is the
receipt text's intrinsic width, not log event content.

The three request/state/render cases in `SourceLogsMobileWidth.test.tsx` fail on
the original implementation and pass with the correction. They retain each
full source, timestamp basis, event and fetch timestamp, the existing card
palette and the truthful log-evidence caveats. They guard the grid-item minimum
without substituting clipping. These are structural render tests; actual CSS
width is verified separately in the browser.

Full frontend validation: 653 tests passed across 33 files. TypeScript no-emit
check and baseline-to-change diff check passed. No compiled assets were changed.

## Actual browser reproduction

A localhost GET-only fixture server rendered the real LogsBoard and LogsPanel
components with the app stylesheet and the recorded receipt fixture. No
PythonAnywhere, broker or provider endpoints were called. The reproduction
matched live geometry before the fix:

- Requested viewport: 390×844; document client width: 375px.
- Before: document scroll width 394px; each card 378.375px wide, right edge
  394.375px; computed min-width `auto`.
- After, light and dark: document scroll width and client width both 375px;
  cards 343px wide, right edge 359px; computed min-width `0px`.
- All eight receipt paragraphs: client and scroll widths both 317px, with the
  existing `overflow-wrap: break-word` rule and full text retained.
- Desktop 1180×757: document scroll width and client width both 1180px; the
  existing three-column layout and colors remain intact.
- No console warnings/errors. The local theme was restored, viewport reset,
  QA tab closed and fixture server stopped. Port5184 had no remaining listener.

Screenshots and exact DOM measurements are in the sibling local review folder
`source-logs-mobile-wrap-evidence`. They include `before-width.json`,
`after-light-width.json`, `after-dark-width.json`, `after-desktop-width.json`
and four visually inspected JPEG captures. A format-aware Pillow decoder
validated the images; the manifest records actual dimensions and hashes. The
requested browser dimensions are independent of screenshot capture scaling.

No push, PR, deployment or production modification was performed.
