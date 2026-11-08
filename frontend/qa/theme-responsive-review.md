# Theme and responsive presentation review

Reviewed 2026-11-08 against base 09f51d2a1b5b47c6631a1debbd9d17dce92c3068.

## Automated checks

- `npm test -- --reporter=dot`: 530 tests passed across 28 files, including 17 new theme and presentation tests.
- `npx tsc -b`: passed.
- `node node_modules/vite/bin/vite.js build --outDir build/theme-qa`: passed. Generated files are not part of this change.
- `git diff --check`: passed.
- Build warnings: existing stale Browserslist database and bundle size above 500 kB. No new dependencies.

## Browser review

Used the supported local browser tooling with Vite bound to loopback. An uncommitted QA config allowed GET-only proxy requests to the existing public PythonAnywhere dashboard API. No trading backend, broker SDK, database connection, or trading operation was started.

- Reviewed all seven tabs in light and dark modes at 1180 x 757 and 390 x 844. No page-level horizontal overflow in the observed states. Horizontal table/navigation scrollers remain intentional.
- Checked Dashboard at 320 x 740. The fixed navigation measured 244 px at 390 px width; main content starts at 260 px, clearing the navigation.
- At 1180 px the monitoring card measured 353.67 px wide and used compact rows. For the observed ARDT row, ticker bounds were x=853.33, y=390, width=34.03, height=18; its $17.30 price was x=813.33, y=430, width=39.36, height=18. The rectangles do not overlap.
- At 900 px the monitoring card measured 837 px wide and displayed all five table columns, approximately 144.2 px each.
- Verified desktop/mobile theme buttons stay in sync after toggling and resizing, and the choice survives tab changes. All screen roots inherit the document theme; no nested `dark` class remains.
- Settled Dashboard and ML API data were inspected in both themes. Light ML heading/secondary colors measured rgb(15,23,42) and rgb(71,85,105). Execute table headers measured rgb(71,85,105) on rgba(241,245,249,0.95).
- Screenshots retained separately in the local review evidence directory, not in source control: dashboard-light-1180.jpg, dashboard-dark-1180.jpg, dashboard-dark-390.jpg, ml-light-390.jpg, ml-dark-390.jpg.

## Limits

This is local presentation QA, not deployed-release verification. The seven-tab sweep includes loading/unavailable states and is not a functional validation of every returned data row, control, or backend source. No production runtime/source behavior changed. Dashboard accent cards deliberately retain dark surfaces in both page themes; their text colors are scoped locally without forcing a page theme.
