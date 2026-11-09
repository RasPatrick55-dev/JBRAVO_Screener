# Mobile Positions field contrast correction

Base: PR795 merge 067c9da61bc9eb623681f1946375a7f825512322, with the existing Source Logs minimum-width correction cherry-picked locally.

Only production change here: MobileField in PositionRow.tsx uses bg-slate-50/75 in light mode, retaining dark:bg-slate-950/45. Labels keep the existing secondary text token, all financial values and positive/negative classes are unchanged. Desktop markup and global styles are untouched.

## Verification

- Five regression tests cover all six field labels and positive, negative, zero and missing captured P/L. Before the surface correction, the contrast-surface test fails; after, all five pass. Source Logs has three separate regression cases that fail without its minimum-width class and pass with it. Combined before run: four failed, four passed.
- Final full frontend suite: 658 tests across 34 files passed. TypeScript no-emit and production build passed. Build emitted only existing large-chunk and stale Browserslist advisories; output was outside the tracked tree. No lint script exists in package.json.
- Supported Edge browser runtime, actual components and app stylesheet, localhost GET-only recorded log fixtures plus synthetic TEST position. Content Security Policy blocks all external providers/fonts/images. No broker/provider or hosted endpoint was called. Fonts therefore use the declared fallback.
- At requested viewport widths 320/375/390/430, actual client and scroll widths both equal 305/360/375/415 respectively, in both Source Logs themes. Desktop 1180 stays three columns with client=scroll=1180. Before correction, the same local reproduction at320 had client305/scroll369. This fixture differs from live font rendering but reproduces the same intrinsic grid-width failure.
- Positions has no page overflow at all five widths in light mode; desktop mobile fields are correctly hidden. Rendered field background is rgba(248,250,252,.75) and label rgb(71,85,105). Dark remains rgba(2,6,23,.45), label rgb(148,163,184).
- Flat screenshot background samples beneath field labels give light contrast 7.03–7.27:1 and dark 6.55–7.04:1, exceeding 4.5:1. Samples use decoded actual JPEG pixels plus DOM foreground colors; screenshot scaling is independent of CSS viewport geometry. Baseline live audit reported2.03–2.25:1, and local before screenshot confirms the same dark-on-gray failure.
- Actual image format decoded with Pillow, dimensions and hashes recorded. Light/dark Positions and320 dark Logs screenshots visually inspected. Temporary browser viewport reset and QA tab closed.

Evidence is local in sibling mobile-layout-evidence: log-widths.json, positions-measurements.json, image-contrast-manifest.json, tests-before.log, full-tests.log, before/after JPEGs, entry.tsx and vite.config.mjs. This is browser responsive QA, not native iOS/Android testing. No publish, PR, merge or deployment.
