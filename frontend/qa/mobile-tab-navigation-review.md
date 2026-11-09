# Mobile tab navigation regression checks

Base: 067c9da61bc9eb623681f1946375a7f825512322 (PR 795).

Each screen mounts its own navbar, so horizontal scroll starts at zero. The navbar
now reveals its active button by adjusting only its own scrollLeft on mount,
active-label changes, and resize. Poll/clock rerenders do not override manual scroll.
App resets document scroll only when the committed top-level tab changes, restores
focus to the new active native button with preventScroll, and supports Back to an
initial empty hash as Dashboard. Unknown hashes remain ignored. Initial direct
hash entry and same-tab activation do not reset scroll or seize focus.

Verification on Node v26.13.0 / npm 11.20.0:
- Mounted jsdom regression: 8 failed / 1 passed against unchanged base source;
  all 9 pass after fixes. Screens are stubbed in App tests to isolate navigation;
  navbar tests mount the real component with explicit geometry because jsdom does
  not calculate layout. Coverage includes 320/375/390/430 widths, remounts, resize,
  repeat/latest/same-tab navigation, StrictMode, empty/nonempty history hash events,
  initial direct hash, focus restoration, unchanged-label renders and cleanup.
- Full frontend: 33 files / 659 tests passed.
- TypeScript: tsc --noEmit -p frontend/tsconfig.json passed.
- npm ci --ignore-scripts passed in this worktree's physical node_modules directory.
- jsdom 26.1 is dev-only, with 37 added dev dependency entries. No existing lockfile
  package entries or runtime dependencies changed. npm reports the same 16 audit
  findings before and after addition; no unrelated dependency upgrades were made.

Browser visual/keyboard/history acceptance is a separate integration check. jsdom
hash events are not a real browser history traversal or layout proof. Confirm
normal pointer and Enter/Space activation, Back/Forward including empty hash,
visible selected Execute/Screener/ML tabs and page-top entry, same-tab preservation,
and light/dark desktop/mobile screenshots using the supported browser/CUA tools.
Static server-render tests emit React's expected useLayoutEffect SSR warning;
the dashboard entrypoint uses client createRoot, not hydration.
