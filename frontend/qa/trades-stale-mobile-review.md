# Trades stale-read and narrow-layout follow-up

Base: 8d953441 (PR801 merge). Frontend source/tests only. No backend, accounting,
database, provider, host or release012 files are changed. Generated build output
is outside the worktree. Nothing has been published or deployed by this task.

## Saved evidence and diagnosis

The release011 UI observation at 2026-11-09 17:31 UTC had a populated PARTIAL
execution-gross leaderboard. An ordinary refresh later displayed UNAVAILABLE,
coverage unknown and a new receipt time. The exact request remains
`/api/trades/leaderboard?range=all&mode=winners&limit=12&basis=execution_gross`.
The source route and parser agree on that shape. A transport error retains the
previous payload as stale, whereas an HTTP200 unavailable envelope replaced it.

Both gross routes call `recorded_trade_analytics.read_view`, which runs a bounded
PostgreSQL projection and canonical rechecks, then converts any exception into
the same CURRENT_EXECUTION_GROSS_INPUTS_UNAVAILABLE envelope. Existing limits are
2 seconds for connection setup, 8 seconds total, bounded statements and lock
waits. Saved sequential/paired comparisons all succeeded. Deadline/contention
is plausible, but connection, policy, identity and cohort drift cannot be
distinguished from the saved evidence. This patch does not claim a native cause
or relax any backend guard.

The saved320px UI measured client width305px, document scroll width308px, and
legacy section width291.56px inside a273px column. The implicit auto grid column
inherits the minimum content width of the leaderboard's non-wrapping five range
buttons. The strategy card is outside that nested grid and already fits.

## Changes and safety boundary

- Explicit single-column minmax(0,1fr) grid at mobile widths, shrinkable grid
  children/cards, and five equal narrow range controls. Desktop classes and
  palette are preserved. No body overflow hiding or content removal.
- Optional source-read policy is enabled only for execution-gross stats,
  leaderboard and strategy history. Other source consumers are unchanged.
- The exact allowlisted gross unavailable envelope may retain a prior
  contract-valid observation from the SAME request. Values are dated STALE;
  current source is explicitly unavailable and not current performance. Partial
  coverage, unknown values and accounting holds remain intact. The underlying
  generic failure might include policy/cohort drift, so retained rows are never
  a current qualification or authority assertion.
- Unexpected envelope keys (including account mismatch), explicit policy holds,
  wrong basis, malformed contracts, future timestamps and other reasons do not
  reuse previous rows. Filter changes start a separate observation and abort the
  old request. There is no cross-range cache.
- A first unavailable read remains unknown. Successful-data time advances only
  after an accepted contract read. Failed-attempt time is separate.
- Strategy history still hides rows on a business failure. Its incorrect
  'Last successful read' timestamp no longer advances on HTTP200 unavailable.
  Existing transport-failure stale behavior remains explicitly labeled.

## Verification

- Offline dependency install: passed, scripts disabled, no network install.
- Full frontend suite:780 tests passed across42 files, including16 new tests.
- New coverage: ready/partial to unavailable to recovery; initial failure;
  explicit identity mismatch, policy hold, wrong basis, malformed/null/future
  input; no change to non-opted-in sources; filter navigation/abort/late response;
  strategy successful versus failed timestamps; responsive class structure.
- TypeScript no-emit check: passed.
- Vite production build: passed, outside the worktree. Existing-style warnings
  reported an old Browserslist database and a chunk above500kB.
- Documentation consistency and diff whitespace check: passed.
- No browser calls were made during preparation, per coordination restriction.
  Actual post-patch320/390px and desktop pixel validation remains to be done in
  a localhost-only fixture before declaring the layout visually accepted.

Local evidence is in the sibling `trades-stale-mobile-followup-evidence` directory.
The source diagnosis is bounded: it improves honest display continuity, not
backend availability, accounting completeness or current account authority.
