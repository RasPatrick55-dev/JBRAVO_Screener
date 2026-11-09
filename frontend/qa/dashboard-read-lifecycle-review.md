# Dashboard read lifecycle review - 2026-11-09

## Scope and basis

Source-only repair based on freshly fetched canonical main
`ab52f68253f799dd39084773deb1a3fd759257fc` (PR797). It does not change backend
contracts, providers, database configuration, schedules, trading actions or themes.
Parent-supplied observations included successful public reads at 10.31 seconds
(Backtest), 11.85 seconds (Account), 24.49 seconds (ML), 26.805 seconds (Execute),
and 16.68 seconds (provider logs). Some reads exceeded 35 seconds during page
bursts. These were supplied diagnostic observations, not repeated by this worker.

## Changes

- Shared maximum of four dispatched reads, FIFO queue, 45-second full-body
  deadlines, abort-on-disposal, and suppression of late results. Queue wait does
  not consume a network deadline. Scorecard's existing 35-second override stays.
- Completion-based polling prevents overlapping requests and immediate timeout
  retry storms. Query changes cancel queued/active reads and clear prior query
  data. Repeated in-flight manual refresh clicks are coalesced.
- Execute state/audit no longer instantiate EventSource or run simultaneous
  stream/fallback timers. Initial failure ends skeletons; a subsequent pending
  retry does not relabel unavailable data as initial loading.
- Same-query refresh failures retain stale rows and receipt timestamps. All
  source validity, held performance, partial/unknown amounts, exact strategy
  report bindings, and diagnostic-versus-broker distinctions remain in force.
  Execute P/L and entry-window claims still fail closed on refresh failure.
- Account's canonical endpoints publish independently. Existing normalizers are
  preserved; unhealthy or held envelopes cannot populate valid-looking amounts.
- ML keeps one-shot behavior, finite read/cancellation, and unchanged artifact
  timestamp/age metadata. Navbar clock reads are now finite and cancellable too.
- Empty/unavailable messages distinguish unknown order counts, protection, log
  activity, portfolio history and audit findings; no synthetic zero fallback.

## Regression coverage

`src/components/ui/readLifecycle.test.ts` covers successful 10.31/11.85/24.49/
26.805/35.001-second reads, hung body decoding, abort/timer cleanup, a transport
ignoring cancellation, polling cadence/manual coalescing, bounded FIFO bursts,
queue deadlines and queued navigation cancellation.

`qa/DashboardReadLifecycle.test.tsx` mounts real consumers in jsdom with mocked
JSON responses. It covers superseded loading/error transitions, new-query reset,
React StrictMode, stale receipts, Execute failure/retry skeletons and no-SSE,
slow Picks/Metrics retention, log-stage receipt reset, independent Account
publication/failure invalidation, and one-shot slow/failed/cancelled ML reads.

The legacy Execute fixture is copied from canonical `ab52f68`, with imports
adapted only. Its test confirms stream `onerror` starts finite state fallback,
cancels the 2.5-second fallback timer, and leaves the existing 20-second poll
working. This does not test a real browser's terminal EventSource handling of
HTTP204; that distinction is explicit in the test. The new production client
never opens either Execute stream.

## Verification

- Full frontend suite: **37 files, 695 tests passed**.
- TypeScript plus Vite production build: passed with `--emptyOutDir false`.
- Existing mobile navigation, theme, source-log width/position contrast, held
  accounting, gross-versus-net and simulation/provenance tests remain passing.
- `python -m scripts.docs_consistency_check`: passed. Generated CLI help changed
  only local line wrapping; that unrelated generated difference is restored.
- `git diff --check`: passed.

Intermediate tests exposed old 5/10-second timeout expectations, hook-order mocks,
a consumed mock Response reused across polls, two new-test fixture mistakes, and
an attempted stale Execute P/L display. All were corrected; the existing P/L
failure gate is retained. No failing stages are omitted from this account.

Build warnings remain the pre-existing stale Browserslist dataset and >500kB
bundle advisory. SSR tests emit the existing Navbar `useLayoutEffect` warning.
No provider/API/browser requests or production deployment were used for testing.

## Release handoff

This commit contains source/tests/documentation only. Tracked dist assets and
index are restored before committing. The integrator must rebuild the combined
backend/frontend release with `--emptyOutDir false`, retaining all prior hashed
assets for cached tabs; do not deploy this worktree's intermediate index.
Acceptance proves local lifecycle behavior, not current provider/database
availability, broker connectivity, live performance qualification, or a
successful production rollout.
