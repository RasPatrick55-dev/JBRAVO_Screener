# Strategy scanner library

The first enabled strategy is **Trend Pullback / Reclaim v2**:
`trend_pullback_reclaim_qualifying_close_v2`. The owner authorized adding it for
paper testing. Research acceptance and strategy promotion remain pending.

The library is in `scripts/strategy_library.py`. Each strategy has an ID,
version, displayed buy rules, code fingerprint and explicit paper-scan enablement.
New entries default to disabled. Adding another catalog entry does not reset
the first strategy's signal identities. Each selected strategy has its own
paper portfolio; shared capital allocation across strategies is a future contract.

## List and scan

```powershell
python -m scripts.screener --mode strategy-library

python -m scripts.screener --mode strategy-scan `
  --strategy trend_pullback_reclaim_qualifying_close_v2 `
  --study-spec C:/research/study.json `
  --as-of 2026-10-23 `
  --output-dir C:/research/scans/pullback-2026-10-23
```

Repeat `--strategy ID` to select several enabled strategies. Omitting it selects
the explicitly enabled library entries. An unknown, duplicate or disabled ID
fails; it never falls back to a different strategy. `--strategy-fold` and
`--strategy-cost-case` select declared episode and cost cases.

The input specification uses the hash-bound raw/split captures, calendar,
corporate actions, fixed universe and risk settings described in
[strategy testing](strategy_testing.md). It must explicitly contain the selected
research method and requested session. A subsequent daily scan needs a fresh
verified capture and a specification extended through the new session. Keeping
the episode start/universe/risk contract fixed preserves signal identities.
The output root must be new and outside the repository. Failed runs keep their
claim and failure receipt; never overwrite them to retry.

The explicit CLI consumes verified captures. The daily worker in
`scripts/daily_strategy_scan.py` refreshes its own captures and invokes the same
scanner. Its configuration is `config/strategy_scan_daily.json`: the initial ten
tested stocks plus SPY, the fixed 2026 paper episode, unchanged risk settings,
SIP feed, raw execution prices and split-adjusted feature prices. It never
substitutes IEX, stale files or invented calendar sessions.

The dedicated PythonAnywhere library task runs daily at **01:20 UTC** (20:20 or
21:20 Eastern on the preceding date). It therefore remains after the assumed
20:15 Eastern availability cutoff across daylight-saving changes. Exchange
calendar non-trading days are skipped. Every trading session has a permanent
single-use output directory; interrupted or completed sessions are not silently
overwritten. A bounded initial bootstrap may scan the latest completed session.

Each completed daily scan is appended to the existing PostgreSQL
`jbravo_research.research_runs` archive under its own daily strategy identity,
then independently read back in a separate read-only transaction. The record
contains the signal proposals, data/source hashes, output location and explicit
non-execution status. No new schema is initialized by the worker.

This task screens the initial cohort and records paper proposals. The existing
legacy primary job, production candidate tables and order executor remain a
separate route. Routing daily library proposals to orders is a later integration;
these captures and simulations do not grant order submission authority. The
library CLI dispatches before legacy credential bootstrap or trading-account probes.

## Follow a signal through the process

The scanner calls the exact research feature/predicate preparation and portfolio
kernel. It reconstructs state from the declared episode start through `--as-of`:

1. Check the completed daily bars and 250-session feature readiness.
2. Apply the Pullback/Reclaim conditions to every selected symbol.
3. Apply held/pending exclusions and strict rearming. Qualifying closes consume
   before ranking/capacity, including setups not selected for a portfolio slot.
4. Rank qualifying setups using the retained composite score; apply the same
   allocation and position limits as the study. No v2 ranker, ML, sentiment or
   target-count filler can create a Pullback/Reclaim signal.
5. Save selected proposals with stable signal IDs, raw signal close/ATR,
   allocation settings, assumed availability and next supplied session/open.
6. Replay subsequent supplied sessions through modeled entries, partial/full
   exits and reconciled cash/equity. Every fill and lifecycle row retains its
   originating signal ID and strategy version.

The strategy uses the tested R02 rules: close above MA50 above MA200, rising
MA50, bounded 2–5-session EMA20 pullback, bullish close above EMA20 and the prior
high, volume above the preceding 20-session mean, and no lower pullback trough.
The original rearm/consumption semantics and risk engine are unchanged.

The signal clock remains an assumption of 20:15 Eastern after the aggregate
session. Paper fills use the next supplied 09:30 regular-session open, with
the study's fees/slippage/stops. They are simulations, not Alpaca paper-account
orders or the existing premarket passive-limit execution model. An unknown next
session stays `WAITING_FOR_NEXT_CALENDAR_SESSION`; no Monday/holiday is invented.

The same logical signal ID survives daily replay and is linked to its input
batch's hashes. A future order consumer must bind that exact batch, check revised
data/expiry and deduplicate the signal; these proposal files grant no submission
authority. A terminal selected setup is retained as a proposal, without inventing
a fill outside the supplied evaluation period.

## Outputs

- `result.json`: run, strategy/version/fingerprint, provenance and funnel counts.
- `top_candidates.csv` and `paper_handoff.json`: this session's selected paper
  proposals. Empty files with headers/empty arrays are valid no-buy outcomes.
- Per-strategy `latest_observations.json`: every symbol's current decision and
  failed entry conditions, including warmup/held/pending/rearm reasons.
- `observations.json`, `intents.json`, `fills.json`, `lifecycle.json`: complete
  chronological audit from setup through modeled outcomes.
- `actions.json`, `ledger.json`, `metrics.json`: the shared accounting outputs.
- `report.html`: standalone human-readable buy rules, signal/entry/exit/P&L
  traces and latest daily decisions.
- `specification.json`, `dataset.json`, `source/` and `manifest.json`: actual
  specification, input provenance, source bytes and artifact hashes.

These artifacts retain `research_qualified=false` and explicit
`order_submission_authorized=false`. They do not alter the separately authorized
existing paper executor or qualify the strategy based on software parity alone.
