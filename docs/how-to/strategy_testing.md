# Swing Research Department: strategy testing and handoff

This is the department's repository entry point for research, paper scanning,
pipeline integration and the evidence needed to follow a strategy into execution.
Research approval, a successful software run and a filled paper trade are
separate decisions. Pullback/Reclaim v2 is enabled for owner-authorized paper
testing; `research_qualified=false` and premarket performance qualification
remain pending. Completing this workflow does not change an earlier research
HOLD or authorize a different trading mode.

## Department workflow

### Six-department review in Paqiyd

Paqiyd coordinates the following review sequence around this existing strategy
pipeline. Each department receives the dated source evidence and preceding
department reports, creates a durable job, and returns its own report.

| Department | Runtime ID | Review output |
| --- | --- | --- |
| Research | `jbravo_research` | Hypothesis, research findings, assumptions and missing qualification |
| Data Provenance | `jbravo_data_provenance` | Session, source hashes, feed/adjustments, lineage and evidence limits |
| Backtest | `jbravo_backtest` | Supplied results, comparison limits and missing evaluation checks |
| Strategy / Trading Strategies | `jbravo_strategy` | Versioned strategy interpretation, risk settings and a qualified disposition |
| Monitoring | `jbravo_monitoring` | Publication/order/fill/protection/exit checklist and exception review |
| Compliance / Governance | `jbravo_compliance_governance` | Approval boundaries, unresolved gates and an owner-facing decision packet |

An `active` registry entry establishes configuration, not successful execution,
research qualification or permission to operate an external system. Retain the
actual worker and department reports, job/run IDs, source bindings and any
`needs_approval`, blocked or failed disposition. Report delivery and approval of
the proposed next action are separate outcomes; do not invent clean background
runs or human reviews to satisfy trust thresholds.

The recurring local department review is scheduled for weekdays at 09:15
America/Chicago. It reviews fresh native receipts and meaningful new session
evidence, preserving prior reports. This Codex follow-up depends on its host;
the existing PythonAnywhere producer, primary, paper-entry and always-on
monitor remain the separate hosted execution path. Browser/voice use also
requires Paqiyd's configured trusted Owner-session issuer and authentication;
local report generation does not prove that interface is provisioned.

Research HOLD and unqualified historical results must not be interpreted as
revoking already authorized experimental paper testing. The existing finite
first-candidate observation is already authorized. Department reports cannot
grant a new trading mode, alter risk settings, replay an old dispatch or submit
an order themselves. New proposed actions retain their owning approval gates.

The [October 27 six-department operations record](../reference/department_operations_2026-10-27.json)
retains this integration's actual results and remaining gates separately from
the earlier technical strategy verification below.

### Strategy pipeline stages

| Stage | Implementation and operator guide | Required handoff |
| --- | --- | --- |
| Research | `research_testing.py`; this guide | Frozen specification, data/action provenance, scenario results and source hashes |
| Strategy library | [Scanner library](strategy_scanner_library.md) | Versioned rules, definition fingerprint and explicit paper-scan enablement |
| Daily data and Screener | [Market-wide screening](market_wide_strategy_screen.md) | Complete asset accounting, current decisions, ranked proposals and independently verified research archive |
| Primary input validation | [Primary strategy route](primary_strategy_screen.md) | Exact-session source-bound producer archive and candidate values |
| Backtest | [Strategy Backtest](strategy_backtest_stage.md) | Same candidate signal IDs, strategy/version/definition and historical cohort |
| Metrics | [Strategy Metrics](strategy_metrics_stage.md) | Bound Backtest result, unchanged candidate population and honest unknown statistics |
| Production publication | [Primary strategy route](primary_strategy_screen.md) | Atomic PostgreSQL publication of candidates, Backtest and Metrics; completed primary report |
| Paper execution and monitor | [Execution handoff](strategy_execution_handoff.md) | Fresh production handoff, actual paper-account ownership, observed orders/fills and protective lifecycle evidence |

The production route currently supports one strategy,
`trend_pullback_reclaim_qualifying_close_v2`, version `2`. Adding an entry to the
scanner registry does not automatically connect its production publication,
account allocation, executor or exit monitoring. Several strategies can be
compared in isolated studies; shared paper-account capital across strategies
still needs an explicit contract.

For a new strategy, retain its research proposal and review decision; freeze
its data convention, causal buy/exit rules, costs, risk settings and versioned
definition. Test the common kernel and a positive signal handoff before enabling
its paper scan. Carry that exact identity through Backtest, Metrics and the
execution state tables. Verify both an empty batch and a nonempty batch, then
observe actual paper fills and monitor protection. Never relabel an existing
strategy ID or infer research promotion from a passing technical test.

### Daily operations and proof

The installed PythonAnywhere task definitions are authoritative. The verified
October 27 configuration is:

| Operation | Task ID | UTC schedule | Dependency |
| --- | --- | --- | --- |
| Market-wide producer/Screener | 1558334 | 01:20 daily | Completed signal data after 20:15 Eastern |
| Primary, Backtest, Metrics and publication | 1322138 | 03:45 daily | Completed exact-session producer archive |
| Paper entry | 1326478 | 07:55 daily start | Fresh primary and monitor; target entry 04:00 Eastern |
| Strategy-aware position monitor | 228489 | Always on | Actual strategy-owned orders/positions and frozen lifecycle rules |

Convert UTC schedules using the exchange date and current Eastern/Chicago
offsets. The entry worker may wait for 04:00 Eastern; its process start is not
the order time. Exchange-calendar non-trading days and changed session hours
must remain explicit. Read schedule metadata and command hashes through the
shared PythonAnywhere MCP; do not copy raw task commands or private environment
contents into departmental records.

Trace one candidate by signal ID through the producer's `scan/paper_handoff.json`,
the exact PostgreSQL batch, primary `handoff`, Backtest candidate IDs and Metrics
rows, then the execution signal/order records. The report locations and recovery
rules are in the execution guide. A source-bound completed report with separate
database readback is required; a nonempty local audit file is insufficient.

Record these outcomes separately: candidate published, order submitted, quantity
filled, protection observed, cancellation confirmed and position exited.
Submitting a limit order does not prove a fill. A healthy flat-account monitor
does not prove stop or exit behavior. A valid zero-signal scan is a successful
screen with an incomplete buy demonstration. Keep the approved rules and wait
for a fresh qualifying session rather than insert a test candidate.

The [October 27 verification record](../reference/pullback_reclaim_verification_2026-10-27.json)
and [dated market-wide findings](market_wide_strategy_screen.md#verified-deployment-and-audit-october-27-2026)
distinguish the completed technical audit from the still-pending production
candidate, paper-fill and monitor demonstration. These are dated evidence,
not current execution inputs or instructions to rerun a consumed packet.

## Run an isolated strategy study

`python -m scripts.research_testing` is the research entry point. The new research
and strategy-scanner paths share the pure `scripts.backtest_core` portfolio kernel,
extracted from the legacy backtest. They do not import the legacy CLI's environment,
logging, provider clients, input backfill or database writes. The deployed legacy
backtest CLI remains unchanged by this library installation.
Importing the research modules performs no provider/database operation.

## Prepare and run

Use a fresh hash-bound capture produced from the connected Alpaca/Massive tools.
The current input adapter consumes the saved Alpaca SIP raw and split daily CSVs
and calendar bound by the capture manifest. Massive is an independent comparison
source; provider disagreement does not silently overwrite the Alpaca series.
The supplementary corporate-action capture must cover the study and retain a
terminal page token. Extend its process-date request beyond the final session to
capture announced dividends whose payment follows that session.

```powershell
python -m scripts.research_testing prepare-study `
  --capture-manifest C:/research/capture/analysis002/manifest.json `
  --actions-capture C:/research/actions.json `
  --output C:/research/study.json

python -m scripts.research_testing inspect-data --spec C:/research/study.json

python -m scripts.research_testing run `
  --spec C:/research/study.json --output C:/research/results/study001
```

The output directory must be new and outside the source tree. The runner freezes
the specification and actual source bytes before evaluating. It saves failed runs
as well as successful ones. Create `STOP` in the output directory, or provide
`--stop-file PATH`, to stop before the next session/trial. A stopped simulation
instance cannot be replayed; new execution requires a new instance/output. The
frozen wall-clock and scenario budgets are enforced. There is no scheduler,
automatic parameter search, automatic winner selection or promotion operation.

The starter specification fixes two methods, three nonoverlapping evaluation
periods and two cost cases, with a separate benchmark for each period/cost case:
18 scenarios total. These are retrospective evaluation periods with prior data
exposure, not certified untouched out-of-sample tests. Features use earlier
warmup observations; cash, positions and challenger consumption start fresh for
each evaluation period. No fitted model, training or optimization runs.

The starter strategies allocate 5% of available cash per entry, at most four
positions. The SPY benchmark invests available cash, reserving fees/slippage.
Their exposure differs substantially; raw return differences alone are not a
matched-risk or matched-exposure comparison.

## Exact signal and execution scope

- `legacy_simulator_core_v1` uses the existing `compute_indicators` and
  `composite_score` formulas. It is the legacy portfolio-simulator baseline,
  **not a screener-v2 signal-parity claim**. The original hammer formula and
  indicator conventions are retained.
- `trend_pullback_reclaim_qualifying_close_v2` applies the retained R02 method:
  trend, rising MA50, 2â€“5 session EMA20 pullback bounds, bullish reclaim above
  yesterday's high, positive ATR, no lower trough, and volume above the preceding
  20 sessions. Both current and preceding required indicators and 250 observations
  must be ready. After common risk handling, qualifying closes consume before
  ranking/capacity even when unselected. Held/pending/unready/failed setups do not
  consume; another setup must start strictly after the consumed close. The R02
  synthetic consumer's historical refusal is preserved in its original artifact;
  this new explicitly exploratory adapter does not relabel historical inputs as
  synthetic or claim earlier acceptance for its historical execution.
- Both methods rank by the legacy composite score and use its portfolio risk
  rules. Indicator construction is causal rolling/EMA computation; append-future
  tests check prefix features, signals and fills through the normal caller.
- Signal availability is an explicit **assumption** of 20:15 Eastern after the
  daily aggregate session. Buys occur at the next supplied regular-session open.
  This differs from the production premarket passive-limit execution model.
- Fees are USD per fill; slippage is basis points per side. Entry stops use the
  signal bar's ATR in raw share units. Opening gaps and preexisting stops precede
  close-derived tightening. Daily bars do not reveal actual intraday path,
  liquidity, spreads or fill probability.
- Cash dividends accrue from prior-close holdings on ex-date. A later ex-date buy
  receives no entitlement. Payments become cash on or after payable date at the
  next evaluated session; unpaid entitlements remain NAV receivables. Gross rates
  assume USD; withholding/tax/cash-in-lieu and completeness remain unqualified.
- Split feature prices and raw fill prices remain separate. v1 refuses changing
  split factors and unsupported corporate actions. It does not invent share,
  stop, basis, symbol-continuity or merger transformations. Missing selected
  sessions and quality flags also fail; they are not filled or silently dropped.
- At the end, entries without another evaluated session are canceled. Existing
  positions and pending exit obligations remain explicitly marked; no invented
  liquidation occurs. Exit-fill win rate includes partials and is labelled as
  such. Zero-exit profit factor/win rate is null rather than fabricated infinity.

The starter uses ten explicitly selected symbols plus SPY. It records NBIS,
VSCO/VSXY, HBAN and RGLD exclusions with their quality/continuity/action reasons.
This narrow selected-symbol result cannot establish market-wide performance or
survivorship-free selection. Original captures and earlier department budgets and
decisions are preserved; this study has its own explicit scenario ledger.

## Artifacts and review

The [strategy scanner library](strategy_scanner_library.md) now exposes the
Pullback/Reclaim strategy through the screener CLI, with stable signal IDs and
paper entry/exit traces. It shares this research engine rather than reimplementing
its predicates or portfolio rules.

Each successful study includes `specification.json`, actual `source/` files,
`dataset.json`, one folder per scenario, `result.json`, `report.html` and a hashed
`manifest.json`. Scenario folders contain complete signals, fills, action events,
daily reconciliation, trades and equity. The HTML is standalone and uses existing
Plotly when available; no paid account or external chart service is required.

All study results retain `research_qualified=false` and
`execution_authorized=false`. Data-use acceptance, historical-universe evidence,
independent strategy review, prospective confirmation and any operational
integration remain separate from completing an exploratory job.

## PostgreSQL

The repository's existing `DATABASE_URL`, or documented `DB_*` fields, supply
read access. `--env-file` explicitly loads an existing local ignored env file in
the process; it never prints its contents. See [the operator runbook](ops_runbook.md)
for the PythonAnywhere SSH tunnel. A configured URL is not successful access.

```powershell
python -m scripts.research_testing --env-file C:/private/jbravo.env db-status
python -m scripts.research_testing --env-file C:/private/jbravo.env db-snapshot `
  --symbols SPY,ALK,FN --start 2023-01-01 --end 2026-10-23 `
  --output C:/research/postgres-snapshot.json
```

Snapshots use a read-only repeatable-read transaction, a 15-second statement
timeout and a 250,000-row cap. Exceeding the cap fails; no truncated snapshot is
accepted. Existing `public.daily_bars` has no feed/adjustment/currency provenance,
so snapshots preserve those values as UNKNOWN and are not automatically combined
with provider data for strategy evaluation.

Optional archival support requires an explicit `RESEARCH_DATABASE_URL`; writes
never fall back to the production URL or `DB_*`. `init-research-db` creates only
`jbravo_research.research_runs`. `publish-result --result PATH` appends one full
study JSON with source/config/dataset/trial hashes. Duplicate run IDs reject;
there is no upsert. These explicit commands are separate from `run`; no schema
or publication occurs during a study or read-only snapshot. Review and authorize
the exact database publication before using them under host operating rules.

## Verified PythonAnywhere study, October 25, 2026

The isolated release at
`/home/RasPatrick/jbravo-research-tools/STRATEGY_TESTING_20261025_002`
completed its first native study and PostgreSQL archive. Python 3.10.12 and the
eight pinned dependency versions matched. All 18 frozen scenarios passed full
parity; independent reconstruction reconciled 5,096 fills, 4,236 daily ledgers and
190 action events across 137 bound artifacts. A separate local read-only connection
verified the committed record in `jbravo_research.research_runs`:

`9d3ae0aa990c74ac3a6a1b3cc6908a78a8ff2d6eeac2a38cf0d5805b361e619a`

The root contains immutable completion/permission/archive receipts and
`results.zip`. The standalone report is
`runtime/study001/report.html`. Full verified local copies and the deployment
handoff are under
`C:/Users/RasPa/PythonAnywhereReviews/2026-10-25-strategy-testing-deployment-002/`.

The approved correction removed group-write bits from the owned home/new
release directories and its four uploaded files. Earlier failed packets remain
unchanged and permanently consumed. Never rerun this release launcher. Future
host studies need new reviewed specifications, output roots and scoped plans;
this completion creates no recurring schedule. Strategy/data qualification,
screener integration and production promotion remain pending separate review.

## Free tools checked October 25, 2026

| Tool | Useful role | Current integration |
| --- | --- | --- |
| [Plotly Python](https://github.com/plotly/plotly.py/blob/main/LICENSE.txt), MIT | Standalone interactive equity reports | Used when installed; existing project dependency |
| [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars), Apache-2.0 | Additional session/holiday/early-close audit | Candidate audit tool; community calendars do not independently prove exchange eligibility |
| [QuantStats](https://github.com/ranaroussi/quantstats), Apache-2.0 | Additional tear sheets and portfolio statistics | Optional follow-up; feed saved return series to avoid implicit market-data downloads |
| [DuckDB](https://duckdb.org/), open-source analytical database | Fast local CSV/Parquet inspection without changing PostgreSQL | Optional follow-up; not another canonical database |
| [Backtesting.py](https://kernc.github.io/backtesting.py/), AGPL-3.0 | Independent simple-strategy comparison | Evaluated; not added as a competing portfolio engine |

The core workflow needs the project's existing pandas, NumPy and optional Plotly;
PostgreSQL commands additionally require psycopg2 and env-file loading uses
python-dotenv, both already in the repository's requirements. No paid software or
additional service account was introduced.
