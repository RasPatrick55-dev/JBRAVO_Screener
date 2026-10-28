# Swing research lab: measure entries, exits and selection

The lab is an isolated research CLI, `python -m scripts.swing_research`. It uses
the existing causal signal detectors, common setup ranking and cash-accounting
kernel. It does not import a broker client, schedule work, change strategy
configuration, publish database results or promote a strategy. Production's
six strategies and no-priority ranking stay unchanged.

## Freeze a comparison

Start with an existing, validated `jbravo.strategy-study.v1` specification and
its locally accessible hash-bound raw/split SIP capture, actions and calendar.
The lab accounts for the complete declared universe; it refuses oversized
universes rather than silently taking a subset. The current market-wide capture
adapter retains its existing single-fold admission contract. Use the historical
capture format for a multi-fold study; a current active-universe capture does
not become a survivorship-free historical universe.

```powershell
python -m scripts.swing_research prepare `
  --base-spec C:/research/study.json --axis exits `
  --output C:/research/exit-plan-001.json

python -m scripts.swing_research run `
  --plan C:/research/exit-plan-001.json `
  --output C:/research/exit-study-001 `
  --registry C:/research/experiment-registry

python -m scripts.swing_research experiments `
  --registry C:/research/experiment-registry
```

Each prepared plan has a unique experiment ID. Keep the same registry for all
attempts, including failed ideas. A claimed plan/source combination cannot be
replayed; preparing a new explicit replication produces a new registered
experiment. Output directories and plan files are new and outside the checkout.
The registry retains planned and completed trial counts separately; an
incomplete claim is not a successful study. `STOP` in the output directory, or
`--stop-file PATH`, stops before the next preparation step or simulated session.

The plan freezes all six strategy IDs, source bytes, input manifest hash, action
bindings, costs, phases, policy variants, wall-clock/trial/signal/symbol budgets
and evidence floors. The default limits are 250 symbols, 2,000 qualifying signals,
500 portfolio trials and one hour. Raising a limit is an explicit new plan;
there is no unattended search or paid provider request. A strict cost-stress
case is added when the base study has only one cost case.

`data_kind` defaults to `UNQUALIFIED_MARKET_CAPTURE`; explicitly set
`SYNTHETIC_TEST_FIXTURE` when using generated demonstrations. The report displays
this declaration prominently. Neither value establishes data authenticity.

Every portfolio keeps the current research risk policy: 5% remaining cash per
entry, three proposals, four positions, 3% initial trailing setting, one-ATR
existing stop setting and seven-calendar-day existing time setting. Each fold
and cost case evaluates every strategy separately and a combined portfolio
sharing one cash balance. Standalone family returns are never added together.

## Compare one axis at a time

Exit experiments freeze the entry cohort and regular-session entry model:

| Policy | Rules |
| --- | --- |
| `existing` | Existing ATR/trailing/indicator/time kernel; pattern structural floor and measured target; no partial pattern exits |
| `structural_target` | Fixed structural stop; existing pattern target, or 2R for Pullback/Reclaim; seven-day time exit |
| `atr_stop` | Fixed 1.5 signal-ATR stop from entry, no target; seven-day time exit |
| `fixed_2r` | Fixed structural stop and target at twice initial risk; seven-day time exit |
| `close_trailing` | Structural floor plus a 3% completed-close trailing stop effective next session; seven-day time exit |
| `time_3d` | Structural stop, no target; time exit queued after three calendar days |

Pullback structural support is 0.1% below the raw low of its completed pullback
through reclaim. Pattern support is its original validated structural stop.
These are explicitly different exploratory exit policies, not new versions of
the deployed rules. All variants keep the original pattern entry-level guards.
An experimental stop that is invalid at the baseline entry is counted as an
inapplicable policy attempt and blocks that trial's evidence checks. Portfolio
capacity and position sizes can differ after exits; the standalone fixed-cash
counterfactuals isolate individual signals. Gap stops execute at the opening price; prior stops precede
targets; completed-bar changes cannot tighten a stop retroactively. Time exits
fill at the next captured open. Terminal holdings remain open.

For entry research, prepare a separate `--axis entries` plan. Its exits remain
`existing`. The normal daily-bar portfolio remains the regular-session baseline.
Premarket is a separate paired entry-price sensitivity experiment; it is not
a full premarket strategy backtest:

```powershell
python -m scripts.swing_research prepare `
  --base-spec C:/research/study.json --axis entries `
  --entry-capture C:/research/entry-quotes.json `
  --output C:/research/entry-plan-001.json
```

`--entry-capture` can be omitted. The result then explicitly reports
`UNAVAILABLE_NO_CAPTURED_INTRADAY_QUOTES`. No daily high, low or opening price
is substituted for a premarket quote.

The capture schema is `jbravo.entry-research-capture.v1`, with exact top-level
fields `schema`, `provider: alpaca`, `feed: sip`, `adjustment: raw`,
`currency: USD`, `size_units: SHARES`,
`availability_basis: PROVIDER_EVENT_TIME_ASSUMPTION`, `quotes`, `references`.
Each quote contains `symbol`, timezone-aware `at`, positive uncrossed `bid` and
`ask`, and integer `ask_size_shares`. Each reference contains `symbol`,
`signal_session`, timezone-aware `at`, and `price`; its timestamp must fall in
16:00–20:00 Eastern on the signal session. Retain original provider responses
and any documented size-unit conversion with the capture. Metadata and hashes
establish declared conventions and integrity, not provider authentication or
point-in-time availability certification.

The comparison uses the first quote within 60 seconds of 04:00 and 09:30 Eastern,
with exchange-date/DST-aware timestamps. Premarket's frozen limit is the
after-hours reference times 1.005, rounded down to cents. Only immediately
marketable orders can receive a modeled fill, capped by displayed ask shares
and the fixed per-signal budget. A nonmarketable first quote does not trigger a
search for a favorable later quote or an invented passive fill. Fees and
slippage use the same frozen cost cases. Quote-event time assumes immediate
availability; actual latency, order queues, cancellations, subsequent fills,
premarket stop behavior and portfolio capital sequencing are not reconstructed.
There is no premarket strategy P&L or production-executor-parity claim.

## Read the scorecard

Open `report.html`; detailed evidence remains in `result.json`, each trial's
fills, attempts, daily cash/NAV ledger and equity CSV, cohort files and independent
counterfactual files. `manifest.json` binds every saved artifact. Source copies
and definition hashes permit later review. A registry completion binds the
manifest; failures retain their own permanent claim and completed-trial count.

The cohort retains **every uncapacitated qualifying close**, including setups
outside the global top three. Pullback's consumed-close/rearm rule is enforced
before ranking. Each record includes source-defined strategy identity, common
score evidence, family/global rank, all matching patterns and one deterministic
exit owner. Account held/pending filters are deliberately absent from this
research cohort; it is not a production lifecycle parity claim. The combined
portfolio then applies actual modeled cash/held/pending capacity, with no
fourth-ranked replacement outside the frozen top three.

Every qualifying setup also receives a standalone fixed-cash experiment using
the same exit/cost policy, including rejected setups. These overlapping
counterfactuals cannot be summed into a tradable portfolio. Headline stock-level
comparisons deduplicate pattern matches. Paired selected-versus-rejected session
comparisons require all owner outcomes to be closed; unfilled/open outcomes
remain visible rather than becoming zero returns.

Statistics count fully closed signals, not partial sell fills. They include
net expectancy, average win/loss, payoff ratio, profit factor, costs, holding
days and R multiples when initial risk is known. Entry costs are allocated once.
No-loss profit factor and zero-trade statistics stay null. Drawdown includes
marked open positions; dividend income is separate from closed-trade P&L.
Exposure and turnover describe the portfolio. Full-exposure SPY, cash and
lagged-exposure SPY are diagnostic references with explicit zero-cost/cash-rate
assumptions; none is a matched-beta or executable-performance claim.

MFE/MAE use observed fills and full interior holding sessions. They are **lower
bounds on excursions**, because daily bars cannot establish the ordering of
entry/exit-day extrema. Weekly cluster bootstrap intervals keep same-week
signals together, but do not eliminate dependence across weeks or correct for
multiple experiments. A few trades or a high win rate cannot establish an edge.

## Chronological evaluation and prospective evidence

Prepared folds are labeled development, validation, then holdout. All default
to `PREVIOUSLY_EXPOSED`; a later `DECLARED_UNSEEN` declaration is not independent
certification. No fitting or winner selection occurs. Freeze rules before
evaluating later periods, and retain each failed trial in the registry.

Evidence checks report the frozen minimum sample and signal-week counts,
positive net expectancy and cluster interval, drawdown limit and declared unseen
evaluation. Review every cost case and evaluation fold together. Passing these
checks never changes `research_qualified=false` or authorizes execution.
Prospective confirmation remains `NOT_ESTABLISHED_BY_RETROSPECTIVE_STUDY`.

To inspect real paper execution separately, use an explicitly configured
application database read-only connection and the exact account ID:

```powershell
python -m scripts.swing_research paper-snapshot `
  --account-id ACCOUNT_ID --start 2026-10-28T00:00:00+00:00 `
  --end 2026-11-28T00:00:00+00:00 `
  --env-file C:/private/jbravo.env --output C:/research/paper-cohort-001
```

This exports the enrollment cohort and latest cumulative broker-order
observations using bounded SELECTs, read-only repeatable-read isolation and
rollback/close. It does not call the writable execution Store or broker.
Cancellation/replacement observations repeating one broker order ID are
deduplicated. Claims, observations, quantities bought/sold, unresolved claims,
full exits and latest stop observations are separate fields. Latest snapshots
do not prove continuous protection. Actual fees/cost attribution is absent from
these tables, so only fully exited **gross** P&L is calculated; net P&L and net
expectancy remain unknown. Enrollment alone does not cover rejected scanner
setups or prove profitable prospective trading.

Research motivation: [Bailey and López de Prado, Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
explains selection bias from many trials. [Alpaca's paper-trading specification](https://docs.alpaca.markets/us/docs/paper-trading)
documents differences between simulated and actual execution. These references
do not qualify the bot or certify the lab's statistical estimates.
