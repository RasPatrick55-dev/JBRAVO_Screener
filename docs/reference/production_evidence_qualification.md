# Production-score qualification and monitor candidates

`python -m scripts.production_evidence` joins actual forward score captures to
subsequently observed, split-adjusted daily closes. It reads exclusively from
an explicitly supplied evidence packet. It never loads a model, queries a
provider/database, publishes a monitor, changes a schedule or places an order.
All input references require exact byte length and SHA-256. Paths are relative
to that packet; symlinks, traversal, ambiguous times, duplicate identities and
changed bytes are rejected.

## Required independent inputs

1. A `jbravo.production-evidence-plan.v1` record approved **before** the first
   baseline trading open. Its approval records `approval_id`, `reviewer` and
   aware `approved_at_utc`. It fixes the full model bundle, three producer source
   identities, ordered features, target `label_5d_pos_300bp`, score `score_5d`,
   symbol universe, bound exchange calendar, disjoint baseline/recent windows,
   explicit `window_lengths` and exact metric policy. The universe policy is
   `all_reviewed_symbols_every_session`. It requires at least 25 symbols;
   actual windows must also meet the 2,000-row calibration minimum.
2. A byte-bound `jbravo.market-session-calendar.v1` JSON record with `venue=XNYS`
   and ordered `sessions`: `session`, `open_utc`, `close_utc`. It covers both
   windows and five subsequent trading sessions. The independent reviewer must
   establish completeness, holidays, early closes and provider authority. A
   supplied calendar's JSON validity alone cannot prove those facts.
3. Full v2 capture packets copied without changing bytes to
   `captures/<capture-id>/{manifest.json,features.csv,predictions.csv}`. The
   reviewed manifest list must represent the complete planned session cohort.
   There must be exactly one reviewed actual production observation per session,
   containing exactly the preregistered symbols. The reviewer must establish
   that these observations correspond to the production prediction publication
   and selection, and that alternative observations were not selected using
   subsequent outcomes. Multiple observations for one session are rejected,
   never silently deduplicated. Do not claim a complete cohort when its canonical
   publication history or capture inventory is incomplete.
4. An immutable daily-bars CSV with `symbol,timestamp,close`, split-adjusted on
   a consistently reviewed basis. `bars_adjustment=split` is an independent
   review claim, not an automatic inference from a CSV name or positive prices.
   Dividend assumptions, revisions and corporate-action treatment require
   review. The complete saved bytes, `bars_observed_at_utc` and
   `outcome_cutoff_utc` are bound. No bar may represent a session that had not
   closed when this snapshot was observed.
5. A later `jbravo.production-evidence-review.v1` record binding that plan,
   captures and bars. Its approval occurs after the bars observation and before
   evaluation. `attestations` must explicitly contain true values for
   `model_deployment_identity_reviewed`, `causal_capture_reviewed`,
   `calendar_completeness_reviewed`, `population_comparability_reviewed` and
   `split_adjusted_outcomes_reviewed`. These are claims by the independent
   reviewer. The consumer cannot independently authenticate their identity or
   treat an agent-created template as their authorization. Bind the review's
   exact SHA externally when authorizing evaluation.

The plan's `window_lengths` object must contain exactly
`baseline_calendar_days` and `recent_calendar_days`, each a positive integer.
Each length must equal its window's inclusive start/end date span. Both windows
must remain disjoint and contain exchange sessions. These lengths are fixed by
the independently approved, byte-bound plan before scoring begins; there is no
CLI override or implicit fallback. Changing them requires a new reviewed plan,
which must still be approved before its first baseline trading open. The report
records the selected lengths, and the health guard checks them against its dates.

The existing monitor CLI defaults of 252 baseline and 63 recent calendar days
are configurable defaults, not a qualification minimum. This consumer does not
require 315 calendar days. Reviewers must justify the planned windows using
population comparability, session coverage and dependence between observations.
Meeting a raw row count alone does not establish statistical adequacy. Shorter
reviewed lengths do not waive complete capture, outcome or freshness checks and
do not guarantee healthy results or operating acceptance.

The module's `POLICY` remains the exact JSON metric policy for the plan:
at least 2,000 rows per calibration window, ten PSI/calibration bins, PSI warn/alert 0.10/0.25,
ECE warn/alert 0.05/0.10, top 25, cost proxy 5 bps, five-session forward horizon,
five-calendar-day strategy rebalance and Sharpe warn/alert 0.30/0.00. Overlapping
windows, lengths inconsistent with reviewed dates or changed thresholds fail.

## Causal outcome join

The score observation must follow its signal session close and precede the very
next trading open. This excludes retrospective rescoring even if the selected
model fingerprint matches. The feature and score symbol/timestamp identities
must match exactly, with finite inputs and probabilities in [0,1]. V2 captures
bind the producer files before and after scoring; a changed producer fails
before prediction publication. Legacy v1 observations remain preserved and
unqualified; their missing source binding is never retrofitted.

For each captured row, the join requires the exact entry bar timestamp and
every daily bar across the signal session and five following exchange sessions.
The outcome session close must be at or before the cutoff and after the score
observation. It computes `fwd_ret_5d = exit_close / entry_close - 1` and the
300-bp label from those closes. It never trusts a previously stored binary zero
or a finite supplied forward-return column as proof of maturity. Missing daily
bars never shift the horizon to a later available row. Captured feature values
remain intact even if a feature is named `close`.

Missing captures, missing outcomes, insufficient populations and mature input
older than seven days leave the candidate in investigation. Every planned
symbol/session must mature; no survivor-only subset silently qualifies. Holiday
or missing-data gaps can therefore keep the unchanged freshness gate blocked.

## Health and acceptance boundaries

Qualified baseline and recent populations use the exact same bundle and producer
identities. Score PSI uses fixed [0,1] bins; continuous feature PSI uses only
baseline quantiles. Constant baseline features use a disclosed three-bin
partition; constant-score shifts cannot disappear into a single bin. All bin
edges/counts and the existing 1e-6 smoothing floor are reported. Binary prevalence,
calibration and strategy calculations use the existing monitor's numerical
routines, including its current cost and calendar rebalance conventions.
Authentication loading in those modules now occurs at their actual monitor and
strategy execution entry points, so importing numerical helpers does not load
server credentials. Existing execution entry points retain environment loading.

These strategy metrics are close-to-close diagnostic proxies; observing a score
after a close does not prove that price was executable, and the cost proxy does
not qualify execution economics. This consumer provides model monitoring
evidence, not a profitable trading strategy or a trading acceptance decision.

Healthy, fully qualified evidence can produce `recommended_action=none` and
`monitor_candidate_eligible=true`. Drift or calibration/performance concerns
retain `investigate` and the underlying `diagnostic_recommended_action`. There
is no automatic production retraining/recalibration. The existing research-fold
monitor continues its current investigation contract.

Outputs are new exclusive UUID directories with `qualified_rows.csv` and
`monitor_candidate.json`, fsync and full readback. The report binds the review,
plan, calendar, bars, capture manifests and final joined rows. It always has
`publication_approved=false` and `operating_acceptance=false`; it cannot replace
`ranker_monitor/latest.json` or database rows. The health guard rejects an
unpublished contract-3 candidate and requires its model and producer identities
to match current prediction metadata. Unknown monitor contracts also warn.

Production publication needs a separately scoped, exact-artifact approval and
approved publisher. This increment does not register such a new MCP publisher,
create a schedule, qualify an absent cohort or authorize a primary run. After
approved publication and current-model verification, the unchanged primary/
postflight source, freshness, invocation and completion checks must succeed.

## Bounded command and verification

The review is a relative path inside an existing packet root, and the output
root must already exist. Sizes are at most 64 MiB per input, 128 MiB per output and 256 MiB
across consumed inputs. No boolean `--accept`, policy override or model path is
available. A healthy candidate exits 0, an investigated candidate exits 1 and
rejected evidence exits 2; none of these exits grants operating acceptance.

```text
python -B -m scripts.production_evidence --packet-root <reviewed-packet> --review-path review.json --review-bytes <approved-length> --review-sha256 <approved-sha256> --output-root <existing-isolated-output>
python -B -m unittest tests.test_production_evidence tests.test_acceptance_evidence tests.test_ml_health_provenance tests.test_pipeline_postflight tests.test_executor_preflight_health tests.test_ranker_monitor_contract
```

Tests use only synthetic temporary inputs and forbid network/DB/process work.
The full synthetic packet demonstrates candidate readiness and preservation of
publication/acceptance gates. It is not real score history or independent
production evidence. Postflight source bindings include this consumer and the
existing numerical monitor/strategy dependencies.
