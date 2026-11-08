# Read-only strategy scorecard

`GET /api/strategies/scorecard` is an additive public dashboard endpoint. It does
not replace the candidate Backtest/Metrics APIs, alter the scanner, run a replay,
change a task, publish to PostgreSQL, or contact a broker. The route passes its
existing native `BASE_DIR` explicitly to `dashboards.strategy_scorecard`; an
isolated web package must not derive native data paths from its own location.

## Evidence and coverage

The authoritative catalog is `jbravo_research.research_runs`. A read-only,
repeatable-read transaction selects the latest **completed** daily archive for
each currently configured strategy ID, displaying that snapshot's exact version,
definition fingerprint and episode ID. Historical versions are not additional
rows. Up to 24 configured strategies are supported; truncation and rejected
archives are explicit. The latest
record in a group is not silently replaced with an older valid record after a
validation or provider failure. Configured strategies without usable archive
metadata remain visible with unknown version/definition and unknown metrics.

These are retained historical archives. `as_of` and `freshness.evidence_as_of`
are the actual source session, not the HTTP fetch time. Newer scheduled attempts
are **not evaluated** by this endpoint; `latest_attempt_status=NOT_EVALUATED`
and `latest_attempt_at=null` make that limitation explicit. A completed old
archive does not establish that today's scanner succeeded.

An archive contains summary/handoff metadata, not full position outcomes. Its
summary's `exit_fill_win_rate_pct` and profit factor are per-exit-fill metrics.
Candidate Backtest reports cover only the current candidate symbols. Neither
can supply strategy-wide whole-position success rates.

Full outcomes come only from the PythonAnywhere files API, using the existing
web environment's token/username aliases. Local outcome files, CSV exports,
broker equity changes and per-pick averages are never fallbacks. Each group
requires three bounded GETs: manifest, specification, and lifecycle JSON (plain
or gzip). The reader checks:

1. PostgreSQL result hash, deterministic daily archive identity, non-execution
   flags, scan/handoff identity and historical strategy-definition fingerprint.
2. Exact output path under the configured producer root and signal session;
   companions must occupy their named companion directory.
3. The manifest hash stored in PostgreSQL, unique safe relative bindings, and
   the producer's exact saved summary/handoff bytes against their manifest pins.
4. Specification bytes against both their manifest pin and the PostgreSQL
   specification hash; reconstructed episode and scan identities, including
   universe policy, portfolio/cost case and source/dataset fingerprints.
5. Lifecycle byte length/hash, bounded decompression, declared symbols/window,
   stable signal identities, strategy/version lineage, chronological fills,
   quantities, cash changes and fully closed-position net/cost reconciliation.

The lifecycle covers historical positions selected by the declared simulated
portfolio and terminal proposals. It does not cover every qualifying or rejected
setup as an independent hypothetical trade. Market-wide scans bind changing
daily universe membership separately from their stable episode policy. The
response exposes the declared universe kind/count, coverage exclusions, portfolio
and modeled cost case. Separate strategies or incompatible episodes are never
added into a combined portfolio.

## Statistics

A whole position is one completed observation only after all bought shares have
been sold. Partial exits do not add trades or independently increment wins.
Open positions and unfilled proposals are counted separately and excluded from
completed statistics. Closed net P&L includes modeled entry/exit fees and
slippage once, excluding separately recorded cash dividends.

- Win rate: positive net closed signals / all closed signals, including breakeven.
- Net expectancy: sum of closed net P&L / completed signal count, in USD.
- Payoff ratio: mean positive outcome / absolute mean negative outcome.
- Profit factor: total positive outcomes / absolute total negative outcomes.
- No-loss profit factor is undefined (`null`), never infinity. No completed
  samples have verified zero counts but unknown rates/expectancy/payoff.
- Missing, conflicting or unverifiable evidence has **null counts and metrics**,
  not zeros. An all-unavailable result still includes validated strategy identities.
- `costs` explicitly says `MODELED`; actual costs are not evaluated by this API.
- The 95% expectancy interval resamples whole signal-week clusters 400 times,
  using a fixed seed. Fewer than two signal weeks leaves bounds unknown. It is
  descriptive uncertainty, not untouched validation or a future success guarantee.
  Fewer than ten weeks also carries a low-cluster warning (a display caution,
  not an acceptance threshold). Week grouping does not remove cross-week holding
  or regime dependence; a larger cluster count does not establish independence.

`scripts/swing_scorecard.py` was evaluated for reuse. Its whole-position and
weekly-cluster concepts are retained, but its fill schema is different and its
module imports NumPy/pandas. This web module uses a bounded standard-library
implementation and validates the archived scanner lifecycle contract directly.
It does not import research engines, live strategy definitions, execution code,
or mutate native source guards. Repeated compatible full snapshots can be
deduplicated by stable signal ID; open fills may only grow by prefix. Conflicting
completed outcomes or disappearance of a filled signal fail closed. The live
reader selects one latest full snapshot per configured strategy, so repeated daily history
cannot be counted more than once.

`actual_paper` remains a separate unavailable/not-evaluated evidence basis. This
is not an assertion of a permanent accounting hold. Future actual realized
performance requires its own complete accounting contract and must never replace
the simulation row's basis. Every response retains `research_qualified=false`
and `order_submission_authorized=false`. There is no 75% acceptance or success
claim.

## Response contract and operating bounds

Envelope: `ok`, `status` (`COMPLETE`, `PARTIAL`, `EMPTY`, `UNAVAILABLE`), `source`,
`artifact_source`, `metric_basis`, `rows`, `rows_final`, `generated_at`,
`cache_age_seconds`, `cache_ttl_seconds`, `coverage`, `limitations`, `actual_paper`.
`PARTIAL` describes missing groups or selection limits; it is not permission to
display an unavailable row's numbers. Only `evidence_status=VERIFIED` rows supply
statistics. Per-row identifiers, `episode`, `freshness`, evidence hashes/counts,
costs and uncertainty accompany the flat statistics fields.

Rows expose `closed_signals`, `open_signals`, `unfilled_signals`, `wins`, `losses`,
`breakeven`, `win_rate_pct`, `net_pnl`, `expectancy_net_per_signal`, `average_win`,
`average_loss`, `payoff_ratio`, `profit_factor`, `profit_factor_status`,
`sample_status`, `evidence_status`, `costs`, `uncertainty` and `evidence`.

Limits are fixed, not expanded by public query parameters:

- One snapshot per configured strategy (six in the initial deployment, at most
  18 artifact GETs). There is a hard cap of 24 configured strategies / 72 GETs.
  Two concurrent readers share a one-second minimum request-start interval;
  the overall deadline may leave some groups unavailable if the catalog grows.
- Two MiB per PostgreSQL payload, 128 KiB manifest, two MiB specification,
  eight MiB compressed/plain lifecycle and 32 MiB decompressed lifecycle.
- 64 MiB request-wide provider bytes and 64 MiB expanded lifecycle bytes.
- 20,000 distinct signals, 100 fills per signal, 100,000 fills per aggregation.
- Five-second SQL statement timeout, one-second lock timeout; DB connection
  and each HTTP read use bounded timeouts. A shared 25-second work deadline
  covers DB and artifact work. A blocking read already in flight can finish up
  to five seconds later. The frontend allows 35 seconds and polls at 60 seconds.
- No HTTP redirects, proxies, retries, arbitrary origins or arbitrary file paths.
  Only standard and EU PythonAnywhere HTTPS API origins are accepted.
- A 15-minute in-process cache for verified/partial historical evidence, a
  separate 60-second all-unavailable cache, and nonblocking single-flight refresh prevent
  repeated public requests from fanning out. Stale data during refresh is explicitly
  labeled with its cache age; source session dates are never refreshed artificially.
- Database/provider exceptions are converted to fixed codes. Credentials, DSNs,
  internal file paths and arbitrary server error text are not returned.

## Verification

Run the offline contract tests without provider/database access:

```text
python -B -m pytest --noconftest -p no:cacheprovider -o addopts= tests/test_strategy_scorecard.py -q
python -m ruff check dashboards/strategy_scorecard.py tests/test_strategy_scorecard.py
python -m scripts.docs_consistency_check
```

Tests cover complete/partial/unfilled positions, duplicate overlapping histories,
revisions/conflicts, unknown and zero samples, modeled costs, deterministic
uncertainty, strict hashes/paths/JSON, gzip bounds, read-only DB cleanup,
sanitized failures, absent configured strategies, provider restrictions/cache,
additive public route wiring, explicit native-root injection and Python 3.10 syntax.
Offline fixtures are not a production database or deployment acceptance claim.
