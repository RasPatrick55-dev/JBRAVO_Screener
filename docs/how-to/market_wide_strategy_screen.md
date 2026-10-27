# Market-wide daily strategy screening

The v2 configuration in `config/strategy_scan_daily.json` replaces the ten-stock
pilot list with daily `GET /v2/assets?status=active&asset_class=us_equity`
discovery from the native Alpaca paper account. It accepts active, tradable U.S.
equities with a supported SIP exchange and the existing execution symbol format.
This asset class includes ETFs; there is no name-based stock/ETF guess or new
price, liquidity, market-cap or performance filter. SPY remains benchmark-only.
The old ten-symbol configuration exists only as an offline regression fixture.

These are source/configuration changes. A local passing test or merged PR does
not establish deployment or successful full-universe execution on PythonAnywhere.
Deployment must preserve the shared dashboard overlays, model, database history
and paper execution rules. No schema migration is needed.

Producer task 1558334 was switched from its immutable pilot launcher to the new
15-source-pinned market-wide worker; the final repair005 activation was verified
October 27. Its enabled state and 01:20 UTC schedule were preserved. The worker
is `/home/RasPatrick/jbravo-research-tools/MARKET_HANDOFF_REPAIR_20261027_005/market_daily_boot.py`;
the primary still consumes the original enrolled `runs` root. Keep old launchers,
claims and session folders unchanged. Editing repository sources alone can
invalidate immutable worker pins: future runtime changes require a new reviewed
launcher/activation plan rather than replaying this completed deployment.

The monitor was restarted and independently observed healthy. The coverage API
was observed serving its new field after one web-reload attempt whose transport
outcome was unknown; that attempt was not repeated. Do not repeat restarts or
reloads merely by following this historical record. The next completed scheduled
dataset must create a fresh 56-source primary handoff before new entries can
pass their source guard. Do not bootstrap an expired signal session.

## Collection and coverage

The worker captures the whole returned asset population. The 30,000-asset bound
is an error ceiling, never a truncation rule. Each discovered asset receives a
decision: screened, benchmark-only, or explicitly excluded with its reason.
Unsupported symbol formats, nontradable assets and unsupported SIP exchanges
are accounted for. There is no fallback to the old ten stocks.

Raw and split-adjusted SIP/USD daily bars are requested in batches of at most
25 symbols. All page tokens are consumed and checked for loops, duplicate bars
and population changes. Corporate actions are paginated for the same population.
Requests are paced at least 0.35 seconds apart, with explicit request, byte and
time budgets. HTTP/authentication/rate-limit failures abort the session without
archiving a partial successful screen or retrying a consumed daily run.

Immutable compressed chunks store each evaluable symbol's raw/split history;
small per-symbol descriptors bind their broker identity and actions. Identical
historical chunks are reused through verified content-addressed hardlinks, so
each capture has its own contained file without copying the same historical
bytes every night. Provider responses are compressed as well.
`capture/analysis/manifest.json` binds every descriptor and chunk,
the raw asset response, supplied calendar and complete `coverage.json` by hash
and size. Full coverage is verified again by the primary pipeline. The compact
coverage counts, universe hash and full coverage-file hash travel with the
candidate batch, Backtest, Metrics and dashboard API even when there are no buys.

Insufficient 250-session history, missing current or historical sessions,
zero-volume history, changing split factors and unsupported corporate actions
receive explicit data exclusions. Cash dividends with due-bill settlement also
exclude the affected symbol until a due-bill accounting adapter is available;
their payment-before-ex-date ordering can be legitimate. Invalid action dates,
dividend rates, payment dates or ex-sessions produce explicit symbol exclusions.
Duplicate action identities and malformed provider envelopes still abort the
capture. Reorganization recipients in `stock_movements` count as affected
symbols alongside top-level participants; unsupported actions exclude those
symbols instead of being mistaken for unrelated provider records. These
exclusions reflect current data/kernel capability; they are not strategy
rejections. A run with no evaluable trading symbols fails. A valid scan with
evaluable symbols and no qualifying setups remains an honest no-buy result.
Newer listings can begin evaluation once the existing 250-session feature
warmup is ready; they need not have been listed in January 2023.

## Cache and replay

The disposable `market-cache` is keyed by broker asset ID, symbol, history start,
feed and currency. It is separate from immutable session captures and is never
an authoritative candidate source. Recent cached history uses a fresh 20-session
overlap; a changed overlap forces a full-history batch refresh. Invalid, expired
or differently scoped cache files cause a full fetch. The manifest explicitly
labels cached history: revisions outside the overlap may require a full refresh.
Remove the disposable cache only under a separately scoped storage operation.

The same strategy preparation examines every evaluable symbol. Portfolio replay
retains all qualifying rows and enough following sessions for every possible
held position and exit. Rows that cannot affect the portfolio are omitted from
the historical trace to bound memory. Complete current decisions remain in
`latest_observations.json.gz`; compressed historical traces describe their
sparse storage explicitly. The HTML report previews 500 rows per table.

The same sparse preparation and unchanged portfolio kernel run in Backtest.
Ranking competition, qualifying-close consumption before capacity, rearming,
cash allocation, exits and cost rules remain shared across the full captured
universe. There is no independent per-batch portfolio. Asset additions/removals
and operational batch-size changes do not reset logical signal identities;
membership and data changes remain bound in the scan/run identity.

Current handoff ranking uses the qualifying pool before simulated portfolio
capacity. A historical full book cannot erase today's valid buy proposals.
The `top_n` limit still applies, and the actual paper executor enforces account
cash and position limits. Historical fills, rearming and portfolio metrics stay
unchanged; current proposals remain unfilled with unknown future outcomes.

The monitor prepares completed-bar features for held and proposed symbols,
including held symbols absent from today's buy list. If a held symbol no longer
has valid completed data, existing protections are retained and the unavailable
input is reported; no replacement bars are fabricated.

## Timing and acceptance

Keep the existing 01:20 UTC producer and 03:45 UTC primary schedule. The producer
still requires the completed signal session after 20:15 Eastern. Capture is
bounded by 90 minutes, scanning by 60 minutes and the whole producer by two hours,
leaving 25 minutes before the primary's normal start. Budget exhaustion fails
closed; it never publishes a subset as full coverage. The first uncached live
run must demonstrate actual runtime, disk capacity, provider entitlement and
complete coverage before broad-market operation can be called verified.

Historical replay uses today's active/tradable universe and therefore has
survivorship bias. It is not a historical point-in-time universe or untouched
validation. Regular-session-open replay still does not establish premarket
performance. Paper order entry remains the separately authorized 04:00 Eastern
limit-order route; sizing, position caps and live-trading restrictions are unchanged.

Provider references: [assets](https://docs.alpaca.markets/us/reference/get-v2-assets-1)
and [historical bars and pagination](https://docs.alpaca.markets/us/reference/stockbars).

## Verified deployment and audit, October 27, 2026

PRs [757](https://github.com/RasPatrick55-dev/JBRAVO_Screener/pull/757),
[758](https://github.com/RasPatrick55-dev/JBRAVO_Screener/pull/758),
[759](https://github.com/RasPatrick55-dev/JBRAVO_Screener/pull/759) and
[760](https://github.com/RasPatrick55-dev/JBRAVO_Screener/pull/760) supplied the
market-wide route, affected-symbol action exclusions and the current-proposal
capacity repair. Native source was verified at
`9b428a91892110c593aadb5c44f0ffb9c5a9c8b0`, with 56 runtime bindings and
78 preserved-file guards. The final handoff repair passed 217 focused tests.
The [structured verification record](../reference/pullback_reclaim_verification_2026-10-27.json)
contains the exact run IDs, hashes, task settings and evidence locations.

The first complete acquisition made 2,840 native Alpaca GETs and retained
1,953,678,560 response bytes. Of 14,407 discovered assets, 6,147 were evaluated,
8,259 received explicit exclusions, and SPY was benchmark-only. This is complete
accounting of the returned population, not evaluation of every discovered asset.
Acquisition took about 58 minutes; each full-cohort Screener/Backtest computation
took about 20 minutes in the observed environment. These measurements do not
guarantee future runtime or provider capacity.

Real-data validation exposed three problems that are now repaired: legitimate
due-bill dividend ordering could abort the capture; nested corporate-action
recipients could be mistaken for unrelated records; and a full simulated
historical book suppressed otherwise valid current proposals. Unsupported
actions now receive explicit affected-symbol exclusions, while current ranking
uses the original qualifying pool before simulated account capacity. Historical
fills, cash, exits and rearming remain unchanged.

The final supervised manual005 audit completed at 19:26:56 UTC. It fully
reverified and reused the immutable capture with zero new provider GETs. The
actual Screener and primary validation adapter passed 12 qualifying setups
through the existing top-three limit: TXN, DK and FOCT. Actual Backtest and
Metrics functions completed with the same strategy and candidate identities.
One research audit row was committed and independently read back.

This was an **isolated input-root audit**. It did not invoke the scheduled primary
task, publish canonical production candidates or submit orders. The October 26
signal's expired entry window was correctly refused. All three current symbols
had zero completed historical trades in the shared cohort; their statistics
remain unknown. The cohort overall profit factor was about 0.9224 and terminal
equity USD 98,799.05 from USD 100,000. A successful technical path is not a
profitability, promotion or premarket-fill qualification.

At the final readiness observation, the production report still represented
the earlier empty ten-symbol pilot with 54 bindings. Its 07:55 UTC executor run
returned zero with `NO_STRATEGY_BUY_SIGNALS`. The monitor was healthy and flat,
but `entry_ready=false` after deployment. The next fresh producer/primary run,
canonical candidate, actual paper fill and monitored lifecycle remained pending.
Recheck current receipts rather than treating this dated snapshot as live state.
