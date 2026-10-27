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

The installed producer task 1558334 calls an immutable October 25 launcher with
exact pilot source/configuration pins. Editing repository files alone would
make that launcher refuse the next run. Deployment therefore needs a new
source-pinned launcher and a verified replacement of only that existing task's
command, retaining its enabled state and 01:20 UTC schedule. Keep the old
launcher, claims and session folders unchanged. Refresh the monitor process
once for its held-symbol feature adapter and refresh the existing web app once
for the coverage API. Do not replay an old deployment or bootstrap today's
expired signal session. The next completed scheduled dataset must create a
fresh 56-source primary handoff before new entries can pass their source guard.

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
receive explicit data exclusions. Malformed data aborts the capture. These
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
