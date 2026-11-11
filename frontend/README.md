# Frontend (React)

## Build base path

The React dashboard is served from the site root (`/`).

When building for production, ensure the base path is `/` so that
`frontend/dist/index.html` references assets under `/assets/`.

## Stock logos

`StockLogo` uses the existing Logo.dev CDN ticker lookup; it does not infer company
websites or switch logo providers. A usable logo requires an existing Logo.dev
**publishable** (`pk_`) key in `VITE_LOGO_DEV_API_KEY` when building this Vite UI.
A server-side setting changed after a build does not modify the compiled browser
bundle. Do not put a Logo.dev secret (`sk_`) API key in frontend configuration.
The legacy `REACT_APP_LOGO_DEV_API_KEY` lookup is retained for an existing runtime
integration; Vite does not automatically expose that unprefixed name from `.env`.

Missing configuration, failed/blank images and an eight-second load deadline show
an accessible ticker in the same compact tile. Failure does not retry an image or
load a placeholder file. The image uses the origin-only referrer policy required
by Logo.dev domain restrictions; no account identifiers or page paths are sent.

Logo configuration is not financial data, broker connectivity or trading health.
Use only the existing approved publishable key and domain policy. Do not create
credentials, weaken restrictions, or commit local `.env` files to repair a logo.
See [Logo.dev ticker lookups](https://www.logo.dev/docs/logo-images/ticker) and
[Logo.dev key/domain policy](https://www.logo.dev/docs/platform/api-keys).

The eight-second deadline is a display deadline: it labels the visible ticker as
delayed while preserving the same pending image request. A later successful load
with valid dimensions may display that exact image. An actual image error remains
terminal until its symbol/source is remounted; no automatic retry is introduced.

## OAuth-exported company logos

Reviewed Logo.dev API assets can be served locally without a CDN key. The exact-ticker
map in src/components/ui/stockLogoAssets.json records the provider resolution,
source etag and downloaded-byte hash. Files have immutable hash-bearing names
under public/assets/company-logos. Never persist the temporary signed download URL.

Current verified local coverage is the 24 observed ALL/Winners and ALL/Losers tickers, resolved by provider ticker lookups.
No company name/domain is inferred. Unmapped tickers use the existing approved
Logo.dev CDN configuration when present; otherwise they retain the visible ticker.
A failed local image also retains its own ticker. This does not establish coverage of other date ranges, future symbols or the full market.

The public Trades leaderboard includes the visible Logos provided by Logo.dev link.
Attribution is retained because the commercial/Community plan status was not queried.
Refresh stored files only after an authorized provider lookup and reviewed etag/hash
change; no polling, webhook subscription or credential setup is introduced.

Logo images preserve their native aspect ratios with object-contain and a light neutral backing inside the original compact tile. MOD/MUR source softness is retained without invented replacements.

## Dashboard request lifecycle

Data reads use a shared FIFO dispatcher with at most four active requests per
page. Queue time remains loading; the 45-second response-and-body deadline starts
when the request is dispatched. The strategy scorecard retains its explicit
35-second deadline and longer refresh/freshness settings. Cancellation removes
queued reads, aborts dispatched reads, clears timers, and prevents late results
from publishing after navigation or a changed query.

Polling waits a full interval after each settled read, rather than starting a
second request on a fixed timer. Existing intervals remain 20 seconds for most
panels, 15 seconds for stored connection diagnostics, and one second for the
server clock. Manual refreshes while a read is queued or active are coalesced.
ML Overview remains one-shot per mount; its diagnostic artifact ages and source
labels are not rewritten by transport success.

Execute state and audit use finite JSON reads, never EventSource streams. Initial
loading is distinct from an unavailable first read. Failed refreshes preserve
last successful samples with stale/unknown notices and their original receipts.
Changed query paths do not inherit old rows or receipts. Account endpoints publish
independently through the same source-state rules. All account reads use the
canonical same-origin `/api` routes (the local Vite proxy supplies development
routing), with no alternate-origin retries.

Transport recovery does not qualify financial results or imply current broker
protection. Held/unavailable business envelopes remain fail-closed. Validated
Execute P/L and current entry-window claims remain hidden on failed refreshes;
retained order/log observations are labeled unverified. PostgreSQL remains the
canonical pipeline/accounting source, CSV remains diagnostic, and execution
remains paper-only.

## Legacy strategy trade history endpoint

Before the combined Trades view, the tab read `/api/trades/strategy-history?limit=25` independently through
the shared request lifecycle. The separate Strategy Trade History card consumes
`jbravo.strategy-history.v1` from PostgreSQL strategy execution observations.
The standalone endpoint remains independent of the legacy ledger. The current Trades tab instead uses the combined observation described below.

Native signal state is separate from observed fill completion. Complete exits,
partial exits, entries without an exit, observed unfilled entries and incomplete
evidence remain distinct. Gross realized P/L requires the server's complete
FULL_EXIT_OBSERVED classification and native CLOSED state. Fees and net remain
unknown. Stored ownership labels are historical and do not establish current
position-management authority. Response receipts and source record times are
separate; polling is not a live fill or position read.

The endpoint returns the latest bounded signals without pagination. The card
shows the `has_more` limitation and reports legacy order-ID overlap exclusions.
Unavailable or malformed responses do not become empty results or zero P/L.
Failed refreshes retain the original sample with a visible stale warning.

For local visual QA without API/database/broker calls, run Vite and open
`/qa/strategyHistoryPreview.html`. This is a standalone synthetic fixture page,
not a production route or source of trading facts. See
`qa/strategy-history-review.md` for the tested scenarios and limits.
## Combined Trades observation

The production Trades tab now reads only `/api/trades/combined-gross`. Python
owns the complete bounded population, durable-order deduplication, exact gross
arithmetic, lifecycle eligibility, UTC range membership and symbol ranking.
The browser displays one `jbravo.trades-overview.v1` observation across the
history, all five summaries and both ranking modes. Range/mode changes and
25-row list pagination reuse that observation; they do not refetch aggregates.

Legacy and strategy account provenance remain distinct. Gross is before fees,
not qualified net or portfolio return. Original Recorded P/L is retained in
legacy row details. Strategy lifecycle rows expose the backend's combined
inclusion/reason and keep the recent25 preview limit explicit. That preview's
`has_more` does not truncate the separately complete realized candidate cohort.

Malformed or duplicate row identities, malformed aggregates, incomplete
population and invalid source contracts fail the entire combined presentation
closed instead of dropping rows while keeping their totals. Null amounts are
Unknown; a genuine zero remains zero. Transport failures and the exact source_busy/source_timeout unavailable envelopes
from broker_account_read/database_cohort_read retain one whole previous sample
with its original receipt and a stale warning. Policy, identity, cohort-change,
overflow and unknown business failures do not retain financial results. No source or
aggregate is combined from different reads. Query changes and unmount cancel
obsolete reads through the shared dispatcher.

The old `/api/trades/overview` qualified-performance contract and Account equity
charts are unchanged. Historical multi-endpoint regression tests use the frozen
QA-only `LegacyTradesTab`; new combined tests exercise the actual production tab.
