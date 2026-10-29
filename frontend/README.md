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
