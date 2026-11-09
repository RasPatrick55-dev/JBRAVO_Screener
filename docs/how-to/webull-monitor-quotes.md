# Webull monitor quote adapter: prepared, not activated

This optional adapter supplies Webull Nasdaq Basic depth quotes to the existing
Alpaca **paper** monitor. It does not change the execution broker, strategy,
ownership, risk, stop maintenance, entry timing, or authoritative PostgreSQL
records. Importing the module does nothing external.

## Evidence and limits as of November 9, 2026

- A production Webull MCP read at 14:55 UTC returned ARDT and SPY positive bid/ask
  books with provider Quote Time values. This is regular-session evidence only.
- The observed SPY quote time was `1794236124260`, Unix milliseconds, with bid
  747.23 x 21 and ask 747.26 x 95. The local fixture reconstructs the SDK's documented
  dictionary shape from those formatted observations; it is **not** a captured
  native JSON response.
- The installed desktop Webull MCP 1.2.6 formatter documents `symbol`,
  `quote_time`, `bids` and `asks`, each side containing `price` and `size`.
  Its SDK 3.0.4 uses GET `/market-data/stocks/depths/list`, version v3.
- Nasdaq Basic is a single-exchange product, not consolidated NBBO. The adapter
  records its venue scope from the configured subscription contract, not from an
  exchange field independently observed in each response.
- Early premarket 04:00-08:00 Eastern and actual PythonAnywhere operation are
  **unverified**. A subscription alone does not establish those capabilities.
- A fresh PythonAnywhere package-directory inventory showed no Webull-named
  package in the Python 3.10 `jbravo-env`. Credential presence was not inspected.
  The existing monitor remains configured for its previously deployed quote path.

## Adapter contract

`scripts.strategy_webull_quotes` accepts only structured depth-1 books for the
requested US equity symbol. It never parses formatted MCP text or substitutes a
snapshot, last trade, receipt timestamp, cached quote, SIP or IEX fallback.

Both sides must have finite positive price and size, with bid no greater than
ask. Provider time must be explicit Unix milliseconds. The existing maximum age
of 60 seconds and maximum future tolerance of 1 second are unchanged. Receipt
time is recorded separately. Earlier results are checked again at batch end.
Prices must also yield a finite positive cent-rounded exit under the unchanged
native calculation. This prevents rejecting an unusable price only after the
native lifecycle has already canceled a protective stop.

At most four distinct symbols and 96 calls per provider instance are permitted.
There are no request retries. Each symbol succeeds or fails independently; a
failed symbol is omitted and a fixed sanitized reason is retained. Missing data
cannot become a zero price or an executable quote.

The explicit `execution_components` factory verifies the existing native method
pins before deriving local classes. Only quote acquisition and quote-error
metadata differ. All broker order methods and engine trading, reconciliation,
risk and ownership methods remain inherited. Provider setup is deferred to quote
acquisition so data failure cannot disable existing order/stop reconciliation.

`DepthTransport` takes an **already initialized**, approved SDK market-data
interface. It calls only `get_quotes(symbol=..., category='US_STOCK', depth=1,
overnight_required=False)`. It does not construct SDK clients or read credentials.
SDK `DataClient` initialization can authenticate and persist tokens, so it must
not be used as an apparently read-only installation or import check.
The SDK response size check bounds accepted content after buffering. It is not
a network memory bound; native transport qualification must address that limit.

## Required before host activation

1. Verify the user's actual Nasdaq Basic Non Display entitlement and its terms
   cover this unattended use. The `non_display_authorized=True` runtime gate is
   not evidence of a license or permission to configure credentials.
2. Obtain per-action permission for persistent Webull credentials/token storage
   on PythonAnywhere. Prefer owner-hidden entry into a dedicated private path.
   Never move desktop encrypted credential files or paste values into chat.
3. Prepare a fixed sanitized host diagnostic returning only configured booleans,
   package versions and quote fields. Use the shared PythonAnywhere MCP operation
   contract, with a fresh reviewed plan; do not use generic console/API workarounds.
4. Qualify a reviewed pinned official SDK under Python 3.10. Bind production US
   endpoints, TLS verification, bounded connect/read timeouts, no automatic retries,
   no debug/header/body logging, and market-data-only interface access. Confirm
   exact native response shape and token lifecycle without order permissions.
5. Build a new source-pinned monitor bootstrap, retaining the current ownership,
   pending-stop and quote-safe wrapper order. Bind new adapter/transport hashes in
   its manifest and report its provider/venue qualification. Do not edit protected
   native lifecycle source to switch the quote feed.
6. Independently review the exact implementation and fresh rollout plan, then
   obtain concrete deployment approval. Coordinate with web-only deployment work.
   Preserve existing schedules and paper account identity. If stop/resume is
   required, retain the existing singleton and fresh broker-protection gates.
7. Verify native read-only quotes and two advancing ordinary monitor cycles.
   Separately observe fresh positive depth at 04:00 Eastern before claiming
   early-session coverage. Do not manually run entries or place test orders.

This adapter alone does not repair SIP after-hours reference capture or producer
data entitlements. The nightly pipeline, source-bound selection, freshness,
monitor health, and entry prerequisites still decide whether paper entry runs.
Do not relabel a blocked or empty result as end-to-end success.

## Offline checks

Run `python -m pytest -q tests/test_strategy_webull_quotes.py` and the existing
quote diagnostics, broker, engine, ownership and pending-stop regressions.
Run `python -m scripts.docs_consistency_check` after documentation changes.
Offline tests do not establish SDK compatibility, market entitlement, host
credentials, premarket availability or deployment success.
