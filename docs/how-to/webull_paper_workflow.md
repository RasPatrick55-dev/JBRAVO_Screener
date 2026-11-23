# Webull-assisted JBravo paper tasks

[Local scheduler definitions](../../config/webull_paper_tasks.json) cover the
Producer, Primary Pipeline and Execute Trades. The monitor continues to track
fills, manage stops and exits, and cancel unfilled entries at the exchange
close. The hourly fill linkage uses Alpaca Paper GET requests and writes
verified linkage data to PostgreSQL.

| Task | UTC slot | Eastern slot | Work |
| --- | --- | --- | --- |
| Producer | 01:20 | 20:20 EST or 21:20 EDT on the previous UTC date | Scan with Alpaca IEX, collect finalist bars from Alpaca first, then use Webull for gaps before committing the archive. |
| Primary Pipeline | 03:45 | 22:45 EST or 23:45 EDT on the previous UTC date | Resolve the signal and target sessions, verify the exact archive, and publish the final selection and paper limits. |
| Execute Trades | 07:55 | 02:55 EST or 03:55 EDT | Verify the Primary and wait until 04:00 ET. Check Alpaca IEX first for each selected symbol, then use Webull if the quote is missing, invalid, or stale. |

The Producer starts after JBravo's 20:15 ET readiness cutoff. On the 2026
early closes, November 27 and December 24, the exchange closes at 13:00 ET
and Webull after-hours data ends at 17:00 ET. The Primary can use that day's
exact archive for the next trading session. An unlisted early close requires
a verified cutoff in the producer configuration. A missing archive blocks
the Primary for that session.
[Webull early-close calendar](https://www.webull.com/help/faq/11110-Market-Closures).

## Screening and paper entry

The Producer checks raw one-minute after-hours bars and the final fifteen
minutes of the regular session for up to twenty finalists. It keeps each usable
Alpaca IEX series and requests Webull only for that symbol's missing or unusable
window. The archive records the provider of each series and comparison price.
The Primary ranks finalists
with positive price change ahead of the other finalists with usable bars,
then preserves the existing technical order within each group. It records
coverage, the last regular minute, the after-hours return, estimated dollar
volume and the age of the last after-hours bar. Missing data and after-hours
bars older than thirty minutes at the cutoff hold the symbol out of selection.
These ranking rules are a local paper policy with performance still to be measured.

Each of up to three selected candidates receives the last after-hours close
plus 0.5%, rounded down to the cent. During the first premarket minute, Execute
Trades checks the provider timestamp and depth-one bid/ask for each eligible
selected symbol. It holds a symbol if the quote is over fifteen seconds old
when received, the spread exceeds 1% of the midpoint, or the ask exceeds the
published limit. Webull can supply a fresh quote when Alpaca's is unavailable.
A usable Alpaca quote that fails the spread or price check holds the symbol.
An admitted symbol may submit an Alpaca Paper day limit with
extended hours enabled at the published price, subject to the existing
account, stop coverage, eligibility, cash and capacity guards.
[Alpaca order model](https://docs.alpaca.markets/us/docs/orders-at-alpaca).

## Hosted qualification

The prepared workflow uses Alpaca IEX first, with the stored Webull sandbox
application for gaps. Alpaca Paper handles simulated orders. The hosted workflow stays
on its Alpaca route until the Webull rollout is approved. On November 23,
PythonAnywhere verified regular-session references, after-hours bars, and
first-minute premarket quotes for AAPL and MSFT. The early-close check remains
pending. Entry input inspection uses the entry command without the placement
flag.

```bash
bash bin/run_webull_paper_task.sh probe after-hours --session 2026-11-20
bash bin/run_webull_paper_task.sh probe premarket --session 2026-11-24
bash bin/run_webull_paper_task.sh probe after-hours --session 2026-11-27
bash bin/run_webull_paper_task.sh entry
```

The premarket probe must run within 04:00â€“04:01 ET on its session date. For
the early close, confirm the date and regular close against the exchange
calendar before testing Webull's published cutoff. Check the environment,
per-symbol status and provider timestamps in each receipt. The bars probe
uses HTTP POST on Webull's historical data endpoint. The quote probe uses
HTTP GET on the depth endpoint.
[Webull market data API](https://developer.webull.com/apis/docs/market-data-api/data-api/).

The sandbox launcher reads the existing private
`~/.config/jbravo/webull-market-data/app.json` directly. Set
`JBR_WEBULL_DATA_ENV=sandbox` and `JBR_WEBULL_NON_DISPLAY_AUTHORIZED=true` in the
host configuration when the app has non-display permission. The environment
must match `webull_data_environment` in the source-bound producer configuration.
Historical requests use exact minute boundaries and archive the environment.
The first-minute quote still has to meet the 15-second freshness requirement.

The Webull fallback requires `JBR_WEBULL_HOSTED_QUALIFIED=true`. Set the flag
after the hosted cutoff check passes and the rollout is approved. An unavailable
Webull client leaves Alpaca IEX usable and holds symbols with unresolved gaps.
Coordinate the pinned Producer, Primary,
entry, and Monitor sources before switching the hosted workflow. The changed
producer configuration creates a new archive revision. The first Primary needs
a verified archive from that revision, including when the signal session
precedes a holiday.
