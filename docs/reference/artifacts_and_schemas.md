# Artifacts And Schemas

## Source Of Truth Policy

- PostgreSQL is authoritative for pipeline state, candidates, backtest output, and dashboard reads.
- CSV artifacts are non-authoritative debug exports and parachute outputs.
- Default candidate reads should use `latest_screener_candidates` and `latest_top_candidates`.

## Canonical Candidate CSV Header

```text
timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown,entry_price,adv20,atrp,source,sma9,ema20,sma180,rsi14,passed_gates,gate_fail_reason
```

## DB-First Data Sources

- `screener_candidates`: scored screener candidates for each run.
- `top_candidates`: narrowed candidate set for downstream execution.
- `pipeline_runs`: pipeline status and summary payloads keyed by run dates/timestamps.
- `backtest_results`: backtest output used by reporting and health views.
- `metrics_daily`: aggregated daily metrics.
- `order_events` and `trades`: execution and trade lifecycle history.
- `latest_screener_candidates` / `latest_top_candidates`: canonical read views for current run state.

## CSV Artifacts (Debug/Parachute)

- `data/latest_candidates.csv`: optional export only; not authoritative.
- `data/top_candidates.csv`: optional export only; not authoritative.
- `data/scored_candidates.csv`: optional export/debug artifact.
- `data/metrics_summary.csv`: summary export for reporting compatibility.
- `data/daily_bars.csv`: Stage 0 ML export when requested.

## Required Pipeline Log Tokens

- `PIPELINE_START`
- `PIPELINE_SUMMARY`
- `FALLBACK_CHECK`
- `PIPELINE_END`
- `DASH RELOAD`

## Common Execution Tokens

- `BUY_SUBMIT`
- `BUY_FILL`
- `BUY_CANCELLED`
- `TRAIL_SUBMIT`
- `TRAIL_CONFIRMED`
- `TIME_WINDOW`
- `CASH`
- `ZERO_QTY`
- `PRICE_BOUNDS`
- `MAX_POSITIONS`

## Artifact Precedence

1. DB views/tables
2. JSON reports generated from DB-backed runs
3. CSV exports for debug or first-run bootstrapping only

## DB Config Precedence

1. `DATABASE_URL` (preferred)
2. `DB_HOST` + `DB_PORT` + `DB_NAME` + `DB_USER` (`DB_PASSWORD` optional)
3. Otherwise DB is disabled with an actionable warning

Dev-only local fallback to `localhost:9999` is available only when `JBR_DEV_DB_DEFAULTS=true`.

## Docs-Check Allow Labels

Use inline allow directives only for intentional exceptions, on the same line or the line immediately before the text:

```html
<!-- docs-check: allow-live_trading_supports_claim -->
```

Available labels:

- `live_alpaca_endpoint`
- `live_trading_enabled_claim`
- `live_trading_supports_claim`
- `production_trading_enabled_claim`
- `csv_source_of_truth_claim`
- `latest_candidates_source_of_truth_claim`
- `latest_candidates_authoritative_claim`
