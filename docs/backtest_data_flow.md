# Backtest Data Flow

## Overview
`scripts/backtest.py` runs the JBRAVO strategy backtest as part of the pipeline,
consuming the latest screened candidates and daily bars to generate per-symbol
performance metrics for downstream analytics and dashboards.

## Inputs
- **screener_candidates (PostgreSQL)**: latest run_date used to select symbols.
- **daily_bars (PostgreSQL)**: historical OHLCV; history depth is controlled by
  `BACKTEST_LOOKBACK_DAYS` and `BACKTEST_MIN_HISTORY_BARS`.
- **Alpaca Market Data API**: IEX primary with optional SIP fallback for backfill.

## Process
- Loads bars up to the run_date; if history is short, backfills from the API and
  upserts into `daily_bars`.
- Uses a custom backtester (Backtrader-style) to simulate the JBRAVO strategy.
- **Entry rules**: close > SMA9, RSI > 50, MA alignment.
- **Exit rules**: RSI > 70 or close < EMA20.
- Runs per symbol; invalid symbols or short-history symbols are skipped.

## Outputs
- **DB**: inserts into `backtest_results` with full per-symbol metrics.
- **CSV**: optional and non-authoritative; CSV export is disabled when DB is enabled.

## Logging
- Per-symbol fetch status:
  - `[INFO] BARS_LOADED_FROM_DB symbol=... rows=...`
  - `[INFO] BACKTEST_BARS_BACKFILL symbol=... rows=...`
  - `[WARN] BAR_CACHE_MISS symbol=... reason=<missing_bars|insufficient_bars>`

## Validation
- Symbols processed match latest `screener_candidates` with >= `BACKTEST_MIN_HISTORY_BARS` bars.
- API backfill grows `daily_bars` when new symbols appear.

## Compatibility
- Output is ready for `scripts/metrics.py` and dashboard ingestion.
- DB is the source of truth; CSV exports are for ad-hoc debugging only.
