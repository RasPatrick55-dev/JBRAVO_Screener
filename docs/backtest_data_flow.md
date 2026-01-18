# Backtest Data Flow

## Overview
`scripts/backtest.py` runs the JBRAVO strategy backtest as part of the pipeline,
consuming the latest screened candidates and cached daily bars to generate
per-symbol performance metrics for downstream analytics and dashboards.

## Inputs
- **screener_candidates (PostgreSQL)**: latest run_date used to select symbols.
- **daily_bars (PostgreSQL)**: historical OHLCV, minimum 750 bars enforced.

## Process
- Uses a custom backtester (Backtrader-style) to simulate the JBRAVO strategy.
- **Entry rules**: close > SMA9, RSI > 50, MA alignment.
- **Exit rules**: RSI > 70 or close < EMA20.
- Runs per symbol; invalid symbols or short-history symbols are skipped.

## Outputs
- **CSV**: `data/backtest_results.csv`
- **DB**: inserts into `backtest_results` with full per-symbol metrics.

## Logging
- Per-symbol fetch status:
  - `[INFO] BARS_LOADED_FROM_DB symbol=... rows=...`
  - `[WARN] BAR_CACHE_MISS symbol=... reason=not_enough_bars`
- Audit summary: `reports/backtest_db_bars_audit.txt`

## Validation
- Symbols processed match latest `screener_candidates` with >=750 bars.
- No fallback or API fetches during the run (DB-only bars).

## Compatibility
- Output is ready for `scripts/metrics.py` and dashboard ingestion.
- CSV and DB contain identical content from the same summary DataFrame.
