# Changelog

## [2026-01-18] DB-first pipeline updates
- Aligned screener bars window with `--run-date` to ensure API windows match the run metadata.
- Added DB views `latest_screener_candidates` and `latest_top_candidates` for final outputs.
- Disabled candidate CSV exports by default when DB is enabled; gate with `JBR_WRITE_CANDIDATE_CSVS=true`.
- Expanded backtest coverage with API backfill into `daily_bars` and optional SIP fallback.
- Documented new backtest controls: `BACKTEST_LOOKBACK_DAYS`, `BACKTEST_MIN_HISTORY_BARS`, `BACKTEST_SIP_FALLBACK`.

## [2025-10-16] Hardening updates
- Added automatic market time-window detection with detailed logging and improved dry-run messaging in the trade executor.
- Hardened Alpaca authentication checks, refined position sizing against minimum order thresholds, and improved trailing stop failure reporting.
- Ensured fallback candidates always emit at least one canonical row with source provenance and updated pipeline health logging and dashboard reload workflow.
- Relaxed metrics reporting by warning when the trades log is missing instead of failing hard.
- Expanded automated tests covering time-window resolution, sizing safeguards, fallback generation, and pipeline summary tokens.
