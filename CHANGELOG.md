# Changelog

## [2025-10-16] Hardening updates
- Added automatic market time-window detection with detailed logging and improved dry-run messaging in the trade executor.
- Hardened Alpaca authentication checks, refined position sizing against minimum order thresholds, and improved trailing stop failure reporting.
- Ensured fallback candidates always emit at least one canonical row with source provenance and updated pipeline health logging and dashboard reload workflow.
- Relaxed metrics reporting by warning when the trades log is missing instead of failing hard.
- Expanded automated tests covering time-window resolution, sizing safeguards, fallback generation, and pipeline summary tokens.
