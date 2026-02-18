# QA Plan and Definition of Done

This document outlines the quality assurance expectations and release hygiene required for the new scoring pipeline rollout.

## 1. QA Plan & Definition of Done

### Unit Tests
- Validate scoring math for each component (e.g., positive MACD yields a bonus, ATR% breaches apply penalties, etc.).
- Ensure `score_breakdown` serialization to a JSON string preserves the canonical header field.

### Integration Tests
- Dry-run pipeline generates `predictions/<date>.csv` and DB writes for `screener_candidates`/`top_candidates`.
- Backtest metrics include the new fields and the dashboard reads them without raising HTTP 500 errors.

### Operations Acceptance (Daily)
- `grep` inspection confirms `PIPELINE_*` tokens (and `FALLBACK_CHECK`, when used) are present.
- `latest_screener_candidates` and `latest_top_candidates` return non-zero rows for active trading days.
- Executor logs display buy/stop events or an explicit skip, and metrics are persisted.
- Dashboard KPIs and screener tables refresh successfully and the evidence bundle renders.

## 2. Shipping Guidelines (PR Hygiene)

- Deliver small PRs per milestone with unified diffs, thorough docstrings, and without breaking CLI flags or canonical headers (executor/screener).
- Introduce feature flags `--ranker-config` and `--featureset v2` to gate the new scoring during rollout.
- Maintain configuration file versioning to enable instant rollback of weight changes.
