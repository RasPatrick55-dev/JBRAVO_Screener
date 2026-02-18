# Architecture (DB-First, Paper-Only)

## Why DB-First

The system writes all operationally significant state to PostgreSQL so every consumer (pipeline, executor, dashboards, checks) can read one authoritative dataset. This prevents drift between file exports and dashboard/API behavior.

## Pipeline Flow

1. `scripts.run_pipeline` runs screener/backtest/metrics/ranker stages.
2. Screener writes candidate rows to DB and emits pipeline tokens.
3. Backtest and metrics consume DB-backed candidate state.
4. Optional CSVs may be exported for diagnostics, but DB remains canonical.
5. Dashboard health and consistency checks read artifacts and logs and validate parity.

## Paper-Only Trading Envelope

- `scripts.execute_trades` enforces a paper-endpoint guard.
- Auth checks hit `/v2/account` before execution proceeds.
- The intended production mode for this repository is simulated/paper execution only.

## Fallback Philosophy

- Fallbacks are for resiliency and observability continuity.
- CSV fallback/parachute output is allowed only when DB data is unavailable.
- Fallback output does not replace DB truth when DB is healthy.

## Operational Contract

- Treat DB views as final data contract.
- Treat CSV artifacts as temporary exports.
- Enforce this contract with `scripts.docs_consistency_check` and `scripts.dashboard_consistency_check`.
