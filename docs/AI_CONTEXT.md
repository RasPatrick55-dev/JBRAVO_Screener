# AI Context (Canonical)

This file is the single source of truth for assistant context in this repo.

## Current Operating Truth

- Trading mode is paper-only. Production live trading is not supported in this repository.
- PostgreSQL is the source of truth for pipeline and dashboard state.
- Canonical candidate reads come from DB views: `latest_screener_candidates` and `latest_top_candidates`.
- CSV candidate files are non-authoritative and exist only for debug/parachute workflows.
- Swing Research Department entry point: [strategy testing and handoff](how-to/strategy_testing.md).
- The enabled daily strategy route uses Pullback/Reclaim v2 across the supported active/tradable Alpaca U.S. equity universe. The ten-symbol study is an offline fixture, not the daily production universe.
- Producer, primary, strategy Backtest/Metrics, paper entry and always-on monitoring are separate stages. Keep strategy/version/definition and signal IDs consistent across the PostgreSQL handoff; do not substitute legacy ML picks.
- Owner-authorized paper testing is separate from research qualification. The [dated October 27 evidence](reference/pullback_reclaim_verification_2026-10-27.json) verifies an isolated positive stage audit, not a production candidate publication, broker fill or protected exit.
- Candidate CSV export is optional via `JBR_WRITE_CANDIDATE_CSVS=true`.
- DB config precedence is `DATABASE_URL` first, then `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`; otherwise DB is treated as disabled (unless `JBR_DEV_DB_DEFAULTS=true` is set for local fallback).
- PythonAnywhere API credentials for local VS Code + Codex use `PA_USERNAME`, `PA_TOKEN`, and `PA_HOST` with the canonical lookup contract in `docs/how-to/pythonanywhere_credentials.md`.
- Never upload `~/.config/jbravo/.env`, `%USERPROFILE%\.config\jbravo\.env`, `.env.local`, or `.env.pythonanywhere.local` to any context pack.
- Before claiming remote PythonAnywhere execution, list consoles, send `pwd`, and confirm `get_latest_output` succeeds. If the API returns HTTP `412`, stop and report that the console must be started in a browser first.

## Scheduled strategy operations

Use the shared PythonAnywhere MCP and its scoped operation contracts to inspect
the installed tasks. The [department workflow](how-to/strategy_testing.md#daily-operations-and-proof)
lists the verified task IDs, dependencies and evidence locations. Runtime source
bindings, current session and account identity must verify before execution.
An old audit, expired signal or a completed schedule edit is not current run proof.

## Legacy manual command reference

The commands below are not the current strategy task definitions. Use them only
within an explicitly scoped legacy/recovery operation; they are not a substitute
for the source-bound producer, primary, paper executor and monitor workflow.

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.run_pipeline --steps screener,backtest,metrics,ranker_eval --reload-web true
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; bash bin/run_premarket_once.sh
cd /home/RasPatrick/jbravo_screener && set -a; . ~/.config/jbravo/.env; set +a; PYTHONANYWHERE_DOMAIN="${PYTHONANYWHERE_DOMAIN:-${PYTHONANYWHERE_USERNAME}.pythonanywhere.com}"; curl -fsS -X POST -H "Authorization: Token ${PYTHONANYWHERE_API_TOKEN}" "https://www.pythonanywhere.com/api/v0/user/${PYTHONANYWHERE_USERNAME}/webapps/${PYTHONANYWHERE_DOMAIN}/reload/"
```

## Core Pipeline Tokens

- `PIPELINE_START`
- `PIPELINE_SUMMARY`
- `FALLBACK_CHECK`
- `PIPELINE_END`
- `DASH RELOAD`

## Doc Hygiene Rule

Before merging behavior/documentation changes, run:

```bash
python -m scripts.docs_consistency_check
```

This regenerates CLI reference docs and fails if docs drift from paper-only + DB-first policy.

For local hook setup, see `docs/how-to/dev_setup.md`.

For PythonAnywhere API credential lookup and the required console handshake, see `docs/how-to/pythonanywhere_credentials.md`.
