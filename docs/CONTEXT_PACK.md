# Context Pack (Canonical)

These are the only files that should be uploaded to the ChatGPT Project Sources for this repo.

Repository invariants:
- Paper-only trading only; no live trading support in this repository.
- DB-first operation: PostgreSQL is the source of truth; CSV outputs are debug/parachute only.
- DB config precedence: `DATABASE_URL` -> `DB_*` -> disabled unless `JBR_DEV_DB_DEFAULTS=true`.
- Docs-as-code: doc changes must pass `docs_consistency_check` and CI.
- Never upload secrets (including `~/.config/jbravo/.env`) to any context pack.

## A) Core Truth (always upload first)

- `docs/AI_CONTEXT.md`
- `docs/INDEX.md`
- `docs/reference/artifacts_and_schemas.md`
- `docs/reference/cli_reference.md`

## B) Operator Pack (Agent)

- `docs/how-to/ops_runbook.md`
- `docs/how-to/dev_setup.md`
- `scripts/docs_consistency_check.py`
- `scripts/dashboard_consistency_check.py`

## C) Guardrails / CI

- `.github/workflows/docs-consistency.yml`
- `.pre-commit-config.yaml`
- `pyproject.toml`
- `requirements-dev.txt`

## D) Runtime Entrypoints (profit/behavior work happens here)

- `scripts/run_pipeline.py`
- `scripts/screener.py`
- `scripts/execute_trades.py`
- `scripts/monitor_positions.py`
- `scripts/db.py`

## Refresh Protocol

After any merge that changes behavior/docs:
- Run: `python -m scripts.docs_consistency_check`
- Commit any regenerated `docs/reference/cli_reference.md`.
- Run: `set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.dashboard_consistency_check`
- If the file list changes, update `docs/CONTEXT_PACK.md` and re-upload the pack to the ChatGPT Project.
- Never upload secrets to Project Sources.

## Project Upload Notes

- Upload in small batches (platform UI limits simultaneous uploads).
- Keep Project Sources minimal to avoid contradictory guidance.
- Prefer refreshing `docs/AI_CONTEXT.md` and `docs/reference/cli_reference.md` first.
