# AI Context (Canonical)

This file is the single source of truth for assistant context in this repo.

## Current Operating Truth

- Trading mode is paper-only. Production live trading is not supported in this repository.
- PostgreSQL is the source of truth for pipeline and dashboard state.
- Canonical candidate reads come from DB views: `latest_screener_candidates` and `latest_top_candidates`.
- CSV candidate files are non-authoritative and exist only for debug/parachute workflows.
- Candidate CSV export is optional via `JBR_WRITE_CANDIDATE_CSVS=true`.

## Canonical Daily One-Liners

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.run_pipeline --steps screener,backtest,metrics,ranker_eval --reload-web true
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; bash bin/run_premarket_once.sh
cd /home/RasPatrick/jbravo_screener && touch /var/www/raspatrick_pythonanywhere_com_wsgi.py
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
