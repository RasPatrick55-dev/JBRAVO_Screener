# JBRAVO Screener

Documentation entry point: `docs/INDEX.md`

## Current Operating Mode

- Paper-only trading workflow.
- PostgreSQL is the source of truth.
- CSV outputs are debug/parachute exports, not authoritative state.

## Canonical Daily One-Liners

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.run_pipeline --steps screener,backtest,metrics,ranker_eval --reload-web true
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; bash bin/run_premarket_once.sh
cd /home/RasPatrick/jbravo_screener && touch /var/www/raspatrick_pythonanywhere_com_wsgi.py
```

## Consistency Checks

```bash
python -m scripts.docs_consistency_check
python -m scripts.dashboard_consistency_check
```
