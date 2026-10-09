# JBRAVO Screener

Documentation entry point: `docs/INDEX.md`

Research declaration boundary: [contract](docs/reference/research_admission_contract.md).
For its scoped synthetic verification, set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and
`PYTHONDONTWRITEBYTECODE=1`, then run under the implementation packet's external
60-second timeout and finite attempt allowance:

```text
python -B -m pytest --noconftest -p no:cacheprovider -o addopts= tests/test_research_admission.py -q
```

These declaration checks do not qualify historical data or authorize execution.

## Current Operating Mode

- Paper-only trading workflow.
- PostgreSQL is the source of truth.
- CSV outputs are debug/parachute exports, not authoritative state.

## Canonical Daily One-Liners

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.run_pipeline --steps screener,backtest,metrics,ranker_eval --reload-web true
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; bash bin/run_premarket_once.sh
cd /home/RasPatrick/jbravo_screener && set -a; . ~/.config/jbravo/.env; set +a; PYTHONANYWHERE_DOMAIN="${PYTHONANYWHERE_DOMAIN:-${PYTHONANYWHERE_USERNAME}.pythonanywhere.com}"; curl -fsS -X POST -H "Authorization: Token ${PYTHONANYWHERE_API_TOKEN}" "https://www.pythonanywhere.com/api/v0/user/${PYTHONANYWHERE_USERNAME}/webapps/${PYTHONANYWHERE_DOMAIN}/reload/"
```

## Consistency Checks

```bash
python -m scripts.docs_consistency_check
python -m scripts.dashboard_consistency_check
```
