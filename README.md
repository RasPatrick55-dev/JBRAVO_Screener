# JBRAVO Screener

Documentation entry point: `docs/INDEX.md`

Monitor exit accounting: [verified fill exports and fresh coverage snapshots](docs/how-to/monitor_exit_accounting.md).

Acceptance evidence preparation: [forward predictor observations and session labels](docs/reference/production_score_observations.md).
The primary preserves new score observations when it refreshes predictions;
these records do not clear model or trading acceptance gates.
The [production evidence consumer](docs/reference/production_evidence_qualification.md)
verifies reviewed comparable score populations and mature outcomes and saves an
isolated monitor candidate. Publication and full acceptance remain separate.

Research declaration boundary: [contract](docs/reference/research_admission_contract.md).
For its scoped synthetic verification, set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and
`PYTHONDONTWRITEBYTECODE=1`, then run under the implementation packet's external
60-second timeout and finite attempt allowance:

```text
python -B -m pytest --noconftest -p no:cacheprovider -o addopts= tests/test_research_admission.py -q
```

These declaration checks do not qualify historical data or authorize execution.

## Current Operating Mode

The fixed primary separates computational completion from research qualification
under the [pipeline-only readiness policy](docs/how-to/pipeline_readiness_only.md).
Valid research diagnostics can remain pending while the pipeline completes.
New trading entries remain blocked, including after a healthy monitor verdict;
current-data, source, candidate and session checks remain required.

- Paper-only trading workflow.
- PostgreSQL is the source of truth.
- CSV outputs are debug/parachute exports, not authoritative state.

Monitor sell fills are recorded as `SELL_FILL_OBSERVED`; the hourly account-sync
reconciler owns canonical `SELL_FILL` recording and trade closure. Deploy the
reviewed monitor and reconciliation source together and keep account sync enabled.
See [the observation and position-closure contract](docs/how-to/hourly_account_sync.md)
for setup, placeholder upgrade rules and offline verification limits.

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
### Current-session feature and prediction inputs

Feature artifacts use the actual daily-bar session date. To generate inputs for a
specific completed session, run `python -m scripts.feature_generator --run-date
YYYY-MM-DD` in the configured environment. A missing or mismatched session fails
before file or database publication. The primary pipeline supplies this date and
refreshes legacy feature metadata that lacks a matching `signal_session`.

Recent rows retain unknown forward labels as blank values so prediction can use
current features. Training still excludes rows without its chosen target label.
Production score captures contain normalized features and scores without outcomes;
capture failures export fixed reason codes and continue to block publication.

See [the October 13 recovery runbook](docs/how-to/oct13_pipeline_recovery.md) for
the separate deployment and prospective research-enrollment steps.
