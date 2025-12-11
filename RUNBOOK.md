# Operations Runbook

## Nightly pipeline

* Use `python -m scripts.run_pipeline --steps screener,backtest,metrics,ranker_eval` in scheduled tasks. Pass additional screener flags via `--screener-args "..."` when you need to tweak liquidity thresholds without editing the wrapper script.
* After each screener run the pipeline inspects `data/top_candidates.csv`. If there are no rows it invokes `scripts.fallback_candidates`, logs `[INFO] FALLBACK_CHECK reason=no_candidates rows_out=<N> source=<scored|predictions|static>`, and rewrites `data/latest_candidates.csv` with canonical headers (`timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown,entry_price,adv20,atrp,source`). This ensures the executor and dashboard always have at least one row to render.
* PythonAnywhere deployments now log `[INFO] DASH_RELOAD method=<pa|touch> rc=<0|ERR>` at the end of each pipeline run. The pipeline first tries `pa_reload_webapp $PYTHONANYWHERE_DOMAIN`; when unavailable it touches `/var/www/${PYTHONANYWHERE_DOMAIN//./_}_wsgi.py` instead.
* `scripts.metrics` can be scheduled before the first trade executes; both the pipeline and the metrics script tolerate a missing `data/trades_log.csv`, creating an empty header stub and still writing `data/metrics_summary.csv` with zeroed metrics.
* The Screener Health dashboard surfaces the latest pipeline tokens (`PIPELINE_START`, `PIPELINE_SUMMARY`, `FALLBACK_CHECK`, `PIPELINE_END`) even on zero-candidate days. When fallback data is active the Top Candidates table displays a subtle `fallback` badge sourced from the `latest_candidates.csv` `source` column.
* The backtester exits cleanly with `Backtest: no candidates today — skipping.` when `data/top_candidates.csv` has no rows, so nightly automation does not fail on quiet days.

### Stage 2 ML: feature generator

* Runs after the nightly pipeline and labels tasks to populate the features stage of ML.
* Inputs: `data/daily_bars.csv`, `data/labels/labels_*.csv`.
* Output: `data/features/features_<date>.csv`.
* Manual run from repo root:

```
cd /home/RasPatrick/jbravo_screener
source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate
set -a; . ~/.config/jbravo/.env; set +a
python -m scripts.feature_generator
```
