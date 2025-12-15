# Operations Runbook

## Nightly pipeline

* Use `python -m scripts.run_pipeline --steps screener,backtest,metrics,ranker_eval` in scheduled tasks. Pass additional screener flags via `--screener-args "..."` when you need to tweak liquidity thresholds without editing the wrapper script.
* After each screener run the pipeline inspects `data/top_candidates.csv`. If there are no rows it invokes `scripts.fallback_candidates`, logs `[INFO] FALLBACK_CHECK reason=no_candidates rows_out=<N> source=<scored|predictions|static>`, and rewrites `data/latest_candidates.csv` with canonical headers (`timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown,entry_price,adv20,atrp,source`). This ensures the executor and dashboard always have at least one row to render.
* PythonAnywhere deployments now log `[INFO] DASH_RELOAD method=<pa|touch> rc=<0|ERR>` at the end of each pipeline run. The pipeline first tries `pa_reload_webapp $PYTHONANYWHERE_DOMAIN`; when unavailable it touches `/var/www/${PYTHONANYWHERE_DOMAIN//./_}_wsgi.py` instead.
* `scripts.metrics` can be scheduled before the first trade executes; both the pipeline and the metrics script tolerate a missing `data/trades_log.csv`, creating an empty header stub and still writing `data/metrics_summary.csv` with zeroed metrics.
* The Screener Health dashboard surfaces the latest pipeline tokens (`PIPELINE_START`, `PIPELINE_SUMMARY`, `FALLBACK_CHECK`, `PIPELINE_END`) even on zero-candidate days. When fallback data is active the Top Candidates table displays a subtle `fallback` badge sourced from the `latest_candidates.csv` `source` column.
* The backtester exits cleanly with `Backtest: no candidates today — skipping.` when `data/top_candidates.csv` has no rows, so nightly automation does not fail on quiet days.

### Nightly ML data pipeline (Stage 0–2)

* Stage 0 (bars) runs at **10:50 UTC**: `python -m scripts.run_pipeline --steps screener,backtest,metrics --export-daily-bars-path data/daily_bars.csv` to emit `data/daily_bars.csv` for downstream ML. The export always writes atomically and logs `[INFO] DAILY_BARS_EXPORTED path=<...> rows=<...> symbols=<...>` when the flag is provided (even if the frame is empty).
* Stage 1 (labels) runs at **10:58 UTC**: `python -m scripts.label_generator --bars-path data/daily_bars.csv --output-dir data/labels --horizons 5 10 --threshold-percent 3.0` writes `data/labels/labels_<RUNDATE>.csv` where `RUNDATE` is the current America/New_York date (stale bars trigger a `[WARN] LABELS_INPUT_STALE …` log rather than shifting the filename to the bars’ max date).
* Stage 2 (features) runs at **11:05 UTC**: `python -m scripts.feature_generator --bars-path data/daily_bars.csv` produces `data/features/features_<RUNDATE>.csv` using the same run date and logs `[WARN] FEATURES_*_STALE …` if either bars or labels lag behind the run day.
* The three stages form the ML data pipeline (Stage 0: bars, Stage 1: labels, Stage 2: features) and must run in order because labels and features depend on the exported daily bars. The nightly rerank step also writes `data/nightly_ml_status.json` summarizing the freshest bars/labels/features/model/predictions/eval artifacts.

### Stage 2 ML: feature generator

* Runs after the Stage 0 bars export and Stage 1 labels generation to populate the Stage 2 features of the ML pipeline.
* Inputs: `data/daily_bars.csv`, `data/labels/labels_*.csv`.
* Output: `data/features/features_<date>.csv`.
* Manual run from repo root:

```
cd /home/RasPatrick/jbravo_screener
source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate
set -a; . ~/.config/jbravo/.env; set +a
python -m scripts.feature_generator
```
