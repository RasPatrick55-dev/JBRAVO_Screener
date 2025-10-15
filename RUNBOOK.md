# Operations Runbook

## Nightly pipeline

* Use `python -m scripts.run_pipeline --steps screener,backtest,metrics` in scheduled tasks. Pass additional screener flags via `--screener-args "..."` when you need to tweak liquidity thresholds without editing the wrapper script.
* After each screener run the pipeline inspects `data/screener_metrics.json`. If the screener returns zero rows the log will contain `FALLBACK_CHECK start` followed by `FALLBACK_CHECK rows_out=N source=fallback` after invoking `scripts.fallback_candidates`. This guarantees at least one candidate row with canonical headers (`timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown,entry_price,adv20,atrp,source`) for the executor and dashboard.
* PythonAnywhere deployments no longer require the `pa_reload_webapp` helper. When it is missing the pipeline touches `/var/www/raspatrick_pythonanywhere_com_wsgi.py` and logs `AUTO-RELOAD fallback touch ok: …` to trigger the reload.
* `scripts.metrics` can be scheduled before the first trade executes; it now tolerates a missing or empty `data/trades_log.csv` and still writes `data/metrics_summary.csv` with zeroed metrics.
* The backtester exits cleanly with `Backtest: no candidates today — skipping.` when `data/top_candidates.csv` has no rows, so nightly automation does not fail on quiet days.
