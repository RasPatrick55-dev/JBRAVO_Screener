# Dashboard Consistency Review

The dashboard consistency checker validates that the Screener dashboard reflects the latest pipeline and executor artifacts. It performs lightweight instrumentation so operators can confirm every KPI on the dashboard lines up with the underlying files even when no candidates are produced.

## Running the checker

```bash
python -m scripts.dashboard_consistency_check
```

The command emits three artifacts under `reports/` (or the directory supplied to `--reports-dir`):

- `dashboard_consistency.json` &mdash; machine-readable payload containing all checks, timestamps, and deltas.
- `dashboard_kpis.csv` &mdash; flattened KPIs that the dashboard can ingest.
- `dashboard_findings.txt` &mdash; quick human-readable summary of the run.

Use `--base` to point at another repository root (useful for tests or sandboxes) and `--reports-dir` to change the output directory.

## What is validated

The checker inspects the latest screener metrics, candidate CSVs, executor logs, and optional prediction artifacts. It records:

- Most recent timestamps from `screener_metrics.json`, `metrics_summary.csv`, and pipeline log markers.
- Universe coverage (`symbols_in`, `symbols_with_bars`, `rows`, and bar totals).
- Candidate file health, canonical headers, first three rows, and whether a fallback source was required.
- Gate pressure breakdown from `screener_metrics.json`.
- Stage timings (`fetch_secs`, `feature_secs`, `rank_secs`, `gate_secs`).
- Pipeline tokens (`PIPELINE_START`, `PIPELINE_SUMMARY`, `FALLBACK_CHECK`, `PIPELINE_END`, `DASH RELOAD`).
- Executor summary (orders submitted/filled/canceled, trailing stops, skip reasons).
- Trades log presence (missing logs are non-fatal but clearly indicated).
- Prediction/decile artifacts when available, including decile lift summaries.
- Prefix guard to detect alphabet bias in scored candidates.

Missing files are tolerated: the checker sets `present=false` flags so monitoring stays operational on zero-candidate or first-run days.

## Using the output

- `dashboard_consistency.json` can backfill health endpoints or feed monitoring dashboards.
- `dashboard_kpis.csv` provides a flat set of metrics that can be consumed by the existing Dash components without additional parsing.
- `dashboard_findings.txt` highlights the essentials for on-call review (last run times, candidate source, trades log availability, executor activity).

Integrate the checker into daily or hourly jobs after the pipeline completes so health dashboards are refreshed alongside the artifacts they represent.
