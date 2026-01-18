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

The checker inspects the latest screener metrics, candidate outputs (DB views with optional CSV exports), executor logs, and optional prediction artifacts. It records:

- Most recent timestamps from `screener_metrics.json`, `metrics_summary.csv`, and pipeline log markers.
- Universe coverage (`symbols_in`, `symbols_with_bars`, `rows`, and bar totals).
- Candidate output health (DB view row counts, canonical headers when CSVs are enabled, and fallback flags).
- Gate pressure breakdown from `screener_metrics.json`.
- Stage timings (`fetch_secs`, `feature_secs`, `rank_secs`, `gate_secs`).
- Pipeline tokens (`PIPELINE_START`, `PIPELINE_SUMMARY`, `FALLBACK_CHECK`, `PIPELINE_END`, `DASH RELOAD`).
- Executor summary (orders submitted/filled/canceled, trailing stops, skip reasons).
- Trades log presence (missing logs are non-fatal but clearly indicated).
- Prediction/decile artifacts when available, including decile lift summaries.
- Prefix guard to detect alphabet bias in scored candidates.

Missing artifacts are tolerated: the checker sets `present=false` flags so monitoring stays operational on zero-candidate or first-run days.

## Using the output

- `dashboard_consistency.json` can backfill health endpoints or feed monitoring dashboards.
- `dashboard_kpis.csv` provides a flat set of metrics that can be consumed by the existing Dash components without additional parsing.
- `dashboard_findings.txt` highlights the essentials for on-call review (last run times, candidate source, trades log availability, executor activity).

Integrate the checker into daily or hourly jobs after the pipeline completes so health dashboards are refreshed alongside the artifacts they represent.

## Paper trading indicators

![Paper Trading Mode badge](images/paper-mode-badge.svg)

The dashboard highlights paper-only environments with a lightweight badge. The Screener and Execution tabs both surface the badge when `APCA_API_BASE_URL` points at the paper endpoint or `JBR_EXEC_PAPER=true` is exported. This keeps operators aware that orders are simulated and allows the checker to assert consistency with paper artifacts.

## Operator checklist

Run the checker and then confirm the live dashboard matches the generated findings:

1. **Last run timestamps** - Screener and Execution panels should show the same UTC timestamps reported in `dashboard_findings.txt`.
2. **Candidate state** - If `latest_screener_candidates` returns zero rows, the Screener tab displays the "No candidates today" hint that links directly to the pipeline panel.
3. **Pipeline tokens** - The pipeline log summary renders the latest `[INFO] PIPELINE_*` markers (START, SUMMARY, FALLBACK_CHECK, END, DASH RELOAD) in the same order as the checker output.
4. **Execution metrics** - When `execute_metrics.json` is missing (paper-only mornings), the Execution tab keeps KPIs at zero and shows "No execution yet today (paper)" while the badge reinforces the environment.
5. **Trades and logs** - If neither `executed_trades.csv` nor `trades_log.csv` is available the tab presents the "No trades yet (paper account)" hint instead of erroring, matching the checker's `present=false` flags.
6. **Deciles and predictions** - Missing evaluation artifacts should resolve to "Not computed yet" placeholders without stack traces; once present they match the checker's decile lift summaries.

Reviewing these items alongside the `reports/dashboard_consistency.json` payload ensures the UI and artifacts stay in lock-step even on zero-candidate or fallback-only days.
