# JBRAVO Runbook (Operations)

## Daily Schedule & Time Zones

All automated tasks run in US Eastern time and respect DST changes. The
executor enforces the pre-market window of **07:00–09:30 America/New_York**.
When the Alpaca clock API is unavailable the executor falls back to the
`America/New_York` timezone automatically and logs
`clock_fetch_failed status=<status> -> using tz_fallback=America/New_York` once
per run. For cron-style deployments use the UTC equivalents:

| Task | Eastern (local) | UTC |
| ---- | ---------------- | --- |
| `scripts/run_pipeline.py` | 05:20 | 09:20 UTC (EST) / 09:20 UTC (EDT) |
| ML Stage 0: bars export (`run_pipeline --export-daily-bars-path`) | 05:50 | 10:50 UTC (EST) / 10:50 UTC (EDT) |
| ML Stage 1: labels (`label_generator --threshold-percent 3.0`) | 05:58 | 10:58 UTC (EST) / 10:58 UTC (EDT) |
| ML Stage 2: features (`feature_generator`) | 06:05 | 11:05 UTC (EST) / 11:05 UTC (EDT) |
| `scripts/execute_trades.py` | 07:05 | 12:05 UTC (EST) / 11:05 UTC (EDT) |
| `scripts/metrics.py` | 16:30 | 21:30 UTC (EST) / 20:30 UTC (EDT) |

Adjust the UTC offset when daylight saving changes; the executor always
normalises to America/New_York internally.

### Pre-Market Autopilot

Trade execution opens positions only within the 07:00–09:30 America/New_York
window when `time_window=premarket`. In UTC that window spans **12:00–14:30**
during Eastern Standard Time (EST, UTC−5) and **11:00–13:30** during Eastern
Daylight Time (EDT, UTC−4). Schedulers should apply the appropriate offset when
running on cron-like systems.

## Pipeline & Dashboard Guarantees

* `PIPELINE_START`, `PIPELINE_SUMMARY`, and `PIPELINE_END rc=<code>` are emitted
  on every run regardless of downstream warnings. A missing
  `data/trades_log.csv` no longer blocks metrics generation; an empty
  `data/metrics_summary.csv` is written with the warning
  `[WARN] no trades_log.csv -> writing empty metrics_summary`.
* Dashboard reloads prefer the `pa_reload_webapp` CLI. If it is not installed,
  the pipeline touches the configured WSGI file and logs
  `[INFO] DASH RELOAD method=touch local rc=<code> path=<path>`.
* Trade execution validates credentials by calling `/v2/account` before
  computing order sizes. HTTP 401 responses emit
  `[ERROR] TRADING_AUTH_FAILED … tip="Reload ~/.config/jbravo/.env"` and exit
  with status `2`. Trailing stops log `TRAIL_SUBMIT … route=trailing_stop` and
  `TRAIL_CONFIRMED …` so the dashboard keeps consistent tokens.

## ML data pipeline (Stage 0–2)

* Stage 0 (bars) runs `python -m scripts.run_pipeline --steps screener,backtest,metrics --export-daily-bars-path data/daily_bars.csv` to emit `data/daily_bars.csv`. The export always writes atomically and logs `[INFO] DAILY_BARS_EXPORTED path=<...> rows=<...> symbols=<...>` when the flag is provided.
* Stage 1 (labels) runs `python -m scripts.label_generator --bars-path data/daily_bars.csv --output-dir data/labels --horizons 5 10 --threshold-percent 3.0` to write `data/labels/labels_<RUNDATE>.csv` using the current America/New_York date instead of the max bar date (stale bars warn via `[WARN] LABELS_INPUT_STALE …`).
* Stage 2 (features) runs `python -m scripts.feature_generator --bars-path data/daily_bars.csv` to write `data/features/features_<RUNDATE>.csv` on the same run date and surfaces `[WARN] FEATURES_*_STALE …` if the inputs lag the run day.
* The nightly rerank step writes `data/nightly_ml_status.json` to snapshot the freshest bars/labels/features/model/predictions/eval artifacts.

## Environment Expectations

All entry points call `load_env()` before parsing CLI arguments. The helper
loads `~/.config/jbravo/.env` followed by the repo `.env`, logs the discovered
paths, and aborts with exit code `2` if any of the following are missing:

```
APCA_API_KEY_ID
APCA_API_SECRET_KEY
APCA_API_BASE_URL
APCA_DATA_API_BASE_URL
ALPACA_DATA_FEED
```

Set those variables once and keep the files readable by the automation user so
every scheduled task starts with a consistent environment.
