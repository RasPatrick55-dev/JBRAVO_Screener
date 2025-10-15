# JBRAVO Runbook (Operations)

## Daily Schedule & Time Zones

All automated tasks run in US Eastern time and respect DST changes. The
executor enforces the pre-market window of **07:00–09:30 America/New_York**.
When the Alpaca clock API is unavailable the executor falls back to the
`America/New_York` timezone automatically and logs
`clock_fetch_failed=<status> -> using tz_fallback=America/New_York` once per
run. For cron-style deployments use the UTC equivalents:

| Task | Eastern (local) | UTC |
| ---- | ---------------- | --- |
| `scripts/run_pipeline.py` | 05:20 | 09:20 UTC (EST) / 09:20 UTC (EDT) |
| `scripts/execute_trades.py` | 07:05 | 12:05 UTC (EST) / 11:05 UTC (EDT) |
| `scripts/metrics.py` | 16:30 | 21:30 UTC (EST) / 20:30 UTC (EDT) |

Adjust the UTC offset when daylight saving changes; the executor always
normalises to America/New_York internally.

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
