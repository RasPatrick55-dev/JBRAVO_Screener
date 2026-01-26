# JBRAVO_Screener
This repository contains a sample workflow for a swing trading automation project. See the [logging metrics reference](docs/logging_metrics.md) for structured event documentation captured by the pipeline.

The `dashboards/dashboard_app.py` file implements a full-featured monitoring
dashboard built with Plotly Dash.  It visualizes trade performance, recent
pipeline runs and multiple log files.  Example CSVs and logs can be found under
`data/` and `logs/`.

## Production Status and Validation

The screener is production-ready (post Phase V) and approved for daily pipeline
runs, backtesting, and downstream execution.

By design:
- Bars fetch via IEX (primary) with a single retry; SIP fallback covers missing bars only.
- Partial bar coverage is accepted; fallback is used only on true no-signal days.
- Every candidate written has indicators populated and gates evaluated (passed_gates TRUE/FALSE).
- run_date is not a unique run identifier; pipeline_health_app.run_ts_utc is canonical.
- screener_run_map_app maps symbols to run_ts_utc for validation, backtesting, and metrics.
- DB is the source of truth; use `latest_screener_candidates` and `latest_top_candidates` for final outputs. Candidate CSVs are optional (`JBR_WRITE_CANDIDATE_CSVS=true`).
- Backtest expands bars via API backfill into `daily_bars` when history is short; SIP fallback can be enabled with `BACKTEST_SIP_FALLBACK=true`.
- Each run emits a [SUMMARY] log line; scripts/verify_e2e.py enforces end-to-end correctness.
- Zero-candidate fallback days are valid PASS cases when recorded in pipeline_health_app (CSV fallback only when DB is disabled).
- ML assists ranking only, never creates candidates, and never overrides gates.
  Training is guarded by minimum sample thresholds; ML can be disabled without
  changing baseline screener behavior.

To launch the dashboard locally run:

```
python dashboards/dashboard_app.py
```

## Logo.dev Stock Logos

The Dashboard Monitoring Positions card supports Logo.dev stock ticker logos. Set the publishable key in `.env`:

```
REACT_APP_LOGO_DEV_API_KEY=your_publishable_logo_dev_key
```

If you are building the Vite frontend directly, mirror it as:

```
VITE_LOGO_DEV_API_KEY=your_publishable_logo_dev_key
```

## Frontend Build (PythonAnywhere)

The React/Vite frontend lives in `frontend/` and is served from `frontend/dist`
by `raspatrick_pythonanywhere_com_wsgi.py`. Build the frontend on
PythonAnywhere after pulling new UI changes:

```bash
cd /home/RasPatrick/jbravo_screener
chmod +x install_node_pythonanywhere.sh build_frontend_pythonanywhere.sh
./install_node_pythonanywhere.sh   # installs Node 20.19.0 via nvm
./build_frontend_pythonanywhere.sh
touch /var/www/raspatrick_pythonanywhere_com_wsgi.py
```

Set `VITE_LOGO_DEV_API_KEY` in `frontend/.env` before building. Keep
`frontend/.env` local (it is ignored by git).

## Dashboard Metrics

The dashboard surfaces executor state in the Ops Summary. Metrics map to the executor JSON as follows:

| Field | Meaning |
|-------|---------|
| `configured_max_positions` | The portfolio cap set via CLI/env |
| `risk_limited_max_positions` | Cap after risk scaler |
| `open_positions` | Number of open positions with nonzero qty |
| `open_orders` | Number of live orders |
| `allowed_new_positions` | Number of new positions permitted this run |
| `in_window` | Boolean indicating premarket/market window |
| `exit_reason` | Final executor logic outcome |

Color coding in the Ops Summary uses the following legend: green for OK, yellow for warnings (for example when outside the trading window), red for errors such as auth failures, and gray when a value is unknown.

## Pre-Market Autopilot

The orchestrator now bootstraps environment variables automatically from
`~/.config/jbravo/.env` and the repository `.env` file. Each entry point
(`scripts/screener.py`, `scripts/run_pipeline.py`, `scripts/execute_trades.py`)
logs which files were loaded and fails fast with exit code `2` when the
required Alpaca credentials are missing. Trade execution validates
authorization against `/v2/account` before sizing orders, emits
`TRADING_AUTH_FAILED` on HTTP 401 responses, and records the failure for the
dashboard.

Default sizing was tuned for pre-market operation: each slot targets the larger
of `5%` of buying power or `$200` notional, and price guardrails automatically
fall back to the previous close when `entry_price` is absent. Trailing stops are
attached immediately after fills with explicit `TRAIL_SUBMIT` and
`TRAIL_CONFIRMED` events. Order submission now rounds prices down to the
nearest SEC Rule 612 tick before hitting the Alpaca API, preventing 422
"sub-penny increment" errors during pre-market execution.

The metrics pipeline is resilient to a missing `data/trades_log.csv`; when the
file is absent an empty `data/metrics_summary.csv` is written and the run still
reports `PIPELINE_END rc=0`. Dashboard reloads continue to work even when the
`pa_reload_webapp` CLI is unavailable - the runner touches the WSGI file as a
fallback.

## Cron Job Setup

To keep CSV files in sync with your Alpaca account, schedule
`update_dashboard_data.py` to run every 10 minutes using cron:

```
*/10 * * * * cd /home/RasPatrick/jbravo_screener && /usr/bin/env python3 scripts/update_dashboard_data.py
```

Logs for these updates are written to `logs/data_update.log`.

## Running the Backtest

Scripts inside the `scripts` directory use relative imports and should be
executed as modules from the project root. To run the backtester, use:

```bash
python -m scripts.backtest
```

Running the module in this way ensures Python resolves package imports
correctly.

## Scheduled Pipeline and Trading Tasks

Set up PythonAnywhere scheduled tasks (or cron jobs) to execute the full pipeline
automatically:

```bash
python scripts/run_pipeline.py
```

Schedule the pre-market executor so it fires at 12:05 UTC during Eastern
Standard Time (and adjust the trigger when New York observes DST) to ensure the
wrapper enters the market window after the 07:00 ET pre-market open.

Set `PYTHONANYWHERE_DOMAIN` (e.g., `RasPatrick.pythonanywhere.com`) or `PA_WSGI_PATH` to enable the automatic web reload.
Disable it per-run via `--reload-web false`.

Pipeline options worth knowing:

* `--screener-args "..."` - pass extra flags directly to `scripts.screener`. For example `--screener-args "--feed iex --dollar-vol-min 2_000_000 --reuse-cache true"` keeps the nightly wrapper thin while still exposing all screener tuning knobs.
* `--exec-args "..."` - forward arguments to `scripts.execute_trades`. Combine with `--steps screener,exec` to run the screener and executor back-to-back while still customizing the execution pass (e.g. `--exec-args "--dry-run true --log-json false"`).
* `--reload-web true` triggers a PythonAnywhere reload when the pipeline finishes. If the `pa_reload_webapp` CLI is not available the runner falls back to touching `/var/www/raspatrick_pythonanywhere_com_wsgi.py` so the dashboard still refreshes automatically.
* `JBR_WRITE_CANDIDATE_CSVS=true` enables optional candidate CSV exports; by default DB views `latest_screener_candidates` and `latest_top_candidates` are the source of truth.
* `BACKTEST_LOOKBACK_DAYS` and `BACKTEST_MIN_HISTORY_BARS` expand backtest coverage; `BACKTEST_SIP_FALLBACK=true` allows SIP fallback when IEX bars are missing.
* `scripts.metrics` tolerates a missing or empty `data/trades_log.csv`, allowing fresh installs (before the first live trade) to produce `data/metrics_summary.csv` without manual scaffolding.
* Nightly ranker evaluation runs by default via the `ranker_eval` pipeline step, writing `data/ranker_eval/latest.json` for the Screener Health decile charts. Temporarily skip it with `--steps screener,backtest,metrics` if needed.

The Bollinger-band squeeze component now applies a shape-safe mask so the ranking pass no longer crashes with NumPy shape mismatch errors.

### Sentiment enrichment (optional)

Enable `USE_SENTIMENT=true` to append per-symbol sentiment to screener outputs. The screener reads `SENTIMENT_API_URL` (required) and `SENTIMENT_API_KEY` (optional) for the JSON HTTP provider, using `SENTIMENT_TIMEOUT_SECS` (default `8`) for requests. Scores are cached at `data/cache/sentiment/YYYY-MM-DD.json` so repeated runs only fetch missing symbols for the day. Adjust impact with `SENTIMENT_WEIGHT` (default `0.0`) and optionally gate out strongly negative values with `MIN_SENTIMENT` (default `-999`, which disables gating). When enabled the screener records `sentiment_enabled`, `sentiment_missing_count`, and `sentiment_avg` in `data/screener_metrics.json`.

### Screener pipeline modes

`scripts/screener.py` now exposes dedicated modes that break the nightly run
into smaller steps. This allows running quick sanity checks when iterating on
the data pipeline or refreshing only one stage:

```bash
# build statistics registry from cached bars only
python scripts/screener.py --mode build-symbol-stats

# compute coarse features for the full universe using cached data when possible
python scripts/screener.py --mode coarse-features

# execute the full nightly flow (coarse + full ranking)
python scripts/screener.py --mode full-nightly
```

Helpful flags include:

* `--prefilter-days` / `--prefilter-top` - tune the universe size before the
  final ranking pass.
* `--full-days` - control the history loaded for shortlisted symbols.
* `--reuse-cache` / `--refresh-latest` - toggle cached parquet reuse and whether
  to refresh optional candidate exports (when `JBR_WRITE_CANDIDATE_CSVS=true`).

Refer to `config/ranker.yml` for the ranking weights applied during the
`full-nightly` mode.

### Executor dry-run mode

The execution script accepts a `--dry-run` flag and always loads candidates
from PostgreSQL (latest screener view).

```bash
# Preview orders without hitting the brokerage API
python scripts/execute_trades.py --dry-run

# Confirm candidate-source log tokens
python scripts/execute_trades.py --dry-run | rg "CANDIDATE_SOURCE db|DB_CANDIDATES|NO_CANDIDATES"

# Tighten liquidity/price guardrails (values shown are the defaults)
python scripts/execute_trades.py --min-adv20 2000000 --min-price 1 --max-price 1000
```

Dry runs log `WOULD SUBMIT` entries while still producing
`data/execute_metrics.json` so the dashboard reflects the simulated run.

When creating a scheduled task for `execute_trades.py` on PythonAnywhere make
sure the working directory is set to the project root so logs and CSV files are
written in the correct location:

```bash
cd /home/RasPatrick/jbravo_screener && /home/RasPatrick/.virtualenvs/jbravo-env/bin/python scripts/execute_trades.py
```

Ensure all tasks activate the virtual environment explicitly. Example entries:

```bash
cd /home/RasPatrick/jbravo_screener && /home/RasPatrick/.virtualenvs/jbravo-env/bin/python scripts/run_pipeline.py
cd /home/RasPatrick/jbravo_screener && /home/RasPatrick/.virtualenvs/jbravo-env/bin/python scripts/metrics.py
cd /home/RasPatrick/jbravo_screener && /home/RasPatrick/.virtualenvs/jbravo-env/bin/python scripts/weekly_summary.py
cd /home/RasPatrick/jbravo_screener && /home/RasPatrick/.virtualenvs/jbravo-env/bin/python scripts/monitor_positions.py
```

## Market Data

Historical prices and volume are now fetched exclusively from the Alpaca Market Data API using the `alpaca-py` SDK. The previous fallback to the `tvdatafeed` package has been removed.
