# JBRAVO_Screener
This repository contains a sample workflow for a swing trading automation project. See the [logging metrics reference](docs/logging_metrics.md) for structured event documentation captured by the pipeline.

The `dashboards/dashboard_app.py` file implements a full-featured monitoring
dashboard built with Plotly Dash.  It visualizes trade performance, recent
pipeline runs and multiple log files.  Example CSVs and logs can be found under
`data/` and `logs/`.

To launch the dashboard locally run:

```
python dashboards/dashboard_app.py
```

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

Set `PYTHONANYWHERE_DOMAIN` (e.g., `RasPatrick.pythonanywhere.com`) or `PA_WSGI_PATH` to enable the automatic web reload.
Disable it per-run via `--reload-web false`.

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

* `--prefilter-days` / `--prefilter-top` — tune the universe size before the
  final ranking pass.
* `--full-days` — control the history loaded for shortlisted symbols.
* `--reuse-cache` / `--refresh-latest` — toggle cached parquet reuse and whether
  to refresh the `latest_candidates.csv` symlink.

Refer to `config/ranker.yml` for the ranking weights applied during the
`full-nightly` mode.

### Executor dry-run mode

The execution script now accepts a `--dry-run` flag and a `--source` override to
load a custom candidates file:

```bash
# Preview orders without hitting the brokerage API
python scripts/execute_trades.py --dry-run

# Execute using a different candidate list
python scripts/execute_trades.py --source data/predictions/2024-05-01.csv
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
