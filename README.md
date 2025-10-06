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
