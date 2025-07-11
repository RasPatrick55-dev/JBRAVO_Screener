# JBRAVO_Screener
This repository contains a sample workflow for a swing trading automation project.

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
