# Positions Tab: Live Data Integration

## Scope

This release wires the `/positions` tab to live broker/account data and backend-calculated metrics.

Implemented components:

- `frontend/src/components/positions/PositionRow.tsx`
- `frontend/src/components/positions/PositionSummary.tsx`
- `frontend/src/components/positions/MonitoringLogsPanel.tsx`
- `frontend/src/components/positions/PositionsTab.tsx`
- `frontend/src/screens/PositionsMonitoring.tsx`

Backend support:

- `dashboards/dashboard_app.py`

## Data Sources and Formulas

### Open positions

- Source: Alpaca open positions (`/api/positions/monitoring` backend route).
- Fields rendered:
  - symbol, qty, entry price, current price, open P/L
  - trailing stop
  - captured P/L
  - days in position

### Trailing stop

- Primary source: live Alpaca trailing-stop orders (`stop_price` per symbol).
- Fallback: DB/event-derived trail metadata or computed trail from current price and trail percent.

### Captured P/L

- Formula (open position stop-trigger scenario):
  - `capturedPL = (trailingStop - entryPrice) * qty`
- This represents expected realized profit/loss if the active trailing stop is triggered now.

### Days in position

- Primary source: Alpaca account activities (`FILL`) from `/v2/account/activities`.
- Logic reconstructs open-lot age per symbol from buy/sell fill history.
- Fallback: DB open-trade earliest entry timestamp.

### Summary row

- Computed server-side and returned by `/api/positions/monitoring` as:
  - `totalShares`
  - `totalOpenPL`
  - `avgDaysHeld`
  - `totalCapturedPL`

### Monitoring logs

- Route: `/api/positions/logs`
- Merged sources:
  - PythonAnywhere monitor logs
  - Alpaca trailing-stop order events
- UI behavior:
  - newest entries first
  - scrollable panel with sticky header
  - sized to show roughly 10 rows at once, with older rows accessible via scrollbar

## API Endpoints Used by Positions Tab

- `GET /api/positions/monitoring`
- `GET /api/positions/logs?limit=200`

## Local Validation

```bash
cd frontend && npm run build
cd .. && python -m dashboards.dashboard_app
```

Open `http://127.0.0.1:8050/` and verify:

- Trailing stop values match Alpaca order stop prices
- Captured P/L matches `(stop - entry) * qty`
- Days reflects activity-derived holding time
- Logs are scrollable and include trailing-stop entries

## GitHub + PythonAnywhere Deploy

From local repo:

```bash
git add dashboards/dashboard_app.py frontend/src docs/deployment/positions_tab_live_data.md docs/ui-reference/Positions_Tab_3Features.jpg docs/ui-reference/Positions_Tab_3Features_FigmaMake.png
git commit -m "Positions tab live data: Alpaca trailing stops, activity-based days held, captured P/L formula, and log panel updates"
git push origin main
```

On PythonAnywhere:

```bash
cd /home/RasPatrick/jbravo_screener
git pull origin main
./build_frontend_pythonanywhere.sh
touch /var/www/raspatrick_pythonanywhere_com_wsgi.py
```

