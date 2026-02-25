# Account Tab: Runtime Data Integration

## Scope

This release updates the Account tab to the new dark Figma layout and wires all cards to Alpaca paper runtime data and order logs.

Implemented frontend files:

- `frontend/src/components/account/AccountTab.tsx`
- `frontend/src/components/account/AccountPerformanceCard.tsx`
- `frontend/src/components/account/EquityCurveCard.tsx`
- `frontend/src/components/account/AccountBreakdownCard.tsx`
- `frontend/src/components/account/OpenOrdersCard.tsx`
- `frontend/src/components/account/DailyOrderLogsCard.tsx`
- `frontend/src/components/account/types.ts`
- `frontend/src/screens/AccountOverview.tsx`

Backend support:

- `dashboards/dashboard_app.py`

## API Endpoints

- `GET /api/account/summary`
- `GET /api/account/performance?range=d|w|m|y|all`
- `GET /api/account/portfolio_history?period=1Y|ALL&timeframe=1D`
- `GET /api/account/open_orders?limit=50`
- `GET /api/account/order_logs?limit=100`

## Data Sources

### Account totals

- Source: Alpaca paper `GET /v2/account`
- Output: live equity, cash, buying power, gross open exposure, and cash-to-positions ratio

### Equity curve and performance

- Source: Alpaca paper `GET /v2/account/portfolio/history`
- Account Total equity source:
  - prefers live equity from `GET /v2/account` (`equityBasis=live`)
  - falls back to last close from portfolio history (`equityBasis=last_close`)
- Performance windows:
  - daily (latest equity vs previous close)
  - weekly (7 days)
  - monthly (30 days)
  - yearly (365 days)
  - all-time (first point delta)

### Open orders

- Source: Alpaca paper `GET /v2/orders?status=open`
- Output rows: symbol, type, side, qty, price/stop, submitted timestamp

### Daily order logs

- Preferred source: Postgres `order_events` (today, newest-first)
- Fallback source: `logs/execute_trades.log` tail parser (today, newest-first)

## Local Validation

```bash
python -m py_compile dashboards/dashboard_app.py
cd frontend && npm run build
```

For Vite local dev (`npm run dev`), API calls are proxied to Flask via `frontend/vite.config.ts`:

- `/api/* -> http://127.0.0.1:8050`
- `/assets/* -> http://127.0.0.1:8050`
- `/ui-assets/* -> http://127.0.0.1:8050`

Optional direct API origin override for frontend dev:

- set `VITE_API_BASE_URL=http://127.0.0.1:8050`

Run the dashboard and verify endpoint payloads:

```bash
python -m dashboards.dashboard_app
curl -s "http://127.0.0.1:8050/api/account/summary" | python -m json.tool
curl -s "http://127.0.0.1:8050/api/account/performance?range=all" | python -m json.tool
curl -s "http://127.0.0.1:8050/api/account/portfolio_history?period=1Y&timeframe=1D" | python -m json.tool
curl -s "http://127.0.0.1:8050/api/account/open_orders?limit=50" | python -m json.tool
curl -s "http://127.0.0.1:8050/api/account/order_logs?limit=100" | python -m json.tool
```

Windows PowerShell equivalent:

```powershell
# stop any stale backend on 8050 first
Get-NetTCPConnection -LocalPort 8050 -State Listen -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# start backend (repo root)
$env:APCA_API_BASE_URL="https://paper-api.alpaca.markets"
python -m dashboards.dashboard_app
```

In a second PowerShell terminal:

```powershell
# repo/frontend
npm run dev

# verify API payloads directly
irm http://127.0.0.1:8050/api/account/summary | ConvertTo-Json -Depth 6
irm "http://127.0.0.1:8050/api/account/performance?range=all" | ConvertTo-Json -Depth 6
irm "http://127.0.0.1:8050/api/account/portfolio_history?period=1Y&timeframe=1D" | ConvertTo-Json -Depth 6
irm "http://127.0.0.1:8050/api/account/open_orders?limit=50" | ConvertTo-Json -Depth 6
irm "http://127.0.0.1:8050/api/account/order_logs?limit=100" | ConvertTo-Json -Depth 6
```

## PythonAnywhere Deploy

From local repo:

```bash
git add dashboards/dashboard_app.py frontend/src/components/account frontend/src/screens/AccountOverview.tsx docs/reference/account_tab_runtime_data.md
git commit -m "Account tab: Figma dark layout + Alpaca paper runtime data endpoints"
git push origin main
```

On PythonAnywhere:

```bash
cd /home/RasPatrick/jbravo_screener
git pull origin main
./build_frontend_pythonanywhere.sh
set -a; . ~/.config/jbravo/.env; set +a
PYTHONANYWHERE_DOMAIN="${PYTHONANYWHERE_DOMAIN:-${PYTHONANYWHERE_USERNAME}.pythonanywhere.com}"
curl -fsS -X POST \
  -H "Authorization: Token ${PYTHONANYWHERE_API_TOKEN}" \
  "https://www.pythonanywhere.com/api/v0/user/${PYTHONANYWHERE_USERNAME}/webapps/${PYTHONANYWHERE_DOMAIN}/reload/"
```
