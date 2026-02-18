# Screener Tab: Runtime Data Integration

## Scope

This release rebuilds the Screener tab into four runtime sections:

- Screener Picks table
- Backtest Results table
- Metrics Results table
- Logs Board (Screener, Backtest, Metrics panels)

Frontend files added:

- `frontend/src/components/screener/ScreenerTab.tsx`
- `frontend/src/components/screener/ScreenerPicksCard.tsx`
- `frontend/src/components/screener/BacktestResultsCard.tsx`
- `frontend/src/components/screener/MetricsResultsCard.tsx`
- `frontend/src/components/screener/LogsBoard.tsx`
- `frontend/src/components/screener/LogsPanel.tsx`
- `frontend/src/components/screener/types.ts`
- `frontend/src/components/screener/utils.ts`

Frontend integration update:

- `frontend/src/screens/ScreenerOverview.tsx`

Backend update:

- `dashboards/dashboard_app.py`

## API Endpoints

### `GET /api/screener/picks?limit=50&q=&filter=all|top10|passed|errors`

Response shape:

```json
{
  "ok": true,
  "run_ts_utc": "2026-02-15T13:45:00Z",
  "status": "COMPLETE",
  "source": "postgres",
  "source_detail": "postgres:screener_candidates run_date=2026-02-15",
  "rows": [
    {
      "rank": 1,
      "symbol": "NVDA",
      "exchange": "NASDAQ",
      "screened_at_utc": "2026-02-15T13:44:31Z",
      "final_score": 92.4,
      "volume": 2543221,
      "dollar_volume": 1832124255.01,
      "price": 720.37,
      "sma_ema_pct": 2.08,
      "entry_price": 718.5,
      "adv20": 1980034,
      "atrp": 3.45
    }
  ]
}
```

Source behavior:

- Primary: Postgres (`screener_candidates`, latest run date; run timestamp scoped when map table is available)
- DB scope fallback: if run timestamp map scope returns no rows, query falls back to unscoped latest `run_date`
- File fallback chain (if DB unavailable/empty): `data/latest_candidates.csv` -> `data/predictions/latest.csv` -> latest `data/predictions/YYYY-MM-DD.csv`

### `GET /api/screener/backtest?run_ts_utc=<optional>&window=6M|3M|1Y|ALL&limit=50&q=`

Response shape:

```json
{
  "ok": true,
  "run_ts_utc": "2026-02-15T13:45:00Z",
  "window": "6M",
  "source": "postgres",
  "rows": [
    {
      "symbol": "NVDA",
      "window": "6M",
      "trades": 18,
      "win_rate_pct": 61.11,
      "avg_return_pct": 2.42,
      "pl_ratio": 1.67,
      "max_dd_pct": 8.94,
      "avg_hold_days": 6.2,
      "total_pl_usd": 7420.33
    }
  ]
}
```

Source behavior:

- Primary: Postgres (`backtest_results`)
- Fallback: `data/backtest_results.csv`

### `GET /api/screener/metrics?run_ts_utc=<optional>&filter=all|gate_failures|data_issues|high_confidence&limit=50&q=`

Response shape:

```json
{
  "ok": true,
  "run_ts_utc": "2026-02-15T13:45:00Z",
  "source": "postgres",
  "rows": [
    {
      "symbol": "NVDA",
      "score_breakdown_short": "Momentum:32 Vol:28 Trend:24",
      "liquidity_gate": "PASS",
      "volatility_gate": "PASS",
      "trend_gate": "PASS",
      "bars_complete": "YES",
      "confidence": "High",
      "source_label": "IEX"
    }
  ]
}
```

Source behavior:

- Primary: Postgres (`screener_candidates`, latest run scope)
- Fallback: `data/latest_candidates.csv`

### `GET /api/screener/logs?stage=screener|backtest|metrics&limit=200&level=all|errors|warnings&today=0|1&q=`

Response shape:

```json
{
  "ok": true,
  "stage": "screener",
  "source": "postgres",
  "source_detail": "postgres:screener_logs",
  "rows": [
    {
      "ts_utc": "2026-02-15T13:45:21Z",
      "level": "INFO",
      "message": "Screening completed for 142 symbols"
    }
  ]
}
```

Source behavior:

- Preferred: PythonAnywhere log files (API token + username required)
- Secondary fallback: local `logs/` files
- Final fallback: DB log-like tables when available (`<stage>_logs`, `pipeline_logs`, `log_events`, `app_logs`)
  - `logs/screener.log`
  - `logs/step.backtest.out`
  - `logs/step.metrics.out`
  - `logs/pipeline.log`

## Validation

```bash
python -m py_compile dashboards/dashboard_app.py
cd frontend && npm run build
```

## Curl Examples

```bash
curl -s "http://127.0.0.1:8050/api/screener/picks?limit=50&q=nvda&filter=all" | python -m json.tool
curl -s "http://127.0.0.1:8050/api/screener/backtest?window=6M&limit=50&q=nvda" | python -m json.tool
curl -s "http://127.0.0.1:8050/api/screener/metrics?filter=high_confidence&limit=50&q=nvda" | python -m json.tool
curl -s "http://127.0.0.1:8050/api/screener/logs?stage=screener&limit=200&level=all&today=0&q=" | python -m json.tool
```

PowerShell examples:

```powershell
irm "http://127.0.0.1:8050/api/screener/picks?limit=50&filter=all" | ConvertTo-Json -Depth 6
irm "http://127.0.0.1:8050/api/screener/backtest?window=3M&limit=50" | ConvertTo-Json -Depth 6
irm "http://127.0.0.1:8050/api/screener/metrics?filter=gate_failures&limit=50" | ConvertTo-Json -Depth 6
irm "http://127.0.0.1:8050/api/screener/logs?stage=metrics&limit=200&level=warnings&today=1" | ConvertTo-Json -Depth 6
```
