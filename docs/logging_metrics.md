# Logging Metrics Reference

This guide summarizes the event payloads emitted by the trading pipeline so you can interpret the JSON lines captured in `logs/execute_metrics.jsonl` (and other structured log sinks).

## Event Envelope

All metric events share a consistent envelope with the following fields:

- `ts` – event timestamp in ISO 8601 format.
- `run_id` – unique identifier of the pipeline run that produced the metric.
- `event` – event type string (for example `CANDIDATE_SKIPPED`).
- `symbol` – instrument ticker relevant to the event, when applicable.
- `reason` – machine readable status or exit code (see [Reason Codes](#reason-codes)).
- `details` – free-form JSON object containing event-specific context.
- `meta` – optional metadata such as account, strategy variant, or order identifiers.

Individual events may include additional keys inside the `details` object; these are illustrated in the examples below.

## Reason Codes

| Code | Description |
| --- | --- |
| `EXISTING_POSITION` | Candidate rejected because an open position already exists. |
| `PENDING_ORDER` | Candidate skipped due to an outstanding order for the same symbol. |
| `RISK_LIMIT` | Risk guardrails (exposure, capital, or concentration) prevented action. |
| `MARKET_DATA` | Required market data was missing or stale. |
| `SESSION_WINDOW` | Event occurred outside the active trading session window. |
| `DUPLICATE_CANDIDATE` | Identical candidate was processed previously in the same run. |
| `ALPACA_REJECT` | Broker rejected the submitted order. |
| `OTHER` | Miscellaneous or uncategorized reason. |
| `MAX_HOLD` | Position exited because the maximum holding period elapsed. |
| `TRAIL_FILLED` | Trailing stop order filled, closing the position. |
| `EMA20_CROSS` | Exit triggered by the EMA20 crossover rule. |
| `MANUAL` | Operator manually instructed the exit. |

## Event Examples

### `CANDIDATE_SKIPPED`

```json
{
  "ts": "2024-02-05T14:31:02.389Z",
  "run_id": "20240205-1430",
  "event": "CANDIDATE_SKIPPED",
  "symbol": "AAPL",
  "reason": "EXISTING_POSITION",
  "details": {
    "position_qty": 100,
    "note": "Already long from 2024-02-02 session"
  }
}
```

### `ORDER_SUBMIT`

```json
{
  "ts": "2024-02-05T14:32:11.023Z",
  "run_id": "20240205-1430",
  "event": "ORDER_SUBMIT",
  "symbol": "MSFT",
  "reason": "OTHER",
  "details": {
    "side": "buy",
    "qty": 50,
    "limit_price": 415.35
  },
  "meta": {
    "order_id": "4b9a9f8c-0e20-4f69-8ce8-5b21894a1da5"
  }
}
```

### `ORDER_FINAL`

```json
{
  "ts": "2024-02-05T14:32:12.812Z",
  "run_id": "20240205-1430",
  "event": "ORDER_FINAL",
  "symbol": "MSFT",
  "reason": "ALPACA_REJECT",
  "details": {
    "status": "rejected",
    "reject_code": "insufficient_margin",
    "submitted_at": "2024-02-05T14:32:11.023Z"
  },
  "meta": {
    "order_id": "4b9a9f8c-0e20-4f69-8ce8-5b21894a1da5"
  }
}
```

### `RETRY`

```json
{
  "ts": "2024-02-05T14:33:44.902Z",
  "run_id": "20240205-1430",
  "event": "RETRY",
  "symbol": "MSFT",
  "reason": "MARKET_DATA",
  "details": {
    "attempt": 2,
    "backoff_seconds": 30,
    "error": "VWAP unavailable"
  }
}
```

### `EXIT_SUBMIT`

```json
{
  "ts": "2024-02-07T19:55:00.500Z",
  "run_id": "20240207-1930",
  "event": "EXIT_SUBMIT",
  "symbol": "AAPL",
  "reason": "TRAIL_FILLED",
  "details": {
    "side": "sell",
    "qty": 100,
    "order_type": "trailing_stop",
    "trail_percent": 2.0
  },
  "meta": {
    "order_id": "5c70a0d9-6af4-4bb7-8848-df066c0c865e"
  }
}
```

### `EXIT_FINAL`

```json
{
  "ts": "2024-02-07T19:55:05.117Z",
  "run_id": "20240207-1930",
  "event": "EXIT_FINAL",
  "symbol": "AAPL",
  "reason": "TRAIL_FILLED",
  "details": {
    "status": "filled",
    "filled_avg_price": 189.42,
    "submitted_at": "2024-02-07T19:55:00.500Z"
  },
  "meta": {
    "order_id": "5c70a0d9-6af4-4bb7-8848-df066c0c865e"
  }
}
```

### `TRAILING_STOP_ATTACH`

```json
{
  "ts": "2024-02-07T19:30:10.204Z",
  "run_id": "20240207-1930",
  "event": "TRAILING_STOP_ATTACH",
  "symbol": "AAPL",
  "reason": "TRAIL_FILLED",
  "details": {
    "parent_order_id": "2c2eae73-6930-4598-a784-e6d64eac668d",
    "trail_percent": 2.0,
    "activation_price": 185.10
  }
}
```

## Backward Compatibility

No dashboard changes are required for this documentation update. The Plotly Dash app continues to rely on the existing keys in `execute_metrics.json` that are already documented in the system guide and pipeline summary.

