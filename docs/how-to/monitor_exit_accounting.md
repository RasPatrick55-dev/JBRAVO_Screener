# Monitor exit accounting

After an exit attempt, the monitor fetches positions again before stop coverage
or its heartbeat uses that position snapshot. A failed refresh fails the cycle;
it cannot certify an empty portfolio or attach stops from the old snapshot.
A one-share profit-target sale is a full exit, with no partial-exit flag.

A terminal filled status triggers a read of that exact broker order. Verified
exports require matching order ID, symbol, sell side and filled quantity, a
positive finite average fill price and an aware broker fill timestamp.
The submitted limit and polling time remain observations, never execution facts.
Unavailable or inconsistent fill details produce a filled_unverified CSV record
with blank exit price/time and no executed-trade database export.
Unknown entry time also prevents an executed-trade export, whose database writer
would otherwise substitute the current time. Verified CSV exits deduplicate by
broker order ID.

The monitor records SELL_FILL_OBSERVED; canonical trade closure and SELL_FILL
ownership remain with reconciliation. This change does not repair old database
or CSV records, alter entry eligibility, or qualify research.

Offline verification uses mocked broker/database boundaries and temporary files:

```powershell
python -m unittest discover -s tests -p test_monitor_task_log_regressions.py
python -m unittest discover -s tests -p test_monitor_positions_db_logging.py
python -m unittest discover -s tests -p test_monitor_reconciliation_boundary.py
```

Deployment must coordinate any existing research enrollment that pins the changed
source bytes. Preserve immutable old registration and claims; never edit them to
conceal a source change.
