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

The saved position frame and in-memory closure snapshot share the same continuity
rules. Entry price always comes from the current broker basis. A prior entry
timestamp is retained only when basis and position side still match; current
aware entry timestamps take precedence, while ambiguous timestamps stay unknown.
This prevents a changed basis from borrowing an earlier holding's price or time.

A partial exit supplies a receipt from the existing exact-order fill readback.
Only verified filled quantity reduces the unrecorded remainder. The partial
order ID is excluded from later residual-fill matching, even when order-history
responses are reordered or the residual fill is not yet visible. Unknown partial
quantity stays blank in a subsequent unverified closure record, rather than
exporting the original full position again. Terminal-status strategy behavior
is unchanged; an accounting receipt does not authorize another order.

After a partial exit, the remaining quantity and `accounted_exit_order_ids` are
atomically checkpointed in `open_positions.csv` before the final broker refresh.
They survive a failed refresh and carry across continuing-position snapshots, so
the next cycle cannot reuse the partial fill as a residual close. A checkpoint
failure fails the cycle and preserves the previous file. This is not a transaction
across both CSV exports or the broker: a process/disk failure between recording
the fill and checkpointing still needs independent reconciliation. Basis/side
continuity alone cannot prove an unobserved close-and-reopen at identical basis.
Canonical broker reconciliation remains the owner of trade history.

The monitor records SELL_FILL_OBSERVED; canonical trade closure and SELL_FILL
ownership remain with reconciliation. This change does not repair old database
or CSV records, alter entry eligibility, or qualify research.

Offline verification uses mocked broker/database boundaries and temporary files:

```powershell
python -m unittest discover -s tests -p test_monitor_task_log_regressions.py
python -m unittest tests.test_monitor_closure_followup
python -m unittest discover -s tests -p test_monitor_positions_db_logging.py
python -m unittest discover -s tests -p test_monitor_reconciliation_boundary.py
```

Deployment must coordinate any existing research enrollment that pins the changed
source bytes. Preserve immutable old registration and claims; never edit them to
conceal a source change.
