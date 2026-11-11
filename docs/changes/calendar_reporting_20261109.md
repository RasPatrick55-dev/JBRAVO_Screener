# Trades calendar reporting periods

The combined Trades report uses the current calendar week and month in
America/New_York. Weekly includes the actual open US equity trading dates from
Monday through Friday. Monthly includes the first through last open trading date
of the current calendar month. Alpaca's unpaginated calendar response supplies
open dates, including holidays and early closes; no weekday-only holiday guess is
used. The UI shows those dates and the period-to-date observation cutoff.

Eligible closed trades are assigned by their exit timestamp's New York date,
including premarket and after-hours completions on an open date. Future executions
are excluded. A new week/month never borrows results from the previous period.
Totals, counts, gross win rates and symbol rankings share the same filter.

The calendar read is bounded and read-only. If unavailable or invalid, weekly and
monthly results become explicitly unknown. Valid history and unchanged daily,
yearly and all-time reporting remain visible. Fee/net accounting holds and legacy
account-association limitations remain unchanged. This reporting change does not
change execution eligibility, orders, monitors or trading strategy.

Offline tests cover holidays, early closes, weekends, DST, year/month boundaries,
leap years, exact observation cutoffs and calendar failure. Actual provider/native
acceptance and deployment are separate verification steps.
