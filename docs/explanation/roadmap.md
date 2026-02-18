# Pipeline Roadmap

This roadmap reflects current DB-first, paper-only operation.

## Current

- Paper-only execution guardrails are enabled.
- Pipeline defaults to `screener,backtest,metrics,ranker_eval`.
- Candidate truth is in DB views, not CSV files.
- Docs and dashboard consistency checks are part of daily ops.

## Next

1. Keep CLI docs auto-generated from `--help` output only.
2. Add CI job that runs docs/dashboard consistency checks on each docs or scripts change.
3. Continue reducing legacy docs that imply file-first or live-trading behavior.
