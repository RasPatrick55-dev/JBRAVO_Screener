# Ops Runbook (Paper + DB-First)

## Invariants

- Paper-only broker mode: `APCA_API_BASE_URL` must point to the paper endpoint.
- DB-first operation: PostgreSQL data is authoritative for candidates, KPIs, and dashboard reads.
- CSV files are debug/parachute artifacts only and must not be treated as the system of record.

## Daily Commands

Run pipeline:

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.run_pipeline --steps screener,backtest,metrics,ranker_eval --reload-web true
```

Run one premarket execution pass:

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; bash bin/run_premarket_once.sh
```

Force web refresh (PythonAnywhere fallback):

```bash
cd /home/RasPatrick/jbravo_screener && touch /var/www/raspatrick_pythonanywhere_com_wsgi.py
```

## Pipeline Flags That Matter

- `--steps screener,backtest,metrics,ranker_eval` controls stage execution order.
- `--screener-args` forwards args directly into `scripts.screener`.
- `--backtest-args` forwards args directly into `scripts.backtest`.
- `--metrics-args` forwards args directly into `scripts.metrics`.
- `--ranker-eval-args` forwards args directly into `scripts.ranker_eval`.
- `--export-daily-bars-path` emits Stage 0 bars CSV for ML downstream jobs.

## Candidate Source Rules

- Primary reads: DB tables/views (`screener_candidates`, `top_candidates`, `latest_screener_candidates`, `latest_top_candidates`).
- `scripts.execute_trades` defaults to `--source db`.
- CSV candidate export is opt-in (`JBR_WRITE_CANDIDATE_CSVS=true`) for troubleshooting only.

## Verification Checklist

1. `PIPELINE_END rc=0` exists in `logs/pipeline.log`.
2. `reports/pipeline_summary.json` and `data/screener_metrics.json` were refreshed.
3. `reports/dashboard_findings.txt` and `reports/docs_findings.txt` exist and report no failures.
4. Dashboard Screener Health shows fresh `PIPELINE_*` and `DASH RELOAD` tokens.

## Failure Handling

1. Re-run docs and dashboard checks:
   `python -m scripts.docs_consistency_check && python -m scripts.dashboard_consistency_check`
2. If DB connectivity fails, stop trade execution and fix DB health first.
3. Use CSVs only for diagnosis; do not promote CSV data above DB outputs.

## Dashboard Consistency Checker Troubleshooting

1. CSV parity mismatch (`[PARITY] top_candidates.csv rows mismatch`):
   - In DB-first mode, stale `data/top_candidates.csv` can trigger legacy CSV parity checks.

   ```bash
   mkdir -p data/parachute
   mv data/top_candidates.csv data/parachute/top_candidates_$(date -u +%Y%m%dT%H%M%SZ).csv
   python -m scripts.dashboard_consistency_check
   ```

2. `DB_CONNECT_FAIL localhost:9999`:
   - `localhost:9999` implies DB config was not sourced (or a local tunnel default was used). On PythonAnywhere, source `~/.config/jbravo/.env` first.

   ```bash
   set -a; . ~/.config/jbravo/.env; set +a
   python -m scripts.dashboard_consistency_check
   ```

   Background: https://help.pythonanywhere.com/pages/AccessingPostgresFromOutsidePythonAnywhere/
