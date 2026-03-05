# Ops Runbook (Paper + DB-First)

## Invariants

- Paper-only broker mode: `APCA_API_BASE_URL` must point to the paper endpoint.
- DB-first operation: PostgreSQL data is authoritative for candidates, KPIs, and dashboard reads.
- CSV files are debug/parachute artifacts only and must not be treated as the system of record.
- DB config precedence: `DATABASE_URL` -> `DB_*` -> disabled unless `JBR_DEV_DB_DEFAULTS=true`.

## DB Connection Playbook (Local VS Code + PythonAnywhere)

Use this section when you see `DB_CONNECT_FAIL` or `DB_MIGRATE_FAILED`.

### Local VS Code (Windows) -> PythonAnywhere Postgres via SSH tunnel

1. Start the SSH tunnel in a dedicated terminal and keep it open:

```powershell
ssh -4 -N -L 9999:RasPatrick-4996.postgres.pythonanywhere-services.com:14996 RasPatrick@ssh.pythonanywhere.com
```

2. In your project shell, load DB env and verify config:

```powershell
cd C:\Users\RasPa\JBravoGit\JBRAVO_Screener
$envFile = Join-Path $HOME ".config/jbravo/.env"
Get-Content $envFile | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $parts = $_ -split '=', 2
  if ($parts.Count -eq 2) {
    Set-Item -Path ("Env:" + $parts[0].Trim()) -Value ($parts[1].Trim().Trim('"'))
  }
}
python -c "from scripts.utils.env import load_env; load_env(); from scripts import db; print(db.db_config_preview())"
```

If you are in Git Bash/WSL instead of PowerShell, use:

```bash
set -a; . ~/.config/jbravo/.env; set +a
```

Good output:
- `enabled: True`
- source should normally be `DATABASE_URL` (preferred), not dev fallback.

3. Verify tunnel + DB reachability:

```powershell
Test-NetConnection -ComputerName 127.0.0.1 -Port 9999
python -c "from scripts.utils.env import load_env; load_env(); from scripts import db; print('db_enabled=', db.db_enabled()); print('db_connect_test=', db.check_db_connection())"
```

Good output:
- `TcpTestSucceeded : True`
- `db_enabled= True`
- `db_connect_test= True`

4. Run bounded DB-first smoke:

```powershell
python -m scripts.run_pipeline --steps screener,labels,ranker_eval --reload-web false --screener-args "--limit 30"
```

Good output:
- no `DB_CONNECT_FAIL`
- no `DB_MIGRATE_FAILED`
- `[INFO] PIPELINE_END rc=0`

### PythonAnywhere runtime shell (production-like)

Use this when running from PythonAnywhere itself (no local tunnel needed):

```bash
cd /home/RasPatrick/jbravo_screener
source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate
set -a; . ~/.config/jbravo/.env; set +a
python -c "from scripts import db; print(db.db_config_preview()); print('db_connect_test=', db.check_db_connection())"
python -m scripts.run_pipeline --steps screener,labels,ranker_eval --reload-web false --screener-args "--limit 30"
```

Good output:
- `db_connect_test= True`
- `[INFO] PIPELINE_END rc=0`

### Common failure signatures and fixes

1. `DB_CONNECT_FAIL ... localhost:9999 ... connection refused`
- Tunnel is down or bound to a different port.
- Restart the SSH tunnel and rerun `Test-NetConnection`.

2. `DB_REQUIRED: DATABASE_URL/DB_* not configured`
- Env was not loaded in the current shell.
- Reload env for your shell (PowerShell loader above, or Bash `set -a; . ~/.config/jbravo/.env; set +a`) and retry.

3. Source unexpectedly shows `JBR_DEV_DB_DEFAULTS`
- Disable local dev fallback for ops runs:
  - unset `JBR_DEV_DB_DEFAULTS`
  - ensure `DATABASE_URL` is present.

4. `DB_MIGRATE_FAILED` immediately after connect failures
- This is usually downstream from connection issues.
- Fix reachability first (`db.check_db_connection() == True`), then rerun pipeline.

## Daily Commands

Run pipeline:

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.run_pipeline --steps screener,backtest,metrics,ranker_eval --reload-web true
```

Run one premarket execution pass:

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; bash bin/run_premarket_once.sh
```

Force web refresh (PythonAnywhere API preferred):

```bash
set -a; . ~/.config/jbravo/.env; set +a
PYTHONANYWHERE_DOMAIN="${PYTHONANYWHERE_DOMAIN:-${PYTHONANYWHERE_USERNAME}.pythonanywhere.com}"
curl -fsS -X POST \
  -H "Authorization: Token ${PYTHONANYWHERE_API_TOKEN}" \
  "https://www.pythonanywhere.com/api/v0/user/${PYTHONANYWHERE_USERNAME}/webapps/${PYTHONANYWHERE_DOMAIN}/reload/"
```

## Pipeline Flags That Matter

- `--steps screener,backtest,metrics,ranker_eval` controls stage execution order.
- `--screener-args` forwards args directly into `scripts.screener`.
- `--backtest-args` forwards args directly into `scripts.backtest`.
- `--metrics-args` forwards args directly into `scripts.metrics`.
- `--ranker-eval-args` forwards args directly into `scripts.ranker_eval`.
- `--allow-no-screener` (opt-in) allows pipeline runs that exclude `screener`.
  - default remains strict: missing `screener` still fails with `SCREENER_STEP_REQUIRED`
    unless this flag (or env `JBR_ALLOW_NO_SCREENER=true`) is set.
  - intended for ML-only maintenance/evaluation runs (for example labels +
    recalibration).
- `--auto-refresh-predictions` (opt-in) enables stale-prediction auto-refresh
  before prediction-consuming steps (eval/enrichment/monitor checks).
  - env fallback: `JBR_AUTO_REFRESH_PREDICTIONS=true|false`
  - default is warn-only (no automatic rerun).
- `--export-daily-bars-path` emits Stage 0 bars CSV for ML downstream jobs.
- `--enrich-candidates-with-ranker` (opt-in) overlays latest ranker `score_5d`
  into candidates as `model_score_5d`.
  - DB-first mode: writes to `screener_ranker_scores_app` and logs:
    - `[INFO] CANDIDATES_ENRICHED destination=db table=screener_ranker_scores_app rows=<n> run_ts_utc=<...>`
  - CSV fallback mode: enriches `data/latest_candidates.csv` with
    `model_score_5d` (debug/parachute path only).
- `--use-champion` (opt-in) loads the latest `ranker_champion` config and
  applies champion-derived ML env overrides to ML analysis steps only:
  - `labels`, `features`, `ranker_train`, `ranker_predict`, `ranker_eval`,
    `ranker_walkforward`, `ranker_strategy_eval`
  - screener/backtest/metrics remain unchanged
  - logs:
    - `[INFO] CHAMPION_LOAD source=<db|fs|none> present=<true|false> run_date=<...>`
    - `[INFO] CHAMPION_APPLIED mode=<fill|force> keys=<comma_list>`
    - `[WARN] CHAMPION_MISSING` (only when enabled and not found)
- `--champion-mode {fill,force}` controls env precedence when champion is enabled:
  - `fill` (default): set only missing env keys
  - `force`: overwrite existing env keys for mapped champion settings
- `JBR_USE_CHAMPION=true` enables champion mode when CLI `--use-champion` is
  not provided.
- `--ml-health-guard` (opt-in) enables enrichment guardrails based on the
  latest `ranker_monitor` artifact.
- `--ml-health-guard-mode {warn,block}` controls behavior when monitor action
  is not `none`:
  - `warn` (default): enrichment continues with warning
  - `block`: ranker enrichment is skipped for that run
- `--ranker-autoremediate-args` forwards args into
  `scripts.ranker_autoremediate` when `ranker_autoremediate` is included in
  `--steps`.
- `--ranker-autoremediate-args-split` supports tokenized forwarding for the
  same step.
- Env precedence follows CLI > env > default:
  - `JBR_ML_HEALTH_GUARD=true|false`
  - `JBR_ML_HEALTH_GUARD_MODE=warn|block`
  - `JBR_ML_HEALTH_SOURCE=auto|db|fs` (default `auto`)
  - `JBR_ML_HEALTH_PATH=<path>` (default `data/ranker_monitor/latest.json`)
  - `JBR_ML_HEALTH_MAX_AGE_DAYS=<int>` (default `7`)
- Optional alert fan-out:
  - `JBR_ML_HEALTH_ALERTS=true` sends best-effort alert when monitor action is
    not `none` (alert failures are warn-only).
  - Auto-remediation trial default can be set via
    `JBR_RANKER_AUTOREMEDIATE_TRIALS=<int>` (script default is `10`).

Expected guard tokens:

- `[INFO] ML_HEALTH_GUARD enabled=<true|false> mode=<warn|block>`
- `[INFO] ML_HEALTH_LOAD source=<db|fs|missing> present=<true|false> run_date=<...>`
- `[INFO] ML_HEALTH_STATUS action=<...> psi_score=<...> recent_sharpe=<...>`
- `[INFO] ML_ENRICHMENT_DECISION decision=<allow|warn|block> mode=<warn|block> action=<...> reason=<...> psi_score=<...> recent_sharpe=<...> monitor_run_date=<...>`
- `[WARN] ML_HEALTH_MISSING reason=no_ranker_monitor_artifact` (if no monitor artifact)
- `[WARN] ML_ENRICHMENT_WARN reason=<...>` (warn mode)
- `[WARN] ML_ENRICHMENT_BLOCKED reason=<...>` (block mode)
- freshness-aware guard reason (when predictions are stale):
  - `reason=...stale_predictions...`
  - `warn` mode continues enrichment; `block` mode skips enrichment.

## Feed And Timeout Controls

- `ALPACA_DATA_FEED` defaults to `iex` when not set.
- `JBR_BARS_ADJUSTMENT` controls Alpaca bars corporate-action adjustment
  (`raw|split|dividend|all`, default `raw`).
  - CLI override: `scripts.screener --bars-adjustment <value>`
  - Precedence: CLI > `JBR_BARS_ADJUSTMENT` > default `raw`
  - Screener logs:
    - `[INFO] BARS_ADJUSTMENT value=<...> source=<cli|env|default>`
  - Use `split` or `all` for ML evaluation stability when raw bars produce
    split-driven forward-return outliers.
- `JBR_SPLIT_ADJUST` controls label-generation fallback split adjustment
  (`off|auto|force`, default `off`).
  - CLI override: `scripts.label_generator --split-adjust <value>`
  - Precedence: CLI > `JBR_SPLIT_ADJUST` > default `off`
  - In `auto` mode, split-like discontinuities are detected and `close_adj`
    is used for `fwd_ret_*` / `label_*` generation only when needed.
- `JBR_STRICT_FWD_RET` controls strict forward-return sanity enforcement for
  labels (default `false`).
  - CLI override: `scripts.label_generator --strict-fwd-ret`
  - When enabled, severe forward-return outliers fail label generation with
    rc=2 for research/hardening runs.
- `sip` feed access requires the appropriate Alpaca subscription. If a `sip`
  bars request receives `401/403`, the screener now retries the same request on
  `iex` and logs:
  - `[WARN] ALPACA_FEED_FALLBACK from=sip to=iex reason=unauthorized endpoint=/v2/stocks/bars`
- `JBR_FEATURES_TIMEOUT_SECS` controls only the `feature_generator` timeout in
  `scripts.run_pipeline` (default: `900` seconds). Increase this if feature
  generation regularly exceeds 5 minutes on your environment.
- `JBR_RANKER_PREDICT_TIMEOUT_SECS` controls only the `ranker_predict` timeout
  in `scripts.run_pipeline` (default: `900` seconds).
  - timeout token:
    - `[INFO] RANKER_PREDICT_TIMEOUT secs=<value> source=<env|default>`
  - failure token:
    - `[WARN] RANKER_PREDICT_FAILED rc=<rc> timeout_secs=<value> predictions_source=<db|fs>:<missing|present>`
  - timeout symptom:
    - `STEP_TIMEOUT name=ranker_predict timeout=<value>`
    - absence of this token is the expected healthy outcome.
- prediction freshness enforcement (DB-first):
  - strict provenance mode (env-only):
    - `JBR_STRICT_PREDICTIONS_META=true|false` (default false)
    - token:
      - `[INFO] STRICT_PREDICTIONS_META enabled=<true|false> source=<env|default>`
  - pipeline emits:
    - `[INFO] PREDICTIONS_FRESHNESS stale=<true|false> reason=<...> model_path=... pred_model_path=... latest_features_set=... latest_features_signature=... pred_features_set=... pred_features_signature=... pred_compatible=<true|false|unknown> pred_missing_frac=<...> pred_compat_reason=<...>`
      - stale can now be triggered by model drift OR feature-schema drift between latest features and the feature schema recorded in predictions metadata.
      - stale also triggers when predictions metadata records an explicit incompatible feature verdict (`reason` includes `pred_feature_incompatible`).
      - with strict provenance enabled, stale also triggers when provenance is incomplete:
        `pred_feature_compat_missing`, `pred_feature_set_missing`,
        `pred_feature_signature_missing`, `pred_model_meta_missing`.
  - if stale and auto-refresh disabled:
    - `[WARN] PREDICTIONS_STALE reason=<...> suggestion=run ranker_predict`
    - with `--ml-health-guard` enabled, stale predictions are also treated as
      unhealthy reason `stale_predictions` for enrichment decisioning:
      - guard `warn`: `ML_ENRICHMENT_WARN ... reason=...stale_predictions...`
      - guard `block`: `ML_ENRICHMENT_BLOCKED ... reason=...stale_predictions...`
  - if stale and auto-refresh enabled:
    - `[INFO] AUTO_REFRESH_PREDICTIONS enabled=true stale=true -> running ranker_predict`
    - optional strict mode via env: `JBR_STRICT_AUTO_REFRESH_PREDICTIONS=true`
      - emits: `[INFO] STRICT_AUTO_REFRESH_PREDICTIONS enabled=true max_missing_feature_fraction=0.2`
      - and forwards strict guards to `ranker_predict`:
        - `--strict-feature-match true`
        - `--max-missing-feature-fraction 0.2`
    - `[INFO] AUTO_REFRESH_PREDICTIONS_DONE rc=<rc>`
    - if refresh still leaves incompatible predictions:
      - `[WARN] AUTO_REFRESH_PREDICTIONS_INEFFECTIVE reason=<...> suggestion=enable_strict_auto_refresh_or_refresh_features`
  - prediction writes emit provenance token:
    - `[INFO] PREDICTIONS_META_WRITTEN source=<db|fs> model_path=... model_mtime_utc=... calibrated=... method=... feature_set=... feature_signature=... feature_meta_source=... compatible=<true|false> missing_frac=<...> compat_reason=<...>`
- `JBR_RANKER_EVAL_TIMEOUT_SECS` controls only the `ranker_eval` timeout in
  `scripts.run_pipeline` (default: `180` seconds).
  - timeout token:
    - `[INFO] RANKER_EVAL_TIMEOUT secs=<value> source=<env|default>`
  - failure token:
    - `[WARN] RANKER_EVAL_FAILED rc=<rc> timeout_secs=<value> eval_source=<db|fs>:<missing|present>`
  - timeout symptom:
    - `STEP_TIMEOUT name=ranker_eval timeout=<value>`
    - absence of this token is the expected healthy outcome.
- `JBR_ML_FEATURE_SET` controls ML feature schema for `scripts.feature_generator`
  (`v1` or `v2`, default `v1`). `v2` keeps legacy columns and adds additional
  time-safe indicator/pattern numeric features.
  - feature metadata token:
    - `[INFO] FEATURES_META_WRITTEN source=<db|fs> feature_set=<v1|v2> feature_signature=<...> rows=<n> path=<...> feature_count=<n>`
  - feature metadata FS sidecars:
    - per-file sidecar: `data/features/features_YYYY-MM-DD.meta.json`
    - convenience pointer: `data/features/latest_meta.json`
- `JBR_ML_CALIBRATION` controls optional probability calibration in
  `scripts.ranker_train` (`none|sigmoid|isotonic`, default `none`).
  - CLI override: `python -m scripts.ranker_train --calibrate <method>`
  - precedence: CLI > `JBR_ML_CALIBRATION` > default `none`
  - train token:
    - `[INFO] RANKER_TRAIN_CALIBRATION method=<none|sigmoid|isotonic> calib_rows=<n> embargo_days=<d>`
  - predict token:
    - `[INFO] RANKER_PREDICT_SCORE_SOURCE calibrated=<true|false> method=<none|sigmoid|isotonic>`
  - predict compatibility token:
    - `[INFO] RANKER_PREDICT_FEATURE_COMPAT ...`
    - `compatible=false` is now truthful for any schema incompatibility,
      including `feature_signature_mismatch`, `feature_set_mismatch`, or
      `missing_feature_fraction_exceeded` (based on
      `--max-missing-feature-fraction`).
  - strict mismatch fatal token:
    - `[ERROR] RANKER_PREDICT_FEATURE_MISMATCH_FATAL ...` (returns rc=2)
  - eval reliability token (test-only):
    - `[INFO] RANKER_EVAL_CALIBRATION bins=<n> ece=<...> mce=<...> score_min=<...> score_max=<...>`
  - eval skip token:
    - `[WARN] RANKER_EVAL_CALIBRATION_SKIPPED reason=<...> score_min=<...> score_max=<...>`
  - `isotonic` can overfit on small calibration windows; prefer `sigmoid`
    unless calibration sample size is large.

Recommended ML-evaluation settings (DB-enabled, paper-only):

- `JBR_BARS_ADJUSTMENT=split` (or `all`) for bars fetches.
- Optional safety net: `JBR_SPLIT_ADJUST=auto` during label generation.
- Optional probability calibration for model-score thresholds:
  - `JBR_ML_CALIBRATION=sigmoid`

Calibration smoke (paper-safe):

```bash
JBR_ML_CALIBRATION=sigmoid python -m scripts.ranker_train --target label_5d_pos_300bp
python -m scripts.ranker_predict
```

Recalibrate-only remediation smoke (paper-safe):

```bash
DB_DISABLED=1 python -m scripts.ranker_recalibrate --target label_5d_pos_300bp --calibrate sigmoid
```

Expected tokens:

- `[INFO] RANKER_RECALIBRATE_START ...`
- `[INFO] RANKER_RECALIBRATE_FIT ...`
- `[INFO] RANKER_RECALIBRATE_END model_path=...`

Optional pipeline integration (opt-in, default pipeline unchanged):

```bash
DB_DISABLED=1 python -m scripts.run_pipeline --steps labels,ranker_recalibrate --reload-web false
```

If labels require fresh bars in your environment, run bounded screener first:

```bash
DB_DISABLED=1 python -m scripts.run_pipeline --steps screener,labels,ranker_recalibrate --reload-web false --screener-args "--limit 30"
```

Good output tokens:

- `[INFO] START ranker_recalibrate ...`
- `[INFO] RANKER_RECALIBRATE_START ...`
- `[INFO] RANKER_RECALIBRATE_FIT ...`
- `[INFO] RANKER_RECALIBRATE_END model_path=...`
- `[INFO] RANKER_RECALIBRATE rc=0 method=<sigmoid|isotonic> model_path=...`
- `[INFO] PIPELINE_END rc=0`

ML-only pipeline smoke (DB-enabled, no screener):

```bash
python -m scripts.run_pipeline --steps labels,ranker_recalibrate --allow-no-screener --reload-web false
```

Good output tokens:

- `[INFO] ALLOW_NO_SCREENER enabled=true ...`
- `[INFO] PIPELINE_START ...`
- `[INFO] START labels ...`
- `[INFO] START ranker_recalibrate ...`
- `[INFO] RANKER_RECALIBRATE rc=0 method=<sigmoid|isotonic> model_path=...`
- `[INFO] PIPELINE_END rc=0`

Recalibrate + refresh predictions (ML-only, no screener):

```bash
python -m scripts.run_pipeline --steps labels,ranker_recalibrate,ranker_predict --allow-no-screener --reload-web false
```

Good output tokens:

- `[INFO] ALLOW_NO_SCREENER enabled=true ...`
- `[INFO] START labels ...`
- `[INFO] START ranker_recalibrate ...`
- `[INFO] RANKER_RECALIBRATE rc=0 ...`
- `[INFO] START ranker_predict ...`
- `[INFO] RANKER_PREDICT rc=0 calibrated=<true|false> method=<...> predictions_source=<db|fs>:<present|missing>`
- `[INFO] PIPELINE_END rc=0`

Prediction freshness smoke (DB-enabled, bounded):

```bash
python -m scripts.run_pipeline --steps screener,labels,ranker_eval --reload-web false --enrich-candidates-with-ranker --use-champion --screener-args "--limit 30"
```

Good output tokens:

- `[INFO] PREDICTIONS_FRESHNESS stale=<true|false> ...`
- if stale with default behavior:
  - `[WARN] PREDICTIONS_STALE ... suggestion=run ranker_predict`
- if stale with auto refresh enabled:
  - `[INFO] AUTO_REFRESH_PREDICTIONS enabled=true stale=true -> running ranker_predict`
  - optional strict env:
    - `[INFO] STRICT_AUTO_REFRESH_PREDICTIONS enabled=true max_missing_feature_fraction=0.2`
  - `[INFO] AUTO_REFRESH_PREDICTIONS_DONE rc=0`
  - if auto-refresh is ineffective due to continued incompatibility:
    - `[WARN] AUTO_REFRESH_PREDICTIONS_INEFFECTIVE reason=pred_feature_incompatible suggestion=enable_strict_auto_refresh_or_refresh_features`
- feature/model compatibility refresh (opt-in, runs before auto repredict):
  - enable with `--auto-refresh-features true` or `JBR_AUTO_REFRESH_FEATURES=true`
  - compatibility token:
    - `[INFO] FEATURES_FRESHNESS stale=<true|false> reason=<...> model_feature_set=... features_feature_set=... model_feature_signature=... features_feature_signature=...`
  - refresh tokens:
    - `[INFO] AUTO_REFRESH_FEATURES enabled=true stale=true -> running feature_generator feature_set=<model_feature_set|env/default>`
    - `[INFO] AUTO_REFRESH_FEATURES_DONE rc=<rc>`
  - if labels are missing for refresh:
    - `[INFO] AUTO_REFRESH_FEATURES enabled=true labels_missing=true -> running labels`

Recommended stale-fix flow after recalibration/retrain:

- `ranker_recalibrate` (or autoremediate retrain path)
- feature refresh (`--auto-refresh-features true`)
- repredict (`--auto-refresh-predictions true`)
- optional `ranker_eval` / `ranker_monitor`

Fresh-but-incompatible predictions troubleshooting:

- Symptom:
  - `PREDICTIONS_FRESHNESS ... reason=pred_feature_incompatible ...`
  - `pred_compatible=false` and high `pred_missing_frac`.
- Root cause:
  - predictions were regenerated, but feature schema remained incompatible with
    the selected model.
- Recommended operator flags:
  - `JBR_AUTO_REFRESH_FEATURES=true`
  - `JBR_AUTO_REFRESH_PREDICTIONS=true`
  - `JBR_STRICT_AUTO_REFRESH_PREDICTIONS=true`
- This combination forces strict compatibility checks during auto repredict and
  prevents treating incompatible predictions as healthy.

Strict predictions provenance troubleshooting:

- Goal:
  - avoid treating "fresh-but-unknown" predictions metadata as healthy.
- Enable:
  - `JBR_STRICT_PREDICTIONS_META=true`
- Typical symptom when old predictions payloads are missing compatibility block:
  - `PREDICTIONS_FRESHNESS ... reason=pred_feature_compat_missing,...`
- Recovery flow:
  - set `JBR_AUTO_REFRESH_PREDICTIONS=true`
  - optionally set `JBR_STRICT_AUTO_REFRESH_PREDICTIONS=true`
  - rerun pipeline so `ranker_predict` rewrites DB/FS provenance with
    `feature_compat`.

Calibration diagnostics smoke (paper-safe, FS fallback):

```bash
DB_DISABLED=1 python -m scripts.ranker_eval --target label_5d_pos_300bp --calibration-bins 10
```

Token grep quick check:

```bash
grep -R "RANKER_EVAL_CALIBRATION" -n logs | head
```

## Champion Workflow (Opt-In)

1. Generate a champion from OOS tune-vs-holdout autotune (DB-first):

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.ranker_autotune --target label_5d_pos_300bp --trials 20 --holdout-days 252
```

2. Run the pipeline with champion applied to ML steps:

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.run_pipeline --steps screener,labels,ranker_eval --reload-web false --enrich-candidates-with-ranker --use-champion --screener-args "--limit 30"
```

3. Verify logs/artifacts:
- `CHAMPION_LOAD` and `CHAMPION_APPLIED` tokens present.
- `PIPELINE_END rc=0`.
- Existing DB enrichment token still present when enabled:
  `CANDIDATES_ENRICHED destination=db ...`.
- Score coverage token after enrichment:
  `MODEL_SCORE_COVERAGE total=... non_null=... pct=...`.
- Join diagnostics token (enrichment + DB candidate read):
  `MODEL_SCORE_JOIN_DIAG candidates=... scores_rows_for_run=... joined_non_null=... joined_null=... run_ts_utc=... score_col=...`.

Score-coverage troubleshooting when `pct=0`:

- `scores_rows_for_run=0`: predictions are missing/stale for that run; rerun ranker
  predict/eval chain and enrichment.
- `scores_rows_for_run>0` but `joined_non_null=0`: symbol/join-key mismatch or run timestamp mismatch.
- Sample unmatched token helps isolate failures:
  `MODEL_SCORE_JOIN_SAMPLE_UNMATCHED symbols=[...] reason=<...>`.
- If enrichment run timestamp is missing in candidates, the pipeline warns with:
  `MODEL_SCORE_ENRICH_RUN_TS_MISSING run_ts_utc=...`.

4. Optional execution-threshold dry-run validation:

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.execute_trades --source db --use-champion-execution true --alloc-weight-key model_score --diagnostic --dry-run true --ignore-market-gate true
```

Expected execution tokens:

- `EXEC_CHAMPION_LOAD ...`
- `EXEC_CHAMPION_APPLIED ... mode=fill`
- `MODEL_SCORE_COVERAGE_EXEC ...`
- `MODEL_SCORE_GATE ...`

ML health action guide (evaluation-only):

- `none`: proceed normally.
- `recalibrate`: review OOS calibration diagnostics (`ECE`/`MCE`) and refresh
  model calibration before changing execution thresholds.
- `retrain`: investigate model drift/performance and run
  `scripts.ranker_autotune`; use `--ml-health-guard-mode block` to prevent
  enrichment overlays during remediation.
- monitor staleness is also treated as unhealthy:
  - if `pipeline_run_date - monitor_run_date > JBR_ML_HEALTH_MAX_AGE_DAYS`,
    decision is `warn` or `block` based on guard mode.

Run monitor with calibration drift diagnostics:

```bash
python -m scripts.ranker_monitor --target label_5d_pos_300bp --recent-days 63 --baseline-days 252 --calibration-bins 10 --calibration-min-rows 2000 --calibration-ece-warn 0.05 --calibration-ece-alert 0.10
```

Expected tokens:

- `RANKER_MONITOR_CALIBRATION ...` when applicable
- or `RANKER_MONITOR_CALIBRATION_SKIPPED reason=...` when skipped
- `RANKER_MONITOR_END ... recommended_action=...`

Deterministic fixture test (no secrets; test-only artifact):

- Fixture path: `data/ranker_monitor/fixture_unhealthy.json`
- Force FS guard input and block mode:

```bash
JBR_ML_HEALTH_SOURCE=fs JBR_ML_HEALTH_PATH=data/ranker_monitor/fixture_unhealthy.json python -m scripts.run_pipeline --steps screener,labels,ranker_eval --reload-web false --enrich-candidates-with-ranker --ml-health-guard --ml-health-guard-mode block --screener-args "--limit 30"
```

Expected output for fixture test:

- `ML_ENRICHMENT_DECISION decision=block ...`
- `ML_ENRICHMENT_BLOCKED ...`
- no `CANDIDATES_ENRICHED destination=db ...` line for that run.

Deterministic recalibration fixture test (fresh, non-stale monitor payload):

- Fixture path: `data/ranker_monitor/fixture_recalibrate.json`
- Block mode (should block enrichment with `action_recalibrate`):

```bash
JBR_ML_HEALTH_SOURCE=fs JBR_ML_HEALTH_PATH=data/ranker_monitor/fixture_recalibrate.json python -m scripts.run_pipeline --steps screener,labels,ranker_eval --reload-web false --enrich-candidates-with-ranker --ml-health-guard --ml-health-guard-mode block --screener-args "--limit 30"
```

- Warn mode (should warn and continue enrichment):

```bash
JBR_ML_HEALTH_SOURCE=fs JBR_ML_HEALTH_PATH=data/ranker_monitor/fixture_recalibrate.json python -m scripts.run_pipeline --steps screener,labels,ranker_eval --reload-web false --enrich-candidates-with-ranker --ml-health-guard --ml-health-guard-mode warn --screener-args "--limit 30"
```

Expected tokens:

- `ML_HEALTH_LOAD source=fs present=true run_date=2026-02-28`
- `ML_HEALTH_STATUS action=recalibrate ...`
- `ML_ENRICHMENT_DECISION ... reason=action_recalibrate ...`
- `ML_ENRICHMENT_BLOCKED ...` in block mode or `ML_ENRICHMENT_WARN ...` in warn mode

## Auto-Remediation Workflow (Opt-In, Evaluation-Only)

Use this when monitor health indicates drift/performance degradation and you
want a bounded re-tune trigger.

Dry-run decision only (deterministic fixture override):

```bash
JBR_ML_HEALTH_SOURCE=fs JBR_ML_HEALTH_PATH=data/ranker_monitor/fixture_recalibrate.json python -m scripts.ranker_autoremediate --target label_5d_pos_300bp --trials 3 --dry-run true
```

Execute remediation when unhealthy:

```bash
JBR_ML_HEALTH_SOURCE=fs JBR_ML_HEALTH_PATH=data/ranker_monitor/fixture_recalibrate.json python -m scripts.ranker_autoremediate --target label_5d_pos_300bp --trials 3
```

Execute remediation and refresh predictions in the same run (opt-in):

```bash
JBR_ML_HEALTH_SOURCE=fs JBR_ML_HEALTH_PATH=data/ranker_monitor/fixture_recalibrate.json python -m scripts.ranker_autoremediate --target label_5d_pos_300bp --trials 3 --refresh-predictions true
```

Execute remediation with feature-safe refresh before repredict (opt-in):

```bash
JBR_ML_HEALTH_SOURCE=fs JBR_ML_HEALTH_PATH=data/ranker_monitor/fixture_recalibrate.json python -m scripts.ranker_autoremediate --target label_5d_pos_300bp --trials 3 --refresh-features true --refresh-predictions true
```

Notes:

- `fixture_recalibrate.json` should route to recalibration path.
- `fixture_unhealthy.json` remains useful for retrain/autotune path tests.
- `--refresh-predictions true` is opt-in and defaults to off.
- `--refresh-only-if-model-changed true` (default) skips repredict if model
  identity (path + mtime) did not change.
- `--refresh-features true` (default off) performs a feature/schema freshness
  check before repredict and can run `feature_generator` first.
- `--refresh-features-only-if-stale true` (default) refreshes only when stale.
- optional predict passthrough:
  - `--ranker-predict-args "<args>"`
  - `--ranker-predict-args-split true`
- optional feature-generator passthrough:
  - `--feature-generator-args "<args>"`
  - `--feature-generator-args-split true`
- when repredict is enabled, autoremediate injects strict compatibility defaults
  unless explicitly overridden in `--ranker-predict-args`:
  - `--strict-feature-match true`
  - `--max-missing-feature-fraction 0.2`

Optional pipeline step integration (default pipeline unchanged):

```bash
python -m scripts.run_pipeline --steps screener,labels,ranker_monitor,ranker_autoremediate,ranker_eval --reload-web false --ranker-autoremediate-args "--target label_5d_pos_300bp --trials 10"
```

Expected tokens:

- `[INFO] AUTOREMEDIATE_START ...`
- `[INFO] AUTOREMEDIATE_DECISION decision=<allow|warn|block> ...`
- `[INFO] AUTOREMEDIATE_RECALIBRATE_START ...` and `..._END ...` (recalibrate path)
- `[INFO] AUTOREMEDIATE_AUTOTUNE_START ...` and `..._END ...` (retrain path)
- `[INFO] AUTOREMEDIATE_FEATURES_FRESHNESS stale=<true|false> reason=<...> model_feature_set=... features_feature_set=... model_feature_signature=... features_feature_signature=...`
- `[INFO] AUTOREMEDIATE_REFRESH_FEATURES enabled=true stale=<true|false> -> running feature_generator feature_set=<...>` (when enabled)
- `[INFO] AUTOREMEDIATE_REFRESH_FEATURES_DONE rc=<rc>` (when refresh runs)
- `[INFO] AUTOREMEDIATE_REPREDICT_START reason=<model_changed|forced> args=...` (when refresh enabled and executed)
- `[INFO] AUTOREMEDIATE_REPREDICT_END rc=<rc> predictions_source=<db|fs>:<present|missing>` (when repredict runs)
- `[WARN] AUTOREMEDIATE_REPREDICT_END rc=2 predictions_source=<db|fs>:<present|missing> reason=feature_mismatch_fatal` (strict mismatch)
- `[WARN] AUTOREMEDIATE_REPREDICT_SKIPPED reason=features_refresh_failed` (refresh failed)
- `[INFO] AUTOREMEDIATE_REPREDICT_SKIPPED reason=<refresh_disabled|model_unchanged|prior_step_failed>` (when skipped)
- `[INFO] AUTOREMEDIATE_END executed=<true|false> output=...`
- `[INFO] AUTOREMEDIATE_DB_WRITTEN artifact_type=ranker_autoremediate run_date=...` (DB-enabled)

## Candidate Source Rules

- Primary reads: DB tables/views (`screener_candidates`, `top_candidates`, `latest_screener_candidates`, `latest_top_candidates`).
- `scripts.execute_trades` defaults to `--source db`.
- CSV candidate export is opt-in (`JBR_WRITE_CANDIDATE_CSVS=true`) for troubleshooting only.
- When ranker enrichment is enabled, DB candidate reads include
  `model_score_5d` from the app-owned overlay table
  `screener_ranker_scores_app` and execution ranking logs:
  - `[INFO] RANK_COLUMN chosen=<model_score_5d|score> non_null=<count> total=<n>`
- Symbol matching for ranker overlay uses normalized `UPPER(TRIM(symbol))`
  semantics to reduce null-score joins from casing/whitespace mismatches.

## ML-Weighted Sizing (Opt-In)

- `scripts.execute_trades` supports `--alloc-weight-key model_score`.
- When `model_score` is not present but `model_score_5d` exists, the executor
  auto-aliases `model_score := model_score_5d` and logs:
  - `[INFO] MODEL_SCORE_ALIAS source=model_score_5d dest=model_score rows=<N> non_null=<K>`
- Optional score-quality gate (opt-in):
  - `--min-model-score <float>` filters candidates below threshold.
  - `--require-model-score true` drops rows with missing model score.
  - `--use-champion-execution true` loads execution threshold defaults from the
    latest `ranker_champion` payload (DB-first, FS fallback), fill mode only.
    - precedence: explicit CLI flags > champion execution defaults > script defaults.
    - tokens:
      - `[INFO] EXEC_CHAMPION_LOAD source=<db|fs> present=<true|false> run_date=<...>`
      - `[INFO] EXEC_CHAMPION_APPLIED min_model_score=<...> require_model_score=<...> mode=fill`
  - gate log token:
    - `[INFO] MODEL_SCORE_GATE min=<...> require=<...> before=<N> after=<M> missing=<K> below_min=<B>`
  - coverage token:
    - `[INFO] MODEL_SCORE_COVERAGE_EXEC total=<N> non_null=<K> pct=<...> col_used=<model_score|model_score_5d>`
- Optional market-window bypass for offline diagnostics only:
  - `--ignore-market-gate true` works only with `--dry-run true` or
    `--diagnostic`.
  - token:
    - `[WARN] MARKET_GATE_BYPASSED reason=dry_run_or_diagnostic`
- This is ranking-assist sizing only. It does not add new candidates and does
  not change paper-only trade semantics.

Safe operator command (no orders submitted):

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.execute_trades --source db --alloc-weight-key model_score --diagnostic --dry-run true
```

Good output indicators:

- `ALLOC_WEIGHT_COLUMNS present=True ...`
- `ALLOCATION_MODE mode=weighted ...`
- `MODEL_SCORE_GATE min=... require=... before=... after=...`
- No `[WARN] ALLOC_WEIGHT_FALLBACK reason=missing_model_score` when
  `model_score_5d` is present.

Example with explicit score gate:

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.execute_trades --source db --alloc-weight-key model_score --min-model-score 0.55 --diagnostic --dry-run true
```

Champion-driven execution threshold validation (paper-safe):

```bash
cd /home/RasPatrick/jbravo_screener && source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate && set -a; . ~/.config/jbravo/.env; set +a; python -m scripts.execute_trades --source db --alloc-weight-key model_score --use-champion-execution true --diagnostic --dry-run true --ignore-market-gate true
```

Good output indicators:

- `EXEC_CHAMPION_LOAD ... present=true ...`
- `EXEC_CHAMPION_APPLIED min_model_score=...`
- `MODEL_SCORE_COVERAGE_EXEC total=... non_null=... pct=...`
- `MODEL_SCORE_GATE min=...`
- no live orders submitted (dry-run + diagnostic)

## Entry-Time ML Context For Attribution (DB-First)

To make trade attribution deterministic for future trades, the executor now
writes entry-time ML score context by `order_id` into the app overlay table
`trade_entry_ml_context_app` (best-effort, non-blocking).

Expected execution token on submit/chase/fill:

- `[INFO] TRADE_ML_CONTEXT_UPSERT stage=<submit|chase_submit|fill> order_id=<...> symbol=<...> model_score=<...> screener_run_ts_utc=<...>`

Attribution matching order (`--match-mode auto`):

1. `entry_context` (`trades.entry_order_id -> trade_entry_ml_context_app.order_id`)
2. `scores_direct`
3. `run_map`

Quick checks:

```bash
python -m scripts.ranker_trade_attribution --self-test
python -m scripts.ranker_trade_attribution --lookback-days 30 --match-mode auto
```

Good output indicators:

- self-test prints `[INFO] TRADE_ATTRIBUTION_SELF_TEST_PASS matched=1`
- attribution prints `TRADE_ATTRIBUTION_DIAG ... entry_context_rows_available=<n> ...`
- attribution summary prints `TRADE_ATTRIBUTION_MATCHED trades_total=... matched=... unmatched=...`

Note: historical trades will only match `entry_context` if this logging existed
at submit/fill time; no automatic backfill is implied.

## Backfill Entry ML Context For Attribution

Use backfill to populate `trade_entry_ml_context_app` for recent closed trades
when direct entry-time logging was not yet available.

Dry-run (no DB writes):

```bash
python -m scripts.trade_entry_ml_context_backfill --lookback-days 30 --dry-run
```

Apply backfill (DB-first):

```bash
python -m scripts.trade_entry_ml_context_backfill --lookback-days 30 --max-lag-hours 24
```

Expected tokens:

- `[INFO] ENTRY_CONTEXT_BACKFILL_START lookback_days=... max_lag_hours=... dry_run=...`
- `[INFO] ENTRY_CONTEXT_BACKFILL_DIAG trades_total=... missing_context=... candidates_from_raw=... candidates_from_scores=...`
- `[INFO] ENTRY_CONTEXT_BACKFILL_UPSERTED rows=...`
- `[INFO] ENTRY_CONTEXT_BACKFILL_END rows_upserted=... rows_skipped=... output=...`
- `[INFO] ENTRY_CONTEXT_BACKFILL_DB_WRITTEN artifact_type=... run_date=...`

Outputs:

- FS fallback/debug:
  - `data/trade_entry_ml_context_backfill/latest.json`
  - `data/trade_entry_ml_context_backfill/backfilled_rows.csv`
- DB artifacts:
  - `trade_entry_ml_context_backfill` (JSON payload)
  - `trade_entry_ml_context_backfill_rows` (CSV payload)

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

## Verifier Commands

```bash
python -m scripts.docs_consistency_check
python -m scripts.dashboard_consistency_check
```

Good output indicators:

- `DOCS_CONSISTENCY PASS`
- Dashboard checker exits `0` and findings contain no `FAIL` lines.

## Dashboard Consistency Checker Troubleshooting

1. CSV parity mismatch (`[PARITY] top_candidates.csv rows mismatch`):
   - In DB-first mode, CSV parity checks run only when `JBR_WRITE_CANDIDATE_CSVS=true`.
   - If parity is not required, unset `JBR_WRITE_CANDIDATE_CSVS` and rerun the checker.

   ```bash
   mkdir -p data/parachute
   mv data/top_candidates.csv data/parachute/top_candidates_$(date -u +%Y%m%dT%H%M%SZ).csv
   python -m scripts.dashboard_consistency_check
   ```

2. `DB_CONNECT_FAIL localhost:9999`:
   - `localhost:9999` should only appear when `JBR_DEV_DB_DEFAULTS=true` (dev fallback) or when explicitly configured.
   - On PythonAnywhere/production, source `~/.config/jbravo/.env` so `DATABASE_URL` is present.

   ```bash
   set -a; . ~/.config/jbravo/.env; set +a
   python -m scripts.dashboard_consistency_check
   ```

   Background: https://help.pythonanywhere.com/pages/AccessingPostgresFromOutsidePythonAnywhere/
