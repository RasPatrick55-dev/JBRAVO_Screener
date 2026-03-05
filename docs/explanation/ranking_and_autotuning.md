# ML Ranker Pipeline And Validation

This doc explains how the current ML ranker pipeline works in this repository
and how to validate ranking changes safely.

## Guardrails (must stay true)

- Paper-only operations only. No live trading support in this repo.
- DB-first operation. PostgreSQL is the source of truth.
- CSV artifacts are debug/parachute outputs, not authoritative.
- ML assists ranking only. It does not create candidates and does not bypass
  screener gates.

## Data sources and joins

The pipeline uses these artifacts/tables:

| Stage | Primary source | Key columns | Output |
| --- | --- | --- | --- |
| Bars | `daily_bars` (`ml_artifacts` or `data/daily_bars.csv`) | `symbol`, `timestamp` | Input to label + feature generation |
| Labels | `labels` (`ml_artifacts` or `data/labels/labels_*.csv`) | `symbol`, `timestamp` | `fwd_ret_*`, `label_*` columns |
| Features | `features` (`ml_artifacts` or `data/features/features_*.csv`) | `symbol`, `timestamp` | v1 or v2 numeric features + labels |
| Predictions | `predictions` (`ml_artifacts` or `data/predictions/predictions_*.csv`) | `symbol`, `timestamp` | `score_5d`, `pred_5d_pos` |
| Eval | `ranker_eval` (`ml_artifacts.payload` or `data/ranker_eval/latest.json`) | N/A | Deciles + quality metrics |
| Screener candidates | `screener_candidates` / `latest_*` views | `symbol`, `timestamp` | Gated candidate universe |

Join keys used in ML scripts are `symbol` + `timestamp`.

## Run locally

End-to-end minimal run through evaluation:

```bash
python -m scripts.run_pipeline --steps screener,labels,ranker_eval --reload-web false
```

Manual sequence (useful for debugging):

```bash
python -m scripts.label_generator --bars-path data/daily_bars.csv --split-adjust auto
JBR_ML_FEATURE_SET=v2 python -m scripts.feature_generator
JBR_ML_CALIBRATION=sigmoid python -m scripts.ranker_train --target label_5d_pos_300bp
python -m scripts.ranker_predict
python -m scripts.ranker_eval --test-size 0.2 --top-k 20 --top-k-fraction 0.1
python -m scripts.ranker_walkforward --target label_5d_pos_300bp
```

When DB is enabled, these scripts read/write `ml_artifacts`. When DB is
disabled, they use filesystem artifacts under `data/`.

## Feature-set toggle (v1 vs v2)

`scripts.feature_generator` supports:

- `JBR_ML_FEATURE_SET=v1|v2` (default: `v1`)
- optional CLI override: `--feature-set v1|v2`

`v1` keeps the legacy momentum/volume feature columns unchanged.

`v2` keeps all `v1` columns and adds daily-bar-derived numeric families:

- Returns/momentum (`ret_*`, `logret_1d`)
- Volatility (`volatility_10d`, `atr14`, `atr_pct`)
- Trend/MA distance (`sma_*`, `ema_20`, distance-to-MA features)
- Oscillators/bands (`rsi14`, `macd_hist`, Bollinger levels/bandwidth)
- Volume/flow (`vol_ma30`, `rel_volume`, `obv`, `obv_delta`)
- Candlestick geometry/pattern-lite flags (`candle_*`)

Time-safety rules:

- Features are computed per symbol after sorting by `symbol,timestamp`.
- Calculations are backward-looking (rolling windows, EWMs, lagged terms).
- No feature uses future bars.
- Any feature NaNs/inf are cleaned (`inf -> NaN -> 0`) without changing label columns.

`ranker_train` selects numeric feature columns dynamically (excluding IDs, `close`,
`fwd_ret_*`, and `label_*`) and saves the exact `feature_columns` to the model
artifact and summary JSON. `ranker_predict` reuses those columns and zero-fills
missing columns for compatibility.

### Time-safe calibration for production scores

`scripts.ranker_train` supports optional probability calibration:

- `--calibrate {none,sigmoid,isotonic}` (env fallback: `JBR_ML_CALIBRATION`)
- `--calibration-fraction` (default `0.1`) controls calibration-window size

Calibration flow is chronological and disjoint from base-model fitting:

1. Sort by `timestamp`.
2. Reserve final validation holdout (existing time-ordered split).
3. From the training portion only, carve:
   - older `train_core`
   - recent `calibration_set`
4. Apply embargo between `train_core` and `calibration_set` based on target
   horizon (for example `label_5d_*` => 5-day embargo).
5. Fit base model on `train_core`, then fit calibrator on `calibration_set`
   only.

This keeps calibration time-safe and avoids fitting calibration parameters on
the same rows used for base-model fitting. Brier score (already reported in
evaluation scripts) is calibration-sensitive: lower is better.

Notes:

- `sigmoid` is usually more stable with smaller calibration windows.
- `isotonic` can overfit when calibration rows are limited.

Split-safe label/feature fallback:

- `scripts.label_generator` supports `--split-adjust {off,auto,force}` and
  env `JBR_SPLIT_ADJUST` (CLI > env > `off`).
- When split adjustment is applied, forward returns are computed from
  `close_adj` while preserving raw `close`.
- `scripts.feature_generator` automatically uses `close_adj` for price-derived
  indicators when available and logs:
  - `[INFO] FEATURES_PRICE_COL price_col=<close|close_adj>`
- `labels` and `features` DB artifacts now include adjustment metadata in
  payload (`bars_adjustment`, `split_adjust_mode`, and applied flags/counts).

## Time-series-safe evaluation (no look-ahead)

`scripts.ranker_eval` now evaluates with chronological splitting:

1. Merge predictions and labels on `symbol,timestamp`.
2. Sort by `timestamp`.
3. Split by time (`train` earlier, `test` later) using `--test-size`.
4. Apply an embargo gap (`--embargo-days`; default inferred from label horizon
   like `label_5d_*` => 5 days) between train and test where feasible.
5. Compute evaluation metrics on the test window only.

This avoids random shuffling and reduces overlap leakage from forward labels.

## Interpreting `ranker_eval` output

Core fields in `data/ranker_eval/latest.json`:

- `validation_scheme`, `split_note`, `embargo_days`, `embargo_applied`
- `train_start`, `train_end`, `test_start`, `test_end`
- `sample_size` (test rows), `train_sample_size`, `population_size`
- `deciles` (still present for dashboard compatibility)
- `top_avg_label`, `bottom_avg_label`, `decile_lift`, `signal_quality`
- `metrics` (test metrics) and `train_metrics`:
  - `auc`
  - `mae`, `rmse`, `brier`
  - `spearman`
  - `precision_at_k`, `top_k_lift_vs_baseline`
  - `precision_at_top_fraction`, `top_fraction_lift_vs_baseline`
- calibration diagnostics (test-only):
  - `calibration_bins`
  - `calibration_curve` (`bin_lo`, `bin_hi`, `count`, `avg_pred`, `frac_pos`)
  - `ece` (Expected Calibration Error)
  - `mce` (Max Calibration Error)
  - `calibration_applicable`, `score_min`, `score_max`

High-level read:

- Higher `decile_lift` and `precision_at_k` suggest better ranking separation.
- `train_metrics` vs `metrics` drift helps spot overfit.
- Time-window boundaries confirm evaluation uses future holdout, not random rows.

### Calibration diagnostics (ECE + reliability curve)

`ranker_eval` computes calibration diagnostics on the **test split only** after
the chronological split and embargo. This keeps reliability measurements OOS and
avoids in-sample optimism.

Diagnostics:

- Reliability table (`calibration_curve`) over fixed bins in `[0,1]`
- `ece`: weighted average absolute gap between predicted probability and
  observed positive rate
- `mce`: worst-bin absolute gap

Interpretation:

- Lower `ece`/`mce` is better calibration.
- `brier` remains useful as a probability-quality metric (lower is better).
- If scores are outside probability range, calibration diagnostics are skipped
  with an explicit token (`RANKER_EVAL_CALIBRATION_SKIPPED`).

Calibration method tradeoffs (training side):

- `sigmoid` is generally safer with smaller calibration windows.
- `isotonic` is more flexible but can overfit small samples and can create
  score plateaus/ties that may reduce ranking granularity in thin datasets.

## Walk-forward Evaluation (`ranker_walkforward`)

`scripts.ranker_walkforward` extends validation from one holdout window to many
rolling out-of-sample folds:

1. Build chronological folds with:
   - train window (`--train-window-days`, default `504`)
   - embargo gap (`--embargo-days`, default inferred from target horizon with
     floor `5`)
   - test window (`--test-window-days`, default `63`)
   - fold step (`--step-days`, default `21`)
2. For each fold, evaluate test-period ranking quality and strategy-like
   outcomes.
3. Aggregate fold distributions into `summary` statistics.

Forward-return sanity diagnostics:

- `ranker_walkforward` now logs a warning when forward-return tails look
  unrealistic (often split-driven on raw bars):
  - `[WARN] FWD_RET_OUTLIER_SUSPECTED col=<fwd_ret_*> p99=<...> max=<...> suggestion=use_adjustment_split_or_all`
- Optional research-only clipping:
  - `--max-abs-fwd-ret <value>` (default `0`, disabled)
  - when enabled and clipping occurs:
    - `[WARN] FWD_RET_CLIPPED max_abs=<...> clipped_rows=<...>`

Why retrain-per-fold matters:

- Precomputed `score_5d` can leak if that score was generated by a model trained
  on data that extends beyond the fold test period.
- `--retrain-per-fold` trains inside each fold (`train -> embargo -> test`) and
  computes metrics only on that fold's OOS predictions (`score_oos`), which is
  safer for tuning.

Key CLI flags:

- `--retrain-per-fold`: enable true OOS fold predictions.
- `--calibrate {none,sigmoid,isotonic}`: optional probability calibration on
  training data only (Brier score is calibration-sensitive; lower is better).
- `--score-col score_5d`: precomputed mode (default) when retrain is not used.

Key metrics per fold:

- Model quality:
  - `auc`
  - `brier` (lower is better; calibration-sensitive)
  - `spearman`
  - `precision_at_k`
- Strategy-like:
  - `top_k_mean_fwd_ret`
  - `top_k_win_rate`
  - `top_k_excess_mean_fwd_ret` vs full test window baseline

Artifact locations:

- FS: `data/ranker_walkforward/latest.json`
- FS: `data/ranker_walkforward/oos_predictions.csv`
- DB-first: `ml_artifacts` row with `artifact_type='ranker_walkforward'`
- DB-first: `ml_artifacts` row with `artifact_type='ranker_oos_predictions'`

Pipeline usage (optional step):

```bash
python -m scripts.run_pipeline \
  --steps screener,labels,ranker_eval,ranker_walkforward \
  --ranker-walkforward-args "--target label_5d_pos_300bp --retrain-per-fold --calibrate sigmoid"
```

This enables comparing out-of-sample fold distributions over time, which is
safer for auto-tuning (`top_k`, thresholds, horizon choices) than relying on a
single split.

## Strategy Evaluation On OOS Predictions

`scripts.ranker_strategy_eval` measures simple profit-capture behavior from
Step 5 OOS predictions without changing any trade execution code.

Data source order:

- DB-first: `ml_artifacts` with `artifact_type='ranker_oos_predictions'`
- FS fallback: `data/ranker_walkforward/oos_predictions.csv`

Simulation design (Step 6 baseline):

1. Infer horizon from `--target` (or use `--horizon-days`).
2. Rebalance every `--rebalance-days` (default: horizon days).
3. At each rebalance timestamp, rank by score and select `--top-k`.
4. Use precomputed `fwd_ret_{horizon}d` for each selected symbol.
5. Apply transaction cost `--cost-bps` per rebalance entry (entry-only model).
6. Compound period returns into an equity curve.

Adjustment note:

- If bars are fetched with `raw` adjustment, split discontinuities can inflate
  `fwd_ret_*` tails and distort strategy metrics.
- Prefer `JBR_BARS_ADJUSTMENT=split` (or `all`) for evaluation-focused runs
  when outlier warnings appear.
- Feed/account caveat: adjustment availability can vary by Alpaca feed and
  subscription. In practice, IEX fallback may not always provide fully adjusted
  OHLCV history, so `--split-adjust auto` is a deterministic label-side safety net.

Why non-overlapping rebalance:

- Rebalancing at the same cadence as the forward label horizon reduces overlap
  bookkeeping and keeps the evaluator deterministic.
- This is intentional for tuning stability; it is not a full execution
  simulator.

Reported metrics include:

- `total_return`, `cagr` (annualized using 252 trading days), `vol_annual`,
  `sharpe` (risk-free set to 0), `max_drawdown`
- `win_rate`, `avg_period_return`, `median_period_return`
- `baseline_*` metrics using mean forward return across all symbols on each
  rebalance timestamp

`ranker_strategy_eval` uses the same forward-return sanity diagnostics and
optional `--max-abs-fwd-ret` clipping behavior as `ranker_walkforward`.

Label generation also runs forward-return sanity diagnostics and logs:

- `[WARN] FWD_RET_OUTLIER_SUSPECTED ... suggestion=set JBR_BARS_ADJUSTMENT=split or enable --split-adjust auto`

Optional strict research mode:

- `--strict-fwd-ret` (or env `JBR_STRICT_FWD_RET=true`) makes label generation
  fail with rc=2 when severe outliers are detected.

Artifacts written:

- FS: `data/ranker_strategy_eval/latest.json`
- FS: `data/ranker_strategy_eval/equity_curve.csv`
- FS (optional): `data/ranker_strategy_eval/param_sweep.csv`
- DB-first:
  - `ranker_strategy_eval` (JSON payload)
  - `ranker_strategy_equity` (CSV payload)
  - `ranker_strategy_sweep` (CSV payload when `--sweep` is used)

Pipeline (optional step):

```bash
python -m scripts.run_pipeline \
  --steps screener,labels,ranker_walkforward,ranker_strategy_eval \
  --ranker-walkforward-args "--target label_5d_pos_300bp --retrain-per-fold" \
  --ranker-strategy-eval-args "--target label_5d_pos_300bp --top-k 25 --cost-bps 5"
```

## Autotune Runner (`ranker_autotune`)

`scripts.ranker_autotune` runs repeated OOS trials by combining:

1. label/feature generation under candidate settings
2. retrain-per-fold walk-forward evaluation
3. strategy evaluation on OOS predictions
4. tune-vs-holdout champion selection

What it sweeps:

- `feature_set`: `v1`, `v2`
- `split_adjust`: `off`, `auto`
- `bars_adjustment`: `raw`, `split`
- `calibration`: `none`, `sigmoid` (and optional `isotonic`)
- `top_k`: default set `10,25,50`
- `cost_bps`: default set `0,5,10`
- `min_score` (execution-style threshold): default grid `0.0,0.55,0.6,0.65`

Objective policy (in code and output payload):

- maximize `sharpe` primarily
- then maximize `cagr`
- include drawdown penalty and discard configs when:
  - `sharpe < 0`
  - `max_drawdown < -0.60`
- holdout gating:
  - `holdout_sharpe >= --min-holdout-sharpe` (default `0.3`)
  - `holdout_periods >= --min-holdout-periods` (default `30`)
  - `holdout_max_drawdown >= max_drawdown_floor`

Holdout concept (simple):

- autotune uses a **tune window** and a **holdout window**.
- tune window is all OOS rows except the last `--holdout-days` calendar days.
- holdout window is the last `--holdout-days` calendar days.
- this reduces selection bias by checking whether a parameter set that looks
  good in tuning still behaves acceptably in data not used for selection.
- set `--holdout-days 0` to disable holdout and keep tune-only behavior.

Sanity caps (default on):

- reject a trial when `holdout_cagr > --max-cagr-cap` (default `5.0`)
- reject a trial when any `|period_return| > --max-abs-period-return-cap`
  (default `1.0`, i.e., 100%)
- these guards help suppress split/artifact-driven unrealistic winners.

This gives a more robust ranking for both win-rate and profit capture while
keeping strict OOS discipline.

Why this is OOS-safe:

- each trial uses `ranker_walkforward --retrain-per-fold` to avoid precomputed-score leakage
- strategy metrics are computed from fold OOS predictions with non-overlapping
  rebalances matched to label horizon assumptions

DB-first + outputs:

- FS:
  - `data/ranker_autotune/latest.json`
  - `data/ranker_autotune/param_sweep.csv`
  - `data/ranker_autotune/champion.json`
- DB-first:
  - `ranker_autotune` (summary JSON payload)
  - `ranker_autotune_sweep` (full sweep CSV payload)
  - `ranker_champion` (selected champion payload)

Run in DB-first mode:

```bash
python -m scripts.ranker_autotune --target label_5d_pos_300bp --trials 20
```

With explicit holdout/caps:

```bash
python -m scripts.ranker_autotune \
  --target label_5d_pos_300bp \
  --trials 20 \
  --holdout-days 252 \
  --min-holdout-sharpe 0.3 \
  --min-holdout-periods 30 \
  --max-cagr-cap 5.0 \
  --max-abs-period-return-cap 1.0
```

With explicit score-threshold grid tuning:

```bash
python -m scripts.ranker_autotune \
  --target label_5d_pos_300bp \
  --trials 20 \
  --min-score-grid "0.0,0.5,0.55,0.6,0.65"
```

Optional pipeline step (opt-in only):

```bash
python -m scripts.run_pipeline \
  --steps screener,labels,ranker_walkforward,ranker_strategy_eval,ranker_autotune \
  --ranker-autotune-args "--target label_5d_pos_300bp --trials 20"
```

Important: `ranker_autotune` is evaluation-only and does not auto-promote or
change live/paper execution behavior.

Champion output now includes execution guidance:

- `execution.min_model_score`
- `execution.require_model_score` (default `false`)

These are recommendations produced from OOS tune/holdout evidence. They are
only used by execution when the operator explicitly enables
`--use-champion-execution`.

## Monitoring & Drift (`ranker_monitor`)

`scripts.ranker_monitor` is a fast, evaluation-only health check that runs on
OOS predictions and emits one health payload with a recommendation.

What it computes:

1. Drift via PSI (Population Stability Index) between an older baseline window
   and the latest recent window.
2. Recent-window strategy metrics using the same non-overlapping, horizon-aware
   evaluation assumptions as `ranker_strategy_eval`.
3. Calibration drift on OOS probabilities (`ECE`/`MCE`) between baseline and
   recent windows when scores are probability-like (`[0,1]`).

Time-series safety reminder:

- Monitoring uses OOS prediction rows in chronological windows.
- It does not retrain on future data and does not mix future rows into the
  baseline window.
- As with all ranker evaluation, avoid leakage by preserving timestamp order.

PSI thresholds (configurable):

- `< 0.10`: stable
- `0.10 - 0.25`: moderate drift (`warn`)
- `>= 0.25`: significant drift (`alert`)

Defaults are configurable via CLI:

- `--recent-days 63`
- `--baseline-days 252`
- `--psi-bins 10`
- `--psi-warn 0.10`
- `--psi-alert 0.25`
- `--calibration-bins 10`
- `--calibration-min-rows 2000`
- `--calibration-ece-warn 0.05`
- `--calibration-ece-alert 0.10`

Calibration drift notes:

- Monitoring computes reliability diagnostics only on OOS windows, not
  training/in-sample rows.
- If score ranges are not probability-like (`[0,1]`) or windows are too small,
  calibration diagnostics are skipped with:
  - `[WARN] RANKER_MONITOR_CALIBRATION_SKIPPED reason=<...> ...`
- On success, monitor logs:
  - `[INFO] RANKER_MONITOR_CALIBRATION bins=... recent_ece=... baseline_ece=... delta_ece=... ...`
- Payload includes `calibration_applicable` plus `calibration` blocks for
  `recent` and `baseline` (`ece`, `mce`, score range, reliability table, and
  `delta_ece`).

Recommended action field (no automation side effects):

- `none`: stable drift + acceptable recent metrics
- `recalibrate`: calibration drift or moderate PSI/performance weakness
- `retrain`: significant PSI drift or severe recent strategy degradation

`ranker_monitor` does **not** auto-run autotune and does **not** alter trading
execution behavior.

Outputs:

- FS: `data/ranker_monitor/latest.json`
- DB-first: `ml_artifacts` `artifact_type='ranker_monitor'`

Pipeline step (opt-in):

```bash
python -m scripts.run_pipeline \
  --steps screener,labels,ranker_walkforward,ranker_strategy_eval,ranker_monitor \
  --ranker-monitor-args "--target label_5d_pos_300bp --recent-days 63 --baseline-days 252"
```

Expected tokens include:

- `[INFO] RANKER_MONITOR_START ...`
- `[INFO] RANKER_MONITOR_DRIFT ...`
- `[WARN] RANKER_MONITOR_PSI ... level=warn|alert ...` (when thresholds are hit)
- `[INFO] RANKER_MONITOR_END ... recommended_action=...`

## Operational Guardrails Using `ranker_monitor`

`scripts.run_pipeline` can optionally gate ML score enrichment based on latest
monitor status, without changing candidate generation or trade execution
semantics.

Opt-in controls:

- `--ml-health-guard`
- `--ml-health-guard-mode {warn,block}`
- env fallback: `JBR_ML_HEALTH_GUARD`, `JBR_ML_HEALTH_GUARD_MODE`

Scope:

- guard applies only to `--enrich-candidates-with-ranker` overlay behavior.
- it does not change screener gates, candidate creation, or order logic.

Behavior:

- if `recommended_action == none`, enrichment proceeds normally.
- if action is non-`none`:
  - `warn` mode logs warning and proceeds.
  - `block` mode skips enrichment for that run.
- `recommended_action=recalibrate` is treated as unhealthy by the guard (same
  policy class as retrain), with deterministic reason tag
  `action_recalibrate`.
- monitor staleness is treated as unhealthy:
  - if monitor run date is older than `JBR_ML_HEALTH_MAX_AGE_DAYS` (default 7)
    relative to pipeline run date, decision becomes `warn` or `block`
    according to guard mode.
- missing monitor payload is also treated as unhealthy (`missing_monitor` reason).

Decision source controls (env):

- `JBR_ML_HEALTH_SOURCE=auto|db|fs` (default `auto`)
- `JBR_ML_HEALTH_PATH=<path>` for forced FS payload testing
- `JBR_ML_HEALTH_MAX_AGE_DAYS=<int>` (default `7`)

Key tokens:

- `[INFO] ML_HEALTH_GUARD enabled=<true|false> mode=<warn|block>`
- `[INFO] ML_HEALTH_LOAD source=<db|fs|missing> present=<true|false> run_date=<...>`
- `[INFO] ML_HEALTH_STATUS action=<...> psi_score=<...> recent_sharpe=<...>`
- `[INFO] ML_ENRICHMENT_DECISION decision=<allow|warn|block> mode=<warn|block> action=<...> reason=<...> psi_score=<...> recent_sharpe=<...> monitor_run_date=<...>`
- `[WARN] ML_ENRICHMENT_WARN ...` or `[WARN] ML_ENRICHMENT_BLOCKED ...`

## Auto-Remediation (`ranker_autoremediate`)

`scripts.ranker_autoremediate` is an evaluation-only workflow that consumes
latest monitor health and conditionally runs either:

- `scripts.ranker_recalibrate` for calibration-only remediation, or
- bounded `scripts.ranker_autotune` for retraining/tuning remediation.

What it does:

1. Load monitor health DB-first (`ranker_monitor` artifact), with FS fallback.
2. Reuse deterministic guard decision logic (`allow|warn|block`) including
   staleness checks.
3. If decision is `allow`, skip remediation.
4. If decision is `warn` or `block` and `--dry-run` is false:
   - `recommended_action=recalibrate` (or reason `action_recalibrate`) runs
     `ranker_recalibrate`.
   - retrain-class actions run `ranker_autotune`.
   - optional post-action repredict (`--refresh-predictions true`) reruns
     `ranker_predict` so prediction artifacts are refreshed against the
     remediated model.
   - optional feature-safe repredict (`--refresh-features true`) checks
     model-vs-features schema freshness first and can refresh features before
     repredict, aligning `JBR_ML_FEATURE_SET` to the remediated model's
     feature-set metadata.
5. Persist a summary artifact as `ranker_autoremediate` (DB-first + FS).

### Recalibrate-only remediation (`ranker_recalibrate`)

`ranker_recalibrate` fits post-hoc probability calibration on a recent,
time-ordered calibration window using an already-trained model.

Time-safety behavior:

1. Load features+labels DB-first (FS fallback only when DB is disabled).
2. Sort rows by timestamp.
3. Take the latest `--calibration-fraction` slice as calibration data.
4. Apply an embargo at the boundary (default inferred from target horizon,
   minimum 5 days) for conservative separation from older rows.

Calibration methods:

- `sigmoid`
- `isotonic`

The script tries estimator-based prefit calibration first when compatible; if
that fails, it falls back to score-mapper calibration on base model scores.

Artifacts:

- Writes updated model + summary under `data/models/` with calibration metadata.
- DB-first payload artifact: `ranker_recalibrate` in `ml_artifacts`.

Why it is safe:

- It is evaluation/tuning only.
- It does not submit orders and does not change `execute_trades` semantics.
- Default daily pipeline behavior is unchanged unless the optional step is
  explicitly included.

Key CLI knobs:

- `--target` (default `label_5d_pos_300bp`)
- `--calibrate {sigmoid,isotonic}` (env fallback: `JBR_ML_CALIBRATION`; defaults to `sigmoid` here)
- `--calibration-fraction` (default `0.1`)
- `--embargo-days` (default inferred from target horizon, minimum 5)

Operational flow:

- Direct run: `python -m scripts.ranker_recalibrate ...`
- Optional pipeline step: include `ranker_recalibrate` in `--steps`
  and pass extra flags via:
  - `--ranker-recalibrate-args`
  - `--ranker-recalibrate-args-split`
- Pipeline emits a summary line:
  - `[INFO] RANKER_RECALIBRATE rc=<rc> method=<...> model_path=<...>`

Why recalibration should be followed by repredict:

- Recalibration updates probability mapping for the trained model.
- Existing prediction artifacts still reflect pre-recalibration scores until
  `ranker_predict` is rerun.
- Operationally, use an ML-only sequence:
  - `labels -> ranker_recalibrate -> ranker_predict`
  - and then optional downstream steps (`ranker_eval`, `ranker_monitor`) to
    refresh health artifacts on the updated score distribution.
- `ranker_autoremediate` supports this as an opt-in one-pass path:
  - `--refresh-predictions true`
  - optional `--refresh-features true` to refresh stale features first
  - default `--refresh-only-if-model-changed true` compares latest model
    identity (`model_path` + `model_mtime_utc`) before and after remediation
    and skips repredict when unchanged.
  - strict compatibility defaults are injected for repredict unless explicitly
    overridden by operator passthrough args:
    - `--strict-feature-match true`
    - `--max-missing-feature-fraction 0.2`

Prediction freshness enforcement:

- `ranker_predict` now writes prediction provenance metadata (selected model
  path + model mtime + calibration flags) alongside predictions artifacts,
  including feature provenance (`feature_set`, `feature_signature`,
  `feature_meta_source`).
- `run_pipeline` compares latest model metadata, latest features metadata, and
  latest predictions metadata before prediction-consuming paths
  (evaluation/enrichment/monitor checks) and emits:
  - `PREDICTIONS_FRESHNESS ...`
  - `PREDICTIONS_STALE ...` when mismatched/unknown.
- predictions are now considered stale for either model drift or feature-schema
  drift (prediction feature set/signature mismatch vs latest features).
- `ranker_predict` writes a prediction-time compatibility verdict
  (`feature_compat`) into predictions metadata. When this verdict is
  `compatible=false`, freshness marks predictions stale with
  `pred_feature_incompatible`.
- Optional strict provenance mode (`JBR_STRICT_PREDICTIONS_META=true`) treats
  missing provenance fields as stale as well (`pred_feature_compat_missing`,
  `pred_feature_set_missing`, `pred_feature_signature_missing`,
  `pred_model_meta_missing`). This prevents "fresh-but-unknown compatibility"
  from silently passing guardrails.
- Default behavior is warn-only (no implicit recompute).
- Operators can opt in to automatic refresh with
  `--auto-refresh-predictions` (or `JBR_AUTO_REFRESH_PREDICTIONS=true`),
  which runs `ranker_predict` before continuing.
- Optional strict auto-refresh guard:
  - `JBR_STRICT_AUTO_REFRESH_PREDICTIONS=true`
  - auto refresh forwards:
    - `--strict-feature-match true`
    - `--max-missing-feature-fraction 0.2`
  - this fails fast instead of accepting broad missing-feature fills during
    pipeline-triggered repredict.
- Operators can also opt in to feature-refresh before auto repredict with
  `--auto-refresh-features` (or `JBR_AUTO_REFRESH_FEATURES=true`):
  - pipeline compares latest model feature metadata vs latest features
    metadata and emits `FEATURES_FRESHNESS ...`
  - pipeline emits model context before refresh:
    - `AUTO_REFRESH_FEATURES_MODEL_CONTEXT model_path=... model_feature_set=... model_feature_signature=...`
  - on mismatch (or missing feature metadata), pipeline can refresh
    `feature_generator` (and `label_generator` only if required) before
    rerunning `ranker_predict`. During this refresh, pipeline forces
    `JBR_ML_FEATURE_SET` to the latest model feature set to avoid schema drift.
  - if the model feature set is missing, feature refresh is skipped with
    `AUTO_REFRESH_FEATURES_SKIPPED reason=model_feature_set_missing`.
- Freshness-aware enrichment guard interaction (opt-in):
  - when `--ml-health-guard` is enabled, stale predictions add deterministic
    reason `stale_predictions` to enrichment decisioning.
  - guard `warn` mode: log warning and proceed.
  - guard `block` mode: skip enrichment overlay until predictions are refreshed.
  - enrichment has an additional deterministic safety gate: if predictions
    remain stale/incompatible after refresh (or `ranker_predict` returns
    non-zero), pipeline skips enrichment and logs:
    - `CANDIDATES_ENRICH_SKIPPED reason=predictions_stale_or_incompatible ...`
  - enrichment DB write is also skipped when merged matches are zero
    (`CANDIDATES_ENRICH_SKIPPED reason=matched_zero ...`) to prevent writing
    all-null overlay rows.

Feature/model compatibility guard in `ranker_predict`:

- `feature_generator` now writes feature provenance metadata (DB payload +
  FS sidecars `data/features/features_YYYY-MM-DD.meta.json` and
  `data/features/latest_meta.json`) with:
  - `feature_set`
  - `feature_signature` (stable hash of ordered ML feature columns)
- `ranker_train` and `ranker_recalibrate` persist `feature_set` and
  `feature_signature` with model artifacts, with signature computed from the
  model's actual `feature_columns` schema.
- `ranker_predict` compares model and features metadata each run and logs:
  - `[INFO] RANKER_PREDICT_FEATURE_COMPAT ...`
  - `compatible=false` now reflects all incompatibility reasons, including:
    `feature_set_mismatch`, `feature_signature_mismatch`, and
    `missing_feature_fraction_exceeded`.
- strict mode options:
  - `--strict-feature-match true`
  - `--max-missing-feature-fraction <float>`
- fatal mismatch behavior:
  - `[ERROR] RANKER_PREDICT_FEATURE_MISMATCH_FATAL ...` and rc=2.

Key tokens:

- `[INFO] AUTOREMEDIATE_START ...`
- `[INFO] AUTOREMEDIATE_DECISION decision=<allow|warn|block> ...`
- `[INFO] AUTOREMEDIATE_RECALIBRATE_START ...` / `[INFO] AUTOREMEDIATE_RECALIBRATE_END ...` (recalibrate path)
- `[INFO] AUTOREMEDIATE_AUTOTUNE_START ...` / `[INFO] AUTOREMEDIATE_AUTOTUNE_END ...`
- `[INFO] AUTOREMEDIATE_FEATURES_FRESHNESS stale=<true|false> reason=<...> ...`
- `[INFO] AUTOREMEDIATE_REFRESH_FEATURES enabled=true stale=<true|false> -> running feature_generator feature_set=<...>` (opt-in)
- `[INFO] AUTOREMEDIATE_REFRESH_FEATURES_DONE rc=<rc>`
- `[INFO] AUTOREMEDIATE_REPREDICT_START reason=<model_changed|forced> args=...` (opt-in)
- `[INFO] AUTOREMEDIATE_REPREDICT_END rc=<rc> predictions_source=<db|fs>:<present|missing>` (when repredict runs)
- `[WARN] AUTOREMEDIATE_REPREDICT_END rc=2 ... reason=feature_mismatch_fatal` (strict mismatch)
- `[WARN] AUTOREMEDIATE_REPREDICT_SKIPPED reason=features_refresh_failed` (refresh failed)
- `[INFO] AUTOREMEDIATE_REPREDICT_SKIPPED reason=<refresh_disabled|model_unchanged|prior_step_failed>` (when skipped)
- `[INFO] AUTOREMEDIATE_END executed=<true|false> output=...`
- `[INFO] AUTOREMEDIATE_DB_WRITTEN artifact_type=ranker_autoremediate run_date=...`

Optional pipeline usage:

```bash
python -m scripts.run_pipeline \
  --steps screener,labels,ranker_monitor,ranker_autoremediate,ranker_eval \
  --ranker-autoremediate-args "--target label_5d_pos_300bp --trials 10"
```

## Live Paper Outcome Attribution (`ranker_trade_attribution`)

`scripts.ranker_trade_attribution` evaluates how model scores at entry time
map to realized closed-trade outcomes in paper mode.

DB-first point-in-time join flow:

1. Load `CLOSED` rows from `trades` in a lookback window.
2. Match each trade to the latest available score at or before `entry_time`
   using one of the match modes:
   - `entry_context`: join by `trades.entry_order_id` to
     `trade_entry_ml_context_app.order_id` (score captured at submit/fill time).
   - `scores_direct`: as-of join directly against
     `screener_ranker_scores_app` per symbol.
   - `run_map`: as-of join through `screener_run_map_app`, then join to
     `screener_ranker_scores_app`.
   - `oos_predictions`: counterfactual fallback from walk-forward OOS prediction
     history (`ranker_oos_predictions` artifact). Matching first tries same UTC
     trade-entry date, then falls back to as-of (`oos_ts <= entry_time`).
   - `auto` (default): prefer `entry_context`, then fallback to
     `scores_direct`, then `run_map`, then `oos_predictions`.
3. Fetch `score_at_entry` (`model_score` alias to `model_score_5d`) and
   annotate per-trade match diagnostics (`match_status`, `match_reason`,
   `score_source`, `score_run_ts_utc`).
4. Compute trade metrics overall and by score buckets/deciles.

Core metrics:

- `trades_total`, `trades_scored`, `trades_unmatched`
- `win_rate_scored`, `avg_return_scored`, `median_return_scored`
- bucket metrics: count, win-rate, avg/median return
- optional Brier score when score values are in `[0,1]`
- diagnostics:
  - `entry_context_rows_available`
  - `score_rows_available`, `score_runs_available`, `run_map_rows_available`,
    `oos_rows_available`
  - `matched_by_source`
  - `unmatched_reason_counts`
  - matched lag distribution (`entry_time - score_run_ts_utc`): min/median/p95
    overall and by source

Output locations:

- FS:
  - `data/ranker_trade_attribution/latest.json`
  - `data/ranker_trade_attribution/trades_scored.csv`
- DB-first:
  - `ranker_trade_attribution` (JSON payload)
  - `ranker_trade_attribution_trades` (CSV payload)

Pipeline step (optional):

```bash
python -m scripts.run_pipeline \
  --steps screener,labels,ranker_eval,ranker_trade_attribution \
  --ranker-trade-attribution-args "--lookback-days 252 --bins 10"
```

Attribution coverage triage:

- `matched=0` with `entry_context_rows_available=0` means no entry-time score
  context has been captured for that trade window yet.
- `matched=0` with `score_rows_available=0` usually means score history was not
  captured for that window (no joinable ranker overlay rows yet).
- `matched=0` with non-zero score rows usually indicates a key/time issue:
  symbol normalization mismatch or no score run at/before entry timestamp.
- `matched=0` with non-zero `oos_rows_available` but zero
  `matched_by_source.oos_predictions` means OOS history exists but did not line
  up by symbol/date/as-of window for the evaluated trades.
- Prefer `--match-mode auto` (or explicit `entry_context`) for deterministic
  future attribution. Historical trades are not backfilled automatically unless
  context existed at submit/fill time.

Counterfactual OOS attribution limitations:

- `oos_predictions` fallback is useful for coverage diagnostics, but it is not a
  guaranteed record of the exact score used at entry unless `entry_context`
  exists for that order.
- Treat OOS fallback attribution as counterfactual analysis for model quality,
  not decision-time provenance.

### Entry-Time Score Logging For Attribution

To avoid fragile reconstruction from sparse historical score snapshots, the
executor now stores point-in-time ML score context keyed by `order_id` in
`trade_entry_ml_context_app` at submit/chase/fill events (best-effort).

This follows point-in-time correctness principles: preserve what was known at
decision time, then attribute later using that exact record.

This step is evaluation-only and does not change trade execution semantics.

## Champion-Driven Pipeline Runs (Opt-In)

`scripts.run_pipeline` can apply the latest holdout-selected champion config to
ML analysis steps with:

- `--use-champion`
- `--champion-mode {fill,force}` (default `fill`)

Champion source order:

- DB-first: `ml_artifacts` `artifact_type='ranker_champion'`
- FS fallback: `data/ranker_autotune/champion.json`

Mapped env overrides (when present in champion payload):

- `JBR_ML_FEATURE_SET`
- `JBR_BARS_ADJUSTMENT`
- `JBR_SPLIT_ADJUST`
- optional `JBR_STRICT_FWD_RET`

Applied steps:

- `labels`, `features`, `ranker_train`, `ranker_predict`, `ranker_eval`,
  `ranker_walkforward`, `ranker_strategy_eval`

Precedence for effective behavior:

- step CLI args > existing env vars > champion values > script defaults

In `--champion-mode fill`, only missing env keys are set from champion.
In `--champion-mode force`, mapped env keys are overwritten.

Expected pipeline tokens when enabled:

- `[INFO] CHAMPION_LOAD source=<db|fs|none> present=<true|false> run_date=<...>`
- `[INFO] CHAMPION_APPLIED mode=<fill|force> keys=<comma_list>`
- `[WARN] CHAMPION_MISSING` when enabled but no champion exists

This is still evaluation/ranking assist only and does not add live-trading
behavior.

## Model-Score Sizing Alias

Execution sizing can opt into ML-weighted notional splits with:

- `--alloc-weight-key model_score`

To keep DB and CSV candidate paths consistent, the executor auto-creates:

- `model_score := model_score_5d` (when `model_score` is missing and
  `model_score_5d` is present)

This keeps behavior opt-in: default sizing logic is unchanged unless the
operator explicitly passes `--alloc-weight-key model_score`.

Safe paper-only validation command:

```bash
python -m scripts.execute_trades --source db --alloc-weight-key model_score --diagnostic --dry-run true
```

Expected logs include weighted-allocation signals and no missing-model-score
fallback warnings when `model_score_5d` is available.

## Opt-In Model-Score Gate For Execution

Execution can optionally filter candidate rows by model score before ranking
and position selection:

- `--min-model-score <float>` (default `0.0`)
- `--require-model-score true|false` (default `false`)

Gate behavior (after candidate loading and score aliasing):

- score precedence: `model_score`, then `model_score_5d`
- when `require_model_score=true`, rows with missing score are dropped
- when `min_model_score>0`, rows below threshold are dropped

Token:

- `[INFO] MODEL_SCORE_GATE min=<...> require=<...> before=<N> after=<M> missing=<K> below_min=<B>`

Defaults preserve behavior: with `--min-model-score 0` and
`--require-model-score false`, no additional filtering is applied.

Threshold selection guidance:

- use `ranker_strategy_eval` sweep outputs (`top_k`, `min_score`, `cost_bps`)
  on OOS predictions to choose a threshold that improves risk-adjusted returns
  without collapsing sample size.
- prefer thresholds validated on walk-forward OOS windows and holdouts, not
  in-sample metrics.

Optional champion execution defaults (fill mode, opt-in):

- `--use-champion-execution true` loads latest `ranker_champion` DB-first (FS
  fallback) and applies `execution.min_model_score` /
  `execution.require_model_score` only when CLI flags were not explicitly set.
- precedence: explicit CLI > champion execution values > defaults.
- logs:
  - `[INFO] EXEC_CHAMPION_LOAD source=<db|fs> present=<true|false> run_date=<...>`
  - `[INFO] EXEC_CHAMPION_APPLIED min_model_score=<...> require_model_score=<...> mode=fill`

Score coverage observability:

- pipeline enrichment logs:
  - `[INFO] MODEL_SCORE_COVERAGE total=<N> non_null=<K> pct=<...> source=<db|csv> run_ts_utc=<...>`
- executor pre-gate logs:
  - `[INFO] MODEL_SCORE_COVERAGE_EXEC total=<N> non_null=<K> pct=<...> col_used=<model_score|model_score_5d>`
