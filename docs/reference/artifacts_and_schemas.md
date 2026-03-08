# Artifacts And Schemas

## Source Of Truth Policy

- PostgreSQL is authoritative for pipeline state, candidates, backtest output, and dashboard reads.
- CSV artifacts are non-authoritative debug exports and parachute outputs.
- Default candidate reads should use `latest_screener_candidates` and `latest_top_candidates`.

## Canonical Candidate CSV Header

```text
timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown,entry_price,adv20,atrp,source,sma9,ema20,sma180,rsi14,passed_gates,gate_fail_reason
```

## DB-First Data Sources

- `screener_candidates`: scored screener candidates for each run.
- `top_candidates`: narrowed candidate set for downstream execution.
- `pipeline_runs`: pipeline status and summary payloads keyed by run dates/timestamps.
- `backtest_results`: backtest output used by reporting and health views.
- `metrics_daily`: aggregated daily metrics.
- `order_events` and `trades`: execution and trade lifecycle history.
- `latest_screener_candidates` / `latest_top_candidates`: canonical read views for current run state.
- `screener_ranker_scores_app`: app-owned overlay table keyed by
  `(run_ts_utc, symbol)` with `model_score_5d` from latest ranker predictions
  for DB candidate ranking assist.
- `trade_entry_ml_context_app`: app-owned overlay table keyed by `order_id`
  that stores entry-time ML score context for deterministic closed-trade
  attribution.

`pipeline_runs.summary` may include additive ML guard/freshness fields:

- `ml_health`:
  - `decision`, `mode`, `reasons`
  - `monitor_run_date`, `psi_score`, `recent_sharpe`, `source`
  - `predictions_stale` (bool)
  - `predictions_stale_reason` (for example `model_path_mismatch`,
    `model_mtime_older`, `predictions_meta_missing`)

`screener_ranker_scores_app` table outline:

- `run_ts_utc` (`TIMESTAMPTZ`, PK part)
- `run_date` (`DATE`, optional convenience)
- `symbol` (`TEXT`, PK part)
- `model_score_5d` (`DOUBLE PRECISION`)
- `score_ts` (`TIMESTAMPTZ`, optional prediction timestamp)
- `ranker_version` (`TEXT`, optional metadata)
- `created_at` (`TIMESTAMPTZ`, defaults `now()`)

Score-overlay join notes:

- Candidate-to-overlay joins use normalized symbol semantics equivalent to
  `UPPER(TRIM(symbol))` plus exact `run_ts_utc` matching.
- Pipeline enrichment and DB candidate reads emit
  `MODEL_SCORE_JOIN_DIAG ...` and optional
  `MODEL_SCORE_JOIN_SAMPLE_UNMATCHED ...` tokens for root-cause diagnosis
  when `model_score` coverage is low.

`trade_entry_ml_context_app` table outline:

- `order_id` (`TEXT`, PK)
- `symbol` (`TEXT`, required)
- `entry_time` (`TIMESTAMPTZ`, optional best-effort)
- `screener_run_ts_utc` (`TIMESTAMPTZ`, optional)
- `model_score` (`DOUBLE PRECISION`, optional)
- `model_score_5d` (`DOUBLE PRECISION`, optional)
- `score_col` (`TEXT`, optional)
- `score_source` (`TEXT`, optional, e.g. `execute_trades.submit`)
- `raw` (`JSONB`, optional debug metadata)
- `created_at` (`TIMESTAMPTZ`, defaults `now()`)
- `updated_at` (`TIMESTAMPTZ`, defaults `now()`)

## ML Artifact Types (`ml_artifacts`)

Common artifact types in `ml_artifacts`:

- `daily_bars` (CSV payload in `csv_data`)
- `labels` (CSV payload in `csv_data`)
- `features` (CSV payload in `csv_data`)
- `predictions` (CSV payload in `csv_data`)
- `ranker_eval` (JSON payload in `payload`)
- `ranker_walkforward` (JSON payload in `payload`)
- `ranker_oos_predictions` (CSV payload in `csv_data`)
- `ranker_strategy_eval` (JSON payload in `payload`)
- `ranker_strategy_equity` (CSV payload in `csv_data`)
- `ranker_strategy_sweep` (CSV payload in `csv_data`, optional)
- `ranker_monitor` (JSON payload in `payload`)
- `ranker_autotune` (JSON payload in `payload`)
- `ranker_autotune_sweep` (CSV payload in `csv_data`)
- `ranker_champion` (JSON payload in `payload`)
- `ranker_recalibrate` (JSON payload in `payload`)
- `ranker_autoremediate` (JSON payload in `payload`)
- `ranker_trade_attribution` (JSON payload in `payload`)
- `ranker_trade_attribution_trades` (CSV payload in `csv_data`)
- `trade_entry_ml_context_backfill` (JSON payload in `payload`)
- `trade_entry_ml_context_backfill_rows` (CSV payload in `csv_data`)

`labels` payload metadata (JSON, optional):

- `bars_adjustment` (`raw|split|dividend|all`)
- `split_adjust_mode` (`off|auto|force`)
- `split_adjust_applied` (bool)
- `split_adjust_counts` (`symbols`, `events`, `rows_affected`)
- `split_adjust_price_col` (`close` or `close_adj`)
- `strict_fwd_ret` (bool)
- `fwd_return_sanity` summary by `fwd_ret_*` column

`features` payload metadata (JSON, optional):

- `feature_set` (`v1|v2`)
- `feature_signature` (stable hash of ordered ML feature columns used for model input)
- `feature_columns` (ordered ML feature column names used for signature)
- `feature_count` (count of `feature_columns`)
- `price_col_used` (`close` or `close_adj`)
- `bars_adjustment` (`raw|split|dividend|all`)
- `split_adjust_mode` (`off|auto|force`)
- `split_adjust_applied` (bool)
- `rows`
- `generated_at_utc`
- `output_path`

Filesystem sidecar for latest feature metadata:

- `data/features/features_YYYY-MM-DD.meta.json` (per-feature-file schema/provenance sidecar)
- `data/features/latest_meta.json` (convenience pointer; debug/parachute, DB payload remains canonical when DB is enabled)

`predictions` payload metadata (JSON, DB-first provenance):

- `model_path` (selected model used for scoring)
- `model_mtime_utc` (UTC mtime of selected model file)
- `model_calibrated` (bool)
- `calibration_method` (`none|sigmoid|isotonic`)
- `model_signature` (cheap signature, filename+mtime)
- `model_feature_set`
- `model_feature_signature`
- `features_feature_set`
- `features_feature_signature`
- `features_meta_source` (`db|fs|missing`)
- `feature_set` (resolved feature-set identity used for prediction provenance)
- `feature_signature` (resolved feature-schema signature used for prediction provenance)
- `feature_meta_source` (`db:run_date|db:latest|fs:sidecar|fs:latest|computed_fallback|missing`)
- `missing_feature_fraction`
- `missing_feature_count`
- `feature_compat` (object; prediction-time compatibility verdict)
  - `strict` (bool)
  - `compatible` (bool)
  - `reason` (comma-separated reason tags or `none`)
  - `missing_feature_fraction` (float)
  - `missing_count` (int)
  - `feature_count` (int)
  - `feature_set_model` (`v1|v2|None`)
  - `feature_set_features` (`v1|v2|None`)
  - `feature_signature_model` (string|None)
  - `feature_signature_features` (string|None)
  - `meta_source` (feature metadata source used by predict)
- `predictions_path` (filesystem export path)
- `snapshot_date` (run-date key for artifact row)
- `rows` (prediction row count)
- `generated_at_utc` (metadata write timestamp)

Freshness/guard reason tags derived from predictions metadata may include:

- `pred_feature_incompatible` (when `feature_compat.compatible=false`)
- `pred_feature_compat_missing` (strict meta mode)
- `pred_feature_set_missing` (strict meta mode)
- `pred_feature_signature_missing` (strict meta mode)
- `pred_model_meta_missing` (strict meta mode)

`ranker_eval` payload outline:

- split/window metadata:
  - `validation_scheme`, `split_note`, `test_size`
  - `embargo_days`, `embargo_applied`
  - `split_timestamp`, `train_start`, `train_end`, `test_start`, `test_end`
- data scope:
  - `sample_size`, `train_sample_size`, `population_size`
  - `label_column`, `score_column`, `label_horizon_days`
- ranking outputs:
  - `deciles`, `top_avg_label`, `bottom_avg_label`, `decile_lift`,
    `signal_quality`
  - `metrics`, `train_metrics` (AUC/Brier/RMSE/MAE/Spearman/precision@k family)
- calibration diagnostics (test-only):
  - `calibration_bins`
  - `calibration_curve` (list of `{bin_lo, bin_hi, count, avg_pred, frac_pos}`)
  - `ece`, `mce`
  - `calibration_applicable`
  - `score_min`, `score_max`
  - optional `calibration_skip_reason` when diagnostics are not applicable

`ranker_walkforward` payload outline:

- `validation_scheme`: `walk_forward_embargo`
- `target`, `score_col`, `oos_score_col`, `score_source`
- `model_type`, `model_types`, `calibration`, `calibration_used`
- `feature_count`, `feature_columns`, `retrain_per_fold`
- `train_window_days`, `test_window_days`, `step_days`
- `requested_embargo_days`, `embargo_days`, `label_horizon_days`
- `population_size`, `fold_count`, `folds_count`, `sample_size_total`
- `oos_prediction_rows`
- `folds`: list of fold records with
  - `fold`, `train_start`, `train_end`, `test_start`, `test_end`,
    `score_source`, `model_type`, `calibration`
  - `sample_size`, `auc`, `brier`, `spearman`, `precision_at_k`
  - `top_k_mean_fwd_ret`, `top_k_win_rate`, `top_k_excess_mean_fwd_ret`
- `summary`: aggregate stats across folds (`mean/std/min/max/median`) for
  quality and strategy metrics
- `overall_oos_metrics`: metrics computed on concatenated fold OOS predictions

`ranker_oos_predictions` CSV outline:

- `symbol`, `timestamp`, `close`, `<target_label>`
- `fwd_ret_*` column when available for target horizon
- `fold_id`, `score_oos`, `score_source`

`ranker_strategy_eval` payload outline:

- `target`, `horizon_days`, `score_col`, `score_col_used`
- optional date filters: `start_date`, `end_date`
- `fwd_return_column`, `top_k`, `rebalance_days`, `cost_bps`
- `cost_model`: `entry_only_bps`
- `metrics`:
  - `periods`, `start`, `end`
  - `total_return`, `cagr`, `vol_annual`, `sharpe`, `max_drawdown`
  - `win_rate`, `avg_period_return`, `median_period_return`
  - `turnover_proxy`
  - `baseline_total_return`, `baseline_cagr`, `baseline_avg_period_return`
  - `alpha_total_return`

`ranker_strategy_equity` CSV outline:

- `date`, `equity`, `period_return`
- `baseline_equity`, `baseline_period_return`
- `selected_count`, `universe_count`

`ranker_strategy_sweep` CSV outline:

- parameter columns: `top_k`, `min_score`, `cost_bps`
- strategy metrics per combination (same metric family as summary payload)

`ranker_monitor` payload outline:

- `target`, `data_source`, `score_col_requested`, `score_col_used`
- `fwd_return_column`, `horizon_days`, `rebalance_days`
- monitoring windows:
  - `dataset_start`, `dataset_end`
  - `baseline_start`, `baseline_end`
  - `recent_start`, `recent_end`
  - `baseline_rows`, `recent_rows`, `population_rows`
- `drift`:
  - `psi_bins`, `psi_warn`, `psi_alert`
  - `columns` map with per-column PSI (`psi`, `level`, `ref_count`,
    `cur_count`, `bins_used`)
  - `max_psi`, `max_psi_col`, `warn_cols`, `alert_cols`
- calibration drift:
  - `calibration_applicable` (bool)
  - `calibration`:
    - `bins`, `min_rows`, `score_col`, `label_col`
    - `recent` / `baseline`:
      - `rows`, `ece`, `mce`, `score_min`, `score_max`
      - `reliability_table` (`<= bins` rows with
        `bin_lo`, `bin_hi`, `count`, `avg_pred`, `frac_pos`)
    - `delta_ece` (`recent.ece - baseline.ece`)
- `recent_strategy` / `baseline_strategy` blocks with strategy metrics:
  - `periods`, `total_return`, `cagr`, `sharpe`, `max_drawdown`,
    `win_rate`, period-return stats, baseline comparisons
- recommendation fields:
  - `recommended_action` (`none|recalibrate|retrain`)
  - `recommendation_reasons`
- optional `champion` summary metadata when available

`ranker_autotune` payload outline:

- `target`
- `trials_requested`, `trials_executed`, `trials_ok`
- `objective`:
  - `description`
  - `min_sharpe`
  - `max_drawdown_floor`
- `search_space`:
  - `feature_set`
  - `split_adjust`
  - `bars_adjustment`
  - `calibration`
  - `top_k`
  - `cost_bps`
- `walkforward` window settings (`train_window_days`, `test_window_days`,
  `step_days`, `embargo_days`, `retrain_per_fold`)
- `champion_status` (`ok` or `no_holdout_pass`)
- `holdout_days`
- `best_config` (champion trial row using tune objective + holdout gates)
- `top_configs` (top N trial rows)

`ranker_autotune_sweep` CSV outline:

- trial/config columns:
  - `trial`, `feature_set`, `split_adjust`, `bars_adjustment`,
    `calibration`, `top_k`, `cost_bps`, `min_score`
- objective columns:
  - `eligible`, `eligibility_reason`, `tune_objective_score`,
    `objective_score`, `status`, `error`, `elapsed_secs`
- quality/outcome columns:
  - tune metrics: `tune_sharpe`, `tune_cagr`, `tune_max_drawdown`,
    `tune_win_rate`, `tune_avg_period_return`, `tune_periods`,
    `tune_total_return`, `tune_max_abs_period_return`
  - holdout metrics: `holdout_sharpe`, `holdout_cagr`,
    `holdout_max_drawdown`, `holdout_win_rate`, `holdout_avg_period_return`,
    `holdout_periods`, `holdout_total_return`,
    `holdout_max_abs_period_return`
  - `top_k_mean_fwd_ret`, `top_k_win_rate`
  - `folds_count`, `sample_size_total`

`ranker_champion` payload outline:

- `run_date`, `champion_status`, `holdout_days`
- `champion_params`:
  - `feature_set`, `split_adjust`, `bars_adjustment`, `calibration`,
    `top_k`, `cost_bps`, `min_score`
- `execution`:
  - `min_model_score` (recommended execution threshold from champion trial)
  - `require_model_score` (recommended missing-score behavior, default `false`)
- `tune_metrics` and `holdout_metrics` blocks (Sharpe/CAGR/drawdown/win-rate
  plus period-return sanity fields)
- `objective` and `thresholds` used for selection
- `reproducibility`: `seed`, `sampled_trials`

`ranker_recalibrate` payload outline:

- `target`, `method`, `method_source`
- calibration scope:
  - `calibration_fraction`, `embargo_days`
  - `calibration_rows`
  - `calibration_window` (`start`, `end`)
  - `reference_rows_before_calibration`
- quality summary:
  - `brier_before`, `brier_after`
- model metadata:
  - `fit_mode` (`prefit_estimator|score_mapper`)
  - `source_model_path`, `model_path`, `summary_path`
  - `feature_count`
- input metadata:
  - `input_source` (`db|fs`)
  - `run_utc`

`ranker_autoremediate` payload outline:

- `run_date`, `target`, `mode`, `dry_run`, `trials`
- health-source controls:
  - `ml_health_source`, `ml_health_path`, `ml_health_max_age_days`
- `decision` block:
  - `decision` (`allow|warn|block`)
  - `mode`, `reasons`
  - `monitor_run_date`, `recommended_action`
  - `psi_score`, `recent_sharpe`, `source`
- execution fields:
  - `executed` (bool)
  - `remediation_kind` (`none|recalibrate|retrain`)
  - `recalibrate`:
    - `requested`, `executed`, `rc`, `elapsed_secs`, `cmd`
  - `autotune`:
    - `requested`, `executed`, `rc`, `elapsed_secs`, `cmd`
    - optional `summary` pointer data from latest `ranker_autotune`
- `repredict`:
  - `enabled`, `executed`, `rc`, `elapsed_secs`, `cmd`
  - `skipped_reason` (`refresh_disabled|model_unchanged|prior_step_failed|features_refresh_failed`)
  - `reason` (`model_changed|forced` plus optional `stale_features`)
  - `model_before` (`model_path`, `model_mtime_utc`)
  - `model_after` (`model_path`, `model_mtime_utc`, `feature_set`, `feature_signature`, `feature_count`)
  - `predictions_source` (`db|fs:present|missing`)
  - `refresh_only_if_model_changed` (bool)
  - `features_freshness_before`
    - `stale`, `reason`
    - `model_feature_set`, `features_feature_set`
    - `model_feature_signature`, `features_feature_signature`
    - `features_meta_source`, `features_path`
  - `features_freshness_after` (same shape as above)
  - `features_refresh`
    - `enabled`, `attempted`, `rc`, `elapsed_secs`, `cmd`
    - `feature_set_used`, `refresh_only_if_stale`
  - `repredict_strict`
    - `strict`, `max_missing_feature_fraction`
    - `strict_injected`, `max_missing_injected`
- champion pointer block:
  - `present`, `path`, `source`, `run_date`, `champion_status`,
    `champion_params`
- `output_files`:
  - `latest_json`, `champion_json`, `autotune_latest_json`

`ranker_trade_attribution` payload outline:

- `status` (`ok|insufficient_trades|no_data`)
- `source` (`db://...` or filesystem path)
- `match_mode_used` (`auto|entry_context|scores_direct|run_map|oos_predictions`)
- diagnostics:
  - `entry_context_rows_available`
  - `score_rows_available`
  - `score_runs_available`
  - `run_map_rows_available`
  - `oos_rows_available`
  - `matched_by_source` (for example `entry_context`, `scores_direct`, `run_map`, `oos_predictions`)
  - `unmatched_reason_counts` (e.g., `missing_scores`, `missing_run_map`,
    `missing_entry_context`, `missing_overlay_score`, `missing_oos_predictions`,
    `symbol_mismatch`, `time_mismatch`)
  - `entry_to_score_lag_minutes` (`min`, `median`, `p95`) for matched rows
  - `entry_to_score_lag_minutes_by_source` (`min`, `median`, `p95` grouped by
    `score_source`)
- score metadata:
  - `score_col_requested`, `score_col_used`
  - `bins_requested`, `bins_used`
- window metadata:
  - `lookback_days`, `window_start`, `window_end`
- `summary`:
  - `trades_total`, `trades_scored`, `trades_unmatched`
  - `win_rate_scored`, `avg_return_scored`, `median_return_scored`
  - `brier`, `brier_rows`, `brier_skipped_reason`
- `buckets`:
  - per bucket: `bucket`, `bucket_label`, `count`, `score_min`, `score_max`,
    `win_rate`, `avg_return`, `median_return`

`ranker_trade_attribution_trades` CSV outline:

- closed trade fields (`trade_id`, `symbol`, `entry_time`, `exit_time`,
  `entry_price`, `exit_price`, `qty`, `realized_pnl`, `exit_reason`)
- score join fields (`score_run_ts_utc`, `model_score_5d`, `model_score`,
  `score_at_entry`, `score_source`, `screener_run_ts_utc`, `score_timestamp_utc`)
- match diagnostics (`match_status`, `match_reason`)
- derived fields (`trade_return_pct`, `win`, `entry_to_score_lag_minutes`)

`trade_entry_ml_context_backfill` payload outline:

- `status` (`ok|dry_run|no_data|no_db`)
- `lookback_days`, `max_lag_hours`, `dry_run`
- `window_start`, `window_end`
- summary counts:
  - `trades_total`
  - `missing_context`
  - `candidates_from_raw`
  - `candidates_from_scores`
  - `rows_candidate_total`
  - `rows_upserted`
  - `rows_skipped`
- `reasons` (for example `missing_order_id`, `missing_entry_time`,
  `no_score_within_lag`)
- `output_files` (`latest_json`, `backfilled_rows_csv`)

`trade_entry_ml_context_backfill_rows` CSV outline:

- `order_id`, `trade_id`, `symbol`, `entry_time`
- `screener_run_ts_utc`
- `model_score`, `model_score_5d`
- `score_col`, `score_source`
- `source_kind` (`raw` or `scores_asof`)
- `lag_hours`

## CSV Artifacts (Debug/Parachute)

- `data/latest_candidates.csv`: optional export only; not authoritative.
- `data/top_candidates.csv`: optional export only; not authoritative.
- `data/scored_candidates.csv`: optional export/debug artifact.
- `data/metrics_summary.csv`: summary export for reporting compatibility.
- `data/daily_bars.csv`: Stage 0 ML export when requested.

## Required Pipeline Log Tokens

- `PIPELINE_START`
- `PIPELINE_SUMMARY`
- `FALLBACK_CHECK`
- `PIPELINE_END`
- `DASH RELOAD`

## Common Execution Tokens

- `BUY_SUBMIT`
- `BUY_FILL`
- `BUY_CANCELLED`
- `TRAIL_SUBMIT`
- `TRAIL_CONFIRMED`
- `TIME_WINDOW`
- `CASH`
- `ZERO_QTY`
- `PRICE_BOUNDS`
- `MAX_POSITIONS`

## Artifact Precedence

1. DB views/tables
2. JSON reports generated from DB-backed runs
3. CSV exports for debug or first-run bootstrapping only

## DB Config Precedence

1. `DATABASE_URL` (preferred)
2. `DB_HOST` + `DB_PORT` + `DB_NAME` + `DB_USER` (`DB_PASSWORD` optional)
3. Otherwise DB is disabled with an actionable warning

Dev-only local fallback to `localhost:9999` is available only when `JBR_DEV_DB_DEFAULTS=true`.

## Docs-Check Allow Labels

Use inline allow directives only for intentional exceptions, on the same line or the line immediately before the text:

```html
<!-- docs-check: allow-live_trading_supports_claim -->
```

Available labels:

- `live_alpaca_endpoint`
- `live_trading_enabled_claim`
- `live_trading_supports_claim`
- `production_trading_enabled_claim`
- `csv_source_of_truth_claim`
- `latest_candidates_source_of_truth_claim`
- `latest_candidates_authoritative_claim`
