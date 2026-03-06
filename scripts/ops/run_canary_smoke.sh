#!/usr/bin/env bash
set -euo pipefail

# Bounded canary smoke for PythonAnywhere scheduled task use.

REPO_DIR="${REPO_DIR:-/home/RasPatrick/jbravo_screener}"
VENV_PATH="${VENV_PATH:-/home/RasPatrick/.virtualenvs/jbravo-env/bin/activate}"
ENV_FILE="${ENV_FILE:-$HOME/.config/jbravo/.env}"
CANARY_LOG="${CANARY_LOG:-${REPO_DIR}/logs/canary_smoke_latest.log}"
SCREENER_ARGS="${SCREENER_ARGS:---limit 30}"

mkdir -p "$(dirname "${CANARY_LOG}")"
: > "${CANARY_LOG}"

{
  echo "[INFO] CANARY_START repo_dir=${REPO_DIR}"
  cd "${REPO_DIR}"

  # shellcheck disable=SC1090
  source "${VENV_PATH}"
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a

  export JBR_STRICT_PREDICTIONS_META="${JBR_STRICT_PREDICTIONS_META:-true}"
  export JBR_AUTO_REFRESH_FEATURES="${JBR_AUTO_REFRESH_FEATURES:-true}"
  export JBR_AUTO_REFRESH_PREDICTIONS="${JBR_AUTO_REFRESH_PREDICTIONS:-true}"
  export JBR_STRICT_AUTO_REFRESH_PREDICTIONS="${JBR_STRICT_AUTO_REFRESH_PREDICTIONS:-true}"
  export JBR_RANKER_PREDICT_TIMEOUT_SECS="${JBR_RANKER_PREDICT_TIMEOUT_SECS:-900}"
  export JBR_RANKER_EVAL_TIMEOUT_SECS="${JBR_RANKER_EVAL_TIMEOUT_SECS:-900}"

  set +e
  python -m scripts.run_pipeline \
    --steps screener,labels,ranker_eval \
    --reload-web false \
    --ml-health-guard \
    --ml-health-guard-mode warn \
    --enrich-candidates-with-ranker \
    --use-champion \
    --screener-args "${SCREENER_ARGS}"
  rc=$?
  set -e

  freshness_line="$(grep 'PREDICTIONS_FRESHNESS' "${REPO_DIR}/logs/pipeline.log" | tail -n 1 || true)"
  predict_line="$(grep 'RANKER_PREDICT rc=' "${REPO_DIR}/logs/pipeline.log" | tail -n 1 || true)"
  coverage_line="$(grep 'MODEL_SCORE_COVERAGE total=' "${REPO_DIR}/logs/pipeline.log" | tail -n 1 || true)"
  echo "[INFO] CANARY_SUMMARY freshness=${freshness_line:-missing} predict=${predict_line:-missing} coverage=${coverage_line:-missing}"
  echo "[INFO] CANARY_END rc=${rc} log=${CANARY_LOG}"
  exit "${rc}"
} 2>&1 | tee -a "${CANARY_LOG}"
