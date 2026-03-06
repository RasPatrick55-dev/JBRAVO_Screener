#!/usr/bin/env bash
set -euo pipefail

# Deploy + verify workflow intended to be launched by PythonAnywhere API task orchestration.
# This script is paper-only safe and runs DB-first checks/smoke before optional web reload.

REPO_DIR="${REPO_DIR:-/home/RasPatrick/jbravo_screener}"
VENV_PATH="${VENV_PATH:-/home/RasPatrick/.virtualenvs/jbravo-env/bin/activate}"
ENV_FILE="${ENV_FILE:-$HOME/.config/jbravo/.env}"
DEPLOY_REF="${1:-main}"
SMOKE_SCREENER_ARGS="${SMOKE_SCREENER_ARGS:---limit 30}"

echo "[INFO] DEPLOY_START repo_dir=${REPO_DIR} ref=${DEPLOY_REF}"

cd "${REPO_DIR}"
git fetch --tags origin
git checkout "${DEPLOY_REF}"
if git ls-remote --exit-code --heads origin "${DEPLOY_REF}" >/dev/null 2>&1; then
  git pull --ff-only origin "${DEPLOY_REF}"
fi
echo "[INFO] DEPLOY_GIT_OK ref=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse HEAD)"

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

python -m scripts.docs_consistency_check
python -m scripts.run_pipeline \
  --steps screener,labels,ranker_eval \
  --reload-web false \
  --ml-health-guard \
  --ml-health-guard-mode warn \
  --enrich-candidates-with-ranker \
  --use-champion \
  --screener-args "${SMOKE_SCREENER_ARGS}"
python -m scripts.dashboard_consistency_check

if [[ -n "${PYTHONANYWHERE_USERNAME:-}" && -n "${PYTHONANYWHERE_API_TOKEN:-}" && -n "${PYTHONANYWHERE_DOMAIN:-}" ]]; then
  curl -fsS -X POST \
    -H "Authorization: Token ${PYTHONANYWHERE_API_TOKEN}" \
    "https://www.pythonanywhere.com/api/v0/user/${PYTHONANYWHERE_USERNAME}/webapps/${PYTHONANYWHERE_DOMAIN}/reload/" \
    >/dev/null
  echo "[INFO] WEBAPP_RELOAD_OK domain=${PYTHONANYWHERE_DOMAIN}"
else
  echo "[WARN] WEBAPP_RELOAD_SKIPPED reason=missing_api_env"
fi

echo "[INFO] DEPLOY_END repo_dir=${REPO_DIR} ref=${DEPLOY_REF}"
