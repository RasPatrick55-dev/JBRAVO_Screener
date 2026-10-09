#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/RasPatrick/jbravo_screener}"
VENV_PATH="${VENV_PATH:-/home/RasPatrick/.virtualenvs/jbravo-env/bin/activate}"
ENV_FILE="${ENV_FILE:-$HOME/.config/jbravo/.env}"
cd "${REPO_DIR}"
mkdir -p logs
# All primary scheduled launches must use this launcher for single-flight scope.
exec 9>logs/primary-pipeline.lock
flock -n 9 || { echo '{"status":"overlap","reason":"primary_pipeline_running"}'; exit 1; }
source "${VENV_PATH}"
set -a
. "${ENV_FILE}"
set +a
exec python -B -m scripts.primary_pipeline "$@"
