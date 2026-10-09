#!/usr/bin/env bash
set -euo pipefail

# Read-only postflight: never rerun the screener or refresh ML artifacts.

REPO_DIR="${REPO_DIR:-/home/RasPatrick/jbravo_screener}"
VENV_PATH="${VENV_PATH:-/home/RasPatrick/.virtualenvs/jbravo-env/bin/activate}"
cd "${REPO_DIR}"
source "${VENV_PATH}"
# No .env loading, application imports, writes, subprocesses, DB or HTTP in checker.
# Scheduler stdout is the health receipt; old canary logs remain untouched.
exec python -B -m scripts.pipeline_postflight "$@"
