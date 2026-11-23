#!/usr/bin/env bash
set -Eeuo pipefail

cd "${PROJECT_HOME:-/home/RasPatrick/jbravo_screener}"
case "${1:-}" in
  producer|entry)
    mkdir -p logs
    exec 9>"logs/webull-paper-$1.lock"
    flock -n 9 || { echo '{"status":"overlap","reason":"webull_paper_task_running"}'; exit 1; }
    ;;
esac
source "${VENV:-/home/RasPatrick/.virtualenvs/jbravo-env}/bin/activate"
set -a
source "$HOME/.config/jbravo/.env"
set +a

exec python -m scripts.strategy_webull_tasks "$@"
