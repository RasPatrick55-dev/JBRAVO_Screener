#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/home/RasPatrick/jbravo_screener}"
VENV="${VENV:-/home/RasPatrick/.virtualenvs/jbravo-env}"
PID_FILE="$PROJECT/data/monitor_positions.pid"
STATUS_FILE="$PROJECT/data/monitor_status.json"
LOG_OUT="$PROJECT/logs/monitor_run.out"
MONITOR_MODULE="scripts.monitor_positions"

cd "$PROJECT"
mkdir -p "$PROJECT/logs" "$PROJECT/data"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[ERROR] Python venv missing at $VENV" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
set +u
set -a
[ -f "$HOME/.config/jbravo/.env" ] && . "$HOME/.config/jbravo/.env"
[ -f "$PROJECT/.env" ] && . "$PROJECT/.env"
set +a
set -u

write_status_file() {
  STATUS="$1" PID="$2" STATUS_FILE="$STATUS_FILE" python - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "status": os.environ.get("STATUS", "unknown"),
    "pid": int(os.environ.get("PID", "0") or 0) or None,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
Path(os.environ["STATUS_FILE"]).write_text(json.dumps(payload, indent=2))
PY
}

restart_notice() {
  echo "[WARN] MONITOR_RESTARTED existing process not running; restarting" | tee -a "$LOG_OUT"
}

start_monitor() {
  nohup "$VENV/bin/python" -m "$MONITOR_MODULE" >> "$LOG_OUT" 2>&1 &
  local child_pid=$!
  echo "$child_pid" > "$PID_FILE"
  write_status_file "running" "$child_pid"
  echo "[INFO] monitor_positions started (pid=$child_pid)" | tee -a "$LOG_OUT"
}

if [ -f "$PID_FILE" ]; then
  existing_pid=$(cat "$PID_FILE" || true)
  if [ -n "$existing_pid" ] && kill -0 "$existing_pid" >/dev/null 2>&1; then
    echo "[INFO] monitor_positions already running with pid $existing_pid" | tee -a "$LOG_OUT"
    exit 0
  else
    restart_notice
    rm -f "$PID_FILE"
  fi
fi

start_monitor
