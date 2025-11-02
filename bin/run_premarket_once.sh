#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[WRAPPER] $*"
}

fail_trap() {
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    log "failure rc=${rc} while executing: ${BASH_COMMAND}"
  fi
  exit $rc
}

trap fail_trap ERR

export TZ="${TZ:-America/New_York}"   # for human-readable timestamps in logs
PROJECT_HOME="${PROJECT_HOME:-/home/RasPatrick/jbravo_screener}"
export PROJECT_HOME
VENV="${VENV:-/home/RasPatrick/.virtualenvs/jbravo-env}"
PYTHON="${VENV}/bin/python"
WSGI_PATH="${PROJECT_HOME}/raspatrick_pythonanywhere_com_wsgi.py"

cd "$PROJECT_HOME"
source "$VENV/bin/activate"
set -a; . ~/.config/jbravo/.env; set +a

log "probing Alpaca credentials"
"$PYTHON" - <<'PY'
import os, requests
from urllib.parse import urljoin

base=os.getenv("APCA_API_BASE_URL", "").strip()
key=os.getenv("APCA_API_KEY_ID", "")
secret=os.getenv("APCA_API_SECRET_KEY", "")
status="ERR"
buying_power="0.00"
if base and key and secret:
    try:
        resp = requests.get(
            urljoin(base, "/v2/account"),
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=10,
        )
        status = "OK" if resp.ok else f"HTTP_{resp.status_code}"
        if resp.ok:
            buying_power = resp.json().get("buying_power", "0.00")
    except Exception as exc:  # pragma: no cover - network guard
        status = f"ERR:{exc.__class__.__name__}"
print(f"[WRAPPER] AUTH status={status} buying_power={buying_power}")
PY

check_pipeline() {
  "$PYTHON" - <<'PY'
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

base = Path(".")
metrics_path = base / "data" / "screener_metrics.json"
today = datetime.now(ZoneInfo("America/New_York")).date()
state = "stale"
iso = ""
epoch = ""
if metrics_path.exists():
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    ts = payload.get("last_run_utc")
    if isinstance(ts, str) and ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            iso = dt.astimezone(ZoneInfo("America/New_York")).isoformat()
            epoch = str(int(dt.timestamp()))
            if dt.astimezone(ZoneInfo("America/New_York")).date() == today:
                state = "fresh"
        except Exception:
            iso = ""
            epoch = ""
print(f"{state}|{iso}|{epoch}")
PY
}

PIPELINE_STATE=$(check_pipeline)
IFS='|' read -r PIPE_STATE PIPE_ISO PIPE_EPOCH <<<"$PIPELINE_STATE"
if [[ "$PIPE_STATE" != "fresh" ]]; then
  log "pipeline summary stale -> running pipeline"
  "$PYTHON" -m scripts.run_pipeline --steps screener --reload-web false
  PIPELINE_STATE=$(check_pipeline)
  IFS='|' read -r PIPE_STATE PIPE_ISO PIPE_EPOCH <<<"$PIPELINE_STATE"
fi

if [[ -z "$PIPE_EPOCH" ]]; then
  log "unable to confirm pipeline metrics timestamp"
  exit 1
fi

log "pipeline summary fresh=${PIPE_STATE} timestamp=${PIPE_ISO:-unknown}"

if [[ -f "$WSGI_PATH" ]]; then
  touch "$WSGI_PATH"
  log "touched wsgi=${WSGI_PATH}"
else
  log "missing wsgi file=${WSGI_PATH}"
fi

update_status() {
  STATUS_NAME="$1" STATUS_VALUE="$2" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

status_path = Path(os.environ["PROJECT_HOME"]) / "data" / "pipeline_status.json"
status = {}
if status_path.exists():
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        status = {}
status[os.environ["STATUS_NAME"]] = int(float(os.environ["STATUS_VALUE"]))
status_path.parent.mkdir(parents=True, exist_ok=True)
status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
print(f"[WRAPPER] STATUS {os.environ['STATUS_NAME']}={status[os.environ['STATUS_NAME']]} written")
PY
}

update_status "Screener" "${PIPE_EPOCH}"

count_candidates() {
  SRC_PATH="$1" "$PYTHON" - <<'PY'
import csv
import os
from pathlib import Path

src = Path(os.environ["SRC_PATH"])
if not src.exists():
    print(0)
else:
    try:
        with src.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header
            rows = sum(1 for _ in reader)
    except Exception:
        rows = 0
    print(rows)
PY
}

[ -z "${SRC:-}" ] && SRC="data/latest_candidates.csv"
CAND_ROWS=$(count_candidates "$SRC")
if [[ "$CAND_ROWS" -le 0 ]]; then
  log "candidates empty; invoking fallback"
  "$PYTHON" -m scripts.fallback_candidates --top-n "${FALLBACK_TOP_N:-3}"
  CAND_ROWS=$(count_candidates "$SRC")
fi

if [[ "$CAND_ROWS" -le 0 ]]; then
  log "no candidates available after fallback"
  exit 1
fi

log "candidates ready count=${CAND_ROWS}"

EXEC_WINDOW="${EXEC_WINDOW:-auto}"
POSITION_SIZER="${POSITION_SIZER:-notional}"
ATR_TARGET_PCT="${ATR_TARGET_PCT:-0.02}"

"$PYTHON" -m scripts.execute_trades \
  --source "${SRC}" \
  --allocation-pct "${ALLOCATION_PCT:-0.06}" \
  --min-order-usd "${MIN_ORDER_USD:-300}" \
  --max-positions "${MAX_POSITIONS:-4}" \
  --trailing-percent "${TRAILING_PCT:-3}" \
  --time-window "${EXEC_WINDOW}" \
  --extended-hours true \
  --cancel-after-min "${CANCEL_AFTER_MIN:-35}" \
  --limit-buffer-pct "${LIMIT_BUFFER_PCT:-1.0}" \
  --position-sizer "${POSITION_SIZER}" \
  --atr-target-pct "${ATR_TARGET_PCT}"

update_status "Execution" "$(date +%s)"
log "execution complete"
