#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  echo "[WRAPPER] $*"
}

# Hold the wrapper until the New York pre-market is open to avoid hitting the
# executor TIME_WINDOW guard too early.
# --- Wait until NY pre-market opens (07:00 NY) ---
python - <<'PY'
import datetime, zoneinfo, time, sys

ny = zoneinfo.ZoneInfo("America/New_York")
now = datetime.datetime.now(ny)
open_t = now.replace(hour=7, minute=0, second=0, microsecond=0)
close_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
if now < open_t:
    sleep_s = (open_t - now).total_seconds()
    print(f"[WRAPPER] Pre-market not open. Sleeping {int(sleep_s)}s until 07:00 NY...")
    sys.stdout.flush()
    time.sleep(sleep_s)
elif now >= close_t:
    print("[WRAPPER] Pre-market window closed in NY; exiting.")
    sys.exit(0)
else:
    print(f"[WRAPPER] Pre-market open (NY now={now}).")
PY

send_alert() {
  local msg="$1"
  if [[ -z "${ALERT_WEBHOOK_URL:-}" || -z "$msg" ]]; then
    return
  fi
  MESSAGE="$msg" "$PYTHON" - <<'PY'
import os

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - requests unavailable
    raise SystemExit(0)

message = os.environ.get("MESSAGE", "")
url = os.environ.get("ALERT_WEBHOOK_URL", "")
if not message or not url:
    raise SystemExit(0)
try:
    requests.post(url, json={"text": message}, timeout=5)
except Exception:
    pass
PY
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
RUN_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$PROJECT_HOME"
source "$VENV/bin/activate"
set -a; . ~/.config/jbravo/.env; set +a

# --- Freshness check (trust nightly) ---
is_fresh_py="$($PYTHON - <<'PY'
import pathlib, datetime as dt, zoneinfo, re

ny = zoneinfo.ZoneInfo("America/New_York")
now = dt.datetime.now(ny)
log = pathlib.Path("logs/pipeline.log")
latest = pathlib.Path("data/latest_candidates.csv")
metrics = pathlib.Path("data/screener_metrics.json")


def ok_csv(path: pathlib.Path) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
    except Exception:
        return False
    need = {
        "timestamp",
        "symbol",
        "score",
        "exchange",
        "close",
        "volume",
        "universe_count",
        "score_breakdown",
        "entry_price",
        "adv20",
        "atrp",
        "source",
    }
    return need.issubset({col.strip().lower() for col in header})


def todays_pipeline_ok() -> bool:
    if not log.exists():
        return False
    try:
        tail = log.read_text(encoding="utf-8", errors="ignore")[-8000:]
    except Exception:
        return False
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}.*PIPELINE_END rc=0", re.M)
    matches = pattern.findall(tail)
    if not matches:
        return False
    last_date = dt.date.fromisoformat(matches[-1])
    return last_date == now.date()


if todays_pipeline_ok() and latest.exists() and ok_csv(latest) and metrics.exists():
    print("FRESH", end="")
else:
    print("STALE", end="")
PY
)"
if [[ "$is_fresh_py" == "FRESH" ]]; then
  log "nightly artifacts fresh; skipping pipeline re-run"
else
  log "nightly artifacts stale; running (fast) refresh"
  if ! "$PYTHON" -m scripts.screener --mode screener --reuse-cache true --skip-fetch true; then
    log "fast refresh failed; falling back to full screener pipeline"
    "$PYTHON" -m scripts.run_pipeline --steps screener --reload-web false || true
  fi
fi

# Sanity check that the latest screener output exists before trading.
rows=$(wc -l < data/latest_candidates.csv 2>/dev/null || echo 0)
if [[ "$rows" -lt 2 ]]; then
  log "latest_candidates.csv missing or too small (rows=${rows}); rerunning screener fallback."
  "$PYTHON" -m scripts.run_pipeline --steps screener --reload-web false
fi

log "probing Alpaca credentials"
AUTH_OUTPUT="$("$PYTHON" - <<'PY'
import json
import os
from urllib.parse import urljoin

import requests

base = os.getenv("APCA_API_BASE_URL", "").strip()
key = os.getenv("APCA_API_KEY_ID", "")
secret = os.getenv("APCA_API_SECRET_KEY", "")
status = "ERR"
buying_power = "0.00"
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
payload = {"status": status, "buying_power": buying_power, "auth_ok": status == "OK"}
print(f"[WRAPPER] AUTH status={status} buying_power={buying_power}")
print(json.dumps(payload))
PY
)"
printf '%s\n' "$AUTH_OUTPUT"
AUTH_JSON=$(printf '%s\n' "$AUTH_OUTPUT" | tail -n1)
[ -z "$AUTH_JSON" ] && AUTH_JSON='{}'

check_pipeline() {
  "$PYTHON" - <<'PY'
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

base = Path(".")
metrics_path = base / "data" / "screener_metrics.json"
log_path = base / "logs" / "pipeline.log"
tz = ZoneInfo("America/New_York")
today = datetime.now(tz).date()

metrics_iso = ""
metrics_epoch = ""
metrics_local = None
if metrics_path.exists():
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    ts = payload.get("last_run_utc")
    if isinstance(ts, str) and ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            local_dt = dt.astimezone(tz)
            metrics_iso = local_dt.isoformat()
            metrics_epoch = str(int(local_dt.timestamp()))
            metrics_local = local_dt
        except Exception:
            metrics_iso = ""
            metrics_epoch = ""
            metrics_local = None

state = "stale"
iso = metrics_iso
epoch = metrics_epoch
rc_text = ""
if log_path.exists():
    try:
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-200000:]
    except Exception:
        tail = ""
    rc_pattern = re.compile(r"rc=(\d+)")
    stamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:Z)?)")
    for line in reversed(tail.splitlines()):
        if "PIPELINE_END" not in line:
            continue
        rc_match = rc_pattern.search(line)
        stamp_match = stamp_pattern.search(line)
        dt_local = None
        if stamp_match:
            stamp = stamp_match.group(1)
            try:
                iso_stamp = stamp.replace(" ", "T")
                dt = datetime.fromisoformat(iso_stamp.replace("Z", "+00:00"))
                dt_local = dt.astimezone(tz)
            except Exception:
                dt_local = None
        if dt_local is None or dt_local.date() != today:
            continue
        iso = dt_local.isoformat()
        epoch = str(int(dt_local.timestamp()))
        if rc_match:
            rc_text = rc_match.group(1)
        state = "fresh" if rc_text == "0" else "stale"
        break

if state != "fresh" and metrics_local is not None and metrics_local.date() == today:
    state = "fast"
    iso = metrics_local.isoformat()
    epoch = str(int(metrics_local.timestamp()))
    if not rc_text:
        rc_text = "fast"

print(f"{state}|{iso}|{epoch}|{rc_text}")
PY
}

PIPELINE_STATE=$(check_pipeline)
IFS='|' read -r PIPE_STATE PIPE_ISO PIPE_EPOCH PIPE_RC <<<"$PIPELINE_STATE"
if [[ "$PIPE_STATE" != "fresh" && "$PIPE_STATE" != "fast" ]]; then
  log "pipeline summary stale -> running pipeline"
  "$PYTHON" -m scripts.run_pipeline --steps screener --reload-web false
  PIPELINE_STATE=$(check_pipeline)
  IFS='|' read -r PIPE_STATE PIPE_ISO PIPE_EPOCH PIPE_RC <<<"$PIPELINE_STATE"
fi

if [[ "$PIPE_STATE" != "fresh" && "$PIPE_STATE" != "fast" ]]; then
  log "pipeline end token missing or failed (state=${PIPE_STATE} rc=${PIPE_RC:-unknown})"
  send_alert "Premarket wrapper aborted: no PIPELINE_END rc=0 today (rc=${PIPE_RC:-unknown})"
  exit 1
fi

if [[ -z "$PIPE_EPOCH" ]]; then
  log "unable to confirm pipeline metrics timestamp"
  send_alert "Premarket wrapper aborted: missing pipeline timestamp"
  exit 1
fi

log "pipeline summary fresh=${PIPE_STATE} timestamp=${PIPE_ISO:-unknown} rc=${PIPE_RC:-unknown}"
export PIPE_RC

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

inspect_candidates() {
  SRC_PATH="$1" "$PYTHON" - <<'PY'
import csv
import os
from pathlib import Path

from scripts.fallback_candidates import CANONICAL_COLUMNS

src = Path(os.environ["SRC_PATH"])
rows = 0
header_ok = False
header_only = False
if src.exists() and src.stat().st_size > 0:
    try:
        with src.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            normalized = [str(col).strip().lower() for col in header]
            header_ok = normalized == [col.lower() for col in CANONICAL_COLUMNS]
            for _ in reader:
                rows += 1
        header_only = rows == 0
    except Exception:
        rows = 0
        header_ok = False
        header_only = False
print(f"{rows}|{int(header_ok)}|{int(header_only)}")
PY
}

[ -z "${SRC:-}" ] && SRC="data/latest_candidates.csv"
IFS='|' read -r CAND_ROWS HEADER_OK HEADER_ONLY <<<"$(inspect_candidates "$SRC")"
if [[ "$HEADER_OK" -ne 1 || "$HEADER_ONLY" -eq 1 || "$CAND_ROWS" -le 0 ]]; then
  log "candidates invalid header_ok=${HEADER_OK} header_only=${HEADER_ONLY} rows=${CAND_ROWS}; invoking fallback"
  "$PYTHON" -m scripts.fallback_candidates --top-n "${FALLBACK_TOP_N:-3}"
  IFS='|' read -r CAND_ROWS HEADER_OK HEADER_ONLY <<<"$(inspect_candidates "$SRC")"
fi

if [[ "$CAND_ROWS" -le 0 || "$HEADER_OK" -ne 1 ]]; then
  log "no candidates available after fallback"
  send_alert "Premarket wrapper aborted: fallback candidates empty"
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

export AUTH_JSON PIPE_STATE PIPE_ISO PIPE_RC CAND_ROWS RUN_STARTED_UTC EXEC_WINDOW
"$PYTHON" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from scripts.execute_trades import _resolve_time_window

base = Path(os.environ["PROJECT_HOME"])
data_dir = base / "data"
data_dir.mkdir(parents=True, exist_ok=True)

auth_raw = os.environ.get("AUTH_JSON", "{}")
try:
    auth_info = json.loads(auth_raw)
except json.JSONDecodeError:
    auth_info = {}

window_req = os.environ.get("EXEC_WINDOW", "auto")
window, in_window, now = _resolve_time_window(window_req)

payload = {
    "started_utc": os.environ.get("RUN_STARTED_UTC"),
    "ny_now": now.isoformat(),
    "window": window,
    "in_window": bool(in_window),
    "candidates_in": int(os.environ.get("CAND_ROWS", "0") or 0),
    "auth_ok": bool(auth_info.get("auth_ok")),
    "auth_status": auth_info.get("status"),
    "buying_power": auth_info.get("buying_power"),
    "orders_submitted": 0,
    "skip_counts": {},
    "pipeline_state": os.environ.get("PIPE_STATE"),
    "pipeline_rc": os.environ.get("PIPE_RC"),
    "pipeline_timestamp": os.environ.get("PIPE_ISO"),
    "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
}

metrics_path = data_dir / "execute_metrics.json"
if metrics_path.exists():
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}
    orders_val = metrics.get("orders_submitted")
    try:
        payload["orders_submitted"] = int(float(orders_val)) if orders_val is not None else 0
    except Exception:
        payload["orders_submitted"] = 0
    skip_map = metrics.get("skip_reasons") or metrics.get("skips")
    if isinstance(skip_map, dict):
        cleaned: dict[str, int] = {}
        for key, value in skip_map.items():
            try:
                cleaned[str(key)] = int(float(value))
            except Exception:
                continue
        payload["skip_counts"] = cleaned

output_path = data_dir / "last_premarket_run.json"
output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"[WRAPPER] wrote {output_path}")
PY
