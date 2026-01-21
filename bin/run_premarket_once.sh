#!/usr/bin/env bash
set -Eeuo pipefail

export TZ="America/New_York"
PROJECT="${PROJECT:-/home/RasPatrick/jbravo_screener}"
VENV="${VENV:-/home/RasPatrick/.virtualenvs/jbravo-env}"
cd "$PROJECT"
source "$VENV/bin/activate"
set -a; . ~/.config/jbravo/.env; set +a

DRY_RUN="${JBRAVO_DRY_RUN:-false}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="${2:-true}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if ! grep -q "PIPELINE_END rc=0" "$PROJECT_HOME/logs/pipeline.log"; then
  echo "[WARN] PIPELINE_MISSING — executor skipped"
  echo "[WARN] Pipeline not completed; skipping trade execution." >> "$PROJECT_HOME/logs/execute_trades.log"
  python - <<PY
import json, datetime
json.dump({
  "timestamp": datetime.datetime.utcnow().isoformat(),
  "status": "SKIPPED_PIPELINE_MISSING"
}, open("$PROJECT_HOME/data/last_premarket_run.json","w"))
PY
  touch "$WSGI_FILE"
  exit 0
fi

if [[ "${APCA_API_BASE_URL:-}" != *paper* ]]; then
  echo "[ERROR] APCA_API_BASE_URL must be a paper endpoint: ${APCA_API_BASE_URL:-unset}" >&2
  exit 1
fi

echo "[WRAPPER] probing Alpaca credentials"
ALPACA_PROBE=$(python - <<'PY'
import os, requests, json
base=os.environ["APCA_API_BASE_URL"].rstrip("/")
headers={"APCA-API-KEY-ID":os.environ["APCA_API_KEY_ID"],"APCA-API-SECRET-KEY":os.environ["APCA_API_SECRET_KEY"]}
r=requests.get(f"{base}/v2/account", headers=headers, timeout=10)
ok=r.status_code==200; bp=(r.json().get("buying_power") if ok else "0")
print(json.dumps({"status":"OK" if ok else "FAIL","buying_power":bp,"auth_ok":ok}))
raise SystemExit(0 if ok else 2)
PY
)
probe_rc=$?
echo "$ALPACA_PROBE"
if [ "$probe_rc" -ne 0 ]; then
  exit "$probe_rc"
fi
export ALPACA_PROBE

echo "[WRAPPER] Alpaca account probe OK"

# Timestamp the wrapper run for snapshotting
PREMARKET_STARTED_UTC=$(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)
export PREMARKET_STARTED_UTC

# Helper to persist the latest premarket snapshot without failing the wrapper
write_premarket_snapshot() {
python - <<'PY'
import json, os, pathlib
from scripts.execute_trades import write_premarket_snapshot

probe_env = os.environ.get("ALPACA_PROBE", "{}")
try:
    probe_payload = json.loads(probe_env)
except Exception:
    probe_payload = {}

started = os.environ.get("PREMARKET_STARTED_UTC")
finished = os.environ.get("PREMARKET_FINISHED_UTC")

try:
    path = write_premarket_snapshot(pathlib.Path("."), probe_payload=probe_payload, started_utc=started, finished_utc=finished)
    print(f"[WRAPPER] premarket snapshot updated: {path}")
except Exception as exc:  # pragma: no cover - defensive wrapper guard
    print(f"[WRAPPER] failed to write premarket snapshot: {exc}")
PY
}

# Run Alpaca connectivity probe for Screener Health tab
python -m scripts.check_connection || echo "[WARN] connection probe failed (non-fatal)"

BASE_ALLOCATION_PCT=0.06
RISK_OUTPUT=$(BASE_ALLOCATION_PCT="$BASE_ALLOCATION_PCT" python - <<'PY'
import json
import os
import pathlib

base_alloc = float(os.environ.get("BASE_ALLOCATION_PCT", "0"))

try:
    ranker_eval = pathlib.Path("data/ranker_eval/latest.json")

    if not ranker_eval.exists():
        print("[WARN] RISK_SCALING_SKIPPED reason=missing_ranker_eval")
        print(f"{base_alloc:.4f}")
        raise SystemExit(0)

    try:
        payload = json.loads(ranker_eval.read_text())
    except Exception:
        print("[WARN] RISK_SCALING_SKIPPED reason=invalid_signal_quality value=read_error")
        print(f"{base_alloc:.4f}")
        raise SystemExit(0)

    signal_quality = payload.get("signal_quality")
    decile_lift = payload.get("decile_lift")

    multipliers = {"HIGH": 1.0, "MEDIUM": 1.0, "LOW": 0.5}
    multiplier = multipliers.get(signal_quality)

    if multiplier is None:
        print(f"[WARN] RISK_SCALING_SKIPPED reason=invalid_signal_quality value={signal_quality}")
        print(f"{base_alloc:.4f}")
        raise SystemExit(0)

    final_alloc = round(base_alloc * multiplier, 4)
    print(
        f"[INFO] RISK_SCALING signal_quality={signal_quality} decile_lift={decile_lift} "
        f"base_alloc={base_alloc} final_alloc={final_alloc}"
    )
    print(f"{final_alloc:.4f}")
except Exception as exc:  # pragma: no cover - defensive fallback
    print(f"[WARN] RISK_SCALING_SKIPPED reason=unexpected_error value={exc}")
    print(f"{base_alloc:.4f}")
PY
)
RISK_LOG_LINE=$(printf "%s\n" "$RISK_OUTPUT" | head -n1)
FINAL_ALLOCATION_PCT=$(printf "%s\n" "$RISK_OUTPUT" | tail -n1)

LOG_TIMESTAMP=$(PYTHONPATH="" python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:23])
PY
)
mkdir -p logs
printf '%s - wrapper - %s\n' "$LOG_TIMESTAMP" "$RISK_LOG_LINE" >> logs/execute_trades.log
echo "$RISK_LOG_LINE"

DEFAULT_MAX_POSITIONS=7
SIGNAL_QUALITY=$(printf "%s" "$RISK_LOG_LINE" | sed -n 's/.*signal_quality=\([A-Z]*\).*/\1/p')
FINAL_MAX_POSITIONS=$DEFAULT_MAX_POSITIONS
FINAL_MAX_NEW_POSITIONS=$DEFAULT_MAX_POSITIONS
if [ "$SIGNAL_QUALITY" = "LOW" ]; then
  FINAL_MAX_NEW_POSITIONS=1
fi

RISK_LIMIT_LOG_LINE="[INFO] RISK_LIMIT signal_quality=${SIGNAL_QUALITY:-UNKNOWN} max_new_positions=${FINAL_MAX_NEW_POSITIONS} (risk-limited) max_total_positions=${FINAL_MAX_POSITIONS}"
LOG_TIMESTAMP=$(PYTHONPATH="" python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:23])
PY
)
printf '%s - wrapper - %s\n' "$LOG_TIMESTAMP" "$RISK_LIMIT_LOG_LINE" >> logs/execute_trades.log
echo "$RISK_LIMIT_LOG_LINE"

EXEC_SOURCE="db"
EXEC_SOURCE_LOG_LINE="[INFO] EXEC_SOURCE path=${EXEC_SOURCE}"
LOG_TIMESTAMP=$(PYTHONPATH="" python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:23])
PY
)
printf '%s - wrapper - %s\n' "$LOG_TIMESTAMP" "$EXEC_SOURCE_LOG_LINE" >> logs/execute_trades.log
echo "$EXEC_SOURCE_LOG_LINE"

DRY_RUN_NORMALIZED=$(printf '%s' "$DRY_RUN" | tr '[:upper:]' '[:lower:]')
if [[ "$DRY_RUN_NORMALIZED" == "1" || "$DRY_RUN_NORMALIZED" == "true" || "$DRY_RUN_NORMALIZED" == "yes" || "$DRY_RUN_NORMALIZED" == "on" ]]; then
  LOG_TIMESTAMP=$(PYTHONPATH="" python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:23])
PY
  )
  DRY_RUN_LOG_LINE="[WARN] DRY_RUN_ENABLED skipping execute_trades"
  printf '%s - wrapper - %s\n' "$LOG_TIMESTAMP" "$DRY_RUN_LOG_LINE" >> logs/execute_trades.log
  echo "$DRY_RUN_LOG_LINE"

  PREMARKET_FINISHED_UTC=$(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
  )
  export PREMARKET_FINISHED_UTC

  write_premarket_snapshot
  exit 0
fi

python -m scripts.execute_trades \
  --source db \
  --time-window auto \
  --extended-hours true \
  --alloc-weight-key score \
  --allocation-pct 0.06 \
  --min-order-usd 300 \
  --max-positions 4 \
  --trailing-percent 3 \
  --limit-buffer-pct 1.0 \
  --cancel-after-min 35

PREMARKET_FINISHED_UTC=$(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)
export PREMARKET_FINISHED_UTC

write_premarket_snapshot

touch /var/www/raspatrick_pythonanywhere_com_wsgi.py || true
