#!/usr/bin/env bash
set -Eeuo pipefail

export TZ="America/New_York"

# Project paths
PROJECT_HOME="${PROJECT_HOME:-/home/RasPatrick/jbravo_screener}"
VENV="${VENV:-/home/RasPatrick/.virtualenvs/jbravo-env}"

cd "$PROJECT_HOME"
source "$VENV/bin/activate"

# Load environment (single source of truth)
set -a
. ~/.config/jbravo/.env
set +a

# Fail fast if Alpaca env missing
: "${APCA_API_KEY_ID:?Missing APCA_API_KEY_ID}"
: "${APCA_API_SECRET_KEY:?Missing APCA_API_SECRET_KEY}"
: "${APCA_API_BASE_URL:?Missing APCA_API_BASE_URL}"

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

# Require successful pipeline
if ! grep -q "PIPELINE_END rc=0" "$PROJECT_HOME/logs/pipeline.log"; then
  echo "[WARN] Pipeline not completed; skipping trade execution." >> "$PROJECT_HOME/logs/execute_trades.log"
  exit 0
fi

echo "[WRAPPER] probing Alpaca credentials"
ALPACA_PROBE=$(python - <<'PY'
import os, requests, json
base = os.environ["APCA_API_BASE_URL"].rstrip("/")
headers = {
    "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
    "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
}
r = requests.get(f"{base}/v2/account", headers=headers, timeout=10)
ok = r.status_code == 200
bp = r.json().get("buying_power") if ok else "0"
print(json.dumps({"status": "OK" if ok else "FAIL", "buying_power": bp, "auth_ok": ok}))
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

# Timestamp run
PREMARKET_STARTED_UTC=$(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)
export PREMARKET_STARTED_UTC

# Health probe (non-fatal)
python -m scripts.check_connection || echo "[WARN] connection probe failed (non-fatal)"

# Execute trades
python -m scripts.execute_trades \
  --source db \
    --price-source blended \
  --time-window auto \
  --extended-hours true \
  --alloc-weight-key score \
  --allocation-pct 0.06 \
  --min-order-usd 300 \
  --max-positions 4 \
  --trailing-percent 3 \
  --cancel-after-min 35

PREMARKET_FINISHED_UTC=$(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)
export PREMARKET_FINISHED_UTC

# Persist snapshot (best effort)
python - <<'PY'
import json, os, pathlib
from scripts.execute_trades import write_premarket_snapshot

try:
    probe = json.loads(os.environ.get("ALPACA_PROBE", "{}"))
except Exception:
    probe = {}

write_premarket_snapshot(
    pathlib.Path("."),
    probe_payload=probe,
    started_utc=os.environ.get("PREMARKET_STARTED_UTC"),
    finished_utc=os.environ.get("PREMARKET_FINISHED_UTC"),
)
print("[WRAPPER] premarket snapshot updated")
PY

# Reload dashboard
touch /var/www/raspatrick_pythonanywhere_com_wsgi.py || true
