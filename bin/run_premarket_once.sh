#!/usr/bin/env bash
# Robust shell settings
set -euo pipefail
IFS=$'\n\t'

# Load venv + .env
# (assumes PA tasks call this script directly)
export TZ="${TZ:-UTC}"
cd /home/RasPatrick/jbravo_screener
source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate || true
set -a
. ~/.config/jbravo/.env
set +a

# --- helper: print NY time for logs
NY_NOW="$(python - <<'PY'
from datetime import datetime
import pytz
ny = pytz.timezone("America/New_York")
print(datetime.now(ny).strftime("%Y-%m-%d %H:%M:%S %Z"))
PY
)"

# Probe Alpaca auth quickly
python - <<'PY'
import os
import requests
from urllib.parse import urljoin

base = os.getenv("APCA_API_BASE_URL")
key = os.getenv("APCA_API_KEY_ID")
secret = os.getenv("APCA_API_SECRET_KEY")
resp = requests.get(
    urljoin(base, "/v2/account"),
    headers={
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    },
    timeout=10,
)
ok = resp.status_code == 200
buying_power = "0"
if ok:
    try:
        payload = resp.json() or {}
    except ValueError:
        payload = {}
    buying_power = payload.get("buying_power", "0")
print(f"[WRAPPER] AUTH_OK={ok} buying_power={buying_power}")
PY

# If today's pipeline missing, run it (best-effort)
python - <<'PY' || true
from pathlib import Path

log_path = Path("logs/pipeline.log")
need = True
if log_path.exists():
    tail = log_path.read_text(encoding="utf-8")[-8000:]
    need = "PIPELINE_END rc=0" not in tail
print("[WRAPPER] PIPELINE_DONE_TODAY=", str(not need))
PY

# Ensure ≥1 candidate row
SRC="data/latest_candidates.csv"
rows="$(wc -l < "$SRC" 2>/dev/null || echo 0)"
if ! [[ "$rows" =~ ^[0-9]+$ ]]; then
  rows=0
fi
if [ "$rows" -lt 2 ]; then
  python -m scripts.fallback_candidates --top-n 3 || true
fi

# Execute (auto resolves by NY time); extended-hours on
# Optional env to widen premarket window (defaults to 07:00–09:30)
#   JBRAVO_PREMARKET_START=0400
#   JBRAVO_PREMARKET_END=0930
echo "[WRAPPER] NY_NOW=${NY_NOW}"
python -m scripts.execute_trades \
  --source data/latest_candidates.csv \
  --allocation-pct 0.06 --min-order-usd 300 --max-positions 4 \
  --trailing-percent 3 --time-window auto --extended-hours true \
  --cancel-after-min 35 --limit-buffer-pct 1.0 || true

echo "[WRAPPER] done."

