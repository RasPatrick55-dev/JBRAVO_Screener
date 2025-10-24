#!/usr/bin/env bash
set -euo pipefail

export TZ="${TZ:-America/New_York}"   # for human-readable timestamps in logs
PROJECT_HOME="${PROJECT_HOME:-/home/RasPatrick/jbravo_screener}"
VENV="/home/RasPatrick/.virtualenvs/jbravo-env"

cd "$PROJECT_HOME"
source "$VENV/bin/activate"
set -a; . ~/.config/jbravo/.env; set +a

# Probe Alpaca (paper)
python - <<'PY'
import os, requests, datetime, json
from urllib.parse import urljoin
b=os.getenv("APCA_API_BASE_URL"); k=os.getenv("APCA_API_KEY_ID"); s=os.getenv("APCA_API_SECRET_KEY")
r=requests.get(urljoin(b,"/v2/account"),headers={"APCA-API-KEY-ID":k,"APCA-API-SECRET-KEY":s},timeout=10)
print(f"[WRAPPER] AUTH_OK={r.status_code==200} buying_power={r.json().get('buying_power') if r.ok else '0.00'}")
PY

# Ensure ≥1 candidate (robust wc)
SRC="data/latest_candidates.csv"
rows="$({ wc -l < "$SRC"; } 2>/dev/null || echo 0)"
if [[ "${rows:-0}" -lt 2 ]]; then
  python -m scripts.fallback_candidates --top-n 3
fi

# Execute once with auto window (premarket enforced by strategy)
python -m scripts.execute_trades \
  --source "$SRC" --allocation-pct 0.06 --min-order-usd 300 --max-positions 4 \
  --trailing-percent 3 --time-window auto --limit-buffer-pct 1.0 --extended-hours true \
  --cancel-after-min 35
