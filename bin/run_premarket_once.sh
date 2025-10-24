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
rows="$(wc -l < "${SRC}" 2>/dev/null || echo 0)"
rows="${rows:-0}"
if [ "${rows}" -lt 2 ]; then
  echo "[WRAPPER] candidates header-only; running fallback..."
  /home/RasPatrick/.virtualenvs/jbravo-env/bin/python -m scripts.fallback_candidates --top-n 3
fi

# Force pre-market window to avoid “window=closed” when scheduled properly.
# (You may switch back to 'auto' once schedule is proven correct.)
EXEC_WINDOW="${EXEC_WINDOW:-premarket}"

/home/RasPatrick/.virtualenvs/jbravo-env/bin/python -m scripts.execute_trades \
  --source "${SRC}" \
  --allocation-pct "${ALLOCATION_PCT:-0.06}" \
  --min-order-usd "${MIN_ORDER_USD:-300}" \
  --max-positions "${MAX_POSITIONS:-4}" \
  --trailing-percent "${TRAILING_PCT:-3}" \
  --time-window "${EXEC_WINDOW}" \
  --extended-hours true \
  --cancel-after-min "${CANCEL_AFTER_MIN:-35}" \
  --limit-buffer-pct "${LIMIT_BUFFER_PCT:-1.0}"
