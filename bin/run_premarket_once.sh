#!/usr/bin/env bash
set -Eeuo pipefail

# Force New York timezone in this shell so hhmm comparisons are unambiguous
export TZ="America/New_York"

cd /home/RasPatrick/jbravo_screener
# venv + .env
source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate
set -a; . ~/.config/jbravo/.env; set +a

log(){ echo "[WRAPPER] $*"; }

# 1) Probe Alpaca auth
resp=$(python - <<'PY'
import os,requests
from urllib.parse import urljoin
b=os.environ["APCA_API_BASE_URL"].rstrip("/")
k=os.environ["APCA_API_KEY_ID"]; s=os.environ["APCA_API_SECRET_KEY"]
r=requests.get(urljoin(b,"/v2/account"),
               headers={"APCA-API-KEY-ID":k,"APCA-API-SECRET-KEY":s}, timeout=10)
print(r.status_code, (r.json().get("buying_power") if r.ok else "0"))
PY
)
status=${resp%% *}; bp=${resp#* }
log "AUTH_OK=$([ "$status" = "200" ] && echo true || echo false) buying_power=${bp}"

# 2) Ensure we have ≥1 candidate row (header+row)
CAND="data/latest_candidates.csv"
rows=$( (wc -l < "$CAND" 2>/dev/null) || echo 0 )
if [[ "${rows:-0}" -lt 2 ]]; then
  log "No candidates (rows=${rows:-0}); running fallback..."
  /home/RasPatrick/.virtualenvs/jbravo-env/bin/python -m scripts.fallback_candidates --top-n 3
fi

# 3) Gate on New York pre-market window (07:00–09:30 ET)
hhmm=$(date +%H%M)   # NY time due to TZ above
if [[ "$hhmm" -ge 0700 && "$hhmm" -lt 0930 ]]; then
  log "NY hhmm=$hhmm → premarket; launching executor"
  /home/RasPatrick/.virtualenvs/jbravo-env/bin/python -m scripts.execute_trades \
    --source "$CAND" \
    --allocation-pct 0.06 --min-order-usd 300 --max-positions 4 \
    --trailing-percent 3 --limit-buffer-pct 1.0 \
    --time-window premarket --extended-hours true --cancel-after-min 35
else
  log "NY hhmm=$hhmm → outside premarket; skip (no-op)"
fi

# 4) Refresh dashboard + status stamp
touch /var/www/raspatrick_pythonanywhere_com_wsgi.py
python - <<'PY'
import json, pathlib, datetime as dt
path=pathlib.Path("data/last_premarket_run.json")
path.write_text(json.dumps({"ts":dt.datetime.utcnow().isoformat()+"Z"}, indent=2))
print(str(path))
PY
log "done."
