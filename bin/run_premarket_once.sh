#!/usr/bin/env bash
set -Eeuo pipefail

export TZ="America/New_York"
PROJECT="${PROJECT:-/home/RasPatrick/jbravo_screener}"
VENV="${VENV:-/home/RasPatrick/.virtualenvs/jbravo-env}"
cd "$PROJECT"
source "$VENV/bin/activate"
set -a; . ~/.config/jbravo/.env; set +a

# (1) Wait until NY pre-market is open (07:00–09:30)
python - <<'PY'
import sys, time, datetime as dt, zoneinfo
ny=zoneinfo.ZoneInfo("America/New_York")
now=dt.datetime.now(ny)
open_t=now.replace(hour=7, minute=0, second=0, microsecond=0)
close_t=now.replace(hour=9, minute=30, second=0, microsecond=0)
if now < open_t:
    sleep_s=(open_t-now).total_seconds()
    print(f"[WRAPPER] Pre-market not open (NY). Sleeping {int(sleep_s)}s…")
    sys.stdout.flush(); time.sleep(sleep_s)
elif now >= close_t:
    print("[WRAPPER] Pre-market window closed (NY); exiting.")
    sys.exit(0)
else:
    print(f"[WRAPPER] Pre-market open (NY now={now}).")
PY

# (2) Freshness decision: MUST NOT re-run pipeline here
echo "[WRAPPER] probing Alpaca credentials"
python - <<'PY'
import os, requests, json
b=os.environ["APCA_API_BASE_URL"].rstrip("/")
h={"APCA-API-KEY-ID":os.environ["APCA_API_KEY_ID"],"APCA-API-SECRET-KEY":os.environ["APCA_API_SECRET_KEY"]}
r=requests.get(f"{b}/v2/account", headers=h, timeout=10)
ok=r.status_code==200; bp=(r.json().get("buying_power") if ok else "0")
print(json.dumps({"status":"OK" if ok else "FAIL","buying_power":bp,"auth_ok":ok}))
PY

# NEVER run the pipeline here; rely on nightly stamp only
fresh=$(python - <<'PY'
import json, pathlib, datetime as dt, zoneinfo, sys
ny=zoneinfo.ZoneInfo("America/New_York")
today=dt.datetime.now(ny).date()
p=pathlib.Path("data/pipeline_fresh.json")
latest=pathlib.Path("data/latest_candidates.csv")
metrics=pathlib.Path("data/screener_metrics.json")
def ok_csv(path):
    try:
        with path.open() as f:
            header=f.readline().strip().lower().split(",")
        need={"timestamp","symbol","score","exchange","close","volume","universe_count","score_breakdown","entry_price","adv20","atrp","source"}
        return need.issubset(set(header))
    except Exception: return False
if not (p.exists() and latest.exists() and metrics.exists() and ok_csv(latest)):
    print("STALE"); sys.exit(0)
try:
    j=json.loads(p.read_text() or "{}")
except Exception:
    print("STALE"); sys.exit(0)
if j.get("rc")!=0 or not j.get("ny_date"):
    print("STALE"); sys.exit(0)
print("FRESH" if j["ny_date"]==today.isoformat() and (j.get("rows") or 0)>0 else "STALE")
PY
)
if [[ "$fresh" != "FRESH" ]]; then
  echo "[WRAPPER] nightly artifacts STALE or missing; NO trading. (Fix Run Pipeline first.)"
  python - <<'PY'
import json, pathlib, datetime as dt
p=pathlib.Path("data/execute_metrics.json")
now=dt.datetime.utcnow().isoformat()+"Z"
m={"last_run_utc":now,"orders_submitted":0,"orders_filled":0,"skips":{"NIGHTLY_STALE":1}}
p.write_text(json.dumps(m, indent=2)); print("[WRAPPER] wrote execute_metrics.json (NIGHTLY_STALE)")
PY
  exit 0
fi
echo "[WRAPPER] nightly artifacts fresh; skipping pipeline re-run"

# (3) Consume nightly candidates only (no fallback generation here)
rows=$(wc -l < data/latest_candidates.csv 2>/dev/null || echo 0)
if [ -z "$rows" ] || [ "$rows" -lt 2 ]; then
  echo "[WRAPPER] ERROR: nightly latest_candidates.csv empty; NO trading."
  exit 0
fi

# (4) Execute (paper mode)
python -m scripts.execute_trades \
  --source data/latest_candidates.csv \
  --allocation-pct 0.06 --min-order-usd 300 --max-positions 4 \
  --trailing-percent 3 --time-window premarket --extended-hours true \
  --cancel-after-min 35 --limit-buffer-pct 1.0

touch /var/www/raspatrick_pythonanywhere_com_wsgi.py || true
