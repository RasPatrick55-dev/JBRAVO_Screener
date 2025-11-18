#!/usr/bin/env bash
set -Eeuo pipefail

export TZ="America/New_York"
PROJECT="${PROJECT:-/home/RasPatrick/jbravo_screener}"
VENV="${VENV:-/home/RasPatrick/.virtualenvs/jbravo-env}"
cd "$PROJECT"
source "$VENV/bin/activate"
set -a; . ~/.config/jbravo/.env; set +a

echo "[WRAPPER] probing Alpaca credentials"
python - <<'PY'
import os, requests, json
base=os.environ["APCA_API_BASE_URL"].rstrip("/")
headers={"APCA-API-KEY-ID":os.environ["APCA_API_KEY_ID"],"APCA-API-SECRET-KEY":os.environ["APCA_API_SECRET_KEY"]}
r=requests.get(f"{base}/v2/account", headers=headers, timeout=10)
ok=r.status_code==200; bp=(r.json().get("buying_power") if ok else "0")
print(json.dumps({"status":"OK" if ok else "FAIL","buying_power":bp,"auth_ok":ok}))
raise SystemExit(0 if ok else 2)
PY

python - <<'PY'
import json, os, pathlib, sys, time
root=pathlib.Path('.')
latest=root/"data"/"latest_candidates.csv"
metrics=root/"data"/"screener_metrics.json"
hours=int(os.environ.get("JBRAVO_STALE_MAX_HOURS","26"))
now=time.time()
issues=[]
def too_stale(path: pathlib.Path) -> bool:
    try:
        return (now-path.stat().st_mtime) > hours*3600
    except FileNotFoundError:
        return True
line_count=0
if latest.exists():
    try:
        with latest.open() as handle:
            for _ in range(2):
                if handle.readline():
                    line_count+=1
    except Exception:
        line_count=0
if not latest.exists() or line_count < 2:
    issues.append("NO_CANDIDATES")
elif too_stale(latest):
    issues.append("STALE_CANDIDATES")
if not metrics.exists():
    issues.append("NO_METRICS")
elif too_stale(metrics):
    issues.append("STALE_METRICS")
if issues:
    summary={"orders_submitted":0,"orders_filled":0,"skips":{"STALE_ARTIFACTS":1}}
    out=root/"data"/"execute_metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[WRAPPER] artifacts stale or missing -> exit. issues={issues}")
    print("EXECUTE_SUMMARY orders_submitted=0 orders_filled=0 skips={'STALE_ARTIFACTS': 1}")
    sys.exit(10)
print("[WRAPPER] artifacts OK; proceeding to execution")
PY
rc=$?
if [ "$rc" -eq 10 ]; then
  exit 0
elif [ "$rc" -ne 0 ]; then
  exit "$rc"
fi

python -m scripts.execute_trades \
  --source data/latest_candidates.csv \
  --allocation-pct 0.06 --min-order-usd 300 --max-positions 7 \
  --trailing-percent 3 --time-window premarket --extended-hours true \
  --submit-at-ny "07:00" --price-source prevclose \
  --cancel-after-min 35 --limit-buffer-pct 0.0

touch /var/www/raspatrick_pythonanywhere_com_wsgi.py || true
