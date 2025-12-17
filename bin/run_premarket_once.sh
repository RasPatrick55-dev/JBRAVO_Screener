#!/usr/bin/env bash
set -Eeuo pipefail

export TZ="America/New_York"
PROJECT="${PROJECT:-/home/RasPatrick/jbravo_screener}"
VENV="${VENV:-/home/RasPatrick/.virtualenvs/jbravo-env}"
cd "$PROJECT"
source "$VENV/bin/activate"
set -a; . ~/.config/jbravo/.env; set +a

echo "[WRAPPER] checking pipeline freshness"
set +e
PIPELINE_INFO=$(python - <<'PY'
import json
import pathlib
from datetime import datetime
import zoneinfo

base = pathlib.Path('.')
summary = base / 'reports' / 'pipeline_summary.json'
ny = zoneinfo.ZoneInfo('America/New_York')
today = datetime.now(ny).date()

fresh = False
reason = 'missing_summary'

try:
    data = json.loads(summary.read_text())
    rc = int(data.get('rc', 1))
    ts = data.get('timestamp')
    ts_date = None
    if isinstance(ts, str) and ts:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            ts_date = dt.astimezone(ny).date()
        except Exception:
            ts_date = None
    fresh = rc == 0 and ts_date == today
    if not fresh:
        reason = f"rc={rc} ts_date={ts_date}"
except FileNotFoundError:
    reason = 'missing_summary'
except Exception as exc:
    reason = f"read_error:{exc}"

print(json.dumps({'fresh': fresh, 'reason': reason}))
raise SystemExit(0 if fresh else 1)
PY
)
PIPELINE_READY=$?
set -e
PIPELINE_REASON=$(PIPELINE_INFO="$PIPELINE_INFO" python - <<'PY'
import json
import os

payload = json.loads(os.environ.get('PIPELINE_INFO', '{}') or '{}')
print(payload.get('reason', ''))
PY
)

if [ "$PIPELINE_READY" -ne 0 ]; then
  echo "[WRAPPER] pipeline stale -> re-running (reason: $PIPELINE_REASON)"
  if ! python -m scripts.run_pipeline; then
    echo "[WRAPPER] pipeline rerun failed"
    exit 1
  fi
  PIPELINE_RERUN_TS=$(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
  )
  export PIPELINE_REASON PIPELINE_RERUN_TS
  python - <<'PY'
import os
from utils.alerts import send_alert

send_alert(
    "JBRAVO wrapper auto-ran pipeline",
    {
        "reason": os.environ.get("PIPELINE_REASON", "unknown"),
        "run_utc": os.environ.get("PIPELINE_RERUN_TS"),
    },
)
PY
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
  PREMARKET_FINISHED_UTC=$(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
  )
  export PREMARKET_FINISHED_UTC
  write_premarket_snapshot
  exit 0
elif [ "$rc" -ne 0 ]; then
  exit "$rc"
fi

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
if [ "$SIGNAL_QUALITY" = "LOW" ]; then
  FINAL_MAX_POSITIONS=1
fi

RISK_LIMIT_LOG_LINE="[INFO] RISK_LIMIT signal_quality=${SIGNAL_QUALITY:-UNKNOWN} max_positions=${FINAL_MAX_POSITIONS}"
LOG_TIMESTAMP=$(PYTHONPATH="" python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:23])
PY
)
printf '%s - wrapper - %s\n' "$LOG_TIMESTAMP" "$RISK_LIMIT_LOG_LINE" >> logs/execute_trades.log
echo "$RISK_LIMIT_LOG_LINE"

EXEC_SOURCE="data/latest_candidates.csv"
EXEC_SOURCE_LOG_LINE="[INFO] EXEC_SOURCE path=${EXEC_SOURCE}"
LOG_TIMESTAMP=$(PYTHONPATH="" python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:23])
PY
)
printf '%s - wrapper - %s\n' "$LOG_TIMESTAMP" "$EXEC_SOURCE_LOG_LINE" >> logs/execute_trades.log
echo "$EXEC_SOURCE_LOG_LINE"

python -m scripts.execute_trades \
  --source "$EXEC_SOURCE" \
  --allocation-pct "$FINAL_ALLOCATION_PCT" --min-order-usd 300 --max-positions "$FINAL_MAX_POSITIONS" \
  --trailing-percent 3 --time-window premarket --extended-hours true \
  --submit-at-ny "07:00" --price-source prevclose \
  --cancel-after-min 35 --limit-buffer-pct 0.0

PREMARKET_FINISHED_UTC=$(python - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)
export PREMARKET_FINISHED_UTC

write_premarket_snapshot

touch /var/www/raspatrick_pythonanywhere_com_wsgi.py || true
