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

PIPELINE_LOG="logs/pipeline.log"
if ! grep -q "PIPELINE_END rc=0" "$PIPELINE_LOG"; then
  echo "[WARN] PIPELINE_MISSING — executor skipped"
  python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "status": "SKIPPED_PIPELINE_MISSING",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
path = Path("data") / "last_premarket_run.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
  touch /var/www/raspatrick_pythonanywhere_com_wsgi.py || true
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

count_candidates() {
python - <<'PY'
from pathlib import Path
path = Path("data") / "latest_candidates.csv"
count = 0
try:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                count += 1
except FileNotFoundError:
    count = 0
print(count)
PY
}

CANDIDATE_LINES=$(count_candidates)
if [ "$CANDIDATE_LINES" -lt 2 ]; then
  echo "[WRAPPER] latest_candidates.csv missing or short -> running fallback_candidates"
  python -m scripts.fallback_candidates
  CANDIDATE_LINES=$(count_candidates)
  if [ "$CANDIDATE_LINES" -lt 2 ]; then
    echo "[ERROR] latest_candidates.csv still missing or short after fallback" >&2
    exit 1
  fi
fi

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

EXEC_SOURCE="data/latest_candidates.csv"
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
  --source "$EXEC_SOURCE" \
  --allocation-pct "$FINAL_ALLOCATION_PCT" --min-order-usd 300 --max-positions "$FINAL_MAX_POSITIONS" --max-new-positions "$FINAL_MAX_NEW_POSITIONS" \
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
