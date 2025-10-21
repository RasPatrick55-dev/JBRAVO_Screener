#!/usr/bin/env bash
set -euo pipefail
# Task-safe nightly pipeline launcher for PythonAnywhere.
# No "source" or backslashes; uses venv python directly.
BASE="/home/RasPatrick/jbravo_screener"
VENV_PY="/home/RasPatrick/.virtualenvs/jbravo-env/bin/python"
cd "$BASE"
# Run pipeline with split args (avoids quoting headaches in schedulers)
"$VENV_PY" -m scripts.run_pipeline \
  --steps screener,backtest,metrics \
  --screener-args-split --feed iex --dollar-vol-min 2000000 --reuse-cache true

# Ensure KPIs are non-null even on strict nights (writes screener_metrics.json)
"$VENV_PY" - <<'PY'
from pathlib import Path
from scripts.run_pipeline import write_complete_screener_metrics
print(write_complete_screener_metrics(Path(".")))
PY

exit 0
