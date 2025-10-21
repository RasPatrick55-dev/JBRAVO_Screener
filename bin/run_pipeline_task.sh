#!/usr/bin/env bash
set -euo pipefail
cd /home/RasPatrick/jbravo_screener

# Use the venv's Python path — this is the PythonAnywhere-recommended way for tasks.
VENV_PY="/home/RasPatrick/.virtualenvs/jbravo-env/bin/python"

# Fresh log for the dashboard Pipeline tab
: > logs/pipeline.log

exec "$VENV_PY" -m scripts.run_pipeline \
  --steps screener,backtest,metrics \
  --screener-args-split --feed iex --dollar-vol-min 2000000 --reuse-cache true
