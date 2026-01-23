#!/bin/bash

cd /home/RasPatrick/jbravo_screener
source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate
set -a
. /home/RasPatrick/.config/jbravo/.env
set +a

python -m scripts.execute_trades \
  --source db \
  --allocation-pct 0.05 \
  --min-order-usd 300 \
  --max-positions 4 \
  --trailing-percent 3 \
  --time-window any \
  --extended-hours True \
  --cancel-after-min 35 \
  --limit-buffer-pct 1.0
