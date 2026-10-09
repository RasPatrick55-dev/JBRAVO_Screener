#!/bin/bash
# Credentials use the existing process environment; never echo them.
set -euo pipefail
cd /home/RasPatrick/jbravo_screener
source /home/RasPatrick/.virtualenvs/jbravo-env/bin/activate
set -a
. /home/RasPatrick/.config/jbravo/.env
set +a
exec python -m scripts.account_sync
