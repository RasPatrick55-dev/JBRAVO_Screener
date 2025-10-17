#!/bin/bash
rows=$(wc -l < data/latest_candidates.csv 2>/dev/null || echo 0)
if [ "$rows" -lt 2 ]; then
  python -m scripts.fallback_candidates --top-n 3 --min-order-usd 300 || true
fi
