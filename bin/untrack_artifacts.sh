#!/usr/bin/env bash
set -euo pipefail

cat <<'MSG'
This script removes already-tracked runtime artifacts from Git index.
It does NOT delete local files from disk.

Reviewing and running:
  git rm -r --cached logs reports || true
  git rm -r --cached frontend/node_modules || true
  git rm --cached data/latest_candidates.csv data/execute_metrics.json || true

Then commit once, for example:
  git commit -m "chore(repo): stop tracking runtime artifacts"
MSG

git rm -r --cached logs reports || true
git rm -r --cached frontend/node_modules || true
git rm --cached data/latest_candidates.csv data/execute_metrics.json || true
