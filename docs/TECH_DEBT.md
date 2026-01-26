# Tech Debt

## Active Items

1. Legacy React sources exist in `src/` alongside the active Vite app in
   `frontend/src`. Confirm no consumers still import from the legacy tree, then
   remove `src/` to avoid duplicate component edits.
2. Frontend builds are still manual on PythonAnywhere. Consider adding a CI job
   to build `frontend/dist` and promote artifacts, or add a deploy script that
   rebuilds and reloads the web app in one step.
3. Dashboard log parsing relies on regex over schedule logs. Consider emitting a
   JSON summary alongside the logs to avoid parsing drift.

## Recently Resolved

1. Updated PythonAnywhere Node.js install/build scripts to use Node 20.19.0,
   matching the Vite 7 requirement.
2. Documented frontend build steps and Logo.dev key setup in the README and
   RUNBOOK.
3. Ignored `frontend/dist` and `frontend/.env` in git to prevent build output
   and secrets from polluting the repo.
