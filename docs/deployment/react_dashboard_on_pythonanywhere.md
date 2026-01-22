# React dashboard on PythonAnywhere

## Architecture overview (React + Flask + Dash)

* **React** is built into `frontend/dist` and served at the site root (`/`).
* **Flask** (WSGI entry point) serves the React `index.html` and the compiled assets.
* **Dash** runs the legacy dashboard UI mounted under `/v2`.
* **Backend API** routes live on the same Flask server as Dash and are exposed at `/api`.

The WSGI entry point is responsible for dispatching these pieces and keeping URL prefixes stable.

## Asset serving rules (/assets)

* React static assets **must** be published under `/assets`.
* The WSGI Flask app is configured with `static_url_path="/assets"` and `static_folder=frontend/dist/assets`.
* Any `/static` or `/ui-assets` references are obsolete and should not be reintroduced.

React builds should use base path `/` so that the compiled `index.html` points to `/assets/*`.

## WSGI responsibilities

The PythonAnywhere WSGI file (`raspatrick_pythonanywhere_com_wsgi.py`) must:

1. Load environment variables.
2. Create the Flask **frontend** app for React assets.
3. Provide a React catch-all route that always returns `frontend/dist/index.html`.
4. Mount the Dash app at `/v2`.
5. Mount the backend API at `/api`.

## PythonAnywhere caveats

* **Static files**: do not rely on PythonAnywhere static mappings for React. The WSGI Flask app is the source of truth for assets.
* **Prefix handling**: DispatcherMiddleware strips mount prefixes, so the WSGI file must re-apply `/v2` and `/api` before handing off to Dash.
* **Build placement**: the React build must live at `frontend/dist`.

## Common failure modes

* **HTML served instead of JS**: assets built with a non-root base path (e.g., `/ui-assets`) cause the browser to fetch HTML at the JS URL.
* **Blank white screen**: JS bundle fails to load due to asset path mismatch or missing build output.
* **Wildcard curl confusion**: curling `/assets/...` returns HTML instead of JS, indicating a missing asset directory.

## Debug checklist

1. Verify the build exists:
   ```bash
   ls frontend/dist/index.html
   ls frontend/dist/assets
   ```
2. Confirm `index.html` references `/assets/`:
   ```bash
   grep -q '/assets/' frontend/dist/index.html
   ```
3. Confirm WSGI mounts:
   * `/` returns React HTML
   * `/assets/*` returns JS/CSS
   * `/v2/` returns Dash
   * `/api/health` returns JSON

## Build guard (fail fast)

Use this check after building React to ensure asset paths are correct:

```bash
grep -q '/assets/' frontend/dist/index.html || exit 1
```
