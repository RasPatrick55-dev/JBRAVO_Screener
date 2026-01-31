# JBRAVO Brand Assets

This doc explains where the JBRAVO Trading Bot brand files live, how they are used, and how to update or request new ones.

## Source Of Truth

All official brand assets live in the repo at:

- `assets/brands/`
  - `jbravo-logo-flat.svg` (primary wordmark)
  - `jbravo-logo-glow.svg` (alternate/glow wordmark)
  - `jbravo-icon-flat.svg` (favicon/app icon)
  - `motion/` (optional motion SVGs)

Important: the SVGs must be real SVG markup (file should start with `<svg ...>`). Avoid Figma preview/iframe HTML wrappers, which render as broken images.

## Frontend Usage (Vite)

The React/Vite app loads brand files from the public folder at:

- `frontend/public/assets/brands/...`

Before building, copy updated assets from `assets/brands/` into the public folder so `vite build` bundles them into `frontend/dist/assets/brands/`.

PowerShell sync:

```powershell
New-Item -ItemType Directory -Path frontend\public\assets\brands, frontend\public\assets\brands\motion -Force | Out-Null
Copy-Item -Path assets\brands\jbravo-logo-flat.svg -Destination frontend\public\assets\brands\jbravo-logo-flat.svg -Force
Copy-Item -Path assets\brands\jbravo-logo-glow.svg -Destination frontend\public\assets\brands\jbravo-logo-glow.svg -Force
Copy-Item -Path assets\brands\jbravo-icon-flat.svg -Destination frontend\public\assets\brands\jbravo-icon-flat.svg -Force
Copy-Item -Path assets\brands\motion\* -Destination frontend\public\assets\brands\motion\ -Force
```

Then build:

```bash
cd frontend
npm run build
```

## Dash / Server Usage

The Dash/Flask server serves `/assets/*` and falls back to the repo `assets/` folder if the file is not in the React build. That means `/assets/brands/...` works both in the Dash UI and the React UI when the source assets are present.

## Current Usage Points

- Navbar logo: `/assets/brands/jbravo-logo-flat.svg`
- Favicon: `/assets/brands/jbravo-icon-flat.svg`

## Requesting New Assets

When requesting updates, specify:

- Format: **SVG** (not PNG, not embedded HTML)
- Naming: keep the existing file names if replacing
- Deliverables: logo flat, logo glow, icon, and motion SVGs as needed

After receiving new assets, replace files in `assets/brands/`, sync to `frontend/public/assets/brands/`, rebuild, and verify in preview.
