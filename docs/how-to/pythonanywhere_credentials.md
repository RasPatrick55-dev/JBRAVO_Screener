# PythonAnywhere Credentials

This is the canonical PythonAnywhere credential contract for local VS Code + Codex work in this repo.

## Rules

- Never commit real PythonAnywhere credentials or tokens.
- Never paste `~/.config/jbravo/.env`, `%USERPROFILE%\\.config\\jbravo\\.env`, `.env.local`, or `.env.pythonanywhere.local` into ChatGPT Project Sources or any context pack.
- PythonAnywhere API auth uses `Authorization: Token <token>`.
- Supported `PA_HOST` values are only:
  - `www.pythonanywhere.com`
  - `eu.pythonanywhere.com`

## Canonical Lookup Order

PythonAnywhere API helpers resolve `PA_USERNAME`, `PA_TOKEN`, and `PA_HOST` in this exact order:

1. Process environment variables.
2. User-level env file:
   - Linux/macOS: `~/.config/jbravo/.env`
   - Windows: `%USERPROFILE%\\.config\\jbravo\\.env`
3. Repo-local ignored env files, checked in this order:
   - `.env.pythonanywhere.local`
   - `.env.local`

If any required value is still missing, the helper must fail with an actionable error that names the missing keys and these exact supported locations.

## Recommended Local Setup

Use placeholders from [`.env.pythonanywhere.example`](../../.env.pythonanywhere.example). Do not commit the real file you create locally.

PowerShell:

```powershell
$env:PA_USERNAME = "your_pythonanywhere_username"
$env:PA_TOKEN = "your_pythonanywhere_api_token"
$env:PA_HOST = "www.pythonanywhere.com"
python -m scripts.pythonanywhere_env_probe
```

PowerShell user-level file example:

```powershell
$envFile = Join-Path $HOME ".config/jbravo/.env"
@"
PA_USERNAME=your_pythonanywhere_username
PA_TOKEN=your_pythonanywhere_api_token
PA_HOST=www.pythonanywhere.com
"@ | Set-Content -Path $envFile
```

bash/zsh:

```bash
export PA_USERNAME=your_pythonanywhere_username
export PA_TOKEN=your_pythonanywhere_api_token
export PA_HOST=www.pythonanywhere.com
python -m scripts.pythonanywhere_env_probe
```

bash/zsh user-level file example:

```bash
mkdir -p ~/.config/jbravo
cat > ~/.config/jbravo/.env <<'EOF'
PA_USERNAME=your_pythonanywhere_username
PA_TOKEN=your_pythonanywhere_api_token
PA_HOST=www.pythonanywhere.com
EOF
```

VS Code note:

- Shared workspace config stays secret-free.
- `.vscode/tasks.json` calls the helper and never hardcodes credentials.
- `.vscode/settings.json` and `.vscode/launch.json` point Python env loading at `.env.local`, which is ignored and local-only.

## Probe Command

```bash
python -m scripts.pythonanywhere_env_probe
```

Expected safe output:

- where lookup was attempted
- `PA_USERNAME`
- `PA_HOST`
- whether `PA_TOKEN` is present
- a masked token hint only

Failure behavior:

- exits nonzero
- names missing variables
- names the exact supported lookup locations
- never prints the raw token

## Required API Handshake Before Remote Commands

Do not claim remote execution unless the `pwd` handshake actually succeeded.

Required sequence:

1. `GET /api/v0/user/{username}/consoles/`
2. `POST /api/v0/user/{username}/consoles/{id}/send_input/` with `input=pwd\n`
3. `GET /api/v0/user/{username}/consoles/{id}/get_latest_output/`

Console rule:

- API-created consoles are not actually started until opened in a browser.
- If `send_input` or `get_latest_output` returns HTTP `412`, stop and report the blocker exactly. Do not continue to deploy or run commands.

Helper command:

```bash
python -m scripts.pythonanywhere_env_probe --handshake
```

Good output:

- consoles were listed first
- an existing Bash console id was selected
- `pwd` was sent
- `get_latest_output` succeeded without HTTP `412`

## Codex / VS Code Network Note

If Codex in VS Code needs PythonAnywhere API access, use a session mode that permits outbound network access and the required tool approvals. If network access is blocked, stop and report that blocker instead of claiming remote execution.
