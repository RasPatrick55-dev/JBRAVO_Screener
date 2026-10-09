# Developer Setup

## Enable Docs Guardrails Locally

Install and enable pre-commit hooks (one-time per machine):

```bash
pip install pre-commit
pre-commit install
```

Run hooks across the repository:

```bash
pre-commit run --all-files
```

## Expected Docs Hook Behavior

The docs hook runs:

```bash
python -m scripts.docs_consistency_check
```

This regenerates `docs/reference/cli_reference.md` and fails if documentation drifts from paper-only + DB-first policy.

## Main Branch Protection (GitHub UI)

For `main`, enable required status checks and include `Docs consistency` as a required check.

## PythonAnywhere API Credentials In VS Code

Use [pythonanywhere_credentials.md](pythonanywhere_credentials.md) as the canonical reference.

- Real credentials are never committed.
- Shared VS Code config stays secret-free.
- Local-only repo files for PythonAnywhere credentials are `.env.pythonanywhere.local` or `.env.local`; both are gitignored.
- The helper task for discovery is:

```bash
python -m scripts.pythonanywhere_env_probe
```

- The handshake task before any remote command execution is:

```bash
python -m scripts.pythonanywhere_env_probe --handshake
```
