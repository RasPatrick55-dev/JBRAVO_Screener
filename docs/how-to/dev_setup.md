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
