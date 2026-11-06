# Alpaca compatibility release: November 6, 2026

This release prepares paper-only Alpaca SDK compatibility against current main.
It does not itself establish that a PythonAnywhere deployment occurred.

Account compatibility prefers finite `equity` and accepts `portfolio_value` only
as a legacy fallback. Snapshot ingestion retains the existing PostgreSQL columns
and raw JSONB payload; missing retired day-trading buying power remains NULL.
Snapshot requests reject non-paper URLs and redirects. Existing financial-view
source/performance boundaries remain in place.

Asset-list reads use `alpaca-py` with `paper=True`, retaining tradability/exchange
filters and the optional universe cache. The cache does not replace canonical
PostgreSQL candidate views. The retired SDK dependency is removed, allowing a
direct `PyYAML>=6.0.3,<7` wheel installation. Both dependency manifests specify
`alpaca-py>=0.45.0,<0.46` and `pydantic>=2.7,<3`.

Execution summaries skip broker counts whenever PostgreSQL or fallback file
metrics exist, preserving explicit zero and incomplete counts. Paper broker
counts remain a labeled fallback when neither metrics source exists. Current
validated-performance and ledger rules are preserved.

## Validation

48 focused tests passed against this current-main port, covering SDK account
models, PostgreSQL snapshot writes, paper-only requests, execution-summary
precedence, asset filters, enum serialization, cache reuse and ranker YAML paths.
Tests use fake credentials, disabled DB access and mocked external boundaries.
The earlier clean Windows Python 3.13 requirements install and `pip check` passed
with Alpaca SDK 0.45.0, PyYAML 6.0.3 and Pydantic 2.13.5.

## PythonAnywhere deployment constraints

The native checkout has newer, separately maintained runtime overlays than the
original local workspace. Deploy only the reviewed compatibility edits onto
captured native preimages; preserve unrelated dashboard statements, source
guards, strategy rules, models, private artifacts and historical receipts.

The current daily producer and paper executor launchers hash-pin changed source
files. Their replacement launchers and command-only task updates must be bound
to the same source manifest. Existing task times, paper flags, position/risk
limits and all other tasks must remain unchanged. Do not run an executor or
force candidate publication to test this release.

Before any host writes, prepare and approve a new byte-bound scoped operation
through the shared PythonAnywhere MCP. The old fixed-release operations do not
authorize this release. Verify native interpreter/dependency compatibility,
source bytes, launcher `--verify-only` results and a single web reload, followed
by bounded dashboard GETs and current pipeline receipts. A source/readiness hold
must remain visible; a successful HTTP response is not pipeline acceptance.

The pre-deployment public read on November 6 returned HTTP 200 for the app and
four inspected API endpoints. Execution summary reported PostgreSQL and a
November 6 run; candidates returned `strategy_primary:unavailable` with
`entry_ready=false`. The latest persisted postflight described completed primary
stages pending unified selection. These are pre-deployment observations, not
proof that the compatibility release is installed or the whole pipeline ready.
