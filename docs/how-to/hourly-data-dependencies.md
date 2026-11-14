# Retained hourly data dependencies

This change brings the five existing PythonAnywhere data-only modules into the canonical repository. It does not activate a job, create a schema, read secrets, or change orders. Native source provenance and hashes are in `docs/verification/hourly-dependencies-20261114.json`.

## Scope and verification

The five modules implement bounded paper-account GET observations, checkpoint-backed fill linkage and fill-proven local signal closure. They do not create, replace, cancel or sell broker orders. Live-account endpoints, redirect responses and unapproved configuration/source hashes fail closed. Unknown commit outcomes require readback; no automatic replay is added.

All five runtime modules are byte-for-byte native readbacks acquired through the supported PythonAnywhere MCP on November 14, 2026. They retain native CRLF bytes through explicit `-text` Git attributes so existing authorization source pins remain valid when files are materialized from Git blobs. The existing checkpoint, activity ingestor and DB support modules match canonical Git blobs exactly.

The regression tests are recovered synthetic fixtures from the prior local review, with `alpaca_optional` markers because they use no provider credentials. Additional tests cover lower-precision timestamp repeat preservation, refusal of material changes, immutable native-source pins and empty-write avoidance. No actual broker or database requests were made by these tests.

## Deployment prerequisites

This is compatibility with the existing PythonAnywhere deployment, not a complete fresh-host installer. The five modules require existing strategy execution, checkpoint, fill-linkage and projection receipt tables and the approved database-policy identity. No DDL or database migrations are included. Verify those preconditions read-only before activation.

The hourly outer adapter owns its generation lease and invokes this data phase in-process. Do not nest a second lease wrapper or assume its dynamically loaded helper shares module state with the canonical helper. These retained modules do not spawn children. New release guards must bind this exact dependency closure, task entrypoint and source/config hashes. Preserve existing authorization configuration semantics and do not reuse consumed operations.

This source preparation is not evidence of a successful hourly run, a complete production cutover or whole-bot readiness. No database connection or PythonAnywhere runtime qualification occurred in this task.

## Offline qualification result

65 tests and 11 subtests passed on Windows and Linux Python 3.10.22. CLI fixtures explicitly mock dotenv, broker and database ports. Ruff, Python 3.10 syntax and Git whitespace checks passed. PostgreSQL integration and actual PythonAnywhere runtime qualification were not performed.
