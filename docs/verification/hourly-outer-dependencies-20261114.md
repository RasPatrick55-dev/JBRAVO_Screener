# Hourly outer dependency recovery, 2026-11-14

Twelve source modules were recovered verbatim via the supported PythonAnywhere MCP read endpoint at 22:05:59 UTC. Each hash and byte count matches the already-frozen candidate007 hourly outer source manifest. The accompanying JSON records exact source paths and hashes. Git attributes retain the native bytes, including line endings. No changes to runtime logic are included.

These modules close the static scripts-import dependencies of the existing hourly lot/origin adapter on canonical base 1d13cf62f60d67188499c616a1e80746751aa45e. They are data-reconciliation and accounting modules, not permission to activate any task, broker operation or database migration.

Offline regression coverage uses synthetic fixtures only. Six historical unit suites were retained; two tests dependent on private actual-trade fixture files were excluded rather than publishing private fixtures. New contracts verify all twelve exact byte hashes, scripts-import closure and import without socket creation. Windows: 100 passed. Linux Python 3.10.22: 100 passed.

Limits: no PostgreSQL integration, current schema, live account binding, production process, loader or hourly source-pin rebinding has been established by these tests. Existing origin and lot schemas and receipts remain mandatory runtime prerequisites. Historical proof pins must remain historical; current canonical source checks require a separate reviewed binding. This is source recovery only, not deployment readiness.

