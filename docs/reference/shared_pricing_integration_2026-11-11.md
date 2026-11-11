# Shared pricing integration checkpoint

Status: NON-DEPLOYABLE integration groundwork. No launchers, scheduler, web
runtime or broker operation is activated by this checkpoint.

Base: GitHub main d37fdfcdb7cff8d0a91697d612f52f4af484cef1, verified through
GitHub on 2026-11-11. The frozen shared-pricing candidate supplied six portable
helpers. Their source bytes were preserved; tests now use repository-local
native sources rather than private desktop preimage paths.

## Reconciliation

- Added the retained pure ticket validator/model, legacy price reconstruction,
  runtime pipeline queue and expected-account validator.
- Preserved canonical source inventory/lifecycle validation, opening-minute
  constraint and monitor readiness. Pipeline handoffs additionally require
  the reconstructed ticket, account identity and ticket validity to agree.
  Verified ticket-bearing records cannot downgrade to the legacy model.
- Applied only the dashboard batch-mismatch readiness fix, preserving the
  canonical root, API signature and completion evidence behavior.
- The shared installer remains an explicit process-local operation. Nothing
  invokes it automatically. Canonical publishing/execution routes remain
  unchanged. The old publisher was not replaced with its host overlay.
- The host-only compatibility-history mechanism and assembly bootstrap were
  deliberately not imported. The portable lifecycle test checks canonical
  fail-closed source validation, not historical compatibility admission.

## Validation

Windows Python focused suite: 83 passed. Combined focused/native regression:
212 passed, 1 failed. The sole failure,
`test_setup_selection.py::test_scanner_dashboard_projects_global_winner_instead_of_primary_family`,
is independently reproduced on untouched base d37fdfc: its snapshot test double
does not accept the existing base_dir keyword. This is not a full suite pass.
Ruff E9 and docs consistency checks passed. External data, database, clock,
account and monitor ports are synthetic where exercised. The guard regression
checks monitor rejection, ticket mutation and model downgrade. No order calls.
No Linux or PythonAnywhere qualification of this integrated tree was performed.

## Remaining synchronization work

This commit is not source-equivalent to the active host. Functional entry,
legacy publisher/reader and ownership overlay reconciliation remain open.
Do not replace the guarded canonical runtime with the retained host file:
its source/lifecycle/monitor behavior differs. Preserve existing positions and
all active ownership checks while constructing the complete source closure.
Then independently review that closure, qualify exact target Python, create
a newly scoped supported deployment operation, verify active web binding and
reload support, and compare repository and host hashes. Do not reuse a consumed
operation or bypass restricted routing. No deployability or operating approval
is implied by passing offline tests or by this local commit.


## Second reconciliation checkpoint

The exact-ticket builder, optional pipeline handoff mode, quote-free execution
mode and market-data-denying broker mode are now integrated. Defaults preserve
the legacy route. Legacy price derivation rejects pipeline-model handoffs.
Historical manual positions retain protection/cancellation/exit behavior;
new manual entries remain rejected. The canonical 04:00-20:00 manage gate is
unchanged even for pipeline engines. Existing lifecycle sync/tick pins match.

Retained monitor source and its read-only lifecycle-history interpretation are
now included. No history writer or database migration is added. Canonical
entry ownership, bounded preflight, monitor ownership and pending-stop adapter
are composed with a fixed named canonical manage-source pin. The old retained
adapter variant remains the default; no arbitrary caller hashes are accepted.
The complete monitor composition is checked before endpoint installation.
The strict pipeline configuration is isolated to the assembled entry facade.
Canonical source_guard and monitor readiness remain, including their failures.

The release builder uses exact commit blobs, not checkout newline conventions.
It exports an immutable complete Python/config/dependency closure and requires
an independently supplied manifest SHA256 to verify it. Assembly requires a
fresh process and blocks network, database/runtime dispatch, subprocesses and
filesystem mutation. Its source inventory can include legitimate zero-byte
package initializers; size and SHA256 still must match exactly.

A real canonical ownership composition test now runs in a fresh process with
network disabled, in addition to synthetic unit tests. Exact-commit Windows and
Linux qualification results must be recorded separately after the checkpoint
is committed. No old host bootstrap, consumed operation or live route is used.

Still not a deployment: active web routing, host preimage agreement, current
position/lifecycle source bindings and supported new operation admission need
host-specific verification. The retained monitor's historical hardcoded pins
are superseded only inside the explicit manifest-bound assembled process.
No scheduler or monitor main/cycle, publisher, database query, or order is run
by assembly. A separate supported deployment operation must bind executable
entry points and rollback/preimage rules before activation.
