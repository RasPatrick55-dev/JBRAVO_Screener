# PythonAnywhere generation-lease compatibility

This change makes the generation-lease helper and subprocess propagation available
in GitHub without imposing a PythonAnywhere path on generic CLI entry points.
It does not install a release, start a job, edit a task, or qualify market data.

## Explicit host entry points

- `scripts.pythonanywhere_primary_pipeline` owns exactly one strict generation
  lease, then calls the unchanged generic `primary_paper_pipeline.main`.
- `scripts.pythonanywhere_account_sync` owns one lease, then calls the unchanged
  generic `account_sync.main` body.
- `scripts/ops/run_pythonanywhere_primary_pipeline.sh` preserves the generic shell
  launcher's venv, environment, arguments and primary single-flight lock, but
  selects the strict PythonAnywhere primary wrapper.

The host wrappers always require the existing private POSIX generation lock. A
missing, replaced, insecure or blocked lock refuses before underlying workflow
import. There is no optional flag or environment setting that disables fencing.
Generic primary/hourly CLI behavior remains portable and does not acquire that
host-specific lease. Existing guarded external adapters may call generic code
within their own lease; they must not nest the strict wrappers inside another
`run_guarded` call. Accidental nesting fails closed before workflow execution.

The helper and existing subprocess propagation hooks retain the reviewed deployed
behavior. Known direct children can inherit the descriptor and outlive the parent
without releasing it. This is NOT a guarantee for arbitrary exec grandchildren:
process-local `_active` is not reconstructed automatically in an exec'ed child,
and any later subprocess that closes the inherited descriptor loses that hold.
A release must separately qualify its actual process tree and adapter boundaries.

## Required release changes

The new PythonAnywhere release must bind the primary task to the strict shell
launcher above, and verify the exact wrapper/helper/source pins. Simply replacing
native source while retaining the old generic primary task command is unsafe.
Hourly's current external adapter already owns its lease; verify that route and
retain its single lease instead of wrapping it twice. The new account-sync wrapper
is for any separately scheduled direct account-sync entry, not automatic nesting.

Rebuild producer, entry, monitor and hourly source manifests against the exact
merged GitHub commit. Freeze fresh host preimages and latest hourly successors.
Use a new reviewed plan, journal and rollback evidence: revision014 is consumed.
Before cutover verify source-bound ownership compatibility, task/worker quiescence
and all adapter boundaries. Enabled flags alone are not run or lease evidence.
Source installation, task binding, web deployment and scheduled run success must
be verified separately, without replaying pipeline or executor as a deploy test.

## Offline verification

Use `python -B -m unittest tests.test_candidate007_worker_fence
 tests.test_primary_paper_pipeline -v` on Linux. Tests use temporary files and inert
children. They cover mutual exclusion, replaced/link identities, direct-child
retention after parent death, maintenance recovery, wrapper fail-closed behavior,
no double fencing, generic portability and exact single-flight shell derivation.

The broader 54-test account-sync suite retains 27 failures and 3 errors on untouched
main and the compatibility version, with identical failing test identifiers.
These baseline failures remain open; focused tests are not a full-suite pass.
