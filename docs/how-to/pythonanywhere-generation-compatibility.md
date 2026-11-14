# PythonAnywhere generation-lease compatibility

This change makes the existing deployed generation-lease behavior canonical in
GitHub. It does not install a release, start a job, change a task, or qualify data.

## What is preserved

The source helper and four call-site changes are taken from the reviewed
candidate007 worker overlay. Primary and hourly entry points require the existing
private POSIX generation lock. Their known child processes inherit its descriptor.
The run_pipeline subprocess and executor telemetry boundaries preserve the lease.
Existing paper checks, source verification, per-worker locks and arguments remain.
The helper has no network or credential access. A missing, replaced, insecure, or
blocked lock refuses work; it must never be silently auto-created or bypassed.

The fixed path remains the already-deployed PythonAnywhere generation-fence path.
This is host compatibility, not a general cross-platform scheduler feature. The
operator must provision and verify its private directory and lock before using
these entry points on a new host. Source installation alone does not create them.

## Remaining release work

- Rebuild external producer, entry, monitor and hourly adapters against the exact
  merged GitHub commit; old checksum manifests are not reusable.
- Bind fresh host source/task preimages and current hourly successor files.
- Use a new reviewed operation identity and journal. Revision014 is consumed.
- Verify active ownership compatibility, task/worker quiescence and rollback before
  cutover. Do not substitute an enabled flag for actual process/lease evidence.
- Verify installed files, task bindings and web deployment separately, without
  submitting orders or replaying the pipeline as a deployment test.

The existing revision014 source transaction, provider task adapter and recovery
code are reusable implementation references, not reusable approval or receipts.
A new release must bind its own exact source mapping and commands. No generic API
transport, hidden metadata disclosure, source-pin bypass or disabled safety guard
is introduced here.

## Tests

Run `python -B -m unittest tests.test_candidate007_worker_fence
 tests.test_primary_paper_pipeline -v` on Linux in a configured test environment.
The fence tests use temporary files and inert child processes. They test mutual
exclusion, replaced/link identities, parent-death descriptor retention, persistent
maintenance blocking/recovery, and primary/hourly refusal before workflow entry.
The old paper-context tests isolate the lease with a stated test double; they do
not independently establish POSIX lease safety. No test places broker orders.

### November 14 offline result

All 18 focused Linux lease/wrapper and paper-context tests passed. Python 3.10
syntax checks and documentation consistency passed. The broader 54-test
account-sync suite has 27 failures and 3 errors on both untouched main and this
branch, with identical failing test/subtest identifiers. These are retained
baseline failures, not a full-suite pass. Source compatibility is not proof of
native runtime deployment or usable market quotes.
