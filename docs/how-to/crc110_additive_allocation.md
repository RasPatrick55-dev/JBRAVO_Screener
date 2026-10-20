# CRC110 additive allocation: review and rollback

This increment is implementation and isolated-test preparation only. It does not authorize a GitHub push/PR/merge, deployment, schema installation, production data read, grant, activation, provider request, order, or network/job/model change. A separate exact production approval is required.

## Scope and economics

The allocation is restricted to CRC trade110, full-row fingerprint `076b62b8ece8ec28a86e3254efad1a86c9c10ad50bdf47bc23bbc9ec9a7800b6` and updated_at `2026-04-17T19:05:44.607608+00:00`. Two entry fills18+12 at65 and three exits1 at63,14 at62.48,15 at63.26 conserve30 shares across both original exit orders. Cost1950, proceeds1886.62 and gross result-63.38 are exact Decimal observations. Fees and net PnL remain unknown. No scalar exit price/order, VWAP, original PnL, ownership or timestamp is synthesized.

The active validated overlay resolves operational missing-exit repair only. It does not make financial inputs complete or turn gross into net. With the observed population otherwise unchanged, AAPL63 remains excluded and performance still has40 unresolved rows, including CRC110. Windowed ML checks only selected allocated records, so an unaffected complete window can remain usable.

## Approval and preflight gates

1. Review the exact patch, source/schema pins, isolated test receipts and any native PostgreSQL gaps. A socket-free SQL protocol double is not native SQL, trigger, privilege or concurrency proof.
2. Verify current deployment/database identity, policy revision and AAPL63 ACTIVE case. The observed current policy revision is1; a successful first CRC activation would advance the shared policy to2. Recheck it at execution rather than silently updating a stale request.
3. Bind the exact current original110 row and five complete activity rows, including raw payloads and all13 stored activity columns. Retained projections and synthetic raw fixtures are insufficient production authority. Review the separate bounded read-only capture packet if necessary; do not discover credentials or run application startup.
4. Obtain one explicit Owner decision covering the exact source rollout, additive schema/ownership plan and CRC activation request/manifest hashes. Distinguish these effects in the same packet; any permission changes must be listed and explicitly included. Do not add permissions automatically. Runtime/ML readers need only SELECT on the policy marker and additive read data; the allocation operator does not need historical-trade/activity UPDATE rights to acquire its ACCESS SHARE locks.
5. Leave `actor`, `approved_by`, `approved_at` and `approval_ref` null in preparation templates. Once full stored evidence is verified, its exact manifest hash may be pinned; that hash is evidence binding, not activation approval. The final approval-bound request digest can be frozen only after distinct concrete activation authority is supplied. The null approval fields intentionally make the preparation request invalid for execution.
6. The operator calls only `change_allocation` with an idle supplied connection, fixed identity and reviewed request/manifest. It uses the existing exclusive policy advisory/CAS contract, bounded waits and exact current evidence. No import, migration or reader activates a case.

All consumers validate the schema marker, complete sidecar visibility, current evidence and exact enabled trigger contract. Partial/missing installed schema, denied reads, RLS/inheritance, changed original/fill/audit data or stale revisions fail closed. The never-installed legacy path is explicitly distinct from a disappeared installed allocation schema.

The exact DDL uses the existing authenticated administrator with `current_user=session_user` and an existing superuser check. It creates the additive objects, assigns only the four new tables and five functions to the existing canonical policy owner `jbravo_v7_owner`, then publishes marker1 and commits within the same transaction. It neither switches to that owner nor grants original-table REFERENCES/TRIGGER privileges. ROOT's observed false table-level REFERENCES flags do not establish the unqueried column-level privileges; this plan avoids relying on either. Original owners and ACLs remain unchanged. Existing service-role read access is a separate attestation limit, not inferred from synthetic grants.

## Production acceptance after separately approved activation

Require a complete acknowledged API receipt, allocation ACTIVE/revision1 and the expected shared policy increment. Independently read the exact case, append-only audit, five native claims and three order claims. Verify all original trades/events/activities counts and full logical-row digests and the named reconcile watermark before/after. Equal snapshots do not prove no intervening mutation or physical heap/MVCC identity.

Verify actual reconciliation, performance, dashboard and ML consumers at the new revision. The AAPL63 exclusion must remain intact. CRC operational repair is suppressed only by validated active evidence, while unknown fees/net remain held. Do not claim full bot-process, service-role or live provider attestation from selected-function tests.

Normal identical activity ingestor replay may allocate sequence values even when ON CONFLICT preserves every logical row. Report sequence effects separately; its returned normalized-row count is not an inserted-row count. Conflicting/new fill identities, field/order changes and destructive buy-upsert replays remain refused.

## Revocation and failure handling

Revocation requires a separate exact approval/request: same allocation ID and manifest, action REVOKE, fresh expected shared policy and allocation revisions. For the initial activation at policy2/allocation1, a successful revoke advances to policy3/allocation2. Never guess those revisions if another authorized change has occurred.

Revocation removes operational resolution, so CRC returns to the missing-exit/unresolved repair set. It also removes the active allocation's trade-wide write ban; a later independently authorized repair of the rightful owner110 may change that original row through ordinary guarded paths. Revocation itself never writes an original. The captured historical manifest, native FILL evidence, claims and transition audits remain immutable and reserved after revoke. A different trade must never obtain either CRC exit order through a candidate or write path. A revoked reader validates historical evidence and current ownership without requiring the current owner110 row to remain the historical snapshot; reactivation still requires its exact frozen full-row CAS and refuses after row drift.

A normal identical request may return its original acknowledged audit receipt only while its current case state/revision still matches. An old APPLY receipt superseded by revoke/reactivation is refused; it cannot imply current activation. No automatic retry, rollback, reapply, original-row rewrite, claim deletion, schema drop or permission fallback is permitted after refusal, timeout or uncertain commit. Retain all evidence and establish the actual committed state through separately authorized reads.

Source rollback is a separately reviewed operation. Do not deploy an older consumer that cannot understand active ownership reservations. Prefer approved revocation and complete consumer/cache verification before a separately approved source rollback. Preserve the additive ledger, marker and audit history; dropping them is not rollback.
