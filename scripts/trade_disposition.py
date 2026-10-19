"""Exact-trade eligibility and explicit, reversible disposition transactions.

UNRESOLVED scoped operational decisions do not establish execution provenance.
No environment discovery, clients, schema creation or activation at import time.
Callers supply a PostgreSQL connection and deployment identity. All policy
errors are fatal to eligibility; absence of policy is not an empty policy.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import re
import math
from uuid import UUID


class TradePolicyError(RuntimeError):
    """Eligibility is unavailable or an approval/CAS precondition failed."""


def fail(reason):
    raise TradePolicyError("TRADE_POLICY_" + reason)


def document(value):
    try:
        def unique_pairs(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    fail("DOCUMENT_DUPLICATE_KEY")
                result[key] = item
            return result
        result = json.loads(value, parse_float=Decimal, object_pairs_hook=unique_pairs) if isinstance(value, str) else value
        if not isinstance(result, dict):
            fail("DOCUMENT_INVALID")
        def checked(item):
            if isinstance(item, dict):
                for key, nested in item.items():
                    if not isinstance(key, str):
                        fail("DOCUMENT_INVALID")
                return {key: checked(nested) for key, nested in item.items()}
            elif isinstance(item, list):
                return [checked(nested) for nested in item]
            elif isinstance(item, (float, Decimal)):
                number = Decimal(str(item))
                if (not number.is_finite() or len(number.as_tuple().digits) > 1000
                        or abs(number.as_tuple().exponent) > 1000):
                    fail("DOCUMENT_INVALID")
                return number
            elif item is None or isinstance(item, (str, int, bool)):
                return item
            fail("DOCUMENT_INVALID")
        return checked(result)
    except TradePolicyError:
        raise
    except (ValueError, TypeError):
        fail("DOCUMENT_INVALID")


def canonical(value):
    # Preserve PostgreSQL numeric precision; binary floating-point coercion can
    # make distinct complete row snapshots share a fingerprint.
    def encode(item):
        if isinstance(item, dict):
            return "{" + ",".join(json.dumps(key) + ":" + encode(item[key]) for key in sorted(item)) + "}"
        if isinstance(item, list):
            return "[" + ",".join(encode(nested) for nested in item) + "]"
        if isinstance(item, Decimal):
            number = format(item, "f")
            if "." in number:
                number = number.rstrip("0").rstrip(".")
            return "0" if item == 0 else number
        return json.dumps(item, allow_nan=False)
    return encode(document(value))


def fingerprint(snapshot):
    return hashlib.sha256(canonical(snapshot).encode("utf-8")).hexdigest()


def text_field(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > 2048:
        fail("IDENTITY_INVALID")
    return value


def identity(deployment_id, database_id):
    text_field(deployment_id)
    try:
        if str(UUID(database_id)) != database_id:
            fail("IDENTITY_INVALID")
    except (ValueError, TypeError, AttributeError):
        fail("IDENTITY_INVALID")
    return deployment_id, database_id


def positive_id(value):
    if type(value) is not int or value <= 0:
        fail("TRADE_ID_INVALID")
    return value


def revision(value):
    if type(value) is not int or value < 0:
        fail("REVISION_INVALID")
    return value


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        fail("DIGEST_INVALID")
    return value


def aware_time(value):
    try:
        stamp = datetime.fromisoformat(text_field(value).replace("Z", "+00:00"))
        if stamp.utcoffset() is None:
            fail("TIME_INVALID")
        return stamp
    except (ValueError, TypeError):
        fail("TIME_INVALID")


def validate_request(request, manifest, expected_identity):
    request, manifest = document(request), document(manifest)
    required = {"action", "case_id", "trade_id", "row_fingerprint", "row_version",
                "deployment_id", "database_id", "expected_policy_revision", "expected_case_revision",
                "manifest_sha256", "actor", "approved_by", "approved_at", "approval_ref"}
    if set(request) != required or request["action"] not in {"APPLY", "REVOKE"}:
        fail("APPROVAL_INVALID")
    if identity(request["deployment_id"], request["database_id"]) != expected_identity:
        fail("IDENTITY_MISMATCH")
    try:
        if str(UUID(request["case_id"])) != request["case_id"]:
            fail("CASE_INVALID")
    except (ValueError, TypeError, AttributeError):
        fail("CASE_INVALID")
    positive_id(request["trade_id"])
    revision(request["expected_policy_revision"])
    revision(request["expected_case_revision"])
    digest(request["row_fingerprint"])
    for key in ("actor", "approved_by", "approval_ref"):
        text_field(request[key])
    approved_at = aware_time(request["approved_at"])
    if approved_at > datetime.now(timezone.utc):
        fail("APPROVAL_TIME_INVALID")
    aware_time(request["row_version"])
    if digest(request["manifest_sha256"]) != fingerprint(manifest):
        fail("EVIDENCE_MISMATCH")
    for key in ("case_id", "trade_id", "row_fingerprint", "row_version", "deployment_id", "database_id"):
        if manifest.get(key) != request[key]:
            fail("EVIDENCE_MISMATCH")
    # Corroboration and exact provenance are separate claims. Neither activates
    # policy: a separately reviewed, case-bound approval is always required.
    if manifest.get("classification") != "UNRESOLVED" or manifest.get("provenance_proved") is not False:
        fail("EVIDENCE_INVALID")
    scopes = manifest.get('operational_scopes')
    if (not isinstance(scopes, list) or not scopes or len(scopes) > 2
        or any(not isinstance(scope, str) or scope not in SCOPES for scope in scopes)
        or len(set(scopes)) != len(scopes)):
        fail('EVIDENCE_SCOPE_INVALID')
    text_field(manifest.get("rationale"))
    evidence = manifest.get("evidence")
    if not isinstance(evidence, list) or not evidence or len(evidence) > 100:
        fail("EVIDENCE_INVALID")
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"reference", "sha256"}:
            fail("EVIDENCE_INVALID")
        text_field(item["reference"])
        digest(item["sha256"])
    return request, manifest


SCOPES = frozenset({'ACTIVE_RECONCILIATION', 'VALIDATED_PERFORMANCE'})
_RUNTIME_IDENTITY = None


def checked_scope(value):
    if value not in SCOPES:
        fail('SCOPE_INVALID')
    return value


def bind_runtime_identity(deployment_id, database_id, *, environment, evidence_ref, verified):
    """Explicit verified paper binding only; no config or connection discovery."""
    global _RUNTIME_IDENTITY
    if environment != 'paper' or verified is not True:
        fail('RUNTIME_IDENTITY_UNVERIFIED')
    proposed = (*identity(deployment_id, database_id), text_field(evidence_ref), environment)
    if _RUNTIME_IDENTITY is not None and _RUNTIME_IDENTITY != proposed:
        fail('RUNTIME_IDENTITY_ALREADY_BOUND')
    _RUNTIME_IDENTITY = proposed
    return proposed[:2]


def runtime_identity():
    if _RUNTIME_IDENTITY is None:
        fail('RUNTIME_IDENTITY_UNBOUND')
    return _RUNTIME_IDENTITY[:2]


@dataclass(frozen=True)
class ActiveCase:
    case_id: str
    trade_id: int
    approval_ref: str
    classification: str
    operational_scopes: tuple


@dataclass(frozen=True)
class Policy:
    deployment_id: str
    database_id: str
    revision: int
    active: tuple
    scope: str | None = None

    @property
    def all_excluded_ids(self):
        # Any active case continues to protect the historical row from writes.
        return frozenset(item.trade_id for item in self.active)

    @property
    def excluded_ids(self):
        return frozenset(item.trade_id for item in self.active
            if self.scope is None or self.scope in item.operational_scopes)

    def metadata(self):
        selected = [item for item in self.active
            if self.scope is None or self.scope in item.operational_scopes]
        return {'deployment_id': self.deployment_id, 'database_id': self.database_id,
            'policy_revision': self.revision, 'scope': self.scope,
            'excluded_count': len(selected),
            'exclusions': [{'case_id': item.case_id, 'trade_id': item.trade_id,
                'approval_ref': item.approval_ref, 'classification': item.classification,
                'operational_scopes': list(item.operational_scopes)} for item in selected],
            'recovery_pending_cases': [{'case_id': item.case_id, 'trade_id': item.trade_id,
                'classification': 'UNRESOLVED'} for item in self.active],
            'recovery_complete': not self.active,
            'accounting_complete': False if self.active else None,
            'provenance_proved': False}


def validate_receipt(result, request):
    result = document(result)
    expected = {"case_id": request["case_id"], "trade_id": request["trade_id"],
                "state": "ACTIVE" if request["action"] == "APPLY" else "REVOKED",
                "case_revision": request["expected_case_revision"] + 1,
                "policy_revision": request["expected_policy_revision"] + 1,
                "approval_ref": request["approval_ref"]}
    if result != expected:
        fail("AUDIT_INVALID")
    return result


def read_policy(cursor, expected_identity, *, exclusive=False, scope=None):
    """Hold the policy lock until the caller ends its transaction.

    Lock order is always policy, then trade. Every caller must use READ
    COMMITTED; an autocommit caller must never use this reader.
    """
    identity(*expected_identity)
    if scope is not None:
        checked_scope(scope)
    cursor.execute("SET LOCAL search_path = pg_catalog")
    cursor.execute("SHOW transaction_isolation")
    isolation = cursor.fetchone()
    if isolation is None or isolation[0] != "read committed":
        fail("ISOLATION_INVALID")
    cursor.execute("SET LOCAL lock_timeout = '1500ms'")
    cursor.execute("SET LOCAL statement_timeout = '5000ms'")
    cursor.execute("SET LOCAL TIME ZONE 'UTC'")
    # SELECT FOR SHARE would require UPDATE privilege on the policy table.
    # A transaction advisory lock keeps runtime roles read-only. Mutation takes
    # the exclusive counterpart before its policy-row lock and CAS.
    lock = "pg_advisory_xact_lock" if exclusive else "pg_advisory_xact_lock_shared"
    cursor.execute("SELECT pg_catalog." + lock + "(pg_catalog.hashtext('jbravo:trade-disposition:policy'))")
    cursor.execute("SELECT deployment_id, database_id, database_name, policy_revision, pg_catalog.current_database() "
                   "FROM public.trade_disposition_policy WHERE singleton=1" + (" FOR UPDATE" if exclusive else ""))
    row = cursor.fetchone()
    if row is None or tuple(row[:2]) != expected_identity or row[2] != row[4]:
        fail("IDENTITY_MISMATCH")
    policy_revision = revision(row[3])
    cursor.execute("SELECT c.request::text, c.row_snapshot::text, c.manifest::text, "
                   "c.case_revision, a.request_digest, a.policy_revision, pg_catalog.to_jsonb(t)::text, "
                   "a.request::text, a.result::text "
                   "FROM public.trade_disposition_cases c LEFT JOIN public.trades t ON t.trade_id=c.trade_id "
                   "LEFT JOIN public.trade_disposition_audit a ON a.approval_ref=c.request->>'approval_ref' "
                   "WHERE c.state='ACTIVE' ORDER BY c.trade_id")
    active, seen = [], set()
    for (saved_request, saved_snapshot, manifest, case_rev, saved_digest, audit_rev,
         current_snapshot, audit_request, audit_result) in cursor.fetchall():
        request, _ = validate_request(saved_request, manifest, expected_identity)
        if document(audit_request) != request:
            fail("AUDIT_INVALID")
        validate_receipt(audit_result, request)
        if (request["action"] != "APPLY" or case_rev != 1 or request["expected_case_revision"] != 0
                or saved_digest != fingerprint(request) or audit_rev != request["expected_policy_revision"] + 1
                or audit_rev > policy_revision or request["trade_id"] in seen or current_snapshot is None):
            fail("CASE_INVALID")
        saved, current = document(saved_snapshot), document(current_snapshot)
        if (saved != current or fingerprint(current) != request["row_fingerprint"]
                or current.get("trade_id") != request["trade_id"]
                or current.get("updated_at") != request["row_version"]):
            fail("ROW_CHANGED")
        seen.add(request["trade_id"])
        active.append(ActiveCase(request["case_id"], request["trade_id"], request["approval_ref"],
            "UNRESOLVED", tuple(sorted(document(manifest)["operational_scopes"]))))
    return Policy(*expected_identity, policy_revision, tuple(active), scope)


def eligible_sql(alias='trades', *, scope):
    """Only after validated read_policy in the same shared-lock transaction."""
    checked_scope(scope)
    if not re.fullmatch(r'[a-z_][a-z_0-9]*', alias):
        fail('SQL_ALIAS_INVALID')
    return ("NOT EXISTS (SELECT 1 FROM public.trade_disposition_cases disposition "
        f"WHERE disposition.trade_id={alias}.trade_id AND disposition.state='ACTIVE' "
        f"AND pg_catalog.jsonb_exists(disposition.manifest->'operational_scopes','{scope}'))")


def require_writable(policy, trade_id):
    if positive_id(trade_id) in policy.all_excluded_ids:
        fail("TRADE_EXCLUDED")


def require_complete_accounting(records):
    """An unknown trade result makes analytics incomplete, never zero-valued."""
    for row in records:
        if row.get("status", "CLOSED") != "CLOSED":
            continue
        for key in ("realized_pnl", "qty", "entry_price", "exit_price"):
            value = row.get(key)
            if isinstance(value, bool):
                fail("ACCOUNTING_INCOMPLETE")
            try:
                number = float(value)
                if (not math.isfinite(number) or (key == "qty" and number == 0)
                        or (key in {"entry_price", "exit_price"} and number <= 0)):
                    fail("ACCOUNTING_INCOMPLETE")
            except (ValueError, TypeError, OverflowError):
                fail("ACCOUNTING_INCOMPLETE")
        for key in ("entry_time", "exit_time"):
            value = row.get(key)
            aware_time(value.isoformat() if isinstance(value, datetime) else value)


def change_disposition(conn, request, manifest, expected_identity):
    """Explicit offline-callable API; never discovers connections or approvals.

    Own one idle PostgreSQL transaction. An approval is data, not authority to
    invoke this function. Operational callers must independently authenticate
    the approver and authorize this exact request before invocation.
    """
    request, manifest = validate_request(request, manifest, expected_identity)
    if conn is None or conn.autocommit or conn.get_transaction_status() != 0:
        fail("TRANSACTION_NOT_IDLE")
    request_digest = fingerprint(request)
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            cursor.execute("SET LOCAL lock_timeout = '1500ms'")
            cursor.execute("SET LOCAL statement_timeout = '5000ms'")
            policy = read_policy(cursor, expected_identity, exclusive=True)
            cursor.execute("SELECT request_digest, result::text FROM public.trade_disposition_audit WHERE approval_ref=%s",
                           (request["approval_ref"],))
            previous = cursor.fetchone()
            if previous:
                if previous[0] != request_digest:
                    fail("APPROVAL_REUSED")
                return validate_receipt(previous[1], request)
            if policy.revision != request["expected_policy_revision"]:
                fail("POLICY_CAS_FAILED")
            cursor.execute("SELECT pg_catalog.to_jsonb(t)::text FROM public.trades t WHERE trade_id=%s FOR UPDATE",
                           (request["trade_id"],))
            row = cursor.fetchone()
            if row is None:
                fail("TRADE_MISSING")
            snapshot = document(row[0])
            if snapshot.get("status") != "CLOSED":
                fail("TRADE_NOT_CLOSED")
            if (snapshot.get("trade_id") != request["trade_id"]
                    or snapshot.get("updated_at") != request["row_version"]
                    or fingerprint(snapshot) != request["row_fingerprint"]):
                fail("ROW_CHANGED")
            cursor.execute("SELECT state, case_revision, request::text, row_snapshot::text, manifest::text "
                           "FROM public.trade_disposition_cases WHERE case_id=%s FOR UPDATE", (request["case_id"],))
            case = cursor.fetchone()
            if request["action"] == "APPLY":
                if case is not None or request["expected_case_revision"] != 0 or request["trade_id"] in policy.all_excluded_ids:
                    fail("CASE_CONFLICT")
                cursor.execute("INSERT INTO public.trade_disposition_cases "
                               "(case_id,trade_id,state,case_revision,request,row_snapshot,manifest) "
                               "VALUES (%s,%s,'ACTIVE',1,%s::pg_catalog.jsonb,%s::pg_catalog.jsonb,%s::pg_catalog.jsonb)",
                               (request["case_id"], request["trade_id"], canonical(request), row[0], canonical(manifest)))
                state, case_revision = "ACTIVE", 1
            else:
                if (case is None or case[0] != "ACTIVE" or case[1] != request["expected_case_revision"]
                        or document(case[3]) != snapshot or document(case[4]) != manifest):
                    fail("CASE_CAS_FAILED")
                original = document(case[2])
                for key in ("trade_id", "row_fingerprint", "row_version", "deployment_id", "database_id", "manifest_sha256"):
                    if original[key] != request[key]:
                        fail("CASE_CONFLICT")
                cursor.execute("UPDATE public.trade_disposition_cases SET state='REVOKED',case_revision=case_revision+1 "
                               "WHERE case_id=%s AND state='ACTIVE' AND case_revision=%s",
                               (request["case_id"], request["expected_case_revision"]))
                if cursor.rowcount != 1:
                    fail("CASE_CAS_FAILED")
                state, case_revision = "REVOKED", case[1] + 1
            cursor.execute("UPDATE public.trade_disposition_policy SET policy_revision=policy_revision+1 "
                           "WHERE singleton=1 AND policy_revision=%s", (policy.revision,))
            if cursor.rowcount != 1:
                fail("POLICY_CAS_FAILED")
            result = {"case_id": request["case_id"], "trade_id": request["trade_id"], "state": state,
                      "case_revision": case_revision, "policy_revision": policy.revision + 1,
                      "approval_ref": request["approval_ref"]}
            cursor.execute("INSERT INTO public.trade_disposition_audit "
                           "(approval_ref,request_digest,request,manifest,result,policy_revision) "
                           "VALUES (%s,%s,%s::pg_catalog.jsonb,%s::pg_catalog.jsonb,%s::pg_catalog.jsonb,%s)",
                           (request["approval_ref"], request_digest, canonical(request), canonical(manifest),
                            canonical(result), policy.revision + 1))
            return result
