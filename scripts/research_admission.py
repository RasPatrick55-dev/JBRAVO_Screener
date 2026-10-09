"""Bounded, inert declaration checks; never data admission or execution authority."""

from dataclasses import dataclass, field
from datetime import date, datetime
import json
import math
import re

SCHEMA = "jbravo.research-admission.v1"
MAX_BYTES = 512 * 1024
MAX_RECORDS = 3000
MAX_DEPTH = 12
VARIANTS = ("screener_v2", "legacy_portfolio_simulator", "ml_forward_return_proxy")
SOURCE_CLASSES = (
    "relational_daily_bars_snapshot", "ml_artifact_set", "selected_symbol_capture"
)


@dataclass(frozen=True)
class FrozenMetadata:
    """JSON object snapshot; arrays are tuples and scalar values are immutable."""

    entries: tuple


def _freeze(value):
    if type(value) is dict:
        return FrozenMetadata(tuple((k, _freeze(v)) for k, v in sorted(value.items())))
    if type(value) is list:
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    if type(value) is FrozenMetadata:
        return {k: _thaw(v) for k, v in value.entries}
    if type(value) is tuple:
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True, order=True)
class Reason:
    field: str
    code: str
    evidence_id: str = ""
    invalid: bool = False


@dataclass(frozen=True)
class AdmissionDecision:
    outcome: str
    specimen_kind: str
    specification: FrozenMetadata | None
    reasons: tuple[Reason, ...]
    metadata_bytes: int | None
    metadata_records: int
    research_qualified: bool = field(default=False, init=False)
    execution_authorized: bool = field(default=False, init=False)
    acceptance_declarations_verified: bool = field(default=False, init=False)

    def to_dict(self):
        """Return a fresh detached built-in copy, without writing it."""
        return {
            "outcome": self.outcome, "specimen_kind": self.specimen_kind,
            "research_qualified": False, "execution_authorized": False,
            "acceptance_declarations_verified": False,
            "specification_retained": self.specification is not None,
            "specification": _thaw(self.specification),
            "metadata_bytes": self.metadata_bytes, "metadata_records": self.metadata_records,
            "reasons": [dict(field=r.field, code=r.code, evidence_id=r.evidence_id,
                             invalid=r.invalid) for r in self.reasons],
        }


def validate_specification(specification) -> AdmissionDecision:
    """Validate exact built-in JSON metadata. References remain unverified assertions.

    INVALID_SPEC outranks HOLD, retaining all safely discoverable reasons. Unsafe
    traversal/serialization stops early without a snapshot; no caller input is changed.
    """
    reasons = []
    records = 0
    size = None

    def reason(code, path, identity="", invalid=False):
        reasons.append(Reason(path, code, identity, invalid))

    class BoundaryError(Exception):
        pass

    def stop(code, path):
        reason(code, path, invalid=True)
        raise BoundaryError

    def inspect(value, path, depth, ancestors):
        nonlocal records
        records += 1
        if records > MAX_RECORDS:
            stop("RECORD_LIMIT", path)
        if depth > MAX_DEPTH:
            stop("DEPTH_LIMIT", path)
        kind = type(value)
        if kind not in (dict, list, str, int, float, bool, type(None)):
            stop("NON_JSON_TYPE", path)
        if kind in (dict, list):
            if id(value) in ancestors:
                stop("CYCLE", path)
            if len(value) > MAX_RECORDS:
                stop("RECORD_LIMIT", path)
            nested = ancestors | {id(value)}
            if kind is dict:
                if any(type(k) is not str for k in value):
                    stop("NON_STRING_KEY", path)
                # Bound keys before sorting or constructing diagnostic paths.
                if any(len(k) > 256 for k in value):
                    stop("KEY_LIMIT", path)
                for key in sorted(value):
                    inspect(key, path, depth + 1, nested)
                    inspect(value[key], path + "." + key, depth + 1, nested)
            else:
                for index, item in enumerate(value):
                    inspect(item, f"{path}[{index}]", depth + 1, nested)
        elif kind is str and len(value) > MAX_BYTES:
            stop("BYTE_LIMIT", path)
        elif kind is int and value.bit_length() > 63:
            stop("INTEGER_LIMIT", path)
        elif kind is float and not math.isfinite(value):
            reason("NONFINITE", path, invalid=True)

    def decision(snapshot, kind="UNKNOWN"):
        ordered = tuple(sorted(set(reasons)))
        outcome = ("INVALID_SPEC" if any(r.invalid for r in ordered) else
                   "HOLD" if ordered else "STRUCTURALLY_READY_FOR_EVIDENCE_REVIEW")
        return AdmissionDecision(outcome, kind, snapshot, ordered, size, records)

    try:
        inspect(specification, "$", 0, set())
        # Always enforce aggregate size before semantic traversal or freezing.
        # NaN/Infinity/-Infinity tokens are budget accounting only, not valid
        # strict JSON, repaired inputs or a canonical metadata_bytes measurement.
        nonfinite = any(r.code == "NONFINITE" for r in reasons)
        budget_bytes = 0
        encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=True,
                                   separators=(",", ":"), allow_nan=True)
        for chunk in encoder.iterencode(specification):
            budget_bytes += len(chunk)  # ASCII characters equal UTF-8 bytes.
            if not nonfinite:
                size = budget_bytes
            if budget_bytes > MAX_BYTES:
                stop("BYTE_LIMIT", "$")
    except BoundaryError:
        return decision(None)

    def obj(value, path, keys):
        if type(value) is not dict:
            reason("TYPE", path, invalid=True)
            return {}
        for key in sorted(set(keys) - value.keys()):
            reason("MISSING_FIELD", path + "." + key, invalid=True)
        for key in sorted(value.keys() - set(keys)):
            reason("UNKNOWN_FIELD", path + "." + key, invalid=True)
        return value

    def text(value, path, reference=False):
        if type(value) is not str or not value.strip():
            reason("VALUE", path, invalid=True)
            return False
        if reference and "latest" in value.lower():
            reason("LATEST_BINDING", path, value, True)
            return False
        return True

    def enum(value, path, choices):
        if type(value) is not str or value not in choices:
            reason("UNSUPPORTED_VALUE", path, invalid=True)

    def number(value, path, low, high, integral=False):
        allowed = (int,) if integral else (int, float)
        if type(value) not in allowed or not math.isfinite(value) or not low <= value <= high:
            reason("NUMBER", path, invalid=True)
            return False
        return True

    def sequence(value, path, nonempty=True):
        if type(value) is not list:
            reason("TYPE", path, invalid=True)
            return []
        if nonempty and not value:
            reason("EMPTY", path, invalid=True)
        return value

    def identifiers(value, path):
        items = sequence(value, path)
        seen = set()
        for index, item in enumerate(items):
            if text(item, f"{path}[{index}]", True):
                if item in seen:
                    reason("DUPLICATE_ID", path, item, True)
                seen.add(item)
        return seen

    def day(value, path):
        if type(value) is not str or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            reason("DATE", path, invalid=True)
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            reason("DATE", path, invalid=True)
            return None

    def interval(value, path):
        data = obj(value, path, ("start", "end"))
        start, end = day(data.get("start"), path + ".start"), day(data.get("end"), path + ".end")
        if start and end:
            if start > end:
                reason("REVERSED_INTERVAL", path, invalid=True)
            else:
                return start, end
        return None

    def digest(value, path, length):
        if type(value) is not str or not re.fullmatch("[0-9a-fA-F]{" + str(length) + "}", value):
            reason("HASH", path, invalid=True)

    def bindings(value, path, keys):
        data = obj(value, path, keys)
        for key in keys:
            identity = data.get(key)
            if identity is None:
                reason("MISSING_EVIDENCE", path + "." + key)
            elif text(identity, path + "." + key, True):
                artifact = artifacts.get(identity)
                if artifact is None:
                    reason("UNBOUND_EVIDENCE", path + "." + key, identity)
                elif artifact.get("role") != key:
                    reason("ROLE_CONFLICT", path + "." + key, identity)
        return data

    qualifier_keys = ("qualified_for_research", "research_qualified", "action_coverage_complete",
                      "executable", "available_for_warmup", "eligibility", "status")

    def qualifiers(data, path, identity=""):
        for key in qualifier_keys[:5]:
            value = data.get(key)
            if value is not None and type(value) is not bool:
                reason("TYPE", path + "." + key, identity, True)
            elif value is not True:
                reason("UNKNOWN_QUALIFIER" if value is None else "NEGATIVE_QUALIFIER", path + "." + key, identity)
        enum(data.get("eligibility"), path + ".eligibility", ("DECLARED", "UNKNOWN", "INELIGIBLE"))
        enum(data.get("status"), path + ".status", ("DECLARED", "UNRESOLVED", "HALTED", "RESUMPTION_DOCUMENTED"))
        if data.get("eligibility") != "DECLARED":
            reason("UNRESOLVED_ELIGIBILITY", path + ".eligibility", identity)
        if data.get("status") != "DECLARED":
            reason("UNRESOLVED_STATUS", path + ".status", identity)

    if type(specification) is not dict:
        reason("TYPE", "$", invalid=True)
        return decision(None)
    root = obj(specification, "$", ("schema", "specimen_kind", "strategy", "experiment",
                                     "inputs", "acceptance_declarations"))
    enum(root.get("schema"), "$.schema", (SCHEMA,))
    enum(root.get("specimen_kind"), "$.specimen_kind", ("SYNTHETIC", "HISTORICAL"))
    inputs = obj(root.get("inputs"), "$.inputs", (
        "source_class", "provider", "feed", "execution_adjustment", "feature_adjustment",
        "execution_units", "feature_units", "knowledge_time", "security_ids", "interval",
        "universe_scope", "selected_symbol_limitation", "artifacts", "bindings", "ml_lineage",
        "qualified_for_research", "research_qualified", "action_coverage_complete",
        "executable", "available_for_warmup", "eligibility", "status"))
    enum(inputs.get("source_class"), "$.inputs.source_class", SOURCE_CLASSES)
    securities = identifiers(inputs.get("security_ids"), "$.inputs.security_ids")
    scope = interval(inputs.get("interval"), "$.inputs.interval")
    for key in ("provider", "feed", "execution_adjustment", "feature_adjustment",
                "execution_units", "feature_units"):
        text(inputs.get(key), "$.inputs." + key)
        if inputs.get(key) == "UNKNOWN":
            reason("UNKNOWN_EVIDENCE", "$.inputs." + key)
    knowledge = inputs.get("knowledge_time")
    try:
        if type(knowledge) is not str or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", knowledge):
            raise ValueError
        datetime.fromisoformat(knowledge.replace("Z", "+00:00"))
    except ValueError:
        reason("DATE", "$.inputs.knowledge_time", invalid=True)
    enum(inputs.get("universe_scope"), "$.inputs.universe_scope", ("HISTORICAL_UNIVERSE", "SELECTED_SYMBOLS"))
    if inputs.get("universe_scope") == "SELECTED_SYMBOLS":
        text(inputs.get("selected_symbol_limitation"), "$.inputs.selected_symbol_limitation")
    elif inputs.get("selected_symbol_limitation") is not None:
        reason("SCOPE_CONFLICT", "$.inputs.selected_symbol_limitation")
    if inputs.get("source_class") == "selected_symbol_capture" and inputs.get("universe_scope") != "SELECTED_SYMBOLS":
        reason("POPULATION_CONFLICT", "$.inputs.universe_scope")
    qualifiers(inputs, "$.inputs")

    artifacts = {}
    artifact_items = sequence(inputs.get("artifacts"), "$.inputs.artifacts", False)
    artifact_roles = set()
    for index, item in enumerate(artifact_items):
        path = f"$.inputs.artifacts[{index}]"
        artifact = obj(item, path, ("id", "role", "sha256", "bytes", "producer_id", "request_id",
                                   "run_id", "source_class", "security_ids", "interval", "upstream_qualifiers"))
        identity = artifact.get("id")
        valid_id = text(identity, path + ".id", True)
        for key in ("role", "producer_id", "request_id", "run_id"):
            text(artifact.get(key), path + "." + key, True)
        digest(artifact.get("sha256"), path + ".sha256", 64)
        number(artifact.get("bytes"), path + ".bytes", 0, 2**63 - 1, True)
        enum(artifact.get("source_class"), path + ".source_class", SOURCE_CLASSES + ("declaration_metadata",))
        artifact_scope = interval(artifact.get("interval"), path + ".interval")
        artifact_securities = identifiers(artifact.get("security_ids"), path + ".security_ids")
        if artifact_scope != scope or artifact_securities != securities:
            reason("SCOPE_CONFLICT", path, identity if valid_id else "")
        if valid_id:
            if identity in artifacts:
                reason("DUPLICATE_ID", path + ".id", identity, True)
                if artifact != artifacts[identity]:
                    reason("IDENTITY_CONFLICT", path, identity)
            else:
                artifacts[identity] = artifact
        role = artifact.get("role")
        enum(role, path + ".role", ("bars", "security", "calendar", "coverage", "quality", "actions", "eligibility",
             "universe", "strategy_source", "strategy_config", "parameters", "experiment_source",
             "experiment_config", "dependencies", "ordering", "baseline", "features", "labels", "predictions"))
        if type(role) is str:
            if role in artifact_roles:
                reason("DUPLICATE_ROLE", path + ".role", identity if valid_id else "", True)
            artifact_roles.add(role)
        if role in ("bars", "features", "labels", "predictions") and artifact.get("source_class") != inputs.get("source_class"):
            reason("POPULATION_CONFLICT", path + ".source_class", identity if valid_id else "")
        if role in ("features", "labels", "predictions") and inputs.get("source_class") != "ml_artifact_set":
            reason("POPULATION_CONFLICT", path + ".role", identity if valid_id else "")
        upstream = artifact.get("upstream_qualifiers")
        if artifact.get("source_class") == "declaration_metadata":
            if upstream is not None:
                reason("CONFLICTING_DECLARATION", path + ".upstream_qualifiers", identity if valid_id else "")
            if role in ("bars", "features", "labels", "predictions"):
                reason("POPULATION_CONFLICT", path + ".source_class", identity if valid_id else "")
        else:
            if artifact.get("source_class") != inputs.get("source_class"):
                reason("POPULATION_CONFLICT", path + ".source_class", identity if valid_id else "")
            upstream = obj(upstream, path + ".upstream_qualifiers", qualifier_keys)
            qualifiers(upstream, path + ".upstream_qualifiers", identity if valid_id else "")
            for key in qualifier_keys:
                if upstream.get(key) != inputs.get(key):
                    reason("QUALIFIER_CONFLICT", path + ".upstream_qualifiers." + key, identity if valid_id else "")

    input_bindings = bindings(inputs.get("bindings"), "$.inputs.bindings",
                              ("bars", "security", "calendar", "coverage", "quality", "actions", "eligibility", "universe"))
    lineage = inputs.get("ml_lineage")
    if inputs.get("source_class") == "ml_artifact_set":
        linked = bindings(lineage, "$.inputs.ml_lineage", ("features", "labels", "predictions"))
        linked_ids = [linked.get(k) for k in ("features", "labels", "predictions")]
        if all(type(v) is str and v in artifacts and type(artifacts[v].get("run_id")) is str for v in linked_ids):
            if len(set(linked_ids)) != 3 or len({artifacts[v].get("run_id") for v in linked_ids}) != 1:
                reason("LINEAGE_CONFLICT", "$.inputs.ml_lineage")
    elif lineage is not None:
        reason("POPULATION_CONFLICT", "$.inputs.ml_lineage")

    strategy = obj(root.get("strategy"), "$.strategy", (
        "id", "version", "repository", "source_commit", "variant", "bindings", "parameters",
        "feature_policy", "warmup_policy", "entry_rules", "exit_rules", "sizing_rules", "capital_rules",
        "signal_availability", "execution_timing", "fill_policy", "stop_policy", "expiry_policy",
        "missing_session_policy", "terminal_position_policy"))
    for key in ("id", "version", "repository"):
        text(strategy.get(key), "$.strategy." + key, True)
    digest(strategy.get("source_commit"), "$.strategy.source_commit", 40)
    enum(strategy.get("variant"), "$.strategy.variant", VARIANTS)
    if strategy.get("variant") == "ml_forward_return_proxy" and inputs.get("source_class") != "ml_artifact_set":
        reason("POPULATION_CONFLICT", "$.strategy.variant")
    bindings(strategy.get("bindings"), "$.strategy.bindings", ("strategy_source", "strategy_config", "parameters"))
    for key in ("feature_policy", "warmup_policy", "entry_rules", "exit_rules", "sizing_rules", "capital_rules",
                "signal_availability", "execution_timing", "fill_policy", "stop_policy", "expiry_policy",
                "missing_session_policy", "terminal_position_policy"):
        text(strategy.get(key), "$.strategy." + key)
    parameter_ids = set()
    for index, item in enumerate(sequence(strategy.get("parameters"), "$.strategy.parameters")):
        path = f"$.strategy.parameters[{index}]"
        parameter = obj(item, path, ("name", "value", "unit"))
        name = parameter.get("name")
        if text(name, path + ".name"):
            if name in parameter_ids:
                reason("DUPLICATE_ID", path, name, True)
            parameter_ids.add(name)
        number(parameter.get("value"), path + ".value", -1e12, 1e12)
        text(parameter.get("unit"), path + ".unit")

    experiment = obj(root.get("experiment"), "$.experiment", (
        "id", "candidate", "baseline", "bindings", "study", "splits", "prior_exposure", "warmup",
        "benchmark", "accounting", "costs", "cost_stress", "cost_basis", "limits", "decision_rules"))
    text(experiment.get("id"), "$.experiment.id", True)
    for name in ("candidate", "baseline"):
        path = "$.experiment." + name
        binding = obj(experiment.get(name), path, ("strategy_id", "version", "repository", "source_commit", "variant"))
        for key in ("strategy_id", "version", "repository"):
            text(binding.get(key), path + "." + key, True)
        digest(binding.get("source_commit"), path + ".source_commit", 40)
        enum(binding.get("variant"), path + ".variant", VARIANTS)
        if name == "candidate":
            for key, target in (("strategy_id", "id"), ("version", "version"), ("repository", "repository"),
                                ("source_commit", "source_commit"), ("variant", "variant")):
                if binding.get(key) != strategy.get(target):
                    reason("IDENTITY_CONFLICT", path + "." + key)
    bindings(experiment.get("bindings"), "$.experiment.bindings",
             ("experiment_source", "experiment_config", "dependencies", "ordering", "baseline"))
    study = interval(experiment.get("study"), "$.experiment.study")
    warmup = interval(experiment.get("warmup"), "$.experiment.warmup")
    if study and warmup and warmup[1] >= study[0]:
        reason("INCOMPATIBLE_INTERVAL", "$.experiment.warmup", invalid=True)
    if scope and study and warmup and not (scope[0] <= warmup[0] and scope[1] >= study[1]):
        reason("SCOPE_CONFLICT", "$.inputs.interval")
    splits = obj(experiment.get("splits"), "$.experiment.splits", ("train", "validation", "test"))
    previous = None
    for name in ("train", "validation", "test"):
        path = "$.experiment.splits." + name
        split = obj(splits.get(name), path, ("interval", "not_used_reason"))
        if split.get("interval") is None:
            text(split.get("not_used_reason"), path + ".not_used_reason")
        else:
            bounds = interval(split.get("interval"), path + ".interval")
            if split.get("not_used_reason") is not None:
                reason("CONFLICTING_DECLARATION", path)
            if study and bounds:
                if bounds[0] < study[0] or bounds[1] > study[1] or (previous and bounds[0] <= previous[1]):
                    reason("INCOMPATIBLE_INTERVAL", path, invalid=True)
                previous = bounds
    exposure = obj(experiment.get("prior_exposure"), "$.experiment.prior_exposure", ("status", "references", "note"))
    enum(exposure.get("status"), "$.experiment.prior_exposure.status", ("KNOWN", "UNKNOWN"))
    refs = sequence(exposure.get("references"), "$.experiment.prior_exposure.references", False)
    exposure_refs = set()
    for index, ref in enumerate(refs):
        if text(ref, f"$.experiment.prior_exposure.references[{index}]", True):
            if ref in exposure_refs:
                reason("DUPLICATE_ID", "$.experiment.prior_exposure.references", ref, True)
            exposure_refs.add(ref)
    text(exposure.get("note"), "$.experiment.prior_exposure.note")
    if exposure.get("status") != "KNOWN" or not refs:
        reason("PRIOR_EXPOSURE_UNRESOLVED", "$.experiment.prior_exposure")
    for key in ("benchmark", "accounting"):
        text(experiment.get(key), "$.experiment." + key)
    enum(experiment.get("cost_basis"), "$.experiment.cost_basis", ("CALIBRATION_DECLARED", "UNCALIBRATED_REFERENCE"))
    if experiment.get("cost_basis") != "CALIBRATION_DECLARED":
        reason("UNCALIBRATED_COSTS", "$.experiment.cost_basis")
    for name in ("costs", "cost_stress"):
        path = "$.experiment." + name
        costs = obj(experiment.get(name), path, ("fee", "slippage"))
        valid = []
        values = []
        for key, unit, ceiling in (("fee", "CURRENCY_PER_FILL", 1e12), ("slippage", "FRACTION_PER_SIDE", 1 - 1e-12)):
            cost = obj(costs.get(key), path + "." + key, ("value", "unit"))
            enum(cost.get("unit"), path + "." + key + ".unit", (unit,))
            valid.append(number(cost.get("value"), path + "." + key + ".value", 0, ceiling))
            values.append(cost.get("value"))
        if name == "cost_stress" and all(valid) and not any(v > 0 for v in values):
            reason("ZERO_COST_STRESS", path, invalid=True)
        if name == "costs" and all(valid) and not any(v > 0 for v in values):
            reason("ZERO_COST_REFERENCE", path)
    limits = obj(experiment.get("limits"), "$.experiment.limits", ("trials", "runs", "seconds", "include_failures"))
    for key, ceiling in (("trials", 10000), ("runs", 10000), ("seconds", 86400)):
        number(limits.get(key), "$.experiment.limits." + key, 1, ceiling, True)
    if limits.get("include_failures") is not True:
        reason("FAILURES_NOT_BUDGETED", "$.experiment.limits.include_failures", invalid=True)
    rule_refs = {}
    for index, item in enumerate(sequence(experiment.get("decision_rules"), "$.experiment.decision_rules")):
        path = f"$.experiment.decision_rules[{index}]"
        rule = obj(item, path, ("reference", "disposition"))
        ref = rule.get("reference")
        if text(ref, path + ".reference", True):
            if ref in rule_refs:
                reason("DUPLICATE_ID", path + ".reference", ref, True)
                if rule_refs[ref] != rule.get("disposition"):
                    reason("CONFLICTING_DECLARATION", path, ref)
            else:
                rule_refs[ref] = rule.get("disposition")
        enum(rule.get("disposition"), path + ".disposition", ("PROPOSED", "ADOPTED"))

    declared_ids = set()
    declaration_ids = set()
    for index, item in enumerate(sequence(root.get("acceptance_declarations"), "$.acceptance_declarations", False)):
        path = f"$.acceptance_declarations[{index}]"
        declaration = obj(item, path, ("id", "declarant", "what", "evidence_id", "decision_reference",
                                      "security_ids", "interval"))
        for key in ("id", "declarant", "what", "decision_reference"):
            text(declaration.get(key), path + "." + key, key != "what")
        declaration_id = declaration.get("id")
        if type(declaration_id) is str:
            if declaration_id in declaration_ids:
                reason("DUPLICATE_ID", path + ".id", declaration_id, True)
            declaration_ids.add(declaration_id)
        identity = declaration.get("evidence_id")
        # Scope syntax is independently discoverable even with a malformed ID.
        bounds = interval(declaration.get("interval"), path + ".interval")
        security_ids = identifiers(declaration.get("security_ids"), path + ".security_ids")
        if bounds != scope or security_ids != securities:
            reason("SCOPE_CONFLICT", path, identity if type(identity) is str else "")
        if text(identity, path + ".evidence_id", True):
            if identity in declared_ids:
                reason("CONFLICTING_DECLARATION", path, identity)
            declared_ids.add(identity)
            artifact = artifacts.get(identity)
            if artifact is None:
                reason("UNBOUND_EVIDENCE", path, identity)
    for identity in sorted(artifacts.keys() - declared_ids):
        reason("MISSING_ACCEPTANCE_DECLARATION", "$.acceptance_declarations", identity)
    # Retain every submitted field, including false/null qualifiers and references.
    kind = root.get("specimen_kind")
    return decision(_freeze(specification), kind if type(kind) is str else "UNKNOWN")
