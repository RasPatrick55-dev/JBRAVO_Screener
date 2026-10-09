"""52 inline synthetic cases. No bot imports, datasets, subprocesses or evaluation."""

import builtins
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError
import datetime  # Preload standard-library dependencies before effect guards.
import importlib
import io
import json
import logging
import math
import os
import pathlib
import re
import socket
import subprocess
import sys

import pytest


@contextmanager
def effects_forbidden(monkeypatch):
    """Guard the actual normal import AND calls; source reads remain available."""
    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden effect boundary")

    original_open, original_io_open, original_os_open = builtins.open, io.open, os.open
    original_import = builtins.__import__

    def read_only_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            forbidden()
        return original_open(file, mode, *args, **kwargs)

    def read_only_io_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            forbidden()
        return original_io_open(file, mode, *args, **kwargs)

    def read_only_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            forbidden()
        return original_os_open(path, flags, *args, **kwargs)

    def restricted_import(name, *args, **kwargs):
        if (name.startswith("scripts.") and name != "scripts.research_admission") or name.split(".")[0] in (
            "alpaca", "requests", "dotenv", "psycopg", "psycopg2", "pandas", "numpy", "utils"
        ):
            forbidden()
        return original_import(name, *args, **kwargs)

    class ForbiddenEnvironment:
        __getitem__ = __setitem__ = __iter__ = __contains__ = get = keys = items = forbidden

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", read_only_open)
        guard.setattr(io, "open", read_only_io_open)
        guard.setattr(os, "open", read_only_os_open)
        guard.setattr(builtins, "__import__", restricted_import)
        guard.setattr(os, "getenv", forbidden)
        guard.setattr(os, "putenv", forbidden)
        guard.setattr(os, "environ", ForbiddenEnvironment())
        for name in ("mkdir", "makedirs", "remove", "unlink", "rename", "replace", "rmdir",
                     "system", "startfile", "fork", "forkpty", "posix_spawn", "posix_spawnp",
                     "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
            if hasattr(os, name):
                guard.setattr(os, name, forbidden)
        for name in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
            guard.setattr(socket, name, forbidden)
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            guard.setattr(subprocess, name, forbidden)
        for name in ("basicConfig", "FileHandler", "StreamHandler"):
            guard.setattr(logging, name, forbidden)
        guard.setattr(logging.Logger, "addHandler", forbidden)
        yield


@pytest.fixture
def admission(monkeypatch):
    # Empty scripts/__init__.py is the only other repository module imported.
    sys.modules.pop("scripts.research_admission", None)
    with effects_forbidden(monkeypatch):
        module = importlib.import_module("scripts.research_admission")
    # Restore between operations so pytest's own bookkeeping is outside the
    # guarded product boundary. Every validator/output-copy call is guarded.
    class GuardedAdmission:
        @staticmethod
        def validate_specification(spec):
            with effects_forbidden(monkeypatch):
                result = module.validate_specification(spec)
                result.to_dict()
                return result

        @staticmethod
        def to_dict(decision):
            with effects_forbidden(monkeypatch):
                return decision.to_dict()
    return GuardedAdmission()


def specimen(variant="screener_v2", source_class="relational_daily_bars_snapshot"):
    """Fictional declarations; true flags deliberately do NOT establish truth."""
    scope = {"start": "2020-01-01", "end": "2020-12-31"}
    qualifiers = dict(qualified_for_research=True, research_qualified=True,
                      action_coverage_complete=True, executable=True,
                      available_for_warmup=True, eligibility="DECLARED", status="DECLARED")
    input_roles = ("bars", "security", "calendar", "coverage", "quality", "actions", "eligibility", "universe")
    strategy_roles = ("strategy_source", "strategy_config", "parameters")
    experiment_roles = ("experiment_source", "experiment_config", "dependencies", "ordering", "baseline")
    roles = input_roles + strategy_roles + experiment_roles
    if source_class == "ml_artifact_set":
        roles += ("features", "labels", "predictions")
    artifacts = []
    declarations = []
    for role in roles:  # Fixed fixture construction, not generated evaluation trials.
        data_role = role in ("bars", "features", "labels", "predictions")
        artifacts.append(dict(id=role + "-001", role=role, sha256="a" * 64, bytes=100,
                              producer_id="synthetic-producer-001", request_id="synthetic-request-001",
                              run_id="synthetic-run-001", source_class=source_class if data_role else "declaration_metadata",
                              security_ids=["synthetic-security-001"], interval=deepcopy(scope),
                              upstream_qualifiers=deepcopy(qualifiers) if data_role else None))
        declarations.append(dict(id="declaration-" + role, declarant="synthetic-independent-reviewer",
                                 what="Fictional evidence review declaration; unverified",
                                 evidence_id=role + "-001", decision_reference="synthetic-decision-" + role,
                                 security_ids=["synthetic-security-001"], interval=deepcopy(scope)))
    strategy = dict(id="synthetic-strategy-001", version="1", repository="synthetic-repository-001",
                    source_commit="c" * 40, variant=variant,
                    bindings={role: role + "-001" for role in strategy_roles},
                    parameters=[dict(name="warmup", value=20, unit="observed_sessions")])
    strategy.update(dict(feature_policy="Completed daily OHLCV; EMA20 declared",
                         warmup_policy="20 qualified observed sessions; reset after interruptions",
                         entry_rules="Explicit synthetic entry definition",
                         exit_rules="Explicit synthetic full/partial exit precedence",
                         sizing_rules="Whole shares; synthetic fixed allocation", capital_rules="USD cash; no borrowing",
                         signal_availability="After daily bar completion", execution_timing="Next qualified open",
                         fill_policy="Opening gap before queued exits", stop_policy="Old stop before close update",
                         expiry_policy="Expire on unresolved interruption", missing_session_policy="No invented fills",
                         terminal_position_policy="Carry stale marks and pending obligations explicitly"))
    candidate = dict(strategy_id=strategy["id"], version="1", repository=strategy["repository"],
                     source_commit=strategy["source_commit"], variant=variant)
    baseline = dict(candidate, strategy_id="synthetic-baseline-001")
    costs = dict(fee=dict(value=1, unit="CURRENCY_PER_FILL"), slippage=dict(value=0.001, unit="FRACTION_PER_SIDE"))
    experiment = dict(id="synthetic-experiment-001", candidate=candidate, baseline=baseline,
                      bindings={role: role + "-001" for role in experiment_roles},
                      study=dict(start="2020-02-01", end="2020-12-31"),
                      splits={"train": dict(interval=dict(start="2020-02-01", end="2020-05-31"), not_used_reason=None),
                              "validation": dict(interval=dict(start="2020-06-01", end="2020-08-31"), not_used_reason=None),
                              "test": dict(interval=dict(start="2020-09-01", end="2020-12-31"), not_used_reason=None)},
                      prior_exposure=dict(status="KNOWN", references=["synthetic-exposure-decision-001"], note="Fictional exposure log"),
                      warmup=dict(start="2020-01-01", end="2020-01-31"), benchmark="Synthetic USD benchmark; dividends explicit",
                      accounting="Gross/net/open costs and stale terminal marks reported separately",
                      costs=costs, cost_stress=dict(fee=dict(value=2, unit="CURRENCY_PER_FILL"),
                                                  slippage=dict(value=0.002, unit="FRACTION_PER_SIDE")),
                      cost_basis="CALIBRATION_DECLARED", limits=dict(trials=1, runs=1, seconds=30, include_failures=True),
                      decision_rules=[dict(reference="synthetic-proposed-rule-001", disposition="PROPOSED")])
    inputs = dict(source_class=source_class, provider="synthetic-provider", feed="synthetic-feed",
                  execution_adjustment="raw with separately declared share-unit policy",
                  feature_adjustment="explicit split/dividend feature policy", execution_units="USD/current declared share unit",
                  feature_units="USD/declared feature share unit", knowledge_time="2021-01-01T00:00:00Z",
                  security_ids=["synthetic-security-001"], interval=scope,
                  universe_scope="SELECTED_SYMBOLS" if source_class == "selected_symbol_capture" else "HISTORICAL_UNIVERSE",
                  selected_symbol_limitation="Selected symbols only; no survivorship or historical membership claim" if source_class == "selected_symbol_capture" else None,
                  artifacts=artifacts, bindings={role: role + "-001" for role in input_roles},
                  ml_lineage={role: role + "-001" for role in ("features", "labels", "predictions")} if source_class == "ml_artifact_set" else None,
                  **qualifiers)
    return dict(schema="jbravo.research-admission.v1", specimen_kind="SYNTHETIC", strategy=strategy,
                experiment=experiment, inputs=inputs, acceptance_declarations=declarations)


def assert_safe(decision):
    assert decision.research_qualified is False
    assert decision.execution_authorized is False
    assert decision.acceptance_declarations_verified is False
    assert "replay_ready" not in decision.to_dict()


def codes(decision):
    assert_safe(decision)
    return {reason.code for reason in decision.reasons}


@pytest.mark.parametrize("variant,population", [
    ("screener_v2", "relational_daily_bars_snapshot"),
    ("legacy_portfolio_simulator", "selected_symbol_capture"),
    ("ml_forward_return_proxy", "ml_artifact_set"),
])
def test_structural_variants(admission, variant, population):
    result = admission.validate_specification(specimen(variant, population))
    assert result.outcome == "STRUCTURALLY_READY_FOR_EVIDENCE_REVIEW"
    assert result.specimen_kind == "SYNTHETIC"
    assert result.reasons == ()
    assert_safe(result)


@pytest.mark.parametrize("value", [False, None])
def test_negative_or_unknown_qualifiers(admission, value):
    spec = specimen("legacy_portfolio_simulator", "selected_symbol_capture")
    for key in ("qualified_for_research", "research_qualified", "action_coverage_complete", "executable", "available_for_warmup"):
        spec["inputs"][key] = value
    spec["inputs"]["status"] = "HALTED"
    spec["inputs"]["eligibility"] = "UNKNOWN"
    result = admission.validate_specification(spec)
    assert result.outcome == "HOLD"
    assert {"UNRESOLVED_STATUS", "UNRESOLVED_ELIGIBILITY", "QUALIFIER_CONFLICT"} <= codes(result)
    assert result.to_dict()["specification"]["inputs"]["executable"] is value


def test_artifact_negative_not_hidden(admission):
    spec = specimen()
    spec["inputs"]["artifacts"][0]["upstream_qualifiers"]["research_qualified"] = False
    result = admission.validate_specification(spec)
    assert result.outcome == "HOLD"
    assert {"NEGATIVE_QUALIFIER", "QUALIFIER_CONFLICT"} <= codes(result)
    assert any(r.evidence_id == "bars-001" and r.code == "NEGATIVE_QUALIFIER" for r in result.reasons)


def test_acceptance_scope(admission):
    spec = specimen()
    spec["acceptance_declarations"][0]["security_ids"] = ["different-security"]
    spec["acceptance_declarations"][1]["interval"]["end"] = "2020-11-30"
    result = admission.validate_specification(spec)
    assert result.outcome == "HOLD"
    assert sum(r.code == "SCOPE_CONFLICT" for r in result.reasons) == 2
    spec["acceptance_declarations"][0]["evidence_id"] = None
    spec["acceptance_declarations"][0]["interval"]["start"] = "bad-date"
    malformed = admission.validate_specification(spec)
    assert malformed.outcome == "INVALID_SPEC"
    assert {"VALUE", "DATE", "SCOPE_CONFLICT"} <= codes(malformed)


def test_missing_evidence(admission):
    spec = specimen()
    spec["inputs"]["bindings"]["actions"] = None
    spec["inputs"]["bindings"]["calendar"] = "missing-calendar"
    spec["acceptance_declarations"].pop()
    result = admission.validate_specification(spec)
    assert result.outcome == "HOLD"
    assert {"MISSING_EVIDENCE", "UNBOUND_EVIDENCE", "MISSING_ACCEPTANCE_DECLARATION"} <= codes(result)


def test_population_and_lineage(admission):
    spec = specimen("ml_forward_return_proxy", "ml_artifact_set")
    spec["inputs"]["artifacts"][0]["source_class"] = "relational_daily_bars_snapshot"
    spec["inputs"]["artifacts"][-1]["run_id"] = "different-run"
    spec["experiment"]["candidate"]["variant"] = "screener_v2"
    result = admission.validate_specification(spec)
    assert result.outcome == "HOLD"
    assert {"POPULATION_CONFLICT", "LINEAGE_CONFLICT", "IDENTITY_CONFLICT"} <= codes(result)
    spec["inputs"]["source_class"] = "relational_daily_bars_snapshot"
    spec["inputs"]["ml_lineage"] = None
    extra_roles = admission.validate_specification(spec)
    assert any(r.code == "POPULATION_CONFLICT" and r.field.endswith(".role") for r in extra_roles.reasons)


def test_exposure(admission):
    spec = specimen()
    spec["experiment"]["prior_exposure"]["status"] = "UNKNOWN"
    spec["experiment"]["prior_exposure"]["references"] = []
    assert "PRIOR_EXPOSURE_UNRESOLVED" in codes(admission.validate_specification(spec))
    spec["experiment"]["prior_exposure"]["references"] = ["same-ref", "same-ref"]
    spec["experiment"]["decision_rules"].append(dict(spec["experiment"]["decision_rules"][0], disposition="ADOPTED"))
    assert {"DUPLICATE_ID", "CONFLICTING_DECLARATION"} <= codes(admission.validate_specification(spec))


@pytest.mark.parametrize("case", ["unit", "nonfinite", "boolean", "negative", "zero_stress", "uncalibrated", "zero_reference"])
def test_cost_boundary(admission, case):
    spec = specimen()
    fee = spec["experiment"]["costs"]["fee"]
    if case == "unit":
        fee["unit"] = "BASIS_POINTS"
    elif case == "nonfinite":
        fee["value"] = math.inf
    elif case == "boolean":
        fee["value"] = True
    elif case == "negative":
        fee["value"] = -1
    elif case == "uncalibrated":
        spec["experiment"]["cost_basis"] = "UNCALIBRATED_REFERENCE"
    else:
        target = "cost_stress" if case == "zero_stress" else "costs"
        spec["experiment"][target]["fee"]["value"] = 0
        spec["experiment"][target]["slippage"]["value"] = 0
    result = admission.validate_specification(spec)
    assert result.outcome == ("HOLD" if case in ("uncalibrated", "zero_reference") else "INVALID_SPEC")
    assert result.reasons
    assert_safe(result)


@pytest.mark.parametrize("case,expected", [
    ("hash", "HASH"), ("date", "DATE"), ("duplicate", "DUPLICATE_ID"),
    ("unknown", "UNKNOWN_FIELD"), ("latest", "LATEST_BINDING"),
    ("reversed", "REVERSED_INTERVAL"), ("splits", "INCOMPATIBLE_INTERVAL"),
])
def test_malformed(admission, case, expected):
    spec = specimen()
    if case == "hash":
        spec["strategy"]["source_commit"] = "bad"
    elif case == "date":
        spec["inputs"]["interval"]["start"] = "2020-02-30"
    elif case == "duplicate":
        spec["inputs"]["artifacts"].append(deepcopy(spec["inputs"]["artifacts"][0]))
    elif case == "unknown":
        spec["inputs"]["arbitrary_manifest"] = {"positive_summary": True}
    elif case == "latest":
        spec["inputs"]["bindings"]["bars"] = "runs/latest/bars"
    elif case == "reversed":
        spec["experiment"]["study"]["end"] = "2019-01-01"
    else:
        spec["experiment"]["splits"]["test"]["interval"]["start"] = "2020-08-31"
    result = admission.validate_specification(spec)
    assert result.outcome == "INVALID_SPEC"
    assert expected in codes(result)


@pytest.mark.parametrize("case,expected", [("size", "BYTE_LIMIT"), ("records", "RECORD_LIMIT"),
                                          ("depth", "DEPTH_LIMIT"), ("cycle", "CYCLE")])
def test_bounds(admission, case, expected):
    spec = specimen()
    if case == "size":
        spec["strategy"]["entry_rules"] = "x" * (512 * 1024)
    elif case == "records":
        spec["oversized"] = [None] * 3001
    elif case == "cycle":
        spec["cycle"] = spec
    else:
        nested = {}
        spec["deep"] = nested
        for _ in range(13):  # Fixed adversarial nesting, no trials.
            child = {}
            nested["child"] = child
            nested = child
    result = admission.validate_specification(spec)
    assert result.outcome == "INVALID_SPEC"
    assert expected in codes(result)
    assert result.specification is None


def test_deterministic_complete_blockers(admission):
    spec = specimen()
    spec["strategy"]["source_commit"] = "bad"
    spec["inputs"]["executable"] = False
    spec["inputs"]["action_coverage_complete"] = None
    spec["inputs"]["bindings"]["actions"] = None
    spec["experiment"]["prior_exposure"]["status"] = "UNKNOWN"
    result = admission.validate_specification(spec)
    reversed_keys = {key: spec[key] for key in reversed(list(spec))}
    assert result == admission.validate_specification(reversed_keys)
    assert result.reasons == tuple(sorted(result.reasons))
    assert {"HASH", "NEGATIVE_QUALIFIER", "UNKNOWN_QUALIFIER", "MISSING_EVIDENCE", "PRIOR_EXPOSURE_UNRESOLVED"} <= codes(result)
    assert result.outcome == "INVALID_SPEC"


def test_detached_unchanged_input(admission):
    spec = specimen()
    original = deepcopy(spec)
    result = admission.validate_specification(spec)
    assert spec == original
    copy = result.to_dict()
    copy["specification"]["strategy"]["parameters"][0]["value"] = 999
    spec["strategy"]["parameters"][0]["value"] = 555
    assert result.to_dict()["specification"]["strategy"]["parameters"][0]["value"] == 20
    with pytest.raises(FrozenInstanceError):
        result.research_qualified = True
    expected_bytes = len(json.dumps(original, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    assert result.metadata_bytes == expected_bytes
    assert_safe(result)


def test_import_and_call_guards(admission, monkeypatch):
    # Fixture imported normally with guards already installed; demonstrate guards.
    with effects_forbidden(monkeypatch):
        with pytest.raises(AssertionError):
            os.getenv("CREDENTIAL_DISCOVERY")
        with pytest.raises(AssertionError):
            socket.create_connection(("invalid", 1))
        with pytest.raises(AssertionError):
            subprocess.Popen(["forbidden"])
        with pytest.raises(AssertionError):
            pathlib.Path("must-not-exist").write_text("forbidden")
        with pytest.raises(AssertionError):
            os.mkdir("must-not-exist")
        with pytest.raises(AssertionError):
            logging.basicConfig()
    assert_safe(admission.validate_specification(specimen()))


def test_malformed_types_and_budget(admission):
    spec = specimen("ml_forward_return_proxy", "ml_artifact_set")
    spec["inputs"]["artifacts"][-1]["run_id"] = {}
    spec["inputs"]["artifacts"][0]["bytes"] = True
    spec["inputs"]["artifacts"][0]["sha256"] = "bad"
    spec["experiment"]["limits"]["runs"] = True
    spec["experiment"]["limits"]["include_failures"] = False
    spec["inputs"]["executable"] = "true"
    result = admission.validate_specification(spec)
    assert result.outcome == "INVALID_SPEC"
    assert {"TYPE", "NUMBER", "HASH", "FAILURES_NOT_BUDGETED", "VALUE"} <= codes(result)
    assert "NON_JSON_TYPE" in codes(admission.validate_specification({"bad": object()}))


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_nonfinite_and_aggregate_size(admission, nonfinite):
    spec = specimen()
    spec["strategy"]["entry_rules"] = "x" * 300_000
    spec["strategy"]["exit_rules"] = "y" * 300_000
    spec["experiment"]["costs"]["fee"]["value"] = nonfinite
    original = deepcopy(spec)
    result = admission.validate_specification(spec)
    assert result.outcome == "INVALID_SPEC"
    assert {"NONFINITE", "BYTE_LIMIT"} <= codes(result)
    assert result.specification is None
    assert result.metadata_bytes is None
    assert result.reasons == tuple(sorted(result.reasons))
    # JSON-like comparison preserves NaN/Infinity tokens without NaN == NaN.
    assert json.dumps(spec, sort_keys=True) == json.dumps(original, sort_keys=True)
    assert result == admission.validate_specification(spec)

    small = specimen()
    small["experiment"]["costs"]["fee"]["value"] = nonfinite
    small_original = deepcopy(small)
    diagnostic = admission.validate_specification(small)
    assert diagnostic.outcome == "INVALID_SPEC"
    assert "NONFINITE" in codes(diagnostic)
    assert "BYTE_LIMIT" not in codes(diagnostic)
    assert diagnostic.specification is not None
    assert diagnostic.metadata_bytes is None
    retained = diagnostic.to_dict()["specification"]["experiment"]["costs"]["fee"]["value"]
    assert math.isnan(retained) if math.isnan(nonfinite) else retained == nonfinite
    assert json.dumps(small, sort_keys=True) == json.dumps(small_original, sort_keys=True)


def test_finite_exact_byte_boundary(admission):
    spec = specimen()
    spec["strategy"]["entry_rules"] = ""
    empty_bytes = len(json.dumps(spec, sort_keys=True, ensure_ascii=True,
                                separators=(",", ":"), allow_nan=False).encode("utf-8"))
    spec["strategy"]["entry_rules"] = "x" * (512 * 1024 - empty_bytes)
    original = deepcopy(spec)
    assert len(json.dumps(spec, sort_keys=True, ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")) == 512 * 1024
    equal = admission.validate_specification(spec)
    assert equal.outcome == "STRUCTURALLY_READY_FOR_EVIDENCE_REVIEW"
    assert equal.metadata_bytes == 512 * 1024
    assert equal.specification is not None
    assert_safe(equal)
    assert spec == original

    spec["strategy"]["entry_rules"] += "x"
    overflow_original = deepcopy(spec)
    overflow = admission.validate_specification(spec)
    assert overflow.outcome == "INVALID_SPEC"
    assert "BYTE_LIMIT" in codes(overflow)
    assert overflow.specification is None
    assert overflow.metadata_bytes == 512 * 1024 + 1
    assert spec == overflow_original


def assert_invalid_root(decision, output):
    assert decision.outcome == output["outcome"] == "INVALID_SPEC"
    assert decision.specimen_kind == output["specimen_kind"] == "UNKNOWN"
    for field in ("research_qualified", "execution_authorized",
                  "acceptance_declarations_verified"):
        assert getattr(decision, field) is False
        assert output[field] is False
    assert decision.reasons == tuple(sorted(decision.reasons))


@pytest.mark.parametrize("root,records", [
    pytest.param(None, 1, id="null"),
    pytest.param(False, 1, id="false"),
    pytest.param(True, 1, id="true"),
    pytest.param(0, 1, id="zero"),
    pytest.param(1.25, 1, id="finite-float"),
    pytest.param("", 1, id="empty-string"),
    pytest.param([], 1, id="empty-list"),
    pytest.param([{"items": [None, False, 0]}], 7, id="nested-list-object"),
])
def test_retained_finite_root(admission, root, records):
    original = deepcopy(root)
    result = admission.validate_specification(root)
    output = admission.to_dict(result)
    assert_invalid_root(result, output)
    assert [(r.field, r.code, r.invalid) for r in result.reasons] == [("$", "TYPE", True)]
    assert result.specification_retained is True
    assert output["specification_retained"] is True
    assert type(output["specification"]) is type(root)
    assert output["specification"] == original
    assert type(result.specification) is (tuple if type(root) is list else type(root))
    expected_bytes = len(json.dumps(root, sort_keys=True, ensure_ascii=True,
                                   separators=(",", ":"), allow_nan=False).encode("utf-8"))
    assert result.metadata_bytes == output["metadata_bytes"] == expected_bytes
    assert result.metadata_records == output["metadata_records"] == records
    assert set(output) == {"outcome", "specimen_kind", "research_qualified",
                           "execution_authorized", "acceptance_declarations_verified",
                           "specification_retained", "specification", "metadata_bytes",
                           "metadata_records", "reasons"}
    assert type(root) is type(original)
    assert root == original


@pytest.mark.parametrize("root", [math.nan, math.inf, -math.inf],
                         ids=["nan", "positive-infinity", "negative-infinity"])
def test_retained_nonfinite_root(admission, root):
    result = admission.validate_specification(root)
    output = admission.to_dict(result)
    assert_invalid_root(result, output)
    assert {(r.field, r.code, r.invalid) for r in result.reasons} == {
        ("$", "NONFINITE", True), ("$", "TYPE", True)}
    assert result.specification_retained is True
    assert output["specification_retained"] is True
    assert result.specification is root
    assert type(output["specification"]) is float
    assert math.isnan(output["specification"]) if math.isnan(root) else output["specification"] == root
    assert result.metadata_bytes is output["metadata_bytes"] is None
    assert result.metadata_records == output["metadata_records"] == 1
    with pytest.raises(ValueError):
        json.dumps(output, allow_nan=False)


@pytest.mark.parametrize("case,expected", [
    ("oversized-string", "BYTE_LIMIT"), ("oversized-list", "BYTE_LIMIT"),
    ("cycle", "CYCLE"), ("non-json", "NON_JSON_TYPE"),
])
def test_unretained_root_boundary(admission, case, expected):
    if case == "oversized-string":
        root = "x" * (512 * 1024)  # Individual limit passes; JSON quotes overflow.
    elif case == "oversized-list":
        root = ["x" * 300_000, "y" * 300_000]
    elif case == "cycle":
        root = []
        root.append(root)
    else:
        root = object()
    result = admission.validate_specification(root)
    output = admission.to_dict(result)
    assert_invalid_root(result, output)
    assert expected in {r.code for r in result.reasons}
    assert result.specification_retained is False
    assert output["specification_retained"] is False
    assert result.specification is output["specification"] is None
    if case.startswith("oversized"):
        assert result.metadata_bytes > 512 * 1024
    else:
        assert result.metadata_bytes is None


def test_retained_root_detachment_and_immutability(admission):
    root = [{"items": [None, False, 0]}, []]
    original = deepcopy(root)
    result = admission.validate_specification(root)
    output = admission.to_dict(result)
    assert_invalid_root(result, output)
    root[0]["items"][1] = True
    root[0]["new"] = "submitted mutation"
    root.append("submitted mutation")
    output["specification"][0]["items"].append("copy mutation")
    output["specification"][1].append("copy mutation")
    output["reasons"][0]["invalid"] = False
    fresh = admission.to_dict(result)
    assert fresh is not output
    assert fresh["specification"] == original
    assert fresh["specification_retained"] is True
    assert fresh["reasons"][0]["invalid"] is True
    assert type(result.specification) is tuple
    assert type(result.specification[0].entries) is tuple
    with pytest.raises(FrozenInstanceError):
        result.specification_retained = False
    with pytest.raises(FrozenInstanceError):
        result.specification[0].entries = ()
    with pytest.raises(TypeError):
        result.specification[0] = None
