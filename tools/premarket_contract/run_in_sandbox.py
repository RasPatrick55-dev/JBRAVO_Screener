"""Qualified synthetic runner, adapted from the retained frozen-input harness.

Only the outer launcher may invoke this file. No trading application is imported.
The payload keeps its historical sandbox filename so the original 65 node IDs stay stable.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys


INPUT = Path("/input")
OUTPUT = Path("/output")
WRAPPER = INPUT / "candidate-wrapper.sh"
CASES = INPUT / "tests/test_premarket_wrapper_contract.py"
CASE_PREFIX = "tests/test_premarket_wrapper_contract.py::"
# Inventory from the accepted prepared-validation/candidate-001 JUnit evidence.
# These are test cases, not the 74 wrapper invocations recorded by the payload.
EXPECTED_NAMES = r"""test_default_forwards_false
test_environment_true_forwarded
test_supported_spellings[true-true]
test_supported_spellings[TRUE-true]
test_supported_spellings[TrUe-true]
test_supported_spellings[1-true]
test_supported_spellings[yes-true]
test_supported_spellings[Y-true]
test_supported_spellings[false-false]
test_supported_spellings[FALSE-false]
test_supported_spellings[FaLsE-false]
test_supported_spellings[0-false]
test_supported_spellings[no-false]
test_supported_spellings[N-false]
test_cli_overrides_environment[true-true]
test_cli_overrides_environment[true-false]
test_cli_overrides_environment[true-invalid]
test_cli_overrides_environment[false-true]
test_cli_overrides_environment[false-false]
test_cli_overrides_environment[false-invalid]
test_bare_dry_run[None]
test_bare_dry_run[]
test_bare_dry_run[false]
test_bare_dry_run[invalid]
test_empty_environment_retains_default
test_configuration_precedence_is_preserved
test_repeated_valid_options_last_wins[args0-false]
test_repeated_valid_options_last_wins[args1-true]
test_repeated_valid_options_last_wins[args2-true]
test_repeated_valid_options_last_wins[args3-false]
test_repeated_valid_options_last_wins[args4-true]
test_invalid_arguments_never_dispatch[args0]
test_invalid_arguments_never_dispatch[args1]
test_invalid_arguments_never_dispatch[args2]
test_invalid_arguments_never_dispatch[args3]
test_invalid_arguments_never_dispatch[args4]
test_invalid_arguments_never_dispatch[args5]
test_invalid_arguments_never_dispatch[args6]
test_invalid_arguments_never_dispatch[args7]
test_invalid_arguments_never_dispatch[args8]
test_invalid_arguments_never_dispatch[args9]
test_invalid_arguments_never_dispatch[args10]
test_invalid_arguments_never_dispatch[args11]
test_invalid_arguments_never_dispatch[args12]
test_invalid_arguments_never_dispatch[args13]
test_invalid_arguments_never_dispatch[args14]
test_invalid_effective_environment_never_dispatches[on]
test_invalid_effective_environment_never_dispatches[off]
test_invalid_effective_environment_never_dispatches[bad]
test_invalid_effective_environment_never_dispatches[ true ]
test_invalid_effective_environment_never_dispatches[--dry-run]
test_invalid_effective_environment_never_dispatches[$(touch /tmp/INJECTED)]
test_denied_pipeline_gate_stops_before_executor
test_failed_prerequisite_prevents_execution[GATE-4]
test_failed_prerequisite_prevents_execution[ACCOUNT-2]
test_missing_required_configuration_prevents_execution
test_connection_failure_remains_nonfatal
test_executor_failure_propagates_without_snapshot_or_reload[1]
test_executor_failure_propagates_without_snapshot_or_reload[2]
test_executor_failure_propagates_without_snapshot_or_reload[7]
test_executor_failure_propagates_without_snapshot_or_reload[125]
test_doubles_reject_unapproved_boundaries[python-args0-import scripts.execute_trades\n]
test_doubles_reject_unapproved_boundaries[python-args1-]
test_doubles_reject_unapproved_boundaries[touch-args2-]
test_doubles_reject_unapproved_boundaries[tee-args3-]""".splitlines()
ORIGINAL_NODEIDS = {CASE_PREFIX + name for name in EXPECTED_NAMES}
REGRESSION_NAMES = [
    "test_command_attempts_rejected[review-suppressed-unknown]",
    "test_command_attempts_rejected[review-absolute-true]",
    "test_command_attempts_rejected[review-suppressed-double]",
    "test_command_attempts_rejected[subshell-unknown]",
    "test_command_attempts_rejected[substitution-absolute]",
    "test_command_attempts_rejected[function-unknown]",
    "test_command_attempts_rejected[redirection-substitution]",
    "test_observation_failure_rejected[absent]",
    "test_observation_failure_rejected[malformed]",
    "test_observation_failure_rejected[incomplete]",
    "test_observation_failure_rejected[write-failure]",
]
REGRESSION_NODEIDS = {CASE_PREFIX + name for name in REGRESSION_NAMES}
EXPECTED_NODEIDS = ORIGINAL_NODEIDS | REGRESSION_NODEIDS


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    temporary = OUTPUT / (name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT / name)


def validate_manifest(result):
    manifest_path = INPUT / "input-manifest.json"
    require(not manifest_path.is_symlink(), "Input manifest must not be a symlink")
    result["input_manifest_sha256"] = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict) and isinstance(manifest.get("files"), list),
            "Input manifest must contain a files list")
    expected = {}
    for entry in manifest["files"]:
        require(isinstance(entry, dict), "Invalid input manifest entry")
        relative = entry.get("path")
        require(isinstance(relative, str) and relative and "\\" not in relative,
                "Manifest path must be relative POSIX text")
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts
                and path.as_posix() == relative and relative != "input-manifest.json",
                "Unsafe or noncanonical manifest path")
        require(relative not in expected, "Duplicate manifest path")
        require(isinstance(entry.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                and type(entry.get("bytes")) is int and entry["bytes"] >= 0,
                "Invalid manifest identity")
        expected[relative] = entry
    actual = set()
    for path in INPUT.rglob("*"):
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), "Input symlinks are forbidden")
        require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "Special input files are forbidden")
        if stat.S_ISREG(mode) and path != manifest_path:
            actual.add(path.relative_to(INPUT).as_posix())
    require(actual == set(expected), "Input manifest does not match the complete staged file set")
    required = {"candidate-wrapper.sh", "tests/test_premarket_wrapper_contract.py",
                "run_in_sandbox.py", "pytest.ini"}
    require(required <= actual and any(name.startswith("deps/") for name in actual),
            "Required wrapper, cases, runner, configuration or offline dependencies missing")
    for relative, entry in expected.items():
        path = INPUT / relative
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
                "Staged input identity mismatch: " + relative)
    result["wrapper_sha256"] = expected["candidate-wrapper.sh"]["sha256"]
    result["case_sha256"] = expected["tests/test_premarket_wrapper_contract.py"]["sha256"]
    result["runner_sha256"] = expected["run_in_sandbox.py"]["sha256"]
    result["pytest_ini_sha256"] = expected["pytest.ini"]["sha256"]


def qualify():
    require(sys.platform == "linux", "Linux filesystem/network isolation is required")
    parent = os.environ.get("JBRAVO_TEST_PARENT_NETNS", "")
    forbidden = ["/mnt/c", "/home/RasPatrick", "/var/www", "/root/.config", "/etc/resolv.conf"]
    qualification = {
        "python": sys.version,
        "parent_network_namespace": parent,
        "test_network_namespace": os.readlink("/proc/self/ns/net"),
        "interfaces": socket.if_nameindex(),
        "routes": Path("/proc/net/route").read_text(),
        "mounts": Path("/proc/mounts").read_text(),
        "forbidden_paths": {path: Path(path).exists() for path in forbidden},
        "home": os.environ.get("HOME"),
        "configuration": "Explicit standalone pytest.ini/root/confcutdir; importlib; no plugin autoload.",
        "qualified": False,
    }
    write_json("isolation.json", qualification)
    require(re.fullmatch(r"net:\[\d+\]", parent)
            and parent != qualification["test_network_namespace"], "Measured private networking is required")
    require(not any(qualification["forbidden_paths"].values()), "Production paths or DNS are exposed")
    require({name for _, name in qualification["interfaces"]} <= {"lo"}, "External network interface exposed")
    require(len(qualification["routes"].splitlines()) == 1, "External network route exposed")
    mounts = [line.split() for line in qualification["mounts"].splitlines()]
    require(any(parts[1:3] == ["/", "tmpfs"] for parts in mounts), "Root must be synthetic tmpfs")
    for target in ("/usr", "/input"):
        mount = next((parts for parts in mounts if parts[1] == target), None)
        require(mount and "ro" in mount[3].split(","), "Runtime and inputs must be read-only")
    require(any(parts[1:3] == ["/tmp", "tmpfs"] for parts in mounts), "Temporary storage must be tmpfs")
    require(qualification["home"] == "/tmp/synthetic-home", "Synthetic HOME is required")
    require(not any(name.startswith(("APCA_", "ALPACA_", "OPENAI_")) for name in os.environ),
            "Inherited production configuration is forbidden")
    require(Path("/usr/bin/bash").is_file(), "Real Bash is required")
    bash = subprocess.run(["/usr/bin/bash", "--version"], env={}, capture_output=True,
                          text=True, check=True, timeout=10)
    qualification["bash"] = bash.stdout
    qualification["qualified"] = True
    write_json("isolation.json", qualification)
    return qualification


class Results:
    """Observe pytest reports; never infer test results from wrapper invocations."""

    def __init__(self, pytest_module, result):
        self.pytest = pytest_module
        self.result = result
        self.selected = []
        self.deselected = []
        self.reports = {}
        self.collection_errors = []

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors.append({"nodeid": report.nodeid, "error": str(report.longrepr)})

    def pytest_collection_finish(self, session):
        self.selected = [item.nodeid for item in session.items]
        if (len(self.selected) != len(EXPECTED_NODEIDS) or set(self.selected) != EXPECTED_NODEIDS
                or self.deselected or self.collection_errors):
            raise self.pytest.UsageError("Exactly the expected 76 contract cases must be selected, without deselection or collection errors")

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(report.nodeid, {})[report.when] = {
            "outcome": report.outcome, "xfail": hasattr(report, "wasxfail"),
        }
        self.update(self.result)
        self.result["wrapper_invocations"] = len((OUTPUT / "cases.jsonl").read_text().splitlines())
        self.result["counts_observation"] = "Latest completed pytest phase report; final only at completion"
        write_json("result.json", self.result)

    def counts_for(self, nodeids, collection_errors=0):
        counts = dict.fromkeys(("collected", "selected", "executed", "passed", "failed",
                                "errored", "skipped", "deselected"), 0)
        counts["errored"] = collection_errors
        counts["collected"] = sum(node in nodeids for node in self.selected + self.deselected)
        counts["selected"] = sum(node in nodeids for node in self.selected)
        counts["deselected"] = sum(node in nodeids for node in self.deselected)
        executed = []
        for nodeid, phases in self.reports.items():
            if nodeid not in nodeids:
                continue
            if "call" in phases:
                executed.append(nodeid)
            if any(phases.get(when, {}).get("outcome") == "failed" for when in ("setup", "teardown")):
                counts["errored"] += 1
            elif phases.get("call", {}).get("outcome") == "failed":
                counts["failed"] += 1
            elif any(phase["outcome"] == "skipped" or phase["xfail"] for phase in phases.values()):
                counts["skipped"] += 1
            elif all(phases.get(when, {}).get("outcome") == "passed" for when in ("setup", "call", "teardown")):
                counts["passed"] += 1
        counts["executed"] = len(executed)
        return counts, executed

    def update(self, result):
        observed = set(self.selected + self.deselected) | set(self.reports)
        counts, executed = self.counts_for(observed, len(self.collection_errors))
        result.update(counts=counts, selected_nodeids=self.selected, executed_nodeids=executed,
                      deselected_nodeids=self.deselected, collection_errors=self.collection_errors,
                      case_groups={
                          "original_behavioral": self.counts_for(ORIGINAL_NODEIDS)[0],
                          "command_attempt_regression": self.counts_for(REGRESSION_NODEIDS)[0],
                      })


def main():
    result = {
        "status": "failed", "phase": "qualification", "exit_code": 2,
        "counts": dict.fromkeys(("collected", "selected", "executed", "passed", "failed",
                                 "errored", "skipped", "deselected"), 0),
        "selected_nodeids": [], "executed_nodeids": [],
        "input_manifest_sha256": None, "wrapper_sha256": None, "case_sha256": None,
        "wrapper_invocations": 0,
    }
    plugin = None
    try:
        write_json("result.json", result)
        require(sys.argv[1:] == ["candidate"], "Only full current-source candidate execution is supported")
        qualification = qualify()
        result["phase"] = "input-validation"
        validate_manifest(result)
        sys.dont_write_bytecode = True
        os.environ.update(JBRAVO_CONTRACT_WRAPPER=str(WRAPPER),
                          JBRAVO_TEST_EVIDENCE=str(OUTPUT / "cases.jsonl"),
                          PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
        os.environ.pop("PYTEST_ADDOPTS", None)
        os.environ.pop("PYTEST_PLUGINS", None)
        Path("/tmp/synthetic-home").mkdir(exist_ok=True)
        Path("/tmp/work").mkdir(exist_ok=True)
        os.chdir("/tmp/work")
        (OUTPUT / "cases.jsonl").touch(exist_ok=False)
        result["phase"] = "syntax"
        command = ["/usr/bin/bash", "--noprofile", "--norc", "-n", str(WRAPPER)]
        syntax = subprocess.run(command, env={}, capture_output=True, text=True, timeout=10)
        write_json("syntax.json", {"command": command, "exit_code": syntax.returncode,
                                   "stdout": syntax.stdout, "stderr": syntax.stderr})
        require(syntax.returncode == 0, "Wrapper Bash syntax check failed")
        result["phase"] = "pytest-import"
        sys.path.insert(0, str(INPUT / "deps"))
        import pytest
        require(Path(pytest.__file__).resolve().is_relative_to(INPUT / "deps"),
                "Pytest must come from the verified offline dependency input")
        qualification["pytest"] = pytest.__version__
        qualification["wrapper_sha256"] = result["wrapper_sha256"]
        qualification["test_module_sha256"] = result["case_sha256"]
        qualification["input_manifest_sha256"] = result["input_manifest_sha256"]
        write_json("isolation.json", qualification)
        args = ["-c", "/input/pytest.ini", "--confcutdir=/input/tests", "--rootdir=/input",
                "--import-mode=importlib", "-p", "no:cacheprovider", "-vv", "-ra",
                "--basetemp=/tmp/pytest", "--junitxml=/output/junit.xml", str(CASES)]
        write_json("command.json", {"mode": "candidate", "pytest_args": args})
        plugin = Results(pytest, result)
        result["phase"] = "pytest"
        write_json("result.json", result)
        result["pytest_exit_code"] = int(pytest.main(args, plugins=[plugin]))
        plugin.update(result)
        result["phase"] = "result-validation"
        invocations = [json.loads(line) for line in (OUTPUT / "cases.jsonl").read_text().splitlines()]
        result["wrapper_invocations"] = len(invocations)
        require(all(case.get("source_wrapper_sha256") == result["wrapper_sha256"] for case in invocations),
                "Wrapper invocation source identities do not match verified input")
        for case in invocations:
            nodeid = case.get("test", "").removesuffix(" (call)")
            observation = case.get("observation", {})
            require(observation.get("observer_protocol") == "bash-debug-v1",
                    "Required command observation protocol missing")
            files = observation.get("files", [])
            require(files, "Required command observation artifacts missing")
            for item in files:
                path = Path(item["path"])
                require(path.is_relative_to(OUTPUT / "command-observations")
                        and not path.is_symlink() and path.is_file()
                        and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"],
                        "Retained command observation artifact mismatch")
            executed_wrapper = Path(observation["directory"]) / "executed-wrapper.sh"
            require(executed_wrapper.is_relative_to(OUTPUT / "command-observations")
                    and digest(executed_wrapper) == case["wrapper_sha256"],
                    "Executed wrapper artifact identity mismatch")
            variant = case.get("variant")
            if variant is None:
                require(case["wrapper_sha256"] == result["wrapper_sha256"],
                        "Unmodified wrapper identity mismatch")
            else:
                require(nodeid in REGRESSION_NODEIDS and variant["source_sha256"] == result["wrapper_sha256"]
                        and variant["sha256"] == case["wrapper_sha256"],
                        "Synthetic wrapper variant provenance mismatch")
                anchor, insertion = variant["anchor"].encode(), variant["insertion"].encode()
                original = WRAPPER.read_bytes()
                require(original.count(anchor) == 1
                        and original.replace(anchor, anchor + insertion, 1) == executed_wrapper.read_bytes(),
                        "Synthetic wrapper variant does not match its declared source modification")
            if nodeid in ORIGINAL_NODEIDS:
                require(variant is None and case.get("validation_status") == "accepted"
                        and observation.get("status") == "passed" and observation.get("events")
                        and not observation.get("violations") and not observation.get("error"),
                        "Original behavior has missing or unsuccessful command observation")
            else:
                require(nodeid in REGRESSION_NODEIDS and case.get("validation_status") == "rejected"
                        and case.get("rejection_reasons"),
                        "Expected command-attempt regression did not retain rejection evidence")
        require(all(case.get("test", "").removesuffix(" (call)") in EXPECTED_NODEIDS for case in invocations),
                "Wrapper evidence names an unexpected test")
        require(result["pytest_exit_code"] == 0, "Inner pytest returned a nonzero status")
        require(result["counts"] == {"collected": 76, "selected": 76, "executed": 76,
                "passed": 76, "failed": 0, "errored": 0, "skipped": 0, "deselected": 0},
                "Contract case execution is incomplete or unsuccessful")
        require(not plugin.collection_errors and set(result["executed_nodeids"]) == EXPECTED_NODEIDS,
                "Executed inventory does not match the expected 76 cases")
        require(len(invocations) == 85, "Expected 85 separately recorded wrapper invocations")
        result["wrapper_invocation_groups"] = {
            "original_behavioral": sum(case["test"].removesuffix(" (call)") in ORIGINAL_NODEIDS for case in invocations),
            "command_attempt_regression": sum(case["test"].removesuffix(" (call)") in REGRESSION_NODEIDS for case in invocations),
        }
        require(result["wrapper_invocation_groups"] == {"original_behavioral": 74, "command_attempt_regression": 11},
                "Original and regression wrapper invocation counts disagree")
        require((OUTPUT / "junit.xml").is_file(), "JUnit evidence missing")
        result.update(status="passed", phase="complete", exit_code=0)
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        native_code = result.get("pytest_exit_code", 0)
        result["exit_code"] = native_code if native_code else 2
        if plugin is not None:
            plugin.update(result)
        evidence = OUTPUT / "cases.jsonl"
        if evidence.is_file():
            result["wrapper_invocations"] = len(evidence.read_text().splitlines())
    finally:
        write_json("result.json", result)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
