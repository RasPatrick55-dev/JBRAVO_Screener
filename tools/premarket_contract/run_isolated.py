"""Run current-source wrapper cases in the qualified, non-trading OS sandbox."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET


EXPECTED_CASES = 76
SOURCE_INPUTS = {
    "bin/run_premarket_once.sh": "candidate-wrapper.sh",
    "tools/premarket_contract/contract.py": "tests/test_premarket_wrapper_contract.py",
    "tools/premarket_contract/run_in_sandbox.py": "run_in_sandbox.py",
    "tools/premarket_contract/pytest.ini": "pytest.ini",
}
REQUIRED_DEPS = (
    "pytest/__init__.py", "_pytest/__init__.py", "pluggy/__init__.py",
    "packaging/__init__.py", "iniconfig/__init__.py", "pygments/__init__.py", "py.py",
)
# A Linux watchdog owns Bubblewrap and its PID namespace. A Windows client timeout
# alone cannot establish child termination, so the deadline is enforced in Linux.
WATCHDOG = r'''
import json, os, signal, subprocess, sys, time
command, seconds, record = json.loads(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
started = time.monotonic()
p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
timed_out = False
try:
    out, err = p.communicate(timeout=seconds)
except subprocess.TimeoutExpired:
    timed_out = True
    os.killpg(p.pid, signal.SIGKILL)
    out, err = p.communicate()
result = {'timed_out': timed_out, 'child_returncode': p.returncode,
          'exit_code': 124 if timed_out else p.returncode,
          'child_reaped': True, 'elapsed_seconds': time.monotonic() - started}
with open(record, 'x', encoding='utf-8') as f:
    json.dump(result, f, indent=2); f.write('\n')
sys.stdout.buffer.write(out); sys.stderr.buffer.write(err)
raise SystemExit(result['exit_code'])
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def no_links(path):
    """Reject redirected inputs/destinations, including Windows junctions."""
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Symlink path is not a qualified input or output: " + str(part))
        if part.exists() and getattr(part.stat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("Reparse-point path is not permitted: " + str(part))
    return path.resolve()


def clean_host_environment():
    if os.name == "nt":
        system = os.environ.get("SystemRoot", r"C:\Windows")
        return {"SystemRoot": system, "WINDIR": system, "PATH": system + r"\System32"}
    return {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"}


def linux_prefix():
    if os.name == "nt":
        system = os.environ.get("SystemRoot", r"C:\Windows")
        wsl = Path(system) / "System32/wsl.exe"
        if not wsl.is_file():
            raise ValueError("Existing Ubuntu WSL is required; no host fallback")
        return [str(wsl), "-d", "Ubuntu", "--exec", "/usr/bin/env", "-i", "PATH=/usr/bin:/bin"]
    if sys.platform == "linux":
        return ["/usr/bin/env", "-i", "PATH=/usr/bin:/bin"]
    raise ValueError("Only existing Ubuntu WSL on Windows or qualified Linux is supported")


def checked_process(argv, timeout=20):
    completed = subprocess.run(argv, env=clean_host_environment(), capture_output=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError("Prerequisite process failed with exit " + str(completed.returncode))
    return completed.stdout.decode("utf-8").strip()


def linux_path(path, prefix):
    if os.name == "nt":
        return checked_process(prefix + ["/usr/bin/wslpath", "-a", "-u", str(path)])
    return str(path)


def bind_copy(source, target, relative):
    no_links(source)
    if not source.is_file() or not stat.S_ISREG(source.stat().st_mode):
        raise ValueError("Regular input file required: " + str(source))
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
    if target.read_bytes() != data or source.read_bytes() != data:
        raise ValueError("Input changed during capture: " + str(source))
    return {"path": relative, "sha256": digest(data), "bytes": len(data)}, {
        "source": str(source), "isolated_path": "/input/" + relative,
        "sha256": digest(data), "bytes": len(data),
    }


def validate_result(output, process_code, manifest_hash, files):
    inner = json.loads((output / "result.json").read_text(encoding="utf-8"))
    expected = {"collected": 76, "selected": 76, "executed": 76, "passed": 76,
                "failed": 0, "errored": 0, "skipped": 0, "deselected": 0}
    if process_code or inner.get("exit_code") != 0 or inner.get("status") != "passed":
        raise ValueError("Isolated child did not pass; see native logs and result.json")
    if inner.get("counts") != expected:
        raise ValueError("Incomplete or failing case inventory; no skipped/empty-suite success")
    for group, count in (("original_behavioral", 65), ("command_attempt_regression", 11)):
        group_expected = {key: count if key in ("collected", "selected", "executed", "passed") else 0 for key in expected}
        if inner.get("case_groups", {}).get(group) != group_expected:
            raise ValueError("Original/regression case accounting incomplete: " + group)
    if (inner.get("wrapper_invocations") != 85 or inner.get("wrapper_invocation_groups") !=
            {"original_behavioral": 74, "command_attempt_regression": 11}):
        raise ValueError("Original/regression wrapper invocation accounting mismatch")
    if inner.get("input_manifest_sha256") != manifest_hash:
        raise ValueError("Executed input manifest identity mismatch")
    hashes = {item["path"]: item["sha256"] for item in files}
    if (inner.get("wrapper_sha256") != hashes["candidate-wrapper.sh"] or
            inner.get("case_sha256") != hashes["tests/test_premarket_wrapper_contract.py"]):
        raise ValueError("Executed wrapper or case identity mismatch")
    selected, executed = inner.get("selected_nodeids", []), inner.get("executed_nodeids", [])
    if len(set(selected)) != EXPECTED_CASES or set(selected) != set(executed):
        raise ValueError("Selected/executed case identities disagree")
    root = ET.parse(output / "junit.xml").getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if totals != {"tests": 76, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError("JUnit does not corroborate the complete successful child result")
    supervision = json.loads((output / "supervision.json").read_text(encoding="utf-8"))
    if supervision["timed_out"] or not supervision["child_reaped"] or supervision["exit_code"]:
        raise ValueError("Supervision did not confirm normal child completion")
    return inner


def run(args):
    output = None
    result = {"status": "failed", "phase": "prerequisites", "exit_code": 2,
              "wrapper_executed_on_host": False, "sandbox_launch_attempted": False,
              "contract_execution_started": False, "counts_complete": True,
              "counts": dict.fromkeys(("collected", "selected", "executed", "passed",
                                       "failed", "errored", "skipped", "deselected"), 0)}
    try:
        # Source selection follows this checked-out launcher, never the caller's cwd.
        launcher = no_links(Path(__file__).absolute())
        source_root = no_links(launcher.parents[2])
        destination = no_links(Path(args.output).absolute())
        deps = no_links(Path(args.deps).absolute())
        if not 0 < args.timeout_seconds <= 240:
            raise ValueError("timeout-seconds must be greater than zero and at most 240")
        if destination.exists() or not destination.parent.is_dir():
            raise ValueError("Output must be absent with an existing parent; nothing is overwritten")
        for root in (source_root, deps):
            if destination.is_relative_to(root) or root.is_relative_to(destination):
                raise ValueError("Output must be separate from source and offline tooling")
        destination.mkdir()
        output = destination
        save(output / "launch-start.json", {"source_root": str(source_root), "launcher": str(launcher),
             "launcher_sha256": digest(launcher.read_bytes()), "output": str(output),
             "deps_root": str(deps), "timeout_seconds": args.timeout_seconds,
             "mode": args.mode, "runtime_environment": "explicit allowlist; no inherited credentials"})
        for name in REQUIRED_DEPS:
            if not (deps / name).is_file():
                raise ValueError("Required offline test dependency missing: " + name)
        prefix = linux_prefix()
        probe = "import json,os,pathlib,sys; print(json.dumps({'python':sys.version,'parent_netns':os.readlink('/proc/self/ns/net'),'bwrap':pathlib.Path('/usr/bin/bwrap').is_file(),'bash':pathlib.Path('/usr/bin/bash').is_file()}))"
        prerequisites = json.loads(checked_process(prefix + ["/usr/bin/python3", "-I", "-S", "-B", "-c", probe]))
        save(output / "prerequisites.json", prerequisites)
        if not prerequisites["bash"] or not prerequisites["bwrap"]:
            raise ValueError("Existing Bubblewrap and Bash are required; no installation or fallback")
        inputs = output / "inputs"
        inputs.mkdir()
        files, selections = [], []
        for relative, target in SOURCE_INPUTS.items():
            entry, selected = bind_copy(source_root / relative, inputs / target, target)
            files.append(entry)
            selections.append(selected)
        dependency_files = []
        for path in sorted(deps.rglob("*")):
            no_links(path)
            if path.is_dir():
                continue
            if path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            if not path.is_file():
                raise ValueError("Special dependency inputs are not supported")
            relative = "deps/" + path.relative_to(deps).as_posix()
            entry, selected = bind_copy(path, inputs / relative, relative)
            files.append(entry)
            dependency_files.append(selected)
        save(output / "source-selection.json", {"source_root": str(source_root),
             "sources": selections, "offline_dependencies": dependency_files})
        save(inputs / "input-manifest.json", {"files": files})
        manifest_hash = digest((inputs / "input-manifest.json").read_bytes())
        linux_output = linux_path(output, prefix)
        command = ["/usr/bin/bwrap", "--unshare-all", "--new-session", "--die-with-parent",
                   "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
                   "--ro-bind", "/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
                   "--symlink", "usr/sbin", "/sbin", "--proc", "/proc", "--dev", "/dev",
                   "--tmpfs", "/tmp", "--dir", "/home", "--ro-bind", linux_output + "/inputs", "/input",
                   "--bind", linux_output, "/output", "--chdir", "/tmp", "--clearenv",
                   "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp/synthetic-home",
                   "--setenv", "LC_ALL", "C.UTF-8", "--setenv", "JBRAVO_TEST_PARENT_NETNS", prerequisites["parent_netns"],
                   "/usr/bin/python3", "-I", "-S", "-B", "/input/run_in_sandbox.py", "candidate"]
        # Do not expose a second writable alias for the otherwise read-only inputs.
        command[command.index("--chdir"):command.index("--chdir")] = ["--ro-bind", linux_output + "/inputs", "/output/inputs"]
        save(output / "launch.json", {"command": command, "input_manifest_sha256": manifest_hash})
        result.update(phase="isolated_child", sandbox_launch_attempted=True,
                      contract_execution_started=None, counts_complete=False, counts=None)
        completed = subprocess.run(prefix + ["/usr/bin/python3", "-I", "-S", "-B", "-c", WATCHDOG,
                                   json.dumps(command), str(args.timeout_seconds), linux_output + "/supervision.json"],
                                   env=clean_host_environment(), capture_output=True, timeout=args.timeout_seconds + 30)
        (output / "stdout.log").write_bytes(completed.stdout)
        (output / "stderr.log").write_bytes(completed.stderr)
        result["native_exit_code"] = completed.returncode
        if (output / "result.json").is_file():
            observed = json.loads((output / "result.json").read_text(encoding="utf-8"))
            result.update(counts=observed.get("counts"), child_phase=observed.get("phase"),
                          wrapper_invocations=observed.get("wrapper_invocations", 0),
                          case_groups=observed.get("case_groups"),
                          wrapper_invocation_groups=observed.get("wrapper_invocation_groups"))
            if observed.get("counts") is not None:
                result["contract_execution_started"] = observed["counts"].get("executed", 0) > 0
        if completed.returncode == 124:
            result.update(phase="timeout", exit_code=124)
            raise ValueError("Isolated child exceeded deadline; Linux watchdog terminated and reaped it")
        # Input snapshots must still agree after execution, including both mount aliases.
        for item in files:
            if digest((inputs / item["path"]).read_bytes()) != item["sha256"]:
                raise ValueError("Input snapshot changed during execution")
        for selected in selections:
            if digest(Path(selected["source"]).read_bytes()) != selected["sha256"]:
                raise ValueError("Current source changed during execution")
        inner = validate_result(output, completed.returncode, manifest_hash, files)
        result.update(status="passed", phase="complete", exit_code=0, counts=inner["counts"], counts_complete=True,
                      input_manifest_sha256=manifest_hash, wrapper_sha256=inner["wrapper_sha256"],
                      case_sha256=inner["case_sha256"], wrapper_invocations=inner["wrapper_invocations"],
                      case_groups=inner["case_groups"], wrapper_invocation_groups=inner["wrapper_invocation_groups"])
    except (OSError, ValueError, RuntimeError, KeyError, ET.ParseError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
        if isinstance(exc, subprocess.TimeoutExpired):
            result.update(phase="launcher_timeout", exit_code=124,
                          child_completion="Not established; inspect retained supervision evidence; no retry")
        elif result["phase"] == "isolated_child":
            result["exit_code"] = 1
    if output is not None:
        save(output / "launcher-result.json", result)
    print(json.dumps({"output": str(output) if output else None, **result}, indent=2))
    return result["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["candidate"])
    parser.add_argument("--output", required=True, help="Absent directory with an existing parent, outside the checkout/tooling")
    parser.add_argument("--deps", required=True, help="Existing qualified offline pytest dependency directory")
    parser.add_argument("--timeout-seconds", type=float, default=240)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
