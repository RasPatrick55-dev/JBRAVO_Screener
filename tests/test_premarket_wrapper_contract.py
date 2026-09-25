"""Discover one host supervisor; all behavioral cases run in the OS sandbox."""

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest


# This exemption belongs on the supervisor as well as the isolated payload.
# It must execute (and visibly fail for missing prerequisites) without credentials.
pytestmark = pytest.mark.alpaca_optional


def test_premarket_wrapper_contract_in_isolation():
    deps = os.environ.get("JBRAVO_CONTRACT_DEPS")
    output_root = os.environ.get("JBRAVO_CONTRACT_OUTPUT_ROOT")
    assert deps, "JBRAVO_CONTRACT_DEPS must name qualified offline pytest tooling"
    assert output_root, "JBRAVO_CONTRACT_OUTPUT_ROOT must name existing disposable evidence storage"
    root = Path(output_root).absolute()
    assert root.is_dir(), "Contract output root must already exist"
    output = root / ("supervisor-" + uuid.uuid4().hex)
    launcher = Path(__file__).absolute().parents[1] / "tools/premarket_contract/run_isolated.py"
    system = os.environ.get("SystemRoot", r"C:\Windows")
    env = ({"SystemRoot": system, "WINDIR": system, "PATH": system + r"\System32"}
           if os.name == "nt" else {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8"})
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(launcher), "candidate",
         "--deps", str(Path(deps).absolute()), "--output", str(output)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"Isolated contract failed ({result.returncode}); evidence: {output}\n{result.stdout}\n{result.stderr}"
    retained = json.loads((output / "launcher-result.json").read_text(encoding="utf-8"))
    assert retained["status"] == "passed" and retained["exit_code"] == 0
    assert retained["counts"] == {
        "collected": 76, "selected": 76, "executed": 76, "passed": 76,
        "failed": 0, "errored": 0, "skipped": 0, "deselected": 0,
    }, f"Incomplete behavioral evidence: {output}"
    for group, count in (("original_behavioral", 65), ("command_attempt_regression", 11)):
        assert retained["case_groups"][group] == {
            "collected": count, "selected": count, "executed": count, "passed": count,
            "failed": 0, "errored": 0, "skipped": 0, "deselected": 0,
        }, f"Incomplete {group} evidence: {output}"
    assert retained["wrapper_invocations"] == 85
    assert retained["wrapper_invocation_groups"] == {
        "original_behavioral": 74, "command_attempt_regression": 11,
    }
