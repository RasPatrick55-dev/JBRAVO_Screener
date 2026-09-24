"""Wrapper argv contract, using only doubles inside an external OS sandbox.

Run a copied module with pytest in an already qualified Linux filesystem/network
namespace, not in a production checkout. The launcher supplies a read-only
JBRAVO_CONTRACT_WRAPPER, the outer JBRAVO_TEST_PARENT_NETNS identity, and a
JBRAVO_TEST_EVIDENCE path inside the sandbox. No application module is imported.
Missing isolation/Bash is a failure, never an essential-test skip. The sandbox
must expose only system runtime, reviewed inputs and disposable output paths.
The root conftest imports Alpaca before its optional-credential fixture; this
standalone copy deliberately collects without that operational dependency.
"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

import pytest


pytestmark = pytest.mark.alpaca_optional

TRADING_ARGS = [
    "--source", "db", "--price-source", "blended", "--ref-buffer-pct", "0.75",
    "--time-window", "auto", "--extended-hours", "true", "--alloc-weight-key",
    "score", "--allocation-pct", "0.06", "--min-order-usd", "300",
    "--max-positions", "4", "--trailing-percent", "3", "--cancel-after-min", "30",
]
WSGI_PATH = "/var/www/raspatrick_pythonanywhere_com_wsgi.py"

# Exact LF heredoc bytes in baseline blob f961ec28b48c0ae0afde276eb911739a7cf7bce7.
# These programs are recognized but NEVER evaluated, executed or imported.
INLINE_PROGRAMS = {
    "90fb9dcf693e37b107508653e1854f52e7e0dd67d82526deee56eb497b554b6d": "gate",
    "70a92061b2bb51ca7e2ee4d05c8bfb01b3fae53eb52559acb9e2a59dfbb5f074": "account",
    "ef920ed8556165b847f7cff4646f78712e1348d62344ec4bdd58b2ca8082b227": "timestamp",
    "fd564213365effd929a9d2ffe5536c73a20c7a87fdeed17673167e5e2e55a837": "snapshot",
}

DOUBLE = r'''#!/usr/bin/python3 -S
import hashlib, json, os, pathlib, sys
tool = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
event = {"tool": tool, "args": args}
def record():
    with open(os.environ["FAKE_TRACE"], "a", encoding="utf-8") as out:
        out.write(json.dumps(event) + "\n")
def deny(reason):
    event["violation"] = reason
    record()
    print("UNEXPECTED_BOUNDARY: " + reason, file=sys.stderr)
    raise SystemExit(97)
if tool == "python":
    if args == ["-"]:
        digest = hashlib.sha256(sys.stdin.buffer.read()).hexdigest()
        event["stdin_sha256"] = digest
        stage = json.loads(os.environ["FAKE_INLINE_PROGRAMS"]).get(digest)
        if stage is None:
            deny("unknown inline program")
    elif args == ["-m", "scripts.check_connection"]:
        stage = "connection"
    elif args[:2] == ["-m", "scripts.execute_trades"]:
        stage = "executor"
    else:
        deny("unknown Python invocation")
    event["stage"] = stage
    record()
    if stage == "gate":
        print("PIPELINE_GATE allow=" + os.environ.get("FAKE_GATE", "1"))
    elif stage == "account":
        print('{"status":"OK","buying_power":"FAKE","auth_ok":true}')
    elif stage == "timestamp":
        print("2000-01-01T00:00:00+00:00")
    raise SystemExit(int(os.environ.get("FAKE_" + stage.upper() + "_RC", "0")))
elif tool == "tee":
    expected = str(pathlib.Path(os.environ["PROJECT_HOME"]) / "logs/execute_trades.log")
    if args != ["-a", expected]:
        deny("unexpected tee destination")
    record()
    data = sys.stdin.read()
    with open(expected, "a", encoding="utf-8") as out:
        out.write(data)
    sys.stdout.write(data)
elif tool == "touch":
    if args != ["/var/www/raspatrick_pythonanywhere_com_wsgi.py"]:
        deny("unexpected touch destination")
    event["effect"] = "recorded only; no file opened or touched"
    record()
else:
    deny("unknown executable")
'''


@pytest.fixture(scope="module", autouse=True)
def require_isolation():
    assert sys.platform == "linux", "OS filesystem/network isolation required"
    parent = os.environ.get("JBRAVO_TEST_PARENT_NETNS")
    assert parent and os.readlink("/proc/self/ns/net") != parent, "private network required"
    assert {name for _, name in socket.if_nameindex()} <= {"lo"}
    assert len(Path("/proc/net/route").read_text().splitlines()) == 1
    for path in ("/mnt/c", "/home/RasPatrick", "/var/www", "/root/.config", "/etc/resolv.conf"):
        assert not Path(path).exists(), "Production paths/DNS must not be mounted"
    mounts = Path("/proc/mounts").read_text().splitlines()
    assert any(line.split()[1:3] == ["/", "tmpfs"] for line in mounts)
    for target in ("/usr", "/input"):
        mount = next((line.split() for line in mounts if line.split()[1] == target), None)
        assert mount and "ro" in mount[3].split(","), "runtime and inputs must be read-only"
    assert Path("/usr/bin/bash").is_file(), "Real Bash is required, not an essential-test skip"
    assert Path(os.environ["JBRAVO_CONTRACT_WRAPPER"]).is_relative_to("/input")


class WrapperHarness:
    def __init__(self, root):
        self.root = root
        self.project = root / "project with spaces"
        self.home = root / "synthetic home"
        self.venv = root / "synthetic venv"
        self.bin = root / "doubles"
        for folder in (self.project / "logs", self.home / ".config/jbravo", self.venv / "bin", self.bin):
            folder.mkdir(parents=True)
        self.wrapper = self.project / "candidate.sh"
        source = Path(os.environ["JBRAVO_CONTRACT_WRAPPER"]).read_bytes()
        self.wrapper.write_bytes(source)  # exact saved bytes: no flag insertion or rewriting
        self.wrapper_sha256 = hashlib.sha256(source).hexdigest()
        self.trace = root / "events.jsonl"
        (self.venv / "bin/activate").write_text("# Synthetic no-op activation; no PATH changes.\n")
        for tool in ("python", "tee", "touch"):
            path = self.bin / tool
            path.write_text(DOUBLE)
            path.chmod(0o700)

    def run(self, args=(), *, setting=None, config_setting=None, overrides=None, omit_key=False):
        self.trace.unlink(missing_ok=True)
        config = ["APCA_API_SECRET_KEY=FAKE_NOT_A_SECRET", "APCA_API_BASE_URL=https://fake.invalid"]
        if not omit_key:
            config.append("APCA_API_KEY_ID=FAKE_NOT_A_KEY")
        if config_setting is not None:
            config.append("JBRAVO_DRY_RUN=" + shlex.quote(config_setting))
        (self.home / ".config/jbravo/.env").write_text("\n".join(config) + "\n")
        env = {
            "HOME": str(self.home), "PROJECT_HOME": str(self.project), "VENV": str(self.venv),
            "PATH": str(self.bin), "LC_ALL": "C", "FAKE_TRACE": str(self.trace),
            "FAKE_INLINE_PROGRAMS": json.dumps(INLINE_PROGRAMS),
        }
        if setting is not None:
            env["JBRAVO_DRY_RUN"] = setting
        env.update(overrides or {})
        result = subprocess.run(
            ["/usr/bin/bash", "--noprofile", "--norc", str(self.wrapper), *args],
            cwd=self.project, env=env, text=True, capture_output=True, timeout=15,
        )
        events = [json.loads(line) for line in self.trace.read_text().splitlines()] if self.trace.exists() else []
        evidence = {
            "test": os.environ.get("PYTEST_CURRENT_TEST"), "wrapper_sha256": self.wrapper_sha256,
            "requested_args": list(args), "setting": setting, "config_setting": config_setting,
            "controls": overrides or {}, "omit_key": omit_key, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr, "events": events,
        }
        with open(os.environ["JBRAVO_TEST_EVIDENCE"], "a", encoding="utf-8") as out:
            out.write(json.dumps(evidence) + "\n")
        assert not any("violation" in event for event in events), evidence
        assert "command not found" not in result.stderr, evidence
        assert not Path(WSGI_PATH).exists(), "The real WSGI destination must remain absent"
        return result, events


@pytest.fixture
def harness(tmp_path):
    return WrapperHarness(tmp_path)


def executor_args(events):
    calls = [event["args"][2:] for event in events if event.get("stage") == "executor"]
    assert len(calls) == 1, events
    return calls[0]


def assert_executor_arguments(events, expected):
    received = executor_args(events)
    assert received.count("--dry-run") == 1, received
    index = received.index("--dry-run")
    assert received[index:index + 2] == ["--dry-run", expected]
    assert received[:index] + received[index + 2:] == TRADING_ARGS


def assert_mode(harness, expected, args=(), **options):
    result, events = harness.run(args, **options)
    assert result.returncode == 0, result.stderr
    assert_executor_arguments(events, expected)
    assert [event.get("stage", event["tool"]) for event in events] == [
        "gate", "tee", "account", "timestamp", "connection", "executor", "timestamp", "snapshot", "touch",
    ]


def test_default_forwards_false(harness):
    assert_mode(harness, "false")


def test_environment_true_forwarded(harness):
    assert_mode(harness, "true", setting="true")


@pytest.mark.parametrize("value,expected", [
    ("true", "true"), ("TRUE", "true"), ("TrUe", "true"), ("1", "true"),
    ("yes", "true"), ("Y", "true"), ("false", "false"), ("FALSE", "false"),
    ("FaLsE", "false"), ("0", "false"), ("no", "false"), ("N", "false"),
])
def test_supported_spellings(harness, value, expected):
    assert_mode(harness, expected, setting=value)
    assert_mode(harness, expected, ["--dry-run", value], setting="invalid-but-overridden")


@pytest.mark.parametrize("setting", ["true", "false", "invalid"])
@pytest.mark.parametrize("choice", ["true", "false"])
def test_cli_overrides_environment(harness, setting, choice):
    assert_mode(harness, choice, ["--dry-run", choice], setting=setting)


@pytest.mark.parametrize("setting", [None, "", "false", "invalid"])
def test_bare_dry_run(harness, setting):
    assert_mode(harness, "true", ["--dry-run"], setting=setting)


def test_empty_environment_retains_default(harness):
    assert_mode(harness, "false", setting="")


def test_configuration_precedence_is_preserved(harness):
    assert_mode(harness, "false", setting="true", config_setting="false")
    assert_mode(harness, "true", ["--dry-run", "true"], setting="true", config_setting="false")


@pytest.mark.parametrize("args,expected", [
    (["--dry-run", "true", "--dry-run", "false"], "false"),
    (["--dry-run", "false", "--dry-run", "true"], "true"),
    (["--dry-run", "false", "--dry-run"], "true"),
    (["--dry-run", "--dry-run", "false"], "false"),
    (["--dry-run", "--dry-run"], "true"),
])
def test_repeated_valid_options_last_wins(harness, args, expected):
    assert_mode(harness, expected, args)


@pytest.mark.parametrize("args", [
    ["--unknown"], ["--dry-run=true"], ["--dry-run", ""], ["--dry-run", "on"],
    ["--dry-run", "off"], ["--dry-run", " true "], ["--dry-run", "garbage"],
    ["--dry-run", "--allocation-pct", "1"], ["--dry-run", "--unknown"],
    ["--dry-run", "true", "extra"], ["--dry-run", "false", "--"],
    ["--dry-run", "bad", "--dry-run", "true"],
    ["--dry-run", "true; touch /var/www/raspatrick_pythonanywhere_com_wsgi.py"],
    ["--dry-run", "$(touch /var/www/raspatrick_pythonanywhere_com_wsgi.py)"],
    ["--dry-run", "true\n--allocation-pct 1"],
])
def test_invalid_arguments_never_dispatch(harness, args):
    result, events = harness.run(args)
    assert result.returncode != 0
    assert not events  # after synthetic config loading, before any Python/executable boundary


@pytest.mark.parametrize("setting", ["on", "off", "bad", " true ", "--dry-run", "$(touch /tmp/INJECTED)"])
def test_invalid_effective_environment_never_dispatches(harness, setting):
    result, events = harness.run(setting=setting)
    assert result.returncode != 0
    assert not events
    assert not Path("/tmp/INJECTED").exists()


def test_denied_pipeline_gate_stops_before_executor(harness):
    result, events = harness.run(["--dry-run"], overrides={"FAKE_GATE": "0"})
    assert result.returncode == 0
    assert [event.get("stage", event["tool"]) for event in events] == ["gate", "tee"]


@pytest.mark.parametrize("stage,code", [("GATE", "4"), ("ACCOUNT", "2")])
def test_failed_prerequisite_prevents_execution(harness, stage, code):
    result, events = harness.run(overrides={"FAKE_" + stage + "_RC": code})
    assert result.returncode == int(code)
    assert not any(event.get("stage") in {"executor", "snapshot"} or event["tool"] == "touch" for event in events)


def test_missing_required_configuration_prevents_execution(harness):
    result, events = harness.run(omit_key=True)
    assert result.returncode != 0
    assert not events


def test_connection_failure_remains_nonfatal(harness):
    assert_mode(harness, "true", ["--dry-run"], overrides={"FAKE_CONNECTION_RC": "3"})


@pytest.mark.parametrize("code", [1, 2, 7, 125])
def test_executor_failure_propagates_without_snapshot_or_reload(harness, code):
    result, events = harness.run(["--dry-run", "false"], overrides={"FAKE_EXECUTOR_RC": str(code)})
    assert result.returncode == code
    assert_executor_arguments(events, "false")
    assert not any(event.get("stage") == "snapshot" or event["tool"] == "touch" for event in events)


@pytest.mark.parametrize("tool,args,stdin", [
    ("python", ["-"], "import scripts.execute_trades\n"),
    ("python", ["-m", "scripts.run_pipeline"], ""),
    ("touch", ["/tmp/unexpected"], ""),
    ("tee", ["-a", "/tmp/unexpected"], ""),
])
def test_doubles_reject_unapproved_boundaries(harness, tool, args, stdin):
    result = subprocess.run(
        [str(harness.bin / tool), *args], input=stdin, text=True, capture_output=True,
        env={"FAKE_TRACE": str(harness.trace), "FAKE_INLINE_PROGRAMS": json.dumps(INLINE_PROGRAMS),
             "PROJECT_HOME": str(harness.project)}, timeout=10,
    )
    assert result.returncode == 97
    assert "UNEXPECTED_BOUNDARY" in result.stderr
    assert json.loads(harness.trace.read_text())["violation"]
    assert not Path("/tmp/unexpected").exists()
