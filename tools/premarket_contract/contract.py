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
import re
import shlex
import socket
import subprocess
import sys
import uuid

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


# The observer runs before the saved wrapper through BASH_ENV. It observes Bash
# command attempts; it does not prevent an executable from running in Bubblewrap.
# DEBUG is before expansion/lookup/redirection; functrace supplies the inner
# attempts in functions, substitutions and subshells. Pipeline simple commands
# are observed in their parent shell. This is not a hostile-shell security layer.
OBSERVER = r'''set -T
__jb_pid=$BASHPID
__jb_seq=0
__jb_fail() {
    builtin printf 'observer-write-failure pid=%s\n' "$BASHPID" >&"$_JBRAVO_OBSERVER_FAILURE_FD"
    builtin exit 96
}
__jb_write() {
    __jb_seq=$((__jb_seq + 1))
    builtin printf '%s\0' "$1" "$BASHPID" "$$" "$BASH_SUBSHELL" "$__jb_seq" "$2" "$3" "$4" >>"$_JBRAVO_OBSERVATION_DIR/$BASHPID.bin" || __jb_fail
}
__jb_finish() {
    trap - DEBUG
    __jb_write end observer 0 "$1"
}
__jb_observe() {
    if [[ $__jb_pid != "$BASHPID" ]]; then
        __jb_pid=$BASHPID
        __jb_seq=0
        trap '__jb_finish "$?"' EXIT
        __jb_write begin observer 0 child
    fi
    __jb_write command "$2" "$3" "$1"
}
trap '__jb_finish "$?"' EXIT
__jb_write begin observer 0 root
trap '__jb_observe "$BASH_COMMAND" "${BASH_SOURCE[0]}" "$LINENO"' DEBUG
'''
OBSERVER_EXIT_LINES = {
    OBSERVER.splitlines().index('__jb_finish() {') + 1,
    OBSERVER.splitlines().index('    trap - DEBUG') + 1,
}

# Static policy for the reviewed wrapper, never learned from selected input or
# probe bytes. Exact double argv/stdin checks remain an additional requirement.
WRAPPER_COMMANDS = {
    'set -Eeuo pipefail', 'export TZ="America/New_York"',
    'PROJECT_HOME="${PROJECT_HOME:-/home/RasPatrick/jbravo_screener}"',
    'VENV="${VENV:-/home/RasPatrick/.virtualenvs/jbravo-env}"',
    'cd "$PROJECT_HOME"', 'source "$VENV/bin/activate"',
    'set -a', '. ~/.config/jbravo/.env', 'set +a',
    ': "${APCA_API_KEY_ID:?Missing APCA_API_KEY_ID}"',
    ': "${APCA_API_SECRET_KEY:?Missing APCA_API_SECRET_KEY}"',
    ': "${APCA_API_BASE_URL:?Missing APCA_API_BASE_URL}"',
    'DRY_RUN="${JBRAVO_DRY_RUN:-false}"', 'DRY_RUN=true', 'DRY_RUN=false',
    'case "${1,,}" in', 'case "$1" in', 'case "$PIPELINE_GATE" in',
    '[[ $# -gt 0 ]]', '[[ $# -eq 1 || "${2-}" == --dry-run ]]',
    'normalize_dry_run "$2"', 'normalize_dry_run "$DRY_RUN"', 'shift', 'shift 2',
    'echo "Invalid dry-run value; use true,false,1,0,yes,no,y,n." 1>&2',
    'echo "Unknown argument: $1" 1>&2', 'exit 0', 'exit 1', 'exit 2',
    'echo "$PIPELINE_GATE"', 'tee -a "$PROJECT_HOME/logs/execute_trades.log"',
    'echo "[WRAPPER] probing Alpaca credentials"', 'probe_rc=$?',
    'echo "$ALPACA_PROBE"', '[ "$probe_rc" -ne 0 ]', 'exit "$probe_rc"',
    'export ALPACA_PROBE', 'echo "[WRAPPER] Alpaca account probe OK"',
    'export PREMARKET_STARTED_UTC', 'export PREMARKET_FINISHED_UTC',
    'python -m scripts.check_connection',
    'echo "[WARN] connection probe failed (non-fatal)"',
    'python -m scripts.execute_trades --dry-run "$DRY_RUN" ' + ' '.join(TRADING_ARGS),
    'touch /var/www/raspatrick_pythonanywhere_com_wsgi.py', 'true',
}
CAPTURE_STAGES = {"PIPELINE_GATE": "gate", "ALPACA_PROBE": "account",
                  "PREMARKET_STARTED_UTC": "timestamp", "PREMARKET_FINISHED_UTC": "timestamp"}


def approved_wrapper_command(command):
    if command.strip() in WRAPPER_COMMANDS:
        return True
    capture = re.fullmatch(r"([A-Z_]+)=\$\((python - <<'PY'\n.*\nPY\n)\)", command, re.DOTALL)
    program = capture[2] if capture else command
    inline = re.fullmatch(r"python - <<'PY'\n(.*\n)PY\n?", program, re.DOTALL)
    if inline is None:
        return False
    stage = INLINE_PROGRAMS.get(hashlib.sha256(inline[1].encode()).hexdigest())
    return stage is not None and (not capture or CAPTURE_STAGES.get(capture[1]) == stage)


def observe_commands(directory, wrapper, bootstrap, config, returncode):
    """Fail closed on channel failure, incomplete shells, or unapproved attempts."""
    observation = {"observer_protocol": "bash-debug-v1", "status": "failed", "error": None,
                   "directory": str(directory), "bootstrap_sha256": hashlib.sha256(bootstrap.read_bytes()).hexdigest(),
                   "events": [], "violations": [], "files": []}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            data = path.read_bytes()
            observation["files"].append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    try:
        if (directory / "recorder-failures.log").read_bytes():
            raise ValueError("Observer record write failed; dedicated failure channel is nonempty")
        paths = sorted(directory.glob("*.bin"))
        if not paths:
            raise ValueError("Required observer streams are absent")
        streams = {}
        lifecycle_errors = []
        for path in paths:
            data = path.read_bytes()
            fields = data.split(b"\0")
            if not data or fields[-1] != b"" or (len(fields) - 1) % 8:
                raise ValueError("Malformed or truncated observer framing: " + path.name)
            rows = []
            for offset in range(0, len(fields) - 1, 8):
                kind, pid, shell, depth, seq, source, line, command = [field.decode("utf-8") for field in fields[offset:offset + 8]]
                if (kind not in {"begin", "command", "end"} or
                        not all(value.isdecimal() for value in (pid, shell, depth, seq, line)) or
                        path.name != pid + ".bin" or int(seq) != len(rows) + 1):
                    raise ValueError("Invalid observer record or sequence: " + path.name)
                event = {"kind": kind, "pid": int(pid), "root_pid": int(shell), "subshell": int(depth),
                         "sequence": int(seq), "source": source, "line": int(line), "command": command}
                rows.append(event)
                observation["events"].append(event)
            if (rows[0]["kind"] != "begin" or rows[-1]["kind"] != "end" or
                    not rows[-1]["command"].isdecimal() or int(rows[-1]["command"]) > 255 or
                    not any(row["kind"] == "command" for row in rows) or
                    any(row["kind"] != "command" for row in rows[1:-1]) or
                    any(row["root_pid"] != rows[0]["root_pid"] or row["subshell"] != rows[0]["subshell"] for row in rows)):
                lifecycle_errors.append("Incomplete observer shell lifecycle: " + path.name)
            streams[rows[0]["pid"]] = rows
        roots = [rows for pid, rows in streams.items() if pid == rows[0]["root_pid"]]
        if len(roots) != 1 or roots[0][0]["command"] != "root" or roots[0][0]["subshell"] != 0:
            raise ValueError("Missing or inconsistent root observer completion")
        if roots[0][-1]["kind"] == "end" and roots[0][-1]["command"].isdecimal():
            if int(roots[0][-1]["command"]) != returncode:
                lifecycle_errors.append("Root observer exit status disagrees with wrapper")
        root_pid = roots[0][0]["pid"]
        for pid, rows in streams.items():
            if rows[0]["root_pid"] != root_pid or (pid != root_pid and rows[0]["command"] != "child"):
                raise ValueError("Observer stream belongs to an unexpected shell")
        wrapper_events = []
        for event in observation["events"]:
            if event["kind"] != "command":
                if event["source"] != "observer" or event["line"] != 0:
                    raise ValueError("Invalid observer lifecycle origin")
                continue
            source, command = event["source"], event["command"]
            if source == str(wrapper):
                event["origin"] = "wrapper"
                wrapper_events.append(event)
                approved = approved_wrapper_command(command)
            elif source == str(config):
                event["origin"] = "synthetic-configuration"
                approved = command in config.read_text().splitlines()
            elif source == str(bootstrap) and event["line"] in OBSERVER_EXIT_LINES:
                # Bash retains the interrupted BASH_COMMAND in EXIT-handler
                # DEBUG callbacks. These two fixed handler lines are runtime.
                event["origin"] = "observer-exit-runtime"
                approved = True
            else:
                event["origin"] = "unexpected-source"
                approved = False
            if not approved:
                observation["violations"].append({**event, "reason": "Unapproved wrapper command attempt or source"})
        root_wrapper = [event for event in roots[0] if event.get("origin") == "wrapper"]
        if not wrapper_events or not root_wrapper or root_wrapper[0]["command"] != "set -Eeuo pipefail":
            raise ValueError("Observer did not positively observe the first wrapper command")
        if observation["violations"]:
            raise ValueError("; ".join(["Unapproved command attempts observed", *lifecycle_errors]))
        if lifecycle_errors:
            raise ValueError("; ".join(lifecycle_errors))
        observation["status"] = "passed"
    except (OSError, UnicodeError, ValueError) as error:
        observation["error"] = str(error)
    return observation


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
        self.source_wrapper_sha256 = self.wrapper_sha256
        self.variant = None
        self.observer = root / "observer.sh"
        self.observer.write_text(OBSERVER)
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
        config_path = self.home / ".config/jbravo/.env"
        config_path.write_text("\n".join(config) + "\n")
        env = {
            "HOME": str(self.home), "PROJECT_HOME": str(self.project), "VENV": str(self.venv),
            "PATH": str(self.bin), "LC_ALL": "C", "FAKE_TRACE": str(self.trace),
            "FAKE_INLINE_PROGRAMS": json.dumps(INLINE_PROGRAMS),
        }
        if setting is not None:
            env["JBRAVO_DRY_RUN"] = setting
        env.update(overrides or {})
        directory = Path(os.environ["JBRAVO_TEST_EVIDENCE"]).parent / "command-observations" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        bootstrap = directory / "bootstrap.sh"
        bootstrap.write_bytes(self.observer.read_bytes())
        (directory / "executed-wrapper.sh").write_bytes(self.wrapper.read_bytes())
        self.wrapper_sha256 = hashlib.sha256(self.wrapper.read_bytes()).hexdigest()
        with (directory / "recorder-failures.log").open("wb") as failure_channel:
            env.update(BASH_ENV=str(bootstrap), _JBRAVO_OBSERVATION_DIR=str(directory),
                       _JBRAVO_OBSERVER_FAILURE_FD=str(failure_channel.fileno()))
            result = subprocess.run(
                ["/usr/bin/bash", "--noprofile", "--norc", str(self.wrapper), *args],
                cwd=self.project, env=env, text=True, capture_output=True, timeout=15,
                pass_fds=(failure_channel.fileno(),),
            )
        observation = observe_commands(directory, self.wrapper, bootstrap, config_path, result.returncode)
        events = [json.loads(line) for line in self.trace.read_text().splitlines()] if self.trace.exists() else []
        rejections = ([observation["error"]] if observation["status"] != "passed" else [])
        if any("violation" in event for event in events):
            rejections.append("Exact command double rejected an unapproved invocation")
        evidence = {
            "test": os.environ.get("PYTEST_CURRENT_TEST"), "wrapper_sha256": self.wrapper_sha256,
            "source_wrapper_sha256": self.source_wrapper_sha256, "variant": self.variant,
            "requested_args": list(args), "setting": setting, "config_setting": config_setting,
            "controls": overrides or {}, "omit_key": omit_key, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr, "events": events, "observation": observation,
            "validation_status": "rejected" if rejections else "accepted", "rejection_reasons": rejections,
        }
        self.last_evidence = evidence
        with open(os.environ["JBRAVO_TEST_EVIDENCE"], "a", encoding="utf-8") as out:
            out.write(json.dumps(evidence) + "\n")
        assert not rejections, evidence
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


COMMAND_PROBES = [
    ("review-suppressed-unknown", 'jbravo_unapproved_probe 2>/dev/null || true\n', "jbravo_unapproved_probe"),
    ("review-absolute-true", '/usr/bin/true\n', "/usr/bin/true"),
    ("review-suppressed-double", 'python -m scripts.unapproved_contract_boundary 2>/dev/null || true\n', "scripts.unapproved_contract_boundary"),
    ("subshell-unknown", '(jbravo_unapproved_probe 2>/dev/null || true)\n', "jbravo_unapproved_probe"),
    ("substitution-absolute", ': "$(/usr/bin/true)"\n', "/usr/bin/true"),
    ("function-unknown", '  jbravo_unapproved_probe 2>/dev/null || true\n', "jbravo_unapproved_probe"),
    ("redirection-substitution", ': > "$(/usr/bin/true; echo /dev/null)"\n', "/usr/bin/true"),
]
REVIEW_PROBE_HASHES = {
    "review-suppressed-unknown": "6577fb8bab421f6e0295f2461743fe7f26cb687cbddce11cd5aa60cff75f0e48",
    "review-absolute-true": "54f55577bb91ad52c3312f0e66ce5c822622c6091df4435a48ffb165db1af473",
    "review-suppressed-double": "6932014055810a10ded56dea727d28b0c28e0023ba4aecf679c0c6d25f04a6fa",
}


def install_probe(harness, name, insertion):
    source = harness.wrapper.read_bytes()
    anchor = 'normalize_dry_run() {\n' if name == "function-unknown" else 'normalize_dry_run "$DRY_RUN"\n'
    assert source.count(anchor.encode()) == 1
    modified = source.replace(anchor.encode(), (anchor + insertion).encode(), 1)
    harness.wrapper.write_bytes(modified)
    harness.variant = {"name": name, "anchor": anchor, "insertion": insertion,
                       "source_sha256": hashlib.sha256(source).hexdigest(), "sha256": hashlib.sha256(modified).hexdigest()}


@pytest.mark.parametrize("name,insertion,attempt", COMMAND_PROBES, ids=[probe[0] for probe in COMMAND_PROBES])
def test_command_attempts_rejected(harness, name, insertion, attempt):
    install_probe(harness, name, insertion)
    if name in REVIEW_PROBE_HASHES:
        assert harness.variant["sha256"] == REVIEW_PROBE_HASHES[name], "Primary regressions must retain exact reviewer probe bytes"
    with pytest.raises(AssertionError):
        assert_mode(harness, "false")
    evidence = harness.last_evidence
    assert evidence["returncode"] == 0, evidence
    assert evidence["validation_status"] == "rejected"
    observation = evidence["observation"]
    assert observation["status"] == "failed" and observation["error"].startswith("Unapproved command attempts observed"), observation
    attempts = [event for event in observation["violations"] if attempt in event["command"]]
    assert attempts, observation
    if name in {"subshell-unknown", "substitution-absolute", "redirection-substitution"}:
        assert any(event["subshell"] > 0 and event["command"].startswith(attempt) for event in attempts), attempts
    if name == "substitution-absolute":
        # Bash may exec a substitution's sole external command without EXIT.
        # The attempt is recorded, and missing completion remains a failure.
        assert "Incomplete observer shell lifecycle" in observation["error"], observation
    if name == "review-suppressed-double":
        assert any(event.get("violation") == "unknown Python invocation" for event in evidence["events"])
    assert_executor_arguments(evidence["events"], "false")


@pytest.mark.parametrize("fault", ["absent", "malformed", "incomplete", "write-failure"])
def test_observation_failure_rejected(harness, fault):
    if fault == "absent":
        harness.observer.write_text("# Deliberately absent observer, synthetic negative control.\n")
    elif fault == "malformed":
        harness.observer.write_text(OBSERVER + '\nbuiltin printf malformed >>"$_JBRAVO_OBSERVATION_DIR/$BASHPID.bin"\n')
    elif fault == "incomplete":
        harness.observer.write_text(OBSERVER.replace('__jb_write end observer 0 "$1"', ': # deliberately omit completion'))
    else:
        # The failed child is explicitly ignored; a separate inherited descriptor
        # must still make this attempt fail validation with wrapper status zero.
        install_probe(harness, "observer-write-failure", '(true) || true\n')
        failure = '''    if [[ "$1" == command && "$4" == true && "$BASH_SUBSHELL" -gt 0 ]]; then
        builtin printf fail >>"$_JBRAVO_OBSERVATION_DIR/missing/record" || __jb_fail
    fi
'''
        harness.observer.write_text(OBSERVER.replace('    __jb_seq=$((__jb_seq + 1))\n', failure + '    __jb_seq=$((__jb_seq + 1))\n'))
    with pytest.raises(AssertionError):
        assert_mode(harness, "false")
    evidence = harness.last_evidence
    assert evidence["returncode"] == 0, evidence
    assert evidence["validation_status"] == "rejected"
    observation = evidence["observation"]
    expected = {"absent": "Required observer streams are absent", "malformed": "Invalid observer record or sequence",
                "incomplete": "Incomplete observer shell lifecycle", "write-failure": "Observer record write failed"}
    assert observation["status"] == "failed" and expected[fault] in observation["error"], observation
    if fault == "malformed":
        # Begin/end alone do not prove commands were observed. This parser-only
        # fixture is not another wrapper invocation or another pytest case.
        empty = Path(observation["directory"]) / "parser-no-command"
        empty.mkdir()
        (empty / "bootstrap.sh").write_text(OBSERVER)
        (empty / "recorder-failures.log").write_bytes(b"")
        fields = ["begin", "999", "999", "0", "1", "observer", "0", "root",
                  "end", "999", "999", "0", "2", "observer", "0", "0"]
        (empty / "999.bin").write_bytes(("\0".join(fields) + "\0").encode())
        parsed = observe_commands(empty, harness.wrapper, empty / "bootstrap.sh", harness.home / ".config/jbravo/.env", 0)
        assert parsed["status"] == "failed" and "did not positively observe" in parsed["error"]
        truncated = empty.parent / "parser-truncated"
        truncated.mkdir()
        (truncated / "bootstrap.sh").write_text(OBSERVER)
        (truncated / "recorder-failures.log").write_bytes(b"")
        (truncated / "999.bin").write_bytes((empty / "999.bin").read_bytes()[:-1])
        parsed = observe_commands(truncated, harness.wrapper, truncated / "bootstrap.sh", harness.home / ".config/jbravo/.env", 0)
        assert parsed["status"] == "failed" and "truncated observer framing" in parsed["error"]
