from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from scripts.utils.pythonanywhere_env import (
    mask_secret,
    resolution_error_message,
    resolve_pythonanywhere_credentials,
)


def _request_json(
    method: str,
    url: str,
    token: str,
    *,
    form: Mapping[str, str] | None = None,
) -> tuple[int, Any]:
    headers = {
        "Authorization": f"Token {token}",
        "Accept": "application/json",
    }
    data = None
    if form is not None:
        data = urllib.parse.urlencode(dict(form)).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = urllib.request.Request(url=url, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(payload) if payload else {}
            return int(response.status), parsed
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        parsed: Any
        try:
            parsed = json.loads(payload) if payload else {}
        except Exception:
            parsed = {"detail": payload}
        return int(exc.code), parsed


def _console_is_bash(console: Mapping[str, Any]) -> bool:
    haystack = " ".join(
        str(console.get(field, ""))
        for field in ("executable", "description", "console_type", "command", "type")
    ).lower()
    return "bash" in haystack


def _select_console(consoles: list[Mapping[str, Any]], console_id: int | None) -> Mapping[str, Any] | None:
    if console_id is not None:
        for console in consoles:
            if int(console.get("id", -1)) == console_id:
                return console
        return None
    for console in consoles:
        if _console_is_bash(console):
            return console
    return None


def _latest_output_excerpt(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""
    for key in ("output", "stdout", "stderr"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            lines = [line for line in value.replace("\r", "").split("\n") if line.strip()]
            if lines:
                return lines[-1].strip()
    return ""


def _print_probe(repo_root: Path | None = None) -> int:
    resolution = resolve_pythonanywhere_credentials(repo_root)
    values = resolution.values
    print("PA_ENV_PROBE")
    for location in resolution.checked_locations:
        print(f"lookup={location}")
    print(f"PA_USERNAME={values.get('PA_USERNAME', '<missing>')}")
    print(f"PA_HOST={values.get('PA_HOST', '<missing>')}")
    print(f"PA_TOKEN_PRESENT={str(bool(values.get('PA_TOKEN'))).lower()}")
    print(f"PA_TOKEN_HINT={mask_secret(values.get('PA_TOKEN'))}")
    for key in ("PA_USERNAME", "PA_TOKEN", "PA_HOST"):
        print(f"{key}_SOURCE={resolution.value_sources.get(key, '<missing>')}")
    if not resolution.ok:
        print(resolution_error_message(resolution, repo_root))
        return 2
    print("PA_ENV_OK")
    return 0


def _print_handshake(repo_root: Path | None = None, console_id: int | None = None) -> int:
    resolution = resolve_pythonanywhere_credentials(repo_root)
    if not resolution.ok:
        print(resolution_error_message(resolution, repo_root))
        return 2

    values = resolution.values
    username = values["PA_USERNAME"]
    token = values["PA_TOKEN"]
    host = values["PA_HOST"]
    base_url = f"https://{host}/api/v0/user/{username}"

    status, consoles_payload = _request_json("GET", f"{base_url}/consoles/", token)
    print(f"PA_API_LIST_CONSOLES status={status} host={host} username={username}")
    if status != 200:
        detail = consoles_payload if isinstance(consoles_payload, Mapping) else {"detail": consoles_payload}
        print(f"PA_API_LIST_CONSOLES_FAILED detail={json.dumps(detail, sort_keys=True)}")
        return 3

    if not isinstance(consoles_payload, list):
        print("PA_API_LIST_CONSOLES_FAILED detail=unexpected_payload")
        return 3

    print(f"PA_API_CONSOLES count={len(consoles_payload)}")
    console = _select_console(consoles_payload, console_id)
    if console is None:
        if console_id is not None:
            print(f"PA_PWD_HANDSHAKE_BLOCKED reason=console_id_not_found console_id={console_id}")
        else:
            print("PA_PWD_HANDSHAKE_BLOCKED reason=no_bash_console_found")
        return 4

    selected_id = int(console.get("id"))
    print(f"PA_API_CONSOLE_SELECTED id={selected_id}")

    send_status, send_payload = _request_json(
        "POST",
        f"{base_url}/consoles/{selected_id}/send_input/",
        token,
        form={"input": "pwd\n"},
    )
    print(f"PA_API_SEND_PWD status={send_status} console_id={selected_id}")
    if send_status == 412:
        detail = send_payload if isinstance(send_payload, Mapping) else {"detail": send_payload}
        print(
            "PA_PWD_HANDSHAKE_BLOCKED "
            f"reason=console_not_started_in_browser console_id={selected_id} "
            f"detail={json.dumps(detail, sort_keys=True)}"
        )
        return 5
    if send_status != 200:
        detail = send_payload if isinstance(send_payload, Mapping) else {"detail": send_payload}
        print(
            "PA_PWD_HANDSHAKE_BLOCKED "
            f"reason=send_input_failed console_id={selected_id} detail={json.dumps(detail, sort_keys=True)}"
        )
        return 5

    latest_status = 0
    latest_payload: Any = {}
    for _ in range(3):
        time.sleep(0.5)
        latest_status, latest_payload = _request_json(
            "GET",
            f"{base_url}/consoles/{selected_id}/get_latest_output/",
            token,
        )
        if latest_status == 200:
            break

    print(f"PA_API_GET_LATEST_OUTPUT status={latest_status} console_id={selected_id}")
    if latest_status == 412:
        detail = latest_payload if isinstance(latest_payload, Mapping) else {"detail": latest_payload}
        print(
            "PA_PWD_HANDSHAKE_BLOCKED "
            f"reason=console_not_started_in_browser console_id={selected_id} "
            f"detail={json.dumps(detail, sort_keys=True)}"
        )
        return 6
    if latest_status != 200:
        detail = latest_payload if isinstance(latest_payload, Mapping) else {"detail": latest_payload}
        print(
            "PA_PWD_HANDSHAKE_BLOCKED "
            f"reason=get_latest_output_failed console_id={selected_id} detail={json.dumps(detail, sort_keys=True)}"
        )
        return 6

    excerpt = _latest_output_excerpt(latest_payload)
    if excerpt:
        print(f"PA_PWD_HANDSHAKE_OK console_id={selected_id} latest_output={excerpt}")
    else:
        print(f"PA_PWD_HANDSHAKE_OK console_id={selected_id} latest_output=<empty>")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe PythonAnywhere API credentials and optionally verify the pwd console handshake."
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repository root override for repo-local env lookup",
    )
    parser.add_argument(
        "--handshake",
        action="store_true",
        help="List consoles, send pwd to an existing Bash console, and verify latest output access",
    )
    parser.add_argument(
        "--console-id",
        type=int,
        default=None,
        help="Optional existing console id to use for the pwd handshake",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    if args.handshake:
        return _print_handshake(repo_root=repo_root, console_id=args.console_id)
    return _print_probe(repo_root=repo_root)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
