import argparse
import json
import sys
from typing import Any, Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


def fetch_json(base_url: str, path: str, timeout: int = 10) -> Optional[Any]:
    url = base_url.rstrip("/") + path
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = resp.read().decode("utf-8", errors="ignore")
            return json.loads(data)
    except (URLError, HTTPError, json.JSONDecodeError):
        return None


def print_result(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def audit_resources(payload: Any) -> bool:
    if not isinstance(payload, dict):
        print_result("PythonAnywhere resources", False, "no response")
        return False
    resources = payload.get("resources") or []
    if not isinstance(resources, list) or not resources:
        print_result("PythonAnywhere resources", False, "empty resources")
        return False
    labels = {str(resource.get("label")) for resource in resources if isinstance(resource, dict)}
    expected = {"CPU Usage", "File Storage", "Postgres Storage"}
    missing = expected - labels
    ok = not missing
    detail = "missing: " + ", ".join(sorted(missing)) if missing else "all present"
    print_result("PythonAnywhere resources", ok, detail)

    value_ok = True
    for resource in resources:
        if not isinstance(resource, dict):
            value_ok = False
            continue
        value = resource.get("value")
        label = resource.get("label") or "resource"
        if value is None:
            print_result(f"Resource {label}", False, "value missing")
            value_ok = False
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            print_result(f"Resource {label}", False, "value not numeric")
            value_ok = False
            continue
        if numeric < 0 or numeric > 100:
            print_result(f"Resource {label}", False, f"value out of range: {numeric}")
            value_ok = False
    if value_ok:
        print_result("Resource values", True, "0-100% range")
    return ok and value_ok


def audit_tasks(payload: Any) -> bool:
    if not isinstance(payload, dict):
        print_result("PythonAnywhere tasks", False, "no response")
        return False
    tasks = payload.get("tasks") or []
    if not isinstance(tasks, list) or not tasks:
        print_result("PythonAnywhere tasks", False, "empty tasks")
        return False
    missing_fields = 0
    for task in tasks:
        if not isinstance(task, dict):
            missing_fields += 1
            continue
        for field in ("name", "frequency", "time"):
            if not task.get(field):
                missing_fields += 1
                break
    ok = missing_fields == 0
    detail = f"{len(tasks)} tasks" + ("" if ok else f", {missing_fields} missing fields")
    print_result("PythonAnywhere tasks", ok, detail)
    return ok


def audit_account(payload: Any) -> bool:
    if not isinstance(payload, dict):
        print_result("Alpaca account", False, "no response")
        return False
    snapshot = payload.get("snapshot") or {}
    ok_flag = bool(payload.get("ok"))
    equity = snapshot.get("equity")
    ok = ok_flag and equity is not None
    detail = f"equity={equity}" if equity is not None else "equity missing"
    print_result("Alpaca account", ok, detail)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit dashboard API payloads.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8050",
        help="Base URL for dashboard server.",
    )
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds.")
    args = parser.parse_args()

    resources = fetch_json(args.base_url, "/api/pythonanywhere/resources", args.timeout)
    tasks = fetch_json(args.base_url, "/api/pythonanywhere/tasks", args.timeout)
    account = fetch_json(args.base_url, "/api/account/overview", args.timeout)

    ok = True
    ok &= audit_resources(resources)
    ok &= audit_tasks(tasks)
    ok &= audit_account(account)

    print("\nSummary:")
    print_result("Overall", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
