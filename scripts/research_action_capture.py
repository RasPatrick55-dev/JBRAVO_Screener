"""Bounded, research-only corporate-action collection. Importing performs no I/O."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"
SYMBOLS = ("ALK", "FN", "HBAN", "HMY", "LOGI", "MIRM", "NBIS", "NVST",
           "PLUG", "RGLD", "SKY", "SPY", "SRRK", "STLD", "VSCO", "VSXY", "YNDX")
REPO = Path(__file__).absolute().parents[1]
SPEC_SHA256 = "6c1bba3f657e01d47a5135dd72842bd1bd68d9c844c3ee339c9c9ae18326fb74"
LIMITATIONS = (
    "Dates filter inclusive process_date, not effective, ex, record or payment dates.",
    "Terminal/empty pagination is not historical completeness or a point-in-time snapshot.",
    "Provider envelope and full event schemas remain unverified; parser is provisional.",
    "Unknown event fields remain raw; ratios, currencies and entitlements are not interpreted.",
    "Aliases/CUSIPs do not establish complete security continuity or historical eligibility.",
    "Revisions, deletions, service latency and earliest historical coverage remain unresolved.",
    "Local deadlines do not establish remote cancellation; wire/incomplete transfer totals unknown.",
)


class CaptureError(Exception):
    """Fixed codes only: never retain arbitrary exception text or authentication headers."""


ERROR_CODES = frozenset((
    "SAVED_BYTE_MISMATCH", "INVALID_ARTIFACT_BINDING", "INVALID_ARTIFACT_FILE",
    "ARTIFACT_BINDING_MISMATCH", "UNSAFE_CONTAINMENT", "INVALID_CAPTURE_ID",
    "OUTPUT_INSIDE_SOURCE", "DESTINATION_COLLISION", "INVALID_PROCESS_DATE",
    "INVALID_PROCESS_DATE_RANGE", "INVALID_SYMBOL_QUERY", "UNBOUND_IDENTIFIER",
    "AMBIGUOUS_IDENTIFIER", "INVALID_LIMIT", "AUTHENTICATION_UNAVAILABLE",
    "TRANSPORT_FAILURE", "LOCAL_REQUEST_TIMEOUT_REMOTE_STATE_UNKNOWN",
    "UNRESOLVED_RESPONSE_ENVELOPE", "UNRESOLVED_PAGINATION_SCHEMA",
    "UNRESOLVED_EVENT_ARRAY_SCHEMA", "UNRESOLVED_EVENT_SCHEMA",
    "LOCAL_ACCEPTANCE_DEADLINE", "GET_LIMIT", "AGGREGATE_BYTE_LIMIT",
    "INVALID_TRANSPORT_RESPONSE", "AUTHENTICATION_ECHO",
    "RESPONSE_OR_AGGREGATE_BYTE_LIMIT", "HTTP_STATUS_FAILURE",
    "UNSUPPORTED_CONTENT_ENCODING", "MALFORMED_RESPONSE_JSON",
    "UNRESOLVED_EVENT_ID_SCHEMA", "CONFLICTING_EVENT_ID_VARIANTS",
    "REPEATED_PAGE_TOKEN", "CHAIN_PAGE_LIMIT", "MANIFEST_READBACK_MISMATCH",
    "FINAL_RECORDING_OR_ELIGIBILITY_FAILURE", "IDENTIFIER_MANIFEST_BINDING_MISMATCH",
    "INVALID_IDENTIFIER_MANIFEST", "IDENTIFIER_MANIFEST_BINDING_REQUIRED",
))


def error_code(error, fallback):
    # A transport/test double can raise CaptureError with arbitrary text as well.
    return str(error) if isinstance(error, CaptureError) and str(error) in ERROR_CODES else fallback


def digest(body):
    return hashlib.sha256(body).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def strict_json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    return json.loads(body.decode("utf-8", errors="strict"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("constant")))


def save_bytes(root, name, body):
    path = root / name
    with path.open("xb") as handle:
        handle.write(body)
    if path.read_bytes() != body:
        raise CaptureError("SAVED_BYTE_MISMATCH")
    return {"path": name, "bytes": len(body), "sha256": digest(body)}


def save_json(root, name, value):
    return save_bytes(root, name, json_bytes(value))


def verify_bindings(root, bindings):
    seen = set()
    for binding in bindings:
        name = binding["path"]
        if name in seen or Path(name).name != name:
            raise CaptureError("INVALID_ARTIFACT_BINDING")
        seen.add(name)
        path = root / name
        if redirected(path) or not path.is_file():
            raise CaptureError("INVALID_ARTIFACT_FILE")
        body = path.read_bytes()
        if len(body) != binding["bytes"] or digest(body) != binding["sha256"]:
            raise CaptureError("ARTIFACT_BINDING_MISMATCH")


def redirected(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def ordinary_ancestors(path):
    for ancestor in reversed((path, *path.parents)):
        if ancestor.exists() or ancestor.is_symlink():
            if redirected(ancestor) or not ancestor.is_dir():
                raise CaptureError("UNSAFE_CONTAINMENT")


def destination(base, capture_id):
    base = Path(base)
    if not base.is_absolute() or ".." in base.parts:
        raise CaptureError("UNSAFE_CONTAINMENT")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", capture_id):
        raise CaptureError("INVALID_CAPTURE_ID")
    target = base / capture_id
    ordinary_ancestors(base)
    ordinary_ancestors(target)
    for path in (base, target):
        if path == REPO or REPO in path.parents:
            raise CaptureError("OUTPUT_INSIDE_SOURCE")
    if target.exists() or target.is_symlink():
        raise CaptureError("DESTINATION_COLLISION")
    return target


def day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise CaptureError("INVALID_PROCESS_DATE")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise CaptureError("INVALID_PROCESS_DATE") from None


@dataclass(frozen=True)
class IdentifierBinding:
    """An independently sourced candidate binding, not proof of complete ID history."""
    cusip: str
    security_key: str
    source_reference: str
    source_sha256: str
    valid_from: str
    valid_through: str
    independently_bound: bool

    def validate(self):
        if (not re.fullmatch(r"[A-Z0-9]{9}", self.cusip)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.security_key)
                or not re.fullmatch(r"[a-f0-9]{64}", self.source_sha256)
                or not re.fullmatch(r"[A-Za-z0-9_./:# -]{1,240}", self.source_reference)
                or self.independently_bound is not True
                or day(self.valid_from) > day(self.valid_through)):
            raise CaptureError("UNBOUND_IDENTIFIER")


@dataclass(frozen=True)
class Settings:
    start: str
    end: str
    symbols: tuple = SYMBOLS
    identifiers: tuple = ()
    identifier_manifest_sha256: str | None = None
    max_gets: int = 12
    max_pages: int = 6
    response_bytes: int = 4 * 1024 * 1024
    aggregate_bytes: int = 32 * 1024 * 1024
    acceptance_seconds: float = 600
    request_seconds: float = 15

    def validate(self):
        if day(self.start) > day(self.end):
            raise CaptureError("INVALID_PROCESS_DATE_RANGE")
        if (not isinstance(self.symbols, tuple) or not self.symbols
                or tuple(sorted(set(self.symbols))) != self.symbols
                or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,19}", s)
                       for s in self.symbols)):
            raise CaptureError("INVALID_SYMBOL_QUERY")
        if not isinstance(self.identifiers, tuple):
            raise CaptureError("UNBOUND_IDENTIFIER")
        if self.identifier_manifest_sha256 is not None and (
                not self.identifiers or not re.fullmatch(r"[a-f0-9]{64}", self.identifier_manifest_sha256)):
            raise CaptureError("INVALID_IDENTIFIER_MANIFEST")
        for binding in self.identifiers:
            if not isinstance(binding, IdentifierBinding):
                raise CaptureError("UNBOUND_IDENTIFIER")
            binding.validate()
        if len({b.cusip for b in self.identifiers}) != len(self.identifiers):
            raise CaptureError("AMBIGUOUS_IDENTIFIER")
        for value, ceiling in ((self.max_gets, 12), (self.max_pages, 6),
                               (self.response_bytes, 4 * 1024 * 1024),
                               (self.aggregate_bytes, 32 * 1024 * 1024)):
            if type(value) is not int or not 0 < value <= ceiling:
                raise CaptureError("INVALID_LIMIT")
        for value, ceiling in ((self.acceptance_seconds, 600), (self.request_seconds, 15)):
            if type(value) not in (int, float) or not 0 < value <= ceiling:
                raise CaptureError("INVALID_LIMIT")

    def chains(self):
        common = {"start": self.start, "end": self.end, "region": "all",
                  "data_quality": "all", "sort": "asc", "limit": 1000}
        chains = [("symbols", {**common, "symbols": ",".join(self.symbols)})]
        if self.identifiers:
            chains.append(("cusips", {**common, "cusips": ",".join(sorted(b.cusip for b in self.identifiers))}))
        return chains


@dataclass
class Response:
    status: int
    body: bytes
    complete: bool = True
    content_encoding: str = "identity"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RestTransport:
    """One local wait per GET. Late workers cannot persist or retry anything."""
    identity = {"client": "Python urllib.request", "retry": False, "redirect": False}

    def __init__(self, key, secret):
        if not key or not secret:
            raise CaptureError("AUTHENTICATION_UNAVAILABLE")
        self._secrets = (key, secret)

    def contains_secret(self, body):
        return any(value.encode("utf-8") in body for value in self._secrets)

    def get(self, params, timeout, byte_limit):
        results = queue.Queue(maxsize=1)
        def operation():
            response = None
            try:
                request = urllib.request.Request(
                    ENDPOINT + "?" + urllib.parse.urlencode(params), method="GET",
                    headers={"APCA-API-KEY-ID": self._secrets[0],
                             "APCA-API-SECRET-KEY": self._secrets[1],
                             "Accept-Encoding": "identity"})
                opener = urllib.request.build_opener(NoRedirect())
                try:
                    response = opener.open(request, timeout=timeout)
                except urllib.error.HTTPError as error:
                    response = error
                body = response.read(byte_limit + 1)
                encoding = response.headers.get("Content-Encoding", "identity")
                results.put(Response(response.code, body[:byte_limit], len(body) <= byte_limit,
                                     encoding))
            except Exception:
                results.put(CaptureError("TRANSPORT_FAILURE"))
            finally:
                if response is not None:
                    response.close()
        threading.Thread(target=operation, daemon=True).start()
        try:
            result = results.get(timeout=timeout)
        except queue.Empty:
            raise CaptureError("LOCAL_REQUEST_TIMEOUT_REMOTE_STATE_UNKNOWN") from None
        if isinstance(result, Exception):
            raise result
        return result


@dataclass
class Node:
    value: object
    start: int
    end: int
    children: object = None


class RawJson:
    """Strict structural parser retaining character spans for exact UTF-8 event slices."""
    def __init__(self, body):
        self.text = body.decode("utf-8", errors="strict")
        self.decoder = json.JSONDecoder(parse_float=Decimal,
                                        parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))

    def ws(self, index):
        while index < len(self.text) and self.text[index] in " \r\n\t":
            index += 1
        return index

    def node(self, index=0, depth=0):
        if depth > 64:
            raise ValueError("depth")
        start = index = self.ws(index)
        if index >= len(self.text):
            raise ValueError("missing")
        char = self.text[index]
        if char not in "[{":
            value, end = self.decoder.raw_decode(self.text, index)
            return Node(value, start, end)
        is_object = char == "{"
        children = {} if is_object else []
        index = self.ws(index + 1)
        close = "}" if is_object else "]"
        if index < len(self.text) and self.text[index] == close:
            return Node({} if is_object else [], start, index + 1, children)
        while True:
            if is_object:
                key, index = self.decoder.raw_decode(self.text, index)
                if not isinstance(key, str) or key in children:
                    raise ValueError("key")
                index = self.ws(index)
                if index >= len(self.text) or self.text[index] != ":":
                    raise ValueError("colon")
                child = self.node(index + 1, depth + 1)
                children[key] = child
            else:
                child = self.node(index, depth + 1)
                children.append(child)
            index = self.ws(child.end)
            if index >= len(self.text):
                raise ValueError("unclosed")
            if self.text[index] == close:
                value = ({k: v.value for k, v in children.items()} if is_object
                         else [v.value for v in children])
                return Node(value, start, index + 1, children)
            if self.text[index] != ",":
                raise ValueError("comma")
            index = self.ws(index + 1)

    def page(self):
        root = self.node()
        if self.ws(root.end) != len(self.text) or not isinstance(root.value, dict):
            raise ValueError("root")
        actions = root.children.get("corporate_actions")
        token = root.children.get("next_page_token")
        if actions is None or not isinstance(actions.value, dict) or token is None:
            raise CaptureError("UNRESOLVED_RESPONSE_ENVELOPE")
        if token.value is not None and (not isinstance(token.value, str) or not token.value
                                       or len(token.value) > 4096):
            raise CaptureError("UNRESOLVED_PAGINATION_SCHEMA")
        events = []
        for category, array in actions.children.items():
            if not isinstance(array.value, list):
                raise CaptureError("UNRESOLVED_EVENT_ARRAY_SCHEMA")
            for event in array.children:
                if not isinstance(event.value, dict):
                    raise CaptureError("UNRESOLVED_EVENT_SCHEMA")
                events.append((category, event, self.text[event.start:event.end].encode("utf-8")))
        return events, token.value


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def capture(transport, settings, output_base, capture_id, *, clock=time.monotonic, now=utc_now):
    settings.validate()
    root = destination(output_base, capture_id)
    started = clock()
    root.mkdir(parents=True, exist_ok=False)
    bindings, receipts, events, duplicates, chain_results = [], [], [], [], []
    get_count = body_bytes = 0
    incomplete_transfer = False
    primary = None
    recording_error = None
    ids = {}

    def check_time():
        if clock() - started >= settings.acceptance_seconds:
            raise CaptureError("LOCAL_ACCEPTANCE_DEADLINE")

    try:
        source_bytes = Path(__file__).read_bytes()
        frozen = {"capture_id": capture_id, "endpoint": ENDPOINT, "chains": settings.chains(),
                  "collector_source": {"bytes": len(source_bytes), "sha256": digest(source_bytes)},
                  "runtime": {"implementation": sys.implementation.name,
                              "python_version": sys.version.split()[0], "sdk": None},
                  "identifier_bindings": [vars(b) for b in settings.identifiers],
                  "identifier_manifest_sha256": settings.identifier_manifest_sha256,
                  "identifier_evidence_adjudication": "OWNER_SUPPLIED_NOT_INDEPENDENTLY_ADJUDICATED",
                  "limits": {k: getattr(settings, k) for k in (
                      "max_gets", "max_pages", "response_bytes", "aggregate_bytes",
                      "acceptance_seconds", "request_seconds")},
                  "transport_identity": transport.identity,
                  "acquisition_specification_sha256": SPEC_SHA256,
                  "response_schema": "PROVISIONAL_UNQUALIFIED",
                  "query_date_field": "process_date", "qualified_for_research": False}
        bindings.append(save_json(root, "frozen-request.json", frozen))
        for chain, fixed_params in settings.chains():
            token = None
            seen_tokens = set()
            terminated = False
            for page in range(1, settings.max_pages + 1):
                check_time()
                if get_count >= settings.max_gets:
                    raise CaptureError("GET_LIMIT")
                remaining = settings.aggregate_bytes - body_bytes
                if remaining <= 0:
                    raise CaptureError("AGGREGATE_BYTE_LIMIT")
                params = dict(fixed_params)
                if token is not None:
                    params["page_token"] = token
                receipt = {"ordinal": get_count + 1, "chain": chain, "page": page,
                           "endpoint": ENDPOINT, "parameters": params,
                           "started_utc": now(), "status": None, "response": None,
                           "transfer_complete": False}
                receipts.append(receipt)
                get_count += 1
                response = transport.get(dict(params), min(settings.request_seconds,
                                        settings.acceptance_seconds - (clock() - started)),
                                         min(settings.response_bytes, remaining))
                receipt["ended_utc"] = now()
                if (not isinstance(response, Response) or type(response.status) is not int
                        or not isinstance(response.body, bytes) or type(response.complete) is not bool):
                    raise CaptureError("INVALID_TRANSPORT_RESPONSE")
                receipt["status"] = response.status
                cap = min(settings.response_bytes, remaining)
                complete = response.complete and len(response.body) <= cap
                retained = response.body[:cap]
                # A provider echo of authentication cannot be saved as raw evidence.
                if transport.contains_secret(retained):
                    receipt["evidence_withheld"] = "AUTHENTICATION_ECHO"
                    incomplete_transfer = True
                    raise CaptureError("AUTHENTICATION_ECHO")
                binding = save_bytes(root, f"response-{get_count:02d}.json", retained)
                bindings.append(binding)
                receipt["response"] = binding
                receipt["transfer_complete"] = complete
                if complete:
                    body_bytes += len(retained)
                else:
                    incomplete_transfer = True
                    raise CaptureError("RESPONSE_OR_AGGREGATE_BYTE_LIMIT")
                check_time()
                if not 200 <= response.status < 300:
                    raise CaptureError("HTTP_STATUS_FAILURE")
                if response.content_encoding not in ("identity", ""):
                    raise CaptureError("UNSUPPORTED_CONTENT_ENCODING")
                try:
                    page_events, next_token = RawJson(retained).page()
                except CaptureError:
                    raise
                except (ValueError, UnicodeError, RecursionError):
                    raise CaptureError("MALFORMED_RESPONSE_JSON") from None
                # Preserve every event on this page before deriving identities or summaries.
                saved_events = []
                for index, (category, event, raw) in enumerate(page_events):
                    check_time()
                    event_binding = save_bytes(root, f"event-{get_count:02d}-{index:04d}.json", raw)
                    bindings.append(event_binding)
                    saved_events.append((category, event, event_binding))
                for index, (category, event, event_binding) in enumerate(saved_events):
                    check_time()
                    event_id = event.value.get("id")
                    record = {"chain": chain, "page": page, "response_ordinal": get_count,
                              "event_index": index, "category": category,
                              "event_id": event_id if isinstance(event_id, str) else None,
                              "payload": event_binding, "response": binding,
                              "fields_present": sorted(event.value),
                              "null_fields": sorted(k for k, v in event.value.items() if v is None)}
                    events.append(record)
                    if not isinstance(event_id, str) or not event_id or len(event_id) > 256:
                        raise CaptureError("UNRESOLVED_EVENT_ID_SCHEMA")
                    if event_id in ids:
                        previous = ids[event_id]
                        identical = (previous["payload"]["sha256"] == event_binding["sha256"]
                                     and previous["category"] == category)
                        duplicates.append({"event_id": event_id, "identical_bytes": identical,
                                           "first": previous, "variant": record})
                        if not identical:
                            raise CaptureError("CONFLICTING_EVENT_ID_VARIANTS")
                    else:
                        ids[event_id] = record
                receipt["next_page_token"] = next_token
                if next_token is None:
                    chain_results.append({"chain": chain, "pages": page, "terminal_observed": True})
                    terminated = True
                    break
                if next_token in seen_tokens:
                    raise CaptureError("REPEATED_PAGE_TOKEN")
                seen_tokens.add(next_token)
                token = next_token
            if not terminated:
                raise CaptureError("CHAIN_PAGE_LIMIT")
        check_time()
    except Exception as error:
        primary = error_code(error, "UNEXPECTED_LOCAL_FAILURE")
        if receipts and "ended_utc" not in receipts[-1]:
            receipts[-1]["ended_utc"] = utc_now()
        incomplete_transfer = incomplete_transfer or any(not r["transfer_complete"] for r in receipts)
    try:
        bindings.append(save_json(root, "request-receipts.json", receipts))
        bindings.append(save_json(root, "event-inventory.json", {"events": events,
                                  "duplicates": duplicates, "unique_ids": len(ids)}))
        if primary:
            bindings.append(save_json(root, "failure.json", {"code": primary,
                            "qualified_for_research": False, "usable_action_ledger": False}))
        verify_bindings(root, bindings)
        if not primary:
            check_time()
    except Exception as error:
        recording_error = error_code(error, "RECORDING_FAILURE")
        if primary is None:
            primary = recording_error
    manifest = {"capture_id": capture_id, "status": "FAILED" if primary else "COLLECTED_UNQUALIFIED",
                "collection_complete": primary is None, "qualified_for_research": False,
                "action_ledger_qualified": False, "historical_completeness_established": False,
                "primary_error": primary, "recording_error": recording_error,
                "get_attempts": get_count, "completed_body_bytes": body_bytes,
                "incomplete_transfer_bytes": "UNKNOWN" if incomplete_transfer else None,
                "elapsed_seconds": clock() - started, "chains": chain_results,
                "event_observations": len(events), "unique_event_ids": len(ids),
                "duplicate_observations": len(duplicates), "limitations": LIMITATIONS,
                "artifacts": bindings, "saved_artifacts_verified": recording_error is None}
    try:
        manifest_binding = save_json(root, "capture-manifest.json", manifest)
        verify_bindings(root, [*bindings, manifest_binding])
        if strict_json((root / "capture-manifest.json").read_bytes()) != json.loads(json_bytes(manifest)):
            raise CaptureError("MANIFEST_READBACK_MISMATCH")
        if not primary:
            check_time()
    except Exception:
        # Partial bytes remain; no rewrite, retry or optimistic success report.
        raise CaptureError("FINAL_RECORDING_OR_ELIGIBILITY_FAILURE") from None
    return {"root": str(root), "exit_code": 1 if primary else 0,
            "manifest": manifest_binding, "status": manifest["status"],
            "qualified_for_research": False}


def load_identifier_bindings(path, expected_sha256):
    with Path(path).open("rb") as handle:
        body = handle.read(65537)
    if len(body) > 65536:
        raise CaptureError("INVALID_IDENTIFIER_MANIFEST")
    if digest(body) != expected_sha256:
        raise CaptureError("IDENTIFIER_MANIFEST_BINDING_MISMATCH")
    try:
        records = strict_json(body)
        if not isinstance(records, list) or not records:
            raise ValueError("population")
        result = tuple(IdentifierBinding(**record) for record in records)
        for binding in result:
            binding.validate()
        return result
    except (TypeError, ValueError, KeyError):
        raise CaptureError("INVALID_IDENTIFIER_MANIFEST") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="inclusive process_date")
    parser.add_argument("--end", required=True, help="inclusive process_date, explicitly frozen")
    parser.add_argument("--symbols", default=",".join(SYMBOLS))
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--identifier-manifest")
    parser.add_argument("--identifier-manifest-sha256")
    args = parser.parse_args(argv)
    try:
        if bool(args.identifier_manifest) != bool(args.identifier_manifest_sha256):
            raise CaptureError("IDENTIFIER_MANIFEST_BINDING_REQUIRED")
        identifiers = (load_identifier_bindings(args.identifier_manifest, args.identifier_manifest_sha256)
                       if args.identifier_manifest else ())
        settings = Settings(args.start, args.end, tuple(args.symbols.split(",")), identifiers,
                            args.identifier_manifest_sha256)
        settings.validate()
        destination(args.output_base, args.capture_id)
        transport = RestTransport(os.environ.get("APCA_API_KEY_ID"),
                                  os.environ.get("APCA_API_SECRET_KEY"))
        result = capture(transport, settings, args.output_base, args.capture_id)
        print(json.dumps(result, sort_keys=True))
        return result["exit_code"]
    except Exception as error:
        code = error_code(error, "LOCAL_PREREQUISITE_OR_RECORDING_FAILURE")
        print(json.dumps({"status": "FAILED", "code": code, "qualified_for_research": False}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
