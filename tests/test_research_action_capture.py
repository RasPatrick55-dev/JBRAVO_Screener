"""Normal module import, synthetic transports and temporary output only."""
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import socket
import subprocess
import urllib.error
import urllib.request

import pytest

from scripts import research_action_capture as collector

pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def block_external(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External network/process forbidden")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(collector.os, "system", forbidden)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", forbidden)


@pytest.fixture
def settings():
    return collector.Settings("2022-01-01", "2026-10-08")


def page(events=(), token=None):
    return json.dumps({"corporate_actions": {"cash_dividends": list(events)},
                       "next_page_token": token}).encode()


def event(identifier="one", **extra):
    return {"id": identifier, "symbol": "SPY", "process_date": "2024-01-05", **extra}


class Transport:
    identity = {"client": "synthetic-fixture"}
    def __init__(self, pages=None):
        self.pages = list(pages if pages is not None else [page([event()])])
        self.calls = []
        self.after = lambda: None
        self.secret = b"synthetic-secret-never-save"

    def contains_secret(self, body):
        return self.secret in body

    def get(self, params, timeout, byte_limit):
        self.calls.append((dict(params), timeout, byte_limit))
        payload = self.pages.pop(0)
        self.after()
        # Mutating this call's copy must not affect frozen pagination parameters.
        params["start"] = "1900-01-01"
        if isinstance(payload, Exception):
            raise payload
        return payload if isinstance(payload, collector.Response) else collector.Response(200, payload)


def run(tmp_path, settings, transport=None, **kwargs):
    transport = transport or Transport()
    result = collector.capture(transport, settings, tmp_path / "research", "fixture", **kwargs)
    root = Path(result["root"])
    manifest = json.loads((root / "capture-manifest.json").read_bytes())
    collector.verify_bindings(root, [*manifest["artifacts"], result["manifest"]])
    return transport, root, manifest


def fail_code(manifest):
    assert manifest["status"] == "FAILED"
    assert manifest["qualified_for_research"] is False
    return manifest["primary_error"]


def test_import_is_inert(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Import performed I/O")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", forbidden)
        patch.setattr(Path, "mkdir", forbidden)
        patch.setattr(collector.os.environ, "get", forbidden)
        importlib.reload(collector)


def test_pagination_exact_frozen_queries(tmp_path, settings):
    provider = Transport([page([event("first")], "opaque +/?"), page([event("second")])])
    provider, root, manifest = run(tmp_path, settings, provider)
    assert provider.calls[0][0] == settings.chains()[0][1]
    assert provider.calls[1][0] == {**settings.chains()[0][1], "page_token": "opaque +/?"}
    assert all(timeout <= 15 and cap <= 4 * 1024 * 1024 for _, timeout, cap in provider.calls)
    assert manifest["collection_complete"] is True
    assert manifest["get_attempts"] == 2
    assert manifest["unique_event_ids"] == 2
    assert manifest["action_ledger_qualified"] is False
    assert manifest["historical_completeness_established"] is False
    frozen = json.loads((root / "frozen-request.json").read_bytes())
    assert frozen["query_date_field"] == "process_date"
    assert frozen["response_schema"] == "PROVISIONAL_UNQUALIFIED"
    assert "asof" not in provider.calls[0][0]


def identifier():
    return collector.IdentifierBinding("123456789", "candidate-key", "independent-record:1",
                                       "a" * 64, "2023-01-01", "2026-10-07", True)


def test_optional_identifier_chain_keeps_dates_and_independent_binding(tmp_path, settings):
    settings = replace(settings, identifiers=(identifier(),))
    provider, root, manifest = run(tmp_path, settings, Transport([page(), page()]))
    assert provider.calls[1][0] == {"start": settings.start, "end": settings.end, "region": "all",
                                   "data_quality": "all", "sort": "asc", "limit": 1000,
                                   "cusips": "123456789"}
    assert manifest["get_attempts"] == 2
    assert [c["chain"] for c in manifest["chains"]] == ["symbols", "cusips"]
    assert json.loads((root / "frozen-request.json").read_bytes())["identifier_bindings"]


@pytest.mark.parametrize("change", [{"independently_bound": False}, {"source_sha256": ""},
                                    {"cusip": "SPY"}, {"valid_from": "unknown"}])
def test_unbound_identifiers_rejected_before_output(tmp_path, settings, change):
    provider = Transport()
    with pytest.raises(collector.CaptureError):
        collector.capture(provider, replace(settings, identifiers=(replace(identifier(), **change),)),
                          tmp_path / "research", "fixture")
    assert not provider.calls
    assert not (tmp_path / "research").exists()


def test_identifier_manifest_expected_identity(tmp_path):
    path = tmp_path / "identifiers.json"
    path.write_bytes(collector.json_bytes([vars(identifier())]))
    bindings = collector.load_identifier_bindings(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert bindings == (identifier(),)
    with pytest.raises(collector.CaptureError, match="BINDING_MISMATCH"):
        collector.load_identifier_bindings(path, "0" * 64)


@pytest.mark.parametrize("body", [b'[]', b'null', b'{"not":"a list"}', b'[{"unknown":true}]',
                                   b'[{"cusip":"a","cusip":"b"}]'])
def test_identifier_manifest_rejects_malformed_population(tmp_path, body):
    path = tmp_path / "identifiers.json"
    path.write_bytes(body)
    with pytest.raises(collector.CaptureError, match="INVALID_IDENTIFIER_MANIFEST"):
        collector.load_identifier_bindings(path, collector.digest(body))


def test_duplicate_identifier_bindings_rejected(tmp_path, settings):
    with pytest.raises(collector.CaptureError, match="AMBIGUOUS_IDENTIFIER"):
        collector.capture(Transport(), replace(settings, identifiers=(identifier(), identifier())),
                          tmp_path / "research", "fixture")
    assert not (tmp_path / "research").exists()


def test_cli_rejects_unsupported_bar_parameter_before_output(tmp_path):
    with pytest.raises(SystemExit) as error:
        collector.main(["--start", "2022-01-01", "--end", "2026-10-08", "--asof", "2026-10-08",
                        "--output-base", str(tmp_path / "research"), "--capture-id", "cli"])
    assert error.value.code == 2
    assert not (tmp_path / "research").exists()


def test_exact_event_numeric_and_missing_null_preservation(tmp_path, settings):
    raw = b'{"id":"lexical", "rate":1.2300, "ratio":1e-7, "amount":null, "symbol":"SPY"}'
    body = b'{"corporate_actions":{"unknown_future_type":[' + raw + b']},"next_page_token":null}'
    _, root, manifest = run(tmp_path, settings, Transport([body]))
    assert (root / "response-01.json").read_bytes() == body
    assert (root / "event-01-0000.json").read_bytes() == raw
    inventory = json.loads((root / "event-inventory.json").read_bytes())["events"][0]
    assert inventory["null_fields"] == ["amount"]
    assert "currency" not in inventory["fields_present"]
    assert manifest["qualified_for_research"] is False


def test_every_payload_preserved_before_conflicting_id_stop(tmp_path, settings):
    provider = Transport([page([event("same", rate=1), event("same", rate=2), event("later")])])
    _, root, manifest = run(tmp_path, settings, provider)
    assert fail_code(manifest) == "CONFLICTING_EVENT_ID_VARIANTS"
    assert len(list(root.glob("event-[0-9]*.json"))) == 3
    inventory = json.loads((root / "event-inventory.json").read_bytes())
    assert inventory["duplicates"][0]["identical_bytes"] is False
    assert manifest["get_attempts"] == 1


def test_exact_duplicates_recorded_not_overwritten(tmp_path, settings):
    _, root, manifest = run(tmp_path, settings, Transport([page([event(), event()])]))
    assert manifest["unique_event_ids"] == 1
    assert manifest["duplicate_observations"] == 1
    assert manifest["event_observations"] == 2
    assert json.loads((root / "event-inventory.json").read_bytes())["duplicates"][0]["identical_bytes"]


@pytest.mark.parametrize("raw", [b'{}', b'{"corporate_actions":{}}',
    b'{"corporate_actions":[],"next_page_token":null}',
    b'{"corporate_actions":{"a":{}},"next_page_token":null}',
    b'{"corporate_actions":{"a":[null]},"next_page_token":null}',
    b'{"corporate_actions":{},"next_page_token":1}',
    b'{"corporate_actions":{"a":[{"id":null}]},"next_page_token":null}',
    b'{"corporate_actions":{},"next_page_token":""}'])
def test_unresolved_schema_preserves_raw_failure(tmp_path, settings, raw):
    _, root, manifest = run(tmp_path, settings, Transport([raw]))
    assert fail_code(manifest).startswith("UNRESOLVED_")
    assert (root / "response-01.json").read_bytes() == raw


@pytest.mark.parametrize("raw", [b'not json', b'{"x":1,"x":2}', b'[]', b'\xff',
    b'{"corporate_actions":{"a":[{"id":"a","rate":NaN}]},"next_page_token":null}',
    b'{"corporate_actions":{},"next_page_token":null} trailing',
    b'{"corporate_actions":{},"next_page_token":null,}'])
def test_malformed_preserved_without_derivation(tmp_path, settings, raw):
    _, root, manifest = run(tmp_path, settings, Transport([raw]))
    assert fail_code(manifest) == "MALFORMED_RESPONSE_JSON"
    assert (root / "response-01.json").read_bytes() == raw


def test_empty_terminal_never_claims_history_completeness(tmp_path, settings):
    _, _, manifest = run(tmp_path, settings, Transport([page()]))
    assert manifest["collection_complete"]
    assert manifest["event_observations"] == 0
    assert not manifest["historical_completeness_established"]
    assert not manifest["qualified_for_research"]


def test_repeated_token_stops_without_retry(tmp_path, settings):
    provider, _, manifest = run(tmp_path, settings, Transport([page(token="same"), page(token="same")]))
    assert fail_code(manifest) == "REPEATED_PAGE_TOKEN"
    assert len(provider.calls) == 2


def test_six_pages_maximum_per_chain(tmp_path, settings):
    provider, _, manifest = run(tmp_path, settings, Transport([page(token=str(n)) for n in range(7)]))
    assert fail_code(manifest) == "CHAIN_PAGE_LIMIT"
    assert len(provider.calls) == 6


def test_twelve_get_ceiling_two_chains(tmp_path, settings):
    pages = [page(token=str(n)) for n in range(5)] + [page()]
    provider, _, manifest = run(tmp_path, replace(settings, identifiers=(identifier(),)),
                                Transport(pages + pages))
    assert manifest["collection_complete"]
    assert len(provider.calls) == 12


def test_lower_get_ceiling_stops_before_next_request(tmp_path, settings):
    provider, _, manifest = run(tmp_path, replace(settings, max_gets=1), Transport([page(token="next")]))
    assert fail_code(manifest) == "GET_LIMIT"
    assert len(provider.calls) == 1


@pytest.mark.parametrize("limit_field", ["response_bytes", "aggregate_bytes"])
def test_oversized_body_retains_bounded_prefix_and_unknown_transfer(tmp_path, settings, limit_field):
    provider, root, manifest = run(tmp_path, replace(settings, **{limit_field: 10}), Transport([page()]))
    assert fail_code(manifest) == "RESPONSE_OR_AGGREGATE_BYTE_LIMIT"
    assert len((root / "response-01.json").read_bytes()) == 10
    assert manifest["completed_body_bytes"] == 0
    assert manifest["incomplete_transfer_bytes"] == "UNKNOWN"
    assert len(provider.calls) == 1


def test_aggregate_budget_exhausted_after_complete_page(tmp_path, settings):
    body = page(token="next")
    provider, _, manifest = run(tmp_path, replace(settings, aggregate_bytes=len(body)), Transport([body]))
    assert fail_code(manifest) == "AGGREGATE_BYTE_LIMIT"
    assert manifest["completed_body_bytes"] == len(body)
    assert len(provider.calls) == 1


def test_late_result_is_preserved_but_ineligible(tmp_path, settings):
    tick = [0.0]
    provider = Transport()
    provider.after = lambda: tick.__setitem__(0, 601.0)
    _, root, manifest = run(tmp_path, settings, provider, clock=lambda: tick[0])
    assert fail_code(manifest) == "LOCAL_ACCEPTANCE_DEADLINE"
    assert (root / "response-01.json").exists()
    assert not list(root.glob("event-[0-9]*.json"))


@pytest.mark.parametrize("status", [301, 401, 403, 429, 500])
def test_http_error_no_fallback_or_retry(tmp_path, settings, status):
    provider, root, manifest = run(tmp_path, settings, Transport([collector.Response(status, b'{}')]))
    assert fail_code(manifest) == "HTTP_STATUS_FAILURE"
    assert len(provider.calls) == 1
    assert (root / "response-01.json").read_bytes() == b'{}'


def test_transport_failure_retains_prior_page_and_no_exception_secret(tmp_path, settings):
    provider = Transport([page([event()], "next"), RuntimeError("synthetic-secret-never-save")])
    _, root, manifest = run(tmp_path, settings, provider)
    assert fail_code(manifest) == "UNEXPECTED_LOCAL_FAILURE"
    assert manifest["get_attempts"] == 2
    assert manifest["incomplete_transfer_bytes"] == "UNKNOWN"
    assert (root / "event-01-0000.json").exists()
    assert all(provider.secret not in path.read_bytes() for path in root.iterdir())


def test_custom_transport_error_cannot_inject_secret_code(tmp_path, settings):
    provider = Transport([collector.CaptureError("synthetic-secret-never-save")])
    _, root, manifest = run(tmp_path, settings, provider)
    assert fail_code(manifest) == "UNEXPECTED_LOCAL_FAILURE"
    assert all(provider.secret not in path.read_bytes() for path in root.iterdir())


def test_actual_rest_transport_single_synthetic_response(tmp_path, settings, monkeypatch):
    seen = []
    class HttpResponse:
        code = 200
        headers = {}
        def read(self, size):
            assert size == settings.response_bytes + 1
            return page()
        def close(self):
            seen.append("closed")
    class Opener:
        def open(self, request, timeout):
            seen.append((request.full_url, request.header_items(), timeout, request.get_method()))
            return HttpResponse()
    monkeypatch.setattr(urllib.request, "build_opener", lambda handler: Opener())
    transport = collector.RestTransport("long-fixture-api-key", "long-fixture-api-secret")
    _, root, manifest = run(tmp_path, settings, transport)
    request = next(item for item in seen if isinstance(item, tuple))
    assert request[3] == "GET"
    assert request[2] <= 15
    assert "start=2022-01-01" in request[0]
    assert "closed" in seen
    assert manifest["collection_complete"]
    assert all(b"long-fixture-api-secret" not in path.read_bytes() for path in root.iterdir())


def test_rest_local_timeout_no_retry(tmp_path, settings, monkeypatch):
    # An inert queue fixture closes the local acceptance window without a real
    # delayed network worker or external process.
    class EmptyQueue:
        def __init__(self, **kwargs):
            pass
        def get(self, timeout):
            raise collector.queue.Empty
    class InertThread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setattr(collector.queue, "Queue", EmptyQueue)
    monkeypatch.setattr(collector.threading, "Thread", InertThread)
    _, _, manifest = run(tmp_path, settings, collector.RestTransport("fixture-key", "fixture-secret"))
    assert fail_code(manifest) == "LOCAL_REQUEST_TIMEOUT_REMOTE_STATE_UNKNOWN"
    assert manifest["get_attempts"] == 1
    assert manifest["incomplete_transfer_bytes"] == "UNKNOWN"


def test_reparse_attributes_rejected_without_request(tmp_path, settings, monkeypatch):
    from types import SimpleNamespace
    original = Path.lstat
    base = tmp_path / "research"
    base.mkdir()
    def metadata(path):
        info = original(path)
        if path == base:
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info
    monkeypatch.setattr(Path, "lstat", metadata)
    provider = Transport()
    with pytest.raises(collector.CaptureError, match="CONTAINMENT"):
        collector.capture(provider, settings, base, "fixture")
    assert not provider.calls
    assert list(base.iterdir()) == []


def test_authentication_echo_not_persisted(tmp_path, settings):
    provider = Transport([b'{"echo":"synthetic-secret-never-save"}'])
    _, root, manifest = run(tmp_path, settings, provider)
    assert fail_code(manifest) == "AUTHENTICATION_ECHO"
    assert not (root / "response-01.json").exists()
    assert all(provider.secret not in path.read_bytes() for path in root.iterdir())


def test_recording_failure_does_not_replace_primary(tmp_path, settings, monkeypatch):
    original = collector.save_json
    def failing(root, name, value):
        if name == "request-receipts.json":
            raise OSError("synthetic recorder failure")
        return original(root, name, value)
    monkeypatch.setattr(collector, "save_json", failing)
    _, root, manifest = run(tmp_path, settings, Transport([collector.Response(500, b'{}')]))
    assert fail_code(manifest) == "HTTP_STATUS_FAILURE"
    assert manifest["recording_error"] == "RECORDING_FAILURE"
    assert not manifest["saved_artifacts_verified"]
    assert (root / "response-01.json").exists()


def test_final_manifest_failure_is_unsuccessful_no_retry(tmp_path, settings, monkeypatch):
    original = collector.save_json
    def failing(root, name, value):
        if name == "capture-manifest.json":
            raise OSError("synthetic failure")
        return original(root, name, value)
    monkeypatch.setattr(collector, "save_json", failing)
    with pytest.raises(collector.CaptureError, match="FINAL_RECORDING"):
        collector.capture(Transport(), settings, tmp_path / "research", "fixture")
    assert (tmp_path / "research" / "fixture" / "request-receipts.json").exists()


@pytest.mark.parametrize("suffix", ["inside/../../source", "../source"])
def test_traversal_rejected_without_output_or_request(tmp_path, settings, suffix):
    provider = Transport()
    with pytest.raises(collector.CaptureError, match="UNSAFE_CONTAINMENT"):
        collector.capture(provider, settings, tmp_path / suffix, "fixture")
    assert not provider.calls
    assert list(tmp_path.iterdir()) == []


def test_source_tree_output_rejected(settings):
    provider = Transport()
    with pytest.raises(collector.CaptureError, match="OUTPUT_INSIDE_SOURCE"):
        collector.capture(provider, settings, collector.REPO / "research-actions", "fixture")
    assert not provider.calls


def test_collision_rejected_and_existing_bytes_unchanged(tmp_path, settings):
    target = tmp_path / "research" / "fixture"
    target.mkdir(parents=True)
    (target / "keep").write_bytes(b'unchanged')
    provider = Transport()
    with pytest.raises(collector.CaptureError, match="COLLISION"):
        collector.capture(provider, settings, target.parent, "fixture")
    assert (target / "keep").read_bytes() == b'unchanged'
    assert not provider.calls


def test_symlink_and_final_destination_redirection(tmp_path, settings):
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(ordinary, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not permit creating symlinks")
    provider = Transport()
    with pytest.raises(collector.CaptureError, match="CONTAINMENT"):
        collector.capture(provider, settings, link, "fixture")
    assert not provider.calls


@pytest.mark.parametrize("change", [{"max_gets": 13}, {"max_pages": 7},
    {"response_bytes": 4194305}, {"aggregate_bytes": 33554433},
    {"acceptance_seconds": 601}, {"request_seconds": 16}, {"max_gets": True},
    {"request_seconds": float('nan')}, {"start": "bad"}, {"end": "2020-01-01"},
    {"symbols": ("SPY", "ALK")}, {"symbols": ("SPY", "SPY")}])
def test_invalid_configuration_before_creation(tmp_path, settings, change):
    provider = Transport()
    with pytest.raises(collector.CaptureError):
        collector.capture(provider, replace(settings, **change), tmp_path / "research", "fixture")
    assert not provider.calls
    assert not (tmp_path / "research").exists()


def test_cli_actual_path_with_process_only_fake_auth(tmp_path, monkeypatch, capsys):
    provider = Transport([page()])
    seen = []
    def client(key, secret):
        seen.append((key, secret))
        return provider
    monkeypatch.setenv("APCA_API_KEY_ID", "fixture-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fixture-secret")
    monkeypatch.setattr(collector, "RestTransport", client)
    assert collector.main(["--start", "2022-01-01", "--end", "2026-10-08",
                           "--output-base", str(tmp_path / "research"), "--capture-id", "cli"]) == 0
    assert seen == [("fixture-key", "fixture-secret")]
    output = json.loads(capsys.readouterr().out)
    assert output["qualified_for_research"] is False
    assert "fixture-secret" not in json.dumps(output)


def test_redirect_handler_refuses_and_transport_headers_not_reported():
    assert collector.NoRedirect().redirect_request(None, None, 302, None, {}, "https://other") is None
    client = collector.RestTransport("fixture-key", "fixture-secret")
    assert client.contains_secret(b'fixture-secret')
    assert "fixture-key" not in json.dumps(client.identity)


def test_final_binding_tampering_detected(tmp_path, settings):
    _, root, manifest = run(tmp_path, settings)
    binding = manifest["artifacts"][0]
    (root / binding["path"]).write_bytes(b'altered')
    with pytest.raises(collector.CaptureError, match="MISMATCH"):
        collector.verify_bindings(root, manifest["artifacts"])
