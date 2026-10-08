"""Normal import; synthetic HTTP responses only, no SDK, DB or provider access."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import socket
import subprocess
import threading

import pytest

from scripts import research_data_capture as collector

pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def no_external_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network/process boundary accessed")
    class ForbiddenSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            forbidden()
    monkeypatch.setattr(socket, "socket", ForbiddenSocket)
    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def settings():
    return collector.Settings("2024-01-01", "2024-01-07", "sip", "raw", "USD", "2026-10-08")


def calendar():
    return [{"date": f"2024-01-{day:02d}", "open": "09:30", "close": "16:00"}
            for day in range(2, 6)]


def bar(day):
    return {"t": f"2024-01-{day:02d}T05:00:00Z", "o": 100, "h": 103,
            "l": 99, "c": 102, "v": 1000, "n": 20, "vw": 101.5}


def bars(days=range(2, 6)):
    return {symbol: [bar(day) for day in days] for symbol in collector.SYMBOLS}


class Provider:
    identity = {"transport": "synthetic", "version": "fixture-v1"}
    def __init__(self, pages=None):
        self.pages = pages if pages is not None else [{"bars": bars(), "next_page_token": None}]
        self.calls = []
        self.responses = []
        self.transform = lambda response: response
        self.after = lambda: None
        self.closed = False

    def get(self, url, params, timeout, byte_limit):
        self.calls.append((url, copy.deepcopy(params), timeout, byte_limit))
        payload = calendar() if url == collector.CALENDAR_URL else self.pages.pop(0)
        if isinstance(payload, Exception):
            raise payload
        if url == collector.DATA_URL:
            payload = {"next_page_token": None, **payload}
        response = collector.Response(200, json.dumps(payload).encode(), dict(params))
        response = self.transform(response)
        self.responses.append(response)
        self.after()
        return response

    def close(self):
        self.closed = True


def run(tmp_path, settings, provider=None, **kwargs):
    provider = provider or Provider()
    result = collector.capture(provider, settings, tmp_path / "jbravo-research-data", "fixture", **kwargs)
    root = Path(result["root"])
    manifest = json.loads((root / "capture-manifest.json").read_text())
    return provider, root, manifest


def failure(root):
    return json.loads((root / "failure.json").read_text())["code"]


def test_multipage_binding_calendar_and_explicit_request(tmp_path, settings):
    provider = Provider([{"bars": bars(range(2, 4)), "next_page_token": "opaque-token"},
                         {"bars": bars(range(4, 6)), "next_page_token": None}])
    provider, root, manifest = run(tmp_path, settings, provider)
    assert manifest["status"] == "CAPTURE_COMPLETE"
    assert manifest["requests_attempted"] == 3
    assert manifest["completed_body_bytes"] == sum(len(r.body) for r in provider.responses)
    assert len(json.loads((root / "dataset.json").read_text())) == 60
    assert list(collector.SYMBOLS) == sorted(set(collector.SYMBOLS))
    assert len(collector.CACHE_SHA256) == 14 and len(collector.SYMBOLS) == 15
    for call in provider.calls[1:]:
        assert {k: v for k, v in call[1].items() if k != "page_token"} == settings.params()
    assert provider.calls[2][1]["page_token"] == "opaque-token"
    receipts = json.loads((root / "request-receipts.json").read_text())
    assert all(r["started_utc"] and r["received_utc"] for r in receipts)
    assert receipts[-1]["raw_body_sha256"] == collector.digest(provider.responses[-1].body)
    assert receipts[-1]["validated_artifact"]["path"] == "page-002.json"
    assert receipts[0]["validated_artifact"]["path"] == "calendar.json"
    assert "opaque-token" not in (root / "request-receipts.json").read_text()
    for binding in manifest["files"]:
        body = (root / binding["path"]).read_bytes()
        assert binding["bytes"] == len(body) and binding["sha256"] == collector.digest(body)
    assert "capture-manifest.json" not in [b["path"] for b in manifest["files"]]


def test_missing_session_reports_holiday_weekend_and_uncertain_eligibility(tmp_path, settings):
    payload = bars()
    payload["NBIS"] = payload["NBIS"][1:]
    _, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    report = json.loads((root / "coverage.json").read_text())
    assert manifest["status"] == "CAPTURE_COMPLETE_WITH_COVERAGE_GAPS"
    assert report["missing_sessions"]["NBIS"] == ["2024-01-02"]
    assert "UNKNOWN" in report["eligibility_on_missing_dates"]
    assert {r["classification"] for r in report["non_sessions"]} == {
        "weekend", "calendar_excluded_weekday_holiday_or_closure"}


@pytest.mark.parametrize("change,code", [
    ("missing_spy", "MISSING_SPY_BENCHMARK_SESSIONS"),
    ("spy_gap", "MISSING_SPY_BENCHMARK_SESSIONS"),
    ("conflict", "CONFLICTING_DUPLICATE"),
    ("order", "PROVIDER_ORDER_CONFLICT"),
    ("unknown_symbol", "UNREQUESTED_SYMBOL_OR_MALFORMED_ROWS"),
    ("negative_volume", "INVALID_OHLCV"),
    ("fractional_volume", "FRACTIONAL_VOLUME_OR_COUNT"),
    ("bad_high", "OHLC_RANGE_CONFLICT"),
    ("naive_timestamp", "INVALID_TIMESTAMP_REPRESENTATION"),
    ("outside_window", "BAR_OUTSIDE_UTC_WINDOW"),
    ("not_daily", "BAR_NOT_AT_EXCHANGE_DAILY_SESSION"),
])
def test_invalid_bars_fail_without_dataset(tmp_path, settings, change, code):
    payload = bars()
    if change == "missing_spy": del payload["SPY"]
    elif change == "spy_gap": payload["SPY"].pop()
    elif change == "conflict": payload["ALK"].insert(1, dict(bar(2), c=101))
    elif change == "order": payload["ALK"].reverse()
    elif change == "unknown_symbol": payload["OTHER"] = [bar(2)]
    elif change == "negative_volume": payload["ALK"][0]["v"] = -1
    elif change == "fractional_volume": payload["ALK"][0]["v"] = 1.5
    elif change == "bad_high": payload["ALK"][0]["h"] = 100
    elif change == "naive_timestamp": payload["ALK"][0]["t"] = "2024-01-02T05:00:00"
    elif change == "outside_window": payload["ALK"][0]["t"] = "2023-12-29T05:00:00Z"
    elif change == "not_daily": payload["ALK"][0]["t"] = "2024-01-02T15:00:00Z"
    _, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    assert manifest["status"] == "FAILED" and failure(root) == code
    assert not (root / "dataset.json").exists()


def test_identical_duplicate_is_reported_once(tmp_path, settings):
    payload = bars()
    payload["ALK"].insert(1, bar(2))
    _, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    assert manifest["status"] == "CAPTURE_COMPLETE"
    assert len(json.loads((root / "dataset.json").read_text())) == 60
    assert json.loads((root / "coverage.json").read_text())["identical_duplicates_removed"] == 1


def test_numeric_duplicate_representation_and_decimal_precision(tmp_path, settings):
    payload = bars()
    payload["ALK"].insert(1, {**bar(2), "o": 100.0})
    _, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    assert manifest["status"] == "CAPTURE_COMPLETE"
    parsed = collector.strict_json(b'{"value":100.000000000000000000000000000001}')
    assert collector.number(parsed["value"]) == "100.000000000000000000000000000001"


def test_repeated_token_and_partial_transport_failure(tmp_path, settings):
    provider = Provider([{"bars": bars(range(2, 4)), "next_page_token": "repeat"},
                         {"bars": bars(range(4, 6)), "next_page_token": "repeat"}])
    _, root, manifest = run(tmp_path, settings, provider)
    assert failure(root) == "REPEATED_PAGE_TOKEN" and manifest["requests_attempted"] == 3
    assert (root / "page-001.json").exists() and not (root / "dataset.json").exists()


def test_partial_failure_retains_first_page_without_raw_error(tmp_path, settings):
    provider = Provider([{"bars": bars(range(2, 4)), "next_page_token": "next"},
                         OSError("credential-shaped synthetic secret")])
    _, root, manifest = run(tmp_path, settings, provider)
    assert failure(root) == "TRANSPORT_ERROR_OSError" and manifest["status"] == "FAILED"
    assert (root / "page-001.json").exists()
    assert not (root / "dataset.json").exists()
    assert all("synthetic secret" not in p.read_text() for p in root.iterdir())


def test_missing_pagination_state_is_not_assumed_terminal(tmp_path, settings):
    provider = Provider()
    def transform(response):
        if "symbols" in response.effective_params:
            response.body = json.dumps({"bars": bars()}).encode()
        return response
    provider.transform = transform
    _, root, manifest = run(tmp_path, settings, provider)
    assert failure(root) == "MISSING_PAGINATION_STATE"
    assert manifest["status"] == "FAILED" and len(provider.calls) == 2


@pytest.mark.parametrize("limits,code", [
    (collector.Limits(requests=1), "REQUEST_LIMIT"),
    (collector.Limits(page_bytes=10), "RESPONSE_BYTE_LIMIT"),
    (collector.Limits(total_bytes=10), "RESPONSE_BYTE_LIMIT"),
])
def test_exhausted_limits(tmp_path, settings, limits, code):
    _, root, manifest = run(tmp_path, settings, limits=limits)
    assert failure(root) == code and manifest["status"] == "FAILED"


def test_elapsed_limit_rejects_late_response(tmp_path, settings):
    now = [0.0]
    provider = Provider()
    provider.after = lambda: now.__setitem__(0, 121.0)
    _, root, manifest = run(tmp_path, settings, provider, clock=lambda: now[0])
    assert failure(root) == "ELAPSED_LIMIT" and len(provider.calls) == 1


def test_wait_deadline_does_not_accept_late_completion(tmp_path, settings):
    release = threading.Event()
    provider = Provider()
    provider.after = lambda: release.wait(2)
    try:
        _, root, manifest = run(tmp_path, settings, provider,
                               limits=collector.Limits(request_seconds=0.02))
        assert failure(root) == "REQUEST_OR_ELAPSED_TIMEOUT"
        assert manifest["status"] == "FAILED" and len(provider.calls) == 1
        before = (root / "capture-manifest.json").read_bytes()
        release.set()
        assert (root / "capture-manifest.json").read_bytes() == before
        assert not (root / "dataset.json").exists()
    finally:
        release.set()


@pytest.mark.parametrize("calendar_data", [[], [dict(calendar()[0], close="08:00")],
                                         [calendar()[0], calendar()[0]]])
def test_invalid_calendar_is_not_a_weekday_fallback(tmp_path, settings, calendar_data):
    provider = Provider()
    provider.transform = lambda response: collector.Response(
        200, json.dumps(calendar_data).encode(), response.effective_params)
    _, root, manifest = run(tmp_path, settings, provider)
    assert manifest["status"] == "FAILED" and len(provider.calls) == 1
    assert not (root / "dataset.json").exists()


@pytest.mark.parametrize("mode,code", [("params", "EFFECTIVE_REQUEST_MISMATCH"),
                                      ("forbidden", "HTTP_STATUS_403"),
                                      ("duplicate_key", "DUPLICATE_JSON_KEY")])
def test_request_and_response_contract_failures(tmp_path, settings, mode, code):
    provider = Provider()
    def transform(response):
        if mode == "params": response.effective_params = {}
        elif mode == "forbidden": response.status = 403
        else: response.body = b'{"bars": {}, "bars": {}}'
        return response
    provider.transform = transform
    _, root, manifest = run(tmp_path, settings, provider)
    assert failure(root) == code and manifest["requests_attempted"] == 1


def test_output_collision_and_invalid_settings_make_no_requests(tmp_path, settings):
    provider = Provider()
    base = tmp_path / "jbravo-research-data"
    (base / "fixture").mkdir(parents=True)
    with pytest.raises(collector.CaptureError, match="OUTPUT_COLLISION"):
        collector.capture(provider, settings, base, "fixture")
    with pytest.raises(collector.CaptureError, match="UNSUPPORTED_CURRENCY"):
        collector.capture(provider, collector.Settings("2024-01-01", "2024-01-07", "sip", "raw", "EUR", "2026-10-08"), base, "other")
    assert not provider.calls


def test_actual_cli_parameter_delivery_and_normal_import(tmp_path, monkeypatch):
    provider = Provider()
    monkeypatch.setenv("APCA_API_KEY_ID", "fixture-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fixture-secret")
    monkeypatch.setattr(collector, "HTTPClient", lambda key, secret: provider)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert collector.main(["--start-date", "2024-01-01", "--end-date", "2024-01-07",
        "--feed", "sip", "--adjustment", "raw", "--currency", "USD", "--asof", "2026-10-08",
        "--capture-id", "cli-fixture"]) == 0
    assert provider.calls[1][1]["feed"] == "sip" and provider.closed


def test_symlink_output_is_rejected_without_requests(tmp_path, settings):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit test symlink creation")
    provider = Provider()
    with pytest.raises(collector.CaptureError, match="OUTPUT_REDIRECTION"):
        collector.capture(provider, settings, link, "fixture")
    assert not provider.calls


def test_traversal_into_source_is_rejected_before_creation_or_request(tmp_path, settings, monkeypatch):
    source = tmp_path / "source"
    monkeypatch.setattr(collector, "REPO", source)
    base = tmp_path / "unused" / ".." / "source"
    assert base.resolve() == source
    provider = Provider()
    with pytest.raises(collector.CaptureError, match="OUTPUT_PARENT_TRAVERSAL"):
        collector.capture(provider, settings, base, "fixture")
    assert not provider.calls
    assert not source.exists() and not (tmp_path / "unused").exists()


def test_ordinary_external_destination_is_allowed(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(collector, "REPO", tmp_path / "source")
    base = tmp_path / "external" / "research"
    provider = Provider()
    result = collector.capture(provider, settings, base, "fixture")
    assert Path(result["root"]) == base / "fixture"
    assert len(provider.calls) == 2
    assert json.loads((base / "fixture" / "capture-manifest.json").read_text())["status"] == "CAPTURE_COMPLETE"
    assert not collector.REPO.exists()


def test_final_destination_inside_source_is_rejected_without_effects(tmp_path, settings, monkeypatch):
    base = tmp_path / "external"
    monkeypatch.setattr(collector, "REPO", base / "fixture")
    provider = Provider()
    with pytest.raises(collector.CaptureError, match="RESEARCH_OUTPUT_INSIDE_SOURCE_TREE"):
        collector.capture(provider, settings, base, "fixture")
    assert not provider.calls and not base.exists()


@pytest.mark.parametrize("redirected", ["base", "destination"])
def test_windows_reparse_attribute_guard_with_inert_metadata(tmp_path, settings, monkeypatch, redirected):
    base = tmp_path / "jbravo-research-data"
    base.mkdir()
    target = base if redirected == "base" else base / "fixture"
    if target != base:
        target.mkdir()
    original = Path.lstat
    def lstat(path):
        actual = original(path)
        return SimpleNamespace(st_mode=actual.st_mode, st_file_attributes=0x400) if path == target else actual
    monkeypatch.setattr(Path, "lstat", lstat)
    provider = Provider()
    with pytest.raises(collector.CaptureError, match="OUTPUT_REDIRECTION"):
        collector.capture(provider, settings, base, "fixture")
    assert not provider.calls
    assert not list(target.iterdir())


def test_http_adapter_actual_prepared_params_and_bounded_stream(monkeypatch):
    import requests
    client = collector.HTTPClient("fixture-key", "fixture-secret")
    class Wire:
        status_code = 200
        def __init__(self, url, params, body):
            self.request = requests.Request("GET", url, params=params).prepare()
            self.raw = io.BytesIO(body)
        def __enter__(self): return self
        def __exit__(self, *args): pass
    observed = []
    def get(url, **kwargs):
        observed.append(kwargs)
        return Wire(url, kwargs["params"], b'[]')
    monkeypatch.setattr(client.session, "get", get)
    result = client.get(collector.CALENDAR_URL, {"start": "2024-01-01"}, 1, 10)
    assert result.body == b'[]' and result.effective_params == {"start": "2024-01-01"}
    assert observed[0]["allow_redirects"] is False and observed[0]["verify"] is True
    assert client.session.trust_env is False
    assert client.session.get_adapter("https://").max_retries.total == 0
    with pytest.raises(collector.CaptureError, match="PAGE_BYTE_LIMIT"):
        client.get(collector.CALENDAR_URL, {}, 1, 2)
    client.close()
