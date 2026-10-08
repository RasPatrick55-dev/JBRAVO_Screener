"""Normal import; synthetic HTTP responses only, no SDK, DB or provider access."""
import copy
from dataclasses import asdict, replace
import io
import json
from decimal import Decimal
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


@pytest.fixture
def supplement():
    return collector.Settings("2026-05-18", "2026-10-07", "sip", "raw", "USD",
                              "2026-10-08", collector.SUPPLEMENT_PROFILE)


SUPPLEMENT_DAYS = ("2026-05-18", "2026-06-02", "2026-10-07")


def supplement_bars(days=SUPPLEMENT_DAYS):
    return {symbol: [{**bar(2), "t": day + "T04:00:00Z"} for day in days]
            for symbol in ("SPY", "VSXY")}


def supplement_provider(pages=None):
    provider = Provider(pages if pages is not None else [{"bars": supplement_bars()}])
    def transform(response):
        if "symbols" not in response.effective_params:
            response.body = collector.json_bytes([
                {"date": day, "open": "09:30", "close": "16:00"}
                for day in SUPPLEMENT_DAYS])
        return response
    provider.transform = transform
    return provider


def test_default_profile_preserves_selection_and_limits(tmp_path, settings):
    _, root, manifest = run(tmp_path, settings)
    selection = json.loads((root / "symbols.json").read_bytes())
    contract = json.loads((root / "request-contract.json").read_bytes())
    assert manifest["profile"] == settings.profile == "default"
    assert manifest["symbols"] == selection["symbols"] == list(collector.SYMBOLS)
    assert selection["selection_source"] == "canonical data/history_cache fourteen filenames plus SPY"
    assert selection["selection_source_sha256"] == collector.CACHE_SHA256
    assert contract["limits"] == manifest["effective_limits"] == asdict(collector.Limits())
    assert not manifest["qualified_for_research"]
    assert_saved_bindings(root, manifest)


def test_supplement_exact_requests_provenance_and_two_pages(tmp_path, supplement):
    provider = supplement_provider([
        {"bars": supplement_bars(SUPPLEMENT_DAYS[:2]), "next_page_token": "next"},
        {"bars": supplement_bars(SUPPLEMENT_DAYS[2:])}])
    _, root, manifest = run(tmp_path, supplement, provider)
    assert manifest["status"] == "CAPTURE_COMPLETE"
    assert manifest["requests_attempted"] == len(provider.calls) == 3
    assert provider.calls[0][1] == {"start": "2026-05-18", "end": "2026-10-07"}
    expected = {"symbols": "SPY,VSXY", "timeframe": "1Day", "sort": "asc",
                "start": "2026-05-18T00:00:00Z", "end": "2026-10-08T00:00:00Z",
                "feed": "sip", "adjustment": "raw", "currency": "USD",
                "asof": "2026-10-08", "limit": 10000}
    assert provider.calls[1][1] == expected
    assert provider.calls[2][1] == {**expected, "page_token": "next"}
    assert all(0 < call[2] <= 15 and call[3] == 4 * 1024 * 1024 for call in provider.calls)
    selection = json.loads((root / "symbols.json").read_bytes())
    contract = json.loads((root / "request-contract.json").read_bytes())
    assert manifest["profile"] == contract["settings"]["profile"] == selection["profile"] == supplement.profile
    assert manifest["symbols"] == selection["symbols"] == ["SPY", "VSXY"]
    assert selection["selection_source_sha256"] == {}
    assert "history_cache" not in json.dumps(selection)
    assert manifest["selection_rationale"] == selection["selection_rationale"]
    assert contract["limits"] == manifest["effective_limits"] == {
        "requests": 3, "page_bytes": 4 * 1024 * 1024, "total_bytes": 8 * 1024 * 1024,
        "elapsed_seconds": 60.0, "request_seconds": 15.0}
    assert len(json.loads((root / "dataset.json").read_bytes())) == 6
    assert manifest["qualified_for_research"] is False
    assert_saved_bindings(root, manifest)


@pytest.mark.parametrize("field,value", [
    ("profile", "unknown"), ("start_date", "2026-05-17"), ("end_date", "2026-10-06"),
    ("feed", "iex"), ("adjustment", "all"), ("currency", "EUR"), ("asof", "2026-10-07")])
def test_incompatible_profile_settings_precede_output_and_requests(tmp_path, supplement, field, value):
    provider = supplement_provider()
    with pytest.raises(collector.CaptureError):
        collector.capture(provider, replace(supplement, **{field: value}), tmp_path / "absent", "fixture")
    assert provider.calls == [] and not (tmp_path / "absent").exists()


@pytest.mark.parametrize("field,value", [
    ("requests", 4), ("page_bytes", 4 * 1024 * 1024 + 1),
    ("total_bytes", 8 * 1024 * 1024 + 1), ("elapsed_seconds", 60.01), ("request_seconds", 15.01)])
def test_supplement_cannot_inherit_or_expand_default_ceilings(tmp_path, supplement, field, value):
    provider = supplement_provider()
    limits = replace(collector.Limits.for_profile(supplement.profile), **{field: value})
    with pytest.raises(collector.CaptureError, match="INVALID_LIMITS"):
        collector.capture(provider, supplement, tmp_path / "absent", "fixture", limits)
    assert provider.calls == [] and not (tmp_path / "absent").exists()


def test_supplement_stops_before_third_bar_page(tmp_path, supplement):
    provider = supplement_provider([
        {"bars": supplement_bars(SUPPLEMENT_DAYS[:1]), "next_page_token": "first"},
        {"bars": supplement_bars(SUPPLEMENT_DAYS[1:2]), "next_page_token": "second"},
        {"bars": supplement_bars(SUPPLEMENT_DAYS[2:])}])
    _, root, manifest = run(tmp_path, supplement, provider)
    assert len(provider.calls) == manifest["requests_attempted"] == 3
    assert len(provider.pages) == 1 and failure(root) == "REQUEST_LIMIT"
    assert not (root / "dataset.json").exists()
    assert_saved_bindings(root, manifest)


def test_supplement_aggregate_capacity_includes_calendar(tmp_path, supplement):
    provider = supplement_provider([
        {"bars": supplement_bars(SUPPLEMENT_DAYS[:1]), "next_page_token": "next"},
        {"bars": supplement_bars(SUPPLEMENT_DAYS[1:])}])
    previous = provider.transform
    def padded(response):
        response = previous(response)
        if "symbols" in response.effective_params:
            # Whitespace is valid JSON, but still consumes decoded-body budget.
            response.body += b" " * (4 * 1024 * 1024 - 1 - len(response.body))
        return response
    provider.transform = padded
    _, root, manifest = run(tmp_path, supplement, provider)
    assert failure(root) == "RESPONSE_BYTE_LIMIT" and manifest["status"] == "FAILED"
    assert provider.calls[-1][3] == 8 * 1024 * 1024 - sum(len(r.body) for r in provider.responses[:-1])
    assert provider.calls[-1][3] < 4 * 1024 * 1024
    assert not (root / "dataset.json").exists()
    assert_saved_bindings(root, manifest)


def test_supplement_sixty_second_deadline_and_reduced_wait(tmp_path, supplement):
    now = [0.0]
    provider = supplement_provider()
    provider.after = lambda: now.__setitem__(0, 60.0)
    _, root, manifest = run(tmp_path, supplement, provider, clock=lambda: now[0])
    assert failure(root) == "ELAPSED_LIMIT" and len(provider.calls) == 1
    assert manifest["elapsed_seconds"] == 60.0
    assert_saved_bindings(root, manifest)


def test_supplement_can_reduce_limits(tmp_path, supplement):
    limits = collector.Limits(3, 4096, 8192, 30, 2)
    provider, root, manifest = run(tmp_path, supplement, supplement_provider(), limits=limits)
    assert manifest["status"] == "CAPTURE_COMPLETE"
    assert manifest["effective_limits"] == asdict(limits)
    assert all(call[2] <= 2 and call[3] <= 4096 for call in provider.calls)
    assert_saved_bindings(root, manifest)


@pytest.mark.parametrize("mode,code", [("quality", None), ("invalid", "OHLC_RANGE_CONFLICT"),
                                      ("missing-spy", "MISSING_SPY_BENCHMARK_SESSIONS"), ("foreign", "UNREQUESTED_SYMBOL_OR_MALFORMED_ROWS")])
def test_supplement_preserves_quality_and_failures(tmp_path, supplement, mode, code):
    payload = supplement_bars()
    if mode == "quality":
        payload["VSXY"][0].update(v=0, vw=0)
    elif mode == "invalid":
        payload["VSXY"][0]["h"] = 90
    elif mode == "missing-spy":
        del payload["SPY"]
    else:
        payload["VSCO"] = payload["VSXY"]
    _, root, manifest = run(tmp_path, supplement, supplement_provider([{"bars": payload}]))
    assert manifest["qualified_for_research"] is False
    if code is not None:
        assert manifest["status"] == "FAILED" and failure(root) == code
        assert not (root / "dataset.json").exists()
        if mode == "invalid":
            diagnostic = json.loads((root / "bar-validation-diagnostic.json").read_bytes())
            assert diagnostic["symbol"] == "VSXY" and diagnostic["field"] == "h"
    else:
        assert manifest["status"] == "CAPTURE_COMPLETE_UNQUALIFIED_DATA"
        quality = json.loads((root / "quality-summary.json").read_bytes())
        assert quality["flagged_bars"] == 1 and quality["total_findings"] == 2
        assert quality["findings"][0]["symbol"] == "VSXY"
    assert_saved_bindings(root, manifest)


def supplement_cli():
    return ["--profile", collector.SUPPLEMENT_PROFILE, "--start-date", "2026-05-18",
            "--end-date", "2026-10-07", "--feed", "sip", "--adjustment", "raw",
            "--currency", "USD", "--asof", "2026-10-08", "--capture-id", "cli"]


def test_supplement_exclusive_end_remains_enforced(tmp_path, supplement):
    payload = supplement_bars()
    payload["VSXY"][-1]["t"] = "2026-10-08T00:00:00Z"
    _, root, manifest = run(tmp_path, supplement, supplement_provider([{"bars": payload}]))
    assert failure(root) == "BAR_OUTSIDE_UTC_WINDOW" and manifest["status"] == "FAILED"
    assert_saved_bindings(root, manifest)


def test_actual_cli_selects_supplement(tmp_path, monkeypatch):
    provider = supplement_provider()
    monkeypatch.setenv("APCA_API_KEY_ID", "synthetic-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "synthetic-secret")
    monkeypatch.setattr(collector, "HTTPClient", lambda *args: provider)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert collector.main(supplement_cli()) == 0 and provider.closed
    manifest = json.loads((tmp_path / "jbravo-research-data" / "cli" / "capture-manifest.json").read_bytes())
    assert manifest["symbols"] == ["SPY", "VSXY"] and manifest["effective_limits"]["requests"] == 3
    assert not manifest["qualified_for_research"]


@pytest.mark.parametrize("option,value", [("--profile", "unknown"), ("--feed", "iex")])
def test_cli_rejects_profile_mismatch_before_client_or_output(tmp_path, monkeypatch, option, value):
    def forbidden(*args):
        raise AssertionError("Client constructed for invalid profile")
    monkeypatch.setattr(collector, "HTTPClient", forbidden)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    args = supplement_cli()
    args[args.index(option) + 1] = value
    if option == "--profile":
        with pytest.raises(SystemExit) as caught:
            collector.main(args)
        assert caught.value.code == 2
    else:
        assert collector.main(args) == 1
    assert not (tmp_path / "jbravo-research-data").exists()


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
    assert "diagnostic_recording" not in manifest
    assert not (root / "bar-validation-diagnostic.json").exists()


def assert_saved_bindings(root, manifest):
    assert {p.name for p in root.iterdir()} == {
        "capture-manifest.json", *(binding["path"] for binding in manifest["files"])}
    for binding in manifest["files"]:
        body = (root / binding["path"]).read_bytes()
        assert binding["bytes"] == len(body)
        assert binding["sha256"] == collector.digest(body)


@pytest.mark.parametrize("field,value,rule,code,kind,rendered", [
    ("c", 0, "strictly_positive", "INVALID_OHLCV", "int", "0"),
    ("v", -1, "nonnegative", "INVALID_OHLCV", "int", "-1"),
    ("n", 1.5, "integral", "FRACTIONAL_VOLUME_OR_COUNT", "Decimal", "1.5"),
    ("vw", -1, "nonnegative", "INVALID_OHLCV", "int", "-1"),
    ("h", 100, "high_at_or_above_open_and_close", "OHLC_RANGE_CONFLICT", "int", "100"),
    ("l", 101, "low_at_or_below_open_and_close", "OHLC_RANGE_CONFLICT", "int", "101"),
    ("t", "2024-01-02T05:00:00", "aware_iso8601_microsecond_precision",
     "INVALID_TIMESTAMP_REPRESENTATION", "str", "2024-01-02T05:00:00"),
])
def test_bar_diagnostic_distinguishes_field_rule_and_body_binding(
        tmp_path, settings, field, value, rule, code, kind, rendered):
    payload = bars()
    payload["ALK"][0][field] = value
    provider, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    diagnostic_body = (root / "bar-validation-diagnostic.json").read_bytes()
    diagnostic = json.loads(diagnostic_body)
    assert len(diagnostic_body) <= 4096
    assert diagnostic["schema"] == "jbravo.bar-validation-diagnostic.v1"
    assert diagnostic["primary_error_code"] == code == failure(root)
    assert diagnostic["usable_dataset"] is False
    assert (diagnostic["field"], diagnostic["validation_rule"]) == (field, rule)
    assert diagnostic["offending_value"] == {
        "type": kind, "value": rendered, "characters": len(rendered), "truncated": False}
    assert diagnostic["request_ordinal"] == 2 and diagnostic["page_ordinal"] == 1
    assert diagnostic["symbol"] == "ALK" and diagnostic["bar_index"] == 0
    assert diagnostic["bar_index_basis"] == "zero_based_within_symbol_page"
    assert diagnostic["supplied_timestamp"]["value"] == payload["ALK"][0]["t"]
    assert diagnostic["response_body_binding"] == {
        "bytes": len(provider.responses[-1].body),
        "sha256": collector.digest(provider.responses[-1].body)}
    assert manifest["status"] == "FAILED" and len(provider.calls) == 2
    assert manifest["diagnostic_recording"]["status"] == "RECORDED"
    assert not (root / "dataset.json").exists() and not (root / "page-001.json").exists()
    assert_saved_bindings(root, manifest)


def test_second_page_failure_retains_validated_page_and_exact_location(tmp_path, settings):
    second = bars(range(4, 6))
    second["SPY"][1]["v"] = -2
    provider = Provider([{"bars": bars(range(2, 4)), "next_page_token": "next"},
                         {"bars": second}])
    provider, root, manifest = run(tmp_path, settings, provider)
    diagnostic = json.loads((root / "bar-validation-diagnostic.json").read_bytes())
    assert (diagnostic["request_ordinal"], diagnostic["page_ordinal"]) == (3, 2)
    assert (diagnostic["symbol"], diagnostic["bar_index"]) == ("SPY", 1)
    assert diagnostic["supplied_timestamp"]["value"] == bar(5)["t"]
    assert diagnostic["response_body_binding"]["sha256"] == collector.digest(provider.responses[2].body)
    assert (root / "page-001.json").exists() and not (root / "page-002.json").exists()
    assert manifest["requests_attempted"] == 3 and manifest["status"] == "FAILED"
    assert not (root / "dataset.json").exists()
    assert_saved_bindings(root, manifest)


@pytest.mark.parametrize("value,rendered,kind", [
    ("9" * 2000, "9" * 128, "str"),
    ("APCA-API-SECRET-KEY=synthetic-credential", "OMITTED_NON_NUMERIC_TEXT", "str"),
    ({"authorization": "synthetic-credential"}, "OMITTED_NON_SCALAR", "dict"),
    (None, "null", "NoneType"),
    (True, "true", "bool"),
])
def test_bar_diagnostic_bounds_values_without_leaking_arbitrary_text(
        tmp_path, settings, value, rendered, kind):
    payload = bars()
    payload["ALK"][0]["c"] = value
    _, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    body = (root / "bar-validation-diagnostic.json").read_bytes()
    diagnostic = json.loads(body)
    assert len(body) <= 4096
    assert diagnostic["offending_value"]["value"] == rendered
    assert diagnostic["offending_value"]["type"] == kind
    assert diagnostic["offending_value"]["truncated"] is (kind in ("str", "dict"))
    assert diagnostic["primary_error_code"] == "INVALID_OHLCV_TYPE"
    assert diagnostic["validation_rule"] == "numeric_type_excluding_bool"
    assert all("synthetic-credential" not in p.read_text() and
               "APCA-API-SECRET-KEY" not in p.read_text() for p in root.iterdir())
    assert manifest["status"] == "FAILED"
    assert_saved_bindings(root, manifest)


@pytest.mark.parametrize("partial", [False, True])
def test_diagnostic_recording_failure_preserves_primary_failure_and_bindings(
        tmp_path, settings, monkeypatch, partial):
    payload = bars()
    payload["ALK"][0]["v"] = -1
    original_save = collector.save
    attempts = []
    def save(root, name, value):
        if name == "bar-validation-diagnostic.json":
            attempts.append(name)
            # The primary outcome must already have been saved.
            assert failure(root) == "INVALID_OHLCV"
            if partial:
                with (root / name).open("xb") as handle:
                    handle.write(b'{"schema":')
            raise OSError("synthetic-credential in diagnostic write exception")
        return original_save(root, name, value)
    monkeypatch.setattr(collector, "save", save)
    provider = Provider([{"bars": payload}])
    result = collector.capture(provider, settings, tmp_path / "jbravo-research-data", "fixture")
    root = Path(result["root"])
    manifest_body = (root / "capture-manifest.json").read_bytes()
    manifest = json.loads(manifest_body)
    assert result["manifest"]["bytes"] == len(manifest_body)
    assert result["manifest"]["sha256"] == collector.digest(manifest_body)
    assert manifest["status"] == result["status"] == "FAILED"
    assert json.loads((root / "failure.json").read_bytes()) == {
        "code": "INVALID_OHLCV", "usable_dataset": False}
    assert len(provider.calls) == 2 and len(attempts) == 1
    assert manifest["diagnostic_recording"] == {
        "status": "FAILED", "path": "bar-validation-diagnostic.json", "error_type": "OSError",
        "unbound_partial_possible": not partial}
    if partial:
        binding = next(b for b in manifest["files"] if b["path"] == "bar-validation-diagnostic.json")
        assert binding["diagnostic_status"] == "PARTIAL_UNVERIFIED"
    else:
        assert not (root / "bar-validation-diagnostic.json").exists()
    assert all("synthetic-credential" not in p.read_text() for p in root.iterdir())
    assert not (root / "dataset.json").exists()
    assert_saved_bindings(root, manifest)


def test_missing_field_and_malformed_bar_are_diagnosed_without_coercion(tmp_path, settings):
    payload = bars()
    del payload["ALK"][0]["v"]
    _, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    diagnostic = json.loads((root / "bar-validation-diagnostic.json").read_bytes())
    assert (diagnostic["field"], diagnostic["validation_rule"]) == ("v", "required_field_present")
    assert failure(root) == "MALFORMED_BAR" and manifest["status"] == "FAILED"
    with pytest.raises(collector.BarValidationError) as caught:
        collector.parse_bar("ALK", ["not a bar"], settings, {r["date"] for r in calendar()})
    assert caught.value.diagnostic["offending_value"]["value"] == "OMITTED_NON_SCALAR"


@pytest.mark.parametrize("mode,field,rule", [
    ("conflict", "c", "identical_duplicate"),
    ("order", "t", "increasing_new_bar_timestamp"),
])
def test_duplicate_and_order_failures_retain_field_diagnostics(tmp_path, settings, mode, field, rule):
    payload = bars()
    if mode == "conflict":
        payload["ALK"].insert(1, dict(bar(2), c=101))
    else:
        payload["ALK"].reverse()
    _, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    diagnostic = json.loads((root / "bar-validation-diagnostic.json").read_bytes())
    assert (diagnostic["field"], diagnostic["validation_rule"], diagnostic["bar_index"]) == (field, rule, 1)
    assert diagnostic["primary_error_code"] == failure(root)
    assert manifest["status"] == "FAILED"
    assert_saved_bindings(root, manifest)


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


@pytest.mark.parametrize("value,projection", [(0, "0"), (0.0, "0"),
                                               (Decimal("0.000"), "0"),
                                               (Decimal("-0.0"), "-0")])
def test_zero_vwap_preserves_exact_numeric_value_without_inventing_price(settings, value, projection):
    supplied = {**bar(2), "vw": value}
    record = collector.parse_bar("NBIS", supplied, settings, {r["date"] for r in calendar()})
    assert record["vw"] == projection and Decimal(record["vw"]) == Decimal(str(value))
    assert record["quality_flags"] == ["ZERO_VWAP_UNQUALIFIED"]
    assert supplied["vw"] is value
    assert record["o"] == "100" and record["c"] == "102"


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), float("-inf"),
                                   "0", "bad", None, True, {}])
def test_invalid_vwap_remains_rejected(settings, value):
    with pytest.raises(collector.BarValidationError) as caught:
        collector.parse_bar("ALK", {**bar(2), "vw": value}, settings,
                            {r["date"] for r in calendar()})
    assert caught.value.diagnostic["field"] == "vw"


@pytest.mark.parametrize("field,code,value", [("vw", "ZERO_VWAP_UNQUALIFIED", "0"),
                                              ("v", "ZERO_VOLUME_UNQUALIFIED", 0)])
def test_quality_findings_bind_data_source_counts_and_manifest(tmp_path, settings, field, code, value):
    payload = bars()
    payload["NBIS"][0][field] = 0
    provider, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    assert manifest["status"] == "CAPTURE_COMPLETE_UNQUALIFIED_DATA"
    assert manifest["quality_disposition"] == "UNQUALIFIED_DATA"
    assert manifest["qualified_for_research"] is False
    dataset = json.loads((root / "dataset.json").read_bytes())
    record = next(r for r in dataset if r["symbol"] == "NBIS" and r["session"] == "2024-01-02")
    assert record[field] == value and record["quality_flags"] == [code]
    quality = json.loads((root / "quality-summary.json").read_bytes())
    assert quality["counts_by_code"][code] == quality["counts_by_field"][field] == 1
    assert quality["total_findings"] == quality["flagged_bars"] == 1
    assert quality["unique_bars_inspected"] == 60
    assert quality["omitted_findings"] == 0 and not quality["sample_truncated"]
    finding = quality["findings"][0]
    assert (finding["code"], finding["field"], finding["symbol"], finding["bar_index"]) == (code, field, "NBIS", 0)
    assert (finding["request_ordinal"], finding["page_ordinal"]) == (2, 1)
    assert finding["supplied_value"]["value"] == "0"
    assert finding["supplied_timestamp"]["value"] == bar(2)["t"]
    assert finding["response_body_binding"] == {
        "bytes": len(provider.responses[1].body), "sha256": collector.digest(provider.responses[1].body)}
    assert len(collector.json_bytes(finding)) <= 4096
    assert manifest["quality_summary"] in manifest["files"]
    assert not (root / "failure.json").exists()
    assert_saved_bindings(root, manifest)


def test_positive_and_absent_optional_vwap_do_not_claim_research_readiness(tmp_path, settings):
    payload = bars()
    del payload["ALK"][0]["vw"]
    _, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    assert manifest["status"] == "CAPTURE_COMPLETE"
    assert manifest["quality_disposition"] == "NO_LISTED_ANOMALIES_OBSERVED"
    assert manifest["qualified_for_research"] is False
    quality = json.loads((root / "quality-summary.json").read_bytes())
    assert quality["total_findings"] == 0 and quality["findings"] == []
    records = json.loads((root / "dataset.json").read_bytes())
    assert "vw" not in records[0]
    assert records[1]["vw"] == "101.5" and "quality_flags" not in records[1]
    assert_saved_bindings(root, manifest)


def test_quality_sample_cap_counts_all_unique_bars_and_preserves_all_flags(tmp_path, settings):
    payload = bars()
    for rows in payload.values():
        for row in rows:
            row.update(v=0, vw=0)
    payload["ALK"].insert(1, copy.deepcopy(payload["ALK"][0]))
    # A coverage gap must not mask the stronger unqualified-data disposition.
    payload["NBIS"].pop()
    _, root, manifest = run(tmp_path, settings, Provider([{"bars": payload}]))
    quality = json.loads((root / "quality-summary.json").read_bytes())
    assert quality["flagged_bars"] == quality["unique_bars_inspected"] == 59
    assert quality["counts_by_code"] == {"ZERO_VWAP_UNQUALIFIED": 59, "ZERO_VOLUME_UNQUALIFIED": 59}
    assert quality["total_findings"] == 118
    assert len(quality["findings"]) == quality["sample_limit"] == 100
    assert quality["omitted_findings"] == 18 and quality["sample_truncated"]
    assert all(len(collector.json_bytes(f)) <= 4096 for f in quality["findings"])
    records = json.loads((root / "dataset.json").read_bytes())
    assert len(records) == 59
    assert all(r["vw"] == "0" and r["v"] == 0 and len(r["quality_flags"]) == 2 for r in records)
    coverage = json.loads((root / "coverage.json").read_bytes())
    assert coverage["identical_duplicates_removed"] == 1
    assert coverage["missing_sessions"]["NBIS"] == ["2024-01-05"]
    assert manifest["status"] == "CAPTURE_COMPLETE_UNQUALIFIED_DATA"
    assert_saved_bindings(root, manifest)


def test_quality_flags_do_not_mask_mandatory_failure(tmp_path, settings):
    first = bars(range(2, 4))
    first["ALK"][0].update(vw=0, v=0)
    second = bars(range(4, 6))
    second["ALK"][0].update(vw=0, c=0)
    provider = Provider([{"bars": first, "next_page_token": "next"}, {"bars": second}])
    _, root, manifest = run(tmp_path, settings, provider)
    assert manifest["status"] == "FAILED" and failure(root) == "INVALID_OHLCV"
    diagnostic = json.loads((root / "bar-validation-diagnostic.json").read_bytes())
    assert (diagnostic["field"], diagnostic["validation_rule"]) == ("c", "strictly_positive")
    assert not (root / "dataset.json").exists()
    quality = json.loads((root / "quality-summary.json").read_bytes())
    assert quality["total_findings"] == 2 and quality["flagged_bars"] == 1
    assert "incomplete" in quality["coverage"]
    assert_saved_bindings(root, manifest)


def test_quality_summary_tampering_fails_final_binding_check(tmp_path, settings, monkeypatch):
    save = collector.save
    def tamper(root, name, value):
        result = save(root, name, value)
        if name == "request-receipts.json":
            (root / "quality-summary.json").write_bytes(b"{}\n")
        return result
    monkeypatch.setattr(collector, "save", tamper)
    with pytest.raises(collector.CaptureError, match="FINAL_ARTIFACT_BINDING_MISMATCH"):
        run(tmp_path, settings)
    assert not (tmp_path / "jbravo-research-data" / "fixture" / "capture-manifest.json").exists()


def test_cli_returns_capture_success_but_explicitly_unqualified_data(tmp_path, settings, monkeypatch):
    payload = bars()
    payload["SPY"][0]["vw"] = 0
    provider = Provider([{"bars": payload}])
    monkeypatch.setenv("APCA_API_KEY_ID", "fixture-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "fixture-secret")
    monkeypatch.setattr(collector, "HTTPClient", lambda key, secret: provider)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert collector.main(["--start-date", settings.start_date, "--end-date", settings.end_date,
        "--feed", "sip", "--adjustment", "raw", "--currency", "USD", "--asof", settings.asof,
        "--capture-id", "cli-quality"]) == 0
    manifest = json.loads((tmp_path / "jbravo-research-data" / "cli-quality" / "capture-manifest.json").read_bytes())
    assert manifest["status"] == "CAPTURE_COMPLETE_UNQUALIFIED_DATA"
    assert manifest["qualified_for_research"] is False and provider.closed
