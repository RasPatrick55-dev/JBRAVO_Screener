"""Isolated, explicitly configured research capture; importing performs no I/O.

Only the CLI constructs an authenticated HTTP client. No production modules,
database, trading actions, automatic retries, alias changes or feed fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as daytime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
from zoneinfo import ZoneInfo


SYMBOLS = ("ALK", "FN", "HBAN", "HMY", "LOGI", "MIRM", "NBIS", "NVST",
           "PLUG", "RGLD", "SKY", "SPY", "SRRK", "STLD", "VSCO")
DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
CALENDAR_URL = "https://paper-api.alpaca.markets/v2/calendar"
REPO = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")
# Current read-only cache identities bind selection, not historical provenance.
CACHE_SHA256 = dict(zip((s for s in SYMBOLS if s != "SPY"), (
    "506b31c4c01927362c29a1937d52f50ce5d416994b0fde5f11c6b66759c041a9",
    "5af1873eba065394180175f7e5492dd1c8c14e7e39a5c3c96361f5d7dde6ff99",
    "40ca54d42b99c15eeaf589c25e114a55694abde57da7f0e69184da529eb66f18",
    "d8efcf509a9107fd68a7f0a98e58070d11c390559726f19ed4449b414928909e",
    "07e4b61f67e705f8972ec9a2ec944662f5bd0f6b850deb73ea3824ea92a901a1",
    "39b5ee50fdcad4dbf720957e00af579c64364768b53dcd1befbaddda5d33b94a",
    "3500ed9534ed3edc777cb62152aa35a4a19da2ced915dc5d4032087c77320e46",
    "0e0cfef2f123524e07d3cdcb14180f13bea46b2f26c7fcf8f83e25ee54ffa833",
    "7e95f258f45b35a56ccde2f295c4b7ba39baa7b4172dbe4bec12139fcff2f955",
    "9c439e5480c420d39431ea932ffd72664354b38d975bb5a76264c46b89f98984",
    "1251a8d76e96832f00a7749638e68086d5a8bab71951e1b222c20ed42653633c",
    "d4e3d15dca9c488ec8dccb19515af30054106ea4f44436e4e43bc384d8a4ad5c",
    "59bca70f19f77f09e9d0332ecac6ba32def89729a5326b920de7dbb8fe7da165",
    "6d2f463b8f7fac83adb8ab70a957dab2aa91e61b3a7fcb6bf7c3d14cbc164095")))


class CaptureError(Exception):
    """Codes contain no provider body, authentication material or raw exception."""


def digest(body):
    return hashlib.sha256(body).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def save(root, name, value):
    body = json_bytes(value)
    path = root / name
    with path.open("xb") as handle:
        handle.write(body)
    actual = path.read_bytes()
    if actual != body:
        raise CaptureError("SAVED_BYTE_MISMATCH")
    return {"path": name, "bytes": len(actual), "sha256": digest(actual)}


def iso_day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise CaptureError("INVALID_DATE")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise CaptureError("INVALID_DATE") from None


@dataclass(frozen=True)
class Settings:
    start_date: str
    end_date: str
    feed: str
    adjustment: str
    currency: str
    asof: str

    def validate(self):
        start, end = iso_day(self.start_date), iso_day(self.end_date)
        iso_day(self.asof)
        if start > end or (end - start).days > 3660:
            raise CaptureError("DATE_WINDOW_LIMIT")
        if self.feed not in ("iex", "sip") or self.adjustment not in ("raw", "split", "all"):
            raise CaptureError("UNSUPPORTED_EXPLICIT_SETTINGS")
        if self.currency != "USD":
            raise CaptureError("UNSUPPORTED_CURRENCY")
        return start, end

    def params(self):
        start, end = self.validate()
        return {"symbols": ",".join(SYMBOLS), "timeframe": "1Day",
                "start": start.isoformat() + "T00:00:00Z",
                "end": (end + timedelta(days=1)).isoformat() + "T00:00:00Z",
                "feed": self.feed, "adjustment": self.adjustment,
                "currency": self.currency, "asof": self.asof,
                "sort": "asc", "limit": 10000}


@dataclass(frozen=True)
class Limits:
    requests: int = 41  # one calendar request plus at most forty bar pages
    page_bytes: int = 4 * 1024 * 1024
    total_bytes: int = 32 * 1024 * 1024
    elapsed_seconds: float = 120.0
    request_seconds: float = 15.0

    def validate(self):
        for field, ceiling in (("requests", 41), ("page_bytes", 4 * 1024 * 1024),
                               ("total_bytes", 32 * 1024 * 1024)):
            value = getattr(self, field)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise CaptureError("INVALID_LIMITS")
        for field, ceiling in (("elapsed_seconds", 120), ("request_seconds", 15)):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= ceiling:
                raise CaptureError("INVALID_LIMITS")


@dataclass
class Response:
    status: int
    body: bytes
    effective_params: dict


class HTTPClient:
    """Fixed GET-only endpoints, TLS verification, no redirects/retries/proxies."""
    def __init__(self, key, secret):
        import requests
        from requests.adapters import HTTPAdapter
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.mount("https://", HTTPAdapter(max_retries=0))
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.identity = {"transport": "requests GET, explicit pagination",
                         "requests_version": importlib.metadata.version("requests"),
                         "python_version": sys.version.split()[0], "alpaca_sdk_used": False}

    def get(self, url, params, timeout, byte_limit):
        if url not in (DATA_URL, CALENDAR_URL):
            raise CaptureError("UNAUTHORIZED_ENDPOINT")
        from urllib.parse import parse_qs, urlsplit
        # Prepared URL readback establishes the actual request parameters.
        with self.session.get(url, params=params, timeout=(timeout, timeout),
                              stream=True, allow_redirects=False, verify=True) as response:
            actual = parse_qs(urlsplit(response.request.url).query, keep_blank_values=True)
            if actual != {k: [str(v)] for k, v in params.items()}:
                raise CaptureError("EFFECTIVE_REQUEST_MISMATCH")
            response.raw.decode_content = True
            body = bytearray()
            while len(body) < byte_limit:
                chunk = response.raw.read(min(65536, byte_limit - len(body)))
                if not chunk:
                    return Response(response.status_code, bytes(body), dict(params))
                body.extend(chunk)
            # Refuse an exactly-full stream rather than overread to prove EOF.
            raise CaptureError("PAGE_BYTE_LIMIT_OR_UNPROVEN_EOF")

    def close(self):
        self.session.close()


def sanitized_params(params):
    return {key: ({"sha256": digest(value.encode()), "bytes": len(value.encode())}
                  if key == "page_token" else value) for key, value in params.items()}


def strict_json(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CaptureError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    try:
        return json.loads(body.decode("utf-8"), object_pairs_hook=unique,
                          parse_float=Decimal,
                          parse_constant=lambda _: (_ for _ in ()).throw(CaptureError("NONFINITE_JSON")))
    except (UnicodeError, ValueError):
        raise CaptureError("MALFORMED_RESPONSE") from None


def number(value, integer=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise CaptureError("INVALID_OHLCV_TYPE")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise CaptureError("INVALID_OHLCV") from None
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        raise CaptureError("INVALID_OHLCV")
    if integer and result != result.to_integral_value():
        raise CaptureError("FRACTIONAL_VOLUME_OR_COUNT")
    if integer:
        return int(result)
    text = format(result, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def parse_calendar(payload, start, end):
    if not isinstance(payload, list) or not payload:
        raise CaptureError("MISSING_CALENDAR")
    sessions = {}
    previous = None
    for item in payload:
        if not isinstance(item, dict) or not {"date", "open", "close"} <= item.keys():
            raise CaptureError("MALFORMED_CALENDAR")
        day = iso_day(item["date"])
        try:
            opening, closing = daytime.fromisoformat(item["open"]), daytime.fromisoformat(item["close"])
        except (ValueError, TypeError):
            raise CaptureError("MALFORMED_CALENDAR_TIME") from None
        if opening.tzinfo or closing.tzinfo or opening >= closing or not start <= day <= end or (previous and day <= previous):
            raise CaptureError("INVALID_CALENDAR_ORDER_OR_RANGE")
        sessions[day.isoformat()] = {"date": day.isoformat(), "open": opening.isoformat(),
                                   "close": closing.isoformat(), "timezone": "America/New_York"}
        previous = day
    return sessions


def parse_bar(symbol, bar, settings, sessions):
    if not isinstance(bar, dict) or not {"t", "o", "h", "l", "c", "v"} <= bar.keys():
        raise CaptureError("MALFORMED_BAR")
    raw = bar["t"]
    if not isinstance(raw, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})", raw):
        raise CaptureError("INVALID_TIMESTAMP_REPRESENTATION")
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        raise CaptureError("INVALID_TIMESTAMP") from None
    start, end = settings.validate()
    if not datetime.combine(start, daytime(), timezone.utc) <= stamp < datetime.combine(end + timedelta(days=1), daytime(), timezone.utc):
        raise CaptureError("BAR_OUTSIDE_UTC_WINDOW")
    local = stamp.astimezone(NY)
    day = local.date().isoformat()
    if day not in sessions or local.time() != daytime():
        raise CaptureError("BAR_NOT_AT_EXCHANGE_DAILY_SESSION")
    record = {"symbol": symbol, "session": day, "timestamp_utc": stamp.isoformat(),
              **{key: number(bar[key], positive=True) for key in ("o", "h", "l", "c")},
              "v": number(bar["v"], integer=True)}
    if Decimal(record["l"]) > min(Decimal(record["o"]), Decimal(record["c"])) or Decimal(record["h"]) < max(Decimal(record["o"]), Decimal(record["c"])):
        raise CaptureError("OHLC_RANGE_CONFLICT")
    if "n" in bar:
        record["n"] = number(bar["n"], integer=True)
    if "vw" in bar:
        record["vw"] = number(bar["vw"], positive=True)
    return record


def new_output(base, capture_id):
    base = Path(base)
    # Reject traversal before absolute-path handling can erase its components.
    # Do not resolve symlinks: their original path remains subject to inspection.
    if ".." in base.parts:
        raise CaptureError("OUTPUT_PARENT_TRAVERSAL")
    base = base.absolute()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", capture_id):
        raise CaptureError("INVALID_CAPTURE_ID")
    root = base / capture_id
    for destination in (base, root):
        if destination == REPO or REPO in destination.parents:
            raise CaptureError("RESEARCH_OUTPUT_INSIDE_SOURCE_TREE")
    for ancestor in (root, *root.parents):
        if ancestor.is_symlink():
            raise CaptureError("OUTPUT_REDIRECTION")
        if ancestor.exists() and getattr(ancestor.lstat(), "st_file_attributes", 0) & 0x400:
            raise CaptureError("OUTPUT_REDIRECTION")
    if root.exists() or root.is_symlink():
        raise CaptureError("OUTPUT_COLLISION")
    base.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    return root


def capture(client, settings, research_base, capture_id, limits=Limits(), *, clock=time.monotonic,
            utcnow=lambda: datetime.now(timezone.utc)):
    """Return a saved manifest. Fail closed; partial receipts remain isolated."""
    start, end = settings.validate()
    limits.validate()
    begun = clock()
    deadline = begun + limits.elapsed_seconds
    root = new_output(research_base, capture_id)
    files, ledger = [], []
    byte_count, duplicates = 0, 0
    data, last = {}, {}
    def remaining():
        seconds = deadline - clock()
        if seconds <= 0:
            raise CaptureError("ELAPSED_LIMIT")
        return seconds
    def fetch(url, params):
        nonlocal byte_count
        seconds = remaining()
        if len(ledger) >= limits.requests:
            raise CaptureError("REQUEST_LIMIT")
        capacity = min(limits.page_bytes, limits.total_bytes - byte_count)
        if capacity <= 0:
            raise CaptureError("TOTAL_BYTE_LIMIT")
        receipt = {"ordinal": len(ledger) + 1, "url": url,
                   "effective_request": sanitized_params(params), "started_utc": utcnow().isoformat(),
                   "disposition": "SUBMITTED"}
        ledger.append(receipt)
        result = queue.Queue(maxsize=1)
        timeout = min(limits.request_seconds, seconds)
        def operation():
            try:
                result.put((True, client.get(url, dict(params), timeout, capacity)))
            except Exception as exc:
                result.put((False, exc))
        threading.Thread(target=operation, daemon=True).start()
        try:
            ok, response = result.get(timeout=timeout)
        except queue.Empty:
            receipt["disposition"] = "LOCAL_TIMEOUT_REMOTE_DISPOSITION_UNKNOWN"
            raise CaptureError("REQUEST_OR_ELAPSED_TIMEOUT") from None
        receipt["received_utc"] = utcnow().isoformat()
        if not ok:
            receipt["disposition"] = "TRANSPORT_FAILED"
            raise CaptureError(response.args[0] if isinstance(response, CaptureError) else "TRANSPORT_ERROR_" + type(response).__name__)
        if not isinstance(response, Response) or not isinstance(response.body, bytes):
            raise CaptureError("INVALID_CLIENT_RESPONSE")
        receipt.update({"status": response.status, "raw_body_bytes": len(response.body),
                        "raw_body_sha256": digest(response.body)})
        byte_count += len(response.body)
        if len(response.body) > capacity:
            raise CaptureError("RESPONSE_BYTE_LIMIT")
        remaining()
        if response.effective_params != params:
            raise CaptureError("EFFECTIVE_REQUEST_MISMATCH")
        if response.status != 200:
            raise CaptureError("HTTP_STATUS_" + str(response.status))
        payload = strict_json(response.body)
        receipt["disposition"] = "RECEIVED_WITHIN_LIMITS"
        return payload
    try:
        files.append(save(root, "symbols.json", {"symbols": SYMBOLS, "benchmark": "SPY",
                         "universe": "selected-symbol research universe",
                         "selection_source": "canonical data/history_cache fourteen filenames plus SPY",
                         "selection_source_sha256": CACHE_SHA256,
                         "historical_eligible_membership": "UNKNOWN"}))
        files.append(save(root, "request-contract.json", {"settings": asdict(settings), "limits": asdict(limits),
                         "bars_params": settings.params(), "end_boundary": "exclusive client validation",
                         "symbol_mapping": "literal identifiers; explicit provider asof; no alias fallback",
                         "client": client.identity, "collector_sha256": digest(Path(__file__).read_bytes())}))
        sessions = parse_calendar(fetch(CALENDAR_URL, {"start": start.isoformat(), "end": end.isoformat()}), start, end)
        files.append(save(root, "calendar.json", list(sessions.values())))
        ledger[-1]["validated_artifact"] = files[-1]
        token, seen = None, set()
        while True:
            params = settings.params()
            if token is not None:
                params["page_token"] = token
            payload = fetch(DATA_URL, params)
            if not isinstance(payload, dict) or not isinstance(payload.get("bars"), dict):
                raise CaptureError("MALFORMED_BAR_PAGE")
            if "next_page_token" not in payload:
                raise CaptureError("MISSING_PAGINATION_STATE")
            next_token = payload.get("next_page_token")
            if next_token is not None and (not isinstance(next_token, str) or not next_token or len(next_token) > 2048):
                raise CaptureError("MALFORMED_PAGE_TOKEN")
            if next_token in seen:
                raise CaptureError("REPEATED_PAGE_TOKEN")
            bars = payload["bars"]
            if set(bars) - set(SYMBOLS) or any(not isinstance(rows, list) for rows in bars.values()):
                raise CaptureError("UNREQUESTED_SYMBOL_OR_MALFORMED_ROWS")
            if sum(map(len, bars.values())) > 10000:
                raise CaptureError("PAGE_RECORD_LIMIT")
            page = []
            for symbol, rows in bars.items():
                for bar in rows:
                    remaining()
                    record = parse_bar(symbol, bar, settings, sessions)
                    key = (symbol, record["session"])
                    if key in data:
                        if data[key] != record:
                            raise CaptureError("CONFLICTING_DUPLICATE")
                        if record["timestamp_utc"] < last[symbol]:
                            raise CaptureError("PROVIDER_ORDER_CONFLICT")
                        duplicates += 1
                    elif symbol in last and record["timestamp_utc"] <= last[symbol]:
                        raise CaptureError("PROVIDER_ORDER_CONFLICT")
                    else:
                        data[key] = record
                        last[symbol] = record["timestamp_utc"]
                    page.append(record)
            files.append(save(root, f"page-{len(ledger)-1:03d}.json", {"bars": page,
                             "next_token_sha256": digest(next_token.encode()) if next_token else None}))
            ledger[-1]["validated_artifact"] = files[-1]
            if next_token is None:
                break
            seen.add(next_token)
            token = next_token
        missing = {symbol: [day for day in sessions if (symbol, day) not in data] for symbol in SYMBOLS}
        non_sessions = []
        for offset in range((end - start).days + 1):
            day = start + timedelta(days=offset)
            if day.isoformat() not in sessions:
                non_sessions.append({"date": day.isoformat(), "classification": "weekend" if day.weekday() >= 5 else "calendar_excluded_weekday_holiday_or_closure"})
        coverage = {"missing_sessions": missing, "non_sessions": non_sessions,
                    "eligibility_on_missing_dates": "UNKNOWN; absence is not proof of ineligibility",
                    "calendar_completeness": "provider-returned calendar; no independent completeness proof",
                    "identical_duplicates_removed": duplicates}
        files.append(save(root, "coverage.json", coverage))
        if missing["SPY"]:
            raise CaptureError("MISSING_SPY_BENCHMARK_SESSIONS")
        files.append(save(root, "dataset.json", [data[key] for key in sorted(data)]))
        remaining()
        disposition = "CAPTURE_COMPLETE_WITH_COVERAGE_GAPS" if any(missing.values()) else "CAPTURE_COMPLETE"
    except Exception as exc:
        code = str(exc) if isinstance(exc, CaptureError) else "LOCAL_ERROR_" + type(exc).__name__
        files.append(save(root, "failure.json", {"code": code, "usable_dataset": False}))
        disposition = "FAILED"
    files.append(save(root, "request-receipts.json", ledger))
    if disposition.startswith("CAPTURE_COMPLETE"):
        try:
            remaining()
        except CaptureError:
            files.append(save(root, "failure.json", {"code": "ELAPSED_LIMIT", "usable_dataset": False}))
            disposition = "FAILED"
    manifest = {"schema": "jbravo.research-capture.v1", "status": disposition, "files": files,
                "requests_attempted": len(ledger), "completed_body_bytes": byte_count,
                "elapsed_seconds": clock() - begun, "sealed_utc": utcnow().isoformat(),
                "qualifications": ["Selected symbols, not historical eligible universe",
                                   "No historical provenance recovery or untouched-period claim",
                                   "Local timeout does not prove remote request termination",
                                   "Failed/incomplete streams have unknown transfer totals"]}
    for binding in files:
        actual = (root / binding["path"]).read_bytes()
        if len(actual) != binding["bytes"] or digest(actual) != binding["sha256"]:
            raise CaptureError("FINAL_ARTIFACT_BINDING_MISMATCH")
    if disposition.startswith("CAPTURE_COMPLETE") and clock() >= deadline:
        files.append(save(root, "failure.json", {"code": "ELAPSED_LIMIT", "usable_dataset": False}))
        manifest["status"] = disposition = "FAILED"
        manifest["elapsed_seconds"] = clock() - begun
    final_binding = save(root, "capture-manifest.json", manifest)  # last; no self hash
    return {"root": str(root), "manifest": final_binding, "status": disposition}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("start-date", "end-date", "feed", "adjustment", "currency", "asof", "capture-id"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    settings = Settings(args.start_date, args.end_date, args.feed, args.adjustment, args.currency, args.asof)
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        print("AUTHENTICATION_UNAVAILABLE", file=sys.stderr)
        return 1
    client = None
    try:
        settings.validate()
        client = HTTPClient(key, secret)
        result = capture(client, settings, Path.home() / "jbravo-research-data", args.capture_id)
        print(json.dumps(result))
        return 0 if result["status"].startswith("CAPTURE_COMPLETE") else 1
    except Exception as exc:
        print(str(exc) if isinstance(exc, CaptureError) else "LOCAL_CAPTURE_FAILURE", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                print("RESEARCH_CLIENT_CLEANUP_FAILED; closure unproved", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
