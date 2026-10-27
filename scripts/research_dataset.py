"""Hash-bound historical inputs for isolated research; no provider or database clients."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class DataError(ValueError):
    pass


def digest_bytes(body):
    return hashlib.sha256(body).hexdigest()


def canonical_hash(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def strict_json(body):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise DataError("Duplicate JSON key")
            obj[key] = value
        return obj
    try:
        return json.loads(body, object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(DataError("Nonfinite JSON")))
    except (ValueError, UnicodeError) as exc:
        raise DataError("Invalid strict JSON input") from exc


def bounded_read(path, limit=128 * 1024 * 1024):
    with Path(path).open("rb") as handle:
        body = handle.read(limit + 1)
    if len(body) > limit:
        raise DataError("Input byte limit exceeded")
    return body


@dataclass(frozen=True)
class ResearchDataset:
    raw: dict[str, pd.DataFrame]
    features: dict[str, pd.DataFrame]
    calendar: pd.DataFrame
    dividends: tuple[dict, ...]
    provenance: dict


def load_dataset(spec):
    manifest_path = Path(spec["capture_manifest"]).resolve()
    root = manifest_path.parent.parent
    manifest_body = bounded_read(manifest_path)
    manifest = strict_json(manifest_body)
    if manifest.get('schema') == 'jbravo.market-capture.v1':
        from scripts.strategy_universe import load_market_dataset
        return load_market_dataset(spec, manifest, manifest_path)
    if manifest.get("research_qualified") is not False:
        raise DataError("Expected original unqualified capture status")
    bindings = {item["path"].replace("\\", "/"): item for item in manifest["bindings"]}
    consumed = []

    def member(name):
        key = manifest_path.parent.name + "/" + name
        binding = bindings.get(key)
        if not binding:
            raise DataError("Required capture member not bound: " + name)
        path = (root / key).resolve()
        if not path.is_relative_to(root):
            raise DataError("Capture path escapes its root")
        body = bounded_read(path)
        if digest_bytes(body) != binding["sha256"] or len(body) != binding["bytes"]:
            raise DataError("Capture hash/size mismatch: " + name)
        consumed.append({"path": str(path), "sha256": digest_bytes(body), "bytes": len(body)})
        return body

    import io
    frames = {}
    for adjustment in ("raw", "split"):
        frame = pd.read_csv(io.BytesIO(member(f"alpaca_{adjustment}_daily.csv")), dtype=str, keep_default_na=False)
        required = {"symbol", "session", "o", "h", "l", "c", "v", "quality_flags", "provider", "feed", "currency", "adjustment", "asof"}
        if not required <= set(frame):
            raise DataError("Missing required captured bar columns")
        if frame.duplicated(["symbol", "session"]).any():
            raise DataError("Duplicate symbol/session input")
        for column, expected in (("provider", "alpaca"), ("feed", "sip"), ("currency", "USD"), ("adjustment", adjustment)):
            if not frame[column].eq(expected).all():
                raise DataError("Mixed or incorrect input " + column)
        if frame["asof"].nunique() != 1:
            raise DataError("Mixed symbol mapping asof dates")
        for value in frame["session"]:
            if date.fromisoformat(value).isoformat() != value:
                raise DataError("Noncanonical session date")
        for field in ("o", "h", "l", "c", "v"):
            frame[field] = pd.to_numeric(frame[field], errors="raise")
            if not np.isfinite(frame[field]).all() or (frame[field] < 0).any():
                raise DataError("Nonfinite or negative bar value")
        if (frame[["o", "h", "l", "c"]] <= 0).any().any():
            raise DataError("Nonpositive price")
        if (frame["l"] > frame[["o", "c"]].min(axis=1)).any() or (frame["h"] < frame[["o", "c"]].max(axis=1)).any():
            raise DataError("Invalid OHLC geometry")
        if not np.equal(frame["v"], np.floor(frame["v"])).all():
            raise DataError("Fractional captured Alpaca volume")
        frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        frames[adjustment] = frame

    calendar_records = strict_json(member("calendar.json"))
    days = [item["date"] for item in calendar_records]
    if days != sorted(set(days)):
        raise DataError("Unsorted or duplicate calendar")
    calendar = pd.DataFrame(calendar_records).set_index("date")
    for day, row in calendar.iterrows():
        opening, closing = datetime.fromisoformat(day + "T" + row["open"]), datetime.fromisoformat(day + "T" + row["close"])
        if opening >= closing or row["timezone"] != "America/New_York":
            raise DataError("Invalid calendar clock")

    all_symbols = set(frames["raw"]["symbol"])
    universe = spec["universe"]
    selected = list(universe["symbols"])
    benchmark = universe["benchmark"]
    exclusions = universe["excluded"]
    if len(selected) != len(set(selected)) or benchmark in selected or not selected:
        raise DataError("Duplicate/empty trading universe or benchmark also traded")
    if set(selected) & set(exclusions) or benchmark in exclusions:
        raise DataError("Excluded symbol selected")
    if set(selected) | {benchmark} | set(exclusions) != all_symbols:
        raise DataError("Every captured symbol must be selected, benchmark, or explicitly excluded")
    if not all(isinstance(reason, str) and reason.strip() for reason in exclusions.values()):
        raise DataError("Every exclusion needs a reason")
    earliest = min(fold["warmup_start"] for fold in spec["folds"])
    latest = max(fold["test_end"] for fold in spec["folds"])
    expected = [d for d in days if earliest <= d <= latest]
    if not expected:
        raise DataError("No calendar sessions in study")
    raw, features = {}, {}
    for symbol in selected + [benchmark]:
        for adjustment, destination in (("raw", raw), ("split", features)):
            frame = frames[adjustment]
            piece = frame.loc[(frame["symbol"] == symbol) & frame["session"].between(earliest, latest)].sort_values("session")
            if list(piece["session"]) != expected:
                raise DataError("Missing or extra eligible session: " + symbol + "/" + adjustment)
            if piece["quality_flags"].ne("").any() or piece["volume"].le(0).any():
                raise DataError("Flagged/zero-volume observations require explicit separate study: " + symbol)
            piece = piece.set_index(pd.to_datetime(piece["session"]))
            destination[symbol] = piece
        ratios = raw[symbol]["close"] / features[symbol]["close"]
        # Current v1 supports action-free share units and cash distributions only.
        # A split must not silently bypass share/stop/basis accounting.
        if not np.allclose(ratios, ratios.iloc[0], rtol=1e-7, atol=1e-9):
            raise DataError("Changing split factor requires a share-action adapter: " + symbol)
        for field in ("open", "high", "low"):
            if not np.allclose(raw[symbol][field], features[symbol][field] * ratios, rtol=1e-5, atol=1e-5):
                raise DataError("Raw/split price convention mismatch: " + symbol)

    actions_spec = spec["actions_capture"]
    action_body = bounded_read(actions_spec["path"])
    if digest_bytes(action_body) != actions_spec["sha256"]:
        raise DataError("Supplementary action capture hash mismatch")
    actions_capture = strict_json(action_body)
    if actions_capture.get("isError"):
        raise DataError("Action provider request failed")
    texts = "\n".join(x["text"] for x in actions_capture["content"] if x["type"] == "text")
    actions = strict_json(texts)["data"]
    if actions.get("next_page_token") is not None:
        raise DataError("Incomplete action pagination")
    request = actions_capture["request"]
    if request["start"] > earliest or request["end"] < latest or not (set(selected) | {benchmark}) <= set(request["symbols"].split(",")):
        raise DataError("Action request does not cover the study")
    dividends, event_ids = [], set()
    for category, events in actions["corporate_actions"].items():
        for event in events:
            if event["id"] in event_ids:
                raise DataError("Duplicate corporate action identity")
            event_ids.add(event["id"])
            mentioned = {v for k, v in event.items() if k == "symbol" or k.endswith("_symbol")}
            if not mentioned & (set(selected) | {benchmark}):
                continue
            event_day = event.get("ex_date") or event.get("effective_date") or event.get("process_date")
            if not earliest <= event_day <= latest:
                continue
            if category != "cash_dividends":
                raise DataError("Unsupported corporate action in selected study: " + category)
            rate = event["rate"]
            if isinstance(rate, bool) or not np.isfinite(rate) or rate < 0:
                raise DataError("Invalid dividend rate")
            if date.fromisoformat(event["payable_date"]) < date.fromisoformat(event["ex_date"]):
                raise DataError("Dividend payment precedes entitlement")
            if event["ex_date"] not in calendar.index:
                raise DataError("Dividend ex-date is not a supplied session")
            dividends.append(dict(event))
    consumed.append({"path": str(Path(actions_spec["path"]).resolve()), "sha256": digest_bytes(action_body), "bytes": len(action_body)})
    provenance = {"capture_manifest": str(manifest_path), "capture_manifest_sha256": digest_bytes(manifest_body),
                  "inputs": consumed, "selection": universe, "feed": "sip", "currency": "USD",
                  "execution_prices": "raw", "feature_prices": "split", "asof": frames["raw"]["asof"].iloc[0],
                  "historical_availability": "REVISED_SNAPSHOT_WITH_ASSUMED_20_15_ET_FINALIZATION",
                  "research_qualified": False, "execution_authorized": False,
                  "limitations": ["Selected-symbol retrospective study; no historical-universe or untouched-holdout claim",
                                  "Provider identity, action completeness and calendar remain independently unqualified",
                                  "Cash dividends use supplied gross USD rates; tax/withholding/cash-in-lieu are not modeled",
                                  "Share-changing actions and changing split factors are refused, not silently ignored"]}
    provenance["dataset_sha256"] = canonical_hash(provenance)
    return ResearchDataset(raw, features, calendar, tuple(sorted(dividends, key=lambda e: (e["ex_date"], e["id"]))), provenance)
