"""Pure research classification/comparison; no I/O or simulator enforcement."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import re
from typing import Mapping, Sequence


NASDAQ_NOTICE = (
    "https://www.globenewswire.com/news-release/2024/10/18/2965726/6948/en/"
    "nasdaq-resumes-trading-in-nebius-group-n-v.html"
)
VS_ANNOUNCEMENT = (
    "https://www.sec.gov/Archives/edgar/data/1856437/000185643726000007/"
    "ex991vscomay212026pressrel.htm"
)
VS_FILING = (
    "https://www.sec.gov/Archives/edgar/data/1856437/000185643726000007/"
    "vsco-20260521.htm"
)
ASOF_DOCUMENTATION = "https://docs.alpaca.markets/us/reference/stockbars"
NEBIUS_KEY = "research:nebius-listed-line"
VS_KEY = "research:victorias-secret-common-stock"
COMPARE_FIELDS = ("o", "h", "l", "c", "v", "n", "vw")
OPTIONAL_FIELDS = ("n", "vw")
POLICY_DECLARATIONS = (
    "Halt-period records cannot supply executable prices, liquidity or warm-up.",
    "Reset indicator/model state across a prolonged halt; require qualified warm-up.",
    "Invalidate pending entry signals at a halt or unresolved interruption.",
    "Preserve positions and pending exit/stop obligations as unresolved; invent no fill.",
    "First qualified executable bar uses existing causal gap/exit rules before adjustment.",
    "Carried valuations are stale/uncertain; ordinary calendar closures are not data gaps.",
)


def _day(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("INVALID_SESSION")
    date.fromisoformat(value)
    return value


@dataclass(frozen=True)
class SymbolInterval:
    """Half-open session interval; aliases are provider labels, not effective tickers."""
    security_key: str
    provider_symbols: tuple[str, ...]
    start: str | None
    end: str | None
    historical_ticker: str | None
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class StatusInterval:
    security_key: str
    start: str | None
    end: str | None
    status: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    symbols: tuple[SymbolInterval, ...]
    statuses: tuple[StatusInterval, ...]


DEFAULT_POLICY = Policy(
    symbols=(
        # The notice establishes former/current names, not the precise rename date.
        SymbolInterval(NEBIUS_KEY, ("YNDX", "NBIS"), None, None, None, (NASDAQ_NOTICE,)),
        SymbolInterval(VS_KEY, ("VSCO", "VSXY"), None, "2026-06-02", "VSCO",
                       (VS_ANNOUNCEMENT, VS_FILING, ASOF_DOCUMENTATION)),
        SymbolInterval(VS_KEY, ("VSCO", "VSXY"), "2026-06-02", None, "VSXY",
                       (VS_ANNOUNCEMENT, VS_FILING, ASOF_DOCUMENTATION)),
    ),
    statuses=(
        StatusInterval(NEBIUS_KEY, "2022-02-28", "2024-10-21", "HALTED", (NASDAQ_NOTICE,)),
        StatusInterval(NEBIUS_KEY, "2024-10-21", None, "RESUMPTION_DOCUMENTED",
                       (NASDAQ_NOTICE,)),
    ),
)


def _contains(interval, session):
    return ((interval.start is None or interval.start <= session)
            and (interval.end is None or session < interval.end))


def _overlaps(left, right):
    return ((left.end is None or right.start is None or right.start < left.end)
            and (right.end is None or left.start is None or left.start < right.end))


def validate_policy(policy: Policy) -> None:
    """Reject ambiguous routes rather than choosing the first matching mapping."""
    if not isinstance(policy, Policy) or not isinstance(policy.symbols, tuple) or not isinstance(
        policy.statuses, tuple
    ):
        raise ValueError("INVALID_POLICY")
    for interval in (*policy.symbols, *policy.statuses):
        if not isinstance(interval, (SymbolInterval, StatusInterval)):
            raise ValueError("INVALID_INTERVAL")
        if not isinstance(interval.security_key, str) or not re.fullmatch(
            r"research:[a-z0-9][a-z0-9-]*", interval.security_key
        ):
            raise ValueError("INTERNAL_KEY_REQUIRED")
        for boundary in (interval.start, interval.end):
            if boundary is not None:
                _day(boundary)
        if interval.start is not None and interval.end is not None and interval.start >= interval.end:
            raise ValueError("EMPTY_OR_REVERSED_INTERVAL")
        if (not isinstance(interval.evidence, tuple) or not interval.evidence
                or any(not isinstance(ref, str) or not ref.strip() for ref in interval.evidence)):
            raise ValueError("EVIDENCE_REQUIRED")
    for index, interval in enumerate(policy.symbols):
        if not isinstance(interval, SymbolInterval):
            raise ValueError("INVALID_SYMBOL_INTERVAL")
        if (not isinstance(interval.provider_symbols, tuple) or not interval.provider_symbols
                or len(set(interval.provider_symbols)) != len(interval.provider_symbols)
                or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]*", s)
                       for s in interval.provider_symbols)):
            raise ValueError("INVALID_PROVIDER_SYMBOLS")
        if interval.historical_ticker is not None and interval.historical_ticker not in interval.provider_symbols:
            raise ValueError("UNBOUND_HISTORICAL_TICKER")
        for other in policy.symbols[:index]:
            if _overlaps(interval, other) and (
                interval.security_key == other.security_key
                or set(interval.provider_symbols) & set(other.provider_symbols)
            ):
                raise ValueError("AMBIGUOUS_SYMBOL_INTERVAL")
    keys = {s.security_key for s in policy.symbols}
    for index, interval in enumerate(policy.statuses):
        if not isinstance(interval, StatusInterval) or interval.status not in (
            "HALTED", "RESUMPTION_DOCUMENTED"
        ):
            raise ValueError("INVALID_STATUS_INTERVAL")
        if interval.security_key not in keys:
            raise ValueError("UNBOUND_STATUS_KEY")
        for other in policy.statuses[:index]:
            if interval.security_key == other.security_key and _overlaps(interval, other):
                raise ValueError("CONTRADICTORY_STATUS_INTERVAL")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("INVALID_NUMBER")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("INVALID_NUMBER") from exc
    if not result.is_finite():
        raise ValueError("NONFINITE_NUMBER")
    return result


def _quality(record):
    reasons = []
    values = {}
    for field in COMPARE_FIELDS:
        if field in OPTIONAL_FIELDS and field not in record:
            continue  # Collector count/VWAP are optional; preserve absent fields.
        try:
            values[field] = _number(record[field])
        except (KeyError, ValueError):
            reasons.append("INVALID_OR_MISSING_FIELD:" + field)
    for field, value in values.items():
        if ((field in ("o", "h", "l", "c") and value <= 0)
                or (field in ("v", "n", "vw") and value < 0)
                or (field in ("v", "n") and value % 1)):
            reasons.append("INVALID_FIELD:" + field)
    if all(field in values for field in ("o", "h", "l", "c")) and not (
        values["l"] <= min(values["o"], values["c"])
        <= max(values["o"], values["c"]) <= values["h"]
    ):
        reasons.append("INVALID_OHLC_RELATIONSHIP")
    if values.get("vw") == 0:
        reasons.append("ZERO_VWAP_UNQUALIFIED")
    if values.get("v") == 0:
        reasons.append("ZERO_VOLUME_UNQUALIFIED")
    return reasons


def _classify(record, policy):
    if not isinstance(record, Mapping):
        raise ValueError("RECORD_MAPPING_REQUIRED")
    symbol = record.get("symbol")
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]*", symbol):
        raise ValueError("INVALID_PROVIDER_SYMBOL")
    session = _day(record.get("session"))
    mapping = next((s for s in policy.symbols if symbol in s.provider_symbols
                    and _contains(s, session)), None)
    key = mapping.security_key if mapping else None
    status = next((s for s in policy.statuses if s.security_key == key
                   and _contains(s, session)), None)
    quality = _quality(record)
    blocked = (status is not None and status.status == "HALTED") or bool(quality)
    reasons = ["RESEARCH_QUALIFICATION_OPEN", "CORPORATE_ACTIONS_UNQUALIFIED",
               "CALENDAR_AND_SECURITY_ELIGIBILITY_UNQUALIFIED"]
    reasons.append("EXTERNAL_SECURITY_IDENTIFIERS_UNVERIFIED" if mapping
                   else "UNRESOLVED_SECURITY_IDENTITY")
    if mapping and mapping.historical_ticker is None:
        reasons.append("HISTORICAL_TICKER_EFFECTIVE_DATE_UNRESOLVED")
    if status and status.status == "HALTED":
        reasons.append("DOCUMENTED_HALT_NONEXECUTABLE_NO_WARMUP")
    if status and status.status == "RESUMPTION_DOCUMENTED":
        reasons.append("RESUMPTION_REQUIRES_QUALIFICATION_AND_FRESH_WARMUP")
    reasons.extend(quality)
    return {
        "original_record": deepcopy(dict(record)),
        "provider_symbol": symbol, "session": session, "security_key": key,
        "security_key_kind": "INTERNAL_RESEARCH_KEY" if key else None,
        "verified_external_identifiers": [],
        "identity_disposition": "POLICY_CONTINUITY_MAPPING" if key else "UNRESOLVED",
        "historical_ticker": mapping.historical_ticker if mapping else None,
        "status": status.status if status else "SESSION_STATUS_UNRESOLVED",
        "execution_eligibility": "BLOCKED" if blocked else "UNRESOLVED",
        "warmup_eligibility": "BLOCKED" if blocked else "UNRESOLVED",
        "executable": False if blocked else None,
        "available_for_warmup": False if blocked else None,
        "research_qualified": False, "reasons": reasons,
        "evidence_references": list(dict.fromkeys(
            (mapping.evidence if mapping else ()) + (status.evidence if status else ()))),
    }


def classify_records(records: Sequence[Mapping], *, policy: Policy = DEFAULT_POLICY) -> dict:
    """Return detached copies; caller provenance remains inside each original_record."""
    validate_policy(policy)
    return {
        "schema": "jbravo.research-security-status.v1", "research_qualified": False,
        "simulator_enforcement": "NOT_IMPLEMENTED",
        "policy_declarations": list(POLICY_DECLARATIONS),
        "documented_events": {
            "halt_start_et": "2022-02-28T06:38:00-05:00",
            "resumption_et": "2024-10-21T09:00:00-04:00",
            "vsxy_effective_session": "2026-06-02",
            "event_scope": "default public evidence; custom policy does not rewrite events",
        },
        "records": [_classify(record, policy) for record in records],
    }


def compare_records(left: Sequence[Mapping], right: Sequence[Mapping], *,
                    sessions: Sequence[str], policy: Policy = DEFAULT_POLICY) -> dict:
    """Compare only an explicit session window; matches never qualify data or stitch it."""
    validate_policy(policy)
    window = [_day(s) for s in sessions]
    if not window or len(set(window)) != len(window) or window != sorted(window):
        raise ValueError("UNIQUE_SORTED_NONEMPTY_SESSIONS_REQUIRED")
    window_set = set(window)
    groups = [defaultdict(list), defaultdict(list)]
    unresolved = []
    for side, records in enumerate((left, right)):
        for original in records:
            row = _classify(original, policy)
            if row["session"] not in window_set:
                continue
            if row["security_key"] is None:
                unresolved.append({"side": "left" if side == 0 else "right", "record": row})
            else:
                groups[side][(row["security_key"], row["session"])].append(row)
    outcomes = []
    for key, session in sorted(set(groups[0]) | set(groups[1])):
        a, b = groups[0][(key, session)], groups[1][(key, session)]
        differences = []
        if len(a) > 1 or len(b) > 1:
            disposition = "CONFLICT_DUPLICATE_IDENTITY"
        elif not a or not b:
            disposition = "MISSING_LEFT" if not a else "MISSING_RIGHT"
        else:
            for field in COMPARE_FIELDS:
                x, y = a[0]["original_record"], b[0]["original_record"]
                if field not in x or field not in y:
                    equal = field in OPTIONAL_FIELDS and field not in x and field not in y
                else:
                    try:
                        equal = _number(x[field]) == _number(y[field])
                    except ValueError:
                        equal = False
                if not equal:
                    differences.append(field)
            invalid = any(r.startswith("INVALID") for row in (*a, *b) for r in row["reasons"])
            disposition = "CONFLICT_INVALID_RECORD" if invalid else (
                "CONFLICT" if differences else "MATCH")
        outcomes.append({
            "security_key": key, "session": session, "disposition": disposition,
            "differing_fields": differences, "left": a, "right": b,
            "research_qualified": False,
        })
    keys = {k for group in groups for k, _ in group}
    missing_both = [{"security_key": k, "session": s, "disposition": "MISSING_BOTH"}
                    for k in sorted(keys) for s in window
                    if (k, s) not in groups[0] and (k, s) not in groups[1]]
    return {
        "schema": "jbravo.research-security-overlap.v1", "research_qualified": False,
        "comparison_scope": list(COMPARE_FIELDS), "sessions": window,
        "calendar_authority": "CALLER_SUPPLIED_UNQUALIFIED",
        "outcomes": outcomes, "unresolved_identities": unresolved,
        "missing_both": missing_both,
        "limitations": ["Only policy-mapped securities observed in the window are inventoried.",
                        "Absence from both inputs cannot identify an entirely unseen security.",
                        "Provider timestamps/provenance are preserved, not equated or verified.",
                        "No merged dataset, execution eligibility or research readiness is produced."],
    }
