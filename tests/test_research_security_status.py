"""Normal module import; in-memory records only; no capture/provider/database access."""
from copy import deepcopy
from dataclasses import replace
import socket
import subprocess

import pytest

from scripts import research_security_status as ledger

pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def forbid_external_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External boundary accessed")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def bar(symbol="NBIS", session="2024-10-21", **changes):
    row = {"symbol": symbol, "session": session, "o": "18.94", "h": "20",
           "l": "18", "c": "19", "v": 100, "n": 5, "vw": "19",
           "provenance": {"capture": "synthetic", "artifact": {"sha256": "fixture"}}}
    row.update(changes)
    return row


def classify(row):
    return ledger.classify_records([row])["records"][0]


@pytest.mark.parametrize("session,status,blocked", [
    ("2022-02-27", "SESSION_STATUS_UNRESOLVED", False),
    ("2022-02-28", "HALTED", True),
    ("2024-10-18", "HALTED", True),
    ("2024-10-20", "HALTED", True),
    ("2024-10-21", "RESUMPTION_DOCUMENTED", False),
    ("2026-10-07", "RESUMPTION_DOCUMENTED", False),
])
def test_halt_half_open_boundaries(session, status, blocked):
    row = classify(bar(session=session))
    assert row["status"] == status
    assert row["executable"] is (False if blocked else None)
    assert row["available_for_warmup"] is (False if blocked else None)
    assert row["research_qualified"] is False


def test_issuer_alias_does_not_invent_rename_date_or_external_identity():
    rows = ledger.classify_records([bar(symbol="YNDX"), bar()])["records"]
    assert rows[0]["security_key"] == rows[1]["security_key"] == ledger.NEBIUS_KEY
    for row in rows:
        assert row["historical_ticker"] is None
        assert row["verified_external_identifiers"] == []
        assert row["security_key_kind"] == "INTERNAL_RESEARCH_KEY"
        assert "HISTORICAL_TICKER_EFFECTIVE_DATE_UNRESOLVED" in row["reasons"]


@pytest.mark.parametrize("symbol,session,ticker", [
    ("VSCO", "2026-06-01", "VSCO"), ("VSXY", "2026-06-01", "VSCO"),
    ("VSXY", "2026-06-02", "VSXY"), ("VSCO", "2026-06-02", "VSXY"),
])
def test_rename_keeps_provider_label_and_separate_historical_ticker(symbol, session, ticker):
    original = bar(symbol=symbol, session=session)
    row = classify(original)
    assert row["provider_symbol"] == symbol
    assert row["historical_ticker"] == ticker
    assert row["security_key"] == ledger.VS_KEY
    assert ledger.VS_FILING in row["evidence_references"]
    assert row["original_record"] == original
    assert row["research_qualified"] is False


def test_unmapped_symbol_is_explicitly_unresolved_not_eligible():
    row = classify(bar(symbol="SPY"))
    assert row["security_key"] is None
    assert row["identity_disposition"] == "UNRESOLVED"
    assert row["execution_eligibility"] == "UNRESOLVED"
    assert "UNRESOLVED_SECURITY_IDENTITY" in row["reasons"]


@pytest.mark.parametrize("changes,reason", [
    ({"vw": 0}, "ZERO_VWAP_UNQUALIFIED"),
    ({"v": 0}, "ZERO_VOLUME_UNQUALIFIED"),
    ({"vw": -1}, "INVALID_FIELD:vw"),
    ({"vw": "NaN"}, "INVALID_OR_MISSING_FIELD:vw"),
    ({"v": True}, "INVALID_OR_MISSING_FIELD:v"),
    ({"c": 0}, "INVALID_FIELD:c"),
    ({"l": 99}, "INVALID_OHLC_RELATIONSHIP"),
    ({"n": 1.5}, "INVALID_FIELD:n"),
])
def test_quality_blocks_without_coercing_original_record(changes, reason):
    original = bar(**changes)
    row = classify(original)
    assert reason in row["reasons"]
    assert row["executable"] is False and row["available_for_warmup"] is False
    assert row["original_record"] == original


def test_inputs_and_nested_provenance_are_detached():
    original = bar(); before = deepcopy(original)
    result = ledger.classify_records([original])
    assert original == before
    result["records"][0]["original_record"]["provenance"]["artifact"]["sha256"] = "changed"
    assert original == before
    original["provenance"]["capture"] = "later"
    assert result["records"][0]["original_record"]["provenance"]["capture"] == "synthetic"


def test_policy_only_no_simulator_enforcement_and_exact_event_times():
    result = ledger.classify_records([bar()])
    assert result["simulator_enforcement"] == "NOT_IMPLEMENTED"
    assert result["documented_events"]["resumption_et"] == "2024-10-21T09:00:00-04:00"
    assert "pending entry" in " ".join(result["policy_declarations"])
    assert result["research_qualified"] is False


@pytest.mark.parametrize("change,code", [
    ({"security_key": "CUSIP:pretend"}, "INTERNAL_KEY_REQUIRED"),
    ({"end": "2020-01-01", "start": "2021-01-01"}, "EMPTY_OR_REVERSED_INTERVAL"),
    ({"evidence": ()}, "EVIDENCE_REQUIRED"),
    ({"provider_symbols": ("NBIS", "NBIS")}, "INVALID_PROVIDER_SYMBOLS"),
    ({"historical_ticker": "OTHER"}, "UNBOUND_HISTORICAL_TICKER"),
])
def test_invalid_mapping_contract_rejected(change, code):
    first = replace(ledger.DEFAULT_POLICY.symbols[0], **change)
    policy = ledger.Policy((first,), ())
    with pytest.raises(ValueError, match=code):
        ledger.classify_records([], policy=policy)


@pytest.mark.parametrize("other_key", [ledger.NEBIUS_KEY, "research:other-security"])
def test_overlapping_alias_mappings_rejected_even_if_same_key(other_key):
    first = ledger.DEFAULT_POLICY.symbols[0]
    policy = ledger.Policy((first, replace(first, security_key=other_key)), ())
    with pytest.raises(ValueError, match="AMBIGUOUS_SYMBOL_INTERVAL"):
        ledger.classify_records([], policy=policy)


def test_contradictory_status_intervals_rejected():
    policy = replace(ledger.DEFAULT_POLICY, statuses=ledger.DEFAULT_POLICY.statuses + (
        ledger.StatusInterval(ledger.NEBIUS_KEY, "2024-10-20", "2024-10-22",
                              "HALTED", (ledger.NASDAQ_NOTICE,)),))
    with pytest.raises(ValueError, match="CONTRADICTORY_STATUS_INTERVAL"):
        ledger.classify_records([], policy=policy)


def compare(left, right, sessions=("2026-06-01",)):
    return ledger.compare_records(left, right, sessions=sessions)


def test_overlap_matches_across_provider_aliases_without_qualifying_or_merging():
    left = [bar(symbol="VSCO", session="2026-06-01")]
    right = [bar(symbol="VSXY", session="2026-06-01", o=18.94)]
    before = deepcopy((left, right))
    result = compare(left, right)
    assert result["outcomes"][0]["disposition"] == "MATCH"
    assert result["outcomes"][0]["left"][0]["original_record"] == left[0]
    assert result["outcomes"][0]["right"][0]["original_record"] == right[0]
    assert (left, right) == before
    assert result["research_qualified"] is False


@pytest.mark.parametrize("field", ledger.COMPARE_FIELDS)
def test_overlap_reports_exact_conflicting_fields(field):
    left = bar(symbol="VSCO", session="2026-06-01")
    right = dict(left, symbol="VSXY", **{field: "19.1" if field in ("o", "c", "vw")
                                         else 101 if field == "v" else 6 if field == "n"
                                         else 21 if field == "h" else 17})
    result = compare([left], [right])["outcomes"][0]
    assert result["disposition"] == "CONFLICT"
    assert result["differing_fields"] == [field]


def test_identical_invalid_values_are_not_reported_as_matches():
    row = bar(symbol="VSCO", session="2026-06-01", v=-1)
    assert compare([row], [row])["outcomes"][0]["disposition"] == "CONFLICT_INVALID_RECORD"


def test_missing_left_right_and_both_with_explicit_window():
    left = [bar(symbol="VSCO", session="2026-06-01")]
    right = [bar(symbol="VSXY", session="2026-06-02")]
    result = compare(left, right, ("2026-06-01", "2026-06-02", "2026-06-03"))
    assert [r["disposition"] for r in result["outcomes"]] == ["MISSING_RIGHT", "MISSING_LEFT"]
    assert result["missing_both"] == [{"security_key": ledger.VS_KEY,
                                       "session": "2026-06-03", "disposition": "MISSING_BOTH"}]


def test_duplicate_identity_is_conflict_not_first_record_selection():
    row = bar(symbol="VSCO", session="2026-06-01")
    result = compare([row, dict(row, symbol="VSXY")], [row])["outcomes"][0]
    assert result["disposition"] == "CONFLICT_DUPLICATE_IDENTITY"
    assert len(result["left"]) == 2


def test_unknown_identity_is_not_joined_by_matching_ticker():
    row = bar(symbol="SPY", session="2026-06-01")
    result = compare([row], [row])
    assert result["outcomes"] == []
    assert len(result["unresolved_identities"]) == 2


@pytest.mark.parametrize("sessions", [(), ("2026-06-01", "2026-06-01"),
                                      ("2026-06-02", "2026-06-01"), ("2026-02-30",)])
def test_comparison_window_rejects_ambiguity(sessions):
    with pytest.raises(ValueError):
        compare([], [], sessions)


def test_empty_inputs_do_not_imply_full_universe_inventory_or_qualification():
    result = compare([], [])
    assert result["outcomes"] == result["missing_both"] == []
    assert result["research_qualified"] is False
    assert "entirely unseen security" in " ".join(result["limitations"])


def test_outside_window_not_inventoried_and_optional_count_presence_compared():
    left = bar(symbol="VSCO", session="2026-06-01"); left.pop("n")
    right = bar(symbol="VSXY", session="2026-06-01")
    result = compare([left, bar(symbol="VSCO", session="2026-05-29")], [right])
    assert len(result["outcomes"]) == 1
    assert result["outcomes"][0]["differing_fields"] == ["n"]


def test_default_constants_are_not_mutated_between_calls():
    first = ledger.classify_records([bar()]); first["policy_declarations"].clear()
    second = ledger.classify_records([bar()])
    assert second["policy_declarations"]
    ledger.validate_policy(ledger.DEFAULT_POLICY)


def test_absent_vwap_remains_absent_without_quality_reason():
    original = bar(); original.pop("vw")
    before = deepcopy(original)
    row = classify(original)
    assert "vw" not in row["original_record"]
    assert not any(reason.endswith(":vw") or reason == "ZERO_VWAP_UNQUALIFIED"
                   for reason in row["reasons"])
    assert row["executable"] is None and row["available_for_warmup"] is None
    assert row["research_qualified"] is False
    assert original == before


@pytest.mark.parametrize("include_count", [True, False])
def test_overlap_absent_vwap_on_both_sides_can_match(include_count):
    left = bar(symbol="VSCO", session="2026-06-01"); left.pop("vw")
    if not include_count:
        left.pop("n")
    right = dict(left, symbol="VSXY")
    result = compare([left], [right])
    outcome = result["outcomes"][0]
    assert outcome["disposition"] == "MATCH"
    assert outcome["differing_fields"] == []
    assert all("vw" not in row["original_record"]
               for row in (*outcome["left"], *outcome["right"]))
    assert result["research_qualified"] is False


@pytest.mark.parametrize("supplied_vwap", ["19", 0])
@pytest.mark.parametrize("absent_on_left", [True, False])
def test_overlap_vwap_presence_asymmetry_conflicts_without_imputation(
    supplied_vwap, absent_on_left
):
    absent = bar(symbol="VSCO", session="2026-06-01"); absent.pop("vw")
    supplied = bar(symbol="VSXY", session="2026-06-01", vw=supplied_vwap)
    left, right = (absent, supplied) if absent_on_left else (supplied, absent)
    before = deepcopy((left, right))
    result = compare([left], [right])
    outcome = result["outcomes"][0]
    assert outcome["disposition"] == "CONFLICT"
    assert outcome["differing_fields"] == ["vw"]
    absent_row = outcome["left" if absent_on_left else "right"][0]
    supplied_row = outcome["right" if absent_on_left else "left"][0]
    assert "vw" not in absent_row["original_record"]
    assert supplied_row["original_record"]["vw"] == supplied_vwap
    assert ("ZERO_VWAP_UNQUALIFIED" in supplied_row["reasons"]) is (supplied_vwap == 0)
    assert "ZERO_VWAP_UNQUALIFIED" not in absent_row["reasons"]
    assert (left, right) == before
    assert result["research_qualified"] is False


@pytest.mark.parametrize("supplied_vwap", [None, "not-a-price"])
def test_supplied_null_or_malformed_vwap_is_not_optional_absence(supplied_vwap):
    original = bar(symbol="VSCO", session="2026-06-01", vw=supplied_vwap)
    row = classify(original)
    assert "INVALID_OR_MISSING_FIELD:vw" in row["reasons"]
    assert "ZERO_VWAP_UNQUALIFIED" not in row["reasons"]
    assert row["original_record"]["vw"] == supplied_vwap
    assert row["executable"] is False and row["available_for_warmup"] is False
    outcome = compare([original], [original])["outcomes"][0]
    assert outcome["disposition"] == "CONFLICT_INVALID_RECORD"
    assert outcome["differing_fields"] == ["vw"]
    assert outcome["research_qualified"] is False


@pytest.mark.parametrize("mandatory_field", ["o", "h", "l", "c", "v"])
def test_optional_vwap_does_not_relax_mandatory_ohlcv(mandatory_field):
    original = bar(symbol="VSCO", session="2026-06-01")
    original.pop("vw"); original.pop(mandatory_field)
    row = classify(original)
    assert "INVALID_OR_MISSING_FIELD:" + mandatory_field in row["reasons"]
    assert "INVALID_OR_MISSING_FIELD:vw" not in row["reasons"]
    assert row["executable"] is False
    outcome = compare([original], [original])["outcomes"][0]
    assert outcome["disposition"] == "CONFLICT_INVALID_RECORD"
    assert outcome["differing_fields"] == [mandatory_field]
