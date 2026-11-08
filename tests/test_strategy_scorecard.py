"""Offline contracts for the read-only strategy-wide scorecard."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dashboards import strategy_scorecard as score

pytestmark = pytest.mark.alpaca_optional
NOW = datetime(2026, 11, 8, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]
SID = "trend_pullback_reclaim_qualifying_close_v2"
PRODUCER = "/home/RasPatrick/research/runs"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(*, compressed=True, market=False):
    strategy = dict(id=SID, version="2", name="Pullback", research_method=SID,
        source_sha256={"strategy_library.py": "a"*64, "rules.py": "b"*64},
        status="PAPER_TEST_CANDIDATE", buy_rules=["synthetic test rule"],
        warmup_sessions=250, paper_scan_enabled=True)
    definition = deepcopy(strategy)
    definition["source_sha256"].pop("strategy_library.py")
    strategy.update(definition_sha256=score.digest(definition), research_qualified=False)
    spec = dict(universe=dict(symbols=["AAA", "BBB", "CCC", "DDD", "EEE"], benchmark="SPY", excluded={}),
        folds=[dict(id="daily-paper-2026", warmup_start="2023-01-03", test_start="2026-01-02", test_end="2026-11-06")],
        cost_cases=[dict(id="base_cost", fee_per_fill=1.0, slippage_bps=5.0)],
        portfolio=dict(initial_cash=100000, alloc_pct=.05, top_n=3, max_positions=4), strategies=[SID])
    coverage = dict(policy=dict(source="alpaca_assets", status="active", asset_class="us_equity", tradable=True),
                    status="COMPLETE", evaluated_count=5, excluded_count=2) if market else None
    population = dict(policy=coverage["policy"], benchmark="SPY") if market else spec["universe"]
    episode = dict(strategy_id=SID, version="2", fold_id="daily-paper-2026", warmup_start="2023-01-03",
        test_start="2026-01-02", universe=population, portfolio=spec["portfolio"],
        cost_case=spec["cost_cases"][0], definition_sha256=strategy["definition_sha256"])
    identity = dict(strategy_id=SID, strategy_name="Pullback", strategy_version="2",
        definition_sha256=strategy["definition_sha256"], episode_id=score.digest(episode), as_of="2026-11-06")
    traces = [trace(identity, "AAA", "2026-09-01", [60, -20]),
              trace(identity, "BBB", "2026-09-08", [-20, -20]),
              trace(identity, "CCC", "2026-09-15", [20, -20]),
              trace(identity, "DDD", "2026-09-22", [20], state="OPEN"),
              trace(identity, "EEE", "2026-11-06", [], state="UNFILLED_PROPOSAL")]
    summary = dict(schema="jbravo.strategy-scan.v1", status="COMPLETED_PAPER_SIMULATION",
        as_of=identity["as_of"], broker_operations=0, research_qualified=False, order_submission_authorized=False,
        dataset_sha256="d"*64, source_sha256={"scanner.py": "e"*64}, top_candidates=[],
        strategies=[dict(strategy=strategy, episode_id=identity["episode_id"], as_of=identity["as_of"],
            metrics=dict(exit_fill_win_rate_pct=99.9, profit_factor=999))])
    summary["run_id"] = score.digest(dict(spec=spec, as_of=identity["as_of"], sources=summary["source_sha256"],
        dataset_sha256=summary["dataset_sha256"], strategies=[identity["definition_sha256"]], cost_case="base_cost"))
    if market:
        summary["coverage"] = coverage
    handoff = dict(schema="jbravo.strategy-paper-proposals.v1", as_of=identity["as_of"],
        scan_run_id=summary["run_id"], paper_only=True, order_submission_authorized=False, proposals=[])
    saved_json = lambda value: (json.dumps(value, sort_keys=True, indent=2)+"\n").encode()
    files = {"specification.json": score.canonical(spec), "result.json": saved_json(summary),
             "paper_handoff.json": saved_json(handoff)}
    filename = SID+"/lifecycle.json"+(".gz" if compressed else "")
    ledger = score.canonical(traces)
    files[filename] = gzip.compress(ledger, mtime=0) if compressed else ledger
    files["manifest.json"] = score.canonical(dict(bindings=[dict(path=name, bytes=len(raw), sha256=sha(raw))
                                                           for name, raw in files.items()]))
    archive = dict(schema="jbravo.daily-strategy-scan.v1", status="COMPLETED_PAPER_SCREEN",
        pipeline="pullback-reclaim-daily-paper-v1", strategy_id=SID, signal_session=identity["as_of"],
        summary=summary, handoff=handoff, output_root=PRODUCER+"/"+identity["as_of"],
        scan_manifest_sha256=sha(files["manifest.json"]), scan_run_id=summary["run_id"],
        research_qualified=False, execution_authorized=False, order_submission_authorized=False)
    archive["run_id"] = score.digest(dict(schema="jbravo.daily-strategy-archive-id.v1", pipeline=archive["pipeline"],
                                         strategy_id=SID, signal_session=identity["as_of"]))
    record = (sha(files["specification.json"]), score.digest(archive), archive)
    folder = archive["output_root"]+"/scan/"
    calls = []
    def reader(path, limit):
        assert path.startswith(folder)
        calls.append(path)
        raw = files[path[len(folder):]]
        assert len(raw) <= limit
        return raw
    return SimpleNamespace(identity=identity, traces=traces, files=files, record=record,
                           reader=reader, calls=calls, filename=filename)


def trace(identity, symbol, session, pnl_parts, *, state="CLOSED"):
    signal = score.digest(dict(strategy_id=identity["strategy_id"], version=identity["strategy_version"],
        episode_id=identity["episode_id"], symbol=symbol, signal_session=session))
    result = dict(signal_id=signal, symbol=symbol, signal_session=session,
        strategy_id=identity["strategy_id"], strategy_version=identity["strategy_version"],
        status=state, bought_quantity=0 if state == "UNFILLED_PROPOSAL" else 10,
        sold_quantity=10 if state == "CLOSED" else 5 if pnl_parts else 0, fills=[])
    if state == "UNFILLED_PROPOSAL":
        return result
    common = {k: result[k] for k in ("signal_id", "symbol", "strategy_id", "strategy_version")}
    from datetime import date, timedelta
    entry = (date.fromisoformat(session)+timedelta(days=1)).isoformat()
    result["fills"].append(dict(common, side="BUY", session=entry, quantity=10, raw_price=100.,
                                fee=1., slippage=1., cash_delta=-1002.))
    for i, net in enumerate(pnl_parts):
        # Allocate half the entry costs into each 5-share exit.
        price = (float(net)+503.)/5
        result["fills"].append(dict(common, side="SELL", session=entry, quantity=5, raw_price=price,
            fee=1., slippage=1., cash_delta=5*price-2., net_pnl=float(net)))
    return result


def aggregate(f):
    return score.aggregate([dict(identity=f.identity, lifecycle=f.traces)])


def test_partial_exit_is_not_a_trade_and_costs_are_not_subtracted_twice():
    result = aggregate(fixture())
    assert result["closed_signals"] == 3
    assert (result["wins"], result["losses"], result["breakeven"]) == (1, 1, 1)
    assert result["win_rate_pct"] == pytest.approx(100/3)
    assert result["net_pnl"] == result["expectancy_net_per_signal"] == 0
    assert result["average_win"] == 40 and result["average_loss"] == -40
    assert result["payoff_ratio"] == result["profit_factor"] == 1
    assert result["costs"] == dict(basis="MODELED", fees=9., slippage=9., total=18., actual_costs_available=False)
    assert result["open_signals"] == result["unfilled_signals"] == 1


def test_repeated_history_is_deduplicated_and_open_positions_can_mature():
    f = fixture()
    old = deepcopy(f.traces)
    old[0]["fills"] = old[0]["fills"][:2]
    old[0].update(status="OPEN", sold_quantity=5)
    first = dict(identity=dict(f.identity, as_of="2026-11-05"), lifecycle=old[:-1])
    later = dict(identity=f.identity, lifecycle=f.traces)
    result = score.aggregate([first, later])
    assert result["closed_signals"] == 3 and result["net_pnl"] == 0
    assert result["evidence"]["duplicate_observations"] == 4


def test_conflicting_closed_history_never_selects_better_result():
    f = fixture()
    newer = deepcopy(f.traces)
    newer[0] = trace(f.identity, "AAA", "2026-09-01", [100, -20])
    with pytest.raises(score.EvidenceError, match="conflicting_signal_history"):
        score.aggregate([dict(identity=f.identity, lifecycle=f.traces), dict(identity=f.identity, lifecycle=newer)])


def test_later_complete_history_cannot_drop_a_filled_signal():
    f = fixture()
    with pytest.raises(score.EvidenceError, match="conflicting_signal_history"):
        score.aggregate([dict(identity=f.identity, lifecycle=f.traces),
                         dict(identity=f.identity, lifecycle=f.traces[1:])])


def test_incompatible_definition_and_episode_never_pool():
    f = fixture()
    with pytest.raises(score.EvidenceError, match="incompatible_episodes"):
        score.aggregate([dict(identity=dict(f.identity, definition_sha256="0"*64), lifecycle=[]),
                         dict(identity=f.identity, lifecycle=f.traces)])


@pytest.mark.parametrize("mutation", [
    lambda t: t[0].update(signal_id="x"*64),
    lambda t: t[0]["fills"][0].update(cash_delta=-1000),
    lambda t: t[0]["fills"][1].update(net_pnl=9999),
    lambda t: t[0]["fills"][1].update(session="2026-09-01"),
    lambda t: t[0]["fills"][1].update(quantity=11),
    lambda t: t[0]["fills"][1].update(strategy_version="3"),
    lambda t: t[0]["fills"][1].update(fee=float("nan")),
    lambda t: t[0].update(sold_quantity=5),
    lambda t: t.append(deepcopy(t[0])),
])
def test_invalid_lifecycle_rejected(mutation):
    f = fixture()
    mutation(f.traces)
    with pytest.raises(score.EvidenceError):
        aggregate(f)


def test_empty_verified_is_distinct_from_unknown():
    f = fixture()
    f.traces.clear()
    result = aggregate(f)
    assert result["closed_signals"] == 0 and result["net_pnl"] == 0
    assert result["win_rate_pct"] is None and result["expectancy_net_per_signal"] is None
    assert result["profit_factor_status"] == "NO_CLOSED_SIGNALS"
    unknown = score.empty_row(f.identity)
    assert all(unknown[k] is None for k in score.METRICS)
    assert unknown["costs"]["total"] is None


def test_no_loss_has_undefined_profit_factor_not_infinity():
    f = fixture()
    f.traces = f.traces[:1]
    result = aggregate(f)
    assert result["profit_factor"] is None
    assert result["profit_factor_status"] == "UNDEFINED_NO_LOSSES"
    assert result["payoff_ratio"] is None
    assert result["uncertainty"]["low"] is None
    assert result["uncertainty"]["status"] == "INSUFFICIENT_SIGNAL_WEEKS"


def test_bootstrap_is_deterministic_and_groups_by_signal_week():
    f = fixture()
    first = aggregate(f)["uncertainty"]
    assert first == aggregate(f)["uncertainty"]
    assert first["signal_weeks"] == 3 and first["samples"] == 400
    assert first["low"] <= 0 <= first["high"]


@pytest.mark.parametrize("compressed,market", [(False, False), (True, False), (True, True)])
def test_complete_bound_reader_handles_fixed_and_market_artifacts(compressed, market):
    f = fixture(compressed=compressed, market=market)
    identity = score.archive_metadata(f.record, now=NOW)
    snapshot, episode = score.bound_lifecycle(f.record, identity, f.reader, PRODUCER)
    assert snapshot["lifecycle"] == f.traces and len(f.calls) == 3
    assert episode["test_start"] == "2026-01-02" and episode["declared_symbols"] == 5
    assert episode["universe_kind"] == ("DAILY_ACTIVE_TRADABLE_UNIVERSE" if market else "FIXED_COHORT")
    assert score.aggregate([snapshot])["win_rate_pct"] != 99.9


@pytest.mark.parametrize("part", ["manifest.json", "specification.json", SID+"/lifecycle.json.gz"])
def test_tampered_artifact_is_rejected(part):
    f = fixture()
    f.files[part] += b" "
    with pytest.raises(score.EvidenceError):
        score.bound_lifecycle(f.record, f.identity, f.reader, PRODUCER)


def test_db_hash_and_definition_hash_are_independently_validated():
    f = fixture()
    f.record[2]["summary"]["strategies"][0]["strategy"]["version"] = "3"
    with pytest.raises(score.EvidenceError, match="archive_hash_mismatch"):
        score.archive_metadata(f.record, now=NOW)
    record = (f.record[0], score.digest(f.record[2]), f.record[2])
    with pytest.raises(score.EvidenceError, match="definition_hash_mismatch"):
        score.archive_metadata(record, now=NOW)


def test_unbound_or_traversal_paths_never_reach_provider():
    f = fixture()
    f.record[2]["output_root"] += "/../secrets"
    with pytest.raises(score.EvidenceError, match="archive_path_mismatch"):
        score.bound_lifecycle(f.record, f.identity, f.reader, PRODUCER)
    assert not f.calls


def test_gzip_expansion_bounded(monkeypatch):
    f = fixture()
    monkeypatch.setattr(score, "MAX_EXPANDED_BYTES", 10)
    with pytest.raises(score.EvidenceError, match="expanded_artifact_size_limit"):
        score.bound_lifecycle(f.record, f.identity, f.reader, PRODUCER)


def test_duplicate_json_keys_and_nonfinite_input_refused():
    for raw in [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}']:
        with pytest.raises(score.EvidenceError):
            score.strict_json(raw)


class Connection:
    def __init__(self, records, *, error=None, oversized=False):
        self.records, self.error, self.oversized = records, error, oversized
        self.queries, self.session, self.closed, self.rolled_back = [], None, False, False
    def set_session(self, **kwargs):
        self.session = kwargs
    def cursor(self):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def execute(self, sql, params=()):
        self.queries.append((sql, params))
        if self.error:
            raise self.error
    def fetchall(self):
        return [(r[2]["run_id"], score.MAX_ARCHIVE_BYTES+1 if self.oversized else len(score.canonical(r[2])))
                for r in self.records]
    def fetchone(self):
        return next(r for r in self.records if r[2]["run_id"] == self.queries[-1][1][0])
    def rollback(self):
        self.rolled_back = True
    def close(self):
        self.closed = True


def config(tmp_path):
    folder = tmp_path/"config"
    folder.mkdir()
    (folder/"strategy_pipeline.json").write_text(json.dumps(dict(schema="jbravo.primary-strategy-route.v1",
        enabled=True, producer_root=PRODUCER, strategy_id=SID, order_submission_authorized=False)))


def test_snapshot_read_only_transaction_closes_before_artifact_fetch(tmp_path):
    f = fixture()
    conn = Connection([f.record])
    config(tmp_path)
    def reader(path, limit):
        assert conn.closed and conn.rolled_back
        return f.reader(path, limit)
    result = score.snapshot(base_dir=tmp_path, now=NOW, connection_factory=lambda: conn, reader=reader)
    assert result["status"] == "COMPLETE" and result["rows"][0]["closed_signals"] == 3
    assert conn.session == dict(readonly=True, isolation_level="REPEATABLE READ", autocommit=False)
    assert all(sql.lstrip().startswith(("SELECT", "SET LOCAL", "WITH")) for sql, _ in conn.queries)
    assert result["actual_paper"]["net_metrics_available"] is False
    assert result["research_qualified"] is result["order_submission_authorized"] is False
    json.dumps(result, allow_nan=False)


def test_provider_failure_keeps_identity_and_unknown_metrics(tmp_path):
    f = fixture()
    config(tmp_path)
    def broken(path, limit):
        raise RuntimeError("secret password with private location")
    result = score.snapshot(base_dir=tmp_path, now=NOW, connection_factory=lambda: Connection([f.record]), reader=broken)
    assert result["status"] == "UNAVAILABLE"
    assert result["rows"][0]["strategy_id"] == SID
    assert result["rows"][0]["closed_signals"] is None
    assert "secret" not in json.dumps(result) and "private" not in json.dumps(result)


def test_oversized_db_record_never_downloaded(tmp_path):
    f = fixture()
    conn = Connection([f.record], oversized=True)
    result = score.snapshot(base_dir=tmp_path, now=NOW, connection_factory=lambda: conn, reader=f.reader)
    assert result["coverage"]["rejected_archives"] == 1
    assert result["status"] == "UNAVAILABLE" and not f.calls
    assert not any(sql == score.ARCHIVE_SQL for sql, _ in conn.queries)


def test_db_errors_are_sanitized_and_connection_closed(tmp_path):
    conn = Connection([], error=RuntimeError("postgres password=secret"))
    result = score.snapshot(base_dir=tmp_path, now=NOW, connection_factory=lambda: conn)
    assert result["failure_code"] == "evidence_unavailable" and conn.closed and conn.rolled_back
    assert "secret" not in json.dumps(result)


def test_no_archive_and_unknown_config_is_not_claimed_complete(tmp_path):
    result = score.snapshot(base_dir=tmp_path, now=NOW, connection_factory=lambda: Connection([]))
    assert result["status"] == "UNAVAILABLE" and result["rows"] == []


def test_every_configured_strategy_survives_missing_or_unavailable_database(tmp_path):
    config(tmp_path)
    (tmp_path/"config/strategy_companions.json").write_text(json.dumps(dict(
        schema="jbravo.strategy-companions.v1", enabled=True, strategy_id="bull_flag_breakout_v1")))
    for conn in (Connection([]), Connection([], error=RuntimeError("no database"))):
        result = score.snapshot(base_dir=tmp_path, now=NOW, connection_factory=lambda: conn)
        assert result["status"] == "UNAVAILABLE" and result["rows_final"] == 2
        assert {r["strategy_id"] for r in result["rows"]} == {SID, "bull_flag_breakout_v1"}
        assert all(r["closed_signals"] is None and r["as_of"] is None for r in result["rows"])


def test_historical_date_is_not_replaced_by_fetch_time(tmp_path):
    config(tmp_path)
    f = fixture()
    result = score.snapshot(base_dir=tmp_path, now=NOW, connection_factory=lambda: Connection([f.record]), reader=f.reader)
    fresh = result["rows"][0]["freshness"]
    assert fresh["evidence_as_of"] == "2026-11-06" and fresh["source_age_calendar_days"] == 2
    assert fresh["latest_attempt_status"] == "NOT_EVALUATED" and fresh["latest_attempt_at"] is None


def test_actual_backend_contract_fixture_matches_complete_response(tmp_path):
    config(tmp_path)
    (tmp_path/"config/strategy_companions.json").write_text(json.dumps(dict(
        schema="jbravo.strategy-companions.v1", enabled=True, strategy_id="bull_flag_breakout_v1")))
    f = fixture(compressed=False, market=True)
    result = score.snapshot(base_dir=tmp_path, now=NOW, connection_factory=lambda: Connection([f.record]), reader=f.reader)
    golden = json.loads((ROOT/"tests/fixtures/strategy_scorecard.json").read_text(encoding="utf-8"))
    assert result == golden
    assert result["status"] == "PARTIAL" and result["rows_final"] == 2
    assert result["rows"][0]["closed_signals"] is None and result["rows"][1]["closed_signals"] == 3


def test_database_whole_request_budget_is_checked():
    f = fixture()
    conn = Connection([f.record])
    with pytest.raises(score.EvidenceError, match="database_budget_exceeded"):
        score.read_archives(conn, deadline=score.time.monotonic()-1)
    assert not any(sql == score.ARCHIVE_SQL for sql, _ in conn.queries)


def test_provider_origin_path_redirect_timeout_and_body_caps(monkeypatch):
    env = dict(PYTHONANYWHERE_USERNAME="RasPatrick", PYTHONANYWHERE_API_TOKEN="dummy")
    with pytest.raises(score.EvidenceError, match="provider_origin_refused"):
        score.ProviderReader(env=dict(env, PYTHONANYWHERE_API_BASE_URL="https://evil.invalid"))
    reader = score.ProviderReader(env=env)
    for path in ("/home/other/x", "/home/RasPatrick/../secret"):
        with pytest.raises(score.EvidenceError, match="provider_path_refused"):
            reader(path, 100)
    with pytest.raises(score.EvidenceError, match="provider_redirect_refused"):
        score.NoRedirect().redirect_request(None, None, 302, None, None, "https://evil.invalid")
    response = io.BytesIO(b"123456")
    response.status, response.headers = 200, {}
    calls = []
    def opening(req, timeout):
        assert req.full_url == "https://www.pythonanywhere.com/api/v0/user/RasPatrick/files/path/home/RasPatrick/file"
        assert req.get_method() == "GET" and 0 < timeout <= 5
        calls.append(req)
        return response
    monkeypatch.setattr(score.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=opening))
    with pytest.raises(score.EvidenceError, match="artifact_size_limit"):
        reader("/home/RasPatrick/file", 5)
    assert len(calls) == 1  # No automatic retry.


def test_cache_single_flight_and_explicit_staleness(monkeypatch):
    score._cache.clear()
    calls = []
    monkeypatch.setattr(score, "snapshot", lambda **kwargs: calls.append(kwargs) or
                        dict(ok=True, status="EMPTY", rows=[], rows_final=0))
    assert score.api_payload(base_dir="test")["status"] == "EMPTY"
    assert score.api_payload(base_dir="test")["cache_age_seconds"] >= 0
    assert len(calls) == 1
    score._cache["test"] = (score.time.monotonic()-1000, score._cache["test"][1])
    score._refresh_lock.acquire()
    try:
        value = score.api_payload(base_dir="test")
        assert value["status"] == "PARTIAL" and value["stale"] and value["cache_age_seconds"] >= 1000
        assert score.api_payload(base_dir="other")["failure_code"] == "refresh_in_progress"
    finally:
        score._refresh_lock.release()
        score._cache.clear()


def test_route_is_additive_public_and_uses_explicit_native_root(monkeypatch):
    tree = ast.parse((ROOT/"dashboards/dashboard_app.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "api_strategy_scorecard")
    assert len(fn.decorator_list) == 1
    assert ast.literal_eval(fn.decorator_list[0].args[0]) == "/api/strategies/scorecard"
    fn.decorator_list = []
    monkeypatch.setattr(score, "api_payload", lambda **kwargs: kwargs)
    namespace = dict(BASE_DIR="native-data-root", _json_no_store=lambda value: value)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "route", "exec"), namespace)
    assert namespace["api_strategy_scorecard"]() == dict(base_dir="native-data-root")


def test_registered_route_accepts_public_get_but_never_writes(monkeypatch):
    from flask import Flask, jsonify
    app = Flask(__name__)
    tree = ast.parse((ROOT/"dashboards/dashboard_app.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "api_strategy_scorecard")
    monkeypatch.setattr(score, "api_payload", lambda **kwargs: dict(ok=True, rows=[], **kwargs))
    def no_store(payload):
        response = jsonify(payload)
        response.headers["Cache-Control"] = "no-store"
        return response
    namespace = dict(server=app, BASE_DIR="native-data-root", _json_no_store=no_store)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "route", "exec"), namespace)
    client = app.test_client()
    response = client.get("/api/strategies/scorecard")
    assert response.status_code == 200 and response.json["base_dir"] == "native-data-root"
    assert response.headers["Cache-Control"] == "no-store"
    assert all(client.open("/api/strategies/scorecard", method=method).status_code == 405
               for method in ("POST", "PUT", "PATCH", "DELETE"))


def test_module_is_python310_and_has_no_runtime_or_provider_import_side_effects():
    source = (ROOT/"dashboards/strategy_scorecard.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 10))
    tree = ast.parse(source)
    native = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
              and (n.module or "").startswith("scripts")]
    assert native == ["scripts.research_store"]
    assert "swing_scorecard" not in source and "strategy_execution" not in source
