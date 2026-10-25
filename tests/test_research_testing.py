"""Normal research caller, causality, cash actions, data refusal and SQL boundaries."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import research_dataset, research_simulation, research_store, research_testing
from scripts.research_dataset import digest_bytes, load_dataset
from scripts.research_simulation import ResearchPortfolio, availability_clock
from scripts.research_strategies import BASELINE, BENCHMARK, CHALLENGER, prepare_signals

pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Research unit tests cannot access the network")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def simple_frame(count=4, *, signal=True):
    dates = pd.bdate_range("2024-01-02", periods=count)
    return pd.DataFrame(dict(open=100.0, high=100.0, low=100.0, close=100.0, volume=1000,
                             score=[5.0 if signal and i == 0 else np.nan for i in range(count)],
                             ATR14=2.0, ema20=1.0, macd=1.0, macd_signal=0.0, rsi=50.0,
                             entry_ready=True, entry_details=[{} for _ in dates]), index=dates)


def engine(data, method=BASELINE, **options):
    defaults = dict(initial_cash=1000.0, alloc_pct=0.5, max_positions=1, top_n=1,
                    trail_pct=None, atr_multiple=0, max_hold_days=1000,
                    enable_macd_exit=False, enable_partial_exit=False, enable_candlestick_exit=False,
                    enable_ema_exit=False, enable_rsi_divergence=False, enable_trailing_exit=False)
    defaults.update(options)
    return ResearchPortfolio(data, method=method, **defaults)


def test_normal_imports_do_not_load_runtime_clients(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"utils", "alpaca", "psycopg2"}:
            raise AssertionError("Research import attempted runtime/client dependency: " + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(research_dataset)
    importlib.reload(research_simulation)
    importlib.reload(research_testing)


def test_next_open_and_cash_fee_reserve_benchmark():
    frame = simple_frame()
    bt = engine({"SPY": frame}, BENCHMARK, alloc_pct=1.0, trade_cost=5, slippage=0.01)
    bt.run()
    assert len(bt.fills) == 1
    fill = bt.fills[0]
    assert fill["signal_session"] == "2024-01-02"
    assert fill["session"] == "2024-01-03"
    assert fill["quantity"] == 9
    assert bt.cash == pytest.approx(86)
    assert bt.summary()["terminal_equity"] == pytest.approx(986)
    assert fill["signal_available_at_assumed"] < fill["execution_at_assumed"]


@pytest.mark.parametrize("ex_offset,pay_offset,expected", [(1, 3, 0), (2, 3, 10), (2, 8, 10)])
def test_dividend_prior_close_entitlement_cash_and_receivable(ex_offset, pay_offset, expected):
    frame = simple_frame()
    ex = frame.index[ex_offset].strftime("%Y-%m-%d")
    pay = (frame.index[0] + pd.offsets.BDay(pay_offset)).strftime("%Y-%m-%d")
    dividend = dict(id="D1", symbol="AAA", ex_date=ex, payable_date=pay, rate=2.0)
    bt = engine({"AAA": frame}, dividends=(dividend,))
    bt.run()
    result = bt.summary()
    assert result["dividend_income"] == expected
    assert result["terminal_equity"] == pytest.approx(1000 + expected)
    if pay > frame.index[-1].strftime("%Y-%m-%d"):
        assert result["dividend_receivable"] == expected
        assert result["dividend_cash"] == 0
    else:
        assert result["dividend_cash"] == expected
    assert len(bt.reconciliations) == len(frame)


def test_challenger_consumes_unselected_closes_and_requires_new_pullback():
    first = simple_frame(count=3)
    first["score"] = 5.0
    first["entry_details"] = [dict(pullback_start="2024-01-01") for _ in range(3)]
    second = first.copy(deep=True)
    bt = engine({"AAA": first, "BBB": second}, CHALLENGER)
    bt.run()
    initial = [x for x in bt.signals if x["session"] == "2024-01-02"]
    assert all(x["consumed"] for x in initial)
    assert {x["symbol"] for x in bt.fills} == {"AAA"}
    assert bt.last_reclaims["BBB"] == "2024-01-02"
    assert any(x["reason"] == "REARM_REQUIRES_PULLBACK_AFTER_CONSUMED_CLOSE" for x in bt.signals)


def test_terminal_signal_canceled_without_fictitious_fill():
    bt = engine({"AAA": simple_frame(count=1)})
    bt.run()
    assert not bt.fills and not bt.pending_entries
    assert bt.terminal_entries[0]["reason"] == "NO_NEXT_SESSION_INSIDE_EVALUATION_WINDOW"


@pytest.mark.parametrize("cause", ["deadline", "stop"])
def test_stop_and_deadline_are_checked_before_any_fill(tmp_path, cause):
    stop = tmp_path / "STOP"
    if cause == "stop":
        stop.touch()
    bt = engine({"AAA": simple_frame()}, stop_file=stop,
                deadline=time.monotonic() - 1 if cause == "deadline" else None)
    with pytest.raises(RuntimeError, match="stopped"):
        bt.run()
    assert not bt.fills
    with pytest.raises(ValueError, match="replayed"):
        bt.run()


def test_expired_entry_refused():
    bt = engine({"AAA": simple_frame()})
    bt.pending_entries["AAA"] = pd.Series(dict(ATR14=1), name=pd.Timestamp("2023-12-20"))
    with pytest.raises(ValueError, match="expiry"):
        bt.run()


def test_dst_signal_clock_is_aware_and_precedes_next_open():
    assert availability_clock("2026-10-30").endswith("+00:00")
    assert "00:15:00" in availability_clock("2026-10-30")
    assert "01:15:00" in availability_clock("2026-11-02")
    assert availability_clock("2026-10-30") < availability_clock("2026-11-02", opening=True)


def featured_source(count=400):
    index = pd.bdate_range("2023-01-03", periods=count)
    seq = np.arange(count)
    close = 100 + seq * 0.12 + 2 * np.sin(seq * 0.4)
    return pd.DataFrame(dict(open=close - 0.2, high=close + 1, low=close - 1,
                             close=close, volume=5000 + 1000 * np.cos(seq * 0.2)), index=index)


@pytest.mark.parametrize("method", [BASELINE, CHALLENGER])
def test_future_append_preserves_features_signals_and_prefix_fills(method):
    source = featured_source()
    shorter = source.iloc[:330]
    early_fold = dict(warmup_start="2023-01-03", test_start=source.index[270].strftime("%Y-%m-%d"), test_end=shorter.index[-1].strftime("%Y-%m-%d"))
    late_fold = dict(early_fold, test_end=source.index[-1].strftime("%Y-%m-%d"))
    a = prepare_signals(shorter, shorter, method, early_fold)
    b = prepare_signals(source, source, method, late_fold)
    pd.testing.assert_frame_equal(a, b.loc[a.index])
    one = engine({"AAA": a}, method)
    two = engine({"AAA": b}, method)
    one.run()
    two.run()
    earlier = a.index[-1].strftime("%Y-%m-%d")
    assert one.fills == [x for x in two.fills if x["session"] <= earlier]
    assert one.signals == [x for x in two.signals if x["session"] <= earlier]
    repeated = engine({"AAA": a}, method)
    repeated.run()
    assert one.summary() == repeated.summary()


def make_capture(tmp_path):
    root = tmp_path / "capture"
    folder = root / "analysis"
    folder.mkdir(parents=True)
    source = featured_source(360)
    source["volume"] = 5000
    records = []
    for symbol in ("AAA", "SPY"):
        for stamp, row in source.iterrows():
            records.append(dict(symbol=symbol, session=stamp.strftime("%Y-%m-%d"),
                                o=row["open"], h=row["high"], l=row["low"], c=row["close"], v=row["volume"],
                                provider="alpaca", feed="sip", currency="USD", asof="2026-10-23", quality_flags=""))
    files = []
    for adjustment in ("raw", "split"):
        frame = pd.DataFrame(records)
        frame["adjustment"] = adjustment
        target = folder / f"alpaca_{adjustment}_daily.csv"
        frame.to_csv(target, index=False)
        files.append(target)
    calendar_path = folder / "calendar.json"
    calendar_path.write_text(json.dumps([dict(date=d.strftime("%Y-%m-%d"), open="09:30:00", close="16:00:00", timezone="America/New_York") for d in source.index]))
    files.append(calendar_path)
    manifest = folder / "manifest.json"
    def bind():
        manifest.write_text(json.dumps(dict(research_qualified=False, execution_authorized=False, bindings=[
            dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=digest_bytes(p.read_bytes())) for p in files])))
    bind()
    actions = tmp_path / "actions.json"
    first, last = source.index[0].strftime("%Y-%m-%d"), source.index[-1].strftime("%Y-%m-%d")
    def write_actions(corporate_actions=None):
        actions.write_text(json.dumps(dict(request=dict(start=first, end="2027-01-31", symbols="AAA,SPY"),
                                          isError=False, content=[dict(type="text", text=json.dumps(dict(data=dict(
                                              corporate_actions=corporate_actions or {}, next_page_token=None))))])))
    write_actions()
    spec = research_testing.starter_spec(manifest, actions)
    spec["universe"] = dict(symbols=["AAA"], benchmark="SPY", excluded={})
    spec["folds"] = [dict(id="test", warmup_start=first, test_start=source.index[270].strftime("%Y-%m-%d"), test_end=last)]
    spec["cost_cases"] = [dict(id="base", fee_per_fill=1.0, slippage_bps=5.0)]
    spec["max_runs"] = 3
    return spec, folder, bind, write_actions


def test_normal_study_writes_hash_bound_results_and_all_trials(tmp_path):
    spec, _, _, _ = make_capture(tmp_path)
    output = tmp_path / "results"
    result = research_testing.run_study(spec, output)
    assert result["completed_runs"] == result["planned_runs"] == 3
    assert result["research_qualified"] is False
    assert result["execution_authorized"] is False
    assert {x["method"] for x in result["runs"]} == {BASELINE, CHALLENGER, BENCHMARK}
    for binding in json.loads((output / "manifest.json").read_text())["bindings"]:
        target = output / binding["path"]
        assert digest_bytes(target.read_bytes()) == binding["sha256"]
    assert (output / "report.html").exists()
    with pytest.raises(FileExistsError):
        research_testing.run_study(spec, output)


def test_publication_verifies_completed_artifacts_before_append(tmp_path, monkeypatch):
    spec, _, _, _ = make_capture(tmp_path)
    output = tmp_path / "archive"
    result = research_testing.run_study(spec, output)
    calls = []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, sql, params):
            assert "INSERT INTO jbravo_research.research_runs" in sql
            assert "ON CONFLICT" not in sql
            assert params[0] == result["run_id"]
            calls.append("append")
    conn = SimpleNamespace(cursor=Cursor, commit=lambda: calls.append("commit"),
                           rollback=lambda: calls.append("rollback"), close=lambda: calls.append("close"))
    def connect(**kwargs):
        assert kwargs["write"] is True
        return conn
    monkeypatch.setattr(research_store, "connect", connect)
    research_store.publish_result(output / "result.json")
    assert calls == ["append", "commit", "close"]
    calls.clear()
    (output / "trial_001/equity.csv").write_text("tampered")
    with pytest.raises(research_store.StoreError, match="hash/size"):
        research_store.publish_result(output / "result.json")
    assert not calls


def test_study_failure_retains_spec_source_and_failure(tmp_path):
    spec, folder, _, _ = make_capture(tmp_path)
    with (folder / "alpaca_raw_daily.csv").open("a") as handle:
        handle.write("tamper")
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="hash"):
        research_testing.run_study(spec, out)
    assert (out / "specification.json").exists()
    failure = json.loads((out / "failure.json").read_text())
    assert failure["status"] == "FAILED" and failure["research_qualified"] is False


@pytest.mark.parametrize("problem", ["flag", "missing", "duplicate", "adjustment", "factor"])
def test_loader_refuses_unsafe_selected_inputs(tmp_path, problem):
    spec, folder, bind, _ = make_capture(tmp_path)
    path = folder / ("alpaca_split_daily.csv" if problem == "factor" else "alpaca_raw_daily.csv")
    frame = pd.read_csv(path, keep_default_na=False)
    if problem == "flag":
        frame.loc[0, "quality_flags"] = "ZERO_VOLUME_UNQUALIFIED"
    elif problem == "missing":
        frame = frame.drop(index=0)
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "adjustment":
        frame.loc[0, "adjustment"] = "all"
    else:
        frame.loc[0, ["o", "h", "l", "c"]] *= 0.5
    frame.to_csv(path, index=False)
    bind()
    with pytest.raises(ValueError):
        load_dataset(spec)


def test_unsupported_action_cannot_be_silently_ignored(tmp_path):
    spec, _, _, write_actions = make_capture(tmp_path)
    write_actions(dict(stock_mergers=[dict(id="A1", acquiree_symbol="AAA", effective_date=spec["folds"][0]["test_start"])]))
    spec["actions_capture"]["sha256"] = digest_bytes(Path(spec["actions_capture"]["path"]).read_bytes())
    with pytest.raises(ValueError, match="Unsupported corporate action"):
        load_dataset(spec)


@pytest.mark.parametrize("problem", ["budget", "overlap", "cost", "unknown", "promotion"])
def test_study_spec_refuses_unsafe_or_unbounded_plan(tmp_path, problem):
    spec, _, _, _ = make_capture(tmp_path)
    if problem == "budget":
        spec["max_runs"] = 1
    elif problem == "overlap":
        spec["folds"].append(dict(spec["folds"][0], id="overlap"))
    elif problem == "cost":
        spec["cost_cases"][0]["slippage_bps"] = float("nan")
    elif problem == "unknown":
        spec["strategies"].append("UNVERSIONED_NEW_STRATEGY")
    else:
        spec["execution_authorized"] = True
    with pytest.raises(ValueError):
        research_testing.validate_spec(spec)


def test_sql_connect_is_explicit_read_only_and_errors_sanitized(monkeypatch):
    import psycopg2
    seen = {}
    conn = SimpleNamespace(set_session=lambda **k: seen.update(session=k), close=lambda: None)
    def connect(dsn=None, **kwargs):
        seen.update(kwargs)
        return conn
    monkeypatch.setattr(psycopg2, "connect", connect)
    assert research_store.connect(env={"DATABASE_URL": "postgresql://USER:SECRET@localhost/DB"}) is conn
    assert seen["session"]["readonly"] is True
    assert seen["session"]["isolation_level"] == "REPEATABLE READ"
    assert "default_transaction_read_only=on" in seen["options"]
    with pytest.raises(research_store.StoreError, match="RESEARCH_DATABASE_URL"):
        research_store.connect(write=True, env={"DATABASE_URL": "postgresql://USER:SECRET@localhost/DB"})
    def fail(*args, **kwargs):
        raise RuntimeError("password authentication failed for SECRET_VALUE")
    monkeypatch.setattr(psycopg2, "connect", fail)
    with pytest.raises(research_store.StoreError) as exc:
        research_store.connect(env={"DATABASE_URL": "postgresql://USER:SECRET@localhost/DB"})
    assert "SECRET" not in str(exc.value)
    assert "AUTHENTICATION_REJECTED" in str(exc.value)


def test_database_snapshot_uses_cap_and_rolls_back(monkeypatch):
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, sql, params):
            assert sql.startswith("SELECT symbol")
            assert "public.daily_bars" in sql and params[-1] == 2
        def fetchall(self): return [("AAA", "2024-01-02", 1, 1, 1, 1, 1)] * 2
    calls = []
    conn = SimpleNamespace(cursor=Cursor, rollback=lambda: calls.append("rollback"), close=lambda: calls.append("close"))
    monkeypatch.setattr(research_store, "connect", lambda **k: conn)
    with pytest.raises(research_store.StoreError, match="row cap"):
        research_store.snapshot_daily_bars(["AAA"], "2024-01-01", "2024-01-10", max_rows=1)
    assert calls == ["rollback", "close"]
