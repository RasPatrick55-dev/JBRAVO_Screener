"""Actual saved-health reader with synthetic files; no app/provider/DB imports."""
import ast
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

NOW = datetime(2026, 10, 29, 17, 30, tzinfo=timezone.utc)
ALIASES = ("timestamp", "ts_utc", "last_run_utc", "last_run")


def reload_data_io(monkeypatch,base_dir):
    monkeypatch.setenv("JBRAVO_HOME",str(base_dir))
    scripts=ModuleType("scripts");scripts.__path__=[]
    db=ModuleType("scripts.db")
    def forbidden(*args,**kwargs):raise AssertionError("DB/provider import or call forbidden")
    db.get_db_conn=forbidden;db.db_enabled=lambda:False;scripts.db=db
    spec=importlib.util.spec_from_file_location("saved_health_contract_reader",Path(__file__).parents[1]/"dashboards/data_io.py")
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{"scripts":scripts,"scripts.db":db}):
        spec.loader.exec_module(module)
    return module


def read(tmp_path, monkeypatch, payload, *, fallback=False):
    base=tmp_path/"data";base.mkdir(exist_ok=True)
    target=base/("connectivity.json" if fallback else "connection_health.json")
    target.write_text(json.dumps(payload),encoding="utf-8")
    module=reload_data_io(monkeypatch,tmp_path)
    return module, module.load_connection_health(base)


@pytest.mark.parametrize("alias",ALIASES)
@pytest.mark.parametrize("stamp",["2026-10-29T17:29:30Z","2026-10-29T13:29:30-04:00","2026-10-29T17:29:30.000000+00:00"])
def test_supported_aliases_use_actual_aware_instant(tmp_path,monkeypatch,alias,stamp):
    module,payload=read(tmp_path,monkeypatch,{alias:stamp,"trading_ok":True})
    check=module.connection_check_snapshot(payload,now=NOW)
    assert check["checked_at"]=="2026-10-29T17:29:30+00:00"
    assert check["status"]=="fresh" and check["trading_ok"] is True
    assert check["freshness"]["age_seconds"]==30


@pytest.mark.parametrize("value",["",False,1,[],{},"not-a-time","2026-02-30T00:00:00Z","2026-10-29T17:30:00","2026-10-29","x"*129,
    "2026-10-29X17:29:30Z","2026-10-29 17:29:30Z","2026-10-29T17:29:30.0000001Z"])
def test_invalid_or_naive_alias_does_not_borrow_a_valid_timestamp(tmp_path,monkeypatch,value):
    module,payload=read(tmp_path,monkeypatch,{"timestamp":value,"ts_utc":"2026-10-29T17:29:30Z","trading_ok":True})
    check=module.connection_check_snapshot(payload,now=NOW)
    assert check["status"]=="unknown" and check["reason"]=="timestamp_invalid"
    assert check["checked_at"] is None and check["freshness"]["age_seconds"] is None


def test_equivalent_aliases_allowed_but_conflicting_instants_unknown(tmp_path,monkeypatch):
    module,payload=read(tmp_path,monkeypatch,{"timestamp":"2026-10-29T17:29:30Z","ts_utc":"2026-10-29T13:29:30-04:00","trading_ok":True})
    assert module.connection_check_snapshot(payload,now=NOW)["status"]=="fresh"
    module,payload=read(tmp_path,monkeypatch,{"timestamp":"2026-10-25T04:04:59Z","ts_utc":"2026-10-29T17:29:30Z","trading_ok":True})
    check=module.connection_check_snapshot(payload,now=NOW)
    assert check["status"]=="unknown" and check["reason"]=="timestamp_conflict"


def test_invalid_primary_does_not_borrow_a_fresh_fallback(tmp_path,monkeypatch):
    module,value=read(tmp_path,monkeypatch,{"ts_utc":"invalid","trading_ok":True})
    (tmp_path/"data/connectivity.json").write_text(json.dumps({"ts_utc":"2026-10-29T17:29:30Z","trading_ok":True}))
    check=module.connection_check_snapshot(module.load_connection_health(tmp_path/"data"),now=NOW)
    assert check["source"]=="connection_health.json" and check["reason"]=="timestamp_invalid"


@pytest.mark.parametrize("offset",["+00:60","-00:60","+01:99","-01:99","+24:00","-24:00","+99:00","-99:99"])
def test_overflow_offsets_never_normalize_to_fresh_evidence(tmp_path,monkeypatch,offset):
    module,payload=read(tmp_path,monkeypatch,{"ts_utc":"2026-10-29T16:29:30"+offset,"trading_ok":True})
    check=module.connection_check_snapshot(payload,now=NOW)
    assert check["status"]=="unknown" and check["reason"]=="timestamp_invalid"
    assert check["checked_at"] is None and check["freshness"]["age_seconds"] is None
    direct=module.connection_check_snapshot({"source":"connection_health.json",
        "timestamp":"2026-10-29T16:29:30"+offset,"trading_ok":True},now=NOW)
    assert direct["reason"]=="timestamp_invalid" and direct["checked_at"] is None


@pytest.mark.parametrize("stamp",["2026-10-29T18:29:30+00:60","2026-10-29T16:29:30-00:60",
    "2026-10-29T20:08:30+01:99","2026-10-29T14:50:30-01:99"])
def test_previously_normalized_overflows_cannot_create_a_thirty_second_old_success(tmp_path,monkeypatch,stamp):
    module,payload=read(tmp_path,monkeypatch,{"ts_utc":stamp,"trading_ok":True})
    check=module.connection_check_snapshot(payload,now=NOW)
    assert check["status"]=="unknown" and check["reason"]=="timestamp_invalid"
    assert check["checked_at"] is None


@pytest.mark.parametrize("stamp",["2026-10-29T17:29:30+00:00","2026-10-29T17:29:30-00:00",
    "2026-10-29T22:59:30+05:30","2026-10-29T13:59:30-03:30","2026-10-30T07:29:30+14:00"])
def test_valid_signed_offsets_preserve_the_exact_instant(tmp_path,monkeypatch,stamp):
    module,payload=read(tmp_path,monkeypatch,{"ts_utc":stamp,"trading_ok":True})
    check=module.connection_check_snapshot(payload,now=NOW)
    assert check["status"]=="fresh" and check["checked_at"]=="2026-10-29T17:29:30+00:00"
    assert check["freshness"]["age_seconds"]==30


@pytest.mark.parametrize("payload",[{}, {"timestamp":None},{"ts_utc":None}])
def test_missing_timestamp_is_not_generated(tmp_path,monkeypatch,payload):
    module,value=read(tmp_path,monkeypatch,{**payload,"trading_ok":True})
    check=module.connection_check_snapshot(value,now=NOW)
    assert check["reason"]=="timestamp_missing" and check["checked_at"] is None


@pytest.mark.parametrize("alias",ALIASES)
def test_future_is_unknown_and_stale_is_not_refreshed_by_reader(tmp_path,monkeypatch,alias):
    module,value=read(tmp_path,monkeypatch,{alias:(NOW+timedelta(microseconds=1)).isoformat(),"trading_ok":True})
    assert module.connection_check_snapshot(value,now=NOW)["reason"]=="timestamp_future"
    module,value=read(tmp_path,monkeypatch,{alias:"2026-10-25T04:04:59Z","trading_ok":True})
    check=module.connection_check_snapshot(value,now=NOW)
    assert check["status"]=="stale" and check["checked_at"]=="2026-10-25T04:04:59+00:00"


@pytest.mark.parametrize("result",[True,False,None])
def test_fallback_filename_and_failed_or_unknown_result_preserved(tmp_path,monkeypatch,result):
    module,payload=read(tmp_path,monkeypatch,{"ts_utc":"2026-10-29T17:29:30Z","trading_ok":result},fallback=True)
    check=module.connection_check_snapshot(payload,now=NOW)
    assert check["source"]=="connectivity.json" and check["trading_ok"] is result
    assert check["status"]==("unknown" if result is None else "fresh")


@pytest.mark.parametrize("result",[True,False])
def test_real_standalone_producer_payload_round_trips_to_reader(tmp_path,monkeypatch,result):
    # Execute only the producer's pure run_probe definition with stubbed probes,
    # clock/config/logger. No dotenv, HTTP or application startup is executed.
    source=Path(__file__).parents[1]/"scripts/check_connection.py"
    node=next(n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=="run_probe")
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return NOW
    data=tmp_path/"data";data.mkdir();file=data/"connection_health.json"
    ns=dict(Dict=dict,Any=object,json=json,datetime=Clock,timezone=timezone,
        _load_dotenv_if_needed=lambda:None,_setup_logger=lambda:logging.getLogger("synthetic-probe"),
        _probe_trading=lambda _:dict(ok=result,status=200 if result else 503,message=""),
        _probe_data=lambda _:dict(ok=True,status=200,message=""),
        _ensure_dirs=lambda:None,CONNECTION_HEALTH_JSON=file)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"actual-run-probe-synthetic", "exec"),ns)
    produced=ns["run_probe"]()
    assert "ts_utc" in produced and "timestamp" not in produced
    module=reload_data_io(monkeypatch,tmp_path)
    check=module.connection_check_snapshot(module.load_connection_health(data),now=NOW)
    assert check["status"]=="fresh" and check["trading_ok"] is result
    assert check["freshness"]["age_seconds"]==0
