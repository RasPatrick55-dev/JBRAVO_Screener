from copy import deepcopy
from pathlib import Path
import pytest
pytestmark = pytest.mark.alpaca_optional
from scripts.strategy_archive_identity import archive_id, session_root, configuration_hash, PRIMARY

def cfg(feed="iex"):
    return {"feed":feed,"strategy_id":PRIMARY,"policy":{"limit":3}}

def test_feed_revision_separates_archive_and_files_without_relabel():
    old,new=cfg("sip"),cfg()
    assert archive_id(old,"2026-11-13") != archive_id(new,"2026-11-13")
    assert session_root("/runs","2026-11-13",old) != session_root("/runs","2026-11-13",new)
    assert old["feed"]=="sip" and new["feed"]=="iex"

def test_identity_is_deterministic_and_all_config_is_bound():
    value=cfg();reordered=dict(reversed(list(value.items())))
    assert archive_id(value,"2026-11-13")==archive_id(reordered,"2026-11-13")
    other=deepcopy(value);other["policy"]["limit"]=4
    assert archive_id(value,"2026-11-13")!=archive_id(other,"2026-11-13")

def test_companions_share_capture_root_not_archive_identity():
    value=cfg();comp=dict(value,strategy_id="bull_flag_breakout_v1")
    assert session_root("/runs","2026-11-13",value)==session_root("/runs","2026-11-13",comp)
    assert archive_id(value,"2026-11-13")!=archive_id(comp,"2026-11-13")
    assert archive_id(value,"2026-11-13",comp["strategy_id"])==archive_id(comp,"2026-11-13")

@pytest.mark.parametrize("day",["../x","2026-1-1","2026-11-31",None])
def test_bad_session_rejected(day):
    with pytest.raises((ValueError,TypeError)):session_root("/runs",day,cfg())

def test_recovery_preserves_old_claim_and_refuses_same_revision_replay(tmp_path):
    old=tmp_path/"2026-11-13";old.mkdir();claim=old/"claim.json";claim.write_text("historic SIP")
    root=session_root(tmp_path,"2026-11-13",cfg());root.mkdir(parents=True,exist_ok=False)
    assert claim.read_text()=="historic SIP"
    with pytest.raises(FileExistsError):root.mkdir(parents=True,exist_ok=False)


@pytest.mark.skipif(__import__('os').name!='posix',reason='POSIX nofollow directory descriptors')
@pytest.mark.parametrize('level',['session','revisions'])
def test_symlinked_parent_refused_without_outside_writes(tmp_path,level):
    from scripts.strategy_archive_identity import create_session_root
    outside=tmp_path/'outside';outside.mkdir();output=tmp_path/'runs';output.mkdir()
    session=output/'2026-11-13'
    if level=='session':session.symlink_to(outside,target_is_directory=True)
    else:
        session.mkdir();(session/'revisions').symlink_to(outside,target_is_directory=True)
    with pytest.raises(OSError):create_session_root(output,'2026-11-13',cfg())
    assert list(outside.iterdir())==[]

@pytest.mark.skipif(__import__('os').name!='posix',reason='POSIX nofollow directory descriptors')
def test_posix_creator_preserves_old_claim_and_refuses_existing_revision(tmp_path):
    from scripts.strategy_archive_identity import create_session_root
    legacy=tmp_path/'2026-11-13';legacy.mkdir();(legacy/'claim').write_text('old')
    root=create_session_root(tmp_path,'2026-11-13',cfg())
    assert root.is_dir() and (legacy/'claim').read_text()=='old'
    with pytest.raises(FileExistsError):create_session_root(tmp_path,'2026-11-13',cfg())


def test_market_capture_uses_explicit_cache_root(monkeypatch,tmp_path):
    from scripts import daily_strategy_scan as daily, strategy_universe as universe
    seen={}
    monkeypatch.setattr(universe,'market_config',lambda cfg:True)
    monkeypatch.setattr(universe,'capture_market',lambda *a,**kw:seen.update(kw))
    cache=tmp_path/'market-cache'
    daily.capture(None,cfg(),[],"2026-11-13",session_root(tmp_path,"2026-11-13",cfg()),now=None,cache_dir=cache)
    assert seen['cache_dir']==cache


def test_frozen_companion_lifecycle_module_is_unchanged():
    import hashlib
    from pathlib import Path
    path=Path(__file__).parents[1]/'scripts/bull_flag_companion.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='a8602c85f0dd49dc2db3ef4a0832095b9c0166570e8420db2e4a9cdc65c11c07'
