"""Real candidate entry with fake HTTP and owned temporary files only."""
import importlib.util
import os
from contextlib import contextmanager
import json
import logging
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import requests
import pandas  # Existing reader fixture restores sys.modules.

pytestmark = pytest.mark.alpaca_optional

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def isolated_provider_environment():
    def affected(name):
        return name.startswith(("APCA_", "ALPACA_")) or name == "JBR_EXEC_PAPER"
    before = {key: value for key, value in os.environ.items() if affected(key)}
    try:
        for key in list(os.environ):
            if affected(key):
                del os.environ[key]
        yield
    finally:
        for key in list(os.environ):
            if affected(key):
                del os.environ[key]
        os.environ.update(before)
        after = {key: value for key, value in os.environ.items() if affected(key)}
        if after != before:
            raise AssertionError("PROVIDER_ENVIRONMENT_RESTORE_FAILED")


@pytest.fixture
def provider_environment():
    with isolated_provider_environment():
        yield


@pytest.fixture
def probe(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('candidate_check_connection', ROOT/'scripts/check_connection.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for key, value in dict(BASE_DIR=tmp_path, DATA_DIR=tmp_path/'data', LOGS_DIR=tmp_path/'logs',
            CONNECTION_HEALTH_JSON=tmp_path/'data/connection_health.json',
            ERROR_LOG=tmp_path/'logs/error.log', LOCAL_LOG=tmp_path/'logs/connection_health.log').items():
        monkeypatch.setattr(m, key, value)
    logger = logging.getLogger('connection_health')
    for h in logger.handlers[:]: h.close(); logger.removeHandler(h)
    monkeypatch.setenv('APCA_API_BASE_URL', m.PAPER_URL)
    monkeypatch.setenv('APCA_DATA_API_BASE_URL', m.DATA_URL)
    monkeypatch.setenv('APCA_API_KEY_ID', 'SYNTHETIC_PRIVATE_KEY')
    monkeypatch.setenv('APCA_API_SECRET_KEY', 'SYNTHETIC_PRIVATE_SECRET')
    monkeypatch.setenv('ALPACA_DATA_FEED', 'iex')
    monkeypatch.delenv('JBR_EXEC_PAPER', raising=False)
    monkeypatch.setattr(m, '_load_dotenv_if_needed', lambda: None)
    yield m
    for h in logger.handlers[:]: h.close(); logger.removeHandler(h)


class Response:
    def __init__(self, status): self.status_code=status; self.close=Mock()
    @property
    def text(self): raise AssertionError('PRIVATE_BODY_READ')
    @property
    def content(self): raise AssertionError('PRIVATE_BODY_READ')
    def json(self): raise AssertionError('PRIVATE_BODY_READ')


def test_actual_entry_two_fixed_gets_no_body_and_canonical_reader(probe, monkeypatch, tmp_path):
    responses = [Response(200), Response(403)]
    get = Mock(side_effect=responses); monkeypatch.setattr(probe.requests, 'get', get)
    result = probe.run_probe()
    assert get.call_count == 2 and result['trading_ok'] is True and result['data_ok'] is False
    a, b = get.call_args_list
    assert a.args == (probe.PAPER_URL+'/v2/account',) and a.kwargs['params'] is None
    assert b.args == (probe.DATA_URL+'/v2/stocks/bars/latest',) and b.kwargs['params'] == {'symbols':'AAPL','feed':'iex'}
    for call in (a,b):
        assert call.kwargs['timeout']==(3,5) and call.kwargs['allow_redirects'] is False and call.kwargs['stream'] is True
    assert all(r.close.call_count==1 for r in responses)
    saved = json.loads(probe.CONNECTION_HEALTH_JSON.read_text())
    assert saved == result and not list(probe.DATA_DIR.glob('.connection_health.*'))
    assert not (probe.DATA_DIR/'health/connectivity.json').exists()
    for p in (probe.CONNECTION_HEALTH_JSON, probe.LOCAL_LOG, probe.ERROR_LOG):
        assert 'SYNTHETIC_PRIVATE' not in p.read_text()
    import test_saved_connection_timestamp as h
    reader=h.reload_data_io(monkeypatch,tmp_path)
    snapshot=reader.connection_check_snapshot(reader.load_connection_health(probe.DATA_DIR),now=datetime.now(timezone.utc))
    assert snapshot['status']=='fresh' and snapshot['trading_ok'] is True and snapshot['source']=='connection_health.json'


@pytest.mark.parametrize('status',[301,302,307,401,429,500])
def test_http_failure_or_redirect_is_closed_no_retry_or_body(probe, monkeypatch, status):
    response=Response(status); get=Mock(return_value=response)
    monkeypatch.setattr(probe.requests,'get',get)
    result=probe._probe_trading(Mock())
    assert result==dict(ok=False,status=status,message='HTTP_STATUS_NOT_200')
    assert get.call_count==1 and response.close.call_count==1


def test_network_failure_and_close_failure_do_not_export_exception(probe, monkeypatch):
    monkeypatch.setattr(probe.requests,'get',Mock(side_effect=requests.ConnectionError('PRIVATE_CANARY')))
    logger=Mock(); result=probe._probe_trading(logger)
    assert result['message']=='REQUEST_FAILED' and 'PRIVATE_CANARY' not in str(logger.mock_calls)
    response=Response(200); response.close.side_effect=OSError('PRIVATE_CANARY')
    monkeypatch.setattr(probe.requests,'get',Mock(return_value=response))
    assert probe._probe_trading(logger)==dict(ok=False,status=0,message='RESPONSE_CLOSE_FAILED')
    assert 'PRIVATE_CANARY' not in str(logger.mock_calls)


@pytest.mark.parametrize('key,value',[('APCA_API_BASE_URL','https://api.alpaca.markets'),
    ('APCA_API_BASE_URL','https://paper-api.alpaca.markets/'),('JBR_EXEC_PAPER','false')])
def test_wrong_account_context_refused_before_any_effect(probe, monkeypatch, key, value):
    monkeypatch.setenv(key,value); get=Mock(); monkeypatch.setattr(probe.requests,'get',get)
    with pytest.raises(ValueError,match='PAPER_CONTEXT_REQUIRED'): probe.run_probe()
    get.assert_not_called(); assert not probe.DATA_DIR.exists() and not probe.LOGS_DIR.exists()


@pytest.mark.parametrize('key,value,reason',[('APCA_DATA_API_BASE_URL','https://unrelated.invalid','DATA_ENDPOINT_REQUIRED'),
    ('ALPACA_DATA_FEED','unknown','DATA_FEED_UNSUPPORTED'),('APCA_API_KEY_ID','','CREDENTIALS_MISSING')])
def test_data_configuration_refusal_does_not_send_credentials(probe, monkeypatch, key, value, reason):
    monkeypatch.setenv(key,value); get=Mock(); monkeypatch.setattr(probe.requests,'get',get)
    assert probe._probe_data(Mock())==dict(ok=False,status=0,message=reason)
    get.assert_not_called()


def test_atomic_publish_failure_preserves_previous_artifact(probe, monkeypatch):
    probe._ensure_dirs(); probe.CONNECTION_HEALTH_JSON.write_text('BEFORE')
    monkeypatch.setattr(probe.os,'replace',Mock(side_effect=OSError('PRIVATE_CANARY')))
    with pytest.raises(OSError): probe._write_health(dict(trading_ok=True))
    assert probe.CONNECTION_HEALTH_JSON.read_text()=='BEFORE' and not list(probe.DATA_DIR.glob('.connection_health.*'))


def test_shared_env_parser_named_files_no_shell_or_values(monkeypatch,tmp_path):
    spec=importlib.util.spec_from_file_location('env_loader_candidate',ROOT/'scripts/check_connection.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    monkeypatch.setattr(m,'BASE_DIR',tmp_path)
    env=ModuleType('utils.env'); env._load_env_file=Mock(); env._normalize_env_aliases=Mock()
    monkeypatch.setitem(sys.modules,'utils.env',env)
    m._load_dotenv_if_needed()
    assert env._load_env_file.call_args_list[0].args==(Path.home()/'.config/jbravo/.env',)
    assert env._load_env_file.call_args_list[1].args==(tmp_path/'.env',)
    assert all(c.kwargs=={'override':False} for c in env._load_env_file.call_args_list)
    env._normalize_env_aliases.assert_called_once()


def test_real_cli_entry_reports_closed_loader_failure(monkeypatch,capsys):
    env=ModuleType('utils.env'); env._load_env_file=Mock(side_effect=RuntimeError('PRIVATE_ENV_CANARY'))
    env._normalize_env_aliases=Mock(); monkeypatch.setitem(sys.modules,'utils.env',env)
    get=Mock(); monkeypatch.setattr(requests,'get',get)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT/'scripts/check_connection.py'),run_name='__main__')
    assert stopped.value.code==1 and capsys.readouterr().out.strip()==json.dumps({'status':'failed','reason':'HEALTH_CHECK_FAILED'})
    get.assert_not_called()


def test_real_cli_entry_success_with_owned_app_and_fake_http(monkeypatch,tmp_path,capsys):
    script=tmp_path/'cli-app/scripts/check_connection.py'; script.parent.mkdir(parents=True)
    script.write_bytes((ROOT/'scripts/check_connection.py').read_bytes())
    env=ModuleType('utils.env'); env._load_env_file=Mock(); env._normalize_env_aliases=Mock()
    monkeypatch.setitem(sys.modules,'utils.env',env)
    monkeypatch.setenv('APCA_API_BASE_URL','https://paper-api.alpaca.markets')
    monkeypatch.setenv('APCA_DATA_API_BASE_URL','https://data.alpaca.markets')
    monkeypatch.setenv('APCA_API_KEY_ID','SYNTHETIC_PRIVATE_KEY'); monkeypatch.setenv('APCA_API_SECRET_KEY','SYNTHETIC_PRIVATE_SECRET')
    monkeypatch.setenv('ALPACA_DATA_FEED','iex'); monkeypatch.delenv('JBR_EXEC_PAPER',raising=False)
    logger=logging.getLogger('connection_health')
    for h in logger.handlers[:]: h.close(); logger.removeHandler(h)
    get=Mock(side_effect=[Response(200),Response(200)]); monkeypatch.setattr(requests,'get',get)
    try:
        with pytest.raises(SystemExit) as completed:
            runpy.run_path(str(script),run_name='__main__')
        assert completed.value.code==0
        result=json.loads((script.parents[1]/'data/connection_health.json').read_text())
        assert result['trading_ok'] is result['data_ok'] is True and get.call_count==2
        assert capsys.readouterr().out==''
    finally:
        for h in logger.handlers[:]: h.close(); logger.removeHandler(h)


@pytest.mark.parametrize('case,exit_code,expected_gets',[
    ('success',0,2),('manual_parser_success',0,2),
    ('trading_http',1,2),('data_http',1,2),('network_failure',1,2),
    ('conflicting_paper_context',1,0),('missing_credentials',1,0),
    ('bad_data_endpoint',1,1),('bad_feed',1,1)])
def test_real_cli_and_real_env_loader_synthetic_configuration_only(monkeypatch,tmp_path,capsys,provider_environment,case,exit_code,expected_gets):
    """No replacement of main/run_probe/env parser; only home/root and HTTP are fixtures."""
    app=tmp_path/'owned-app';script=app/'scripts/check_connection.py';script.parent.mkdir(parents=True)
    script.write_bytes((ROOT/'scripts/check_connection.py').read_bytes())
    home=tmp_path/'owned-home';user_env=home/'.config/jbravo/.env';user_env.parent.mkdir(parents=True)
    key='' if case=='missing_credentials' else 'SYNTHETIC_PRIVATE_USER_KEY'
    secret='' if case=='missing_credentials' else 'SYNTHETIC_PRIVATE_USER_SECRET'
    base='https://api.alpaca.markets' if case=='conflicting_paper_context' else 'https://paper-api.alpaca.markets/v2/'
    user_env.write_text(f'APCA_API_KEY_ID="{key}"\nAPCA_API_SECRET_KEY=\'{secret}\'\nALPACA_API_BASE_URL="{base}"\n',encoding='utf-8')
    data_base='https://unrelated.invalid' if case=='bad_data_endpoint' else 'https://data.alpaca.markets/'
    feed='unknown' if case=='bad_feed' else 'iex'
    (app/'.env').write_text('APCA_API_KEY_ID="SYNTHETIC_PRIVATE_REPO_KEY"\nAPCA_API_SECRET_KEY="SYNTHETIC_PRIVATE_REPO_SECRET"\n'+f'APCA_DATA_API_BASE_URL="{data_base}"\nALPACA_DATA_FEED="{feed}"\n',encoding='utf-8')
    # Real helper, including its actual python-dotenv/manual paths. It reads only
    # the two explicit fixture files because home and script root are owned.
    spec=importlib.util.spec_from_file_location('utils.env',ROOT/'utils/env.py')
    env=importlib.util.module_from_spec(spec);spec.loader.exec_module(env)
    if case=='manual_parser_success':monkeypatch.setattr(env,'load_dotenv',None)
    monkeypatch.setitem(sys.modules,'utils.env',env)
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:home))
    outcomes=[Response(401 if case=='trading_http' else 200),Response(503 if case=='data_http' else 200)]
    if case=='network_failure':outcomes[0]=requests.ConnectionError('PRIVATE_EXCEPTION_CANARY')
    get=Mock(side_effect=outcomes);monkeypatch.setattr(requests,'get',get)
    logger=logging.getLogger('connection_health')
    for h in logger.handlers[:]:h.close();logger.removeHandler(h)
    try:
        with pytest.raises(SystemExit) as finished:runpy.run_path(str(script),run_name='__main__')
        assert finished.value.code==exit_code and get.call_count==expected_gets
        out=capsys.readouterr()
        assert 'PRIVATE_' not in out.out+out.err
        artifact=app/'data/connection_health.json'
        if case=='conflicting_paper_context':
            assert not artifact.exists() and not (app/'logs').exists()
        else:
            saved=json.loads(artifact.read_text())
            assert saved['trading_ok']==(case not in ('trading_http','network_failure','missing_credentials'))
            assert saved['data_ok']==(case not in ('data_http','missing_credentials','bad_data_endpoint','bad_feed'))
            for file in (artifact,app/'logs/connection_health.log',app/'logs/error.log'):
                assert 'PRIVATE_' not in file.read_text()
        for call in get.call_args_list:
            assert call.kwargs['headers']['APCA-API-KEY-ID']==key
            assert call.kwargs['headers']['APCA-API-SECRET-KEY']==secret
    finally:
        for h in logger.handlers[:]:h.close();logger.removeHandler(h)


def test_provider_environment_restores_existing_and_newly_added_keys(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "SYNTHETIC_BEFORE")
    monkeypatch.delenv("ALPACA_ISOLATION_NEW_CANARY", raising=False)
    with isolated_provider_environment():
        assert "APCA_API_KEY_ID" not in os.environ
        os.environ["APCA_API_KEY_ID"] = "SYNTHETIC_DURING"
        os.environ["ALPACA_ISOLATION_NEW_CANARY"] = "SYNTHETIC_NEW"
    assert os.environ["APCA_API_KEY_ID"] == "SYNTHETIC_BEFORE"
    assert "ALPACA_ISOLATION_NEW_CANARY" not in os.environ
