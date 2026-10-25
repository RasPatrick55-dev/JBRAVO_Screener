"""Offline library, scan-to-paper trace, readiness and causality contracts."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys

import pandas as pd
import pytest

from scripts import strategy_library, strategy_scanner
from scripts.research_simulation import ResearchPortfolio
from scripts.research_strategies import CHALLENGER
from test_research_testing import make_capture, simple_frame

pytestmark = pytest.mark.alpaca_optional


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError('Strategy scanner cannot contact a broker, provider or database')
    monkeypatch.setattr(socket.socket, 'connect', reject)
    monkeypatch.setattr(socket, 'create_connection', reject)


def test_only_first_strategy_enabled_and_definition_bound():
    definitions = strategy_library.list_strategies()
    assert len(definitions) == 1
    definition = definitions[0]
    assert definition['id'] == CHALLENGER and definition['version'] == '2'
    assert definition['paper_scan_enabled'] is True and definition['research_qualified'] is False
    assert len(definition['definition_sha256']) == 64 and len(definition['buy_rules']) == 8


def test_disabled_new_registration_does_not_activate_existing_daily_scan(monkeypatch):
    first = strategy_library.get_strategy(CHALLENGER)
    monkeypatch.setitem(strategy_library._STRATEGIES, 'future', replace(first, id='future', paper_scan_enabled=False))
    assert [s.id for s in strategy_library.select_strategies()] == [CHALLENGER]
    with pytest.raises(ValueError, match='not enabled'):
        strategy_library.select_strategies(['future'])


def test_catalog_only_source_change_preserves_existing_strategy_identity(monkeypatch):
    first = strategy_library.get_strategy(CHALLENGER)
    original = first.describe()
    read = Path.read_bytes
    def changed(path):
        raw = read(path)
        return raw + b'\n# another catalog entry\n' if path.name == 'strategy_library.py' else raw
    monkeypatch.setattr(Path, 'read_bytes', changed)
    later = first.describe()
    assert original['definition_sha256'] == later['definition_sha256']
    assert original['source_sha256']['strategy_library.py'] != later['source_sha256']['strategy_library.py']


@pytest.mark.parametrize('ids', [[], [CHALLENGER, CHALLENGER], ['unknown']])
def test_unknown_empty_duplicate_selections_reject(ids):
    with pytest.raises(ValueError):
        strategy_library.select_strategies(ids)


def consumer(data, *, scanner=True):
    options = dict(initial_cash=1000., alloc_pct=.5, max_positions=1, top_n=1,
        trail_pct=None, atr_multiple=0, max_hold_days=1000, enable_macd_exit=False,
        enable_partial_exit=False, enable_candlestick_exit=False, enable_ema_exit=False,
        enable_rsi_divergence=False, enable_trailing_exit=False)
    if scanner:
        return strategy_scanner.ScannerPortfolio(data, strategy=strategy_library.get_strategy(CHALLENGER),
                                                 episode_id='frozen-episode', **options)
    return ResearchPortfolio(data, method=CHALLENGER, **options)


def data_for_consumer():
    frame = simple_frame(count=4)
    frame['score'] = 5.
    frame['entry_details'] = [dict(pullback_start='2024-01-01', signal=True, failures=[]) for _ in range(4)]
    return {'AAA': frame.copy(deep=True), 'BBB': frame.copy(deep=True)}


def test_same_consumer_preserves_unselected_consumption_and_rearm():
    actual = consumer(data_for_consumer())
    expected = consumer(data_for_consumer(), scanner=False)
    actual.run()
    expected.run()
    assert actual.signals == expected.signals and actual.summary() == expected.summary()
    assert actual.last_reclaims == expected.last_reclaims
    assert len(actual.intents) == 1 and actual.intents[0]['symbol'] == 'AAA'
    first = actual.daily_scans[:2]
    assert all(row['consumed'] for row in first)
    assert [row['selected'] for row in first] == [True, False]
    assert any(row['reason'] == 'REARM_REQUIRES_PULLBACK_AFTER_CONSUMED_CLOSE' for row in actual.daily_scans)
    assert actual.fills[0]['signal_id'] == actual.intents[0]['signal_id']
    assert actual.fills[0]['session'] > actual.intents[0]['signal_session']
    assert actual.intents[0]['order_submission_authorized'] is False


def test_prefix_signal_ids_and_paper_fills_survive_future_append():
    longer = data_for_consumer()
    shorter = {symbol: frame.iloc[:2].copy() for symbol, frame in longer.items()}
    early, late = consumer(shorter), consumer(longer)
    early.run()
    late.run()
    assert early.daily_scans == [x for x in late.daily_scans if x['session'] <= '2024-01-03']
    assert early.fills == [x for x in late.fills if x['session'] <= '2024-01-03']


def test_terminal_selected_setup_is_saved_as_proposal_not_fictitious_fill():
    frame = data_for_consumer()['AAA'].iloc[:1]
    bt = consumer({'AAA': frame})
    bt.run()
    assert len(bt.intents) == 1 and not bt.fills and not bt.pending_entries
    assert bt.terminal_entries[0]['reason'] == 'NO_NEXT_SESSION_INSIDE_EVALUATION_WINDOW'


def test_library_uses_exact_research_feature_preparation():
    from test_research_testing import featured_source
    from scripts.research_strategies import prepare_signals
    frame = featured_source()
    fold = dict(warmup_start='2023-01-03', test_start=frame.index[270].strftime('%Y-%m-%d'),
                test_end=frame.index[-1].strftime('%Y-%m-%d'))
    actual = strategy_library.get_strategy(CHALLENGER).prepare(frame, frame, fold)
    expected = prepare_signals(frame, frame, CHALLENGER, fold)
    pd.testing.assert_frame_equal(actual, expected)


def test_real_capture_adapter_writes_valid_empty_day_and_bound_trace(tmp_path):
    spec, _, _, _ = make_capture(tmp_path)
    day = spec['folds'][0]['test_end']
    output = tmp_path/'scan'
    result = strategy_scanner.run_scan(spec, as_of=day, output=output, cost_case_id='base')
    assert result['status'] == 'COMPLETED_PAPER_SIMULATION'
    assert result['broker_operations'] == 0 and result['order_submission_authorized'] is False
    assert result['top_candidates'] == []
    handoff = json.loads((output/'paper_handoff.json').read_bytes())
    assert handoff['proposals'] == [] and handoff['paper_only'] is True
    csv = pd.read_csv(output/'top_candidates.csv')
    assert csv.empty and 'strategy_id' in csv.columns
    for binding in json.loads((output/'manifest.json').read_bytes())['bindings']:
        body = (output/binding['path']).read_bytes()
        assert len(body) == binding['bytes']
        assert strategy_scanner.hashlib.sha256(body).hexdigest() == binding['sha256']
    with pytest.raises(FileExistsError):
        strategy_scanner.run_scan(spec, as_of=day, output=output, cost_case_id='base')


@pytest.mark.parametrize('moment', [datetime(2026, 5, 26, 23, tzinfo=timezone.utc), datetime(2026, 5, 26, 23)])
def test_incomplete_session_or_naive_clock_rejects_before_claim(tmp_path, moment):
    spec, _, _, _ = make_capture(tmp_path)
    with pytest.raises(ValueError, match='cutoff'):
        strategy_scanner.run_scan(spec, as_of='2026-05-26', output=tmp_path/'scan', cost_case_id='base', now=moment)
    assert not (tmp_path/'scan').exists()


def test_unknown_cost_or_episode_rejects_before_claim(tmp_path):
    spec, _, _, _ = make_capture(tmp_path)
    day = spec['folds'][0]['test_end']
    with pytest.raises(ValueError, match='cost case'):
        strategy_scanner.run_scan(spec, as_of=day, output=tmp_path/'scan', cost_case_id='missing')
    assert not (tmp_path/'scan').exists()


def test_changed_input_fails_and_preserves_claim(tmp_path):
    spec, folder, _, _ = make_capture(tmp_path)
    with (folder/'alpaca_split_daily.csv').open('ab') as handle:
        handle.write(b' ')
    with pytest.raises(ValueError, match='hash/size mismatch'):
        strategy_scanner.run_scan(spec, as_of=spec['folds'][0]['test_end'], output=tmp_path/'scan', cost_case_id='base')
    assert (tmp_path/'scan/claim.json').exists() and (tmp_path/'scan/failure.json').exists()
    assert not (tmp_path/'scan/paper_handoff.json').exists()


def test_screener_main_library_route_skips_private_bootstrap_and_account_probe(monkeypatch, capsys):
    from scripts import screener
    def reject(*args, **kwargs):
        raise AssertionError('Library route must not bootstrap credentials or broker clients')
    monkeypatch.setattr(screener, '_bootstrap_env', reject)
    monkeypatch.setattr(screener, 'assert_alpaca_creds', reject)
    monkeypatch.setattr(screener, 'probe_trading_only', reject)
    assert screener.main(['--mode', 'strategy-library']) == 0
    assert json.loads(capsys.readouterr().out)['strategies'][0]['id'] == CHALLENGER


def test_actual_screener_cli_does_not_import_legacy_clients_or_credentials():
    repo = Path(__file__).resolve().parents[1]
    code = """import builtins,runpy,socket,sys
sys.path.insert(0,sys.argv[1])
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'utils','alpaca','requests','psycopg2','dotenv'}:
        raise AssertionError('legacy import attempted: '+name)
    return original(name,*args,**kwargs)
def blocked(*args,**kwargs):
    raise AssertionError('network attempted')
builtins.__import__=guarded
socket.socket.connect=socket.create_connection=blocked
sys.argv=['scripts.screener','--mode','strategy-library']
runpy.run_module('scripts.screener',run_name='__main__')
"""
    result = subprocess.run([sys.executable,'-I','-c',code,str(repo)],capture_output=True,text=True,timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['strategies'][0]['id'] == CHALLENGER
    assert 'credentials' not in result.stderr.lower()


def test_existing_main_generator_arguments_survive_mode_probe(monkeypatch):
    from scripts import screener
    monkeypatch.setattr(screener,'_bootstrap_env',lambda:None)
    monkeypatch.setattr(screener,'_ensure_logger',lambda:None)
    monkeypatch.setattr(screener.logger_utils,'make_screener_file_handler',lambda:__import__('logging').NullHandler())
    seen = []
    class Parsed(Exception): pass
    def parse(args):
        seen.extend(args)
        raise Parsed()
    monkeypatch.setattr(screener,'parse_args',parse)
    with pytest.raises(Parsed):
        screener.main(iter(['--source-csv','fixture.csv']))
    assert seen == ['--mode','screener','--source-csv','fixture.csv']
