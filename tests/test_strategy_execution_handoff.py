from datetime import datetime,timedelta
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from scripts import strategy_execution_handoff as handoff
from scripts import strategy_pipeline_store
from test_strategy_pipeline import NOW
from test_strategy_execution import positive

pytestmark=pytest.mark.alpaca_optional


def setup(tmp_path,monkeypatch):
    record,rows=positive()
    directory=tmp_path/'reports/pipeline_postflight';directory.mkdir(parents=True)
    raw=json.dumps(record).encode();path=directory/'latest.json';path.write_bytes(raw)
    expected=dict(status='ok',route='pullback_reclaim_strategy',screen_ready=True,
        report_binding=handoff.health._binding(raw))
    health=MagicMock(return_value=expected)
    monkeypatch.setattr(handoff.health,'check_health',health)
    cur=MagicMock();cur.__enter__.return_value=cur
    batch=record['batch']
    cur.fetchone.return_value=(record['run_id'],datetime.fromisoformat(record['session']['signal_session']).date(),
        NOW,len(rows),batch)
    conn=SimpleNamespace(readonly=True,autocommit=False,isolation_level=2,cursor=lambda:cur)
    frame=MagicMock(return_value=(pd.DataFrame(rows),NOW))
    monkeypatch.setattr(strategy_pipeline_store,'candidate_frame',frame)
    return conn,cur,health,frame,path


def test_one_repeatable_snapshot_requires_exact_latest_header_and_population(tmp_path,monkeypatch):
    conn,cur,health,frame,path=setup(tmp_path,monkeypatch)
    value=handoff.load_handoff(conn,base_dir=tmp_path,now=NOW+timedelta(seconds=10))
    assert value['status']=='PREPARED' and value['broker_operations']==0
    health.assert_called_once()
    assert health.call_args.kwargs['require_session'] is True
    assert health.call_args.kwargs['require_completion'] is True
    assert frame.call_args.kwargs['exact_run_ts']==NOW
    assert 'ORDER BY run_date DESC,run_ts_utc DESC LIMIT 1' in cur.execute.call_args.args[0]


@pytest.mark.parametrize('change',['readonly','isolation','primary','newer_database','changed_file'])
def test_missing_operational_proof_or_racing_publication_cannot_be_consumed(tmp_path,monkeypatch,change):
    conn,cur,health,frame,path=setup(tmp_path,monkeypatch)
    if change=='readonly':conn.readonly=False
    elif change=='isolation':conn.isolation_level=1
    elif change=='primary':health.return_value=dict(status='failed')
    elif change=='newer_database':
        old=cur.fetchone.return_value;cur.fetchone.return_value=('2'*32,*old[1:])
    else:
        original=frame.return_value
        def mutate(*a,**kw):path.write_bytes(path.read_bytes()+b' ');return original
        frame.side_effect=mutate
    with pytest.raises(ValueError):
        handoff.load_handoff(conn,base_dir=tmp_path,now=NOW+timedelta(seconds=10))
