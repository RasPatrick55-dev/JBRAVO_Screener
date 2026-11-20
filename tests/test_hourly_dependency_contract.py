"""Offline regressions for retained timestamp and source-contract semantics."""
from copy import deepcopy
from pathlib import Path
import ast
import hashlib
import json
import pytest
from scripts import strategy_data_projection as projection
from scripts import strategy_fill_auto_sync as auto
from scripts import strategy_fill_linkage as core
from test_data_projection import cohort, state, EXIT
from test_strategy_fill_linkage import A

pytestmark = pytest.mark.alpaca_optional


def fixture():
    f = cohort()
    f['orders'][1]['observation']['updated_at'] = '2026-11-05T09:30:00.123456789Z'
    snapshots = {o['observation']['id']: deepcopy(o['observation']) for o in f['orders']}
    snapshots[EXIT]['updated_at'] = '2026-11-05T09:30:00.123456Z'
    return f, snapshots


def plan(f, snapshots):
    return projection.snapshot_plan(state(f), snapshots, account_id=A,
                                    observed_at='2026-11-05T10:00:00Z')


def test_lower_precision_repeat_keeps_original_no_update():
    f, snapshots = fixture()
    before = deepcopy(f)
    p = plan(f, snapshots)
    assert p['orders'] == []
    assert len(p['precision_preserved_observation_evidence']) == 1
    assert p['precision_preserved_observation_evidence'][0]['chronology_certified'] is False
    assert f == before


@pytest.mark.parametrize('field,value', [
    ('filled_avg_price', '999'), ('status', 'new'), ('filled_qty', '1'),
    ('updated_at', '2026-11-05T09:30:00.123455Z'),
])
def test_precision_change_does_not_hide_material_or_nonoverlap_change(field, value):
    f, snapshots = fixture()
    snapshots[EXIT][field] = value
    with pytest.raises(ValueError):
        plan(f, snapshots)


def test_retained_source_files_match_reviewed_source_hashes():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'docs/verification/hourly-dependencies-20261114.json').read_text())
    # Keep the dated native readback intact. These two current files now
    # include the optional manual-sell disposition and have successor pins.
    successors = {
        "strategy_fill_auto_sync.py": "31ff5e5ba966da51e215c5f395fb60ec7ed4ec415e85cdc1b52ba9c67920bfdf",
        "strategy_fill_sync_adapter.py": "19da7c1bc0c28106ee552c8d67f7d8416cd989c9f9dd1e0a63dbe6ce46cf230a",
    }
    for name, entry in manifest['sources'].items():
        # These retained native modules have -text attributes to preserve exact pinned bytes.
        content = (root/'scripts'/name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == successors.get(name, entry["sha256"])


def test_no_import_time_provider_database_or_executor_dependency():
    root = Path(auto.__file__).parent
    for name in ('strategy_fill_auto_sync.py', 'strategy_fill_sync_adapter.py',
                 'strategy_fill_linkage.py', 'strategy_fill_linkage_store.py',
                 'strategy_data_projection.py'):
        tree = ast.parse((root/name).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith(('requests', 'psycopg', 'alpaca')) for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not any(a.name in ('db', 'fetch_account_activities', 'execute_trades',
                                          'monitor_positions') for a in node.names)


def test_empty_projection_never_opens_write_connection():
    def forbidden(*args, **kwargs):
        raise AssertionError('No DB access for empty plan')
    assert auto.write_projection({'orders': [], 'signals': []}, connect=forbidden,
                                 policy_lock=forbidden, recheck=forbidden) == {'status': 'NO_CHANGE'}
