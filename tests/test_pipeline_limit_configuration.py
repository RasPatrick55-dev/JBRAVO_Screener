"""The native deployment config preserves published pipeline limits, offline."""
import json
from pathlib import Path
import pytest
from scripts import strategy_execution_runtime as runtime
from scripts import strategy_execution as rules

pytestmark = pytest.mark.alpaca_optional


def test_native_configuration_matches_pipeline_contract():
    config = runtime.load_config()
    shared = json.loads((runtime.ROOT / 'config/shared_pricing_execution.json').read_text())
    assert config == shared
    assert config['execution_model'] == rules.PIPELINE_LIMIT_MODEL
    assert config['extra_limit_buffer_pct'] == 0
    assert config['limit_source'] == 'PIPELINE_PUBLISHED_LIMIT'
    assert 'limit_buffer_pct' not in config
    # Feed selection belongs to data providers, not an entry-only price policy.
    assert 'reference_feed' not in config


@pytest.mark.parametrize('change', [
    {'execution_model': rules.EXECUTION_MODEL},
    {'extra_limit_buffer_pct': .5},
    {'limit_buffer_pct': .5},
    {'limit_source': 'CAPTURED_AFTER_HOURS_CLOSE'},
    {'schema': 'jbravo.premarket-paper-execution-config.v1'},
    {'reference_feed': 'sip'},
    {'paper_only': False},
    {'allocation_pct': .1},
])
def test_configuration_drift_fails_closed(tmp_path, change):
    config = runtime.load_config() | change
    target = tmp_path / runtime.CONFIG
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='strategy_execution_configuration_changed'):
        runtime.load_config(tmp_path)

