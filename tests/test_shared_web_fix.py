from pathlib import Path
from datetime import datetime, timezone
import pytest
from tests.test_shared_web_api import web_case, NOW
from tests.test_shared_pricing_wiring import wiring
from tests.test_isolated_pricing_integration import fixture
from tests.test_pricing_adapter_prototype import contract, load
pytestmark=pytest.mark.alpaca_optional

@pytest.mark.parametrize('kind',['candidates','metrics','backtest'])
def test_isolated_web_copy_clears_readiness_on_batch_mismatch(web_case,kind):
 c=web_case;c.commit()
 fixed=load('shared_web_fixed',Path(__file__).resolve().parents[1]/'scripts/strategy_dashboard.py')
 fixed.snapshot=c.web.snapshot
 result=fixed.api_payload(kind,{'run_ts_utc':'2026-11-10T02:00:00+00:00'},now=NOW)
 assert result['status']=='ERROR' and result['rows']==[]
 assert result['entry_ready'] is False
 assert result['entry_block_reason']=='strategy_primary:batch_changed'
 assert result['entry_block_reasons']==['strategy_primary:batch_changed']
 assert result['computation_complete'] is False and result['completion_evidence'] is None
 assert result['order_submission_authorized'] is False
