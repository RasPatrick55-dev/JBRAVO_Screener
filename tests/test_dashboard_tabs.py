from dashboards import dashboard_app


import pytest


@pytest.mark.alpaca_optional
def test_trade_performance_tab_present():
    tabs = dashboard_app.build_tabs()
    tab_ids = [tab.tab_id for tab in tabs.children]
    labels = [tab.label for tab in tabs.children]
    assert "tab-trade-performance" in tab_ids
    assert any(str(label).lower().strip() == "trade performance" for label in labels)
