from dashboards import dashboard_app


def test_tab_from_triggered_id_returns_tab_id_for_valid_inputs():
    assert dashboard_app.tab_from_triggered_id("tab-overview") == "tab-overview"
    assert dashboard_app.tab_from_triggered_id("tab-trade-performance") == "tab-trade-performance"
    assert dashboard_app.tab_from_triggered_id("trade-performance") == "tab-trade-performance"


def test_tab_from_triggered_id_ignores_unknown_trigger():
    assert dashboard_app.tab_from_triggered_id(None) is None
    assert dashboard_app.tab_from_triggered_id("unknown-component") is None
