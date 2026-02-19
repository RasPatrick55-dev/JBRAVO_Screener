import pytest

pytestmark = pytest.mark.alpaca_optional


import pandas as pd

from scripts import export_latest_candidates as mod


def test_export_latest_candidates_writes_canonical_header(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-02-01T14:30:00Z",
                "symbol": "AAPL",
                "score": 1.23,
                "exchange": "NASDAQ",
                "close": 100.0,
                "volume": 1000,
                "universe_count": 10,
                "score_breakdown": "{}",
                "entry_price": 100.0,
                "adv20": 500000,
                "atrp": 2.1,
                "source": "screener",
                "extra": "ignore",
            }
        ]
    )
    monkeypatch.setattr(
        mod, "get_latest_screener_candidates", lambda run_date: (frame, "2026-02-01T15:00:00Z")
    )
    out = tmp_path / "latest_candidates.csv"
    rows = mod.export_latest_candidates("2026-02-01", out)
    assert rows == 1
    text = out.read_text(encoding="utf-8").splitlines()
    assert (
        text[0]
        == "timestamp,symbol,score,exchange,close,volume,universe_count,score_breakdown,entry_price,adv20,atrp,source"
    )
    assert text[1].startswith("2026-02-01T14:30:00Z,AAPL,")
