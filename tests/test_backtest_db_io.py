from datetime import date, datetime
import logging

import pandas as pd
import pytest

from scripts import backtest


@pytest.mark.alpaca_optional
def test_main_uses_csv_when_db_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    csv_path = tmp_path / "candidates.csv"
    pd.DataFrame({"symbol": ["AAA", "BBB"]}).to_csv(csv_path, index=False)

    captured = {}

    def _fake_run_backtest(symbols, **kwargs):
        captured["symbols"] = symbols
        captured["export_csv"] = kwargs["export_csv"]
        captured["enable_db"] = kwargs["enable_db"]
        return {"tested": len(symbols), "skipped": 0}

    monkeypatch.setattr(backtest, "run_backtest", _fake_run_backtest)

    rc = backtest.main(["--source", str(csv_path)])
    assert rc == 0
    assert captured["symbols"] == ["AAA", "BBB"]
    assert captured["export_csv"] is True
    assert captured["enable_db"] is False


@pytest.mark.alpaca_optional
def test_run_backtest_upserts_db(monkeypatch, caplog):
    base = list(range(300))
    dummy_df = pd.DataFrame(
        {
            "open": [10.0 + i * 0.01 for i in base],
            "high": [10.5 + i * 0.01 for i in base],
            "low": [9.5 + i * 0.01 for i in base],
            "close": [10.2 + i * 0.01 for i in base],
            "volume": [1000 + i for i in base],
            "ma50": [10.0 for _ in base],
            "ma200": [9.5 for _ in base],
            "rsi": [55 for _ in base],
            "macd": [0.1 for _ in base],
            "macd_signal": [0.05 for _ in base],
            "macd_hist": [0.05 for _ in base],
            "adx": [25 for _ in base],
            "aroon_up": [80 for _ in base],
            "aroon_down": [20 for _ in base],
            "obv": [float(i) for i in base],
            "vol_avg30": [900 for _ in base],
            "month_high": [10.2 + i * 0.01 for i in base],
            "ema20": [10.0 for _ in base],
            "ATR14": [0.5 for _ in base],
            "atr": [0.5 for _ in base],
        }
    )

    class DummyBT:
        def __init__(self, *_args, **_kwargs):
            self._equity = pd.DataFrame({"timestamp": [1, 2, 3], "equity": [100, 101, 102]})

        def run(self):
            return None

        def results(self):
            return pd.DataFrame(
                {
                    "symbol": ["ABC", "ABC"],
                    "entry_time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                    "exit_time": [datetime(2024, 1, 3), datetime(2024, 1, 4)],
                    "pnl": [1.0, -0.5],
                    "entry_price": [10.0, 10.0],
                    "qty": [1, 1],
                    "exit_reason": ["rule", "rule"],
                }
            )

        def equity(self):
            return self._equity

    monkeypatch.setattr(backtest, "get_data", lambda *_args, **_kwargs: dummy_df)
    monkeypatch.setattr(backtest, "compute_indicators", lambda df: df)
    monkeypatch.setattr(backtest, "composite_score", lambda df: pd.Series(1, index=df.index))
    monkeypatch.setattr(backtest, "prepare_series", lambda df: df)
    monkeypatch.setattr(backtest, "PortfolioBacktester", DummyBT)

    inserted = {}

    def _fake_insert(run_date, df_results):
        inserted["run_date"] = run_date
        inserted["rows"] = len(df_results)
        return True

    monkeypatch.setattr(backtest.db, "insert_backtest_results", _fake_insert)

    caplog.set_level(logging.INFO)
    result = backtest.run_backtest(
        ["ABC"],
        max_symbols=None,
        max_days=None,
        quick=False,
        run_date=date(2024, 1, 1),
        export_csv=False,
        enable_db=True,
    )

    assert result["tested"] == 1
    assert inserted["run_date"] == date(2024, 1, 1)
    assert inserted["rows"] == 1
    assert any("BACKTEST_DB_OK" in message for message in caplog.messages)
