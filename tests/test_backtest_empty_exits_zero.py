from scripts import backtest


def test_backtest_empty_exits_zero(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "top_candidates.csv").write_text("symbol\n", encoding="utf-8")

    monkeypatch.setattr(backtest, "BASE_DIR", str(tmp_path))

    rc = backtest.main()
    assert rc == 0
