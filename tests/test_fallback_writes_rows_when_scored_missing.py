import logging

import pandas as pd
import pytest

from scripts import fallback_candidates


@pytest.mark.alpaca_optional
def test_fallback_static_writes_three_rows_when_scored_missing(tmp_path, caplog, monkeypatch):
    base_dir = tmp_path
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(fallback_candidates, "PROJECT_ROOT", base_dir)
    monkeypatch.setattr(fallback_candidates, "DATA_DIR", data_dir)
    monkeypatch.setattr(
        fallback_candidates,
        "LATEST_CANDIDATES",
        data_dir / "latest_candidates.csv",
    )
    monkeypatch.setattr(
        fallback_candidates,
        "SCORED_CANDIDATES",
        data_dir / "scored_candidates.csv",
    )

    caplog.set_level(logging.INFO, logger=fallback_candidates.LOGGER.name)

    frame, source = fallback_candidates.build_latest_candidates(base_dir)

    assert frame.shape[0] == 3
    assert list(frame.columns) == list(fallback_candidates.CANONICAL_COLUMNS)
    assert (frame["source"] == "fallback:static").all()
    assert source == "static"

    latest_path = data_dir / "latest_candidates.csv"
    assert latest_path.exists()
    latest = pd.read_csv(latest_path)
    assert latest.shape[0] == 3
    assert list(latest.columns) == list(fallback_candidates.CANONICAL_COLUMNS)
    assert (latest["source"] == "fallback:static").all()

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "[INFO] FALLBACK_CHECK rows_out=3 source=fallback:static" in message for message in messages
    )
