from pathlib import Path

import pandas as pd

from scripts import fallback_candidates as fallback_mod


def test_build_latest_candidates_uses_scored_and_canonical_columns(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    scored_path = data_dir / "scored_candidates.csv"
    pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "score": 1.5,
                "exchange": "NYSE",
                "close": 10.0,
                "volume": 200_000,
                "universe count": 10,
                "score breakdown": "{}",
                "adv20": 1_000_000,
            },
            {
                "symbol": "BBB",
                "score": 3.2,
                "exchange": "NASDAQ",
                "close": 11.5,
                "volume": 400_000,
                "universe count": 10,
                "score breakdown": "{}",
                "adv20": 2_000_000,
            },
        ]
    ).to_csv(scored_path, index=False)

    frame, source = fallback_mod.build_latest_candidates(tmp_path, max_rows=2)

    latest_path = data_dir / "latest_candidates.csv"
    latest = pd.read_csv(latest_path)

    assert source == "scored"
    assert latest.columns.tolist() == list(fallback_mod.CANONICAL_COLUMNS)
    assert latest.iloc[0]["symbol"] == "BBB"
    assert latest.iloc[0]["source"] == "fallback:scored"
    assert frame.iloc[0]["symbol"] == "BBB"


def test_build_latest_candidates_invokes_atomic_write(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    calls: list[str] = []
    original = fallback_mod.write_csv_atomic

    def spy(path: str, rows):
        calls.append(Path(path).name)
        return original(path, rows)

    monkeypatch.setattr(fallback_mod, "write_csv_atomic", spy)

    frame, source = fallback_mod.build_latest_candidates(tmp_path)

    assert source == "static"
    assert "latest_candidates.csv" in calls
    latest_path = data_dir / "latest_candidates.csv"
    assert latest_path.exists()
    persisted = pd.read_csv(latest_path)
    assert persisted.iloc[0]["source"].startswith("fallback")
    assert frame.equals(persisted.head(len(frame)))
