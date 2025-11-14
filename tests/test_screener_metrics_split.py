import json
from pathlib import Path

import pandas as pd

from scripts.run_pipeline import compose_metrics_from_artifacts


def test_compose_metrics_from_artifacts_uses_stage_and_csv(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fetch_payload = {
        "symbols_with_bars_fetch": 9,
        "bars_rows_total_fetch": 180,
    }
    post_payload = {
        "symbols_with_bars_post": 7,
        "candidates_final": 3,
    }
    (data_dir / "screener_stage_fetch.json").write_text(json.dumps(fetch_payload))
    (data_dir / "screener_stage_post.json").write_text(json.dumps(post_payload))

    top_df = pd.DataFrame(
        [
            {"symbol": "AAA", "score": 1},
            {"symbol": "BBB", "score": 2},
            {"symbol": "CCC", "score": 3},
        ]
    )
    top_df.to_csv(data_dir / "top_candidates.csv", index=False)

    payload = compose_metrics_from_artifacts(
        tmp_path,
        symbols_in=15,
        fallback_symbols_with_bars=11,
        fallback_bars_rows_total=200,
        latest_source="screener",
    )

    assert payload["symbols_in"] == 15
    assert payload["symbols_with_bars_fetch"] == 9
    assert payload["symbols_with_bars"] == 9
    assert payload["bars_rows_total_fetch"] == 180
    assert payload["bars_rows_total"] == 180
    assert payload["symbols_with_bars_post"] == 7
    assert payload["rows"] == 3
    assert payload["rows_premetrics"] == 3
    assert payload["candidates_final"] == 3
    assert payload["latest_source"] == "screener"
