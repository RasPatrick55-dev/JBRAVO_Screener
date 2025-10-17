from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import dashboard_consistency_check as checker


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.alpaca_optional
def test_dashboard_consistency_report_generation(tmp_path: Path) -> None:
    base = tmp_path
    data_dir = base / "data"
    logs_dir = base / "logs"
    reports_dir = base / "reports"
    data_dir.mkdir()
    logs_dir.mkdir()

    (data_dir / "screener_metrics.json").write_text(
        json.dumps(
            {
                "last_run_utc": "2024-01-01T09:05:00Z",
                "symbols_in": 12,
                "symbols_with_bars": 10,
                "rows": 2,
                "source": "screener",
                "gate_fail_total": 1,
                "gate_breakdown": {"TIME_WINDOW": 1},
                "fetch_secs": 12.3,
                "feature_secs": 4.5,
                "rank_secs": 1.2,
                "gate_secs": 0.8,
            }
        ),
        encoding="utf-8",
    )

    _write_csv(
        data_dir / "metrics_summary.csv",
        [
            {
                "total_trades": 1,
                "net_pnl": 25.5,
                "win_rate": 100.0,
                "expectancy": 25.5,
                "profit_factor": 2.0,
                "max_drawdown": 0.0,
            }
        ],
    )

    _write_csv(
        data_dir / "latest_candidates.csv",
        [
            {
                "timestamp": "2024-01-01T09:00:00Z",
                "symbol": "AAPL",
                "score": 0.91,
                "exchange": "NASDAQ",
                "close": 195.1,
                "volume": 1_000_000,
                "universe_count": 12,
                "score_breakdown": "{}",
                "entry_price": 195.1,
                "adv20": 5_000_000,
                "atrp": 0.75,
                "source": "pipeline",
            },
            {
                "timestamp": "2024-01-01T09:00:00Z",
                "symbol": "BETA",
                "score": 0.85,
                "exchange": "NYSE",
                "close": 22.3,
                "volume": 250_000,
                "universe_count": 12,
                "score_breakdown": "{}",
                "entry_price": 22.3,
                "adv20": 1_500_000,
                "atrp": 1.25,
                "source": "pipeline",
            },
        ],
    )

    _write_csv(
        data_dir / "top_candidates.csv",
        [
            {"symbol": "AAPL", "score": 0.91},
            {"symbol": "BETA", "score": 0.85},
        ],
    )

    _write_csv(
        data_dir / "scored_candidates.csv",
        [
            {"symbol": "AAPL", "score": 0.91, "feature": 1.0},
            {"symbol": "BETA", "score": 0.85, "feature": 0.9},
        ],
    )

    (data_dir / "execute_metrics.json").write_text(
        json.dumps(
            {
                "orders_submitted": 2,
                "orders_filled": 1,
                "orders_canceled": 1,
                "trailing_attached": 1,
                "skip_reasons": {"CASH": 1},
            }
        ),
        encoding="utf-8",
    )

    _write_csv(
        data_dir / "predictions" / "latest.csv",
        [
            {"symbol": "AAPL", "score": 0.91},
            {"symbol": "BETA", "score": 0.85},
        ],
    )

    ranker_path = data_dir / "ranker_eval" / "latest.json"
    ranker_path.parent.mkdir(parents=True, exist_ok=True)
    ranker_path.write_text(
        json.dumps({"deciles": [{"name": "D1", "lift": 1.5}]}),
        encoding="utf-8",
    )

    (logs_dir / "pipeline.log").write_text(
        "\n".join(
            [
                "2024-01-01 09:00:00 [INFO] PIPELINE_START steps=screener,metrics",
                "2024-01-01 09:05:00 [INFO] PIPELINE_SUMMARY symbols_in=12 with_bars=10 rows=2 fetch_secs=12.3 feature_secs=4.5 rank_secs=1.2 gate_secs=0.8",
                "2024-01-01 09:05:01 [INFO] FALLBACK_CHECK rows_out=2 source=screener",
                "2024-01-01 09:05:02 [INFO] PIPELINE_END rc=0 duration=62.5s",
                "2024-01-01 09:05:03 [INFO] DASH RELOAD method=touch rc=0 path=/tmp/app.wsgi",
            ]
        ),
        encoding="utf-8",
    )

    (logs_dir / "execute_trades.log").write_text(
        "\n".join(
            [
                "2024-01-01 09:05:10 [INFO] Submitting limit buy order for AAPL",
                "2024-01-01 09:05:11 [INFO] Order filled for AAPL",
                "2024-01-01 09:05:12 [INFO] Order canceled for BETA",
                "2024-01-01 09:05:13 [INFO] Creating trailing stop for AAPL",
                "2024-01-01 09:05:14 [INFO] SKIP due to CASH",
            ]
        ),
        encoding="utf-8",
    )

    report = checker.generate_report(base_dir=base)

    assert report["checks"]["candidates_ok"] is True
    assert report["checks"]["candidates"]["latest"]["canonical"] is True
    assert report["checks"]["candidates"]["latest"]["missing_score_breakdown"] is False
    pipeline_tokens = report["checks"]["pipeline_tokens"]
    assert pipeline_tokens["PIPELINE_START"]["data"]["steps"] == ["screener", "metrics"]
    assert pipeline_tokens["PIPELINE_SUMMARY"]["data"]["rows"] == 2
    assert report["checks"]["trades_log"]["present"] is False
    assert report["checks"]["executor"]["orders_submitted"] == 2
    assert report["checks"]["predictions"]["prediction_rows"] == 2
    assert report["checks"]["prefix_sanity"]["ok"] is True

    assert (reports_dir / "dashboard_consistency.json").exists()
    assert (reports_dir / "dashboard_kpis.csv").exists()
    assert (reports_dir / "dashboard_findings.txt").exists()
    findings_text = (reports_dir / "dashboard_findings.txt").read_text(encoding="utf-8")
    assert "candidates_header=canonical:true" in findings_text
