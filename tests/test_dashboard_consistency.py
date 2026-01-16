from __future__ import annotations

import importlib
import json
import sys
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
                "symbols_with_any_bars": 10,
                "symbols_with_required_bars": 10,
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
                "sma9": 194.9,
                "ema20": 193.8,
                "sma180": 150.2,
                "rsi14": 58.0,
                "passed_gates": True,
                "gate_fail_reason": "[]",
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
                "sma9": 22.1,
                "ema20": 21.9,
                "sma180": 18.4,
                "rsi14": 52.0,
                "passed_gates": True,
                "gate_fail_reason": "[]",
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
                "2024-01-01 09:05:00 [INFO] PIPELINE_SUMMARY symbols_in=12 with_bars=10 rows=2 fetch_secs=12.3 feature_secs=4.5 rank_secs=1.2 gate_secs=0.8 bars_rows_total=240 source=screener",
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
                "2024-01-01 09:05:10 [INFO] BUY_SUBMIT symbol=AAPL qty=10 limit=125.50",
                "2024-01-01 09:05:11 [INFO] BUY_FILL symbol=AAPL filled_qty=10 avg_price=125.55",
                "2024-01-01 09:05:12 [INFO] BUY_CANCELLED symbol=BETA remaining_qty=10 status=cancelled",
                "2024-01-01 09:05:13 [INFO] TRAIL_SUBMIT symbol=AAPL trail_pct=3 route=trailing_stop",
                "2024-01-01 09:05:14 [INFO] TRAIL_CONFIRMED symbol=AAPL qty=10 order_id=abc123",
                "2024-01-01 09:05:15 [INFO] TIME_WINDOW reason=outside_window",
                "2024-01-01 09:05:16 [INFO] CASH symbol=BETA detail=insufficient",
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
    assert report["checks"]["executor"]["orders_canceled"] == 1
    assert report["checks"]["executor"]["tokens"]["BUY_SUBMIT"]["count"] == 1
    assert report["checks"]["executor"]["tokens"]["TIME_WINDOW"]["count"] == 1
    assert report["checks"]["executor"]["skip_reasons"]["CASH"] == 1
    assert report["checks"]["executor"]["skip_reasons"]["TIME_WINDOW"] == 1
    assert report["checks"]["predictions"]["prediction_rows"] == 2
    assert report["checks"]["prefix_sanity"]["ok"] is True

    assert (reports_dir / "dashboard_consistency.json").exists()
    assert (reports_dir / "dashboard_kpis.csv").exists()
    assert (reports_dir / "dashboard_findings.txt").exists()
    findings_text = (reports_dir / "dashboard_findings.txt").read_text(encoding="utf-8")
    assert "candidates_header=canonical:true" in findings_text

    evidence_dir = base / "evidence"
    bundle = checker.collect_evidence(report, base_dir=base, evidence_dir=evidence_dir)
    assert bundle.exists()
    assert (bundle / "pipeline_tokens.json").exists()
    assert (bundle / "executor_tokens.json").exists()
    assert (bundle / "candidate_headers.json").exists()
    assert (bundle / "metrics_snapshot.json").exists()


@pytest.mark.alpaca_optional
def test_dashboard_kpis_recover_from_pipeline_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path
    data_dir = base_dir / "data"
    logs_dir = base_dir / "logs"
    data_dir.mkdir()
    logs_dir.mkdir()

    (data_dir / "screener_metrics.json").write_text(
        json.dumps(
            {
                "last_run_utc": "2024-06-01T12:00:00Z",
                "symbols_in": None,
                "symbols_with_bars": None,
                "rows": None,
                "bars_rows_total": None,
            }
        ),
        encoding="utf-8",
    )

    (logs_dir / "pipeline.log").write_text(
        "2024-06-01T12:00:05Z [INFO] PIPELINE_SUMMARY symbols_in=42 with_bars=36 rows=9 bar_rows=540\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("JBRAVO_HOME", str(base_dir))
    sys.modules.pop("dashboards.screener_health", None)
    screener_health = importlib.import_module("dashboards.screener_health")
    try:
        kpis = screener_health.load_kpis()

        assert kpis["symbols_in"] == 42
        assert kpis["symbols_with_bars"] == 36
        assert kpis["rows"] == 9
        assert kpis["bars_rows_total"] == 540
        assert kpis["source"] == "pipeline summary (recovered)"
        assert kpis["_kpi_inferred_from_log"] is True
    finally:
        sys.modules.pop("dashboards.screener_health", None)
