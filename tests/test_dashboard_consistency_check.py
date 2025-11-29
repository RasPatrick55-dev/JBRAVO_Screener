import json
import shutil
from pathlib import Path

import pytest

from scripts import dashboard_consistency_check as checker


pytestmark = pytest.mark.alpaca_optional


FIXTURE_DIR = Path(__file__).parent / "data"


def _copy_fixture(name: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE_DIR / name, target)


def _build_test_layout(base: Path) -> None:
    data_dir = base / "data"
    logs_dir = base / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    _copy_fixture("screener_metrics_ok.json", data_dir / "screener_metrics.json")
    _copy_fixture("execute_metrics.json", data_dir / "execute_metrics.json")
    _copy_fixture("metrics_summary.csv", data_dir / "metrics_summary.csv")
    _copy_fixture("latest_candidates.csv", data_dir / "latest_candidates.csv")
    _copy_fixture("top_candidates.csv", data_dir / "top_candidates.csv")
    _copy_fixture("scored_candidates.csv", data_dir / "scored_candidates.csv")
    _copy_fixture("connection_health.json", data_dir / "connection_health.json")

    predictions_dir = data_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    _copy_fixture("predictions_latest.csv", predictions_dir / "latest.csv")

    ranker_dir = data_dir / "ranker_eval"
    ranker_dir.mkdir(parents=True, exist_ok=True)
    _copy_fixture("ranker_eval_latest.json", ranker_dir / "latest.json")

    _copy_fixture("pipeline.log", logs_dir / "pipeline.log")
    _copy_fixture("execute_trades.log", logs_dir / "execute_trades.log")


def _stub_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checker, "send_alert", lambda *args, **kwargs: None)


def test_dashboard_consistency_check_runs_with_ok_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_alert(monkeypatch)
    _build_test_layout(tmp_path)
    reports_dir = tmp_path / "reports"

    rc = checker.main(["--base", str(tmp_path), "--reports-dir", str(reports_dir)])

    assert rc == 0
    findings_path = reports_dir / "dashboard_findings.txt"
    assert findings_path.exists()
    report_json = json.loads((reports_dir / "dashboard_consistency.json").read_text())
    assert report_json["inputs"]["screener_metrics"]["present"] is True


def test_dashboard_consistency_detects_missing_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_alert(monkeypatch)
    _build_test_layout(tmp_path)
    (tmp_path / "data" / "screener_metrics.json").unlink()
    reports_dir = tmp_path / "reports"

    rc = checker.main(["--base", str(tmp_path), "--reports-dir", str(reports_dir)])

    assert rc != 0
    findings_path = reports_dir / "dashboard_findings.txt"
    assert findings_path.exists()
    report_json = json.loads((reports_dir / "dashboard_consistency.json").read_text())
    assert report_json["inputs"]["screener_metrics"]["present"] is False
