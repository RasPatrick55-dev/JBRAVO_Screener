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


def _write_minimal_latest_candidates(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("timestamp,symbol,score\n", encoding="utf-8")


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


def test_dashboard_consistency_allows_fetch_superset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_alert(monkeypatch)
    _build_test_layout(tmp_path)
    data_dir = tmp_path / "data"
    metrics = {
        "symbols_in": 57,
        "symbols_with_required_bars": 57,
        "symbols_with_any_bars": 100,
        "symbols_with_bars_fetch": 419,
        "bars_rows_total_fetch": 1000,
        "rows": 2,
    }
    (data_dir / "screener_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    errors = checker.run_assertions(tmp_path)

    assert errors == []


def test_no_fallback_or_warning_when_bar_totals_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _stub_alert(monkeypatch)
    _build_test_layout(tmp_path)
    caplog.set_level("WARNING")

    errors = checker.run_assertions(tmp_path)

    assert errors == []
    assert "[WARN] bars_rows_total_fetch" not in caplog.text
    assert "[FALLBACK]" not in "".join(errors)


def test_bar_total_mismatch_warns_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _stub_alert(monkeypatch)
    _build_test_layout(tmp_path)
    metrics_path = tmp_path / "data" / "screener_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["bars_rows_total_fetch"] = 1000
    metrics["bars_rows_total"] = 200
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(checker.LOGGER, "propagate", True)
    caplog.set_level("WARNING", logger=checker.LOGGER.name)

    errors = checker.run_assertions(tmp_path)

    assert errors == []
    assert "[WARN] bars_rows_total_fetch" in caplog.text


def test_fallback_triggers_only_when_candidates_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_alert(monkeypatch)
    _build_test_layout(tmp_path)
    _write_minimal_latest_candidates(tmp_path / "data" / "latest_candidates.csv")

    errors = checker.run_assertions(tmp_path)

    assert any(error.startswith("[FALLBACK] latest_candidates.csv empty") for error in errors)
