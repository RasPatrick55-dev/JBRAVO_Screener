import json
import shutil
from pathlib import Path

import plotly.graph_objects as go
import pytest

from dashboards import screener_health


pytestmark = pytest.mark.alpaca_optional


class DummyApp:
    def __init__(self) -> None:
        self.callbacks = []

    def callback(self, *args, **kwargs):  # noqa: ANN001
        def decorator(func):
            self.callbacks.append(func)
            return func

        return decorator


def _prepare_artifacts(tmp_path: Path) -> None:
    fixture_dir = Path(__file__).parent / "data"
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for name, target in [
        ("screener_metrics_ok.json", data_dir / "screener_metrics.json"),
        ("execute_metrics.json", data_dir / "execute_metrics.json"),
        ("last_premarket_run.json", data_dir / "last_premarket_run.json"),
    ]:
        shutil.copy(fixture_dir / name, target)

    ranker_dir = data_dir / "ranker_eval"
    ranker_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture_dir / "ranker_eval_latest.json", ranker_dir / "latest.json")

    for name in ("pipeline.log", "screener.log"):
        shutil.copy(fixture_dir / name, logs_dir / name)


def _patch_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(screener_health, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(screener_health, "DATA_DIR", data_dir)
    monkeypatch.setattr(screener_health, "LOG_DIR", logs_dir)
    monkeypatch.setattr(screener_health, "METRICS_JSON", data_dir / "screener_metrics.json")
    monkeypatch.setattr(screener_health, "EXECUTE_METRICS_JSON", data_dir / "execute_metrics.json")
    monkeypatch.setattr(screener_health, "PREMARKET_JSON", data_dir / "last_premarket_run.json")
    monkeypatch.setattr(
        screener_health, "RANKER_EVAL_LATEST", data_dir / "ranker_eval" / "latest.json"
    )


def _render_callback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    app = DummyApp()
    _patch_paths(tmp_path, monkeypatch)
    screener_health.register_callbacks(app)
    assert app.callbacks, "callback should be registered"
    return app.callbacks[-1]


def test_screener_health_layout_renders_with_full_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_artifacts(tmp_path)
    _patch_paths(tmp_path, monkeypatch)

    layout = screener_health.build_layout()
    assert layout is not None


def test_decile_panels_show_not_computed_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_artifacts(tmp_path)
    _patch_paths(tmp_path, monkeypatch)

    render = _render_callback(monkeypatch, tmp_path)
    outputs = render({}, [], [], None, {}, {}, {})
    hit_fig, ret_fig = outputs[14], outputs[15]
    for fig in (hit_fig, ret_fig):
        assert isinstance(fig, go.Figure)
        annotation_texts = [ann.text for ann in fig.layout.annotations]
        joined = " ".join(annotation_texts)
        assert "no ranker_eval file present" in joined


def test_decile_panels_use_ranker_eval_data_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_artifacts(tmp_path)
    _patch_paths(tmp_path, monkeypatch)
    ev = screener_health.load_ranker_eval(tmp_path)

    render = _render_callback(monkeypatch, tmp_path)
    outputs = render({}, [], [], ev, {}, {}, {})
    hit_fig, ret_fig = outputs[14], outputs[15]
    assert list(hit_fig.data[0].x) == [1, 2, 3]
    assert list(hit_fig.data[0].y) == [0.1, 0.15, 0.2]
    assert list(ret_fig.data[0].y) == [0.02, 0.03, 0.04]


def test_decile_panels_surface_reason_and_sample_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare_artifacts(tmp_path)
    _patch_paths(tmp_path, monkeypatch)

    render = _render_callback(monkeypatch, tmp_path)
    ev = {
        "deciles": [],
        "label_horizon_days": 5,
        "reason": "no_labeled_samples",
        "run_utc": "2024-08-01T00:00:00Z",
        "sample_size": 0,
    }
    outputs = render({}, [], [], ev, {}, {}, {})
    hit_fig, ret_fig = outputs[14], outputs[15]
    for fig in (hit_fig, ret_fig):
        annotation_texts = [ann.text for ann in fig.layout.annotations]
        joined = " ".join(annotation_texts)
        assert "no_labeled_samples" in joined
        assert "sample_size=0" in joined
