import importlib
from pathlib import Path

import pytest


def reload_data_io(monkeypatch: pytest.MonkeyPatch, base_dir: Path):
    monkeypatch.setenv("JBRAVO_HOME", str(base_dir))
    import dashboards.data_io as data_io  # local import for reload

    return importlib.reload(data_io)
