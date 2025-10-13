"""PythonAnywhere WSGI entry point for the JBRAVO dashboard."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

env_path = ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

os.environ.setdefault("JBRAVO_HOME", str(ROOT))

from dashboards.dashboard_app import app as dash_app  # noqa: E402

application = dash_app.server

LOGGER = logging.getLogger("jbravo.wsgi")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

if not getattr(LOGGER, "_jbravo_logged", False):
    LOGGER.info("[WSGI] JBRAVO_HOME=%s venv ok", os.environ.get("JBRAVO_HOME", ""))
    LOGGER._jbravo_logged = True  # type: ignore[attr-defined]
