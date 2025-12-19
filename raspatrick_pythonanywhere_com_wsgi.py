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

# ROOT_REDIRECT_TO_V2
_real_application = application


def application(environ, start_response):
    path = environ.get("PATH_INFO", "")
    if path in ("", "/"):
        qs = environ.get("QUERY_STRING", "")
        loc = "/v2/" + (f"?{qs}" if qs else "")
        start_response(
            "302 Found",
            [
                ("Location", loc),
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Cache-Control", "no-store"),
            ],
        )
        return [b"Redirecting to /v2/"]
    return _real_application(environ, start_response)

LOGGER = logging.getLogger("jbravo.wsgi")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

if not getattr(LOGGER, "_jbravo_logged", False):
    LOGGER.info("[WSGI] JBRAVO_HOME=%s venv ok", os.environ.get("JBRAVO_HOME", ""))
    LOGGER._jbravo_logged = True  # type: ignore[attr-defined]
