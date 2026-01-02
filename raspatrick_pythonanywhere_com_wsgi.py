"""PythonAnywhere WSGI entry point for the JBRAVO dashboard."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROJECT_HOME = Path("/home/RasPatrick/jbravo_screener")
os.environ.setdefault("PROJECT_HOME", str(PROJECT_HOME))
if str(PROJECT_HOME) not in sys.path:
    sys.path.insert(0, str(PROJECT_HOME))

env_path = Path.home() / ".config" / "jbravo" / ".env"
if env_path.exists():
    try:
        with env_path.open("r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key and key not in os.environ:
                    os.environ.setdefault(key, value)
    except OSError:
        pass

print(
    "WSGI_ENV_LOADED",
    str(env_path),
    "DATABASE_URL_SET",
    bool(os.getenv("DATABASE_URL")),
)

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
