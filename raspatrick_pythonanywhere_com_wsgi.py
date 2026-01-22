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

from flask import Flask, send_from_directory  # noqa: E402
from werkzeug.middleware.dispatcher import DispatcherMiddleware  # noqa: E402

from dashboards.dashboard_app import app as dash_app  # noqa: E402

FRONTEND_BUILD = ROOT / "frontend" / "dist"
FRONTEND_ASSETS = FRONTEND_BUILD / "assets"


class PrefixMiddleware:
    """Re-apply a URL prefix stripped by DispatcherMiddleware."""

    def __init__(self, app, prefix: str) -> None:
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        environ = environ.copy()
        environ["PATH_INFO"] = f"{self.prefix}{environ.get('PATH_INFO', '')}"
        return self.app(environ, start_response)


frontend_app = Flask(
    __name__,
    static_folder=str(FRONTEND_ASSETS),
    static_url_path="/assets",
)


@frontend_app.route("/", defaults={"path": ""})
@frontend_app.route("/<path:path>")
def react_catch_all(path: str):
    index_path = FRONTEND_BUILD / "index.html"
    if not index_path.exists():
        message = "React build not found. Run the frontend build and place output in frontend/dist."
        return message, 503
    return send_from_directory(FRONTEND_BUILD, "index.html")


dash_wsgi = PrefixMiddleware(dash_app.server, "/v2")
api_wsgi = PrefixMiddleware(dash_app.server, "/api")

application = DispatcherMiddleware(
    frontend_app,
    {
        "/v2": dash_wsgi,
        "/api": api_wsgi,
    },
)

LOGGER = logging.getLogger("jbravo.wsgi")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

if not getattr(LOGGER, "_jbravo_logged", False):
    LOGGER.info("[WSGI] JBRAVO_HOME=%s venv ok", os.environ.get("JBRAVO_HOME", ""))
    LOGGER._jbravo_logged = True  # type: ignore[attr-defined]
