import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if "PYTHONANYWHERE_USAGE_URL" not in os.environ:
    os.environ["PYTHONANYWHERE_USAGE_URL"] = "http://localhost:8001/pythonanywhere_usage.json"

from dashboards import dashboard_app

if __name__ == "__main__":
    dashboard_app.app.run(debug=False)
