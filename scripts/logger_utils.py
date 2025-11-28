import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def make_screener_file_handler() -> logging.Handler:
    """Create a file handler for ``logs/screener.log``.

    The handler uses a simple INFO-level formatter and ensures the logs
    directory exists before opening the file.
    """

    log_path = LOG_DIR / "screener.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    fmt = logging.Formatter("%(asctime)s - screener - %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    handler.setLevel(logging.INFO)
    return handler
