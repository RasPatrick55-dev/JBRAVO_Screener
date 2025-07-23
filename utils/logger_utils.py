import logging
import os
from logging.handlers import RotatingFileHandler


def init_logging(module_name: str, log_filename: str) -> logging.Logger:
    """Return a logger writing to the project's ``logs`` directory."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None, level=logging.INFO, filename: str | None = None) -> logging.Logger:
    """Return a configured :class:`logging.Logger` instance.

    Parameters
    ----------
    name:
        Name of the logger, typically ``__name__``.
    level:
        The logging level. Defaults to :data:`logging.INFO`.
    filename:
        Optional path to a log file. When provided, messages are written to this
        file in addition to ``stdout``.
    """

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if filename:
        handlers.append(logging.FileHandler(filename))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    return logging.getLogger(name)

