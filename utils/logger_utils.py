import logging
import os
from logging.handlers import RotatingFileHandler


def init_logging(module_name: str, log_filename: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    )

    file_handler = RotatingFileHandler(
        f"logs/{log_filename}", maxBytes=2 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(log_formatter)

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    error_handler = RotatingFileHandler(
        "logs/error.log", maxBytes=2 * 1024 * 1024, backupCount=5
    )
    error_handler.setFormatter(log_formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

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
