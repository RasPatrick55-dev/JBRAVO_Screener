"""Label generation utilities for nightly ranker evaluation."""

from .make_nextday_labels import main as make_nextday_labels

__all__ = ["make_nextday_labels"]
