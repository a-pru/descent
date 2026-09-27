"""Rank-zero-aware python logger."""

import logging

from lightning.pytorch.utilities import rank_zero_only


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Returns a logger that only logs on rank zero in multi-GPU runs."""
    logger = logging.getLogger(name)
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger
