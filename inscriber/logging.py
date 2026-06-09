"""Logging & progress setup (DESIGN §16).

Progress / logs go to **stderr**; the final list of written file paths goes to
**stdout** (one per line, see ``output.py``) so a run is machine-parseable even
under ``-q``.

``-v`` -> DEBUG, default -> INFO, ``-q`` -> WARNING.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "inscriber"


def setup_logging(verbose: int = 0, quiet: bool = False) -> logging.Logger:
    """Configure the ``inscriber`` logger to emit to stderr at the chosen level."""
    if quiet:
        level = logging.WARNING
    elif verbose >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    # Replace handlers so repeated setup (e.g. in tests) doesn't duplicate output.
    logger.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
