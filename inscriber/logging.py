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
    stream = sys.stderr
    # A cp1252 stderr (redirected/piped Windows console) raises UnicodeEncodeError
    # inside the logging handler when a message interpolates a non-ASCII model
    # path / URL / paper title — escape unencodable characters instead of losing
    # the whole record. (argparse help text got the same treatment in cli.py.)
    try:
        stream.reconfigure(errors="backslashreplace")
    except (AttributeError, OSError, ValueError):
        pass  # not a reconfigurable text stream (test doubles, exotic redirects)
    handler = logging.StreamHandler(stream=stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)
