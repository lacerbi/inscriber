"""Logging setup (DESIGN §16) — cp1252-stderr survival (review A3).

On a redirected/piped Windows console, stderr can be cp1252; a non-ASCII model
path / URL / paper title inside a log message must be escaped, not raise
``UnicodeEncodeError`` inside the logging handler (which loses the record).
"""

from __future__ import annotations

import io
import logging

from inscriber.logging import LOGGER_NAME, setup_logging


def test_nonascii_log_survives_cp1252_stderr(monkeypatch):
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    monkeypatch.setattr("inscriber.logging.sys.stderr", stream)
    logger = setup_logging()
    try:
        # "ï" is cp1252-encodable, "→" (U+2192) is not.
        logger.info("reading %s", "naïve→paper.pdf")
        for h in logger.handlers:
            h.flush()
        stream.flush()
        text = buf.getvalue().decode("cp1252")
        assert "naïve" in text  # encodable characters pass through unchanged
        assert "\\u2192" in text  # the arrow is escaped instead of fatal
    finally:
        # Drop the handler bound to this test's buffer so later tests that log
        # without re-running setup_logging don't write into a dead stream.
        logging.getLogger(LOGGER_NAME).handlers.clear()
