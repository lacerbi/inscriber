"""Shared base exception so the CLI can present a clean message for any expected
pipeline failure (DESIGN §16) instead of an uncaught traceback.

Config errors map to exit 2 (usage); every other :class:`InscriberError` maps to
exit 1.
"""

from __future__ import annotations


class InscriberError(Exception):
    """Base class for all expected inscriber failures."""


class InferenceError(InscriberError):
    """Raised on a failed model inference call (mtmd-cli / VLM)."""
