"""OCR backend registry: name → backend class (DESIGN §8.1).

Adding a future backend is purely additive: implement :class:`OcrBackend` and
register it here. v1 has exactly one (``deepseek-ocr``); the deferred text-OCR
backends (§22.1) plug in the same way.
"""

from __future__ import annotations

from inscriber.ocr.base import OcrBackend
from inscriber.ocr.deepseek import DeepSeekOcrBackend

_REGISTRY: dict[str, type[OcrBackend]] = {
    DeepSeekOcrBackend.name: DeepSeekOcrBackend,
}


def known_ocr_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_ocr_backend(name: str, **kwargs) -> OcrBackend:
    """Construct the named OCR backend, forwarding kwargs to its constructor."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"unknown OCR backend {name!r}; known: {known_ocr_backends()}"
        )
    return cls(**kwargs)
