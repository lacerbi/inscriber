"""VLM backend registry: name → backend class (DESIGN §9)."""

from __future__ import annotations

from inscriber.vlm.base import VlmBackend
from inscriber.vlm.gemma import GemmaVlmBackend

_REGISTRY: dict[str, type[VlmBackend]] = {
    GemmaVlmBackend.name: GemmaVlmBackend,
}


def known_vlm_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_vlm_backend(name: str, **kwargs) -> VlmBackend:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"unknown VLM backend {name!r}; known: {known_vlm_backends()}")
    return cls(**kwargs)
