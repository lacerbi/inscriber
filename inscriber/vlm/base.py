"""``VlmBackend`` abstraction for figure description (DESIGN §9.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VlmBackend(ABC):
    """A vision→text figure describer (image in, prose out)."""

    name: str = "base"

    @abstractmethod
    def describe(self, image_png: bytes, context_text: str | None) -> str:
        """Return the cleaned description text (already extracted from tags)."""

    def build_prompt(self, context_text: str | None) -> str:
        """The fully-assembled prompt (context included) — also the VLM cache key
        material (DESIGN §9.6)."""
        raise NotImplementedError

    def server_flags(self) -> list[str]:
        return []

    def sampling(self) -> dict:
        return {"temperature": 0, "seed": 0}

    def max_tokens(self) -> int:
        return 1536
