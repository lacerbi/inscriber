"""``VlmBackend`` abstraction for figure description (DESIGN §9.2) and table
restructuring (dev/docs/table-reconstruction-findings.md)."""

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

    def build_table_prompt(
        self, table_blob: str, page_text: str, *, table_index: int, table_count: int
    ) -> str:
        """The fully-assembled table-restructuring prompt — also the table cache
        key material."""
        raise NotImplementedError

    def restructure_table(
        self,
        page_png: bytes,
        table_blob: str,
        page_text: str,
        *,
        table_index: int,
        table_count: int,
    ) -> str | None:
        """One table restructure (whole-page image + OCR blob → Markdown table).

        Returns the raw model response, or ``None`` when the response was
        truncated — a truncated table silently loses rows, while the original
        OCR blob still has every value, so the caller keeps the blob.
        """
        raise NotImplementedError

    def server_flags(self) -> list[str]:
        return []

    def sampling(self) -> dict:
        return {"temperature": 0, "seed": 0}

    def chat_template_kwargs(self) -> dict | None:
        """Extra jinja chat-template kwargs sent with every request (or None)."""
        return None
