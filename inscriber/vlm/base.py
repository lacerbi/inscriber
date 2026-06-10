"""``VlmBackend`` abstraction for figure description (DESIGN §9.2) and table
restructuring (dev/notes/2026-06-10-table-reconstruction-findings.md)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inscriber.llama.client import ChatClient


class VlmBackend(ABC):
    """A vision→text figure describer (image in, prose out).

    One instance serves both roles (DESIGN §9.2): the ``build_*`` prompt
    assemblers double as the cache-key material (§9.6), and the same instance
    performs the inference once the pipeline's VLM session attaches a chat
    ``client``. The caller assembles each prompt exactly once and passes the
    same string into ``describe``/``restructure_table``, so a cached key can
    never drift from the request actually sent.
    """

    name: str = "base"
    client: ChatClient | None = None  # attached by the VLM session at server launch

    def build_prompt(self, context_text: str | None) -> str:
        """The fully-assembled figure prompt (context included) — also the VLM
        cache key material (DESIGN §9.6)."""
        raise NotImplementedError

    def build_table_prompt(
        self,
        table_blob: str,
        page_text: str,
        *,
        table_index: int,
        table_count: int,
        cropped: bool = False,
    ) -> str:
        """The fully-assembled table-restructuring prompt — also the table cache
        key material. ``cropped`` selects the cropped-table-image variant
        (DESIGN §9.7); ``table_index``/``table_count`` are unused there."""
        raise NotImplementedError

    def build_bibtex_probe_prompt(self, page_text: str) -> str:
        """The fully-assembled BibTeX citability/metadata probe prompt — also
        the probe cache key material (DESIGN §12, auto mode)."""
        raise NotImplementedError

    @abstractmethod
    def describe(self, image_png: bytes, prompt: str) -> str:
        """One figure description (crop + a :meth:`build_prompt` prompt).
        Returns the cleaned description text (already extracted from tags)."""

    def restructure_table(self, image_png: bytes, prompt: str) -> str | None:
        """One table restructure (a :meth:`build_table_prompt` prompt plus its
        matching image — the table crop, or the whole page on the fallback
        path; DESIGN §9.7).

        Returns the raw model response, or ``None`` when the response was
        truncated — a truncated table silently loses rows, while the original
        OCR blob still has every value, so the caller keeps the blob.
        """
        raise NotImplementedError

    def probe_metadata(self, prompt: str) -> str | None:
        """One BibTeX metadata probe (a :meth:`build_bibtex_probe_prompt`
        prompt). **Text-only** — the project's only image-less inference call.

        Returns the raw model response, or ``None`` when truncated (the caller
        treats it as "citability unknown" and must not cache it).
        """
        raise NotImplementedError

    def server_flags(self) -> list[str]:
        return []

    def sampling(self) -> dict:
        return {"temperature": 0, "seed": 0}

    def chat_template_kwargs(self) -> dict | None:
        """Extra jinja chat-template kwargs sent with every request (or None)."""
        return None
