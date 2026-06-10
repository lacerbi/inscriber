"""Gemma 4 VLM backend (DESIGN §2.3, §9).

Used as a vision→text describer (figure crops → prose, DESIGN §9) and as the
table restructurer (whole page image + DeepSeek ``<table>`` blob → Markdown pipe
table, dev/docs/table-reconstruction-findings.md). Prompts are assembled once by
the orchestrator via ``build_prompt``/``build_table_prompt`` — the same strings
are the cache-key material (DESIGN §9.6) — and passed into
``describe``/``restructure_table``.

Gemma 4 is a **thinking model**: hard tasks spend reasoning tokens before the
answer (llama-server strips the thought channel from ``content``). Thinking is
activated **explicitly** via the per-request chat-template kwarg
``enable_thinking`` rather than relying on the build's default.

No ``max_tokens`` is sent: generation runs until EOS or the context window fills
(``ctx_size`` is the single size knob). Hitting the window yields
``finish_reason: "length"``, which is how truncation is detected.
"""

from __future__ import annotations

from inscriber.bibtex.probe import format_probe_prompt
from inscriber.errors import InferenceError
from inscriber.llama.client import ChatClient
from inscriber.postprocess.prompt import extract_description_from_tags, format_image_prompt
from inscriber.postprocess.tables import format_table_prompt
from inscriber.vlm.base import VlmBackend

TRUNCATED_MARKER = "[...]"


class GemmaVlmBackend(VlmBackend):
    name = "gemma"

    def __init__(
        self,
        *,
        client: ChatClient | None = None,
        seed: int = 0,
        request_timeout: float = 600.0,
        image_first: bool = True,
    ) -> None:
        self.client = client
        self.seed = seed
        self.request_timeout = request_timeout
        self.image_first = image_first

    def sampling(self) -> dict:
        return {"temperature": 0, "seed": self.seed}

    def chat_template_kwargs(self) -> dict | None:
        # Explicit thinking activation (needs llama-server jinja templating; on
        # builds where the kwarg is a no-op the model's default applies).
        return {"enable_thinking": True}

    def build_prompt(self, context_text: str | None) -> str:
        return format_image_prompt(context_text)

    def build_table_prompt(
        self, table_blob: str, page_text: str, *, table_index: int, table_count: int
    ) -> str:
        return format_table_prompt(
            table_blob, page_text, table_index=table_index, table_count=table_count
        )

    def build_bibtex_probe_prompt(self, page_text: str) -> str:
        return format_probe_prompt(page_text)

    def _truncated(self) -> bool:
        """Whether the last response was cut off (``finish_reason != "stop"``,
        i.e. generation hit the context window rather than ending at EOS)."""
        finish_reason = getattr(self.client, "last_finish_reason", None)
        return isinstance(finish_reason, str) and finish_reason != "stop"

    def describe(self, image_png: bytes, prompt: str) -> str:
        if self.client is None:
            raise InferenceError("GemmaVlmBackend has no chat client (no VLM endpoint)")
        raw = self.client.chat_image(
            image_png=image_png,
            prompt=prompt,
            sampling=self.sampling(),
            timeout_s=self.request_timeout,
            image_first=self.image_first,
            chat_template_kwargs=self.chat_template_kwargs(),
        )
        desc = extract_description_from_tags(raw)
        if self._truncated() and not desc.rstrip().endswith(TRUNCATED_MARKER):
            return desc.rstrip() + f" {TRUNCATED_MARKER}"
        return desc

    def restructure_table(self, page_png: bytes, prompt: str) -> str | None:
        if self.client is None:
            raise InferenceError("GemmaVlmBackend has no chat client (no VLM endpoint)")
        raw = self.client.chat_image(
            image_png=page_png,
            prompt=prompt,
            sampling=self.sampling(),
            timeout_s=self.request_timeout,
            image_first=True,  # image before text, as in the validated experiment
            chat_template_kwargs=self.chat_template_kwargs(),
        )
        if self._truncated():
            return None  # a truncated table silently loses rows; keep the OCR blob
        return raw

    def probe_metadata(self, prompt: str) -> str | None:
        # Text-only: a hand-built single-user-message list straight to chat()
        # (which pins a temperature-0 baseline before applying sampling).
        if self.client is None:
            raise InferenceError("GemmaVlmBackend has no chat client (no VLM endpoint)")
        raw = self.client.chat(
            [{"role": "user", "content": prompt}],
            sampling=self.sampling(),
            timeout_s=self.request_timeout,
            chat_template_kwargs=self.chat_template_kwargs(),
        )
        if self._truncated():
            return None  # truncated JSON is unusable; treat citability as unknown
        return raw
