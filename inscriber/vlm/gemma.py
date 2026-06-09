"""Gemma 4 VLM backend (DESIGN §2.3, §9).

Used purely as a vision→text describer: build the figure-description prompt
(context included), send the figure crop as a base64 data URL, and extract the
text from the ``<img_desc>`` tags.
"""

from __future__ import annotations

from inscriber.errors import InferenceError
from inscriber.llama.client import ChatClient
from inscriber.postprocess.prompt import extract_description_from_tags, format_image_prompt
from inscriber.vlm.base import VlmBackend


class GemmaVlmBackend(VlmBackend):
    name = "gemma"

    def __init__(
        self,
        *,
        client: ChatClient | None = None,
        seed: int = 0,
        max_tokens: int = 1536,
        request_timeout: float = 600.0,
        image_first: bool = True,
    ) -> None:
        self.client = client
        self.seed = seed
        self._max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.image_first = image_first

    def sampling(self) -> dict:
        return {"temperature": 0, "seed": self.seed}

    def max_tokens(self) -> int:
        return self._max_tokens

    def build_prompt(self, context_text: str | None) -> str:
        return format_image_prompt(context_text)

    def describe(self, image_png: bytes, context_text: str | None) -> str:
        if self.client is None:
            raise InferenceError("GemmaVlmBackend has no chat client (no VLM endpoint)")
        prompt = self.build_prompt(context_text)
        raw = self.client.chat_image(
            image_png=image_png,
            prompt=prompt,
            max_tokens=self.max_tokens(),
            sampling=self.sampling(),
            timeout_s=self.request_timeout,
            image_first=self.image_first,
        )
        return extract_description_from_tags(raw)
