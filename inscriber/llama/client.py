"""OpenAI-compatible chat client for llama-server (DESIGN §2.1, §8.2).

Images are passed as base64 data URLs in the standard OpenAI ``image_url``
content-part shape, to ``POST {base_url}/v1/chat/completions``.
"""

from __future__ import annotations

import base64

import httpx

from inscriber.errors import InscriberError
from inscriber.logging import get_logger

logger = get_logger()


class ChatError(InscriberError):
    """Raised on a non-2xx chat response or malformed payload."""


def image_data_url(png_bytes: bytes, mime: str = "image/png") -> str:
    """Encode raw image bytes as a base64 ``data:`` URL."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


class ChatClient:
    """Thin wrapper over ``/v1/chat/completions`` (single-client, no streaming)."""

    def __init__(self, base_url: str, *, default_timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_timeout = default_timeout

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        sampling: dict | None = None,
        timeout_s: float | None = None,
    ) -> str:
        """Send a chat request, return ``choices[0].message.content`` (a string)."""
        body: dict = {
            "model": "local",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # sampling carries temperature / seed / etc. (DESIGN §8.2). Defaults to
        # deterministic temperature 0 when unspecified.
        body["temperature"] = 0
        if sampling:
            for key, value in sampling.items():
                if value is not None:
                    body[key] = value

        url = f"{self.base_url}/v1/chat/completions"
        try:
            resp = httpx.post(url, json=body, timeout=timeout_s or self.default_timeout)
        except httpx.HTTPError as e:
            raise ChatError(f"chat request to {url} failed: {e}") from e

        if resp.status_code != 200:
            raise ChatError(
                f"chat request to {url} returned {resp.status_code}: {resp.text[:500]}"
            )
        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise ChatError(f"malformed chat response from {url}: {e}") from e

    def chat_image(
        self,
        *,
        image_png: bytes,
        prompt: str,
        max_tokens: int,
        sampling: dict | None = None,
        timeout_s: float | None = None,
        image_first: bool = True,
    ) -> str:
        """One image + text-prompt turn → assistant text.

        Uses the OpenAI content-part shape: a ``text`` part plus an ``image_url``
        part whose ``url`` is a base64 ``data:`` URL.

        ``image_first`` (default True) puts the image content-part BEFORE the text.
        This is **required** for DeepSeek-OCR grounding to activate on llama-server
        (M1a finding: text-first yields plain markdown with no ``<|ref|>``/layout
        boxes; image-first yields the grounded ``label[[bbox]]`` output).
        """
        text_part = {"type": "text", "text": prompt}
        image_part = {
            "type": "image_url",
            "image_url": {"url": image_data_url(image_png)},
        }
        content = [image_part, text_part] if image_first else [text_part, image_part]
        messages = [{"role": "user", "content": content}]
        return self.chat(
            messages, max_tokens=max_tokens, sampling=sampling, timeout_s=timeout_s
        )
