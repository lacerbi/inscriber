"""GLM-OCR backend — **experimental, TEXT-ONLY** (DESIGN §22.1, deferred set).

GLM-OCR is SOTA at text / table / equation OCR, but in llama.cpp it emits **no
figure bounding boxes** — layout/figure detection lives in a separate
PP-DocLayoutV3 model that is not part of llama.cpp. So this backend is text-only:
``supports_grounding = False`` and ``ocr_page`` returns ``regions = []`` (DESIGN
§8.2). Run it with ``--figure-detect none`` for a pure text/table comparison, or
pair it with a separate figure pass (the DeepSeek-for-boxes hybrid) later.

Pinned to the llama.cpp GLM-OCR usage notes (ggml-org/llama.cpp discussion #19721):

* prompt   = ``"Text Recognition:"``
* server   = **flash-attention must be off** (``--flash-attn off``) on current builds
* sampling = ``temperature 0.02`` (near-deterministic, per the upstream example)
* image content-part goes **before** the text prompt — already handled by
  ``ChatClient.chat_image(image_first=True)`` (the same M1a finding as DeepSeek).
"""

from __future__ import annotations

from typing import Literal

from inscriber.models import OcrPageResult, PageImage, ResolutionMode
from inscriber.ocr.base import Inferencer, OcrBackend

GLM_PROMPT = "Text Recognition:"


class GlmOcrBackend(OcrBackend):
    """Experimental text-only GLM-OCR backend (no figure grounding; DESIGN §22.1)."""

    name = "glm-ocr"
    supports_grounding = False

    def __init__(
        self,
        *,
        figures_enabled: bool = False,  # accepted for registry parity; GLM can't ground
        seed: int = 0,
        max_tokens: int = 8192,
        request_timeout: float = 900.0,
    ) -> None:
        # figures_enabled is intentionally ignored: GLM-OCR has no grounding, so
        # figure detection must come from elsewhere (--figure-detect none / a
        # separate detector). Kept in the signature so get_ocr_backend(...,
        # figures_enabled=...) constructs it the same way as DeepSeek.
        self.seed = seed
        self._max_tokens = max_tokens
        self.request_timeout = request_timeout

    def sampling(self) -> dict:
        # Upstream GLM-OCR example uses temperature 0.02 (near-deterministic). Seed
        # is part of the OCR cache key, so keep it fixed for reproducibility.
        return {"temperature": 0.02, "seed": self.seed}

    def max_tokens(self) -> int:
        return self._max_tokens

    def chat_template(self, path: Literal["server", "mtmd-cli"]) -> str | None:
        # Server applies the model's built-in template; don't override it.
        return None

    def server_flags(self) -> list[str]:
        # GLM-OCR currently requires Flash Attention OFF in llama.cpp (#19721);
        # the default is 'auto', which can enable it and break the image path.
        return ["--flash-attn", "off"]

    def prompt(self, *, figures_enabled: bool | None = None) -> str:
        return GLM_PROMPT

    def ocr_page(
        self, inf: Inferencer, image: PageImage, mode: ResolutionMode
    ) -> OcrPageResult:
        raw = inf.infer(
            image,
            self.prompt(),
            sampling=self.sampling(),
            chat_template=self.chat_template("server"),
            max_tokens=self.max_tokens(),
            timeout_s=self.request_timeout,
        )
        # Text-only: GLM emits clean markdown (tables/equations) with no grounding
        # markers — pass it straight through with no regions (DESIGN §8.2).
        return OcrPageResult(
            page_number=image.page_number, markdown=raw.strip(), regions=[]
        )
