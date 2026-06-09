"""OCR inference layer + ``OcrBackend`` abstraction (DESIGN §8.2).

This module holds:

* :class:`Inferencer` — the one multimodal (image+prompt → text) call backends
  depend on, with **two** implementations so the ``llama-mtmd-cli`` fallback
  (DESIGN §2.1) is swappable without touching any backend:
    - :class:`HttpInferencer`    → ``llama-server`` ``/v1/chat/completions``
    - :class:`MtmdCliInferencer` → one-shot ``llama-mtmd-cli`` subprocess
* :class:`OcrBackend` — the ABC each OCR model implements (added M1b; the v1
  ``DeepSeekOcrBackend`` lives in ``ocr/deepseek.py``).

``Region`` / ``OcrPageResult`` are imported from :mod:`inscriber.models` (their
canonical home) and re-exported here for convenience.
"""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from inscriber.config import find_binary
from inscriber.errors import InferenceError
from inscriber.llama.client import ChatClient
from inscriber.logging import get_logger
from inscriber.models import OcrPageResult, PageImage, Region, ResolutionMode

__all__ = [
    "Inferencer",
    "HttpInferencer",
    "MtmdCliInferencer",
    "OcrBackend",
    "OcrPageResult",
    "Region",
]

logger = get_logger()


@runtime_checkable
class Inferencer(Protocol):
    """One multimodal (image + prompt → text) call (DESIGN §8.2).

    Backends depend on THIS, not on an HTTP client directly, so the mtmd-cli
    fallback is implementable without changing any signatures.
    """

    def infer(
        self,
        image: PageImage,
        prompt: str,
        *,
        sampling: dict,
        chat_template: str | None,
        max_tokens: int,
        timeout_s: float,
    ) -> str: ...


class HttpInferencer:
    """Inference over a managed ``llama-server`` (the primary path, DESIGN §2.1).

    ``chat_template`` is ignored here: on the server path the model's built-in
    template is applied and we must NOT pass ``--chat-template`` for DeepSeek-OCR
    (DESIGN §2.2). It is part of the signature only for the mtmd-cli path.
    """

    def __init__(self, base_url: str) -> None:
        self.client = ChatClient(base_url)
        self.last_raw: str = ""  # last response, for cache raw-output capture

    def infer(
        self,
        image: PageImage,
        prompt: str,
        *,
        sampling: dict,
        chat_template: str | None,
        max_tokens: int,
        timeout_s: float,
    ) -> str:
        out = self.client.chat_image(
            image_png=image.png_bytes,
            prompt=prompt,
            max_tokens=max_tokens,
            sampling=sampling,
            timeout_s=timeout_s,
        )
        self.last_raw = out
        return out


class MtmdCliInferencer:
    """One-shot ``llama-mtmd-cli`` fallback (DESIGN §2.1).

    Reloads the model on every call (slow), so it is NOT the primary path — it
    exists because the server image path has had model-specific bugs (issue
    #21022). Unlike the HTTP path, the chat template flag IS used here for
    DeepSeek-OCR (``--chat-template deepseek-ocr``, DESIGN §2.2).

    ⚠️ Output parsing is best-effort and must be confirmed empirically in M1a on
    real hardware (mtmd-cli prints generation to stdout, logs to stderr).
    """

    def __init__(
        self,
        bin_dir: str,
        model: str,
        mmproj: str,
        *,
        n_gpu_layers: int = 0,
        ctx_size: int = 8192,
    ) -> None:
        self.bin_dir = bin_dir
        self.model = model
        self.mmproj = mmproj
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.last_raw: str = ""

    def _resolve_exe(self) -> Path:
        exe = find_binary(self.bin_dir, "llama-mtmd-cli")
        if exe is None:
            raise InferenceError(
                "llama-mtmd-cli binary not found "
                f"(llama.bin_dir={self.bin_dir!r}; not on PATH either)"
            )
        return exe

    def build_args(self, exe: Path, image_path: Path, prompt: str, *,
                   chat_template: str | None, max_tokens: int) -> list[str]:
        args = [
            str(exe),
            "-m", self.model,
            "--mmproj", self.mmproj,
            "--image", str(image_path),
            "-p", prompt,
            "--temp", "0",
            "-n", str(max_tokens),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx_size),
            "--no-display-prompt",
        ]
        if chat_template:
            args += ["--chat-template", chat_template]
        return args

    def infer(
        self,
        image: PageImage,
        prompt: str,
        *,
        sampling: dict,
        chat_template: str | None,
        max_tokens: int,
        timeout_s: float,
    ) -> str:
        exe = self._resolve_exe()
        with tempfile.TemporaryDirectory(prefix="inscriber-mtmd-") as td:
            img_path = Path(td) / f"page_{image.page_number}.png"
            img_path.write_bytes(image.png_bytes)
            args = self.build_args(
                exe, img_path, prompt, chat_template=chat_template, max_tokens=max_tokens
            )
            logger.debug("mtmd-cli: %s", " ".join(args))
            proc = subprocess.run(
                args, capture_output=True, text=True, timeout=timeout_s
            )
        if proc.returncode != 0:
            raise InferenceError(
                f"llama-mtmd-cli exited {proc.returncode}: {proc.stderr[-1000:]}"
            )
        self.last_raw = proc.stdout.strip()
        return self.last_raw


# --------------------------------------------------------------------------- #
# OcrBackend ABC (DESIGN §8.2) — the backend OWNS the whole per-page inference.
# --------------------------------------------------------------------------- #


class OcrBackend(ABC):
    """Pluggable OCR backend. The backend owns the whole inference for one page.

    For the "second backend, zero pipeline changes" promise to hold (DESIGN §8.1):
    (a) the backend owns the inference call (not just prompt/parse); (b) the
    returned ``bbox_norm`` is in the ORIGINAL-PAGE [0,1] frame; (c) it declares
    whether it can ground figures via :attr:`supports_grounding`.
    """

    name: str = "base"
    # Can this model locate figures from its own output? DeepSeek → True.
    supports_grounding: bool = False

    @abstractmethod
    def ocr_page(
        self, inf: Inferencer, image: PageImage, mode: ResolutionMode
    ) -> OcrPageResult:
        """Own the WHOLE inference for one page: build prompt(s), call ``inf``
        (possibly more than once), return clean markdown + regions in the
        original-page frame."""

    def server_flags(self) -> list[str]:
        """Extra ``llama-server`` flags (e.g. DRY / repeat-penalty, DESIGN §2.2)."""
        return []

    def sampling(self) -> dict:
        """Per-request sampling params (OCR determinism — temperature 0)."""
        return {"temperature": 0}

    def max_tokens(self) -> int:
        """Hard generation cap (a real guard against repetition loops, DESIGN §2.2)."""
        return 8192

    def chat_template(self, path: Literal["server", "mtmd-cli"]) -> str | None:
        """Path-aware chat template (DESIGN §2.2): the value (or None) to use on
        the llama-server path vs. the mtmd-cli path — they differ for DeepSeek-OCR."""
        return None

    def prompt(self, *, figures_enabled: bool | None = None) -> str:
        """The OCR prompt for this backend/mode (the backend may also carry its own
        ``figures_enabled`` from construction; ``None`` defers to that)."""
        raise NotImplementedError
