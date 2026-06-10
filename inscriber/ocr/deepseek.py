"""DeepSeek-OCR backend (DESIGN §8.3) — pinned to real-hardware findings.

**Confirmed facts (DESIGN §2.1–2.2/§8.3; evidence in dev/notes/2026-06-09-m1a-findings.md and
dev/notes/2026-06-10-build-9587-verification.md), locked here:**

* Output is a **block layout list**, one region per block::

      LABEL[[x1, y1, x2, y2]]
      <region markdown text, until the next LABEL[[…]] or blank line>

  …NOT the inline ``<|ref|>…<|/ref|><|det|>…<|/det|>`` spans upstream docs describe.
* Coordinates are on a 0–999 **per-axis** grid relative to the original image
  (``norm = grid / 999`` per axis) — verified on llama.cpp build 9587.
  ⚠️ **Build-scoped:** builds ≤ 9028 padded the image to a square first
  (the M1a finding), shifting every x on portrait pages; that frame is no
  longer supported — :attr:`DeepSeekOcrBackend.min_server_build` makes the
  pipeline refuse servers older than 9587.
* The figure label is ``image``; its caption is the following ``image_caption``
  block.
* Grounding only activates with the image content-part **before** the text
  (handled in ``ChatClient.chat_image(image_first=True)``).
"""

from __future__ import annotations

import re
from typing import Literal

from inscriber.logging import get_logger
from inscriber.models import (
    FIGURE_LABELS,
    OcrPageResult,
    PageImage,
    Region,
    ResolutionMode,
    fig_placeholder,
)
from inscriber.ocr.base import Inferencer, OcrBackend, inference_truncated

logger = get_logger()

GROUNDING_PROMPT = "<|grounding|>Convert the document to markdown."
PLAIN_PROMPT = "Convert the document to markdown."

# Caption-class labels whose text becomes the preceding figure's Region.text.
CAPTION_LABELS = frozenset({"image_caption", "caption", "figure_caption"})

# A grounding marker: LABEL[[x1, y1, x2, y2]] (M1a-confirmed format).
MARKER_RE = re.compile(
    r"(?P<label>[A-Za-z_][A-Za-z_0-9]*)\s*\[\[\s*"
    r"(?P<coords>\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)\s*\]\]"
)

GRID_MAX = 999.0  # DeepSeek-OCR grounding grid is 0..999.

# Minimum llama.cpp build (DESIGN §2.2): the grounding frame changed upstream
# between builds 9028 (padded-square) and 9587 (per-axis); this backend pins the
# per-axis frame, so older servers would silently shift every figure crop
# (dev/notes/2026-06-10-build-9587-verification.md). Enforced by the pipeline at OCR time.
MIN_SERVER_BUILD = 9587


def grid_to_norm(
    coords: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Map a 0–999 grounding box to the original-page [0,1] frame.

    Per-axis (verified on build 9587, dev/notes/2026-06-10-build-9587-verification.md §2):
    the grid spans the original image independently per axis, so
    ``norm = clamp(grid / 999, 0, 1)`` — no padding terms. The frame is
    render-size-invariant (identical grid coords at 1280–2560 px renders).
    """
    x1, y1, x2, y2 = coords

    def n(g: int) -> float:
        return min(max(g / GRID_MAX, 0.0), 1.0)

    return (n(x1), n(y1), n(x2), n(y2))


def _parse_blocks(raw: str) -> list[tuple[str, tuple[int, int, int, int], str]]:
    """Split grounding output into ordered ``(label, coords, text)`` blocks."""
    markers = list(MARKER_RE.finditer(raw))
    blocks: list[tuple[str, tuple[int, int, int, int], str]] = []
    for i, m in enumerate(markers):
        label = m.group("label")
        nums = [int(c) for c in re.split(r"\s*,\s*", m.group("coords").strip())]
        coords = (nums[0], nums[1], nums[2], nums[3])
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
        text = raw[start:end].strip()
        blocks.append((label, coords, text))
    return blocks


class DeepSeekOcrBackend(OcrBackend):
    """v1 OCR backend (`name="deepseek-ocr"`, grounding-capable)."""

    name = "deepseek-ocr"
    supports_grounding = True
    min_server_build = MIN_SERVER_BUILD  # per-axis grounding frame (DESIGN §2.2)

    def __init__(
        self,
        *,
        figures_enabled: bool = True,
        seed: int = 0,
        max_tokens: int = 8192,
        request_timeout: float = 900.0,
    ) -> None:
        self.figures_enabled = figures_enabled
        self.seed = seed
        self._max_tokens = max_tokens
        self.request_timeout = request_timeout

    # --- capability / config knobs (DESIGN §8.2) --- #

    def sampling(self) -> dict:
        # Deterministic OCR (DESIGN §2.2): temperature 0 + fixed seed.
        return {"temperature": 0, "seed": self.seed}

    def max_tokens(self) -> int:
        return self._max_tokens

    def chat_template(self, path: Literal["server", "mtmd-cli"]) -> str | None:
        # Server applies the model's built-in template — do NOT pass one (DESIGN
        # §2.2). The mtmd-cli fallback would use "deepseek-ocr", but that path
        # crashed when last tested (build 9028, M1a; untested since) — kept for
        # documentation.
        return None if path == "server" else "deepseek-ocr"

    def server_flags(self) -> list[str]:
        # Partial mitigation for the missing n-gram repetition penalty (DESIGN §2.2):
        # DRY tuned to mirror the upstream ngram_size≈30 / window≈90. The real guards
        # remain f16/Q8 weights + max_tokens cap + soft-fail on a looping page.
        return [
            "--dry-multiplier", "0.5",
            "--dry-base", "1.75",
            "--dry-allowed-length", "30",
            "--dry-penalty-last-n", "90",
        ]

    def prompt(self, *, figures_enabled: bool | None = None) -> str:
        enabled = self.figures_enabled if figures_enabled is None else figures_enabled
        return GROUNDING_PROMPT if enabled else PLAIN_PROMPT

    # --- the per-page inference (DESIGN §8.3) --- #

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
        result = self.parse(raw, image)
        # Generation that stopped at the cap instead of EOS is the repetition-loop
        # signature (DESIGN §2.2): the page tail is missing. Keep the best-effort
        # parse but flag it — the pipeline warns and caches it as explicitly
        # truncated (DESIGN §8.6).
        result.truncated = inference_truncated(inf)
        return result

    def parse(self, raw: str, image: PageImage) -> OcrPageResult:
        """Parse grounding output → clean markdown (figure placeholders) + regions.

        Robustness (DESIGN §8.3): if no grounding markers are present (malformed /
        ungrounded output), treat the whole thing as plain markdown with no regions
        and warn — the pipeline still succeeds, just without figures.
        """
        blocks = _parse_blocks(raw)
        if not blocks:
            logger.warning(
                "page %d: no grounding markers found; treating output as plain markdown",
                image.page_number,
            )
            return OcrPageResult(
                page_number=image.page_number, markdown=raw.strip(), regions=[]
            )

        regions: list[Region] = []
        md_parts: list[str] = []
        fig_index = 0
        for idx, (label, coords, text) in enumerate(blocks):
            bbox = grid_to_norm(coords)
            if label.lower() in FIGURE_LABELS:
                fig_index += 1
                fig_id = f"fig_p{image.page_number}_{fig_index}"
                caption = None
                if idx + 1 < len(blocks) and blocks[idx + 1][0].lower() in CAPTION_LABELS:
                    caption = blocks[idx + 1][2] or None
                regions.append(Region(label=label, bbox_norm=bbox, text=caption))
                # Splice the placeholder in place of the figure (DESIGN §8.3 step 4).
                md_parts.append(fig_placeholder(fig_id))
            else:
                regions.append(Region(label=label, bbox_norm=bbox, text=text or None))
                if text:
                    md_parts.append(text)

        markdown = "\n\n".join(md_parts)
        return OcrPageResult(
            page_number=image.page_number, markdown=markdown, regions=regions
        )
