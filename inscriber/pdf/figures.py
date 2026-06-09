"""Figure detection strategies (DESIGN §8.4).

Figure detection is a separate step from OCR text so future text-only backends can
plug in a different detector. ``figure.detect``:

* ``auto`` (default) — OCR-backend grounding when the backend supports it (v1:
  DeepSeek). Falls back to ``pdf-embedded`` only for a non-grounding backend.
* ``grounding`` — force grounding; error if the backend can't.
* ``none`` — no figures.
* ``pdf-embedded`` — experimental: PyMuPDF embedded raster rects. Catches raster
  figures only, **misses vector figures common in LaTeX papers** — never selected
  by ``auto`` while DeepSeek grounds.
"""

from __future__ import annotations

import fitz  # PyMuPDF

from inscriber.config import ConfigError
from inscriber.logging import get_logger
from inscriber.models import OcrPageResult, Region

logger = get_logger()


def figure_regions_from_grounding(result: OcrPageResult) -> list[Region]:
    """Figure-class regions from the OCR backend's grounding output."""
    return [r for r in result.regions if r.is_figure]


def figure_regions_from_pdf_embedded(
    pdf_bytes: bytes, page_number: int
) -> list[Region]:
    """Embedded raster image rects via PyMuPDF (experimental; DESIGN §8.4).

    ``page.get_images`` + ``page.get_image_rects`` → ``bbox_norm`` in the page's
    point frame. Ordered by ``y0`` (top-to-bottom) for stable placeholder order.
    """
    regions: list[Region] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_number - 1]
        pr = page.rect
        pw, ph = pr.width, pr.height
        for img in page.get_images(full=True):
            xref = img[0]
            for rect in page.get_image_rects(xref):
                bbox = (
                    min(max(rect.x0 / pw, 0.0), 1.0),
                    min(max(rect.y0 / ph, 0.0), 1.0),
                    min(max(rect.x1 / pw, 0.0), 1.0),
                    min(max(rect.y1 / ph, 0.0), 1.0),
                )
                regions.append(Region(label="image", bbox_norm=bbox, text=None))
    regions.sort(key=lambda r: r.bbox_norm[1])  # by y0
    return regions


def select_figure_regions(
    detect: str,
    *,
    supports_grounding: bool,
    result: OcrPageResult,
    pdf_bytes: bytes,
    page_number: int,
) -> list[Region]:
    """Dispatch figure detection per ``figure.detect`` (DESIGN §8.4)."""
    if detect == "none":
        return []
    if detect == "grounding":
        if not supports_grounding:
            raise ConfigError(
                "figure.detect='grounding' but the OCR backend cannot ground figures"
            )
        return figure_regions_from_grounding(result)
    if detect == "pdf-embedded":
        return figure_regions_from_pdf_embedded(pdf_bytes, page_number)
    if detect == "auto":
        if supports_grounding:
            return figure_regions_from_grounding(result)
        # Non-grounding backend under auto → experimental embedded path (DESIGN §8.4).
        logger.debug("auto: backend cannot ground; falling back to pdf-embedded")
        return figure_regions_from_pdf_embedded(pdf_bytes, page_number)
    raise ConfigError(f"unknown figure.detect mode: {detect!r}")
