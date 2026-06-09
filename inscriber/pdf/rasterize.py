"""PDF rasterization via PyMuPDF (DESIGN §7).

PyMuPDF ships prebuilt wheels with no system dependency (unlike pdf2image/poppler)
— the cross-platform choice. Renders each selected page to a PNG at the long-edge
pixel target for the chosen resolution mode.
"""

from __future__ import annotations

import fitz  # PyMuPDF

from inscriber.errors import InscriberError
from inscriber.logging import get_logger
from inscriber.models import PageImage, ResolutionMode

logger = get_logger()


class RasterizeError(InscriberError):
    """Raised on an unreadable / invalid PDF or a bad page range."""


def page_count(pdf_bytes: bytes) -> int:
    """Number of pages in the PDF."""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return doc.page_count
    except Exception as e:  # pragma: no cover - corrupt PDF
        raise RasterizeError(f"could not open PDF: {e}") from e


def parse_page_range(spec: str | None, total: int) -> list[int]:
    """Resolve a 1-indexed inclusive page range, clamped to ``[1, total]``.

    Accepts ``"1-10"``, ``"3"``, ``"5-"``, ``"-12"``, ``"all"`` and ``None``
    (= all). These open-ended/shorthand forms are an inscriber convenience, not
    ported paper2llm behavior (DESIGN §7).
    """
    if total <= 0:
        return []
    s = (spec or "").strip().lower()
    if s in ("", "all"):
        return list(range(1, total + 1))

    try:
        if "-" in s:
            left, right = s.split("-", 1)
            start = int(left) if left.strip() else 1
            end = int(right) if right.strip() else total
        else:
            start = end = int(s)
    except ValueError as e:
        raise RasterizeError(
            f"invalid page range {spec!r}; use forms like '1-10', '3', '5-', '-12', 'all'"
        ) from e

    if start < 1 or end < 1:
        raise RasterizeError(f"page range {spec!r} must use positive page numbers")

    # Intersect the requested window with the available pages [1, total]
    # (DESIGN §7 "clamped to [1, page_count]"). A window fully outside the
    # document (e.g. "5-9" on a 1-page doc) intersects to empty.
    start_eff = max(start, 1)
    end_eff = min(end, total)
    if (start_eff, end_eff) != (start, end):
        logger.debug("page range %r intersected with [1, %d] -> [%d, %d]",
                     spec, total, start_eff, end_eff)
    if start_eff > end_eff:
        return []
    return list(range(start_eff, end_eff + 1))


def rasterize(
    pdf_bytes: bytes,
    mode: ResolutionMode,
    pages: str | None = None,
) -> list[PageImage]:
    """Render the selected pages to PNGs at the mode's long-edge target.

    The zoom matrix is ``fitz.Matrix(zoom, zoom)`` with
    ``zoom = target_px / max(page_pt_w, page_pt_h)`` — PyMuPDF points are already
    1/72 inch and the matrix is a unit scale, so there is **no ``* 72``** (an
    earlier draft's bug, DESIGN §7).
    """
    target_px = mode.long_edge_px
    results: list[PageImage] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total = doc.page_count
            wanted = parse_page_range(pages, total)
            if not wanted:
                raise RasterizeError(
                    f"page range {pages!r} selected no pages (document has {total})"
                )
            for page_number in wanted:
                page = doc[page_number - 1]  # 0-indexed internally
                rect = page.rect
                long_edge_pt = max(rect.width, rect.height)
                if long_edge_pt <= 0:  # pragma: no cover - degenerate page
                    raise RasterizeError(f"page {page_number} has zero size")
                zoom = target_px / long_edge_pt
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                results.append(
                    PageImage(
                        page_number=page_number,
                        png_bytes=pix.tobytes("png"),
                        width_px=pix.width,
                        height_px=pix.height,
                    )
                )
                logger.debug(
                    "rasterized page %d -> %dx%d px (mode=%s)",
                    page_number, pix.width, pix.height, mode.value,
                )
    except RasterizeError:
        raise
    except Exception as e:  # pragma: no cover - corrupt PDF mid-render
        raise RasterizeError(f"rasterization failed: {e}") from e
    return results
