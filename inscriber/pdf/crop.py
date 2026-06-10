"""Crop figure regions from page images (DESIGN §8.4).

Bboxes are already in the original-page ``[0,1]`` frame (the backend converted
them, DESIGN §8.2), so cropping is model-agnostic: pixel box = ``bbox × (W,H)``,
plus a ``crop_padding`` margin, clamped, near-zero-area boxes skipped.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from inscriber.logging import get_logger
from inscriber.models import Figure, PageImage, Region

logger = get_logger()

# Conventional figures subdir name; Figure.crop_path is stored relative to its parent.
FIGURES_DIRNAME = "figures"

_MIN_PX = 2  # skip near-zero-area crops


def padded_pixel_box(
    bbox_norm: tuple[float, float, float, float],
    width: int,
    height: int,
    padding: float,
) -> tuple[int, int, int, int] | None:
    """``bbox_norm`` + a ``padding`` margin → a clamped pixel box, or ``None``
    when the box is near-zero-area (DESIGN §8.4). Shared by the figure and
    table crop paths so their box math cannot drift."""
    x1, y1, x2, y2 = bbox_norm
    bx1 = max(0.0, x1 - padding)
    by1 = max(0.0, y1 - padding)
    bx2 = min(1.0, x2 + padding)
    by2 = min(1.0, y2 + padding)
    px1, py1 = int(round(bx1 * width)), int(round(by1 * height))
    px2, py2 = int(round(bx2 * width)), int(round(by2 * height))
    if px2 - px1 < _MIN_PX or py2 - py1 < _MIN_PX:
        return None
    return (px1, py1, px2, py2)


def crop_region_bytes(
    png_bytes: bytes,
    bbox_norm: tuple[float, float, float, float],
    *,
    padding: float,
) -> bytes | None:
    """Crop one region from a page raster, in memory → PNG bytes (DESIGN §9.7).

    Used by the table pass: crops are ephemeral per-call inputs (never bundle
    artifacts), and the cache key is (raster hash + bbox + padding) — the
    deterministic inputs of this function — so the encoded bytes themselves are
    never key material. ``None`` means the box was near-zero-area.
    """
    img = Image.open(io.BytesIO(png_bytes))
    box = padded_pixel_box(bbox_norm, img.width, img.height, padding)
    if box is None:
        return None
    buf = io.BytesIO()
    img.crop(box).save(buf, format="PNG")
    return buf.getvalue()


def crop_figures(
    page: PageImage,
    regions: list[Region],
    *,
    crop_padding: float,
    figures_dir: Path,
) -> list[Figure]:
    """Crop each figure-class region of ``page`` → ``figures/fig_p{page}_{i}.png``.

    ``regions`` is the full region list; figure-class regions are enumerated in
    order so the ``fig_p{page}_{i}`` ids align with the placeholders the parser
    spliced (DESIGN §8.3/§8.4). The crop's caption is the region's ``text``.
    """
    width, height = page.width_px, page.height_px
    img = Image.open(io.BytesIO(page.png_bytes))
    figures_dir = Path(figures_dir)
    figures: list[Figure] = []
    fig_index = 0
    for region in regions:
        if not region.is_figure:
            continue
        fig_index += 1
        fig_id = f"fig_p{page.page_number}_{fig_index}"
        # crop_padding is a fraction of page dims (DESIGN §8.4); clamp to [0,1].
        box = padded_pixel_box(region.bbox_norm, width, height, crop_padding)
        if box is None:
            logger.warning("skipping near-zero-area figure %s (bbox=%s)", fig_id, region.bbox_norm)
            continue
        figures_dir.mkdir(parents=True, exist_ok=True)
        img.crop(box).save(figures_dir / f"{fig_id}.png")
        figures.append(
            Figure(
                id=fig_id,
                page=page.page_number,
                bbox_norm=region.bbox_norm,
                crop_path=f"{FIGURES_DIRNAME}/{fig_id}.png",
                caption=region.text,
            )
        )
    return figures
