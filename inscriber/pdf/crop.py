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
        x1, y1, x2, y2 = region.bbox_norm
        # crop_padding is a fraction of page dims (DESIGN §8.4); clamp to [0,1].
        bx1 = max(0.0, x1 - crop_padding)
        by1 = max(0.0, y1 - crop_padding)
        bx2 = min(1.0, x2 + crop_padding)
        by2 = min(1.0, y2 + crop_padding)
        px1, py1 = int(round(bx1 * width)), int(round(by1 * height))
        px2, py2 = int(round(bx2 * width)), int(round(by2 * height))
        if px2 - px1 < _MIN_PX or py2 - py1 < _MIN_PX:
            logger.warning("skipping near-zero-area figure %s (bbox=%s)", fig_id, region.bbox_norm)
            continue
        figures_dir.mkdir(parents=True, exist_ok=True)
        crop = img.crop((px1, py1, px2, py2))
        crop.save(figures_dir / f"{fig_id}.png")
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
