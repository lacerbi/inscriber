"""M5: experimental pdf-embedded figure detection (DESIGN §8.4)."""

from __future__ import annotations

import io

import fitz
from PIL import Image

from inscriber.models import OcrPageResult, ResolutionMode, fig_placeholder
from inscriber.pdf.crop import crop_figures
from inscriber.pdf.figures import figure_regions_from_pdf_embedded, select_figure_regions
from inscriber.pdf.rasterize import rasterize
from inscriber.postprocess.inject import ensure_placeholders


def _png() -> bytes:
    img = Image.new("RGB", (200, 160), "white")
    for x in range(200):
        for y in range(0, 160, 20):
            img.putpixel((x, y), (0, 0, 255))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _embedded_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((72, 72), "Embedded Figure Doc", fontsize=18)
    page.insert_image(fitz.Rect(100, 200, 400, 500), stream=_png())
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_embedded_detection_and_crop(tmp_path):
    pdf = _embedded_pdf()
    regions = figure_regions_from_pdf_embedded(pdf, 1)
    assert len(regions) >= 1
    assert regions[0].label == "image"
    # bbox is a sensible sub-region near the inserted rect (PyMuPDF preserves the
    # image aspect ratio, so the placed rect is fitted/centered within (100,200)-(400,500)).
    x0, y0, x1, y1 = regions[0].bbox_norm
    assert 0.0 < x0 < x1 < 1.0 and 0.0 < y0 < y1 < 1.0
    assert 0.1 <= x0 <= 0.25 and 0.55 <= x1 <= 0.7
    assert 0.2 <= y0 <= 0.35 and 0.5 <= y1 <= 0.65

    page = rasterize(pdf, ResolutionMode.LARGE)[0]
    figs = crop_figures(page, regions, crop_padding=0.0, figures_dir=tmp_path / "figs")
    assert len(figs) >= 1
    assert (tmp_path / "figs" / "fig_p1_1.png").is_file()


def test_select_pdf_embedded_and_append_placeholder(tmp_path):
    pdf = _embedded_pdf()
    # A non-grounding OCR result (plain text, no placeholders).
    result = OcrPageResult(page_number=1, markdown="Embedded Figure Doc", regions=[])
    regions = select_figure_regions(
        "pdf-embedded", supports_grounding=True, result=result, pdf_bytes=pdf, page_number=1
    )
    assert len(regions) >= 1
    page = rasterize(pdf, ResolutionMode.LARGE)[0]
    figs = crop_figures(page, regions, crop_padding=0.0, figures_dir=tmp_path / "f")
    md = ensure_placeholders("Embedded Figure Doc", figs)
    assert fig_placeholder("fig_p1_1") in md  # placeholder appended after page text


def test_auto_uses_grounding_when_supported():
    # With a grounding backend, 'auto' uses grounding regions, NOT pdf-embedded.
    result = OcrPageResult(page_number=1, markdown="x", regions=[])
    regions = select_figure_regions(
        "auto", supports_grounding=True, result=result, pdf_bytes=_embedded_pdf(), page_number=1
    )
    assert regions == []  # grounding found no figures; embedded path not taken
