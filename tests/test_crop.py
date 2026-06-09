"""M2: figure cropping (DESIGN §8.4)."""

from __future__ import annotations

from pathlib import Path

from inscriber.models import PageImage, Region, ResolutionMode
from inscriber.ocr.deepseek import DeepSeekOcrBackend
from inscriber.pdf.crop import crop_figures
from inscriber.pdf.rasterize import rasterize

FIXTURES = Path(__file__).parent / "fixtures"


def _calibration_page() -> PageImage:
    pdf = (FIXTURES / "calibration.pdf").read_bytes()
    return rasterize(pdf, ResolutionMode.LARGE)[0]


def test_crop_figures_creates_png_and_figure(tmp_path):
    page = _calibration_page()
    raw = (FIXTURES / "deepseek_calibration_raw.txt").read_text(encoding="utf-8")
    result = DeepSeekOcrBackend().parse(raw, page)

    figs = crop_figures(page, result.regions, crop_padding=0.02, figures_dir=tmp_path / "figures")
    assert len(figs) == 1
    fig = figs[0]
    assert fig.id == "fig_p1_1"
    assert fig.crop_path == "figures/fig_p1_1.png"
    assert fig.caption == "<center>Figure 1: calibration box. </center>"
    assert (tmp_path / "figures" / "fig_p1_1.png").is_file()


def test_crop_padding_widens_box(tmp_path):
    page = _calibration_page()
    region = Region(label="image", bbox_norm=(0.4, 0.4, 0.6, 0.6))
    from PIL import Image

    crop_figures(page, [region], crop_padding=0.0, figures_dir=tmp_path / "a")
    crop_figures(page, [region], crop_padding=0.1, figures_dir=tmp_path / "b")
    w0 = Image.open(tmp_path / "a" / "fig_p1_1.png").width
    w1 = Image.open(tmp_path / "b" / "fig_p1_1.png").width
    assert w1 > w0


def test_crop_skips_near_zero_area(tmp_path):
    page = _calibration_page()
    tiny = Region(label="image", bbox_norm=(0.5, 0.5, 0.5001, 0.5001))
    figs = crop_figures(page, [tiny], crop_padding=0.0, figures_dir=tmp_path / "z")
    assert figs == []


def test_crop_ignores_non_figure_regions(tmp_path):
    page = _calibration_page()
    text = Region(label="text", bbox_norm=(0.1, 0.1, 0.9, 0.2))
    figs = crop_figures(page, [text], crop_padding=0.0, figures_dir=tmp_path / "z")
    assert figs == []
