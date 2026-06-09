"""M1a: PDF rasterization + page-range parsing (DESIGN §7).

The calibration test validates the zoom math end-to-end (no model needed): it
renders the known red box and checks its pixel bbox matches the analytically
predicted ``expected_box_px`` from ``calibration.json``.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from inscriber.models import ResolutionMode
from inscriber.pdf.rasterize import RasterizeError, page_count, parse_page_range, rasterize

FIXTURES = Path(__file__).parent / "fixtures"
CALIB_PDF = FIXTURES / "calibration.pdf"
CALIB_JSON = FIXTURES / "calibration.json"


# --------------------------------------------------------------------------- #
# parse_page_range
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec,total,expected",
    [
        (None, 5, [1, 2, 3, 4, 5]),
        ("all", 5, [1, 2, 3, 4, 5]),
        ("", 5, [1, 2, 3, 4, 5]),
        ("3", 5, [3]),
        ("1-3", 5, [1, 2, 3]),
        ("2-", 4, [2, 3, 4]),
        ("-2", 4, [1, 2]),
        ("3-100", 5, [3, 4, 5]),  # clamp high end
        ("100-200", 5, []),  # window fully past the doc -> empty (intersection)
        ("9-5", 10, []),  # reversed window -> empty
    ],
)
def test_parse_page_range(spec, total, expected):
    assert parse_page_range(spec, total) == expected


def test_parse_page_range_invalid():
    with pytest.raises(RasterizeError):
        parse_page_range("abc", 5)


def test_parse_page_range_zero_total():
    assert parse_page_range("1-3", 0) == []


# --------------------------------------------------------------------------- #
# Calibration render — validates the zoom matrix (no *72) end-to-end
# --------------------------------------------------------------------------- #


def _red_bbox(png_bytes: bytes) -> tuple[int, int, int, int]:
    """Bounding box (x0,y0,x1,y1 inclusive-exclusive-ish) of red pixels."""
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    px = img.load()
    w, h = img.size
    xs, ys = [], []
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if r > 200 and g < 80 and b < 80:
                xs.append(x)
                ys.append(y)
    assert xs and ys, "no red pixels found"
    return min(xs), min(ys), max(xs), max(ys)


def test_calibration_box_maps_to_expected_pixels():
    meta = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
    pdf = CALIB_PDF.read_bytes()
    assert page_count(pdf) == 1

    pages = rasterize(pdf, ResolutionMode.LARGE)
    assert len(pages) == 1
    page = pages[0]
    large = meta["modes"]["large"]
    assert page.width_px == large["render_w_px"]
    assert page.height_px == large["render_h_px"]

    x0, y0, x1, y1 = _red_bbox(page.png_bytes)
    ex0, ey0, ex1, ey1 = large["expected_box_px"]
    tol = 4  # anti-aliasing + 2pt frame stroke slack
    assert abs(x0 - ex0) <= tol, (x0, ex0)
    assert abs(y0 - ey0) <= tol, (y0, ey0)
    assert abs(x1 - ex1) <= tol, (x1, ex1)
    assert abs(y1 - ey1) <= tol, (y1, ey1)


def test_rasterize_page_dims_scale_with_mode():
    pdf = CALIB_PDF.read_bytes()
    small = rasterize(pdf, ResolutionMode.SMALL)[0]
    large = rasterize(pdf, ResolutionMode.LARGE)[0]
    # Long edge hits the mode target exactly (page is 600x800 → height is long).
    assert max(small.width_px, small.height_px) == 640
    assert max(large.width_px, large.height_px) == 1280


def test_rasterize_empty_range_errors():
    pdf = CALIB_PDF.read_bytes()
    with pytest.raises(RasterizeError):
        rasterize(pdf, ResolutionMode.LARGE, pages="5-9")  # 1-page doc, start>total
