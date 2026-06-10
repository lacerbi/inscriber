"""Generate the M1a calibration fixture (review Fix 4 / DESIGN §8.3, §2.2).

Produces a non-square PDF with a solid black rectangle at EXACT known PDF
coordinates, plus ``calibration.json`` recording the page size, the box in points,
and — for each resolution mode — the render dims, the expected pixel box, and the
**two competing 0–999 grounding-coordinate predictions** (reference per-axis vs.
padded-square). When DeepSeek-OCR is run on this page on real hardware (M1a Q2),
the emitted ``<|det|>`` coords are matched against these to pick the frame.

Run once to (re)generate the committed fixtures:

    python tests/fixtures/make_calibration.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

HERE = Path(__file__).parent

# Deliberately non-square page so per-axis vs padded-square frames diverge.
PAGE_W_PT = 600.0
PAGE_H_PT = 800.0
# Known black-box rectangle in points (asymmetric on both axes).
BOX_PT = (150.0, 200.0, 450.0, 520.0)  # x0, y0, x1, y1

MODES = {
    "tiny": 512,
    "small": 640,
    "base": 1024,
    "large": 1280,
    "gundam": 2048,
}


def _chart_png(w_px: int = 580, h_px: int = 600) -> bytes:
    """A simple line-chart PNG so DeepSeek-OCR grounds the region as a figure.

    Drawn in blue/black (NO red) so the red calibration frame stays the only red.
    """
    img = Image.new("RGB", (w_px, h_px), "white")
    d = ImageDraw.Draw(img)
    m = 60  # margin
    # Axes.
    d.line([(m, h_px - m), (w_px - m, h_px - m)], fill="black", width=3)  # x-axis
    d.line([(m, m), (m, h_px - m)], fill="black", width=3)  # y-axis
    # A zig-zag "data" line.
    pts = [
        (m, h_px - m - 40),
        (m + 120, h_px - m - 220),
        (m + 240, h_px - m - 120),
        (m + 360, h_px - m - 340),
        (w_px - m, h_px - m - 260),
    ]
    d.line(pts, fill=(20, 60, 200), width=5)
    for px, py in pts:
        d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(20, 60, 200))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def build_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W_PT, height=PAGE_H_PT)
    # Title + caption so the page also reads as a "document" for OCR.
    page.insert_text((60, 80), "Calibration Page", fontsize=24)
    x0, y0, x1, y1 = BOX_PT
    rect = fitz.Rect(x0, y0, x1, y1)
    # Embed a real line-chart image so DeepSeek-OCR grounds it as a figure...
    page.insert_image(rect, stream=_chart_png())
    # ...and overlay a thin RED frame at exactly BOX_PT so rasterize tests can
    # recover the box's pixel extent (the only red on the page).
    page.draw_rect(rect, color=(1, 0, 0), width=2)
    page.insert_text((x0, y1 + 24), "Figure 1: calibration box.", fontsize=12)
    data = doc.tobytes()
    doc.close()
    return data


def expected_for_mode(target_px: int) -> dict:
    long_edge_pt = max(PAGE_W_PT, PAGE_H_PT)
    zoom = target_px / long_edge_pt
    w_px = round(PAGE_W_PT * zoom)
    h_px = round(PAGE_H_PT * zoom)
    x0, y0, x1, y1 = BOX_PT
    box_px = [x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom]

    # Reference mapping (DESIGN default): grid = px / dim * 999, per-axis, original image.
    ref = [
        round(box_px[0] / w_px * 999),
        round(box_px[1] / h_px * 999),
        round(box_px[2] / w_px * 999),
        round(box_px[3] / h_px * 999),
    ]

    # Padded-square mapping: encoder pads the SHORT axis to a square of side L=long.
    long_px = max(w_px, h_px)
    pad_x = (long_px - w_px) / 2.0
    pad_y = (long_px - h_px) / 2.0
    padded = [
        round((box_px[0] + pad_x) / long_px * 999),
        round((box_px[1] + pad_y) / long_px * 999),
        round((box_px[2] + pad_x) / long_px * 999),
        round((box_px[3] + pad_y) / long_px * 999),
    ]

    return {
        "target_px": target_px,
        "zoom": zoom,
        "render_w_px": w_px,
        "render_h_px": h_px,
        "expected_box_px": [round(v, 2) for v in box_px],
        "predicted_grid_reference": ref,
        "predicted_grid_padded_square": padded,
    }


def main() -> None:
    pdf_bytes = build_pdf()
    (HERE / "calibration.pdf").write_bytes(pdf_bytes)

    meta = {
        "page_pt": [PAGE_W_PT, PAGE_H_PT],
        "box_pt": list(BOX_PT),
        "box_color": "red (1,0,0) frame outline; interior = embedded line-chart image",
        "note": (
            "On real hardware, run DeepSeek-OCR on calibration.pdf at a given mode "
            "and compare the emitted <|det|>[[...]] coords to predicted_grid_reference "
            "(DESIGN default: per-axis, original image) vs predicted_grid_padded_square. "
            "Whichever it matches IS the coordinate frame (DESIGN §2.2 / §8.3 step 3)."
        ),
        "modes": {name: expected_for_mode(px) for name, px in MODES.items()},
    }
    (HERE / "calibration.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote calibration.pdf and calibration.json")
    print("large-mode predictions:")
    print("  reference     :", meta["modes"]["large"]["predicted_grid_reference"])
    print("  padded-square :", meta["modes"]["large"]["predicted_grid_padded_square"])


if __name__ == "__main__":
    main()
