"""Generate a richer single-page 'mini paper' fixture for M1a/M1b format capture.

Exercises more DeepSeek-OCR grounding labels than the calibration page (title,
body text, a figure + caption, a small table, an equation). Run on real hardware
via dev/scripts/m1a_spike.py --paper to capture the grounded output for the M1b parser
golden test.

    python tests/fixtures/make_sample_paper.py
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from make_calibration import _chart_png  # reuse the line-chart image

HERE = Path(__file__).parent

ABSTRACT = (
    "We present a calibration study for local optical character recognition. "
    "Our method converts academic documents into Markdown using a vision model "
    "running entirely on device. We report layout grounding accuracy and discuss "
    "coordinate-frame conventions used by the decoder."
)
INTRO_1 = (
    "Document understanding has progressed rapidly. Converting PDFs to clean text "
    "is a prerequisite for downstream language-model pipelines. We focus on the "
    "offline setting, where no cloud service is involved."
)
INTRO_2 = (
    "We summarize related work and motivate the need for figure grounding. "
    "Existing tools transcribe text well but rarely localize figures, which is "
    "essential when figures must be replaced by generated descriptions."
)
METHOD = (
    "Let x denote the input page and y the predicted markdown. We model p(y | x) "
    "with a decoder conditioned on encoder features. The loss is L = -log p(y | x)."
)


def build_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # US Letter

    def text(x, y, s, size=11, font="helv"):
        page.insert_text((x, y), s, fontsize=size, fontname=font)

    def box(x0, y0, x1, y1, s, size=11):
        page.insert_textbox(fitz.Rect(x0, y0, x1, y1), s, fontsize=size, fontname="helv")

    text(72, 80, "A Calibration Study for Local OCR", size=20)
    text(72, 110, "Ada Researcher, Boris Scientist", size=11)

    text(72, 150, "Abstract", size=13)
    box(72, 160, 540, 230, ABSTRACT)

    text(72, 250, "1. Introduction", size=14)
    box(72, 262, 540, 320, INTRO_1)
    box(72, 322, 540, 380, INTRO_2)

    # Figure with caption.
    fig_rect = fitz.Rect(150, 400, 462, 600)
    page.insert_image(fig_rect, stream=_chart_png())
    page.draw_rect(fig_rect, color=(0.3, 0.3, 0.3), width=1)
    text(150, 618, "Figure 1: Calibration curve for the proposed method.", size=10)

    text(72, 650, "2. Method", size=14)
    box(72, 662, 540, 710, METHOD)

    data = doc.tobytes()
    doc.close()
    return data


def main() -> None:
    (HERE / "sample_paper.pdf").write_bytes(build_pdf())
    print("wrote sample_paper.pdf")


if __name__ == "__main__":
    main()
