"""M1b highest-value test (DESIGN §8.3, §17): the DeepSeek-OCR grounding parser +
coordinate mapping, pinned to **real captured output** from the M1a spike.

The single-pass grounding design hinges on exact parsing; these are golden tests
against committed real fixtures (`tests/fixtures/deepseek_*_raw.txt`) and the
M1a-confirmed **padded-square** coordinate frame.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inscriber.models import PageImage, ResolutionMode
from inscriber.ocr.base import Inferencer
from inscriber.ocr.deepseek import DeepSeekOcrBackend, grid_to_norm
from inscriber.pdf.rasterize import rasterize

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeInferencer(Inferencer):
    """Returns a canned raw string (mocks the chat-client boundary, DESIGN §17)."""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.calls: list[tuple] = []

    def infer(self, image, prompt, *, sampling, chat_template, max_tokens, timeout_s):
        self.calls.append((prompt, sampling, chat_template, max_tokens))
        return self.raw


def _page_for(pdf_name: str) -> PageImage:
    pdf = (FIXTURES / pdf_name).read_bytes()
    return rasterize(pdf, ResolutionMode.LARGE)[0]


# --------------------------------------------------------------------------- #
# Coordinate frame (pure) — the M1a padded-square mapping
# --------------------------------------------------------------------------- #


def test_grid_to_norm_padded_square_calibration():
    # calibration page renders 960x1280; figure emitted image[[305,245,690,653]].
    norm = grid_to_norm((305, 245, 690, 653), 960, 1280)
    assert norm[0] == pytest.approx(0.2404, abs=1e-3)
    assert norm[1] == pytest.approx(0.2453, abs=1e-3)
    assert norm[2] == pytest.approx(0.7542, abs=1e-3)
    assert norm[3] == pytest.approx(0.6537, abs=1e-3)


def test_grid_to_norm_long_axis_has_no_padding():
    # On the long axis, padded-square == per-axis (pad=0): grid/999 directly.
    _, y0, _, y1 = grid_to_norm((0, 250, 0, 750), 960, 1280)
    assert y0 == pytest.approx(250 / 999, abs=1e-4)
    assert y1 == pytest.approx(750 / 999, abs=1e-4)


def test_grid_to_norm_clamps():
    norm = grid_to_norm((0, 0, 999, 999), 960, 1280)
    assert all(0.0 <= v <= 1.0 for v in norm)


# --------------------------------------------------------------------------- #
# Calibration fixture
# --------------------------------------------------------------------------- #


def test_parse_calibration_fixture():
    raw = (FIXTURES / "deepseek_calibration_raw.txt").read_text(encoding="utf-8")
    page = _page_for("calibration.pdf")
    res = DeepSeekOcrBackend().parse(raw, page)

    labels = [r.label for r in res.regions]
    assert labels == ["title", "image", "image_caption"]

    figs = [r for r in res.regions if r.is_figure]
    assert len(figs) == 1
    assert figs[0].label == "image"
    # caption pulled from the following image_caption block:
    assert figs[0].text == "<center>Figure 1: calibration box. </center>"
    # figure bbox uses padded-square mapping:
    assert figs[0].bbox_norm[0] == pytest.approx(0.2404, abs=1e-3)

    assert "⟦INSCRIBER_FIG:fig_p1_1⟧" in res.markdown
    assert "Calibration Page" in res.markdown
    # markers are stripped from the markdown:
    assert "image[[" not in res.markdown
    assert "[[305" not in res.markdown


# --------------------------------------------------------------------------- #
# Sample-paper fixture (richer: title/text/image/caption/math)
# --------------------------------------------------------------------------- #


def test_parse_sample_paper_fixture():
    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8")
    page = _page_for("sample_paper.pdf")
    res = DeepSeekOcrBackend().parse(raw, page)

    # 11 grounded blocks; exactly one figure.
    assert len(res.regions) == 11
    figs = [r for r in res.regions if r.is_figure]
    assert len(figs) == 1
    assert figs[0].label == "image"
    assert figs[0].text == "<center>Figure 1: Calibration curve for the proposed method. </center>"

    md = res.markdown
    # Headings preserved (already markdown-formatted in the model output).
    assert "## Abstract" in md
    assert "## 1. Introduction" in md
    assert "## 2. Method" in md
    # Inline LaTeX math preserved.
    assert r"\(x\)" in md
    # Figure placeholder spliced in place — after the intro text, before the caption.
    assert "⟦INSCRIBER_FIG:fig_p1_1⟧" in md
    assert md.index("We summarize") < md.index("⟦INSCRIBER_FIG:fig_p1_1⟧")
    assert md.index("⟦INSCRIBER_FIG:fig_p1_1⟧") < md.index("Figure 1: Calibration curve")
    # No grounding markers leak into the markdown.
    assert "sub_title[[" not in md
    assert "image_caption[[" not in md


def test_ocr_page_matches_parse_and_uses_grounding_prompt():
    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8")
    page = _page_for("sample_paper.pdf")
    backend = DeepSeekOcrBackend()
    inf = _FakeInferencer(raw)
    res = backend.ocr_page(inf, page, ResolutionMode.LARGE)

    assert res.markdown == backend.parse(raw, page).markdown
    # grounding prompt + deterministic sampling were used:
    prompt, sampling, chat_template, _ = inf.calls[0]
    assert prompt == "<|grounding|>Convert the document to markdown."
    assert sampling["temperature"] == 0
    assert chat_template is None  # server path


# --------------------------------------------------------------------------- #
# Robustness: malformed / ungrounded output → plain markdown, no regions
# --------------------------------------------------------------------------- #


def test_parse_no_markers_falls_back_to_plain():
    page = _page_for("calibration.pdf")
    res = DeepSeekOcrBackend().parse("# Just text\n\nNo grounding here.", page)
    assert res.regions == []
    assert res.markdown == "# Just text\n\nNo grounding here."


def test_figures_disabled_uses_plain_prompt():
    backend = DeepSeekOcrBackend(figures_enabled=False)
    assert backend.prompt() == "Convert the document to markdown."
