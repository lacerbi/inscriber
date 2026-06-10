"""Highest-value test (DESIGN §8.3, §17): the DeepSeek-OCR grounding parser +
coordinate mapping, pinned to **real captured output**.

The single-pass grounding design hinges on exact parsing; these are golden tests
against committed real fixtures (`tests/fixtures/deepseek_*_raw.txt`, captured on
llama.cpp build 9587) and the verified **per-axis** coordinate frame
(dev/notes/2026-06-10-build-9587-verification.md; the older padded-square frame of build
9028 is no longer supported — `min_server_build` gates it).
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
# Coordinate frame (pure) — the per-axis mapping verified on build 9587
# --------------------------------------------------------------------------- #


def test_grid_to_norm_per_axis_calibration():
    # Build 9587 emitted image[[242,243,753,653]] for the calibration box whose
    # true position is (0.25, 0.25, 0.75, 0.65) — per-axis grid/999, no padding.
    norm = grid_to_norm((242, 243, 753, 653))
    assert norm[0] == pytest.approx(242 / 999, abs=1e-6)
    assert norm[1] == pytest.approx(243 / 999, abs=1e-6)
    assert norm[2] == pytest.approx(753 / 999, abs=1e-6)
    assert norm[3] == pytest.approx(653 / 999, abs=1e-6)
    for got, want in zip(norm, (0.25, 0.25, 0.75, 0.65), strict=True):
        assert got == pytest.approx(want, abs=0.01)


def test_grid_to_norm_clamps():
    norm = grid_to_norm((0, 0, 999, 999))
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
    # figure bbox uses the per-axis mapping (raw emits image[[242,243,753,653]]):
    assert figs[0].bbox_norm[0] == pytest.approx(242 / 999, abs=1e-3)

    assert "⟦INSCRIBER_FIG:fig_p1_1⟧" in res.markdown
    assert "Calibration Page" in res.markdown
    # markers are stripped from the markdown:
    assert "image[[" not in res.markdown
    assert "[[242" not in res.markdown


def test_parse_calibration_gundam2048_fixture_same_global_frame():
    """Real output captured at a 1536x2048 render — a gundam-sized input
    (dev/notes/2026-06-10-build-9587-verification.md): build 9587 does NOT tile, so the
    grounding frame is the SAME per-axis frame at every input size. Grid coords
    are render-size-invariant and grid_to_norm recovers the true calibration box
    ((150,200,450,520)pt on the 600x800pt page → (0.25, 0.25, 0.75, 0.65))."""
    raw = (FIXTURES / "deepseek_calibration_gundam2048_raw.txt").read_text(encoding="utf-8")
    page = PageImage(page_number=1, png_bytes=b"", width_px=1536, height_px=2048)
    res = DeepSeekOcrBackend().parse(raw, page)

    figs = [r for r in res.regions if r.is_figure]
    assert len(figs) == 1
    expected = (0.25, 0.25, 0.75, 0.65)
    for got, want in zip(figs[0].bbox_norm, expected, strict=True):
        assert got == pytest.approx(want, abs=0.02)


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
