"""M2: figure injection — blockquote format, modes, placeholders (DESIGN §10.2)."""

from __future__ import annotations

from inscriber.models import Figure, fig_placeholder
from inscriber.postprocess.inject import (
    ensure_placeholders,
    format_blockquote,
    inject_descriptions,
)


def _fig(fid="fig_p1_1", caption="Figure 1: x."):
    return Figure(id=fid, page=1, bbox_norm=(0.1, 0.2, 0.8, 0.6),
                  crop_path=f"figures/{fid}.png", caption=caption)


def test_format_blockquote_header_and_prefixes():
    out = format_blockquote("First line.\nSecond line.")
    assert out == "> **Image description.** First line.\n> Second line."


def test_format_blockquote_blank_lines_become_bare_gt():
    out = format_blockquote("Para one.\n\nPara two.")
    assert out == "> **Image description.** Para one.\n>\n> Para two."


def test_inject_describe_only():
    md = f"Intro.\n\n{fig_placeholder('fig_p1_1')}\n\nFigure 1: x."
    out = inject_descriptions(md, descriptions={"fig_p1_1": "A chart."},
                              figures={"fig_p1_1": _fig()}, mode="describe-only")
    assert "> **Image description.** A chart." in out
    assert "⟦INSCRIBER_FIG" not in out
    assert "![" not in out  # describe-only does not keep the image ref


def test_inject_describe_and_keep_adds_image_ref():
    md = fig_placeholder("fig_p1_1")
    out = inject_descriptions(md, descriptions={"fig_p1_1": "A chart."},
                              figures={"fig_p1_1": _fig(caption="Figure 1: x.")},
                              mode="describe-and-keep")
    assert "![Figure 1: x.](figures/fig_p1_1.png)" in out
    assert "> **Image description.** A chart." in out


def test_inject_placeholder_mode():
    md = fig_placeholder("fig_p1_1")
    out = inject_descriptions(md, descriptions={}, figures={"fig_p1_1": _fig()},
                              mode="placeholder")
    assert out.strip() == "> **Image.** [not displayed]"


def test_inject_missing_description_is_unavailable():
    md = fig_placeholder("fig_p1_1")
    out = inject_descriptions(md, descriptions={}, figures={"fig_p1_1": _fig()},
                              mode="describe-only")
    assert "> **Image description.** [figure description unavailable]" in out


def test_ensure_placeholders_appends_when_missing():
    md = "Page text only."
    out = ensure_placeholders(md, [_fig("fig_p1_1")])
    assert out == "Page text only.\n\n" + fig_placeholder("fig_p1_1")


def test_ensure_placeholders_noop_when_present():
    md = f"Text\n\n{fig_placeholder('fig_p1_1')}\n\nmore"
    assert ensure_placeholders(md, [_fig("fig_p1_1")]) == md
