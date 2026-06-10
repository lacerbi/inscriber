"""M3: stitching + cleanup (DESIGN §10.1, §10.3)."""

from __future__ import annotations

from inscriber.models import fig_placeholder
from inscriber.postprocess.stitch import (
    dehyphenate,
    ensure_image_description_spacing,
    normalize_line_breaks,
    stitch_pages,
    strip_running_headers_footers,
)


def test_normalize_line_breaks():
    assert normalize_line_breaks("a\n\n\n\nb") == "a\n\nb"
    assert normalize_line_breaks("a\n\nb") == "a\n\nb"


def test_ensure_spacing_around_description_block():
    md = "text\n> **Image description.** A chart.\nmore text"
    out = ensure_image_description_spacing(md)
    lines = out.split("\n")
    # blank line before and after the blockquote:
    assert "" in lines
    assert out == "text\n\n> **Image description.** A chart.\n\nmore text"


def test_ensure_spacing_tolerates_all_header_spellings():
    for header in ("Image description", "Image Description", "Image"):
        md = f"x\n> **{header}.** body\ny"
        out = ensure_image_description_spacing(md)
        assert f"> **{header}.** body" in out
        assert "\n\n> " in out


def test_ensure_spacing_figure_caption_after_block():
    md = "> **Image description.** A chart.\nFigure 1: the chart."
    out = ensure_image_description_spacing(md)
    assert "\n\nFigure 1: the chart." in out


def test_dehyphenate_joins_line_break():
    assert dehyphenate("exam-\nple") == "example"
    assert dehyphenate("multi-\n  word") == "multiword"
    # a real hyphenated word not at a break is left alone:
    assert dehyphenate("well-known") == "well-known"


def test_stitch_pages_plain():
    out = stitch_pages([(1, "page one"), (2, "page two")])
    # Pages are always separated by a blank line, never welded together.
    assert out == "page one\n\npage two"


def test_stitch_pages_never_fuses_page_boundary():
    # Regression: a page ending in "(MAR)" must not weld onto the next page's
    # "2. A researcher…" (the observed "(MAR)2. A researcher" fusion bug).
    out = stitch_pages([(1, "…random (MAR)"), (2, "2. A researcher uses…")])
    assert "(MAR)2." not in out
    assert out == "…random (MAR)\n\n2. A researcher uses…"


def test_stitch_pages_with_numbers_and_separators():
    out = stitch_pages(
        [(1, "one"), (2, "two")], page_numbers=True, page_separators=True
    )
    assert "#### Page 1" in out
    assert "#### Page 2" in out
    assert "---" in out
    assert out.index("#### Page 1") < out.index("one") < out.index("---")


def test_strip_running_headers_footers():
    pages = [
        "Journal of Examples\n\nPage one body.\n\n1",
        "Journal of Examples\n\nPage two body.\n\n2",
        "Journal of Examples\n\nPage three body.\n\n3",
    ]
    out = strip_running_headers_footers(pages)
    for cleaned in out:
        assert "Journal of Examples" not in cleaned
    assert "Page one body." in out[0]


def test_strip_running_headers_never_removes_figure_placeholders():
    pages = [
        "Title page body.",
        f"Page two body.\n\n{fig_placeholder('fig_p2_1')}",
        "Page three body.",
        f"Page four body.\n\n{fig_placeholder('fig_p4_1')}",
    ]
    out = strip_running_headers_footers(pages)
    assert fig_placeholder("fig_p2_1") in out[1]
    assert fig_placeholder("fig_p4_1") in out[3]


def test_strip_running_headers_uses_ceiling_threshold():
    pages = [
        "Repeated Header\n\nPage one body.",
        "Repeated Header\n\nPage two body.",
        "Different Header\n\nPage three body.",
        "Another Header\n\nPage four body.",
    ]
    # 60% of 4 pages is 2.4, so only two occurrences must not qualify.
    assert strip_running_headers_footers(pages) == pages


def test_strip_skips_short_docs():
    pages = ["Header\n\nbody one", "Header\n\nbody two"]  # only 2 pages < min_pages
    assert strip_running_headers_footers(pages) == pages


def test_pipe_table_rows_never_stripped_as_chrome():
    # Pages ending in a pipe table: the digit-normalized last rows recur
    # (`| # | # |`) and would be stripped as a running footer, silently losing
    # table data — but table rows are protected artifact lines (never chrome).
    intros = ["Alpha intro.", "Beta methods.", "Gamma results.", "Delta discussion."]
    pages = [
        f"{intro}\n\n| A | B |\n| --- | --- |\n| {i} | {i * 2} |"
        for i, intro in enumerate(intros, 1)
    ]
    out = strip_running_headers_footers(pages)
    assert out == pages
