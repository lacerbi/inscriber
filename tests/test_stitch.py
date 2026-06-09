"""M3: stitching + cleanup (DESIGN §10.1, §10.3)."""

from __future__ import annotations

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
    assert out == "page onepage two"


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


def test_strip_skips_short_docs():
    pages = ["Header\n\nbody one", "Header\n\nbody two"]  # only 2 pages < min_pages
    assert strip_running_headers_footers(pages) == pages
