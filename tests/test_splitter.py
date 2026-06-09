"""M3: section splitting (DESIGN §11)."""

from __future__ import annotations

from inscriber.postprocess.splitter import (
    extract_title,
    get_all_parts,
    prepare_formatted_sections,
    split_markdown_content,
)


def test_extract_title_h1():
    assert extract_title("# My Great Paper\n\nbody") == "My Great Paper"


def test_extract_title_bibtex_fallback():
    assert extract_title("no heading\ntitle={Fallback Title}\n") == "Fallback Title"


def test_extract_title_default():
    assert extract_title("just text, no markers") == "Untitled_Paper"


def test_split_main_and_backmatter():
    doc = "# Paper\n\nIntro.\n\n## References\n\n[1] A. Author."
    s = split_markdown_content(doc)
    assert s.title == "Paper"
    assert s.main_content.startswith("# Paper")
    assert "Intro." in s.main_content
    assert s.backmatter is not None and s.backmatter.startswith("## References")
    assert s.appendix is None


def test_split_main_appendix_backmatter():
    doc = (
        "# Paper\n\nBody.\n\n"
        "## Acknowledgments\n\nThanks.\n\n"
        "## Appendix\n\nExtra.\n"
    )
    s = split_markdown_content(doc)
    assert "Body." in s.main_content
    assert s.backmatter is not None and "Thanks." in s.backmatter
    assert s.appendix is not None and s.appendix.startswith("## Appendix")
    # backmatter must not bleed into appendix:
    assert "Appendix" not in s.backmatter


def test_a_pattern_rejected_before_ack():
    # The first "A " heading is a body heading before the ack → guard rejects it,
    # so there's no false-positive appendix (faithful to paper2llm: only the FIRST
    # "A " heading per pattern is considered).
    doc = "# Paper\n\n## A Method\n\nbody\n\n## Acknowledgments\n\nthanks\n"
    s = split_markdown_content(doc)
    assert s.appendix is None
    assert "## A Method" in s.main_content
    assert s.backmatter is not None and "thanks" in s.backmatter


def test_a_pattern_accepted_when_first_after_ack():
    doc = "# Paper\n\nbody\n\n## Acknowledgments\n\nthanks\n\n## A Supplement\n\nstuff\n"
    s = split_markdown_content(doc)
    assert s.appendix is not None and "A Supplement" in s.appendix
    assert s.backmatter is not None and "thanks" in s.backmatter


def test_a_pattern_rejected_without_ack():
    # DESIGN §11 intent (stricter than paper2llm): a bare "A " heading with NO
    # acknowledgments anchor is NOT treated as an appendix — guards against titles
    # like "## A Calibration Study" being misread as the appendix start.
    doc = "## A Calibration Study\n\nbody\n\n## 2. Method\n\nmore body\n"
    s = split_markdown_content(doc)
    assert s.appendix is None
    assert "A Calibration Study" in s.main_content


def test_non_a_appendix_accepted_without_ack():
    # A non-"A " appendix pattern (e.g. "Appendix") is still accepted with no ack.
    doc = "# Paper\n\nbody\n\n## Appendix\n\nstuff\n"
    s = split_markdown_content(doc)
    assert s.appendix is not None and s.appendix.startswith("## Appendix")


def test_page_marker_boundary_shift():
    doc = "# Paper\n\ntext\n\n#### Page 2\n\n## References\n\nrefs\n"
    s = split_markdown_content(doc)
    # the dangling page marker moves into the backmatter, out of main:
    assert s.backmatter is not None and s.backmatter.startswith("#### Page 2")
    assert "#### Page 2" not in s.main_content


def test_prepare_formatted_sections_headers():
    s = split_markdown_content(
        "# Paper\n\nBody.\n\n## Acknowledgments\n\nThx.\n\n## Appendix\n\nApp.\n"
    )
    main, backmatter, appendix = prepare_formatted_sections(s)
    assert main.startswith("# Paper")
    assert appendix.startswith("# Paper - Appendix\n\n---\n\n")
    assert backmatter.startswith("# Paper - Backmatter\n\n---\n\n")


def test_allparts_order_main_appendix_backmatter():
    s = split_markdown_content(
        "# P\n\nMAIN.\n\n## Acknowledgments\n\nBACK.\n\n## Appendix\n\nAPP.\n"
    )
    out = get_all_parts(s, add_title=True)
    # deliberate reorder: appendix precedes backmatter even though backmatter is
    # positionally first in the source.
    assert out.index("MAIN.") < out.index("# P - Appendix")
    assert out.index("# P - Appendix") < out.index("# P - Backmatter")
    assert out.index("APP.") < out.index("BACK.")
