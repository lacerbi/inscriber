"""Generated-document transcription notice."""

from __future__ import annotations

from inscriber.postprocess.notice import append_transcription_notice, build_transcription_notice


def test_notice_text_only():
    assert (
        build_transcription_notice("Plain OCR text.")
        == "*Transcribed with OCR; text may contain mistakes.*"
    )


def test_notice_adapts_to_equations_tables_and_figures():
    md = (
        "Text with \\(x^2\\).\n\n"
        "<table><tr><td>1</td></tr></table>\n\n"
        "> **Image description.** A chart."
    )
    assert build_transcription_notice(md) == (
        "*Transcribed with OCR and VLMs; text, equations, tables, "
        "and figure descriptions may contain mistakes.*"
    )


def test_append_notice_uses_horizontal_rule():
    out = append_transcription_notice("Body\n")
    assert out == (
        "Body\n\n---\n\n"
        "*Transcribed with OCR; text may contain mistakes.*\n"
    )


def test_notice_credits_vlm_for_restructured_tables():
    md = "Text.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert build_transcription_notice(md, vlm_tables=True) == (
        "*Transcribed with OCR and VLMs; text and tables may contain mistakes.*"
    )
    # Without the flag, tables alone do not imply a VLM was involved.
    assert build_transcription_notice(md) == (
        "*Transcribed with OCR; text and tables may contain mistakes.*"
    )
