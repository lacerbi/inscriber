"""Generated-document notice appended to Markdown outputs."""

from __future__ import annotations

import re

_EQUATION_RE = re.compile(r"(\\\(|\\\[|\$\$|^\s*\\\[[\s\S]*?\\\]\s*$)", re.MULTILINE)
_HTML_TABLE_RE = re.compile(r"</?table\b|</?tr\b|</?td\b|</?th\b", re.IGNORECASE)
_PIPE_TABLE_RE = re.compile(r"^\s*\|.+\|\s*\n\s*\|(?:\s*:?-{3,}:?\s*\|)+", re.MULTILINE)
_FIGURE_DESC_RE = re.compile(r"^> \*\*Image description\.\*\*", re.MULTILINE)


def build_transcription_notice(markdown: str) -> str:
    """Return a compact, content-aware OCR/VLM caveat for one Markdown document."""
    has_equations = _EQUATION_RE.search(markdown) is not None
    has_tables = (
        _HTML_TABLE_RE.search(markdown) is not None
        or _PIPE_TABLE_RE.search(markdown) is not None
    )
    has_figure_descriptions = _FIGURE_DESC_RE.search(markdown) is not None

    items = ["text"]
    if has_equations:
        items.append("equations")
    if has_tables:
        items.append("tables")
    if has_figure_descriptions:
        items.append("figure descriptions")

    source = "OCR and VLMs" if has_figure_descriptions else "OCR"
    return f"*Transcribed with {source}; {_join_items(items)} may contain mistakes.*"


def append_transcription_notice(markdown: str) -> str:
    """Append the generated-document notice after a horizontal rule."""
    return markdown.rstrip() + "\n\n---\n\n" + build_transcription_notice(markdown) + "\n"


def _join_items(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
