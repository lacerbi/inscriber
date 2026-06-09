"""Split a document into main / appendix / backmatter (DESIGN §11).

Ported from paper2llm ``core/utils/markdown-splitter.ts`` (section regexes +
boundary logic) and ``content-utils.ts`` (the ``allparts`` reassembly). Heading
detection is case-insensitive, any level ``#+``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- Backmatter (acknowledgments / impact / references family) — markdown-splitter.ts:95 ---
_ACK_PATTERNS = [
    re.compile(r"^#+\s+(Acknowledgments?)\b", re.M | re.I),
    re.compile(r"^#+\s+(Acknowledgements?)\b", re.M | re.I),
    re.compile(r"^#+\s+Author\s+(Contributions|contributions)", re.M | re.I),
    re.compile(r"^#+\s+Funding", re.M | re.I),
    re.compile(r"^#+\s+Impact\s+(Statement|statement)", re.M | re.I),
    re.compile(r"^#+\s+Broader\s+(Impact|impact)", re.M | re.I),
    re.compile(r"^#+\s+Societal\s+(Impact|impact)", re.M | re.I),
    re.compile(r"^#+\s+Ethical\s+(Considerations|considerations)", re.M | re.I),
    re.compile(r"^#+\s+(References|Bibliography)\b", re.M | re.I),
    re.compile(r"^#+\s+Works\s+Cited\b", re.M | re.I),
    re.compile(r"^#+\s+Literature\s+Cited\b", re.M | re.I),
    re.compile(r"^#+\s+Citations?\b", re.M | re.I),
    re.compile(r"^#+\s+References\s+and\s+Notes\b", re.M | re.I),
    re.compile(r"^#+\s+References\s+Cited\b", re.M | re.I),
    re.compile(r"^#+\s+Cited\s+(Works|Literature)\b", re.M | re.I),
]

# --- Appendix / supplementary — markdown-splitter.ts:115. The (is_a_guard) flag marks
# the "A "/"A. " patterns, accepted only after the ack match (false-positive guard). ---
_APPENDIX_PATTERNS = [
    (re.compile(r"^#+\s+(Appendix|Appendices|appendix|appendices)\b", re.M | re.I), False),
    (re.compile(
        r"^#+\s+(Supplementary|Supporting|supplementary|supporting)\s+"
        r"(Material|Materials|Information|Data|material|materials|information|data)",
        re.M | re.I), False),
    (re.compile(r"^#+\s+(Supplemental|supplemental)\s+", re.M | re.I), False),
    (re.compile(r"^#+\s+SI\s+", re.M | re.I), False),
    (re.compile(r"^#+\s+S\d+\.\s+", re.M | re.I), False),
    (re.compile(r"^#+\s+A\s+", re.M | re.I), True),
    (re.compile(r"^#+\s+A\.\s+", re.M | re.I), True),
]

_PAGE_MARKER = re.compile(r"^#{3,4}\s+Page\s+\d+\s*$", re.I)


@dataclass
class MarkdownSections:
    main_content: str
    backmatter: str | None
    appendix: str | None
    title: str


def extract_title(content: str) -> str:
    """First ``# Title`` → BibTeX ``title={…}`` → ``"Untitled_Paper"`` (§11)."""
    m = re.search(r"^# (.+?)$", content, re.M)
    if m:
        return m.group(1).strip()
    m = re.search(r"title=\{([^}]*)\}", content)
    if m:
        return m.group(1).strip()
    return "Untitled_Paper"


def _has_content_between(content: str, marker_pos: int, heading_pos: int) -> bool:
    between = content[marker_pos:heading_pos].strip()
    lines = between.split("\n")[1:]  # drop the page-marker line itself
    return any(line.strip() for line in lines)


def _shift_before_page_marker(content: str, start: int) -> int:
    """If a page marker sits just before ``start`` with nothing between, move the
    boundary before the marker so page markers don't dangle (DESIGN §11)."""
    before = content[:start]
    lines = before.split("\n")
    for i in range(len(lines) - 1, max(0, len(lines) - 5) - 1, -1):
        if _PAGE_MARKER.search(lines[i]):
            line_pos = before.rfind(lines[i])
            if line_pos >= 0 and not _has_content_between(content, line_pos, start):
                return line_pos
            break
    return start


def find_section_boundaries(content: str) -> tuple[int | None, int | None]:
    """Return ``(ack_start, appendix_start)`` char indices (or None)."""
    ack_start: int | None = None
    for pat in _ACK_PATTERNS:
        m = pat.search(content)
        if m and (ack_start is None or m.start() < ack_start):
            ack_start = m.start()

    appendix_start: int | None = None
    for pat, is_a in _APPENDIX_PATTERNS:
        m = pat.search(content)
        if m and (appendix_start is None or m.start() < appendix_start):
            if is_a:
                # DESIGN §11 intent: bare "A "/"A. " headings are accepted ONLY when
                # they occur AFTER an acknowledgments/backmatter match. This is
                # stricter than paper2llm's literal code (which accepts them when no
                # ack exists) and prevents catastrophic false positives such as a
                # title that begins with the word "A" (e.g. "A Calibration Study").
                if ack_start is not None and m.start() > ack_start:
                    appendix_start = m.start()
            else:
                appendix_start = m.start()

    if ack_start is not None:
        ack_start = _shift_before_page_marker(content, ack_start)
    if appendix_start is not None:
        appendix_start = _shift_before_page_marker(content, appendix_start)

    # Order check: if ack appears after appendix, the ack we found is inside the
    # appendix — re-search for an ack before the appendix.
    if ack_start is not None and appendix_start is not None and ack_start > appendix_start:
        ack_content = content[:appendix_start]
        ack_start = None
        for pat in _ACK_PATTERNS:
            m = pat.search(ack_content)
            if m:
                ack_start = _shift_before_page_marker(content, m.start())
                break

    return ack_start, appendix_start


def split_markdown_content(content: str) -> MarkdownSections:
    """Split into main / backmatter / appendix + title (DESIGN §11)."""
    title = extract_title(content)
    ack_start, appendix_start = find_section_boundaries(content)

    main = content
    backmatter: str | None = None
    appendix: str | None = None

    if appendix_start is not None:
        appendix = content[appendix_start:]
        main = content[:appendix_start]

    if ack_start is not None and not (
        appendix_start is not None and ack_start > appendix_start
    ):
        backmatter = main[ack_start:]
        main = main[:ack_start]

    main = re.sub(r"---\s*$", "", main).strip()
    if backmatter:
        backmatter = re.sub(r"---\s*$", "", backmatter).strip()
    if appendix:
        appendix = re.sub(r"---\s*$", "", appendix).strip()

    return MarkdownSections(main, backmatter, appendix, title)


def format_section_with_header(content: str, title: str, section_name: str) -> str:
    return f"# {title} - {section_name}\n\n---\n\n{content}"


def prepare_formatted_sections(
    sections: MarkdownSections,
) -> tuple[str, str | None, str | None]:
    """Standalone-file framing (DESIGN §11): main's first H1 → canonical title;
    appendix/backmatter prefixed with their section header."""
    main = re.sub(r"^# .*$", f"# {sections.title}", sections.main_content, count=1, flags=re.M)
    backmatter = (
        format_section_with_header(sections.backmatter, sections.title, "Backmatter")
        if sections.backmatter
        else None
    )
    appendix = (
        format_section_with_header(sections.appendix, sections.title, "Appendix")
        if sections.appendix
        else None
    )
    return main, backmatter, appendix


def get_all_parts(sections: MarkdownSections, *, add_title: bool = True) -> str:
    """Reassemble in the deliberate **main → appendix → backmatter** order
    (content-utils.ts:43-66 — faithful, NOT a bug to "fix"). DESIGN §11."""
    parts: list[str] = []
    if sections.main_content:
        parts.append(sections.main_content)
    if sections.appendix:
        parts.append(
            format_section_with_header(sections.appendix, sections.title, "Appendix")
            if add_title
            else sections.appendix
        )
    if sections.backmatter:
        parts.append(
            format_section_with_header(sections.backmatter, sections.title, "Backmatter")
            if add_title
            else sections.backmatter
        )
    return "\n\n".join(parts)
