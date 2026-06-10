"""Page stitching + cleanup (DESIGN §10.1, §10.3).

Two tiers:

* **(a) ported from paper2llm** (always on): ``normalizeLineBreaks`` (collapse 3+
  newlines) and ``ensureImageDescriptionSpacing`` (blank lines around description
  blockquotes + ``Figure …`` captions) — ``markdown-processor.ts``.
* **(b) new for inscriber** (per-page OCR artifacts; ``--no-clean`` toggles):
  running header/footer stripping and conservative de-hyphenation.
"""

from __future__ import annotations

import re
from collections import Counter
from math import ceil

from inscriber.logging import get_logger
from inscriber.models import FIG_PLACEHOLDER_PREFIX

logger = get_logger()

# ensureImageDescriptionSpacing markers (markdown-processor.ts:112) — tolerates all
# three header spellings.
_IMG_BLOCK_RE = re.compile(r"^> \*\*(?:Image description|Image Description|Image)\.\*\*")
_FIGURE_RE = re.compile(r"^Figure ")


def normalize_line_breaks(markdown: str) -> str:
    """Collapse 3+ consecutive newlines to a single blank line (``\\n{3,}`` → ``\\n\\n``)."""
    return re.sub(r"\n{3,}", "\n\n", markdown)


def ensure_image_description_spacing(markdown: str) -> str:
    """Guarantee blank lines before/after description blockquotes and ``Figure …``
    captions that follow them (verbatim port of ``ensureImageDescriptionSpacing``)."""
    if not markdown:
        return markdown

    lines = markdown.split("\n")
    result: list[str] = []
    in_image_block = False
    after_image_block = False
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not in_image_block and _IMG_BLOCK_RE.match(line):
            in_image_block = True
            after_image_block = False
            if i > 0 and result and result[-1] != "":
                result.append("")
            result.append(line)
        elif in_image_block and line.startswith(">"):
            result.append(line)
        elif in_image_block:
            in_image_block = False
            after_image_block = True
            if line != "":
                result.append("")
            if line != "":
                result.append(line)
        elif after_image_block and _FIGURE_RE.match(line):
            if result and result[-1] != "":
                result.append("")
            result.append(line)
            if i < n - 1 and lines[i + 1] != "":
                result.append("")
        else:
            result.append(line)
            if line != "" and not _FIGURE_RE.match(line):
                after_image_block = False
        i += 1

    if in_image_block:
        result.append("")
    return "\n".join(result)


# --------------------------------------------------------------------------- #
# (b) New cleanup — conservative, never deletes content we're unsure about.
# --------------------------------------------------------------------------- #


def _norm_line(line: str) -> str:
    """Normalize a line for recurrence comparison (digits → ``#``)."""
    return re.sub(r"\d+", "#", line.strip())


def _is_protected_artifact_line(line: str) -> bool:
    """Lines generated from figure/table handling must never be treated as page
    chrome (a pipe-table row as a page's first/last line could otherwise recur —
    e.g. identical separator rows — and be stripped, silently losing table data)."""
    stripped = line.strip()
    lowered = stripped.lower()
    return (
        FIG_PLACEHOLDER_PREFIX in stripped
        or _IMG_BLOCK_RE.match(stripped) is not None
        or stripped.startswith("> ")
        or stripped.startswith("![")
        or stripped.startswith("|")
        or stripped.startswith("Figure ")
        or lowered.startswith("<center>figure ")
    )


def strip_running_headers_footers(
    page_texts: list[str], *, min_pages: int = 3, min_fraction: float = 0.6
) -> list[str]:
    """Strip short lines that recur at the same page position across many pages.

    Conservative: needs ``>= min_pages`` pages and recurrence on ``>= min_fraction``
    of them; only the first/last non-empty line of each page is considered; lines
    longer than 80 chars are never treated as headers/footers. Logs what's removed.
    """
    if len(page_texts) < min_pages:
        return page_texts

    tops: list[str] = []
    bottoms: list[str] = []
    for md in page_texts:
        nonempty = [ln for ln in md.split("\n") if ln.strip()]
        top = nonempty[0] if nonempty else ""
        bottom = nonempty[-1] if nonempty else ""
        tops.append("" if _is_protected_artifact_line(top) else _norm_line(top))
        bottoms.append("" if _is_protected_artifact_line(bottom) else _norm_line(bottom))

    threshold = max(2, ceil(len(page_texts) * min_fraction))
    top_counts = Counter(t for t in tops if t and len(t) <= 80)
    bot_counts = Counter(b for b in bottoms if b and len(b) <= 80)
    headers = {t for t, c in top_counts.items() if c >= threshold}
    footers = {b for b, c in bot_counts.items() if c >= threshold}
    if not headers and not footers:
        return page_texts

    out: list[str] = []
    removed = 0
    for md in page_texts:
        lines = md.split("\n")
        # Strip first non-empty line if it is a recurring header.
        idx_top = next((j for j, ln in enumerate(lines) if ln.strip()), None)
        if (
            idx_top is not None
            and not _is_protected_artifact_line(lines[idx_top])
            and _norm_line(lines[idx_top]) in headers
        ):
            del lines[idx_top]
            removed += 1
        # Strip last non-empty line if it is a recurring footer.
        idx_bot = next((j for j in range(len(lines) - 1, -1, -1) if lines[j].strip()), None)
        if (
            idx_bot is not None
            and not _is_protected_artifact_line(lines[idx_bot])
            and _norm_line(lines[idx_bot]) in footers
        ):
            del lines[idx_bot]
            removed += 1
        out.append("\n".join(lines).strip("\n"))
    if removed:
        logger.info("stripped %d running header/footer line(s)", removed)
    return out


def dehyphenate(markdown: str) -> str:
    """Join words split by a hyphen at a line break: ``exam-\\nple`` → ``example``."""
    return re.sub(r"(\w)-\n\s*(\w)", r"\1\2", markdown)


# --------------------------------------------------------------------------- #
# Stitching
# --------------------------------------------------------------------------- #


def stitch_pages(
    pages: list[tuple[int, str]],
    *,
    page_numbers: bool = False,
    page_separators: bool = False,
    normalize: bool = True,
) -> str:
    """Concatenate per-page markdown in order (DESIGN §10.1).

    Optional ``#### Page {n}`` headings and ``---`` separators (both default off).
    The ``#### Page N`` shape is what the splitter recognizes. Pages are **always**
    separated by a blank line — even with both options off — so the last token of
    one page cannot fuse into the first token of the next (e.g. a page ending in
    ``(MAR)`` welding onto the next page's ``2. A researcher…``).
    """
    parts: list[str] = []
    for idx, (page_number, content) in enumerate(pages):
        if idx > 0:
            # ``---`` when separators are requested (it carries its own blank
            # lines); a plain blank line otherwise.
            parts.append("\n\n---\n\n" if page_separators else "\n\n")
        if page_numbers:
            parts.append(f"#### Page {page_number}\n\n")
        page_md = normalize_line_breaks(content) if normalize else content
        # Strip the page's own leading/trailing newlines so the join above is the
        # single source of inter-page spacing (no 3+ newline pile-ups).
        parts.append(page_md.strip("\n"))
    return "".join(parts)
