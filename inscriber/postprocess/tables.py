"""Table restructuring: DeepSeek ``<table>`` blobs → clean Markdown pipe tables.

DeepSeek-OCR emits tables as degenerate HTML — ``<table>…</table>`` with most cell
boundaries missing, so adjacent cells concatenate. The values are all present but
the grid is gone and is not post-fixable. The fix (validated in
``dev/notes/2026-06-10-table-reconstruction-findings.md``): for each blob, ask the VLM to
**restructure** it from the page image — the blob supplies the values, the image
supplies the layout, and the rest of the page's text supplies correct spellings
for merged labels. Low-risk *structuring*, not from-scratch re-OCR.

This module owns the text side: blob detection, the verbatim validated prompt,
output sanitation, and splicing. The inference call lives in the VLM backend.
"""

from __future__ import annotations

import re

from inscriber.postprocess.inject import PLACEHOLDER_RE

# A well-formed DeepSeek table blob. Non-greedy so adjacent tables stay separate;
# an unclosed <table> never matches (left untouched — never risk eating content).
TABLE_BLOB_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)

_TABLE_OPEN_RE = re.compile(r"<table\b", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CODE_FENCE_RE = re.compile(r"^```[A-Za-z]*\s*\n(?P<body>.*?)\n?```\s*$", re.DOTALL)

# The validated prompt, verbatim from dev/notes/2026-06-10-table-reconstruction-findings.md.
# The page image is sent BEFORE this text (image_first=True), sampling temperature 0.
TABLE_PROMPT_TEMPLATE = """You are reconstructing ONE table from a scientific paper as clean GitHub-flavored Markdown.

{locator}

You are given the page image, the rest of the page's text as context, and a raw OCR transcription of that table. The OCR is generally accurate but NOT perfect: it may have MERGED adjacent labels or values that run together, and the table may be IRREGULAR (column groups with different numbers of sub-columns).

Guidelines:
- Use the IMAGE to determine the true structure: the real columns and rows, and any grouped/multi-level headers (represent column groups with a second header row).
- Use the PAGE TEXT to resolve ambiguous or run-together labels: the caption and surrounding prose usually spell out the correct column/row names and what the rows and columns mean. Prefer those spellings when fixing merged labels.
- When you are CERTAIN, fix clear OCR mistakes: split labels or values the OCR ran together and place them in the correct cells. Do not invent unsupported data.
- Keep irregular groups as they are; never drop or merge values to look uniform.
- Preserve each value's exact formatting (e.g. "2.57 (0.020)").
- Output ONLY the markdown table. No commentary.

Page text (context):
<page_text>
{page_text}
</page_text>

Raw OCR of the table:
{table_blob}"""


def find_table_blobs(markdown: str) -> list[tuple[int, int, str]]:
    """All ``<table>…</table>`` blobs as ``(start, end, text)`` spans, in order."""
    return [(m.start(), m.end(), m.group(0)) for m in TABLE_BLOB_RE.finditer(markdown)]


def blob_is_refinable(blob: str) -> bool:
    """Whether a blob is safe and worthwhile to restructure.

    * A figure placeholder inside the blob must survive — splicing would destroy
      the only anchor for the description, so the blob is left alone.
    * A nested ``<table>`` (unobserved from DeepSeek, but model output is
      untrusted) means the non-greedy match stopped at the INNER ``</table>`` —
      splicing that span would orphan the outer blob's tail, so skip.
    * An empty/value-less blob gives the VLM nothing to anchor on: its task would
      degrade to from-scratch re-OCR of the image (hallucination risk), so skip.
    """
    if PLACEHOLDER_RE.search(blob):
        return False
    if len(_TABLE_OPEN_RE.findall(blob)) > 1:
        return False
    text = _HTML_TAG_RE.sub(" ", blob)
    return re.search(r"[0-9A-Za-z]", text) is not None


def table_page_context(markdown: str) -> str:
    """The prompt's page text: the page markdown with all ``<table>`` blobs and
    figure placeholders removed (per the validated findings prompt)."""
    text = TABLE_BLOB_RE.sub("", markdown)
    text = PLACEHOLDER_RE.sub("", text)
    return text.strip()


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def locator_text(table_index: int, table_count: int) -> str:
    """The count-aware locator (findings ingredient 1). ``table_index`` is 1-based."""
    if table_count == 1:
        return "This page contains a single table; reconstruct it."
    return (
        f"This page contains {table_count} tables; reconstruct the "
        f"{_ordinal(table_index)} table (the one whose values match the OCR text below)."
    )


def format_table_prompt(
    table_blob: str, page_text: str, *, table_index: int, table_count: int
) -> str:
    """Assemble the full restructuring prompt — also the table cache key material."""
    return TABLE_PROMPT_TEMPLATE.format(
        locator=locator_text(table_index, table_count),
        page_text=page_text,
        table_blob=table_blob,
    )


def sanitize_table_output(raw: str | None) -> str | None:
    """Validate + clean a restructured table; ``None`` means "keep the OCR blob".

    Tolerates a wrapping code fence; rejects anything that isn't purely a pipe
    table (commentary despite the instruction, or no table at all). Conservative:
    a rejected output costs nothing — the original blob still has every value.
    """
    if not raw:
        return None
    text = raw.strip()
    fence = _CODE_FENCE_RE.match(text)
    if fence:
        text = fence.group("body").strip()
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return None
    if not all(ln.lstrip().startswith("|") for ln in lines):
        return None
    return "\n".join(ln.strip() for ln in lines)


def splice_tables(markdown: str, replacements: list[tuple[int, int, str]]) -> str:
    """Replace blob spans with their tables, blank-line-separated from neighbors.

    Spans are from :func:`find_table_blobs` against the ORIGINAL ``markdown``;
    applied in reverse order so earlier offsets stay valid.
    """
    md = markdown
    for start, end, table_md in sorted(replacements, reverse=True):
        before = md[:start].rstrip()
        after = md[end:].lstrip()
        parts = [p for p in (before, table_md.strip(), after) if p]
        md = "\n\n".join(parts)
    return md
