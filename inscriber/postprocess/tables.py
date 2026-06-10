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

from inscriber.models import TABLE_LABELS, Region
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

# The cropped-input variant (DESIGN §9.7): same prompt with the count-aware
# locator replaced by a crop preamble — a cropped image needs no on-page
# disambiguation — and "the page image" reworded accordingly. Everything from
# "The OCR is generally accurate" onward is byte-identical to the validated
# template (pinned by a test). ⚠️ Pending real-hardware validation (§9.7
# pinned-prompt rule); the opening line is also the pinned test-mock
# discriminator — keep it verbatim in both templates.
TABLE_PROMPT_TEMPLATE_CROPPED = """You are reconstructing ONE table from a scientific paper as clean GitHub-flavored Markdown.

The image is a cropped view of the table to reconstruct, taken from the page it appears on; it may include a sliver of surrounding page content at the edges.

You are given the cropped table image, the rest of the page's text as context, and a raw OCR transcription of that table. The OCR is generally accurate but NOT perfect: it may have MERGED adjacent labels or values that run together, and the table may be IRREGULAR (column groups with different numbers of sub-columns).

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

# Margin around a matched table region, as a fraction of page dims — covers the
# observed bbox jitter on build 9587 (Δ≈4–6 grid units ≈ 0.5%). Cache-key
# material (make_table_key crop_padding): changing it recomputes the crops.
TABLE_CROP_PADDING = 0.02

# A matched region narrower than this (per axis, [0,1] frame) cannot hold a
# readable table — treat it as unmatched and fall back to the whole-page path.
MIN_TABLE_REGION_SPAN = 0.01


def match_table_regions(
    blobs: list[str], regions: list[Region]
) -> list[Region | None]:
    """Match each ``<table>`` blob to the grounded table region that anchors it.

    A table region's **anchor text** is its own text when present, or — the
    shape confirmed on build 9587 (real capture:
    ``tests/fixtures/deepseek_paper_table_p27_raw.txt``) — the immediately
    following region's text: ``table[[bbox]]`` is an EMPTY block, exactly like
    ``image``, and the following ``table_caption[[bbox]]`` block carries the
    caption AND the ``<table>`` HTML. Matching is content-based (exact anchor
    match preferred, then containment — so a blob that is a substring of some
    other anchor can never steal that region from its true blob), label-gated
    (``TABLE_LABELS`` only — a ``text``-block bbox is not a table bbox), with
    document order as the tiebreak for duplicate blobs. Degenerate bboxes are
    not candidates. ``None`` entries (hand-edited bundle markdown, an
    ungrounded table, a stale region) fall back to the whole-page input path.
    """
    candidates: list[tuple[Region, str]] = []
    for i, r in enumerate(regions):
        if r.label.lower() not in TABLE_LABELS:
            continue
        x1, y1, x2, y2 = r.bbox_norm
        if x2 - x1 < MIN_TABLE_REGION_SPAN or y2 - y1 < MIN_TABLE_REGION_SPAN:
            continue
        anchor = r.text or (regions[i + 1].text if i + 1 < len(regions) else None)
        if not anchor:
            continue
        candidates.append((r, anchor))
    used: set[int] = set()
    matched: list[Region | None] = []
    for blob in blobs:
        match = None
        for exact in (True, False):
            for j, (r, anchor) in enumerate(candidates):
                if j in used:
                    continue
                if (blob == anchor) if exact else (blob in anchor):
                    match = r
                    used.add(j)
                    break
            if match is not None:
                break
        matched.append(match)
    return matched


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
    table_blob: str,
    page_text: str,
    *,
    table_index: int,
    table_count: int,
    cropped: bool = False,
) -> str:
    """Assemble the full restructuring prompt — also the table cache key material.

    ``cropped`` selects the cropped-input variant (the image is the table crop,
    so the locator is replaced by the crop preamble; index/count are unused).
    """
    if cropped:
        return TABLE_PROMPT_TEMPLATE_CROPPED.format(
            page_text=page_text, table_blob=table_blob
        )
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
