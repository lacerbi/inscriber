"""BibTeX citability + metadata probe (DESIGN §12, auto mode).

One **text-only** VLM call per document: is this a citable scholarly work, and
what front-matter metadata is visible on its first page? The result drives the
auto-mode source chain (``chain.py``) and the local best-effort entry
(``local.py``).

The prompt is pinned, model-facing behavior (the table-pass discipline,
DESIGN §9.2): assembled exactly once per document via the backend
(``build_bibtex_probe_prompt``), used verbatim as cache-key material AND as the
request. Changes require re-validation on real hardware recorded in
``dev/notes/2026-06-10-bibtex-probe-findings.md``. The phrase "bibliographic metadata" is
the pinned mock-dispatch discriminator (AGENTS.md) and must survive any tuning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Truncation cap on the page-1 text embedded in the prompt. Front matter
# (title/authors/venue) fits comfortably; deliberately its own constant, NOT
# the [figure].context_chars knob (that one is a figure-context setting).
PROBE_PAGE_CHARS = 3000

_CODE_FENCE_RE = re.compile(r"^```[A-Za-z]*\s*\n(?P<body>.*?)\n?```\s*$", re.DOTALL)

# The pinned probe prompt. Citability is abstain-biased (decision 1: with a
# default-on feature a false positive is worse than a false negative), and
# extraction is transcription-not-recall (decision 5: only fields visible in
# the supplied text; absent fields are omitted, never filled in).
PROBE_PROMPT_TEMPLATE = """You are extracting bibliographic metadata from the first page of a document.

First decide whether the document is CITABLE: a self-contained scholarly work \
— a research paper, preprint, thesis, or technical report — whose title and \
authors are identifiable in the text below. Slides, lecture notes, invoices, \
forms, manuals, web pages, and other non-scholarly documents are NOT citable. \
When unsure, answer "citable": false.

Then extract the metadata fields that are VISIBLE in the text. Copy them \
exactly as written; do not guess and do not recall from memory. Omit any field \
that is not visible in the text.

Answer with a single JSON object and nothing else, in this shape:
{{"citable": true, "title": "...", "authors": ["...", "..."], "year": "...", "venue": "..."}}

- "citable": required boolean.
- "title": the document's full title, when visible.
- "authors": the list of author names, when visible.
- "year": the publication year, when visible.
- "venue": the journal, conference, or repository name, when visible.

First page text:
<page_text>
{page_text}
</page_text>"""


@dataclass
class ProbeResult:
    """The parsed probe answer (front-matter metadata + citability verdict)."""

    citable: bool
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: str | None = None
    venue: str | None = None
    raw: str = ""  # the model's JSON (fence-stripped), for debugging + caching


def format_probe_prompt(page_text: str) -> str:
    """Assemble the full probe prompt — also the probe cache key material."""
    text = page_text.strip()
    if len(text) > PROBE_PAGE_CHARS:
        text = text[: PROBE_PAGE_CHARS - 3] + "..."
    return PROBE_PROMPT_TEMPLATE.format(page_text=text)


def parse_probe_response(raw: str | None) -> ProbeResult | None:
    """Parse + type-check a probe response; ``None`` means "treat as unknown".

    Tolerates a wrapping code fence (like the table pass's sanitizer); otherwise
    strict: a single JSON object with a boolean ``citable`` and correctly-typed
    optional fields. Anything malformed → ``None`` (and the caller must NOT
    cache it).
    """
    if not raw:
        return None
    text = raw.strip()
    fence = _CODE_FENCE_RE.match(text)
    if fence:
        text = fence.group("body").strip()
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("citable"), bool):
        return None

    title = data.get("title")
    if title is not None and not isinstance(title, str):
        return None
    authors = data.get("authors", [])
    if authors is None:
        authors = []
    if not isinstance(authors, list) or not all(isinstance(a, str) for a in authors):
        return None
    year = data.get("year")
    if isinstance(year, int):  # tolerate a bare-number year
        year = str(year)
    if year is not None and not isinstance(year, str):
        return None
    venue = data.get("venue")
    if venue is not None and not isinstance(venue, str):
        return None

    return ProbeResult(
        citable=data["citable"],
        title=title.strip() or None if title else None,
        authors=[a.strip() for a in authors if a.strip()],
        year=year.strip() or None if year else None,
        venue=venue.strip() or None if venue else None,
        raw=text,
    )
