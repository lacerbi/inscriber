"""Local best-effort BibTeX entry from probe-extracted front matter (DESIGN §12).

The last link in the auto-mode source chain: a clearly-marked ``@misc`` entry
assembled purely from what the probe transcribed off the document's first page.
Fully offline — no citation database is consulted. Transcription, not recall
(decision 5): absent fields are absent, never ``Unknown Journal``-style filler;
the extracted venue goes in ``note``, not ``journal`` (decision 4 — no
venue-type guessing).
"""

from __future__ import annotations

from inscriber.bibtex.probe import ProbeResult
from inscriber.bibtex.semantic_scholar import generate_citation_key, sanitize_bibtex_text

# Canonical header (pinned by tests/fixtures/bibtex_best_effort.txt).
BEST_EFFORT_HEADER = (
    "% NOTE: Best-effort entry generated from the document's own front matter\n"
    "% by inscriber (no citation database was consulted). Verify before use.\n"
    "%\n"
)


def best_effort_bibtex(probe: ProbeResult | None) -> str | None:
    """Assemble the marked ``@misc`` entry, or ``None`` when there is no usable
    title (an entry without a title is noise, not a citation)."""
    if probe is None or not probe.title:
        return None
    key = generate_citation_key(probe.title, probe.authors, probe.year)
    fields = [f"  title={{{sanitize_bibtex_text(probe.title)}}}"]
    if probe.authors:
        fields.append(
            "  author={" + " and ".join(sanitize_bibtex_text(a) for a in probe.authors) + "}"
        )
    if probe.year:
        fields.append(f"  year={{{probe.year}}}")
    if probe.venue:
        fields.append(f"  note={{{sanitize_bibtex_text(probe.venue)}}}")
    return BEST_EFFORT_HEADER + f"@misc{{{key},\n" + ",\n".join(fields) + "\n}"
